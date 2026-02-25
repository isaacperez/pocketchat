from pathlib import Path
from unittest.mock import patch

import torch

from pocketchat.data import CharCorpus
from pocketchat.train import (
    TrainConfig,
    _decode_next_tokens,
    _collect_special_token_ids,
    create_dataloaders,
    evaluate,
    generate_autoregressive_completion,
    load_config,
    train,
)


def _write_text(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_create_dataloaders_returns_variable_train_and_fixed_validation_lengths(tmp_path) -> None:
    text_path = _write_text(tmp_path, "corpus.txt", "abcdefghijklmnopqrstuvwxyz" * 8)
    corpus = CharCorpus.from_file(text_path, vocab_size=128)
    config = TrainConfig(
        batch_size=4,
        min_seq_len=4,
        max_seq_len=10,
        num_next_chars=3,
        val_split=0.2,
        num_workers=0,
        pin_memory=False,
    )

    train_loader, val_loader = create_dataloaders(corpus=corpus, config=config)
    train_inputs, train_targets, train_lengths = next(iter(train_loader))

    assert train_inputs.ndim == 2
    assert train_targets.ndim == 3
    assert train_lengths.ndim == 1
    assert train_inputs.shape[0] == config.batch_size
    assert train_targets.shape[2] == config.num_next_chars
    assert int(train_lengths.min().item()) >= config.min_seq_len
    assert int(train_lengths.max().item()) <= config.max_seq_len

    for row_index, sample_len in enumerate(train_lengths.tolist()):
        if sample_len < train_inputs.shape[1]:
            assert torch.all(train_inputs[row_index, sample_len:] == corpus.pad_id)
            assert torch.all(train_targets[row_index, sample_len:, :] == corpus.pad_id)

    assert val_loader is not None
    val_inputs, val_targets, val_lengths = next(iter(val_loader))
    assert val_inputs.shape[1] == config.max_seq_len
    assert val_targets.shape[1] == config.max_seq_len
    assert torch.all(val_lengths == config.max_seq_len)


def test_evaluate_passes_lengths_to_model_and_loss(tmp_path) -> None:
    text_path = _write_text(tmp_path, "eval.txt", "abcdefghijklmnopqrstuvwxyz" * 6)
    corpus = CharCorpus.from_file(text_path, vocab_size=128)
    config = TrainConfig(
        batch_size=3,
        min_seq_len=5,
        max_seq_len=9,
        num_next_chars=2,
        val_split=0.0,
        num_workers=0,
        pin_memory=False,
    )
    train_loader, _ = create_dataloaders(corpus=corpus, config=config)

    class DummyModel:
        def __init__(self) -> None:
            self.seen_lengths: list[torch.Tensor | None] = []

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, token_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
            self.seen_lengths.append(None if lengths is None else lengths.detach().cpu())
            batch_size = token_ids.shape[0]
            return {"next_char_logits": torch.zeros(batch_size, config.num_next_chars, corpus.vocab_size)}

    model = DummyModel()
    seen_loss_lengths: list[torch.Tensor] = []

    def fake_compute_batch_loss(
        outputs: dict[str, torch.Tensor],
        batch_targets: torch.Tensor,
        pad_id: int,
        lengths: torch.Tensor,
        horizon_decay: float = 1.0,
    ) -> torch.Tensor:
        assert horizon_decay > 0
        seen_loss_lengths.append(lengths.detach().cpu())
        return torch.tensor(0.5)

    with patch("pocketchat.train.compute_batch_loss", side_effect=fake_compute_batch_loss):
        loss, preview_prompt = evaluate(
            model=model,  # type: ignore[arg-type]
            dataloader=train_loader,
            device=torch.device("cpu"),
            pad_id=corpus.pad_id,
            max_batches=2,
        )

    assert abs(loss - 0.5) < 1e-8
    assert preview_prompt is not None
    assert preview_prompt.ndim == 1
    assert len(model.seen_lengths) > 0
    assert len(seen_loss_lengths) > 0
    assert all(lengths is not None for lengths in model.seen_lengths)
    assert all(lengths.ndim == 1 for lengths in seen_loss_lengths)


def test_train_smoke_with_variable_sequence_lengths(tmp_path) -> None:
    text_path = _write_text(tmp_path, "smoke.txt", "abcdefghijklmnopqrstuvwxyz" * 12)
    out_dir = tmp_path / "runs"
    config = TrainConfig(
        device="cpu",
        steps=2,
        log_every=1,
        save_every=2,
        val_every=1,
        val_split=0.2,
        val_max_batches=1,
        batch_size=2,
        min_seq_len=4,
        max_seq_len=8,
        num_next_chars=2,
        embedding_dim=8,
        num_hypotheses=2,
        num_parts=2,
        window_size=4,
        num_iterations=1,
        num_workers=0,
        pin_memory=False,
        max_chars=0,
        vocab_size=128,
    )

    train(text_path=text_path, config=config, out_dir=out_dir)

    assert (out_dir / "checkpoint_step_0000002.pt").exists()


def test_load_config_rejects_deprecated_seq_len_key(tmp_path) -> None:
    cfg_path = tmp_path / "deprecated.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seq_len: 128",
                "min_seq_len: 16",
                "max_seq_len: 128",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_config(str(cfg_path))
        raise AssertionError("Expected ValueError for deprecated seq_len.")
    except ValueError as exc:
        assert "deprecated" in str(exc)


def test_load_config_requires_min_and_max_seq_len(tmp_path) -> None:
    cfg_path = tmp_path / "missing_lengths.yaml"
    cfg_path.write_text("min_seq_len: 16\n", encoding="utf-8")

    try:
        load_config(str(cfg_path))
        raise AssertionError("Expected ValueError when max_seq_len is missing.")
    except ValueError as exc:
        assert "min_seq_len" in str(exc)
        assert "max_seq_len" in str(exc)


def test_train_validates_loss_horizon_decay_range(tmp_path) -> None:
    text_path = _write_text(tmp_path, "corpus.txt", "abcdefghij" * 8)
    out_dir = tmp_path / "runs"
    config = TrainConfig(
        device="cpu",
        steps=1,
        log_every=1,
        save_every=1,
        val_every=1,
        batch_size=2,
        min_seq_len=4,
        max_seq_len=8,
        num_next_chars=2,
        embedding_dim=8,
        num_hypotheses=2,
        num_parts=2,
        window_size=4,
        num_iterations=1,
        num_workers=0,
        pin_memory=False,
        max_chars=0,
        vocab_size=128,
        loss_horizon_decay=0.0,
    )

    try:
        train(text_path=text_path, config=config, out_dir=out_dir)
        raise AssertionError("Expected ValueError for invalid loss_horizon_decay.")
    except ValueError as exc:
        assert "loss_horizon_decay" in str(exc)


def test_generate_autoregressive_completion_uses_growing_lengths(tmp_path) -> None:
    text_path = _write_text(tmp_path, "completion.txt", "abcdefghijklmnopqrstuvwxyz" * 4)
    corpus = CharCorpus.from_file(text_path, vocab_size=128)
    special_ids = _collect_special_token_ids(corpus)
    non_special_id = corpus.char_to_id["a"]

    class DummyCompletionModel:
        def __init__(self) -> None:
            self.training = True
            self.seen_lengths: list[int] = []
            self.num_next_chars = 2
            self.vocab_size = corpus.vocab_size

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def __call__(self, token_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
            assert lengths is not None
            self.seen_lengths.append(int(lengths[0].item()))
            logits = torch.full((1, self.num_next_chars, self.vocab_size), -5.0)
            logits[:, :, non_special_id] = 5.0
            return {
                "next_char_logits": logits,
                "next_char_predictions": torch.full((1, self.num_next_chars), non_special_id, dtype=torch.long),
            }

    model = DummyCompletionModel()
    prompt_ids = corpus.encode("abcd")

    generated = generate_autoregressive_completion(
        model=model,  # type: ignore[arg-type]
        prompt_ids=prompt_ids,
        num_generate_chars=5,
        device=torch.device("cpu"),
        special_token_ids=special_ids,
        decoding="top_k",
        temperature=1.0,
        top_k=1,
    )

    assert generated.shape == (5,)
    assert generated.tolist() == [non_special_id] * 5
    assert model.seen_lengths == [4, 6, 8]
    assert model.training is True


def test_decode_next_tokens_greedy_and_top_k_behaviour() -> None:
    logits = torch.tensor(
        [
            [0.1, 2.0, -1.0],
            [3.0, 1.5, 2.5],
        ],
        dtype=torch.float32,
    )

    greedy = _decode_next_tokens(logits=logits, decoding="greedy", temperature=1.0, top_k=0)
    top_k_1 = _decode_next_tokens(logits=logits, decoding="top_k", temperature=1.0, top_k=1)

    assert greedy.tolist() == [1, 0]
    assert top_k_1.tolist() == [1, 0]


def test_decode_next_tokens_validates_strategy_and_parameters() -> None:
    logits = torch.randn(2, 5)

    try:
        _decode_next_tokens(logits=logits, decoding="unknown", temperature=1.0, top_k=0)
        raise AssertionError("Expected ValueError for unsupported decoding strategy.")
    except ValueError as exc:
        assert "Unsupported decoding" in str(exc)

    try:
        _decode_next_tokens(logits=logits, decoding="sample", temperature=0.0, top_k=0)
        raise AssertionError("Expected ValueError for non-positive temperature.")
    except ValueError as exc:
        assert "temperature must be > 0" in str(exc)

    try:
        _decode_next_tokens(logits=logits, decoding="top_k", temperature=1.0, top_k=0)
        raise AssertionError("Expected ValueError for invalid top_k.")
    except ValueError as exc:
        assert "top_k must be > 0" in str(exc)
