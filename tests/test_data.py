import torch

from pocketchat.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from pocketchat.data import (
    CharCorpus,
    NextCharSequenceDataset,
    build_variable_length_collate_fn,
    create_next_char_dataloader,
)


def _write_text(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_from_file_builds_capped_frequency_vocab(tmp_path) -> None:
    text = "aaaaabbbbcccdde"
    path = _write_text(tmp_path, "corpus.txt", text)

    corpus = CharCorpus.from_file(path, vocab_size=7)  # 4 special + top 3 chars

    assert corpus.vocab_size == 7
    assert corpus.id_to_char[:4] == [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    assert corpus.id_to_char[4:] == ["a", "b", "c"]

    unk_id = corpus.unk_id
    assert corpus.data.dtype == torch.long
    assert corpus.data.numel() == len(text)
    assert corpus.data[-2:].tolist() == [unk_id, unk_id]


def test_vocab_size_is_bounded_by_unique_characters(tmp_path) -> None:
    path = _write_text(tmp_path, "small.txt", "abca")

    corpus = CharCorpus.from_file(path, vocab_size=100)

    assert corpus.vocab_size == 7  # 4 special + 3 unique chars
    assert set(corpus.id_to_char[4:]) == {"a", "b", "c"}


def test_encode_unknown_character_maps_to_unk(tmp_path) -> None:
    path = _write_text(tmp_path, "encode.txt", "aaaab")
    corpus = CharCorpus.from_file(path, vocab_size=5)  # only "a" retained
    unk_id = corpus.unk_id

    encoded = corpus.encode("abz")
    assert encoded.tolist() == [corpus.char_to_id["a"], unk_id, unk_id]


def test_decode_skips_special_tokens_and_marks_invalid_ids(tmp_path) -> None:
    path = _write_text(tmp_path, "decode.txt", "abc")
    corpus = CharCorpus.from_file(path, vocab_size=10)
    ids = torch.tensor(
        [
            corpus.bos_id,
            corpus.char_to_id["a"],
            corpus.eos_id,
            999,
        ],
        dtype=torch.long,
    )

    decoded = corpus.decode(ids)
    assert decoded == "a?"


def test_sample_batch_shape_dtype_and_device(tmp_path) -> None:
    path = _write_text(tmp_path, "batch.txt", "abcd" * 30)
    corpus = CharCorpus.from_file(path, vocab_size=10)

    batch = corpus.sample_batch(batch_size=8, seq_len=16, device=torch.device("cpu"))

    assert batch.shape == (8, 16)
    assert batch.dtype == torch.long
    assert batch.device.type == "cpu"


def test_sample_batch_raises_for_short_corpus(tmp_path) -> None:
    path = _write_text(tmp_path, "short.txt", "abc")
    corpus = CharCorpus.from_file(path, vocab_size=10)

    try:
        corpus.sample_batch(batch_size=2, seq_len=3, device=torch.device("cpu"))
        raise AssertionError("Expected ValueError for seq_len >= corpus length")
    except ValueError as exc:
        assert "must be greater than seq_len" in str(exc)


def test_from_file_validates_vocab_size_and_special_tokens(tmp_path) -> None:
    path = _write_text(tmp_path, "validations.txt", "hello")

    invalid_kwargs = (
        {"vocab_size": 0},
        {"vocab_size": 3},  # fewer than default special tokens
        {"vocab_size": 10, "special_tokens": (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)},
    )

    for kwargs in invalid_kwargs:
        try:
            CharCorpus.from_file(path, **kwargs)
            raise AssertionError(f"Expected ValueError for kwargs={kwargs}")
        except ValueError:
            pass


def test_special_token_id_properties(tmp_path) -> None:
    path = _write_text(tmp_path, "token_ids.txt", "abc")
    corpus = CharCorpus.from_file(path, vocab_size=10)

    assert corpus.pad_id == corpus.char_to_id[PAD_TOKEN]
    assert corpus.unk_id == corpus.char_to_id[UNK_TOKEN]
    assert corpus.bos_id == corpus.char_to_id[BOS_TOKEN]
    assert corpus.eos_id == corpus.char_to_id[EOS_TOKEN]


def test_next_char_sequence_dataset_shapes_and_alignment(tmp_path) -> None:
    path = _write_text(tmp_path, "next_chars.txt", "abcdef")
    corpus = CharCorpus.from_file(path, vocab_size=16)
    dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=4,
        num_next_chars=2,
        pad_id=corpus.pad_id,
    )

    assert len(dataset) == 3

    x0, y0 = dataset[0]
    assert x0.shape == (4,)
    assert y0.shape == (4, 2)
    assert x0.tolist() == [corpus.char_to_id[c] for c in "abcd"]
    assert y0.tolist() == [[corpus.char_to_id[c] for c in row] for row in ("bc", "cd", "de", "ef")]

    x2, y2 = dataset[2]
    assert x2.tolist() == [corpus.char_to_id[c] for c in "cdef"]
    assert y2.tolist() == [
        [corpus.char_to_id["d"], corpus.char_to_id["e"]],
        [corpus.char_to_id["e"], corpus.char_to_id["f"]],
        [corpus.char_to_id["f"], corpus.pad_id],
        [corpus.pad_id, corpus.pad_id],
    ]


def test_next_char_sequence_dataset_stride_changes_window_count(tmp_path) -> None:
    path = _write_text(tmp_path, "stride.txt", "abcdef")
    corpus = CharCorpus.from_file(path, vocab_size=16)
    dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=4,
        num_next_chars=2,
        pad_id=corpus.pad_id,
        stride=2,
    )

    assert len(dataset) == 2
    x0, _ = dataset[0]
    x1, _ = dataset[1]
    assert x0.tolist() == [corpus.char_to_id[c] for c in "abcd"]
    assert x1.tolist() == [corpus.char_to_id[c] for c in "cdef"]


def test_next_char_sequence_dataset_validates_arguments(tmp_path) -> None:
    path = _write_text(tmp_path, "dataset_validations.txt", "abcd")
    corpus = CharCorpus.from_file(path, vocab_size=16)

    invalid_configs = (
        {"seq_len": 0, "num_next_chars": 2, "pad_id": corpus.pad_id},
        {"seq_len": 2, "num_next_chars": 0, "pad_id": corpus.pad_id},
        {"seq_len": 2, "num_next_chars": 2, "pad_id": corpus.pad_id, "stride": 0},
        {"seq_len": 10, "num_next_chars": 2, "pad_id": corpus.pad_id},
        {"seq_len": 2, "num_next_chars": 2, "pad_id": -1},
    )

    for kwargs in invalid_configs:
        try:
            NextCharSequenceDataset(corpus.data, **kwargs)
            raise AssertionError(f"Expected ValueError for kwargs={kwargs}")
        except ValueError:
            pass

    try:
        NextCharSequenceDataset(corpus.data.float(), seq_len=2, num_next_chars=2, pad_id=corpus.pad_id)
        raise AssertionError("Expected TypeError for floating token_ids.")
    except TypeError:
        pass


def test_create_next_char_dataloader_returns_expected_batch_shapes(tmp_path) -> None:
    path = _write_text(tmp_path, "loader.txt", "abcdefgh")
    corpus = CharCorpus.from_file(path, vocab_size=20)

    loader = create_next_char_dataloader(
        corpus=corpus,
        seq_len=4,
        num_next_chars=3,
        batch_size=2,
        shuffle=False,
    )
    batch_inputs, batch_targets = next(iter(loader))

    assert batch_inputs.shape == (2, 4)
    assert batch_targets.shape == (2, 4, 3)
    assert batch_inputs.dtype == torch.long
    assert batch_targets.dtype == torch.long


def test_create_next_char_dataloader_requires_pad_token(tmp_path) -> None:
    path = _write_text(tmp_path, "no_pad.txt", "abcdef")
    corpus = CharCorpus.from_file(
        path,
        vocab_size=16,
        special_tokens=(UNK_TOKEN, BOS_TOKEN, EOS_TOKEN),
    )

    try:
        create_next_char_dataloader(corpus=corpus, seq_len=4, num_next_chars=2, batch_size=2)
        raise AssertionError("Expected ValueError when PAD token is missing.")
    except ValueError as exc:
        assert "PAD token" in str(exc)


def test_variable_length_collate_fn_train_mode_ranges_and_padding(tmp_path) -> None:
    path = _write_text(tmp_path, "collate_train.txt", "abcdefghijklmnopqrstuvwxyz")
    corpus = CharCorpus.from_file(path, vocab_size=64)
    dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=8,
        num_next_chars=2,
        pad_id=corpus.pad_id,
        stride=1,
    )
    batch_samples = [dataset[0], dataset[1], dataset[2], dataset[3]]
    collate_fn = build_variable_length_collate_fn(
        min_seq_len=3,
        max_seq_len=8,
        pad_id=corpus.pad_id,
        random_lengths=True,
        random_offset=True,
    )

    batch_inputs, batch_targets, lengths = collate_fn(batch_samples)

    assert batch_inputs.ndim == 2
    assert batch_targets.ndim == 3
    assert lengths.ndim == 1
    assert batch_inputs.shape[0] == 4
    assert batch_targets.shape[0] == 4
    assert batch_targets.shape[2] == 2
    assert int(lengths.min().item()) >= 3
    assert int(lengths.max().item()) <= 8

    for row_index, sample_len in enumerate(lengths.tolist()):
        if sample_len < batch_inputs.shape[1]:
            assert torch.all(batch_inputs[row_index, sample_len:] == corpus.pad_id)
            assert torch.all(batch_targets[row_index, sample_len:, :] == corpus.pad_id)


def test_variable_length_collate_fn_validation_mode_uses_fixed_max_seq_len(tmp_path) -> None:
    path = _write_text(tmp_path, "collate_val.txt", "abcdefghijklmnopqrstuvwxyz")
    corpus = CharCorpus.from_file(path, vocab_size=64)
    dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=7,
        num_next_chars=3,
        pad_id=corpus.pad_id,
        stride=1,
    )
    batch_samples = [dataset[0], dataset[1], dataset[2]]
    collate_fn = build_variable_length_collate_fn(
        min_seq_len=7,
        max_seq_len=7,
        pad_id=corpus.pad_id,
        random_lengths=False,
        random_offset=False,
    )

    batch_inputs, batch_targets, lengths = collate_fn(batch_samples)

    assert batch_inputs.shape == (3, 7)
    assert batch_targets.shape == (3, 7, 3)
    assert torch.all(lengths == 7)


def test_variable_length_collate_fn_with_equal_min_max_matches_fixed_behavior(tmp_path) -> None:
    path = _write_text(tmp_path, "collate_equal.txt", "abcdefghijklmnopqrstuvwxyz")
    corpus = CharCorpus.from_file(path, vocab_size=64)
    dataset = NextCharSequenceDataset(
        token_ids=corpus.data,
        seq_len=6,
        num_next_chars=2,
        pad_id=corpus.pad_id,
        stride=1,
    )
    sample = dataset[4]
    collate_fn = build_variable_length_collate_fn(
        min_seq_len=6,
        max_seq_len=6,
        pad_id=corpus.pad_id,
        random_lengths=True,
        random_offset=False,
    )

    batch_inputs, batch_targets, lengths = collate_fn([sample])

    assert batch_inputs.shape == (1, 6)
    assert batch_targets.shape == (1, 6, 2)
    assert lengths.tolist() == [6]
    assert torch.equal(batch_inputs[0], sample[0])
    assert torch.equal(batch_targets[0], sample[1])
