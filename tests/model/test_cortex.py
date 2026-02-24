import torch

from pocketchat.model import Cortex


def test_cortex_forward_batched_pipeline_shapes() -> None:
    model = Cortex(
        vocab_size=32,
        embedding_dim=8,
        padding_idx=0,
        hidden_state_dim=6,
        num_hypotheses=3,
        num_parts=2,
        window_size=4,
        num_iterations=3,
        num_next_chars=4,
        hypothesis_dim=5,
        observation_dim=5,
        updated_hypothesis_dim=5,
    )
    token_ids = torch.randint(1, 32, (2, 7), dtype=torch.long)

    outputs = model(token_ids)

    assert outputs["token_embeddings"].shape == (2, 7, 8)
    assert outputs["base_hidden_state"].shape == (2, 6)
    assert outputs["final_hidden_state"].shape == (2, 6)
    assert outputs["hidden_state_delta_history"].shape == (2, 6, 3)
    assert outputs["hypothesis_update_history"].shape == (2, 3, 3, 5)
    assert outputs["last_updated_hypotheses"].shape == (2, 3, 5)
    assert outputs["next_char_logits"].shape == (2, 4, 32)
    assert outputs["next_char_predictions"].shape == (2, 4)
    assert outputs["next_char_predictions"].dtype == torch.long
    assert model.next_char_predictor.tie_with_embedding


def test_cortex_forward_unbatched_shapes() -> None:
    model = Cortex(
        vocab_size=20,
        embedding_dim=6,
        padding_idx=0,
        hidden_state_dim=6,
        num_hypotheses=2,
        num_parts=2,
        window_size=3,
        num_iterations=2,
        num_next_chars=3,
        hypothesis_dim=4,
        observation_dim=4,
        updated_hypothesis_dim=4,
    )
    token_ids = torch.randint(1, 20, (5,), dtype=torch.long)

    outputs = model(token_ids)

    assert outputs["token_embeddings"].shape == (5, 6)
    assert outputs["base_hidden_state"].shape == (6,)
    assert outputs["final_hidden_state"].shape == (6,)
    assert outputs["hidden_state_delta_history"].shape == (6, 2)
    assert outputs["hypothesis_update_history"].shape == (2, 2, 4)
    assert outputs["last_updated_hypotheses"].shape == (2, 4)
    assert outputs["next_char_logits"].shape == (3, 20)
    assert outputs["next_char_predictions"].shape == (3,)
    assert outputs["next_char_predictions"].dtype == torch.long


def test_cortex_supports_lengths_for_hidden_state_and_glimpse() -> None:
    model = Cortex(
        vocab_size=16,
        embedding_dim=4,
        padding_idx=0,
        hidden_state_dim=4,
        num_hypotheses=2,
        num_parts=2,
        window_size=3,
        num_iterations=2,
    )
    token_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([3, 5], dtype=torch.long)

    outputs = model(token_ids, lengths=lengths)

    assert outputs["token_embeddings"].shape == (2, 5, 4)
    assert outputs["base_hidden_state"].shape == (2, 4)
    assert outputs["final_hidden_state"].shape == (2, 4)


def test_cortex_backprop_reaches_embedding_parameters() -> None:
    model = Cortex(
        vocab_size=24,
        embedding_dim=6,
        padding_idx=0,
        hidden_state_dim=6,
        num_hypotheses=3,
        num_parts=2,
        window_size=4,
        num_iterations=2,
    )
    token_ids = torch.randint(1, 24, (2, 6), dtype=torch.long)

    outputs = model(token_ids)
    loss = outputs["final_hidden_state"].pow(2).mean()
    loss.backward()

    assert model.char_embeddings.weight.grad is not None
    assert model.char_embeddings.weight.grad.abs().sum().item() > 0
    assert model.base_hidden_state.grad is not None
    assert model.base_hidden_state.grad.abs().sum().item() > 0


def test_cortex_uses_trainable_base_hidden_state_parameter() -> None:
    model = Cortex(vocab_size=12, embedding_dim=4, hidden_state_dim=6, num_hypotheses=2, num_parts=2, num_iterations=1)
    token_ids = torch.randint(1, 12, (3, 5), dtype=torch.long)

    outputs = model(token_ids)

    assert model.base_hidden_state.requires_grad
    assert outputs["base_hidden_state"].shape == (3, 6)
    assert torch.allclose(outputs["base_hidden_state"][0], model.base_hidden_state.detach(), atol=1e-6)
    assert torch.allclose(outputs["base_hidden_state"][1], model.base_hidden_state.detach(), atol=1e-6)


def test_cortex_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"vocab_size": 0, "embedding_dim": 8},
        {"vocab_size": 10, "embedding_dim": 0},
        {"vocab_size": 10, "embedding_dim": 8, "num_hypotheses": 0},
        {"vocab_size": 10, "embedding_dim": 8, "num_parts": 0},
        {"vocab_size": 10, "embedding_dim": 8, "window_size": 0},
        {"vocab_size": 10, "embedding_dim": 8, "num_iterations": 0},
        {"vocab_size": 10, "embedding_dim": 8, "num_next_chars": 0},
        {"vocab_size": 10, "embedding_dim": 8, "matcher_eps": 0.0},
    )

    for config in invalid_configs:
        try:
            Cortex(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass


def test_cortex_validates_token_ids_input() -> None:
    model = Cortex(vocab_size=10, embedding_dim=4)

    invalid_inputs = (
        torch.randn(2, 5),  # floating point token ids
        torch.ones(2, 3, 4, dtype=torch.long),  # invalid rank
    )

    for token_ids in invalid_inputs:
        try:
            model(token_ids)
            raise AssertionError("Expected validation error for invalid token_ids.")
        except (TypeError, ValueError):
            pass
