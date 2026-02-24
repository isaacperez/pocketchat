import torch

from pocketchat.model import NextCharPredictor


def test_next_char_predictor_unbatched_shape() -> None:
    module = NextCharPredictor(
        num_hypotheses=3,
        num_iterations=2,
        updated_hypothesis_dim=4,
        vocab_size=11,
        num_next_chars=5,
        hidden_dim=13,
    )
    history = torch.randn(3, 2, 4)  # [H, N_iter, D_u]

    logits = module(history)

    assert logits.shape == (5, 11)


def test_next_char_predictor_batched_shape() -> None:
    module = NextCharPredictor(
        num_hypotheses=2,
        num_iterations=3,
        updated_hypothesis_dim=6,
        vocab_size=17,
        num_next_chars=4,
    )
    history = torch.randn(2, 2, 3, 6)  # [B, H, N_iter, D_u]

    logits = module(history)

    assert logits.shape == (2, 4, 17)


def test_next_char_predictor_backprop_reaches_inputs_and_parameters() -> None:
    module = NextCharPredictor(
        num_hypotheses=2,
        num_iterations=2,
        updated_hypothesis_dim=5,
        vocab_size=9,
        num_next_chars=3,
    )
    history = torch.randn(2, 2, 2, 5, requires_grad=True)

    logits = module(history)
    loss = logits.pow(2).mean()
    loss.backward()

    assert history.grad is not None
    assert module.linear_in.weight.grad is not None
    assert module.linear_out.weight.grad is not None
    assert module.linear_in.weight.grad.abs().sum().item() > 0
    assert module.linear_out.weight.grad.abs().sum().item() > 0


def test_next_char_predictor_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"num_hypotheses": 0, "num_iterations": 2, "updated_hypothesis_dim": 4, "vocab_size": 10, "num_next_chars": 2},
        {"num_hypotheses": 2, "num_iterations": 0, "updated_hypothesis_dim": 4, "vocab_size": 10, "num_next_chars": 2},
        {"num_hypotheses": 2, "num_iterations": 2, "updated_hypothesis_dim": 0, "vocab_size": 10, "num_next_chars": 2},
        {"num_hypotheses": 2, "num_iterations": 2, "updated_hypothesis_dim": 4, "vocab_size": 0, "num_next_chars": 2},
        {"num_hypotheses": 2, "num_iterations": 2, "updated_hypothesis_dim": 4, "vocab_size": 10, "num_next_chars": 0},
        {"num_hypotheses": 2, "num_iterations": 2, "updated_hypothesis_dim": 4, "vocab_size": 10, "num_next_chars": 2, "hidden_dim": 0},
        {
            "num_hypotheses": 2,
            "num_iterations": 2,
            "updated_hypothesis_dim": 4,
            "vocab_size": 10,
            "num_next_chars": 2,
            "tie_with_embedding": True,
            "tied_embedding_dim": 0,
        },
    )

    for config in invalid_configs:
        try:
            NextCharPredictor(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass


def test_next_char_predictor_tied_mode_shape() -> None:
    module = NextCharPredictor(
        num_hypotheses=2,
        num_iterations=3,
        updated_hypothesis_dim=4,
        vocab_size=7,
        num_next_chars=5,
        tie_with_embedding=True,
        tied_embedding_dim=6,
    )
    history = torch.randn(2, 2, 3, 4)
    embedding_weight = torch.randn(7, 6)

    logits = module(history, embedding_weight=embedding_weight)

    assert logits.shape == (2, 5, 7)


def test_next_char_predictor_validates_forward_input() -> None:
    module = NextCharPredictor(
        num_hypotheses=2,
        num_iterations=3,
        updated_hypothesis_dim=4,
        vocab_size=8,
        num_next_chars=2,
    )

    invalid_histories = (
        torch.randn(2, 4),  # rank too low
        torch.randn(2, 2, 2, 4),  # N_iter mismatch
        torch.randn(2, 3, 3, 4),  # H mismatch
        torch.randn(2, 2, 3, 5),  # D_u mismatch
        torch.ones(2, 2, 3, 4, dtype=torch.long),  # non-float input
    )

    for history in invalid_histories:
        try:
            module(history)
            raise AssertionError("Expected validation error for invalid hypothesis_update_history input.")
        except (ValueError, TypeError):
            pass


def test_next_char_predictor_tied_mode_validates_embedding_weight() -> None:
    module = NextCharPredictor(
        num_hypotheses=2,
        num_iterations=2,
        updated_hypothesis_dim=3,
        vocab_size=8,
        num_next_chars=2,
        tie_with_embedding=True,
        tied_embedding_dim=5,
    )
    history = torch.randn(2, 2, 2, 3)

    invalid_embedding_weights = (
        None,  # required in tied mode
        torch.randn(8),  # invalid rank
        torch.randn(7, 5),  # vocab mismatch
        torch.randn(8, 4),  # embedding dim mismatch
        torch.ones(8, 5, dtype=torch.long),  # non-float
    )

    for embedding_weight in invalid_embedding_weights:
        try:
            module(history, embedding_weight=embedding_weight)
            raise AssertionError("Expected validation error for invalid embedding_weight.")
        except (ValueError, TypeError):
            pass
