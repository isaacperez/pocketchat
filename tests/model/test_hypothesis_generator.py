import torch

from pocketchat.model import HypothesisGenerator


def test_hypothesis_generator_unbatched_shape() -> None:
    module = HypothesisGenerator(hidden_state_dim=8, num_hypotheses=4, hypothesis_dim=6)
    hidden_state = torch.randn(8)

    hypotheses = module(hidden_state)

    assert hypotheses.shape == (4, 6)


def test_hypothesis_generator_batched_shape() -> None:
    module = HypothesisGenerator(hidden_state_dim=8, num_hypotheses=3, hypothesis_dim=5)
    hidden_state = torch.randn(2, 8)

    hypotheses = module(hidden_state)

    assert hypotheses.shape == (2, 3, 5)


def test_hypothesis_generator_keeps_prefix_dims() -> None:
    module = HypothesisGenerator(hidden_state_dim=7, num_hypotheses=2, hypothesis_dim=4)
    hidden_state = torch.randn(2, 3, 7)

    hypotheses = module(hidden_state)

    assert hypotheses.shape == (2, 3, 2, 4)


def test_hypothesis_generator_uses_hidden_dim_when_hypothesis_dim_not_provided() -> None:
    module = HypothesisGenerator(hidden_state_dim=9, num_hypotheses=5)
    hidden_state = torch.randn(4, 9)

    hypotheses = module(hidden_state)

    assert hypotheses.shape == (4, 5, 9)


def test_hypothesis_generator_backprop_reaches_parameters_and_inputs() -> None:
    module = HypothesisGenerator(hidden_state_dim=6, num_hypotheses=3, hypothesis_dim=4)
    hidden_state = torch.randn(2, 6, requires_grad=True)

    hypotheses = module(hidden_state)
    loss = hypotheses.pow(2).mean()
    loss.backward()

    assert hidden_state.grad is not None
    assert module.rms_norm.weight.grad is not None
    assert module.linear_in.weight.grad is not None
    assert module.linear_out.weight.grad is not None
    assert module.rms_norm.weight.grad.abs().sum().item() > 0
    assert module.linear_in.weight.grad.abs().sum().item() > 0
    assert module.linear_out.weight.grad.abs().sum().item() > 0


def test_hypothesis_generator_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"hidden_state_dim": 0, "num_hypotheses": 2, "hypothesis_dim": 4},
        {"hidden_state_dim": 8, "num_hypotheses": 0, "hypothesis_dim": 4},
        {"hidden_state_dim": 8, "num_hypotheses": 2, "hypothesis_dim": 0},
        {"hidden_state_dim": 8, "num_hypotheses": 2, "hypothesis_dim": 4, "eps": 0.0},
    )

    for config in invalid_configs:
        try:
            HypothesisGenerator(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass


def test_hypothesis_generator_validates_forward_input() -> None:
    module = HypothesisGenerator(hidden_state_dim=5, num_hypotheses=2, hypothesis_dim=3)

    invalid_inputs = (
        torch.randn(2, 4),  # last-dim mismatch
        torch.ones(2, 5, dtype=torch.long),  # non-floating input
    )

    for hidden_state in invalid_inputs:
        try:
            module(hidden_state)
            raise AssertionError("Expected validation error for invalid hidden_state input.")
        except (ValueError, TypeError):
            pass
