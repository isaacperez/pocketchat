import torch

from pocketchat.model import HiddenStateDeltaPredictor, HypothesisGenerator, HypothesisUpdater


def test_hidden_state_delta_predictor_unbatched_shape() -> None:
    module = HiddenStateDeltaPredictor(num_hypotheses=3, updated_hypothesis_dim=5, hidden_state_dim=7, hidden_dim=9)
    updated_hypotheses = torch.randn(3, 5)  # [H, D_u]

    delta = module(updated_hypotheses)

    assert delta.shape == (7,)


def test_hidden_state_delta_predictor_batched_shape() -> None:
    module = HiddenStateDeltaPredictor(num_hypotheses=4, updated_hypothesis_dim=6, hidden_state_dim=8)
    updated_hypotheses = torch.randn(2, 4, 6)  # [B, H, D_u]

    delta = module(updated_hypotheses)

    assert delta.shape == (2, 8)


def test_hidden_state_delta_predictor_backprop_reaches_inputs_and_parameters() -> None:
    module = HiddenStateDeltaPredictor(num_hypotheses=3, updated_hypothesis_dim=4, hidden_state_dim=6)
    updated_hypotheses = torch.randn(2, 3, 4, requires_grad=True)

    delta = module(updated_hypotheses)
    loss = delta.pow(2).mean()
    loss.backward()

    assert updated_hypotheses.grad is not None
    assert module.linear_in.weight.grad is not None
    assert module.linear_out.weight.grad is not None
    assert module.linear_in.weight.grad.abs().sum().item() > 0
    assert module.linear_out.weight.grad.abs().sum().item() > 0


def test_hidden_state_delta_predictor_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"num_hypotheses": 0, "updated_hypothesis_dim": 4},
        {"num_hypotheses": 3, "updated_hypothesis_dim": 0},
        {"num_hypotheses": 3, "updated_hypothesis_dim": 4, "hidden_state_dim": 0},
        {"num_hypotheses": 3, "updated_hypothesis_dim": 4, "hidden_dim": 0},
    )

    for config in invalid_configs:
        try:
            HiddenStateDeltaPredictor(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass


def test_hidden_state_delta_predictor_validates_forward_input() -> None:
    module = HiddenStateDeltaPredictor(num_hypotheses=3, updated_hypothesis_dim=4, hidden_state_dim=5)

    invalid_inputs = (
        torch.randn(4),  # rank too low
        torch.randn(2, 2, 4),  # H mismatch
        torch.randn(2, 3, 5),  # D_u mismatch
        torch.ones(2, 3, 4, dtype=torch.long),  # non-float input
    )

    for updated_hypotheses in invalid_inputs:
        try:
            module(updated_hypotheses)
            raise AssertionError("Expected validation error for invalid updated_hypotheses input.")
        except (ValueError, TypeError):
            pass


def test_hidden_state_delta_predictor_integrates_with_generator_and_updater() -> None:
    batch_size = 2
    hidden_state_dim = 8
    num_hypotheses = 4
    hypothesis_dim = 6

    hidden_state = torch.randn(batch_size, hidden_state_dim, requires_grad=True)

    generator = HypothesisGenerator(
        hidden_state_dim=hidden_state_dim,
        num_hypotheses=num_hypotheses,
        hypothesis_dim=hypothesis_dim,
    )
    hypotheses = generator(hidden_state)  # [B, H, D_h]

    observations = torch.randn(batch_size, num_hypotheses, hypothesis_dim)
    updater = HypothesisUpdater(
        hypothesis_dim=hypothesis_dim,
        observation_dim=hypothesis_dim,
        updated_dim=hypothesis_dim,
    )
    updated_hypotheses = updater(hypotheses, observations)  # [B, H, D_u]

    delta_predictor = HiddenStateDeltaPredictor(
        num_hypotheses=num_hypotheses,
        updated_hypothesis_dim=hypothesis_dim,
        hidden_state_dim=hidden_state_dim,
    )
    hidden_state_delta = delta_predictor(updated_hypotheses)  # [B, D_hidden]
    next_hidden_state = hidden_state + hidden_state_delta

    assert hidden_state_delta.shape == (batch_size, hidden_state_dim)
    assert next_hidden_state.shape == (batch_size, hidden_state_dim)

    # End-to-end gradient reaches the original hidden state.
    loss = next_hidden_state.pow(2).mean()
    loss.backward()
    assert hidden_state.grad is not None
