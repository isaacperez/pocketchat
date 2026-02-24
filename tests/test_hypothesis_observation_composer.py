import torch

from pocketchat.model import HypothesisObservationComposer


def test_hypothesis_observation_composer_unbatched_shape() -> None:
    module = HypothesisObservationComposer(num_parts=3, window_size=5, observation_dim=6, hidden_dim=10)
    part_heatmaps = torch.randn(3, 5)  # [K, W]

    observations = module(part_heatmaps)

    assert observations.shape == (6,)


def test_hypothesis_observation_composer_batched_shape() -> None:
    module = HypothesisObservationComposer(num_parts=2, window_size=5, observation_dim=4, hidden_dim=8)
    part_heatmaps = torch.randn(2, 3, 2, 5)  # [B, H, K, W]

    observations = module(part_heatmaps)

    assert observations.shape == (2, 3, 4)


def test_hypothesis_observation_composer_aux_shapes() -> None:
    module = HypothesisObservationComposer(num_parts=4, window_size=6, observation_dim=3)
    part_heatmaps = torch.randn(2, 4, 6)  # [H, K, W]

    observations, aux = module(part_heatmaps, return_aux=True)

    assert observations.shape == (2, 3)
    assert aux["flat_heatmaps"].shape == (2, 24)
    assert aux["hidden_features"].shape == (2, 24)


def test_hypothesis_observation_composer_backprop_reaches_inputs_and_parameters() -> None:
    module = HypothesisObservationComposer(num_parts=3, window_size=4, observation_dim=5)
    part_heatmaps = torch.randn(2, 3, 4, requires_grad=True)

    observations = module(part_heatmaps)
    loss = observations.pow(2).mean()
    loss.backward()

    assert part_heatmaps.grad is not None
    assert module.linear_in.weight.grad is not None
    assert module.linear_out.weight.grad is not None
    assert module.linear_in.weight.grad.abs().sum().item() > 0
    assert module.linear_out.weight.grad.abs().sum().item() > 0


def test_hypothesis_observation_composer_validates_inputs() -> None:
    module = HypothesisObservationComposer(num_parts=3, window_size=4, observation_dim=4)

    invalid_heatmaps = (
        torch.randn(4),  # invalid rank
        torch.randn(2, 2, 4),  # K mismatch
        torch.randn(2, 3, 5),  # W mismatch
        torch.ones(2, 3, 4, dtype=torch.long),  # non-float heatmaps
    )

    for part_heatmaps in invalid_heatmaps:
        try:
            module(part_heatmaps)
            raise AssertionError("Expected validation error for invalid heatmaps.")
        except (ValueError, TypeError):
            pass


def test_hypothesis_observation_composer_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"num_parts": 0, "window_size": 4, "observation_dim": 6},
        {"num_parts": 2, "window_size": 0, "observation_dim": 6},
        {"num_parts": 2, "window_size": 4, "observation_dim": 0},
        {"num_parts": 2, "window_size": 4, "observation_dim": 6, "hidden_dim": 0},
    )

    for config in invalid_configs:
        try:
            HypothesisObservationComposer(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass
