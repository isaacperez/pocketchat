import torch

from pocketchat.model import (
    Glimpse,
    HypothesisDecomposer,
    HypothesisObservationComposer,
    HypothesisUpdater,
    Matcher,
)


def test_hypothesis_updater_unbatched_shape() -> None:
    module = HypothesisUpdater(hypothesis_dim=6)
    hypotheses = torch.randn(6)
    observations = torch.randn(6)

    updated = module(hypotheses, observations)

    assert updated.shape == (6,)


def test_hypothesis_updater_batched_shape_with_custom_dims() -> None:
    module = HypothesisUpdater(hypothesis_dim=5, observation_dim=7, updated_dim=4, hidden_dim=9)
    hypotheses = torch.randn(2, 3, 5)
    observations = torch.randn(2, 3, 7)

    updated = module(hypotheses, observations)

    assert updated.shape == (2, 3, 4)


def test_hypothesis_updater_backprop_reaches_inputs_and_parameters() -> None:
    module = HypothesisUpdater(hypothesis_dim=4)
    hypotheses = torch.randn(2, 3, 4, requires_grad=True)
    observations = torch.randn(2, 3, 4, requires_grad=True)

    updated = module(hypotheses, observations)
    loss = updated.pow(2).mean()
    loss.backward()

    assert hypotheses.grad is not None
    assert observations.grad is not None
    assert module.linear_in.weight.grad is not None
    assert module.linear_out.weight.grad is not None
    assert module.linear_in.weight.grad.abs().sum().item() > 0
    assert module.linear_out.weight.grad.abs().sum().item() > 0


def test_hypothesis_updater_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"hypothesis_dim": 0},
        {"hypothesis_dim": 4, "observation_dim": 0},
        {"hypothesis_dim": 4, "updated_dim": 0},
        {"hypothesis_dim": 4, "hidden_dim": 0},
    )

    for config in invalid_configs:
        try:
            HypothesisUpdater(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass


def test_hypothesis_updater_validates_forward_input() -> None:
    module = HypothesisUpdater(hypothesis_dim=4, observation_dim=5)

    invalid_cases = (
        (torch.randn(2, 4), torch.randn(2, 3, 5)),  # prefix mismatch
        (torch.randn(2, 3, 5), torch.randn(2, 3, 5)),  # hypothesis dim mismatch
        (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),  # observation dim mismatch
        (torch.ones(2, 3, 4, dtype=torch.long), torch.randn(2, 3, 5)),  # non-float hypotheses
        (torch.randn(2, 3, 4), torch.ones(2, 3, 5, dtype=torch.long)),  # non-float observations
    )

    for hypotheses, observations in invalid_cases:
        try:
            module(hypotheses, observations)
            raise AssertionError("Expected validation error for invalid inputs.")
        except (ValueError, TypeError):
            pass


def test_hypothesis_updater_integrates_with_current_pipeline() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, window_size, embedding_dim = 8, 5, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim)
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=6,
    )
    center_logits, zoom_logits, references, adapters = decomposer(hypotheses)

    glimpse = Glimpse(window_size=window_size, squeeze_output=False)
    windows_flat = glimpse(
        input_embeddings,
        center_logits=center_logits.reshape(batch_size, -1),
        zoom_logits=zoom_logits.reshape(batch_size, -1),
    )  # [B, H*K, W, D]
    part_windows = windows_flat.reshape(batch_size, num_hypotheses, num_parts, window_size, embedding_dim)

    matcher = Matcher()
    part_heatmaps = matcher(part_windows, references, adapters)  # [B, H, K, W]

    composer = HypothesisObservationComposer(
        num_parts=num_parts,
        window_size=window_size,
        observation_dim=embedding_dim,
        hidden_dim=12,
    )
    observations = composer(part_heatmaps)  # [B, H, D]

    updater = HypothesisUpdater(hypothesis_dim=embedding_dim, observation_dim=embedding_dim, updated_dim=embedding_dim)
    updated_hypotheses = updater(hypotheses, observations)

    assert updated_hypotheses.shape == (batch_size, num_hypotheses, embedding_dim)
