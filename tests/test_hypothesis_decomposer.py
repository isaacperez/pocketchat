import torch

from pocketchat.model import Glimpse, HypothesisDecomposer, Matcher


def test_hypothesis_decomposer_unbatched_shapes() -> None:
    module = HypothesisDecomposer(hypothesis_dim=8, num_parts=3, matcher_dim=6, hidden_dim=10)
    hypotheses = torch.randn(8)

    center_logits, zoom_logits, references, adapters = module(hypotheses)

    assert center_logits.shape == (3,)
    assert zoom_logits.shape == (3,)
    assert references.shape == (3, 6)
    assert adapters.shape == (3, 6)


def test_hypothesis_decomposer_batched_shapes() -> None:
    module = HypothesisDecomposer(hypothesis_dim=7, num_parts=2, matcher_dim=5, hidden_dim=9)
    hypotheses = torch.randn(2, 4, 7)

    center_logits, zoom_logits, references, adapters = module(hypotheses)

    assert center_logits.shape == (2, 4, 2)
    assert zoom_logits.shape == (2, 4, 2)
    assert references.shape == (2, 4, 2, 5)
    assert adapters.shape == (2, 4, 2, 5)


def test_hypothesis_decomposer_defaults_matcher_and_hidden_dim() -> None:
    module = HypothesisDecomposer(hypothesis_dim=11, num_parts=4)
    hypotheses = torch.randn(3, 11)

    center_logits, zoom_logits, references, adapters = module(hypotheses)

    assert center_logits.shape == (3, 4)
    assert zoom_logits.shape == (3, 4)
    assert references.shape == (3, 4, 11)
    assert adapters.shape == (3, 4, 11)


def test_hypothesis_decomposer_backprop_reaches_all_heads() -> None:
    module = HypothesisDecomposer(hypothesis_dim=6, num_parts=3, matcher_dim=6, hidden_dim=8)
    hypotheses = torch.randn(2, 3, 6, requires_grad=True)

    center_logits, zoom_logits, references, adapters = module(hypotheses)
    loss = (
        center_logits.pow(2).mean()
        + zoom_logits.pow(2).mean()
        + references.pow(2).mean()
        + adapters.pow(2).mean()
    )
    loss.backward()

    assert hypotheses.grad is not None
    assert module.rms_norm.weight.grad is not None
    assert module.center_head[0].weight.grad is not None
    assert module.zoom_head[0].weight.grad is not None
    assert module.reference_head[0].weight.grad is not None
    assert module.adapter_head[0].weight.grad is not None
    assert module.rms_norm.weight.grad.abs().sum().item() > 0
    assert module.center_head[0].weight.grad.abs().sum().item() > 0
    assert module.zoom_head[0].weight.grad.abs().sum().item() > 0
    assert module.reference_head[0].weight.grad.abs().sum().item() > 0
    assert module.adapter_head[0].weight.grad.abs().sum().item() > 0


def test_hypothesis_decomposer_connects_with_glimpse_and_matcher() -> None:
    hypotheses = torch.randn(2, 3, 4)  # [B, H, D]
    input_embeddings = torch.randn(2, 7, 4)  # [B, L, D]

    decomposer = HypothesisDecomposer(hypothesis_dim=4, num_parts=2, matcher_dim=4, hidden_dim=6)
    center_logits, zoom_logits, references, adapters = decomposer(hypotheses)

    glimpse = Glimpse(window_size=5, squeeze_output=False)
    # Glimpse expects [B, G] logits, so flatten (H, K) into G = H*K.
    windows = glimpse(
        input_embeddings,
        center_logits=center_logits.reshape(2, -1),
        zoom_logits=zoom_logits.reshape(2, -1),
    )
    assert windows.shape == (2, 6, 5, 4)

    matcher = Matcher()
    heatmap = matcher(input_embeddings, references, adapters)
    assert heatmap.shape == (2, 7, 3, 2)


def test_hypothesis_decomposer_validates_constructor_arguments() -> None:
    invalid_configs = (
        {"hypothesis_dim": 0, "num_parts": 2},
        {"hypothesis_dim": 8, "num_parts": 0},
        {"hypothesis_dim": 8, "num_parts": 2, "matcher_dim": 0},
        {"hypothesis_dim": 8, "num_parts": 2, "hidden_dim": 0},
        {"hypothesis_dim": 8, "num_parts": 2, "eps": 0.0},
    )

    for config in invalid_configs:
        try:
            HypothesisDecomposer(**config)
            raise AssertionError(f"Expected ValueError for config={config}")
        except ValueError:
            pass


def test_hypothesis_decomposer_validates_forward_input() -> None:
    module = HypothesisDecomposer(hypothesis_dim=5, num_parts=3, matcher_dim=4)

    invalid_inputs = (
        torch.randn(2, 4),  # last-dim mismatch
        torch.ones(2, 5, dtype=torch.long),  # non-floating input
    )

    for hypothesis_embeddings in invalid_inputs:
        try:
            module(hypothesis_embeddings)
            raise AssertionError("Expected validation error for invalid hypothesis_embeddings input.")
        except (ValueError, TypeError):
            pass
