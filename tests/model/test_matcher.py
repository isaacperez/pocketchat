import torch

from pocketchat.model import Matcher


def test_matcher_returns_expected_shape_with_global_references_and_adapters() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 5, 4)  # [B, T, D]
    adapters = torch.randn(3, 4)  # [M, D]
    references = torch.randn(3, 4)  # [M, D]

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (2, 5, 3)  # [B, T, M]
    assert float(heatmap.max().item()) <= 1.0 + 1e-5
    assert float(heatmap.min().item()) >= -1.0 - 1e-5


def test_matcher_returns_expected_shape_with_batched_references_and_adapters() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 7, 6)  # [B, T, D]
    adapters = torch.randn(2, 4, 6)  # [B, M, D]
    references = torch.randn(2, 4, 6)  # [B, M, D]

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (2, 7, 4)  # [B, T, M]


def test_matcher_preserves_input_structure_for_hierarchical_inputs() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 3, 5, 4)  # [B, H, T, D]
    adapters = torch.randn(2, 3, 6, 4)  # [B, H, M, D]
    references = torch.randn(2, 3, 6, 4)  # [B, H, M, D]

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (2, 3, 5, 6)  # [B, H, T, M]


def test_matcher_simple_numeric_case() -> None:
    matcher = Matcher()

    # B=1, T=2, D=2
    input_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    # B=1, M=1, D=2
    adapters = torch.tensor([[[1.0, 1.0]]])
    references = torch.tensor([[[1.0, 0.0]]])

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (1, 2, 1)  # [B, T, M]
    # position 0: cos([1,0], [1,0]) = 1
    assert heatmap[0, 0, 0].item() == 1.0
    # position 1: cos([0,1], [1,0]) = 0
    assert abs(heatmap[0, 1, 0].item()) < 1e-6


def test_matcher_adapter_masks_dimension_with_different_reference() -> None:
    matcher = Matcher()

    # B=1, T=2, D=3
    input_embeddings = torch.tensor(
        [
            [
                [1.0, 100.0, 0.0],
                [0.0, -50.0, 1.0],
            ]
        ]
    )
    # B=1, M=1, D=3; adapter masks out dimension 1
    adapters = torch.tensor([[[1.0, 0.0, 1.0]]])
    # Different vector than adapter on purpose
    references = torch.tensor([[[1.0, 1.0, 0.0]]])

    assert not torch.allclose(adapters, references)

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (1, 2, 1)
    # pos0 adapted: [1,0,0], reference: [1,1,0] -> cos = 1/sqrt(2)
    assert torch.allclose(heatmap[0, 0, 0], torch.tensor(1.0 / (2.0**0.5)), atol=1e-6)
    # pos1 adapted: [0,0,1], reference: [1,1,0] -> cos = 0
    assert abs(heatmap[0, 1, 0].item()) < 1e-6


def test_matcher_validates_shapes_and_types() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 5, 4)

    bad_shape_cases = (
        (torch.randn(3, 4), torch.randn(2, 3, 4)),  # references/adapters shape mismatch
        (torch.randn(3, 5), torch.randn(3, 5)),  # D mismatch with input
    )

    for references, adapters in bad_shape_cases:
        try:
            _ = matcher(input_embeddings, references, adapters)
            raise AssertionError("Expected ValueError for invalid shape configuration.")
        except ValueError:
            pass

    try:
        _ = matcher(input_embeddings.long(), torch.randn(3, 4), torch.randn(3, 4))
        raise AssertionError("Expected TypeError for non-floating input embeddings.")
    except TypeError:
        pass
