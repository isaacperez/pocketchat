import torch

from pocketchat.model import Matcher


def test_matcher_returns_expected_shape_with_global_adapters_and_references() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 5, 4)  # [B, T, D]
    adapters = torch.randn(3, 4)  # [M, D]
    references = torch.randn(3, 4)  # [M, D]

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (2, 3, 5)
    assert float(heatmap.max().item()) <= 1.0 + 1e-5
    assert float(heatmap.min().item()) >= -1.0 - 1e-5


def test_matcher_returns_expected_shape_with_batched_adapters_and_references() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 7, 6)  # [B, T, D]
    adapters = torch.randn(2, 4, 6)  # [B, M, D]
    references = torch.randn(2, 4, 6)  # [B, M, D]

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (2, 4, 7)


def test_matcher_simple_numeric_case() -> None:
    matcher = Matcher()

    # B=1, T=2, D=2
    input_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    # M=1, D=2
    adapters = torch.tensor([[1.0, 1.0]])
    references = torch.tensor([[1.0, 0.0]])

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (1, 1, 2)
    # position 0: cos([1,0], [1,0]) = 1
    assert heatmap[0, 0, 0].item() == 1.0
    # position 1: cos([0,1], [1,0]) = 0
    assert abs(heatmap[0, 0, 1].item()) < 1e-6


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
    # M=1, D=3; adapter masks out dimension 1
    adapters = torch.tensor([[1.0, 0.0, 1.0]])
    # Different vector than adapter on purpose
    references = torch.tensor([[1.0, 1.0, 0.0]])

    assert not torch.allclose(adapters, references)

    heatmap = matcher(input_embeddings, references, adapters)

    assert heatmap.shape == (1, 1, 2)
    # pos0 adapted: [1,0,0], reference: [1,1,0] -> cos = 1/sqrt(2)
    assert torch.allclose(heatmap[0, 0, 0], torch.tensor(1.0 / (2.0**0.5)), atol=1e-6)
    # pos1 adapted: [0,0,1], reference: [1,1,0] -> cos = 0
    assert abs(heatmap[0, 0, 1].item()) < 1e-6


def test_matcher_validates_shapes() -> None:
    matcher = Matcher()
    input_embeddings = torch.randn(2, 5, 4)

    invalid_cases = (
        (torch.randn(3, 4), torch.randn(2, 3, 4)),  # ndim mismatch adapters/references
        (torch.randn(3, 5), torch.randn(3, 5)),  # D mismatch with input
        (torch.randn(2, 3, 4), torch.randn(3, 3, 4)),  # B mismatch in 3D mode
        (torch.randn(2, 3, 4), torch.randn(2, 4, 4)),  # shape mismatch refs/adapters
    )

    for references, adapters in invalid_cases:
        try:
            _ = matcher(input_embeddings, references, adapters)
            raise AssertionError("Expected ValueError for invalid shape configuration.")
        except ValueError:
            pass
