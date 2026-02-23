import torch

from pocketchat.model import Glimpse


def _fraction_to_logit(fraction: float) -> float:
    if fraction <= 0.0:
        return float("-inf")
    if fraction >= 1.0:
        return float("inf")
    return float(torch.logit(torch.tensor(fraction)).item())


def test_glimpse_matches_contiguous_slice_at_min_zoom() -> None:
    embeddings = torch.arange(9, dtype=torch.float32).unsqueeze(-1)  # [L, D=1]
    glimpse = Glimpse(
        window_size=5,
        num_glimpses=1,
        init_center_fraction=0.5,
        init_zoom_fraction=0.0,
        learnable=False,
    )

    out = glimpse(embeddings, zoom_logits=torch.tensor(-20.0))  # [W, D]

    expected = embeddings[2:7]
    assert out.shape == (5, 1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_glimpse_linear_interpolation_non_integer_positions() -> None:
    embeddings = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0]])  # L=5, D=1
    glimpse = Glimpse(
        window_size=3,
        num_glimpses=1,
        init_center_fraction=0.625,  # c = 1 + 0.625*(3-1) = 2.25
        init_zoom_fraction=0.0,  # s = 1
        learnable=False,
    )

    out = glimpse(embeddings, zoom_logits=torch.tensor(-20.0))  # positions: ~1.25, ~2.25, ~3.25

    expected = torch.tensor([[12.5], [22.5], [32.5]])
    assert torch.allclose(out, expected, atol=1e-5)


def test_glimpse_clamps_to_borders_when_positions_go_outside_sequence() -> None:
    embeddings = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0]])  # L=5, D=1
    glimpse = Glimpse(
        window_size=3,
        num_glimpses=2,
        init_center_fraction=0.5,
        init_zoom_fraction=0.0,
        learnable=False,
    )

    center_logits = torch.tensor([-20.0, 20.0])  # approx center_fraction=0 and 1
    zoom_logits = torch.tensor([20.0, 20.0])  # approx zoom_fraction=1 (max zoom)
    out = glimpse(embeddings, center_logits=center_logits, zoom_logits=zoom_logits)  # [G, W, D]

    assert out.shape == (2, 3, 1)
    # Left-border glimpse: first sampled position is < 0, should replicate index 0.
    assert out[0, 0, 0].item() == embeddings[0, 0].item()
    # Right-border glimpse: last sampled position is > L-1, should replicate index L-1.
    assert out[1, -1, 0].item() == embeddings[-1, 0].item()


def test_glimpse_supports_batch_and_multiple_glimpses_with_lengths() -> None:
    embeddings = torch.randn(2, 6, 4)  # [B=2, L=6, D=4]
    lengths = torch.tensor([6, 4])  # second sequence has valid region [0..3]
    glimpse = Glimpse(window_size=4, num_glimpses=3, learnable=False)

    out, aux = glimpse(
        embeddings,
        center_logits=torch.zeros(3),  # per-glimpse center, shared across batch
        zoom_logits=torch.zeros(2, 1),  # per-batch zoom, shared across glimpses
        lengths=lengths,
        return_aux=True,
    )

    assert out.shape == (2, 3, 4, 4)
    assert aux["sample_positions"].shape == (2, 3, 4)
    assert int(aux["left_indices"][1].max().item()) <= 3
    assert int(aux["right_indices"][1].max().item()) <= 3


def test_glimpse_backprop_reaches_center_and_zoom_logits() -> None:
    glimpse = Glimpse(
        window_size=5,
        num_glimpses=1,
        init_center_fraction=0.35,
        init_zoom_fraction=0.65,
        learnable=True,
        squeeze_output=False,
    )
    embeddings = torch.randn(2, 8, 3)

    out = glimpse(embeddings)  # [B, G, W, D]
    loss = out.pow(2).mean()
    loss.backward()

    assert glimpse.center_logits.grad is not None
    assert glimpse.zoom_logits.grad is not None
    assert glimpse.center_logits.grad.abs().sum().item() > 0
    assert glimpse.zoom_logits.grad.abs().sum().item() > 0


def test_glimpse_rescales_correctly_for_all_center_zoom_grid_values() -> None:
    seq_len = 9
    window_size = 5
    embeddings = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)  # [L, D=1]

    center_fractions = [0.0, 0.5, 1.0]
    zoom_fractions = [0.0, 0.5, 1.0]
    center_grid = [c for c in center_fractions for _ in zoom_fractions]
    zoom_grid = [z for _ in center_fractions for z in zoom_fractions]
    num_glimpses = len(center_grid)

    glimpse = Glimpse(
        window_size=window_size,
        num_glimpses=num_glimpses,
        learnable=False,
    )
    center_logits = torch.tensor([_fraction_to_logit(c) for c in center_grid], dtype=torch.float32)
    zoom_logits = torch.tensor([_fraction_to_logit(z) for z in zoom_grid], dtype=torch.float32)

    out, aux = glimpse(
        embeddings,
        center_logits=center_logits,
        zoom_logits=zoom_logits,
        return_aux=True,
    )

    assert out.shape == (num_glimpses, window_size, 1)

    half_window = (window_size - 1) / 2.0
    center_min = half_window
    center_max = (seq_len - 1) - half_window
    max_spacing = (seq_len - 1) / (window_size - 1)
    offsets = torch.arange(window_size, dtype=torch.float32) - half_window

    expected_positions = []
    for center_fraction, zoom_fraction in zip(center_grid, zoom_grid):
        center = center_min + center_fraction * (center_max - center_min)
        spacing = 1.0 + zoom_fraction * (max_spacing - 1.0)
        expected_positions.append(center + spacing * offsets)
    expected_positions_t = torch.stack(expected_positions, dim=0)
    expected_clamped_positions_t = expected_positions_t.clamp(0.0, float(seq_len - 1))

    assert torch.allclose(aux["sample_positions"][0], expected_positions_t, atol=1e-6)
    # With E[i]=i and linear interpolation + border clamp, output equals clamped sample positions.
    assert torch.allclose(out.squeeze(-1), expected_clamped_positions_t, atol=1e-5)


def test_glimpse_rescales_center_zoom_grid_correctly_with_batched_lengths() -> None:
    seq_len = 9
    window_size = 5
    lengths = torch.tensor([9, 7])
    emb0 = torch.arange(seq_len, dtype=torch.float32)  # value = 0 + 1*i
    emb1 = 100.0 + (2.0 * torch.arange(seq_len, dtype=torch.float32))  # value = 100 + 2*i
    embeddings = torch.stack([emb0, emb1], dim=0).unsqueeze(-1)  # [B=2, L=9, D=1]

    center_fractions = [0.0, 0.5, 1.0]
    zoom_fractions = [0.0, 0.5, 1.0]
    center_grid = [c for c in center_fractions for _ in zoom_fractions]
    zoom_grid = [z for _ in center_fractions for z in zoom_fractions]
    num_glimpses = len(center_grid)

    glimpse = Glimpse(
        window_size=window_size,
        num_glimpses=num_glimpses,
        learnable=False,
        squeeze_output=False,
    )
    center_logits = torch.tensor([_fraction_to_logit(c) for c in center_grid], dtype=torch.float32)
    zoom_logits = torch.tensor([_fraction_to_logit(z) for z in zoom_grid], dtype=torch.float32)

    out, aux = glimpse(
        embeddings,
        center_logits=center_logits,  # shared center grid across batch
        zoom_logits=zoom_logits,  # shared zoom grid across batch
        lengths=lengths,
        return_aux=True,
    )

    assert out.shape == (2, num_glimpses, window_size, 1)
    assert aux["sample_positions"].shape == (2, num_glimpses, window_size)

    offsets = torch.arange(window_size, dtype=torch.float32) - ((window_size - 1) / 2.0)
    expected_outputs = []
    expected_positions = []
    for length, bias, slope in ((9, 0.0, 1.0), (7, 100.0, 2.0)):
        half_window = (window_size - 1) / 2.0
        center_min = half_window
        center_max = (length - 1) - half_window
        max_spacing = (length - 1) / (window_size - 1)

        positions_for_batch = []
        outputs_for_batch = []
        for center_fraction, zoom_fraction in zip(center_grid, zoom_grid):
            center = center_min + center_fraction * (center_max - center_min)
            spacing = 1.0 + zoom_fraction * (max_spacing - 1.0)
            positions = center + spacing * offsets
            clamped_positions = positions.clamp(0.0, float(length - 1))
            positions_for_batch.append(positions)
            outputs_for_batch.append(bias + (slope * clamped_positions))

        expected_positions.append(torch.stack(positions_for_batch, dim=0))
        expected_outputs.append(torch.stack(outputs_for_batch, dim=0))

    expected_positions_t = torch.stack(expected_positions, dim=0)
    expected_outputs_t = torch.stack(expected_outputs, dim=0).unsqueeze(-1)

    assert torch.allclose(aux["sample_positions"], expected_positions_t, atol=1e-6)
    assert torch.allclose(out, expected_outputs_t, atol=1e-5)
