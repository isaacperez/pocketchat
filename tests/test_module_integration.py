import torch

from pocketchat.model import Glimpse, HypothesisDecomposer, HypothesisObservationComposer, Matcher


def test_integration_decomposer_glimpse() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, window_size, embedding_dim = 9, 5, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim)  # [B, H, D]
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)  # [B, L, D]

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=6,
    )
    center_logits, zoom_logits, _, _ = decomposer(hypotheses)  # [B, H, K], [B, H, K], ...

    glimpse = Glimpse(window_size=window_size, squeeze_output=False)
    windows = glimpse(
        input_embeddings,
        center_logits=center_logits.reshape(batch_size, -1),
        zoom_logits=zoom_logits.reshape(batch_size, -1),
    )  # [B, H*K, W, D]

    assert windows.shape == (batch_size, num_hypotheses * num_parts, window_size, embedding_dim)


def test_integration_decomposer_matcher() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, embedding_dim = 7, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim)  # [B, H, D]
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)  # [B, L, D]

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=6,
    )
    _, _, references, adapters = decomposer(hypotheses)  # [B, H, K, D], [B, H, K, D]

    matcher = Matcher()
    heatmaps = matcher(input_embeddings, references, adapters)

    assert heatmaps.shape == (batch_size, seq_len, num_hypotheses, num_parts)


def test_integration_glimpse_matcher() -> None:
    batch_size, seq_len, embedding_dim = 2, 8, 4
    num_glimpses, window_size = 5, 4

    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    center_logits = torch.randn(batch_size, num_glimpses)
    zoom_logits = torch.randn(batch_size, num_glimpses)

    glimpse = Glimpse(window_size=window_size, squeeze_output=False)
    windows = glimpse(input_embeddings, center_logits=center_logits, zoom_logits=zoom_logits)  # [B, G, W, D]

    references = torch.randn(batch_size, num_glimpses, embedding_dim)  # [B, G, D]
    adapters = torch.randn(batch_size, num_glimpses, embedding_dim)  # [B, G, D]

    matcher = Matcher()
    heatmaps = matcher(windows, references, adapters)  # [B, G, W]

    assert heatmaps.shape == (batch_size, num_glimpses, window_size)


def test_integration_matcher_composer() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    window_size, embedding_dim = 5, 4

    part_windows = torch.randn(batch_size, num_hypotheses, num_parts, window_size, embedding_dim)
    references = torch.randn(batch_size, num_hypotheses, num_parts, embedding_dim)
    adapters = torch.randn(batch_size, num_hypotheses, num_parts, embedding_dim)

    matcher = Matcher()
    part_heatmaps = matcher(part_windows, references, adapters)  # [B, H, K, W]

    composer = HypothesisObservationComposer(
        num_parts=num_parts,
        window_size=window_size,
        observation_dim=embedding_dim,
        hidden_dim=10,
    )
    observations = composer(part_heatmaps)

    assert observations.shape == (batch_size, num_hypotheses, embedding_dim)


def test_integration_full_pipeline_decomposer_glimpse_matcher_composer() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, window_size, embedding_dim = 8, 5, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim, requires_grad=True)
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=6,
    )
    center_logits, zoom_logits, references, adapters = decomposer(hypotheses)  # [B,H,K], [B,H,K], [B,H,K,D], [B,H,K,D]

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

    assert observations.shape == (batch_size, num_hypotheses, embedding_dim)

    # Sanity check that gradients flow end-to-end through all modules.
    loss = observations.pow(2).mean()
    loss.backward()
    assert hypotheses.grad is not None
