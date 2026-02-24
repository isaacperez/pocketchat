import torch

from pocketchat.model import (
    Glimpse,
    HiddenStateDeltaPredictor,
    HypothesisDecomposer,
    HypothesisObservationComposer,
    HypothesisUpdater,
    Matcher,
    NextCharPredictor,
)


def test_integration_decomposer_glimpse_keeps_part_layout() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, window_size, embedding_dim = 9, 5, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim)
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=6,
    )
    center_logits, zoom_logits, _, _ = decomposer(hypotheses)

    glimpse = Glimpse(window_size=window_size, squeeze_output=False)
    windows_flat = glimpse(
        input_embeddings,
        center_logits=center_logits.reshape(batch_size, -1),
        zoom_logits=zoom_logits.reshape(batch_size, -1),
    )
    part_windows = windows_flat.reshape(batch_size, num_hypotheses, num_parts, window_size, embedding_dim)

    assert part_windows.shape == (batch_size, num_hypotheses, num_parts, window_size, embedding_dim)


def test_integration_decomposer_matcher_produces_hypothesis_part_heatmaps() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, embedding_dim = 7, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim)
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=6,
    )
    _, _, references, adapters = decomposer(hypotheses)

    matcher = Matcher()
    heatmaps = matcher(input_embeddings, references, adapters)

    assert heatmaps.shape == (batch_size, seq_len, num_hypotheses, num_parts)


def test_integration_glimpse_matcher_with_parallel_glimpses() -> None:
    batch_size, seq_len, embedding_dim = 2, 8, 4
    num_glimpses, window_size = 5, 4

    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    center_logits = torch.randn(batch_size, num_glimpses)
    zoom_logits = torch.randn(batch_size, num_glimpses)

    glimpse = Glimpse(window_size=window_size, squeeze_output=False)
    windows = glimpse(input_embeddings, center_logits=center_logits, zoom_logits=zoom_logits)

    references = torch.randn(batch_size, num_glimpses, embedding_dim)
    adapters = torch.randn(batch_size, num_glimpses, embedding_dim)

    matcher = Matcher()
    heatmaps = matcher(windows, references, adapters)

    assert heatmaps.shape == (batch_size, num_glimpses, window_size)


def test_integration_matcher_composer_builds_observation_embeddings() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    window_size, embedding_dim = 5, 4

    part_windows = torch.randn(batch_size, num_hypotheses, num_parts, window_size, embedding_dim)
    references = torch.randn(batch_size, num_hypotheses, num_parts, embedding_dim)
    adapters = torch.randn(batch_size, num_hypotheses, num_parts, embedding_dim)

    matcher = Matcher()
    part_heatmaps = matcher(part_windows, references, adapters)

    composer = HypothesisObservationComposer(
        num_parts=num_parts,
        window_size=window_size,
        observation_dim=embedding_dim,
        hidden_dim=10,
    )
    observations = composer(part_heatmaps)

    assert part_heatmaps.shape == (batch_size, num_hypotheses, num_parts, window_size)
    assert observations.shape == (batch_size, num_hypotheses, embedding_dim)


def test_integration_composer_updater_fuses_hypotheses_and_observations() -> None:
    batch_size, num_hypotheses, num_parts, window_size = 2, 4, 3, 5
    hypothesis_dim, observation_dim, updated_dim = 6, 7, 8

    part_heatmaps = torch.randn(batch_size, num_hypotheses, num_parts, window_size)
    hypotheses = torch.randn(batch_size, num_hypotheses, hypothesis_dim)

    composer = HypothesisObservationComposer(
        num_parts=num_parts,
        window_size=window_size,
        observation_dim=observation_dim,
        hidden_dim=9,
    )
    observations = composer(part_heatmaps)

    updater = HypothesisUpdater(
        hypothesis_dim=hypothesis_dim,
        observation_dim=observation_dim,
        updated_dim=updated_dim,
        hidden_dim=11,
    )
    updated_hypotheses = updater(hypotheses, observations)

    assert observations.shape == (batch_size, num_hypotheses, observation_dim)
    assert updated_hypotheses.shape == (batch_size, num_hypotheses, updated_dim)


def test_integration_updater_delta_predictor_and_next_char_predictor() -> None:
    batch_size, num_hypotheses = 2, 3
    hypothesis_dim, observation_dim, updated_dim = 5, 4, 6
    hidden_state_dim, num_iterations = 7, 3
    vocab_size, num_next_chars = 17, 4

    hypotheses = torch.randn(batch_size, num_hypotheses, hypothesis_dim, requires_grad=True)
    observations = torch.randn(batch_size, num_hypotheses, observation_dim, requires_grad=True)

    updater = HypothesisUpdater(
        hypothesis_dim=hypothesis_dim,
        observation_dim=observation_dim,
        updated_dim=updated_dim,
        hidden_dim=10,
    )
    updated_hypotheses = updater(hypotheses, observations)

    delta_predictor = HiddenStateDeltaPredictor(
        num_hypotheses=num_hypotheses,
        updated_hypothesis_dim=updated_dim,
        hidden_state_dim=hidden_state_dim,
        hidden_dim=12,
    )
    hidden_state_delta = delta_predictor(updated_hypotheses)

    history = torch.stack([updated_hypotheses for _ in range(num_iterations)], dim=2)
    predictor = NextCharPredictor(
        num_hypotheses=num_hypotheses,
        num_iterations=num_iterations,
        updated_hypothesis_dim=updated_dim,
        vocab_size=vocab_size,
        num_next_chars=num_next_chars,
    )
    next_char_logits = predictor(history)

    assert hidden_state_delta.shape == (batch_size, hidden_state_dim)
    assert next_char_logits.shape == (batch_size, num_next_chars, vocab_size)

    loss = hidden_state_delta.pow(2).mean() + next_char_logits.pow(2).mean()
    loss.backward()
    assert hypotheses.grad is not None
    assert observations.grad is not None


def test_integration_full_pipeline_decomposer_to_next_char_predictor() -> None:
    batch_size, num_hypotheses, num_parts = 2, 3, 2
    seq_len, window_size, embedding_dim = 9, 5, 4
    num_iterations, num_next_chars = 3, 4
    hidden_state_dim = 6
    vocab_size = 21

    hypotheses = torch.randn(batch_size, num_hypotheses, embedding_dim, requires_grad=True)
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
    embedding_weight = torch.randn(vocab_size, embedding_dim, requires_grad=True)

    decomposer = HypothesisDecomposer(
        hypothesis_dim=embedding_dim,
        num_parts=num_parts,
        matcher_dim=embedding_dim,
        hidden_dim=8,
    )
    glimpse = Glimpse(window_size=window_size, squeeze_output=False)
    matcher = Matcher()
    composer = HypothesisObservationComposer(
        num_parts=num_parts,
        window_size=window_size,
        observation_dim=embedding_dim,
        hidden_dim=10,
    )
    updater = HypothesisUpdater(
        hypothesis_dim=embedding_dim,
        observation_dim=embedding_dim,
        updated_dim=embedding_dim,
        hidden_dim=10,
    )
    delta_predictor = HiddenStateDeltaPredictor(
        num_hypotheses=num_hypotheses,
        updated_hypothesis_dim=embedding_dim,
        hidden_state_dim=hidden_state_dim,
        hidden_dim=12,
    )
    predictor = NextCharPredictor(
        num_hypotheses=num_hypotheses,
        num_iterations=num_iterations,
        updated_hypothesis_dim=embedding_dim,
        vocab_size=vocab_size,
        num_next_chars=num_next_chars,
        tie_with_embedding=True,
        tied_embedding_dim=embedding_dim,
    )

    current_hypotheses = hypotheses
    updated_hypotheses_history: list[torch.Tensor] = []

    for _ in range(num_iterations):
        center_logits, zoom_logits, references, adapters = decomposer(current_hypotheses)

        windows_flat = glimpse(
            input_embeddings,
            center_logits=center_logits.reshape(batch_size, -1),
            zoom_logits=zoom_logits.reshape(batch_size, -1),
        )
        part_windows = windows_flat.reshape(batch_size, num_hypotheses, num_parts, window_size, embedding_dim)

        part_heatmaps = matcher(part_windows, references, adapters)
        observations = composer(part_heatmaps)
        updated_hypotheses = updater(current_hypotheses, observations)
        hidden_state_delta = delta_predictor(updated_hypotheses)

        updated_hypotheses_history.append(updated_hypotheses)
        current_hypotheses = updated_hypotheses + hidden_state_delta.unsqueeze(1)[..., :embedding_dim]

    history = torch.stack(updated_hypotheses_history, dim=2)
    next_char_logits = predictor(history, embedding_weight=embedding_weight)

    assert history.shape == (batch_size, num_hypotheses, num_iterations, embedding_dim)
    assert next_char_logits.shape == (batch_size, num_next_chars, vocab_size)

    loss = next_char_logits.pow(2).mean()
    loss.backward()

    assert hypotheses.grad is not None
    assert input_embeddings.grad is not None
    assert embedding_weight.grad is not None
    assert decomposer.center_head[0].weight.grad is not None
    assert predictor.linear_out.weight.grad is not None
