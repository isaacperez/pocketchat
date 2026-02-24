import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pocketchat.constants import DEFAULT_MATCHER_EPS


class Cortex(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        hidden_state_dim: int | None = None,
        num_hypotheses: int = 4,
        num_parts: int = 2,
        window_size: int = 8,
        num_iterations: int = 2,
        num_next_chars: int = 4,
        hypothesis_dim: int | None = None,
        observation_dim: int | None = None,
        updated_hypothesis_dim: int | None = None,
        matcher_eps: float = DEFAULT_MATCHER_EPS,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")
        if num_hypotheses <= 0:
            raise ValueError(f"num_hypotheses must be > 0, got {num_hypotheses}.")
        if num_parts <= 0:
            raise ValueError(f"num_parts must be > 0, got {num_parts}.")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}.")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be > 0, got {num_iterations}.")
        if num_next_chars <= 0:
            raise ValueError(f"num_next_chars must be > 0, got {num_next_chars}.")
        if matcher_eps <= 0:
            raise ValueError(f"matcher_eps must be > 0, got {matcher_eps}.")
        if padding_idx is not None and not (0 <= padding_idx < vocab_size):
            raise ValueError(
                f"padding_idx must be in [0, {vocab_size - 1}] when provided, got {padding_idx}."
            )

        resolved_hidden_state_dim = embedding_dim if hidden_state_dim is None else hidden_state_dim
        if resolved_hidden_state_dim <= 0:
            raise ValueError(f"hidden_state_dim must be > 0, got {resolved_hidden_state_dim}.")

        resolved_hypothesis_dim = resolved_hidden_state_dim if hypothesis_dim is None else hypothesis_dim
        if resolved_hypothesis_dim <= 0:
            raise ValueError(f"hypothesis_dim must be > 0, got {resolved_hypothesis_dim}.")

        resolved_observation_dim = resolved_hypothesis_dim if observation_dim is None else observation_dim
        if resolved_observation_dim <= 0:
            raise ValueError(f"observation_dim must be > 0, got {resolved_observation_dim}.")

        resolved_updated_hypothesis_dim = (
            resolved_hypothesis_dim if updated_hypothesis_dim is None else updated_hypothesis_dim
        )
        if resolved_updated_hypothesis_dim <= 0:
            raise ValueError(f"updated_hypothesis_dim must be > 0, got {resolved_updated_hypothesis_dim}.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.hidden_state_dim = resolved_hidden_state_dim
        self.num_hypotheses = num_hypotheses
        self.num_parts = num_parts
        self.window_size = window_size
        self.num_iterations = num_iterations
        self.num_next_chars = num_next_chars
        self.hypothesis_dim = resolved_hypothesis_dim
        self.observation_dim = resolved_observation_dim
        self.updated_hypothesis_dim = resolved_updated_hypothesis_dim
        self.char_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.base_hidden_state = nn.Parameter(torch.zeros(resolved_hidden_state_dim))
        self.hypothesis_generator = HypothesisGenerator(
            hidden_state_dim=resolved_hidden_state_dim,
            num_hypotheses=num_hypotheses,
            hypothesis_dim=resolved_hypothesis_dim,
        )
        self.hypothesis_decomposer = HypothesisDecomposer(
            hypothesis_dim=resolved_hypothesis_dim,
            num_parts=num_parts,
            matcher_dim=embedding_dim,
        )
        self.glimpse = Glimpse(window_size=window_size, squeeze_output=False, eps=matcher_eps)
        self.matcher = Matcher(eps=matcher_eps)
        self.hypothesis_observation_composer = HypothesisObservationComposer(
            num_parts=num_parts,
            window_size=window_size,
            observation_dim=resolved_observation_dim,
        )
        self.hypothesis_updater = HypothesisUpdater(
            hypothesis_dim=resolved_hypothesis_dim,
            observation_dim=resolved_observation_dim,
            updated_dim=resolved_updated_hypothesis_dim,
        )
        self.hidden_state_delta_predictor = HiddenStateDeltaPredictor(
            num_hypotheses=num_hypotheses,
            updated_hypothesis_dim=resolved_updated_hypothesis_dim,
            hidden_state_dim=resolved_hidden_state_dim,
        )
        self.next_char_predictor = NextCharPredictor(
            num_hypotheses=num_hypotheses,
            num_iterations=num_iterations,
            updated_hypothesis_dim=resolved_updated_hypothesis_dim,
            vocab_size=vocab_size,
            num_next_chars=num_next_chars,
            tie_with_embedding=True,
            tied_embedding_dim=embedding_dim,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if token_ids.dtype.is_floating_point:
            raise TypeError(f"token_ids must be integer tensor, got dtype={token_ids.dtype}.")
        was_unbatched = token_ids.ndim == 1
        if was_unbatched:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape [L] or [B, L], got shape={tuple(token_ids.shape)}.")

        token_ids = token_ids.long()
        token_embeddings = self.char_embeddings(token_ids)
        batch_size = token_ids.shape[0]
        hidden_state = self.base_hidden_state.unsqueeze(0).expand(batch_size, -1)
        base_hidden_state = hidden_state

        updated_hypotheses_history: list[torch.Tensor] = []
        hidden_state_deltas: list[torch.Tensor] = []

        for _ in range(self.num_iterations):
            hypotheses = self.hypothesis_generator(hidden_state)  # [B, H, D_h]
            center_logits, zoom_logits, references, adapters = self.hypothesis_decomposer(hypotheses)  # [B,H,K], [B,H,K], [B,H,K,D], [B,H,K,D]

            part_windows_flat = self.glimpse(
                token_embeddings,
                center_logits=center_logits.reshape(token_embeddings.shape[0], -1),
                zoom_logits=zoom_logits.reshape(token_embeddings.shape[0], -1),
                lengths=lengths,
            )  # [B, H*K, W, D]
            part_windows = part_windows_flat.reshape(
                token_embeddings.shape[0],
                self.num_hypotheses,
                self.num_parts,
                self.window_size,
                self.embedding_dim,
            )  # [B, H, K, W, D]

            part_heatmaps = self.matcher(part_windows, references, adapters)  # [B, H, K, W]
            observations = self.hypothesis_observation_composer(part_heatmaps)  # [B, H, D_o]
            updated_hypotheses = self.hypothesis_updater(hypotheses, observations)  # [B, H, D_u]
            hidden_state_delta = self.hidden_state_delta_predictor(updated_hypotheses)  # [B, D_hidden]

            updated_hypotheses_history.append(updated_hypotheses)
            hidden_state_deltas.append(hidden_state_delta)
            hidden_state = hidden_state + hidden_state_delta

        # [B, H, N, D_u] where N is the number of iterations and D_u is last.
        hypothesis_update_history = torch.stack(updated_hypotheses_history, dim=2)
        # [B, D_hidden, N]
        hidden_state_delta_history = torch.stack(hidden_state_deltas, dim=-1)
        next_char_logits = self.next_char_predictor(
            hypothesis_update_history,
            embedding_weight=self.char_embeddings.weight,
        )  # [B, N_pred, V]
        next_char_predictions = next_char_logits.argmax(dim=-1)  # [B, N_pred]

        outputs: dict[str, torch.Tensor] = {
            "token_embeddings": token_embeddings,
            "base_hidden_state": base_hidden_state,
            "final_hidden_state": hidden_state,
            "hidden_state_delta_history": hidden_state_delta_history,
            "hypothesis_update_history": hypothesis_update_history,
            "last_updated_hypotheses": updated_hypotheses_history[-1],
            "next_char_logits": next_char_logits,
            "next_char_predictions": next_char_predictions,
        }
        if was_unbatched:
            outputs = {name: tensor.squeeze(0) for name, tensor in outputs.items()}
        return outputs


class HypothesisGenerator(nn.Module):
    """
    Generate H hypothesis embeddings from a hidden state tensor.

    Architecture:
    - hidden_state -> RMSNorm -> linear_in -> GeLU -> linear_out

    Input:
    - hidden_state: [..., hidden_state_dim]

    Output:
    - hypotheses: [..., num_hypotheses, hypothesis_dim]
    """

    def __init__(
        self,
        hidden_state_dim: int,
        num_hypotheses: int,
        hypothesis_dim: int | None = None,
        eps: float = DEFAULT_MATCHER_EPS,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_state_dim <= 0:
            raise ValueError(f"hidden_state_dim must be > 0, got {hidden_state_dim}.")
        if num_hypotheses <= 0:
            raise ValueError(f"num_hypotheses must be > 0, got {num_hypotheses}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        output_hypothesis_dim = hidden_state_dim if hypothesis_dim is None else hypothesis_dim
        if output_hypothesis_dim <= 0:
            raise ValueError(f"hypothesis_dim must be > 0, got {output_hypothesis_dim}.")

        self.hidden_state_dim = hidden_state_dim
        self.num_hypotheses = num_hypotheses
        self.hypothesis_dim = output_hypothesis_dim
        flat_output_dim = num_hypotheses * output_hypothesis_dim
        self.rms_norm = nn.RMSNorm(hidden_state_dim, eps=eps)
        self.linear_in = nn.Linear(hidden_state_dim, hidden_state_dim, bias=bias)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(hidden_state_dim, flat_output_dim, bias=bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim < 1:
            raise ValueError(
                f"hidden_state must have shape [..., hidden_state_dim], got ndim={hidden_state.ndim}."
            )
        if hidden_state.shape[-1] != self.hidden_state_dim:
            raise ValueError(
                "Last dim mismatch: "
                f"expected hidden_state_dim={self.hidden_state_dim}, got {hidden_state.shape[-1]}."
            )
        if not hidden_state.dtype.is_floating_point:
            raise TypeError(f"hidden_state must be floating-point tensor, got dtype={hidden_state.dtype}.")

        normalized_state = self.rms_norm(hidden_state)
        flat_hypotheses = self.linear_out(self.activation(self.linear_in(normalized_state)))
        return flat_hypotheses.reshape(*hidden_state.shape[:-1], self.num_hypotheses, self.hypothesis_dim)


class HypothesisDecomposer(nn.Module):
    """
    Decompose each hypothesis embedding into control and matching signals.

    For every hypothesis vector `[..., D_h]`, this module generates `K` parts.
    Each part contains:
    - one `center_logit` for Glimpse
    - one `zoom_logit` for Glimpse
    - one `reference` embedding for Matcher
    - one `adapter` embedding for Matcher

    Output shapes:
    - `center_logits`: [..., K]
    - `zoom_logits`: [..., K]
    - `references`: [..., K, D_m]
    - `adapters`: [..., K, D_m]

    Architecture:
    - Shared pre-normalization: RMSNorm on hypothesis embeddings.
    - Four independent heads (one per output), each with:
      Linear -> GeLU -> Linear
    """

    def __init__(
        self,
        hypothesis_dim: int,
        num_parts: int,
        matcher_dim: int | None = None,
        hidden_dim: int | None = None,
        eps: float = DEFAULT_MATCHER_EPS,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hypothesis_dim <= 0:
            raise ValueError(f"hypothesis_dim must be > 0, got {hypothesis_dim}.")
        if num_parts <= 0:
            raise ValueError(f"num_parts must be > 0, got {num_parts}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        output_matcher_dim = hypothesis_dim if matcher_dim is None else matcher_dim
        if output_matcher_dim <= 0:
            raise ValueError(f"matcher_dim must be > 0, got {output_matcher_dim}.")

        head_hidden_dim = hypothesis_dim if hidden_dim is None else hidden_dim
        if head_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {head_hidden_dim}.")

        self.hypothesis_dim = hypothesis_dim
        self.num_parts = num_parts
        self.matcher_dim = output_matcher_dim
        self.hidden_dim = head_hidden_dim
        self.rms_norm = nn.RMSNorm(hypothesis_dim, eps=eps)

        self.center_head = self._build_head(hypothesis_dim, head_hidden_dim, num_parts, bias=bias)
        self.zoom_head = self._build_head(hypothesis_dim, head_hidden_dim, num_parts, bias=bias)
        self.reference_head = self._build_head(
            hypothesis_dim,
            head_hidden_dim,
            num_parts * output_matcher_dim,
            bias=bias,
        )
        self.adapter_head = self._build_head(
            hypothesis_dim,
            head_hidden_dim,
            num_parts * output_matcher_dim,
            bias=bias,
        )

    @staticmethod
    def _build_head(input_dim: int, hidden_dim: int, output_dim: int, bias: bool) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(
        self,
        hypothesis_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hypothesis_embeddings.ndim < 1:
            raise ValueError(
                "hypothesis_embeddings must have shape [..., hypothesis_dim], "
                f"got ndim={hypothesis_embeddings.ndim}."
            )
        if hypothesis_embeddings.shape[-1] != self.hypothesis_dim:
            raise ValueError(
                "Last dim mismatch: "
                f"expected hypothesis_dim={self.hypothesis_dim}, got {hypothesis_embeddings.shape[-1]}."
            )
        if not hypothesis_embeddings.dtype.is_floating_point:
            raise TypeError(
                "hypothesis_embeddings must be floating-point tensor, "
                f"got dtype={hypothesis_embeddings.dtype}."
            )

        normalized_hypotheses = self.rms_norm(hypothesis_embeddings)
        center_logits = self.center_head(normalized_hypotheses)
        zoom_logits = self.zoom_head(normalized_hypotheses)
        references = self.reference_head(normalized_hypotheses).reshape(
            *hypothesis_embeddings.shape[:-1],
            self.num_parts,
            self.matcher_dim,
        )
        adapters = self.adapter_head(normalized_hypotheses).reshape(
            *hypothesis_embeddings.shape[:-1],
            self.num_parts,
            self.matcher_dim,
        )
        return center_logits, zoom_logits, references, adapters


class HypothesisObservationComposer(nn.Module):
    """
    Compose part-level heatmaps into one observation embedding per hypothesis.

    Input:
    - part_heatmaps: [..., K, W]
      K = number of parts, W = glimpse window size

    Output:
    - observation_embeddings: [..., D_obs]

    Architecture:
    - flatten [..., K, W] -> [..., K*W]
    - Linear -> GeLU -> Linear
    """

    def __init__(
        self,
        num_parts: int,
        window_size: int,
        observation_dim: int,
        hidden_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_parts <= 0:
            raise ValueError(f"num_parts must be > 0, got {num_parts}.")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}.")
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be > 0, got {observation_dim}.")

        self.num_parts = num_parts
        self.window_size = window_size
        self.observation_dim = observation_dim
        self.input_dim = num_parts * window_size
        self.hidden_dim = self.input_dim if hidden_dim is None else hidden_dim
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}.")

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(self.hidden_dim, observation_dim, bias=bias)

    def forward(
        self,
        part_heatmaps: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if part_heatmaps.ndim < 2:
            raise ValueError(
                "part_heatmaps must have shape [..., K, W], "
                f"got shape={tuple(part_heatmaps.shape)}."
            )
        if part_heatmaps.shape[-2] != self.num_parts:
            raise ValueError(
                f"Part dimension mismatch: expected K={self.num_parts}, got K={part_heatmaps.shape[-2]}."
            )
        if part_heatmaps.shape[-1] != self.window_size:
            raise ValueError(
                f"Window dimension mismatch: expected W={self.window_size}, got W={part_heatmaps.shape[-1]}."
            )
        if not part_heatmaps.dtype.is_floating_point:
            raise TypeError(f"part_heatmaps must be floating-point tensor, got dtype={part_heatmaps.dtype}.")

        flat_heatmaps = part_heatmaps.reshape(*part_heatmaps.shape[:-2], self.input_dim)
        hidden = self.activation(self.linear_in(flat_heatmaps))
        observation_embeddings = self.linear_out(hidden)

        if not return_aux:
            return observation_embeddings
        aux = {
            "flat_heatmaps": flat_heatmaps,
            "hidden_features": hidden,
        }
        return observation_embeddings, aux


class HypothesisUpdater(nn.Module):
    """
    Fuse each hypothesis embedding with its observation embedding.

    Inputs:
    - hypotheses: [..., D_h]
    - observations: [..., D_o]

    Output:
    - updated_hypotheses: [..., D_u]

    Architecture:
    - concat([hypothesis, observation]) -> Linear -> GeLU -> Linear
    """

    def __init__(
        self,
        hypothesis_dim: int,
        observation_dim: int | None = None,
        updated_dim: int | None = None,
        hidden_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hypothesis_dim <= 0:
            raise ValueError(f"hypothesis_dim must be > 0, got {hypothesis_dim}.")

        resolved_observation_dim = hypothesis_dim if observation_dim is None else observation_dim
        if resolved_observation_dim <= 0:
            raise ValueError(f"observation_dim must be > 0, got {resolved_observation_dim}.")

        resolved_updated_dim = hypothesis_dim if updated_dim is None else updated_dim
        if resolved_updated_dim <= 0:
            raise ValueError(f"updated_dim must be > 0, got {resolved_updated_dim}.")

        input_dim = hypothesis_dim + resolved_observation_dim
        resolved_hidden_dim = input_dim if hidden_dim is None else hidden_dim
        if resolved_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {resolved_hidden_dim}.")

        self.hypothesis_dim = hypothesis_dim
        self.observation_dim = resolved_observation_dim
        self.updated_dim = resolved_updated_dim
        self.hidden_dim = resolved_hidden_dim

        self.linear_in = nn.Linear(input_dim, resolved_hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(resolved_hidden_dim, resolved_updated_dim, bias=bias)

    def forward(self, hypotheses: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        if hypotheses.ndim < 1:
            raise ValueError(f"hypotheses must have shape [..., D_h], got ndim={hypotheses.ndim}.")
        if observations.ndim < 1:
            raise ValueError(f"observations must have shape [..., D_o], got ndim={observations.ndim}.")
        if hypotheses.shape[:-1] != observations.shape[:-1]:
            raise ValueError(
                "Prefix shape mismatch: hypotheses and observations must share leading dimensions. "
                f"Got {hypotheses.shape} and {observations.shape}."
            )
        if hypotheses.shape[-1] != self.hypothesis_dim:
            raise ValueError(
                f"Hypothesis dim mismatch: expected D_h={self.hypothesis_dim}, got {hypotheses.shape[-1]}."
            )
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Observation dim mismatch: expected D_o={self.observation_dim}, got {observations.shape[-1]}."
            )
        if not hypotheses.dtype.is_floating_point:
            raise TypeError(f"hypotheses must be floating-point tensor, got dtype={hypotheses.dtype}.")
        if not observations.dtype.is_floating_point:
            raise TypeError(f"observations must be floating-point tensor, got dtype={observations.dtype}.")

        fused_inputs = torch.cat([hypotheses, observations], dim=-1)
        hidden = self.activation(self.linear_in(fused_inputs))
        return self.linear_out(hidden)


class HiddenStateDeltaPredictor(nn.Module):
    """
    Predict a hidden-state delta from all updated hypotheses.

    Input:
    - updated_hypotheses: [..., H, D_u]

    Output:
    - hidden_state_delta: [..., D_h]

    Architecture:
    - flatten [..., H, D_u] -> [..., H*D_u]
    - Linear -> GeLU -> Linear
    """

    def __init__(
        self,
        num_hypotheses: int,
        updated_hypothesis_dim: int,
        hidden_state_dim: int | None = None,
        hidden_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_hypotheses <= 0:
            raise ValueError(f"num_hypotheses must be > 0, got {num_hypotheses}.")
        if updated_hypothesis_dim <= 0:
            raise ValueError(f"updated_hypothesis_dim must be > 0, got {updated_hypothesis_dim}.")

        resolved_hidden_state_dim = updated_hypothesis_dim if hidden_state_dim is None else hidden_state_dim
        if resolved_hidden_state_dim <= 0:
            raise ValueError(f"hidden_state_dim must be > 0, got {resolved_hidden_state_dim}.")

        input_dim = num_hypotheses * updated_hypothesis_dim
        resolved_hidden_dim = input_dim if hidden_dim is None else hidden_dim
        if resolved_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {resolved_hidden_dim}.")

        self.num_hypotheses = num_hypotheses
        self.updated_hypothesis_dim = updated_hypothesis_dim
        self.hidden_state_dim = resolved_hidden_state_dim
        self.hidden_dim = resolved_hidden_dim
        self.input_dim = input_dim

        self.linear_in = nn.Linear(input_dim, resolved_hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(resolved_hidden_dim, resolved_hidden_state_dim, bias=bias)

    def forward(self, updated_hypotheses: torch.Tensor) -> torch.Tensor:
        if updated_hypotheses.ndim < 2:
            raise ValueError(
                "updated_hypotheses must have shape [..., H, D_u], "
                f"got shape={tuple(updated_hypotheses.shape)}."
            )
        if updated_hypotheses.shape[-2] != self.num_hypotheses:
            raise ValueError(
                "Num hypotheses mismatch: "
                f"expected H={self.num_hypotheses}, got H={updated_hypotheses.shape[-2]}."
            )
        if updated_hypotheses.shape[-1] != self.updated_hypothesis_dim:
            raise ValueError(
                "Updated hypothesis dim mismatch: "
                f"expected D_u={self.updated_hypothesis_dim}, got D_u={updated_hypotheses.shape[-1]}."
            )
        if not updated_hypotheses.dtype.is_floating_point:
            raise TypeError(
                "updated_hypotheses must be floating-point tensor, "
                f"got dtype={updated_hypotheses.dtype}."
            )

        flat_hypotheses = updated_hypotheses.reshape(*updated_hypotheses.shape[:-2], self.input_dim)
        hidden = self.activation(self.linear_in(flat_hypotheses))
        return self.linear_out(hidden)


class NextCharPredictor(nn.Module):
    """
    Predict logits for the next N characters from hypothesis update history.

    Input:
    - hypothesis_update_history: [..., H, N_iter, D_u]

    Output:
    - next_char_logits: [..., N_pred, V]
      N_pred = number of next characters to predict
      V = vocabulary size

    Architecture:
    - flatten [..., H, N_iter, D_u] -> [..., H*N_iter*D_u]
    - Linear -> GeLU -> Linear
    - If `tie_with_embedding=False`: reshape -> [..., N_pred, V]
    - If `tie_with_embedding=True`: reshape -> [..., N_pred, D_e], then logits via
      dot-product with shared embedding weights [V, D_e]
    """

    def __init__(
        self,
        num_hypotheses: int,
        num_iterations: int,
        updated_hypothesis_dim: int,
        vocab_size: int,
        num_next_chars: int,
        hidden_dim: int | None = None,
        tie_with_embedding: bool = False,
        tied_embedding_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_hypotheses <= 0:
            raise ValueError(f"num_hypotheses must be > 0, got {num_hypotheses}.")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be > 0, got {num_iterations}.")
        if updated_hypothesis_dim <= 0:
            raise ValueError(f"updated_hypothesis_dim must be > 0, got {updated_hypothesis_dim}.")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        if num_next_chars <= 0:
            raise ValueError(f"num_next_chars must be > 0, got {num_next_chars}.")
        if tie_with_embedding and (tied_embedding_dim is None or tied_embedding_dim <= 0):
            raise ValueError(
                "tied_embedding_dim must be > 0 when tie_with_embedding=True, "
                f"got {tied_embedding_dim}."
            )

        self.num_hypotheses = num_hypotheses
        self.num_iterations = num_iterations
        self.updated_hypothesis_dim = updated_hypothesis_dim
        self.vocab_size = vocab_size
        self.num_next_chars = num_next_chars
        self.tie_with_embedding = tie_with_embedding
        self.tied_embedding_dim = tied_embedding_dim
        self.input_dim = num_hypotheses * num_iterations * updated_hypothesis_dim
        self.hidden_dim = self.input_dim if hidden_dim is None else hidden_dim
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}.")

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.activation = nn.GELU()
        output_dim = (
            num_next_chars * tied_embedding_dim
            if tie_with_embedding
            else num_next_chars * vocab_size
        )
        self.linear_out = nn.Linear(self.hidden_dim, output_dim, bias=bias)

    def forward(
        self,
        hypothesis_update_history: torch.Tensor,
        embedding_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hypothesis_update_history.ndim < 3:
            raise ValueError(
                "hypothesis_update_history must have shape [..., H, N_iter, D_u], "
                f"got shape={tuple(hypothesis_update_history.shape)}."
            )
        if hypothesis_update_history.shape[-3] != self.num_hypotheses:
            raise ValueError(
                "Num hypotheses mismatch: "
                f"expected H={self.num_hypotheses}, got H={hypothesis_update_history.shape[-3]}."
            )
        if hypothesis_update_history.shape[-2] != self.num_iterations:
            raise ValueError(
                "Num iterations mismatch: "
                f"expected N_iter={self.num_iterations}, got N_iter={hypothesis_update_history.shape[-2]}."
            )
        if hypothesis_update_history.shape[-1] != self.updated_hypothesis_dim:
            raise ValueError(
                "Updated hypothesis dim mismatch: "
                f"expected D_u={self.updated_hypothesis_dim}, got D_u={hypothesis_update_history.shape[-1]}."
            )
        if not hypothesis_update_history.dtype.is_floating_point:
            raise TypeError(
                "hypothesis_update_history must be floating-point tensor, "
                f"got dtype={hypothesis_update_history.dtype}."
            )

        flat_history = hypothesis_update_history.reshape(*hypothesis_update_history.shape[:-3], self.input_dim)
        hidden = self.activation(self.linear_in(flat_history))
        flat_outputs = self.linear_out(hidden)

        if not self.tie_with_embedding:
            return flat_outputs.reshape(*hypothesis_update_history.shape[:-3], self.num_next_chars, self.vocab_size)

        if embedding_weight is None:
            raise ValueError("embedding_weight must be provided when tie_with_embedding=True.")
        if embedding_weight.ndim != 2:
            raise ValueError(
                f"embedding_weight must have shape [V, D_e], got shape={tuple(embedding_weight.shape)}."
            )
        if embedding_weight.shape[0] != self.vocab_size:
            raise ValueError(
                f"Embedding vocab mismatch: expected V={self.vocab_size}, got V={embedding_weight.shape[0]}."
            )
        if embedding_weight.shape[1] != self.tied_embedding_dim:
            raise ValueError(
                "Embedding dim mismatch for tied projection: "
                f"expected D_e={self.tied_embedding_dim}, got D_e={embedding_weight.shape[1]}."
            )
        if not embedding_weight.dtype.is_floating_point:
            raise TypeError(f"embedding_weight must be floating-point tensor, got dtype={embedding_weight.dtype}.")

        token_features = flat_outputs.reshape(
            *hypothesis_update_history.shape[:-3],
            self.num_next_chars,
            self.tied_embedding_dim,
        )
        return torch.einsum("...nd,vd->...nv", token_features, embedding_weight)


class Matcher(nn.Module):
    """
    Apply adapter modulation to input embeddings and compare with references.

    Input:
    - input_embeddings: [..., D]
    - references: [..., D]
    - adapters: [..., D] (same shape as references)

    The module finds the longest shared prefix between input/reference shapes,
    then treats the remaining input dimensions as "positions" and the remaining
    reference dimensions as "match axes".

    Output:
    - heatmap similarities in [-1, 1] with shape:
      [*common_prefix, *input_specific_dims, *reference_specific_dims]
      preserving the input shape structure.
    """

    def __init__(self, eps: float = DEFAULT_MATCHER_EPS) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.eps = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        references: torch.Tensor,
        adapters: torch.Tensor,
    ) -> torch.Tensor:
        if input_embeddings.ndim < 1:
            raise ValueError(
                f"input_embeddings must have shape [..., D], got ndim={input_embeddings.ndim}."
            )
        if references.ndim < 1:
            raise ValueError(f"references must have shape [..., D], got ndim={references.ndim}.")
        if adapters.ndim < 1:
            raise ValueError(f"adapters must have shape [..., D], got ndim={adapters.ndim}.")
        if references.shape != adapters.shape:
            raise ValueError(
                f"adapters and references shapes must match, got {adapters.shape} and {references.shape}."
            )
        if not input_embeddings.dtype.is_floating_point:
            raise TypeError(f"input_embeddings must be floating-point tensor, got dtype={input_embeddings.dtype}.")
        if not references.dtype.is_floating_point:
            raise TypeError(f"references must be floating-point tensor, got dtype={references.dtype}.")
        if not adapters.dtype.is_floating_point:
            raise TypeError(f"adapters must be floating-point tensor, got dtype={adapters.dtype}.")

        emb_dim = input_embeddings.shape[-1]
        if references.shape[-1] != emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input D={emb_dim}, refs/adapters D={references.shape[-1]}."
            )

        input_prefix = tuple(input_embeddings.shape[:-1])
        ref_prefix = tuple(references.shape[:-1])

        # Longest common prefix allows shared context dims (e.g. batch, hierarchy).
        common_prefix_len = 0
        for input_dim, ref_dim in zip(input_prefix, ref_prefix):
            if input_dim == ref_dim:
                common_prefix_len += 1
            else:
                break

        common_prefix = input_prefix[:common_prefix_len]
        input_specific_dims = input_prefix[common_prefix_len:]
        ref_specific_dims = ref_prefix[common_prefix_len:]

        common_size = math.prod(common_prefix) if common_prefix else 1
        input_size = math.prod(input_specific_dims) if input_specific_dims else 1
        ref_size = math.prod(ref_specific_dims) if ref_specific_dims else 1

        input_view = input_embeddings.reshape(common_size, input_size, emb_dim)
        references_view = references.reshape(common_size, ref_size, emb_dim)
        adapters_view = adapters.reshape(common_size, ref_size, emb_dim)

        adapted = input_view.unsqueeze(2) * adapters_view.unsqueeze(1)  # [C, S, M, D]
        adapted_norm = F.normalize(adapted, dim=-1, eps=self.eps)
        references_norm = F.normalize(references_view, dim=-1, eps=self.eps).unsqueeze(1)  # [C, 1, M, D]
        similarities = (adapted_norm * references_norm).sum(dim=-1)  # [C, S, M]

        return similarities.reshape(*common_prefix, *input_specific_dims, *ref_specific_dims)


class Glimpse(nn.Module):
    """
    Differentiable 1D glimpse over character embeddings with linear interpolation.

    The module samples a fixed-size window `W` from sequences of length `L` using
    real-valued positions and border-clamped interpolation.

    Inputs:
    - `input_embeddings`: [L, D] or [B, L, D]
    - `center_logits`, `zoom_logits`: scalar, [G], or [B, G]-like (including [1, G], [B, 1])
      where `G` is the number of glimpses. Controls are external: this module has no
      internal center/zoom parameters.
    - `lengths` (optional): scalar or [B], valid length per sequence for clamped sampling.

    Glimpse count:
    - `G` is inferred per forward pass from control shapes.
    - If one control has shape [*, 1] and the other [*, G], the singleton control is broadcast.

    Outputs:
    - If `squeeze_output=True`:
      - unbatched + G=1 -> [W, D]
      - unbatched + G>1 -> [G, W, D]
      - batched + G=1 -> [B, W, D]
      - batched + G>1 -> [B, G, W, D]
    - If `squeeze_output=False`: always [B, G, W, D]
    """

    def __init__(
        self,
        window_size: int,
        eps: float = DEFAULT_MATCHER_EPS,
        squeeze_output: bool = True,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.window_size = window_size
        self.eps = eps
        self.squeeze_output = squeeze_output

    def _prepare_lengths(
        self,
        lengths: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # If no per-sequence valid lengths are provided, all sequences use full length L.
        if lengths is None:
            return torch.full((batch_size,), float(seq_len), device=device, dtype=dtype)

        valid_lengths = torch.as_tensor(lengths, device=device, dtype=dtype)
        if valid_lengths.ndim == 0:
            valid_lengths = valid_lengths.view(1).expand(batch_size)
        elif valid_lengths.ndim == 1:
            if valid_lengths.shape[0] == 1:
                valid_lengths = valid_lengths.expand(batch_size)
            elif valid_lengths.shape[0] != batch_size:
                raise ValueError(f"lengths must have shape [B] with B={batch_size}, got {valid_lengths.shape}.")
        else:
            raise ValueError(f"lengths must be scalar or 1D tensor, got ndim={valid_lengths.ndim}.")

        if torch.any(valid_lengths < 1):
            raise ValueError("All lengths must be >= 1.")
        if torch.any(valid_lengths > seq_len):
            raise ValueError(f"All lengths must be <= seq_len ({seq_len}).")
        return valid_lengths

    def _normalize_logits_to_matrix(
        self,
        logit_values: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Normalize controls to a 2D matrix [batch_like, glimpse_like].
        # Broadcasting to the actual [B, G] happens in `_expand_logits_to_batch_and_glimpses`.
        expanded_logits = torch.as_tensor(logit_values, device=device, dtype=dtype).to(device=device, dtype=dtype)

        if expanded_logits.ndim == 0:
            return expanded_logits.view(1, 1)

        if expanded_logits.ndim == 1:
            return expanded_logits.view(1, expanded_logits.shape[0])

        if expanded_logits.ndim == 2:
            return expanded_logits

        raise ValueError(f"logits must be scalar, 1D, or 2D tensor, got ndim={expanded_logits.ndim}.")

    def _expand_logits_to_batch_and_glimpses(
        self,
        logits_matrix: torch.Tensor,
        batch_size: int,
        num_glimpses: int,
        logits_name: str,
    ) -> torch.Tensor:
        # Expand [batch_like, glimpse_like] controls to the concrete [B, G] shape.
        # Each dimension can be either exact-size or singleton for broadcasting.
        logits_batch, logits_glimpses = logits_matrix.shape
        if logits_batch not in (1, batch_size):
            raise ValueError(f"{logits_name} batch dim must be 1 or {batch_size}, got {logits_batch}.")
        if logits_glimpses not in (1, num_glimpses):
            raise ValueError(f"{logits_name} glimpse dim must be 1 or {num_glimpses}, got {logits_glimpses}.")
        return logits_matrix.expand(batch_size, num_glimpses)

    def _format_output(
        self,
        output: torch.Tensor,
        input_was_unbatched: bool,
        num_glimpses: int,
    ) -> torch.Tensor:
        if not self.squeeze_output:
            return output
        if input_was_unbatched:
            return output[0, 0] if num_glimpses == 1 else output[0]
        return output[:, 0] if num_glimpses == 1 else output

    def forward(
        self,
        input_embeddings: torch.Tensor,
        center_logits: torch.Tensor,
        zoom_logits: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Normalize embeddings to [B, L, D] so all downstream ops are fully vectorized.
        was_unbatched = input_embeddings.ndim == 2
        if was_unbatched:
            input_embeddings = input_embeddings.unsqueeze(0)
        if input_embeddings.ndim != 3:
            raise ValueError(
                f"input_embeddings must have shape [L, D] or [B, L, D], got shape={tuple(input_embeddings.shape)}."
            )

        batch_size, seq_len, emb_dim = input_embeddings.shape
        device = input_embeddings.device
        dtype = input_embeddings.dtype
        if not input_embeddings.dtype.is_floating_point:
            raise TypeError(f"input_embeddings must be floating-point tensor, got dtype={input_embeddings.dtype}.")

        # Resolve valid sequence lengths and broadcast control logits to [B, G].
        valid_lengths = self._prepare_lengths(lengths, batch_size, seq_len, device, dtype)
        center_logits_matrix = self._normalize_logits_to_matrix(center_logits, device, dtype)
        zoom_logits_matrix = self._normalize_logits_to_matrix(zoom_logits, device, dtype)

        num_glimpses = max(center_logits_matrix.shape[1], zoom_logits_matrix.shape[1])
        if num_glimpses <= 0:
            raise ValueError("num_glimpses must be > 0 after expanding logits.")

        center_logits_expanded = self._expand_logits_to_batch_and_glimpses(
            center_logits_matrix,
            batch_size,
            num_glimpses,
            "center_logits",
        )
        zoom_logits_expanded = self._expand_logits_to_batch_and_glimpses(
            zoom_logits_matrix,
            batch_size,
            num_glimpses,
            "zoom_logits",
        )

        # Convert unconstrained logits to normalized controls in [0, 1].
        center_fraction = torch.sigmoid(center_logits_expanded)
        zoom_fraction = torch.sigmoid(zoom_logits_expanded)

        # Compute geometric parameters (center bounds and zoom-dependent spacing).
        half_window = (self.window_size - 1) / 2.0
        if self.window_size == 1:
            max_spacing = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            max_spacing = (valid_lengths - 1.0) / float(self.window_size - 1)

        center_min_raw = torch.full((batch_size,), half_window, device=device, dtype=dtype)
        center_max_raw = (valid_lengths - 1.0) - half_window
        center_midpoint = (valid_lengths - 1.0) / 2.0
        window_fits_inside_valid_length = center_min_raw <= center_max_raw
        center_min = torch.where(window_fits_inside_valid_length, center_min_raw, center_midpoint)
        center_max = torch.where(window_fits_inside_valid_length, center_max_raw, center_midpoint)

        # Compute center and spacing per [batch, glimpse].
        center_positions = center_min.unsqueeze(1) + center_fraction * (center_max - center_min).unsqueeze(1)
        if self.window_size == 1:
            sampling_spacings = torch.ones_like(center_positions)
        else:
            sampling_spacings = 1.0 + zoom_fraction * (max_spacing - 1.0).unsqueeze(1)

        # Real-valued sampling coordinates per slot: u_j = center + spacing * offset_j.
        slot_offsets = torch.arange(self.window_size, device=device, dtype=dtype) - half_window
        sample_positions = center_positions.unsqueeze(-1) + sampling_spacings.unsqueeze(-1) * slot_offsets.view(1, 1, self.window_size)

        # Linear interpolation endpoints and mixing factor:
        # y = (1 - alpha) * E[left] + alpha * E[right].
        left_indices_float = torch.floor(sample_positions)
        interpolation_alpha = (sample_positions - left_indices_float).unsqueeze(-1)
        left_indices = left_indices_float.to(torch.long)
        right_indices = left_indices + 1

        # Border policy: replicate edges by clamping indices to [0, valid_length - 1].
        max_valid_index = (valid_lengths.to(torch.long) - 1).view(batch_size, 1, 1)
        zero_idx = torch.zeros((), device=device, dtype=torch.long)
        left_indices = torch.minimum(torch.maximum(left_indices, zero_idx), max_valid_index)
        right_indices = torch.minimum(torch.maximum(right_indices, zero_idx), max_valid_index)

        # Gather interpolation endpoints for all [B, G, W] coordinates in parallel.
        flat_left_indices = left_indices.reshape(batch_size, -1)
        flat_right_indices = right_indices.reshape(batch_size, -1)

        left_embeddings = torch.gather(
            input_embeddings,
            dim=1,
            index=flat_left_indices.unsqueeze(-1).expand(batch_size, flat_left_indices.shape[1], emb_dim),
        ).reshape(batch_size, num_glimpses, self.window_size, emb_dim)
        right_embeddings = torch.gather(
            input_embeddings,
            dim=1,
            index=flat_right_indices.unsqueeze(-1).expand(batch_size, flat_right_indices.shape[1], emb_dim),
        ).reshape(batch_size, num_glimpses, self.window_size, emb_dim)

        glimpse_windows = (1.0 - interpolation_alpha) * left_embeddings + interpolation_alpha * right_embeddings
        formatted = self._format_output(
            glimpse_windows,
            input_was_unbatched=was_unbatched,
            num_glimpses=num_glimpses,
        )

        if not return_aux:
            return formatted
        aux = {
            "center_fraction": center_fraction,
            "zoom_fraction": zoom_fraction,
            "center_positions": center_positions,
            "sampling_spacings": sampling_spacings,
            "sample_positions": sample_positions,
            "interpolation_alpha": interpolation_alpha.squeeze(-1),
            "left_indices": left_indices,
            "right_indices": right_indices,
        }
        return formatted, aux
