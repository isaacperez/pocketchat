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
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")
        if padding_idx is not None and not (0 <= padding_idx < vocab_size):
            raise ValueError(
                f"padding_idx must be in [0, {vocab_size - 1}] when provided, got {padding_idx}."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.char_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype.is_floating_point:
            raise TypeError(f"token_ids must be integer tensor, got dtype={token_ids.dtype}.")
        return self.char_embeddings(token_ids.long())


class Decomposer(nn.Module):
    """
    Decompose each input embedding into K component embeddings.

    Architecture:
    1) RMSNorm on the last dimension D
    2) Linear projection D -> (K * D)
    3) GeLU activation
    4) Residual add with the original embedding broadcast to K slots

    Supported input ranks:
    - [B, D] -> [B, K, D]
    - [B, H, D] -> [B, H, K, D]
    - [B, H, P, D] -> [B, H, P, K, D]
    - and in general any [..., D] -> [..., K, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_components: int,
        eps: float = DEFAULT_MATCHER_EPS,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}.")
        if num_components <= 0:
            raise ValueError(f"num_components must be > 0, got {num_components}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.embedding_dim = embedding_dim
        self.num_components = num_components
        self.eps = eps

        self.rms_norm = nn.RMSNorm(embedding_dim, eps=eps)
        self.proj = nn.Linear(embedding_dim, num_components * embedding_dim, bias=bias)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2:
            raise ValueError(f"inputs must have at least 2 dims [..., D], got shape={tuple(inputs.shape)}.")
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Last dim mismatch: expected D={self.embedding_dim}, got D={inputs.shape[-1]}."
            )
        if not inputs.dtype.is_floating_point:
            raise TypeError(f"inputs must be floating-point tensor, got dtype={inputs.dtype}.")

        normalized = self.rms_norm(inputs)
        deltas = self.activation(self.proj(normalized))
        deltas = deltas.reshape(*inputs.shape[:-1], self.num_components, self.embedding_dim)

        residual_base = inputs.unsqueeze(-2)
        return residual_base + deltas


class Matcher(nn.Module):
    """
    Apply adapter modulation to input embeddings and compare with references.

    Input:
    - input_embeddings: [B, T, D]
    - adapters: [M, D] or [B, M, D]
    - references: [M, D] or [B, M, D]

    Output:
    - heatmap similarities in [-1, 1] with shape [B, M, T]
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
        if input_embeddings.ndim != 3:
            raise ValueError(
                f"input_embeddings must have shape [B, T, D], got ndim={input_embeddings.ndim}."
            )
        if references.ndim not in (2, 3):
            raise ValueError(f"references must have shape [M, D] or [B, M, D], got ndim={references.ndim}.")
        if adapters.ndim != references.ndim:
            raise ValueError(
                f"adapters and references must have the same ndim, got {adapters.ndim} and {references.ndim}."
            )

        batch_size, _, emb_dim = input_embeddings.shape

        if references.ndim == 2:
            if references.shape != adapters.shape:
                raise ValueError(
                    f"references and adapters shapes must match for 2D mode, got {references.shape} and {adapters.shape}."
                )
            if references.shape[-1] != emb_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: input D={emb_dim}, refs/adapters D={references.shape[-1]}."
                )

            adapted = input_embeddings.unsqueeze(2) * adapters.unsqueeze(0).unsqueeze(0)
            adapted_norm = F.normalize(adapted, dim=-1, eps=self.eps)
            references_norm = F.normalize(references, dim=-1, eps=self.eps)
            similarities = torch.einsum("btmd,md->btm", adapted_norm, references_norm)
            return similarities.transpose(1, 2).contiguous()

        if references.shape != adapters.shape:
            raise ValueError(
                f"references and adapters shapes must match for 3D mode, got {references.shape} and {adapters.shape}."
            )
        if references.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: input B={batch_size}, refs/adapters B={references.shape[0]}."
            )
        if references.shape[-1] != emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input D={emb_dim}, refs/adapters D={references.shape[-1]}."
            )

        adapted = input_embeddings.unsqueeze(2) * adapters.unsqueeze(1)
        adapted_norm = F.normalize(adapted, dim=-1, eps=self.eps)
        references_norm = F.normalize(references, dim=-1, eps=self.eps)
        similarities = torch.einsum("btmd,bmd->btm", adapted_norm, references_norm)
        return similarities.transpose(1, 2).contiguous()


class Glimpse(nn.Module):
    """
    Differentiable 1D glimpse over character embeddings using linear interpolation.

    Input embeddings:
    - unbatched: [L, D]
    - batched: [B, L, D]

    Output windows:
    - with squeeze_output=True (default):
      - [W, D] when input is unbatched and num_glimpses=1
      - [G, W, D] when input is unbatched and num_glimpses=G
      - [B, W, D] when input is batched and num_glimpses=1
      - [B, G, W, D] when input is batched and num_glimpses=G
    - with squeeze_output=False:
      - always [B, G, W, D]

    Center/zoom controls are represented as logits and passed through sigmoid.
    This keeps optimization unconstrained while ensuring valid [0, 1] control values.
    """

    def __init__(
        self,
        window_size: int,
        num_glimpses: int = 1,
        init_center_fraction: float = 0.5,
        init_zoom_fraction: float = 0.0,
        learnable: bool = True,
        eps: float = DEFAULT_MATCHER_EPS,
        squeeze_output: bool = True,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}.")
        if num_glimpses <= 0:
            raise ValueError(f"num_glimpses must be > 0, got {num_glimpses}.")
        if not (0.0 <= init_center_fraction <= 1.0):
            raise ValueError(f"init_center_fraction must be in [0, 1], got {init_center_fraction}.")
        if not (0.0 <= init_zoom_fraction <= 1.0):
            raise ValueError(f"init_zoom_fraction must be in [0, 1], got {init_zoom_fraction}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.window_size = window_size
        self.num_glimpses = num_glimpses
        self.eps = eps
        self.squeeze_output = squeeze_output

        center_logits = torch.full((num_glimpses,), self._prob_to_logit(init_center_fraction), dtype=torch.float32)
        zoom_logits = torch.full((num_glimpses,), self._prob_to_logit(init_zoom_fraction), dtype=torch.float32)
        if learnable:
            self.center_logits = nn.Parameter(center_logits)
            self.zoom_logits = nn.Parameter(zoom_logits)
        else:
            self.register_buffer("center_logits", center_logits)
            self.register_buffer("zoom_logits", zoom_logits)

    @staticmethod
    def _prob_to_logit(prob: float, eps: float = 1e-6) -> float:
        clipped = min(max(prob, eps), 1.0 - eps)
        return float(torch.logit(torch.tensor(clipped)).item())

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

    def _expand_logits(
        self,
        logit_values: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Accepted shapes:
        # - scalar: shared across batch and glimpses
        # - [G]: one value per glimpse, shared across batch
        # - [B]: one value per batch element, shared across glimpses
        # - [B, G] (or broadcastable [1, G] / [B, 1])
        expanded_logits = torch.as_tensor(logit_values, device=device, dtype=dtype).to(device=device, dtype=dtype)

        if expanded_logits.ndim == 0:
            return expanded_logits.view(1, 1).expand(batch_size, self.num_glimpses)

        if expanded_logits.ndim == 1:
            n = expanded_logits.shape[0]
            if n == self.num_glimpses:
                return expanded_logits.view(1, self.num_glimpses).expand(batch_size, self.num_glimpses)
            if n == batch_size:
                return expanded_logits.view(batch_size, 1).expand(batch_size, self.num_glimpses)
            if n == 1:
                return expanded_logits.view(1, 1).expand(batch_size, self.num_glimpses)
            raise ValueError(
                f"1D logits must have size 1, num_glimpses ({self.num_glimpses}), or batch_size ({batch_size}); got {n}."
            )

        if expanded_logits.ndim == 2:
            logits_batch, logits_glimpses = expanded_logits.shape
            if logits_batch not in (1, batch_size):
                raise ValueError(f"2D logits batch dim must be 1 or {batch_size}, got {logits_batch}.")
            if logits_glimpses not in (1, self.num_glimpses):
                raise ValueError(f"2D logits glimpse dim must be 1 or {self.num_glimpses}, got {logits_glimpses}.")
            return expanded_logits.expand(batch_size, self.num_glimpses)

        raise ValueError(f"logits must be scalar, 1D, or 2D tensor, got ndim={expanded_logits.ndim}.")

    def _format_output(self, output: torch.Tensor, input_was_unbatched: bool) -> torch.Tensor:
        if not self.squeeze_output:
            return output
        if input_was_unbatched:
            return output[0, 0] if self.num_glimpses == 1 else output[0]
        return output[:, 0] if self.num_glimpses == 1 else output

    def forward(
        self,
        input_embeddings: torch.Tensor,
        center_logits: torch.Tensor | None = None,
        zoom_logits: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Normalize input to [B, L, D] for unified vectorized computation.
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

        valid_lengths = self._prepare_lengths(lengths, batch_size, seq_len, device, dtype)
        center_logits_expanded = self._expand_logits(
            self.center_logits if center_logits is None else center_logits,
            batch_size,
            device,
            dtype,
        )
        zoom_logits_expanded = self._expand_logits(
            self.zoom_logits if zoom_logits is None else zoom_logits,
            batch_size,
            device,
            dtype,
        )

        # Convert unconstrained logits to valid normalized controls in [0, 1].
        center_fraction = torch.sigmoid(center_logits_expanded)
        zoom_fraction = torch.sigmoid(zoom_logits_expanded)

        # Geometric window parameters.
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

        # Center c and spacing s for each [batch, glimpse].
        center_positions = center_min.unsqueeze(1) + center_fraction * (center_max - center_min).unsqueeze(1)
        if self.window_size == 1:
            sampling_spacings = torch.ones_like(center_positions)
        else:
            sampling_spacings = 1.0 + zoom_fraction * (max_spacing - 1.0).unsqueeze(1)

        # Real-valued sample positions u_j = c + s*(j - h).
        slot_offsets = torch.arange(self.window_size, device=device, dtype=dtype) - half_window
        sample_positions = center_positions.unsqueeze(-1) + sampling_spacings.unsqueeze(-1) * slot_offsets.view(1, 1, self.window_size)

        # Linear interpolation setup: y = (1-alpha)*E[i0] + alpha*E[i1].
        left_indices_float = torch.floor(sample_positions)
        interpolation_alpha = (sample_positions - left_indices_float).unsqueeze(-1)
        left_indices = left_indices_float.to(torch.long)
        right_indices = left_indices + 1

        # Border policy = replicate/clamp to valid range [0, length-1].
        max_valid_index = (valid_lengths.to(torch.long) - 1).view(batch_size, 1, 1)
        zero_idx = torch.zeros((), device=device, dtype=torch.long)
        left_indices = torch.minimum(torch.maximum(left_indices, zero_idx), max_valid_index)
        right_indices = torch.minimum(torch.maximum(right_indices, zero_idx), max_valid_index)

        # Gather left/right endpoints in parallel for all batch x glimpse x slot.
        flat_left_indices = left_indices.reshape(batch_size, -1)
        flat_right_indices = right_indices.reshape(batch_size, -1)

        left_embeddings = torch.gather(
            input_embeddings,
            dim=1,
            index=flat_left_indices.unsqueeze(-1).expand(batch_size, flat_left_indices.shape[1], emb_dim),
        ).reshape(batch_size, self.num_glimpses, self.window_size, emb_dim)
        right_embeddings = torch.gather(
            input_embeddings,
            dim=1,
            index=flat_right_indices.unsqueeze(-1).expand(batch_size, flat_right_indices.shape[1], emb_dim),
        ).reshape(batch_size, self.num_glimpses, self.window_size, emb_dim)

        glimpse_windows = (1.0 - interpolation_alpha) * left_embeddings + interpolation_alpha * right_embeddings
        formatted = self._format_output(glimpse_windows, input_was_unbatched=was_unbatched)

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
