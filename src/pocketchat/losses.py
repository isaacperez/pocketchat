import torch
import torch.nn.functional as F


def _select_targets_for_prediction_position(
    targets: torch.Tensor,
    prediction_position: int,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if targets.ndim == 2:
        return targets
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape [B, N_pred] or [B, L, N_pred], "
            f"got shape={tuple(targets.shape)}."
        )
    if targets.shape[1] == 0:
        raise ValueError("targets sequence length L must be > 0.")

    if lengths is not None:
        if lengths.ndim != 1:
            raise ValueError(f"lengths must have shape [B], got shape={tuple(lengths.shape)}.")
        if lengths.shape[0] != targets.shape[0]:
            raise ValueError(
                f"lengths batch mismatch: expected B={targets.shape[0]}, got {lengths.shape[0]}."
            )
        if lengths.dtype.is_floating_point:
            raise TypeError(f"lengths must be integer tensor, got dtype={lengths.dtype}.")

        lengths = lengths.to(device=targets.device, dtype=torch.long)
        if torch.any(lengths < 1):
            raise ValueError("All lengths must be >= 1.")
        if torch.any(lengths > targets.shape[1]):
            raise ValueError(f"All lengths must be <= target sequence length ({targets.shape[1]}).")

        row_index = torch.arange(targets.shape[0], device=targets.device)
        return targets[row_index, lengths - 1, :]

    return targets[:, prediction_position, :]


def _build_horizon_weights(
    num_predictions: int,
    horizon_decay: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_predictions <= 0:
        raise ValueError(f"num_predictions must be > 0, got {num_predictions}.")
    if horizon_decay <= 0:
        raise ValueError(f"horizon_decay must be > 0, got {horizon_decay}.")

    if horizon_decay == 1.0:
        return torch.ones(num_predictions, device=device, dtype=dtype)

    positions = torch.arange(num_predictions, device=device, dtype=dtype)
    decay_base = torch.tensor(float(horizon_decay), device=device, dtype=dtype)
    return torch.pow(decay_base, positions)


def next_char_cross_entropy_loss(
    next_char_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    prediction_position: int = -1,
    lengths: torch.Tensor | None = None,
    horizon_decay: float = 1.0,
) -> torch.Tensor:
    """
    Cross-entropy loss for Cortex next-char prediction.

    Inputs:
    - next_char_logits: [B, N_pred, V]
    - targets:
      - [B, N_pred], or
      - [B, L, N_pred] (e.g. dataset targets per input position)
    - lengths (optional): [B], last valid position per sample is lengths-1
    - horizon_decay: weight decay across prediction horizon.
      Weight for horizon h is horizon_decay**h (h=0 is immediate next char).

    If targets are [B, L, N_pred], this function selects one prediction position
    before computing CE:
    - if `lengths` is provided, uses per-sample index `lengths - 1`
    - otherwise uses `prediction_position` (default: last position)
    """

    if next_char_logits.ndim != 3:
        raise ValueError(
            "next_char_logits must have shape [B, N_pred, V], "
            f"got shape={tuple(next_char_logits.shape)}."
        )
    if not next_char_logits.dtype.is_floating_point:
        raise TypeError(f"next_char_logits must be floating-point tensor, got dtype={next_char_logits.dtype}.")

    selected_targets = _select_targets_for_prediction_position(targets, prediction_position, lengths=lengths)
    if not selected_targets.dtype.is_floating_point and selected_targets.dtype != torch.long:
        selected_targets = selected_targets.long()
    if selected_targets.dtype != torch.long:
        raise TypeError(f"targets must be integer tensor, got dtype={selected_targets.dtype}.")

    batch_size, num_predictions, vocab_size = next_char_logits.shape
    if selected_targets.shape != (batch_size, num_predictions):
        raise ValueError(
            "Target shape mismatch after selection: "
            f"expected {(batch_size, num_predictions)}, got {tuple(selected_targets.shape)}."
        )

    flat_logits = next_char_logits.reshape(batch_size * num_predictions, vocab_size)
    flat_targets = selected_targets.reshape(batch_size * num_predictions)

    token_losses = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(batch_size, num_predictions)

    horizon_weights = _build_horizon_weights(
        num_predictions=num_predictions,
        horizon_decay=horizon_decay,
        device=next_char_logits.device,
        dtype=next_char_logits.dtype,
    )
    weighted_losses = token_losses * horizon_weights.unsqueeze(0)

    valid_mask = selected_targets != ignore_index
    weighted_valid_counts = (valid_mask.to(next_char_logits.dtype) * horizon_weights.unsqueeze(0)).sum()
    if weighted_valid_counts <= 0:
        raise ValueError("No valid targets available to compute loss (all targets are ignore_index).")

    return weighted_losses.sum() / weighted_valid_counts
