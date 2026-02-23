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


class AdapterReferenceMatcher(nn.Module):
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
