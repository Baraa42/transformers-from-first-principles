"""A small, explicit decoder-only language model."""

import torch
from torch import nn

from .layers import DecoderBlock


class TinyDecoderLM(nn.Module):
    """Token embeddings, repeated decoder blocks, and an LM head."""

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 16,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map IDs ``(B, n)`` to raw vocabulary logits ``(B, n, V)``.

        Embedding lookup selects learned rows from a ``(V, d_model)`` matrix,
        then blocks/final norm preserve ``(B, n, d_model)``. No softmax is
        applied because loss and sampling are deliberately deferred.
        """
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token IDs (B, n), got {tuple(token_ids.shape)}")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
