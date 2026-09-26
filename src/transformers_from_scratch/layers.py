"""The feed-forward network and pre-norm decoder block."""

import torch
from torch import nn
from torch.nn import functional as F

from .attention import KVCache, SelfAttention


class MLP(nn.Module):
    """Per-token feed-forward network: ``d_model -> d_ff -> d_model``."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up, self.down = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n, d_model) -> (B, n, d_ff) -> (B, n, d_model).
        return self.down(F.silu(self.up(x)))


class DecoderBlock(nn.Module):
    """Pre-norm block preserving ``(B, n, d_model)``.

    Attention communicates between tokens; the MLP transforms features within
    a token. Each residual adds a learned update back to its input, and
    LayerNorm operates over the final feature dimension.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.mlp = SelfAttention(d_model, n_heads), MLP(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        *,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        if kv_cache is not None and not use_cache:
            raise ValueError("kv_cache requires use_cache=True")

        if use_cache:
            attn_output, new_cache = self.attn(
                self.norm1(x),
                use_cache=True,
                kv_cache=kv_cache,
            )
            x = x + attn_output
            x = x + self.mlp(self.norm2(x))
            return x, new_cache

        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
