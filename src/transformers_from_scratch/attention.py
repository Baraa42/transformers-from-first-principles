"""Explicit causal multi-head self-attention."""

import torch
from torch import nn

from .rope import apply_rope

KVCache = tuple[torch.Tensor, torch.Tensor]


class SelfAttention(nn.Module):
    """Causal multi-head attention with RoPE applied to queries and keys."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.Wq: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.Wk: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.Wv: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.Wo: nn.Linear[torch.Tensor, torch.Tensor] = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, KVCache]
        | tuple[torch.Tensor, torch.Tensor, KVCache]
    ):
        """Attend causally over ``x`` shaped ``(B, n, d_model)``.

        With caching enabled, prefill returns RoPE-rotated keys and ordinary values.
        A supplied cache restricts input to one new token and determines its RoPE offset.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected (B, n, {self.d_model}), got {tuple(x.shape)}")
        if kv_cache is not None and not use_cache:
            raise ValueError("kv_cache requires use_cache=True")

        B, n, _ = x.shape
        if n < 1:
            raise ValueError("SelfAttention requires at least one token")

        offset = 0
        k_cache: torch.Tensor | None = None
        v_cache: torch.Tensor | None = None
        if kv_cache is not None:
            if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
                raise ValueError("kv_cache must be a (K, V) tensor tuple")
            k_cache, v_cache = kv_cache
            if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
                raise ValueError("kv_cache must contain tensors")
            if n != 1:
                raise ValueError("cached decode requires exactly one input token")
            if k_cache.ndim != 4 or v_cache.ndim != 4:
                raise ValueError("cached K and V must have shape (B, H, T, d_head)")
            if k_cache.shape != v_cache.shape:
                raise ValueError("cached K and V must have the same shape")
            if (
                k_cache.shape[0] != B
                or k_cache.shape[1] != self.n_heads
                or k_cache.shape[2] < 1
                or k_cache.shape[3] != self.d_head
            ):
                raise ValueError(
                    f"kv_cache must have shape ({B}, {self.n_heads}, T, {self.d_head}) with T > 0"
                )
            if k_cache.device != x.device or v_cache.device != x.device:
                raise ValueError("kv_cache and x must be on the same device")
            offset = k_cache.shape[2]

        # Projection: (B, n, d_model); reshape + transpose: (B, H, n, d_head).
        q = self.Wq(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        q = apply_rope(q, offset=offset)
        k = apply_rope(k, offset=offset)

        if k_cache is not None and v_cache is not None:
            if k_cache.dtype != k.dtype or v_cache.dtype != v.dtype:
                raise ValueError("kv_cache dtype must match projected K and V dtype")
            k = torch.cat((k_cache, k), dim=2)
            v = torch.cat((v_cache, v), dim=2)

        # Q K^T yields (B, H, n, n) normally or (B, H, 1, P+1) during cached decode.
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        if kv_cache is None:
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # attention @ V -> (B, H, n, d_head); concatenate heads -> (B, n, d_model).
        output = (attn @ v).transpose(1, 2).contiguous().view(B, n, self.d_model)
        output = self.Wo(output)

        if use_cache:
            new_cache = (k, v)
            if return_attention:
                return output, attn, new_cache
            return output, new_cache
        if return_attention:
            return output, attn
        return output
