"""Explicit causal multi-head self-attention."""

import torch
from torch import nn

from .rope import apply_rope


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
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend causally over ``x`` shaped ``(B, n, d_model)``.

        The optional attention return is an inspection path for tests; normal
        callers receive only transformed token representations.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected (B, n, {self.d_model}), got {tuple(x.shape)}")
        B, n, _ = x.shape
        # Projection: (B, n, d_model); reshape + transpose: (B, H, n, d_head).
        q = self.Wq(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        q, k = apply_rope(q), apply_rope(k)
        # Q K^T: (B, H, n, n), followed by the explicit causal mask and softmax.
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # attention @ V -> (B, H, n, d_head); concatenate heads -> (B, n, d_model).
        output = (attn @ v).transpose(1, 2).contiguous().view(B, n, self.d_model)
        output = self.Wo(output)
        return (output, attn) if return_attention else output
