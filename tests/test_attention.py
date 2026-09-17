import pytest
import torch

from transformers_from_scratch.attention import SelfAttention


def test_attention_shapes_and_causal_mask() -> None:
    B, n, d_model, n_heads = 2, 5, 16, 4
    attention = SelfAttention(d_model, n_heads)
    output, weights = attention(torch.randn(B, n, d_model), return_attention=True)

    assert output.shape == (B, n, d_model)
    assert weights.shape == (B, n_heads, n, n)
    # Every probability above the causal diagonal is exactly zero after softmax.
    upper_triangle = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[:, :, upper_triangle]) == 0


def test_attention_requires_even_head_split() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SelfAttention(d_model=15, n_heads=4)
