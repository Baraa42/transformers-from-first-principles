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


def test_cached_decode_matches_full_attention_and_grows_cache() -> None:
    torch.manual_seed(7)
    batch_size, prefix_length, d_model, n_heads = 2, 5, 16, 4
    d_head = d_model // n_heads
    attention = SelfAttention(d_model, n_heads)
    x = torch.randn(batch_size, prefix_length + 1, d_model)

    full_output = attention(x)
    reference = full_output[:, -1:, :]

    prefill_output, cache = attention(x[:, :prefix_length, :], use_cache=True)
    old_k, old_v = cache
    cached_output, updated_cache = attention(
        x[:, prefix_length:, :],
        use_cache=True,
        kv_cache=cache,
    )
    updated_k, updated_v = updated_cache

    assert prefill_output.shape == (batch_size, prefix_length, d_model)
    assert old_k.shape == (batch_size, n_heads, prefix_length, d_head)
    assert old_v.shape == (batch_size, n_heads, prefix_length, d_head)
    assert updated_k.shape == (batch_size, n_heads, prefix_length + 1, d_head)
    assert updated_v.shape == (batch_size, n_heads, prefix_length + 1, d_head)
    assert torch.equal(updated_k[:, :, :prefix_length, :], old_k)
    assert torch.equal(updated_v[:, :, :prefix_length, :], old_v)
    assert torch.allclose(cached_output, reference, atol=1e-6, rtol=1e-5)


def test_attention_can_return_weights_and_cache_together() -> None:
    batch_size, sequence_length, d_model, n_heads = 2, 4, 16, 4
    attention = SelfAttention(d_model, n_heads)

    output, weights, cache = attention(
        torch.randn(batch_size, sequence_length, d_model),
        return_attention=True,
        use_cache=True,
    )

    assert output.shape == (batch_size, sequence_length, d_model)
    assert weights.shape == (batch_size, n_heads, sequence_length, sequence_length)
    assert cache[0].shape == (batch_size, n_heads, sequence_length, d_model // n_heads)
    assert cache[1].shape == cache[0].shape


def test_attention_rejects_cache_when_caching_is_disabled() -> None:
    attention = SelfAttention(d_model=8, n_heads=2)
    cache = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

    with pytest.raises(ValueError, match="requires use_cache"):
        attention(torch.randn(1, 1, 8), kv_cache=cache)


def test_attention_rejects_multi_token_cached_decode() -> None:
    attention = SelfAttention(d_model=8, n_heads=2)
    _, cache = attention(torch.randn(1, 3, 8), use_cache=True)

    with pytest.raises(ValueError, match="exactly one"):
        attention(torch.randn(1, 2, 8), use_cache=True, kv_cache=cache)


def test_attention_rejects_malformed_cache_shape() -> None:
    attention = SelfAttention(d_model=8, n_heads=2)
    malformed = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))

    with pytest.raises(ValueError, match="shape"):
        attention(torch.randn(1, 1, 8), use_cache=True, kv_cache=malformed)
