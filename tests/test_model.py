import pytest
import torch

from transformers_from_scratch.layers import DecoderBlock
from transformers_from_scratch.model import TinyDecoderLM


def test_tiny_decoder_default_shape() -> None:
    B, n, V = 2, 5, 100
    token_ids = torch.randint(0, V, (B, n))
    model = TinyDecoderLM(vocab_size=V, d_model=16, n_heads=4, d_ff=64, n_layers=3)

    logits = model(token_ids)

    assert token_ids.shape == (2, 5)
    assert logits.shape == (2, 5, 100)


def test_decoder_block_preserves_shape() -> None:
    block = DecoderBlock(d_model=16, n_heads=4, d_ff=64)
    output = block(torch.randn(2, 5, 16))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 5, 16)


def test_decoder_block_prefill_returns_cache() -> None:
    batch_size, prefix_length, d_model, n_heads = 2, 5, 16, 4
    d_head = d_model // n_heads
    block = DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=64)

    output, cache = block(
        torch.randn(batch_size, prefix_length, d_model),
        use_cache=True,
    )
    k, v = cache

    assert output.shape == (batch_size, prefix_length, d_model)
    assert k.shape == (batch_size, n_heads, prefix_length, d_head)
    assert v.shape == k.shape


def test_decoder_block_cached_decode_grows_cache_and_preserves_prefix() -> None:
    batch_size, prefix_length, d_model, n_heads = 2, 5, 16, 4
    d_head = d_model // n_heads
    block = DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=64)
    prefix = torch.randn(batch_size, prefix_length, d_model)

    _, cache = block(prefix, use_cache=True)
    old_k, old_v = cache
    output, updated_cache = block(
        torch.randn(batch_size, 1, d_model),
        use_cache=True,
        kv_cache=cache,
    )
    updated_k, updated_v = updated_cache

    assert output.shape == (batch_size, 1, d_model)
    assert old_k.shape == (batch_size, n_heads, prefix_length, d_head)
    assert old_v.shape == old_k.shape
    assert updated_k.shape == (batch_size, n_heads, prefix_length + 1, d_head)
    assert updated_v.shape == updated_k.shape
    assert torch.equal(updated_k[:, :, :prefix_length, :], old_k)
    assert torch.equal(updated_v[:, :, :prefix_length, :], old_v)


def test_decoder_block_cached_decode_matches_full_block_output() -> None:
    torch.manual_seed(11)
    batch_size, prefix_length, d_model = 2, 5, 16
    block = DecoderBlock(d_model=d_model, n_heads=4, d_ff=64)
    x = torch.randn(batch_size, prefix_length + 1, d_model)

    full_output = block(x)
    reference = full_output[:, -1:, :]
    _, cache = block(x[:, :prefix_length, :], use_cache=True)
    cached_output, _ = block(
        x[:, prefix_length:, :],
        use_cache=True,
        kv_cache=cache,
    )

    assert torch.allclose(cached_output, reference, atol=1e-6, rtol=1e-5)


def test_decoder_block_rejects_cache_when_caching_is_disabled() -> None:
    block = DecoderBlock(d_model=16, n_heads=4, d_ff=64)
    _, cache = block(torch.randn(2, 5, 16), use_cache=True)

    with pytest.raises(ValueError, match="requires use_cache"):
        block(torch.randn(2, 1, 16), kv_cache=cache)
