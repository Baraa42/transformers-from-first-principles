import pytest
import torch

from transformers_from_scratch.layers import DecoderBlock
from transformers_from_scratch.model import TinyDecoderLM


def test_tiny_decoder_default_shape() -> None:
    B, n, V = 2, 5, 100
    token_ids = torch.randint(0, V, (B, n))
    model = TinyDecoderLM(vocab_size=V, d_model=16, n_heads=4, d_ff=64, n_layers=3)

    logits = model(token_ids)

    assert isinstance(logits, torch.Tensor)
    assert token_ids.shape == (2, 5)
    assert logits.shape == (2, 5, 100)


def test_tiny_decoder_prefill_returns_one_cache_per_layer() -> None:
    batch_size, prefix_length, vocab_size = 2, 5, 100
    d_model, n_heads, n_layers = 16, 4, 3
    d_head = d_model // n_heads
    model = TinyDecoderLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=64,
        n_layers=n_layers,
    )

    logits, cache = model(
        torch.randint(0, vocab_size, (batch_size, prefix_length)),
        use_cache=True,
    )

    assert logits.shape == (batch_size, prefix_length, vocab_size)
    assert isinstance(cache, tuple)
    assert len(cache) == n_layers
    for k, v in cache:
        assert k.shape == (batch_size, n_heads, prefix_length, d_head)
        assert v.shape == k.shape


def test_tiny_decoder_cached_decode_grows_every_layer_cache() -> None:
    batch_size, prefix_length, vocab_size = 2, 5, 100
    d_model, n_heads, n_layers = 16, 4, 3
    d_head = d_model // n_heads
    model = TinyDecoderLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=64,
        n_layers=n_layers,
    )
    _, cache = model(
        torch.randint(0, vocab_size, (batch_size, prefix_length)),
        use_cache=True,
    )

    logits, updated_cache = model(
        torch.randint(0, vocab_size, (batch_size, 1)),
        use_cache=True,
        kv_cache=cache,
    )

    assert logits.shape == (batch_size, 1, vocab_size)
    assert len(updated_cache) == n_layers
    for (old_k, old_v), (updated_k, updated_v) in zip(cache, updated_cache, strict=True):
        assert old_k.shape == (batch_size, n_heads, prefix_length, d_head)
        assert old_v.shape == old_k.shape
        assert updated_k.shape == (batch_size, n_heads, prefix_length + 1, d_head)
        assert updated_v.shape == updated_k.shape
        assert torch.equal(updated_k[:, :, :prefix_length, :], old_k)
        assert torch.equal(updated_v[:, :, :prefix_length, :], old_v)


def test_tiny_decoder_cached_decode_matches_full_model_output() -> None:
    torch.manual_seed(17)
    batch_size, prefix_length, vocab_size = 2, 5, 100
    model = TinyDecoderLM(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        d_ff=64,
        n_layers=3,
    )
    all_ids = torch.randint(0, vocab_size, (batch_size, prefix_length + 1))

    full_logits = model(all_ids)
    reference = full_logits[:, -1:, :]
    _, cache = model(all_ids[:, :prefix_length], use_cache=True)
    cached_logits, updated_cache = model(
        all_ids[:, prefix_length:],
        use_cache=True,
        kv_cache=cache,
    )

    assert all(layer_cache[0].shape[2] == prefix_length + 1 for layer_cache in updated_cache)
    assert torch.allclose(cached_logits, reference, atol=1e-6, rtol=1e-5)


def test_tiny_decoder_multi_step_cached_decode_matches_full_prefixes() -> None:
    torch.manual_seed(23)
    batch_size, prefix_length, decode_steps, vocab_size = 2, 4, 3, 100
    model = TinyDecoderLM(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        d_ff=64,
        n_layers=3,
    )
    all_ids = torch.randint(0, vocab_size, (batch_size, prefix_length + decode_steps))
    _, cache = model(all_ids[:, :prefix_length], use_cache=True)

    for step in range(decode_steps):
        prefix_end = prefix_length + step + 1
        cached_logits, cache = model(
            all_ids[:, prefix_end - 1 : prefix_end],
            use_cache=True,
            kv_cache=cache,
        )
        reference = model(all_ids[:, :prefix_end])[:, -1:, :]

        assert all(layer_cache[0].shape[2] == prefix_end for layer_cache in cache)
        assert torch.allclose(cached_logits, reference, atol=1e-6, rtol=1e-5)


def test_tiny_decoder_rejects_ambiguous_or_inconsistent_model_cache() -> None:
    model = TinyDecoderLM(vocab_size=100, d_model=16, n_heads=4, d_ff=64, n_layers=3)
    prefix = torch.randint(0, 100, (2, 5))
    _, cache = model(prefix, use_cache=True)

    with pytest.raises(ValueError, match="requires use_cache"):
        model(torch.randint(0, 100, (2, 1)), kv_cache=cache)

    with pytest.raises(ValueError, match="3 layer caches"):
        model(torch.randint(0, 100, (2, 1)), use_cache=True, kv_cache=cache[:-1])

    with pytest.raises(ValueError, match="exactly one"):
        model(torch.randint(0, 100, (2, 2)), use_cache=True, kv_cache=cache)

    short_k = cache[1][0][:, :, :-1, :]
    short_v = cache[1][1][:, :, :-1, :]
    inconsistent_cache = (cache[0], (short_k, short_v), cache[2])
    with pytest.raises(ValueError, match="same sequence length"):
        model(
            torch.randint(0, 100, (2, 1)),
            use_cache=True,
            kv_cache=inconsistent_cache,
        )


def test_tiny_decoder_cache_is_runtime_only_and_does_not_change_state_dict() -> None:
    model = TinyDecoderLM(vocab_size=100, d_model=16, n_heads=4, d_ff=64, n_layers=3)
    state_before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    _, cache = model(torch.randint(0, 100, (2, 5)), use_cache=True)
    model(torch.randint(0, 100, (2, 1)), use_cache=True, kv_cache=cache)
    state_after = model.state_dict()

    assert state_after.keys() == state_before.keys()
    for name, tensor in state_before.items():
        assert torch.equal(state_after[name], tensor)


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
