"""Every-step equivalence, RoPE offsets, ownership, validation, and fresh requests."""

from dataclasses import replace

import pytest
import torch

from transformers_from_scratch.generation import generate_cached, generate_greedy_cached
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.rope import apply_rope

DEVICES = [
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable"),
    ),
]


def tiny(device="cpu"):
    torch.manual_seed(79)
    return (
        TinyDecoderLM(vocab_size=29, d_model=16, n_heads=4, d_ff=32, n_layers=2).to(device).eval()
    )


def prefill(model, prompt, capacity=12):
    return model(prompt, use_cache=True, cache_mode="preallocated", cache_capacity=capacity)


def decode(model, token, cache, **kwargs):
    return model(token, use_cache=True, cache_mode="preallocated", kv_cache=cache, **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch,p,n", [(1, 1, 1), (1, 3, 5), (4, 8, 7)])
@torch.inference_mode()
def test_every_step_equivalence_rope_and_storage(device, batch, p, n, monkeypatch):
    model = tiny(device)
    prompt = torch.randint(29, (batch, p), device=device)
    reference, cat = model(prompt, use_cache=True)
    logits, cache = prefill(model, prompt, p + n)
    pointers = [t.data_ptr() for pair in cache.layers for t in pair]
    full = prompt
    for step in range(n + 1):
        assert cache.length == p + step
        assert cache.capacity == p + n
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-6)
        # Uncached forward independently verifies absolute RoPE positions.
        torch.testing.assert_close(logits[:, -1], model(full)[:, -1], rtol=1e-5, atol=1e-6)
        assert torch.equal(logits[:, -1].argmax(-1), reference[:, -1].argmax(-1))
        for a, e in zip(cache.logical_layers, cat, strict=True):
            for actual, expected in zip(a, e, strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert pointers == [t.data_ptr() for pair in cache.layers for t in pair]
        if step == n:
            break
        token = reference[:, -1].argmax(-1, keepdim=True)
        # Explicit first-layer projected key at its absolute position.
        block = model.blocks[0]
        x = block.norm1(model.token_embedding(token))
        raw = block.attn.Wk(x).view(batch, 1, 4, 4).transpose(1, 2)
        expected_key = apply_rope(raw, offset=cache.length)
        previous = cache
        reference, cat = model(token, use_cache=True, kv_cache=cat)
        original_cat = torch.cat

        def no_kv_cat(tensors, dim=0, **kwargs):
            assert not (dim == 2 and tensors[0].ndim == 4)
            return original_cat(tensors, dim=dim, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "cat", no_kv_cat)
            logits, cache = decode(model, token, cache)
        torch.testing.assert_close(
            cache.layers[0][0][:, :, cache.length - 1 : cache.length],
            expected_key,
            rtol=1e-5,
            atol=1e-6,
        )
        with pytest.raises(ValueError, match="stale"):
            decode(model, token, previous)
        full = torch.cat((full, token), dim=1)


@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_no_stale_tail_or_request_aliasing(device):
    model = tiny(device)
    a = torch.tensor([[1, 2, 3]], device=device)
    b = torch.tensor([[4, 5]], device=device)
    logits, first = prefill(model, a)
    for k, v in first.layers:
        k[:, :, first.length :].fill_(float("nan"))
        v[:, :, first.length :].fill_(float("nan"))
    _, second = prefill(model, b)
    first_ptrs = {t.data_ptr() for pair in first.layers for t in pair}
    assert first_ptrs.isdisjoint(t.data_ptr() for pair in second.layers for t in pair)
    snapshot = [(k.clone(), v.clone()) for k, v in second.layers]
    token = logits[:, -1].argmax(-1, keepdim=True)
    decoded, _ = decode(model, token, first)
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(
        decoded[:, -1], model(torch.cat((a, token), 1))[:, -1], rtol=1e-5, atol=1e-6
    )
    for actual, expected in zip(second.layers, snapshot, strict=True):
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x, y, equal_nan=True)
    assert not any("cache" in key for key in model.state_dict())


@torch.inference_mode()
def test_validation_before_any_mutation():
    model = tiny()
    prompt = torch.tensor([[1, 2, 3]])
    token = prompt[:, :1]
    _, cache = prefill(model, prompt, 5)
    original = [(k.clone(), v.clone()) for k, v in cache.layers]
    k, v = cache.layers[-1]
    invalid = [
        replace(cache, length=0),
        replace(cache, capacity=True),
        replace(cache, layers=cache.layers[:1]),
        replace(cache, layers=(cache.layers[0], (k.double(), v))),
        replace(cache, layers=(cache.layers[0], (k[:, :, :4], v))),
        replace(cache, layers=(cache.layers[0], (k.to("meta"), v))),
        replace(cache, layers=(cache.layers[0], cache.layers[0])),
        replace(cache, layers=(cache.layers[0], (v, v))),
        replace(cache, layers=(cache.layers[0], (k.transpose(1, 3), v))),
    ]
    for bad in invalid:
        with pytest.raises(ValueError):
            decode(model, token, bad)
    for input_ids, kwargs in [
        (token.repeat(2, 1), {}),
        (prompt, {}),
        (token, {"cache_capacity": 6}),
    ]:
        with pytest.raises(ValueError):
            decode(model, input_ids, cache, **kwargs)
    with pytest.raises(ValueError, match="belong"):
        decode(tiny(), token, cache)
    for actual, expected in zip(cache.layers, original, strict=True):
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x, y, equal_nan=True)
    _, cache = decode(model, token, cache)
    _, cache = decode(model, token, cache)
    with pytest.raises(ValueError, match="overflow"):
        decode(model, token, cache)
    for capacity in (None, 0, -1, True, 2, 4.5):
        with pytest.raises(ValueError, match="capacity"):
            prefill(model, prompt, capacity)
    with pytest.raises(ValueError):
        model(prompt, cache_mode="preallocated", cache_capacity=5)
    with pytest.raises(ValueError):
        model(token, use_cache=True, kv_cache=cache)
    with pytest.raises(ValueError):
        model(prompt, use_cache=True, cache_mode="unknown")


@pytest.mark.parametrize("device", DEVICES)
def test_generation_api_and_training_unchanged(device):
    model = tiny(device).train()
    prompt = torch.tensor([[1, 2, 3]], device=device)
    kwargs = dict(max_new_tokens=6, context_length=8)
    cat = generate_greedy_cached(model, prompt, **kwargs)
    pre = generate_greedy_cached(model, prompt, cache_mode="preallocated", **kwargs)
    assert torch.equal(cat, pre)
    assert model.training
    for mode in ("cat", "preallocated"):
        assert torch.equal(
            generate_greedy_cached(
                model, prompt, max_new_tokens=0, context_length=1, cache_mode=mode
            ),
            prompt,
        )
        sampled = generate_cached(
            model, prompt, do_sample=False, eos_token_id=cat[0, 3].item(), cache_mode=mode, **kwargs
        )
        assert sampled.shape[1] == 4
    with pytest.raises(ValueError, match="no_grad"):
        prefill(model, prompt)
    loss = model(prompt).sum()
    loss.backward()
    assert model.blocks[0].attn.Wk.weight.grad is not None


@torch.inference_mode()
def test_kv_storage_allocated_only_during_prefill(monkeypatch):
    model = tiny()
    original = torch.Tensor.new_empty
    allocations = []

    def observe(tensor, shape, **kwargs):
        allocations.append(tuple(shape))
        return original(tensor, shape, **kwargs)

    monkeypatch.setattr(torch.Tensor, "new_empty", observe)
    _, cache = prefill(model, torch.tensor([[1, 2, 3]]), 7)
    assert allocations == [(1, 4, 7, 4)] * 4
    allocations.clear()
    for _ in range(4):
        _, cache = decode(model, torch.tensor([[4]]), cache)
    assert not allocations
