"""Request-local inference cache. Handles are consumed by in-place append.

Only [:length] is initialized. Independent requests must prefill fresh storage;
use the returned handle for decode, never fork an existing mutable buffer.
"""

from dataclasses import dataclass

import torch

from .rope import apply_rope


@dataclass(frozen=True)
class PreallocatedKVCache:
    """A logical prefix and its reserved per-layer K/V buffers."""

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int
    capacity: int
    owner: object
    _cursor: list[int]

    @property
    def logical_layers(self):
        return tuple((k[:, :, : self.length], v[:, :, : self.length]) for k, v in self.layers)


def positive(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def forward_preallocated(model, token_ids, cache, capacity):
    """Validate all layers before any append; reference cat path is separate."""
    if torch.is_grad_enabled():
        raise ValueError("preallocated caching requires no_grad or inference_mode")
    if torch.is_autocast_enabled(token_ids.device.type):
        raise ValueError("preallocated caching requires explicit model dtype; disable autocast")
    if token_ids.ndim != 2 or min(token_ids.shape) < 1:
        raise ValueError("expected nonempty token IDs (B, T)")
    batch, count = token_ids.shape
    offset = 0
    if cache is None:
        if not positive(capacity) or capacity < count:
            raise ValueError("cache_capacity must be a positive integer fitting the prompt")
    else:
        if not isinstance(cache, PreallocatedKVCache) or cache.owner is not model:
            raise ValueError("cache must belong to this model and use preallocated mode")
        if not positive(cache.length) or not positive(cache.capacity):
            raise ValueError("invalid cache length or capacity")
        if cache._cursor != [cache.length]:
            raise ValueError("stale cache handle: use the returned cache after each decode")
        if capacity is not None and capacity != cache.capacity:
            raise ValueError("cache_capacity cannot change during decode")
        capacity, offset = cache.capacity, cache.length
        if count != 1:
            raise ValueError("cached decode requires exactly one input token")
        if offset + count > capacity:
            raise ValueError("preallocated cache capacity overflow")
        if len(cache.layers) != len(model.blocks):
            raise ValueError("wrong number of layer caches")
        storages = set()
        for block, pair in zip(model.blocks, cache.layers, strict=True):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("each layer requires a K/V tensor tuple")
            attn = block.attn
            expected = (batch, attn.n_heads, capacity, attn.d_head)
            for tensor in pair:
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
                    raise ValueError(f"cache storage must have shape {expected}")
                if tensor.device != token_ids.device:
                    raise ValueError("cache and input must use the same device")
                if tensor.dtype != attn.Wk.weight.dtype:
                    raise ValueError("cache dtype must match projected K/V dtype")
                if not tensor.is_contiguous() or tensor.storage_offset() != 0:
                    raise ValueError("cache requires contiguous independent buffers")
                key = (str(tensor.device), tensor.untyped_storage().data_ptr())
                if key in storages:
                    raise ValueError("cache buffers must not alias")
                storages.add(key)
    x = model.token_embedding(token_ids)
    layers = []
    for index, block in enumerate(model.blocks):
        attn = block.attn
        normalized = block.norm1(x)
        shape = (batch, count, attn.n_heads, attn.d_head)
        q = apply_rope(attn.Wq(normalized).view(shape).transpose(1, 2), offset=offset)
        k = apply_rope(attn.Wk(normalized).view(shape).transpose(1, 2), offset=offset)
        v = attn.Wv(normalized).view(shape).transpose(1, 2)
        if cache is None:
            buffer_shape = (batch, attn.n_heads, capacity, attn.d_head)
            keys, values = k.new_empty(buffer_shape), v.new_empty(buffer_shape)
        else:
            keys, values = cache.layers[index]
        keys[:, :, offset : offset + count].copy_(k)
        values[:, :, offset : offset + count].copy_(v)
        valid_keys, valid_values = keys[:, :, : offset + count], values[:, :, : offset + count]
        scores = (q @ valid_keys.transpose(-2, -1)) / (attn.d_head**0.5)
        if cache is None:
            mask = torch.triu(
                torch.ones(count, count, dtype=torch.bool, device=x.device), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = (weights @ valid_values).transpose(1, 2).reshape(batch, count, attn.d_model)
        x = x + attn.Wo(output)
        x = x + block.mlp(block.norm2(x))
        layers.append((keys, values))
    logits = model.lm_head(model.final_norm(x))
    cursor = [offset + count] if cache is None else cache._cursor
    cursor[0] = offset + count
    return logits, PreallocatedKVCache(tuple(layers), offset + count, capacity, model, cursor)
