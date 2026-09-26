"""A small, explicit decoder-only language model."""

import torch
from torch import nn

from .attention import KVCache
from .layers import DecoderBlock

ModelKVCache = tuple[KVCache, ...]


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

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        use_cache: bool = False,
        kv_cache: ModelKVCache | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelKVCache]:
        """Map IDs ``(B, n)`` to raw vocabulary logits ``(B, n, V)``.

        Embedding lookup selects learned rows from a ``(V, d_model)`` matrix,
        then blocks/final norm preserve ``(B, n, d_model)``. No softmax is
        applied because loss and sampling are deliberately deferred.
        """
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token IDs (B, n), got {tuple(token_ids.shape)}")
        if kv_cache is not None and not use_cache:
            raise ValueError("kv_cache requires use_cache=True")

        if kv_cache is not None:
            if len(kv_cache) != len(self.blocks):
                raise ValueError(
                    f"kv_cache must contain {len(self.blocks)} layer caches, got {len(kv_cache)}"
                )
            if token_ids.shape[1] != 1:
                raise ValueError("cached decode requires exactly one input token")

            cache_lengths = {
                layer_cache[0].shape[2]
                for layer_cache in kv_cache
                if isinstance(layer_cache, tuple)
                and len(layer_cache) == 2
                and isinstance(layer_cache[0], torch.Tensor)
                and layer_cache[0].ndim >= 3
            }
            if len(cache_lengths) > 1:
                raise ValueError("all layer caches must have the same sequence length")

        x = self.token_embedding(token_ids)

        if not use_cache:
            for block in self.blocks:
                x = block(x)
            return self.lm_head(self.final_norm(x))

        layer_caches: tuple[KVCache | None, ...]
        if kv_cache is None:
            layer_caches = (None,) * len(self.blocks)
        else:
            layer_caches = kv_cache

        new_caches: list[KVCache] = []
        for block, layer_cache in zip(self.blocks, layer_caches, strict=True):
            x, new_cache = block(x, use_cache=True, kv_cache=layer_cache)
            new_caches.append(new_cache)

        logits = self.lm_head(self.final_norm(x))
        return logits, tuple(new_caches)
