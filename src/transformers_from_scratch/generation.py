"""Minimal autoregressive sampling for the manual decoder-only model."""

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample one token from batch logits shaped ``(B, V)``."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    filtered = logits / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)

    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Select the highest-logit token from batch logits shaped ``(B, V)``."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B, V)")
    return logits.argmax(dim=-1, keepdim=True)


def generate_greedy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
) -> torch.Tensor:
    """Append deterministic argmax tokens using naive uncached decoding."""
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a torch.long tensor shaped (B, T)")
    if max_new_tokens < 0 or context_length < 1:
        raise ValueError("max_new_tokens must be non-negative and context_length positive")

    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(generated[:, -context_length:])
                next_token = greedy_next_token(logits[:, -1, :])
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)
    return generated


def generate_greedy_cached(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
) -> torch.Tensor:
    """Append deterministic argmax tokens using an untruncated KV cache.

    The complete requested sequence, including generated tokens, must fit within
    ``context_length`` because cache eviction is not implemented.
    """
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a torch.long tensor shaped (B, T)")
    if max_new_tokens < 0 or context_length < 1:
        raise ValueError("max_new_tokens must be non-negative and context_length positive")
    if input_ids.shape[1] + max_new_tokens > context_length:
        raise ValueError("prompt and generated tokens must fit within context_length")

    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    try:
        with torch.inference_mode():
            if max_new_tokens == 0:
                return generated

            logits, cache = model(input_ids, use_cache=True)
            next_token = greedy_next_token(logits[:, -1, :])
            generated = torch.cat((generated, next_token), dim=1)

            for _ in range(max_new_tokens - 1):
                logits, cache = model(next_token, use_cache=True, kv_cache=cache)
                next_token = greedy_next_token(logits[:, -1, :])
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)
    return generated


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Append sampled tokens while feeding at most ``context_length`` IDs to the model."""
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a torch.long tensor shaped (B, T)")
    if max_new_tokens < 0 or context_length < 1:
        raise ValueError("max_new_tokens must be non-negative and context_length positive")

    was_training = model.training
    model.eval()
    generated = input_ids.clone()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(generated[:, -context_length:])
                next_token = sample_next_token(
                    logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
                )
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)
    return generated
