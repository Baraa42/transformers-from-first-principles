"""Measurement helpers for deterministic uncached autoregressive inference."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from transformers_from_scratch.generation import greedy_next_token
from transformers_from_scratch.training import synchronize_device


@dataclass(frozen=True)
class DecodeStepTiming:
    """Latency for one uncached forward at a known prefix length."""

    prefix_length: int
    latency_ms: float


@dataclass(frozen=True)
class InferenceBenchmarkResult:
    """Prefill and uncached-decode measurements for one prompt workload."""

    prompt_length: int
    generated_tokens: int
    prefill_ms: float
    decode_total_ms: float
    mean_decode_ms: float
    tokens_per_sec: float
    total_latency_ms: float
    decode_steps: tuple[DecodeStepTiming, ...]


def construct_exact_prompt(
    source_token_ids: Sequence[int],
    prompt_length: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic exact-length ``(1, T)`` prompt of valid token IDs."""
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if len(source_token_ids) < prompt_length:
        raise ValueError("source token sequence is shorter than prompt_length")

    selected = list(source_token_ids[:prompt_length])
    if any(token_id < 0 or token_id >= vocab_size for token_id in selected):
        raise ValueError("source token IDs must be within the model vocabulary")
    return torch.tensor([selected], dtype=torch.long, device=device)


def calculate_inference_metrics(
    *,
    prompt_length: int,
    generated_tokens: int,
    prefill_ms: float,
    decode_steps: Sequence[DecodeStepTiming],
) -> InferenceBenchmarkResult:
    """Calculate forward-only latency and throughput metrics for one workload."""
    expected_decode_forwards = generated_tokens - 1
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    if generated_tokens < 1:
        raise ValueError("generated_tokens must be positive")
    if prefill_ms < 0:
        raise ValueError("prefill_ms must be non-negative")
    if len(decode_steps) != expected_decode_forwards:
        raise ValueError("decode step count must equal generated_tokens - 1")
    if any(step.latency_ms < 0 for step in decode_steps):
        raise ValueError("decode latencies must be non-negative")

    decode_total_ms = sum(step.latency_ms for step in decode_steps)
    if expected_decode_forwards:
        mean_decode_ms = decode_total_ms / expected_decode_forwards
        tokens_per_sec = (
            expected_decode_forwards / (decode_total_ms / 1000.0)
            if decode_total_ms > 0
            else float("inf")
        )
    else:
        mean_decode_ms = 0.0
        tokens_per_sec = 0.0

    return InferenceBenchmarkResult(
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_ms=prefill_ms,
        decode_total_ms=decode_total_ms,
        mean_decode_ms=mean_decode_ms,
        tokens_per_sec=tokens_per_sec,
        total_latency_ms=prefill_ms + decode_total_ms,
        decode_steps=tuple(decode_steps),
    )


def _timed_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Time one synchronized model forward and return milliseconds."""
    synchronize_device(device)
    started_at = time.perf_counter()
    logits = model(input_ids)
    synchronize_device(device)
    return logits, (time.perf_counter() - started_at) * 1000.0


def benchmark_uncached_greedy(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    generated_tokens: int,
    context_length: int,
    device: torch.device,
    warmup_forwards: int = 2,
) -> tuple[InferenceBenchmarkResult, torch.Tensor]:
    """Measure prefill plus naive full-prefix forwards for deterministic decoding.

    The prefill forward produces generated token number one. The decode phase then
    performs ``generated_tokens - 1`` full-prefix forwards for the remaining tokens.
    Token selection and concatenation are intentionally outside the forward timers.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a torch.long tensor shaped (1, T)")
    prompt_length = prompt_ids.shape[1]
    if prompt_length < 1:
        raise ValueError("prompt length must be positive")
    if generated_tokens < 1:
        raise ValueError("generated_tokens must be positive")
    required_context_length = prompt_length + generated_tokens - 1
    if context_length < required_context_length:
        raise ValueError("context_length must fit the largest uncached prefix forward")
    if warmup_forwards < 0:
        raise ValueError("warmup_forwards must be non-negative")

    was_training = model.training
    model.eval()
    generated = prompt_ids.clone()
    decode_steps: list[DecodeStepTiming] = []
    try:
        with torch.inference_mode():
            for _ in range(warmup_forwards):
                model(prompt_ids)
            synchronize_device(device)

            # Prefill is exactly one full-prompt forward and yields new token number one.
            logits, prefill_ms = _timed_forward(model, prompt_ids, device)
            next_token = greedy_next_token(logits[:, -1, :])
            generated = torch.cat((generated, next_token), dim=1)

            # Every later token reruns the model over the complete growing prefix.
            for _ in range(generated_tokens - 1):
                prefix_length = generated.shape[1]
                logits, latency_ms = _timed_forward(model, generated[:, -context_length:], device)
                decode_steps.append(
                    DecodeStepTiming(prefix_length=prefix_length, latency_ms=latency_ms)
                )
                next_token = greedy_next_token(logits[:, -1, :])
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)

    result = calculate_inference_metrics(
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_ms=prefill_ms,
        decode_steps=decode_steps,
    )
    return result, generated
