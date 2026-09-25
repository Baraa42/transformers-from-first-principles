"""Measurement helpers for deterministic uncached autoregressive inference."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median

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


@dataclass(frozen=True)
class DecodeWindowMetrics:
    """Mean latency at the beginning and end of one decode run."""

    first_mean_ms: float
    last_mean_ms: float
    growth_ratio: float


@dataclass(frozen=True)
class InferenceBenchmarkSummary:
    """Median measurements across repeated identical inference workloads."""

    prompt_length: int
    generated_tokens: int
    repetitions: int
    window_size: int
    median_prefill_ms: float
    median_decode_total_ms: float
    median_mean_decode_ms: float
    median_tokens_per_sec: float
    median_total_latency_ms: float
    median_first_window_ms: float
    median_last_window_ms: float
    median_growth_ratio: float


@dataclass(frozen=True)
class RepeatedInferenceBenchmark:
    """Individual results and outputs from repeated deterministic workloads."""

    results: tuple[InferenceBenchmarkResult, ...]
    generated_outputs: tuple[torch.Tensor, ...]


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


def decode_window_metrics(
    decode_steps: Sequence[DecodeStepTiming],
    *,
    window_size: int = 8,
) -> DecodeWindowMetrics:
    """Compare the first and last fixed-size windows of one decode run."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(decode_steps) < window_size:
        raise ValueError("decode_steps must contain at least window_size entries")
    if any(step.latency_ms < 0 for step in decode_steps):
        raise ValueError("decode latencies must be non-negative")

    first_mean_ms = float(mean(step.latency_ms for step in decode_steps[:window_size]))
    last_mean_ms = float(mean(step.latency_ms for step in decode_steps[-window_size:]))
    if first_mean_ms == 0:
        raise ValueError("first decode window mean must be non-zero")
    return DecodeWindowMetrics(
        first_mean_ms=first_mean_ms,
        last_mean_ms=last_mean_ms,
        growth_ratio=last_mean_ms / first_mean_ms,
    )


def summarize_inference_results(
    results: Sequence[InferenceBenchmarkResult],
    *,
    window_size: int = 8,
) -> InferenceBenchmarkSummary:
    """Aggregate repeated identical workloads using medians."""
    if not results:
        raise ValueError("results must not be empty")

    prompt_length = results[0].prompt_length
    generated_tokens = results[0].generated_tokens
    if any(
        result.prompt_length != prompt_length or result.generated_tokens != generated_tokens
        for result in results
    ):
        raise ValueError("all results must describe the same workload")

    windows = [
        decode_window_metrics(result.decode_steps, window_size=window_size) for result in results
    ]
    return InferenceBenchmarkSummary(
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        repetitions=len(results),
        window_size=window_size,
        median_prefill_ms=float(median(result.prefill_ms for result in results)),
        median_decode_total_ms=float(median(result.decode_total_ms for result in results)),
        median_mean_decode_ms=float(median(result.mean_decode_ms for result in results)),
        median_tokens_per_sec=float(median(result.tokens_per_sec for result in results)),
        median_total_latency_ms=float(median(result.total_latency_ms for result in results)),
        median_first_window_ms=float(median(window.first_mean_ms for window in windows)),
        median_last_window_ms=float(median(window.last_mean_ms for window in windows)),
        median_growth_ratio=float(median(window.growth_ratio for window in windows)),
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


def benchmark_uncached_repetitions(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    generated_tokens: int,
    context_length: int,
    device: torch.device,
    repetitions: int = 5,
    warmup_forwards: int = 2,
) -> RepeatedInferenceBenchmark:
    """Run identical uncached workloads, warming up only before the first run."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if warmup_forwards < 0:
        raise ValueError("warmup_forwards must be non-negative")

    results: list[InferenceBenchmarkResult] = []
    generated_outputs: list[torch.Tensor] = []
    for repetition in range(repetitions):
        result, generated = benchmark_uncached_greedy(
            model,
            prompt_ids,
            generated_tokens=generated_tokens,
            context_length=context_length,
            device=device,
            warmup_forwards=warmup_forwards if repetition == 0 else 0,
        )
        if generated_outputs and not torch.equal(generated_outputs[0], generated):
            raise RuntimeError("greedy output changed across identical benchmark repetitions")
        results.append(result)
        generated_outputs.append(generated)

    return RepeatedInferenceBenchmark(
        results=tuple(results),
        generated_outputs=tuple(generated_outputs),
    )
