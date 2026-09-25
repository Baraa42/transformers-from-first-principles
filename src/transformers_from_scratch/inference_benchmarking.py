"""Measurement helpers for deterministic uncached autoregressive inference."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import torch

from transformers_from_scratch.generation import greedy_next_token
from transformers_from_scratch.training import synchronize_device


@dataclass(frozen=True)
class DecodeWindowMetrics:
    """Per-token latency derived from synchronized decode blocks."""

    first_mean_ms: float
    last_mean_ms: float
    growth_ratio: float


@dataclass(frozen=True)
class InferenceBenchmarkResult:
    """Prefill and end-to-end uncached-decode measurements for one run."""

    prompt_length: int
    generated_tokens: int
    prefill_ms: float
    decode_total_ms: float
    mean_decode_ms: float
    tokens_per_sec: float
    total_latency_ms: float
    first_window_ms: float
    last_window_ms: float
    growth_ratio: float
    window_size: int


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


def decode_window_metrics(
    *,
    first_block_total_ms: float,
    last_block_total_ms: float,
    window_size: int = 8,
) -> DecodeWindowMetrics:
    """Convert synchronized first/last block totals into per-token latency."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if first_block_total_ms <= 0:
        raise ValueError("first block timing must be positive")
    if last_block_total_ms < 0:
        raise ValueError("last block timing must be non-negative")

    first_mean_ms = first_block_total_ms / window_size
    last_mean_ms = last_block_total_ms / window_size
    return DecodeWindowMetrics(
        first_mean_ms=first_mean_ms,
        last_mean_ms=last_mean_ms,
        growth_ratio=last_mean_ms / first_mean_ms,
    )


def calculate_inference_metrics(
    *,
    prompt_length: int,
    generated_tokens: int,
    prefill_ms: float,
    decode_total_ms: float,
    first_block_total_ms: float,
    last_block_total_ms: float,
    window_size: int = 8,
) -> InferenceBenchmarkResult:
    """Calculate authoritative block-level decode metrics for one workload."""
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    if generated_tokens < 1:
        raise ValueError("generated_tokens must be positive")
    if prefill_ms < 0:
        raise ValueError("prefill_ms must be non-negative")
    if decode_total_ms < 0:
        raise ValueError("decode_total_ms must be non-negative")

    decode_forwards = generated_tokens - 1
    if decode_forwards < 2 * window_size:
        raise ValueError("decode forwards must fit non-overlapping first and last windows")
    windows = decode_window_metrics(
        first_block_total_ms=first_block_total_ms,
        last_block_total_ms=last_block_total_ms,
        window_size=window_size,
    )
    mean_decode_ms = decode_total_ms / decode_forwards
    tokens_per_sec = (
        decode_forwards / (decode_total_ms / 1000.0) if decode_total_ms > 0 else float("inf")
    )

    return InferenceBenchmarkResult(
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_ms=prefill_ms,
        decode_total_ms=decode_total_ms,
        mean_decode_ms=mean_decode_ms,
        tokens_per_sec=tokens_per_sec,
        total_latency_ms=prefill_ms + decode_total_ms,
        first_window_ms=windows.first_mean_ms,
        last_window_ms=windows.last_mean_ms,
        growth_ratio=windows.growth_ratio,
        window_size=window_size,
    )


def summarize_inference_results(
    results: Sequence[InferenceBenchmarkResult],
) -> InferenceBenchmarkSummary:
    """Aggregate repeated identical workloads using medians."""
    if not results:
        raise ValueError("results must not be empty")

    first = results[0]
    if any(
        result.prompt_length != first.prompt_length
        or result.generated_tokens != first.generated_tokens
        or result.window_size != first.window_size
        for result in results
    ):
        raise ValueError("all results must describe the same workload")

    return InferenceBenchmarkSummary(
        prompt_length=first.prompt_length,
        generated_tokens=first.generated_tokens,
        repetitions=len(results),
        window_size=first.window_size,
        median_prefill_ms=float(median(result.prefill_ms for result in results)),
        median_decode_total_ms=float(median(result.decode_total_ms for result in results)),
        median_mean_decode_ms=float(median(result.mean_decode_ms for result in results)),
        median_tokens_per_sec=float(median(result.tokens_per_sec for result in results)),
        median_total_latency_ms=float(median(result.total_latency_ms for result in results)),
        median_first_window_ms=float(median(result.first_window_ms for result in results)),
        median_last_window_ms=float(median(result.last_window_ms for result in results)),
        median_growth_ratio=float(median(result.growth_ratio for result in results)),
    )


def _timed_prefill(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Time one synchronized full-prompt model forward."""
    synchronize_device(device)
    started_at = time.perf_counter()
    logits = model(prompt_ids)
    synchronize_device(device)
    return logits, (time.perf_counter() - started_at) * 1000.0


def _decode_iterations(
    model: torch.nn.Module,
    generated: torch.Tensor,
    *,
    iterations: int,
    context_length: int,
) -> torch.Tensor:
    """Run naive uncached decode iterations without internal synchronization."""
    for _ in range(iterations):
        logits = model(generated[:, -context_length:])
        next_token = greedy_next_token(logits[:, -1, :])
        generated = torch.cat((generated, next_token), dim=1)
    return generated


def _timed_decode_block(
    model: torch.nn.Module,
    generated: torch.Tensor,
    *,
    iterations: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Time one decode block with a single synchronization at each boundary."""
    synchronize_device(device)
    started_at = time.perf_counter()
    generated = _decode_iterations(
        model,
        generated,
        iterations=iterations,
        context_length=context_length,
    )
    synchronize_device(device)
    return generated, (time.perf_counter() - started_at) * 1000.0


def _start_from_prompt(model: torch.nn.Module, prompt_ids: torch.Tensor) -> torch.Tensor:
    """Run unmeasured prefill and append the first greedy token."""
    logits = model(prompt_ids)
    first_token = greedy_next_token(logits[:, -1, :])
    return torch.cat((prompt_ids.clone(), first_token), dim=1)


def _measure_decode_windows(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    decode_forwards: int,
    window_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    """Measure first and last windows during a separate deterministic execution."""
    middle_iterations = decode_forwards - 2 * window_size
    generated = _start_from_prompt(model, prompt_ids)
    generated, first_total_ms = _timed_decode_block(
        model,
        generated,
        iterations=window_size,
        context_length=context_length,
        device=device,
    )
    generated = _decode_iterations(
        model,
        generated,
        iterations=middle_iterations,
        context_length=context_length,
    )
    generated, last_total_ms = _timed_decode_block(
        model,
        generated,
        iterations=window_size,
        context_length=context_length,
        device=device,
    )
    return first_total_ms, last_total_ms, generated


def benchmark_uncached_greedy(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    generated_tokens: int,
    context_length: int,
    device: torch.device,
    warmup_forwards: int = 2,
    window_size: int = 8,
) -> tuple[InferenceBenchmarkResult, torch.Tensor]:
    """Measure synchronized prefill, decode-loop total, and window blocks.

    Prefill is a model-forward-only measurement and produces generated token one.
    The authoritative decode total contains every subsequent uncached model forward,
    greedy argmax, and token concatenation in one synchronized block.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a torch.long tensor shaped (1, T)")
    prompt_length = prompt_ids.shape[1]
    if prompt_length < 1:
        raise ValueError("prompt length must be positive")
    if generated_tokens < 1:
        raise ValueError("generated_tokens must be positive")
    decode_forwards = generated_tokens - 1
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if decode_forwards < 2 * window_size:
        raise ValueError("decode forwards must fit non-overlapping first and last windows")
    required_context_length = prompt_length + decode_forwards
    if context_length < required_context_length:
        raise ValueError("context_length must fit the largest uncached prefix forward")
    if warmup_forwards < 0:
        raise ValueError("warmup_forwards must be non-negative")

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup_forwards):
                model(prompt_ids)
            synchronize_device(device)

            logits, prefill_ms = _timed_prefill(model, prompt_ids, device)
            first_token = greedy_next_token(logits[:, -1, :])
            generated = torch.cat((prompt_ids.clone(), first_token), dim=1)
            generated, decode_total_ms = _timed_decode_block(
                model,
                generated,
                iterations=decode_forwards,
                context_length=context_length,
                device=device,
            )

            first_block_ms, last_block_ms, window_generated = _measure_decode_windows(
                model,
                prompt_ids,
                decode_forwards=decode_forwards,
                window_size=window_size,
                context_length=context_length,
                device=device,
            )
            if not torch.equal(generated, window_generated):
                raise RuntimeError("greedy output changed between total and window measurements")
    finally:
        model.train(was_training)

    result = calculate_inference_metrics(
        prompt_length=prompt_length,
        generated_tokens=generated_tokens,
        prefill_ms=prefill_ms,
        decode_total_ms=decode_total_ms,
        first_block_total_ms=first_block_ms,
        last_block_total_ms=last_block_ms,
        window_size=window_size,
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
    window_size: int = 8,
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
            window_size=window_size,
        )
        if generated_outputs and not torch.equal(generated_outputs[0], generated):
            raise RuntimeError("greedy output changed across identical benchmark repetitions")
        results.append(result)
        generated_outputs.append(generated)

    return RepeatedInferenceBenchmark(
        results=tuple(results),
        generated_outputs=tuple(generated_outputs),
    )
