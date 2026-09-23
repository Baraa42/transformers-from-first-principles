"""Parsing helpers for the controlled precision benchmark."""

import re
import statistics
from dataclasses import dataclass

_FIELD_PATTERN = re.compile(r"([a-z_]+)=([^\s]+)")


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Metrics extracted from one completed precision-training log."""

    precision: str
    training_wall_time_sec: float
    median_tokens_per_sec: float
    final_train_loss: float
    final_val_loss: float
    final_val_perplexity: float
    amp_overflow_count: int
    final_grad_scale: float | None


def parse_training_log(log: str, precision: str) -> BenchmarkMetrics:
    """Extract final losses, synchronized timing, and AMP diagnostics."""
    throughputs: list[float] = []
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_val_perplexity: float | None = None
    wall_time: float | None = None
    overflow_count: int | None = None
    final_grad_scale: float | None = None
    saw_scale = False

    for line in log.splitlines():
        fields = dict(_FIELD_PATTERN.findall(line))
        if "tokens_per_sec" in fields:
            throughputs.append(float(fields["tokens_per_sec"]))
        if "train_loss" in fields:
            final_train_loss = float(fields["train_loss"])
        if "val_loss" in fields:
            final_val_loss = float(fields["val_loss"])
            final_val_perplexity = float(fields["val_perplexity"])
        if "training_wall_time_sec" in fields:
            wall_time = float(fields["training_wall_time_sec"])
        if "amp_overflow_count" in fields:
            overflow_count = int(fields["amp_overflow_count"])
        if "final_grad_scale" in fields:
            saw_scale = True
            value = fields["final_grad_scale"]
            final_grad_scale = None if value == "none" else float(value)

    missing = [
        name
        for name, value in (
            ("tokens_per_sec", throughputs),
            ("train_loss", final_train_loss),
            ("val_loss", final_val_loss),
            ("val_perplexity", final_val_perplexity),
            ("training_wall_time_sec", wall_time),
            ("amp_overflow_count", overflow_count),
            ("final_grad_scale", saw_scale),
        )
        if value is None or value is False or value == []
    ]
    if missing:
        raise ValueError(f"Training log is missing required metrics: {', '.join(missing)}")

    return BenchmarkMetrics(
        precision=precision,
        training_wall_time_sec=wall_time,
        median_tokens_per_sec=statistics.median(throughputs),
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_val_perplexity=final_val_perplexity,
        amp_overflow_count=overflow_count,
        final_grad_scale=final_grad_scale,
    )


def relative_throughput(results: list[BenchmarkMetrics]) -> dict[str, float]:
    """Return per-mode median throughput relative to FP32."""
    fp32 = next((result for result in results if result.precision == "fp32"), None)
    if fp32 is None:
        return {}
    if fp32.median_tokens_per_sec <= 0:
        raise ValueError("FP32 median throughput must be positive")
    return {
        result.precision: result.median_tokens_per_sec / fp32.median_tokens_per_sec
        for result in results
    }
