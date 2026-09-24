"""Opt-in synchronized timing helpers for training diagnostics."""

import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from transformers_from_scratch.training import synchronize_device

TRAIN_COMPONENTS = (
    "zero_grad",
    "batch_fetch",
    "host_to_device",
    "forward_loss",
    "backward",
    "grad_processing",
    "optimizer_step",
)


@dataclass(frozen=True)
class TimingStatistics:
    """Distribution statistics for one train-step component."""

    median_ms: float
    p95_ms: float
    pct_measured_step: float


@dataclass
class TimingProfile:
    """Collect successful-step component samples and separate event timings."""

    enabled: bool
    warmup_steps: int
    successful_steps: int = 0
    component_samples: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in TRAIN_COMPONENTS}
    )
    step_samples: list[float] = field(default_factory=list)
    validation_samples: list[float] = field(default_factory=list)
    checkpoint_samples: list[float] = field(default_factory=list)

    def record_successful_step(
        self, component_times: MutableMapping[str, float], step_time: float
    ) -> None:
        """Record one optimizer step after excluding initial successful warmup steps."""
        if not self.enabled:
            return
        self.successful_steps += 1
        if self.successful_steps <= self.warmup_steps:
            return
        for name in TRAIN_COMPONENTS:
            self.component_samples[name].append(component_times[name])
        self.step_samples.append(step_time)

    def record_event(self, name: str, elapsed: float | None) -> None:
        """Record validation/checkpoint invocation time when profiling is active."""
        if not self.enabled or elapsed is None:
            return
        if name == "validation":
            self.validation_samples.append(elapsed)
        elif name == "checkpoint":
            self.checkpoint_samples.append(elapsed)
        else:
            raise ValueError(f"Unknown profiling event: {name}")


def validate_profile_config(enabled: object, warmup_steps: object) -> tuple[bool, int]:
    """Validate opt-in timing settings without accepting bool as an integer."""
    if not isinstance(enabled, bool):
        raise ValueError("profile_timing must be true or false")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("profile_warmup_steps must be an integer >= 0")
    return enabled, warmup_steps


@contextmanager
def timed_section(
    name: str,
    timings: MutableMapping[str, float] | None,
    device: torch.device,
    *,
    enabled: bool,
) -> Iterator[None]:
    """Synchronize around a section and accumulate elapsed seconds when enabled."""
    if not enabled:
        yield
        return
    if timings is None:
        raise ValueError("Enabled timing requires a timings mapping")
    synchronize_device(device)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        synchronize_device(device)
        elapsed = time.perf_counter() - started_at
        timings[name] = timings.get(name, 0.0) + elapsed


def start_timer(device: torch.device, *, enabled: bool) -> float | None:
    """Start an outer synchronized timer, or return no timer when disabled."""
    if not enabled:
        return None
    synchronize_device(device)
    return time.perf_counter()


def stop_timer(started_at: float | None, device: torch.device, *, enabled: bool) -> float | None:
    """Stop an outer synchronized timer, or return no sample when disabled."""
    if not enabled:
        return None
    if started_at is None:
        raise ValueError("Enabled timing requires a start time")
    synchronize_device(device)
    return time.perf_counter() - started_at


def component_statistics(profile: TimingProfile) -> dict[str, TimingStatistics]:
    """Compute median, p95, and median-share statistics for each component."""
    if not profile.step_samples:
        return {}
    medians = {
        name: float(np.median(samples)) for name, samples in profile.component_samples.items()
    }
    measured_total = sum(medians.values())
    return {
        name: TimingStatistics(
            median_ms=median * 1000.0,
            p95_ms=float(np.percentile(profile.component_samples[name], 95)) * 1000.0,
            pct_measured_step=(median / measured_total * 100.0 if measured_total else 0.0),
        )
        for name, median in medians.items()
    }


def _event_statistics(samples: list[float]) -> tuple[int, float, float]:
    if not samples:
        return 0, 0.0, 0.0
    return len(samples), sum(samples), float(np.median(samples))


def print_timing_summary(profile: TimingProfile) -> None:
    """Print the concise Stage 5.1 timing report."""
    if not profile.enabled:
        return

    print("timing_profile=true")
    print(f"successful_steps={profile.successful_steps}")
    print(f"profiled_steps={len(profile.step_samples)}")
    print(f"profile_warmup_steps={profile.warmup_steps}")
    print()
    print("component        median_ms   p95_ms   pct_measured_step")
    for name, stats in component_statistics(profile).items():
        print(
            f"{name:<16} {stats.median_ms:>9.3f} "
            f"{stats.p95_ms:>8.3f} {stats.pct_measured_step:>19.2f}"
        )

    if profile.step_samples:
        print()
        print(f"step_time_median_ms={float(np.median(profile.step_samples)) * 1000.0:.3f}")
        print(f"step_time_p95_ms={float(np.percentile(profile.step_samples, 95)) * 1000.0:.3f}")

    validation_count, validation_total, validation_median = _event_statistics(
        profile.validation_samples
    )
    checkpoint_count, checkpoint_total, checkpoint_median = _event_statistics(
        profile.checkpoint_samples
    )
    print()
    print(f"validation_count={validation_count}")
    print(f"validation_total_sec={validation_total:.3f}")
    print(f"validation_median_sec={validation_median:.3f}")
    print()
    print(f"checkpoint_count={checkpoint_count}")
    print(f"checkpoint_total_sec={checkpoint_total:.3f}")
    print(f"checkpoint_median_sec={checkpoint_median:.3f}")
