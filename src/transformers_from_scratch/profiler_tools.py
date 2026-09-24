"""Small validation and path helpers for operator profiling."""

from pathlib import Path

import torch


def validate_profiler_steps(warmup_steps: object, profile_steps: object) -> tuple[int, int]:
    """Validate short profiler phase lengths."""
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be an integer >= 0")
    if isinstance(profile_steps, bool) or not isinstance(profile_steps, int) or profile_steps < 1:
        raise ValueError("profile_steps must be an integer >= 1")
    return warmup_steps, profile_steps


def profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    """Use CPU operator activity everywhere and CUDA device activity when available."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def trace_output_path(project_root: Path, requested: str | Path) -> Path:
    """Resolve an optional profiler trace beneath the project unless absolute."""
    path = Path(requested)
    return path if path.is_absolute() else project_root / path
