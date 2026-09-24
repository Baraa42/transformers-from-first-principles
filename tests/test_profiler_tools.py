from pathlib import Path

import pytest
import torch

from transformers_from_scratch.profiler_tools import (
    profiler_activities,
    trace_output_path,
    validate_profiler_steps,
)


@pytest.mark.parametrize(
    ("warmup_steps", "profile_steps"),
    [(-1, 1), (0, 0), (True, 1), (0, False), (1.5, 1), (0, "1")],
)
def test_validate_profiler_steps_rejects_invalid_values(
    warmup_steps: object, profile_steps: object
) -> None:
    with pytest.raises(ValueError):
        validate_profiler_steps(warmup_steps, profile_steps)


def test_validate_profiler_steps_accepts_defaults() -> None:
    assert validate_profiler_steps(5, 15) == (5, 15)


def test_mps_profiler_uses_cpu_activity_without_assuming_mps_activity() -> None:
    assert profiler_activities(torch.device("mps")) == [torch.profiler.ProfilerActivity.CPU]


def test_trace_output_path_resolves_relative_path_under_project() -> None:
    project_root = Path("/project")
    assert trace_output_path(project_root, "benchmarks/profiler/trace.json") == (
        project_root / "benchmarks/profiler/trace.json"
    )


def test_trace_output_path_preserves_absolute_path() -> None:
    path = Path("/tmp/custom-trace.json")
    assert trace_output_path(Path("/project"), path) == path
