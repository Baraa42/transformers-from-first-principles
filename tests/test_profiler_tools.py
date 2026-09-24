import runpy
from pathlib import Path

import pytest
import torch

from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.profiler_tools import (
    profiler_activities,
    trace_output_path,
    validate_profiler_steps,
)


def test_profiler_training_step_returns_detached_scalar_tensor() -> None:
    run_training_step = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "profile_training_ops.py")
    )["run_training_step"]
    model = TinyDecoderLM(vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = iter(
        [
            (
                torch.randint(0, 7, (2, 3)),
                torch.randint(0, 7, (2, 3)),
            )
        ]
    )

    loss = run_training_step(
        model=model,
        optimizer=optimizer,
        batches=batches,
        device=torch.device("cpu"),
        precision="fp32",
        max_grad_norm=None,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not loss.requires_grad
    assert loss.grad_fn is None


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
