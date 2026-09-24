import copy
from pathlib import Path

import pytest
import yaml

from transformers_from_scratch.benchmarking import (
    parse_profile_step_times,
    relative_scaling_throughput,
    scaling_metrics,
)

MATRIX = (
    ("context64-batch16", 64, 16),
    ("context128-batch8", 128, 8),
    ("context128-batch16", 128, 16),
    ("context128-batch32", 128, 32),
    ("context256-batch16", 256, 16),
)


def fake_profile(median_ms: float, p95_ms: float) -> str:
    return f"timing_profile=true\nstep_time_median_ms={median_ms}\nstep_time_p95_ms={p95_ms}\n"


def test_scaling_metric_math_uses_batch_times_context() -> None:
    result = scaling_metrics(
        context_length=128,
        batch_size=16,
        log=fake_profile(median_ms=20.0, p95_ms=25.0),
    )

    assert result.tokens_per_step == 2048
    assert result.median_tokens_per_sec == pytest.approx(102_400)


def test_relative_scaling_throughput_normalizes_baseline() -> None:
    baseline = scaling_metrics(
        context_length=128,
        batch_size=16,
        log=fake_profile(median_ms=20.0, p95_ms=25.0),
    )
    faster = scaling_metrics(
        context_length=128,
        batch_size=32,
        log=fake_profile(median_ms=30.0, p95_ms=35.0),
    )

    relative = relative_scaling_throughput([baseline, faster])

    assert relative[(128, 16)] == pytest.approx(1.0)
    assert relative[(128, 32)] == pytest.approx((4096 / 0.030) / (2048 / 0.020))


def test_profile_step_time_parser_extracts_median_and_p95() -> None:
    median, p95 = parse_profile_step_times(fake_profile(median_ms=21.5, p95_ms=29.75))

    assert median == 21.5
    assert p95 == 29.75


def test_scaling_configs_differ_only_in_batch_context_and_checkpoint_dir() -> None:
    configs = []
    actual_matrix = set()
    checkpoint_dirs = set()

    for name, expected_context, expected_batch in MATRIX:
        path = Path(f"configs/benchmarks/scaling/{name}.yaml")
        config = yaml.safe_load(path.read_text())
        assert config["training"]["precision"] == "fp32"
        assert config["training"]["profile_timing"] is True
        actual_matrix.add((config["data"]["context_length"], config["data"]["batch_size"]))
        assert (config["data"]["context_length"], config["data"]["batch_size"]) == (
            expected_context,
            expected_batch,
        )

        comparable = copy.deepcopy(config)
        del comparable["data"]["context_length"]
        del comparable["data"]["batch_size"]
        checkpoint_dirs.add(comparable["training"].pop("checkpoint_dir"))
        configs.append(comparable)

    assert actual_matrix == {(context, batch) for _, context, batch in MATRIX}
    assert configs[0] == configs[1] == configs[2] == configs[3] == configs[4]
    assert len(checkpoint_dirs) == len(MATRIX)
