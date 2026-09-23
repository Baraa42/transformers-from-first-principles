import copy

import pytest
import yaml

from transformers_from_scratch.benchmarking import (
    BenchmarkMetrics,
    parse_training_log,
    relative_throughput,
)


def test_parse_training_log_uses_final_losses_and_median_throughput() -> None:
    log = """
step=20 train_loss=4.0000 lr=3.00e-04 grad_norm=1.0 tokens_per_sec=100
step=20 val_loss=3.9000 val_perplexity=49.40
step=40 train_loss=3.5000 lr=3.00e-04 grad_norm=1.0 tokens_per_sec=300
step=40 val_loss=3.4000 val_perplexity=29.96
step=60 train_loss=3.0000 lr=3.00e-04 grad_norm=1.0 tokens_per_sec=200
training_wall_time_sec=12.345
amp_overflow_count=2 final_grad_scale=32768
"""

    result = parse_training_log(log, "fp16")

    assert result.precision == "fp16"
    assert result.training_wall_time_sec == 12.345
    assert result.median_tokens_per_sec == 200
    assert result.final_train_loss == 3.0
    assert result.final_val_loss == 3.4
    assert result.final_val_perplexity == 29.96
    assert result.amp_overflow_count == 2
    assert result.final_grad_scale == 32768


def test_relative_throughput_uses_fp32_median_as_baseline() -> None:
    def result(precision: str, throughput: float) -> BenchmarkMetrics:
        return BenchmarkMetrics(
            precision=precision,
            training_wall_time_sec=1.0,
            median_tokens_per_sec=throughput,
            final_train_loss=1.0,
            final_val_loss=1.0,
            final_val_perplexity=1.0,
            amp_overflow_count=0,
            final_grad_scale=None,
        )

    relative = relative_throughput([result("fp32", 100), result("fp16", 150), result("bf16", 80)])

    assert relative == {"fp32": 1.0, "fp16": 1.5, "bf16": 0.8}


def test_precision_benchmark_configs_differ_only_in_precision_and_checkpoint_dir() -> None:
    configs = {}
    for precision in ("fp32", "fp16", "bf16"):
        path = f"configs/benchmarks/precision-{precision}.yaml"
        with open(path) as config_file:
            configs[precision] = yaml.safe_load(config_file)

    normalized = []
    checkpoint_dirs = set()
    for precision, config in configs.items():
        assert config["training"]["precision"] == precision
        checkpoint_dirs.add(config["training"]["checkpoint_dir"])
        comparable = copy.deepcopy(config)
        del comparable["training"]["precision"]
        del comparable["training"]["checkpoint_dir"]
        normalized.append(comparable)

    assert normalized[0] == normalized[1] == normalized[2]
    assert len(checkpoint_dirs) == 3


def test_parse_training_log_requires_all_summary_metrics() -> None:
    with pytest.raises(ValueError, match="missing required metrics"):
        parse_training_log("tokens_per_sec=100", "fp32")
