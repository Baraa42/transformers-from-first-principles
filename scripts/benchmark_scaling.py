"""Run the synchronized FP32 batch/context scaling benchmark."""

import subprocess
import sys
from pathlib import Path

from transformers_from_scratch.benchmarking import (
    ScalingBenchmarkMetrics,
    relative_scaling_throughput,
    scaling_metrics,
)
from transformers_from_scratch.training import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "benchmarks" / "scaling"
LOG_DIR = PROJECT_ROOT / "benchmarks" / "scaling"
RUN_NAMES = (
    "context64-batch16",
    "context128-batch8",
    "context128-batch16",
    "context128-batch32",
    "context256-batch16",
)


def run_training(name: str, config_path: Path, log_path: Path) -> tuple[int, str]:
    """Run one fresh training process while teeing combined output to disk."""
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "train.py"),
        "--config",
        str(config_path),
    ]
    lines: list[str] = []
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Training subprocess stdout was not captured")
        for line in process.stdout:
            lines.append(line)
            log_file.write(line)
            log_file.flush()
            print(f"[{name}] {line}", end="")
        return_code = process.wait()
    return return_code, "".join(lines)


def print_summary(results: list[ScalingBenchmarkMetrics]) -> None:
    """Print latency and derived throughput relative to T=128, B=16."""
    relative = relative_scaling_throughput(results)
    print()
    print("context | batch | tokens/step | median_step_ms | p95_ms | median_tok_s | vs_baseline")
    for result in results:
        speedup = relative.get((result.context_length, result.batch_size))
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        print(
            f"{result.context_length:>7} | {result.batch_size:>5} "
            f"| {result.tokens_per_step:>11} | {result.median_step_ms:>14.3f} "
            f"| {result.p95_step_ms:>6.3f} | {result.median_tokens_per_sec:>12.0f} "
            f"| {speedup_text}"
        )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ScalingBenchmarkMetrics] = []
    failures: list[str] = []

    for name in RUN_NAMES:
        config_path = CONFIG_DIR / f"{name}.yaml"
        config = load_config(config_path)
        data_config = config["data"]
        training_config = config["training"]
        if training_config.get("precision") != "fp32":
            failures.append(f"{name} must use precision=fp32")
            continue
        if training_config.get("profile_timing") is not True:
            failures.append(f"{name} must enable profile_timing")
            continue

        context_length = data_config["context_length"]
        batch_size = data_config["batch_size"]
        print(f"starting name={name} context={context_length} batch={batch_size}")
        log_path = LOG_DIR / f"{name}.log"
        return_code, output = run_training(name, config_path, log_path)
        if return_code:
            failures.append(f"{name} failed with exit code {return_code}; see {log_path}")
            continue
        try:
            results.append(
                scaling_metrics(
                    context_length=context_length,
                    batch_size=batch_size,
                    log=output,
                )
            )
        except ValueError as error:
            failures.append(f"{name} metrics could not be parsed: {error}; see {log_path}")

    if results:
        print_summary(results)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
