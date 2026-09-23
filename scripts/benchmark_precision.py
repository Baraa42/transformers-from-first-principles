"""Run the controlled FP32/FP16/BF16 training benchmark."""

import subprocess
import sys
from pathlib import Path

from transformers_from_scratch.benchmarking import (
    BenchmarkMetrics,
    parse_training_log,
    relative_throughput,
)
from transformers_from_scratch.training import (
    load_config,
    resolve_device,
    validate_precision_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "benchmarks" / "precision"
RUNS = (
    ("fp32", PROJECT_ROOT / "configs" / "benchmarks" / "precision-fp32.yaml"),
    ("fp16", PROJECT_ROOT / "configs" / "benchmarks" / "precision-fp16.yaml"),
    ("bf16", PROJECT_ROOT / "configs" / "benchmarks" / "precision-bf16.yaml"),
)


def run_training(precision: str, config_path: Path, log_path: Path) -> tuple[int, str]:
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
            print(f"[{precision}] {line}", end="")
        return_code = process.wait()
    return return_code, "".join(lines)


def print_summary(results: list[BenchmarkMetrics]) -> None:
    """Print the completed benchmark rows and FP32-relative throughput."""
    relative = relative_throughput(results)
    print()
    print(
        "precision | wall_sec | median_tok_s | train_loss | val_loss | val_ppl "
        "| overflows | final_scale | vs_fp32"
    )
    for result in results:
        scale = "none" if result.final_grad_scale is None else f"{result.final_grad_scale:g}"
        speedup = relative.get(result.precision)
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        print(
            f"{result.precision:<9} | {result.training_wall_time_sec:>8.3f} "
            f"| {result.median_tokens_per_sec:>12.0f} "
            f"| {result.final_train_loss:>10.4f} | {result.final_val_loss:>8.4f} "
            f"| {result.final_val_perplexity:>7.2f} "
            f"| {result.amp_overflow_count:>9} | {scale:>11} | {speedup_text}"
        )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkMetrics] = []
    failures: list[str] = []

    for precision, config_path in RUNS:
        config = load_config(config_path)
        device = resolve_device(config["device"])
        try:
            validate_precision_support(device, precision)
        except ValueError as error:
            message = f"{precision}=unsupported: {error}"
            print(message)
            (LOG_DIR / f"{precision}.log").write_text(message + "\n")
            continue

        print(f"starting precision={precision} device={device.type}")
        log_path = LOG_DIR / f"{precision}.log"
        return_code, output = run_training(precision, config_path, log_path)
        if return_code:
            failures.append(f"{precision} failed with exit code {return_code}; see {log_path}")
            continue
        try:
            results.append(parse_training_log(output, precision))
        except ValueError as error:
            failures.append(f"{precision} metrics could not be parsed: {error}; see {log_path}")

    if results:
        print_summary(results)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
