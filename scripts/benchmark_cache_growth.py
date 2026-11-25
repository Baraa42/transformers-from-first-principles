"""Sequential torch.cat KV-cache baseline; no preallocation or copy-cost attribution."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from transformers_from_scratch.generation import (
    generate_greedy,
    generate_greedy_cached,
    greedy_next_token,
)
from transformers_from_scratch.inference_benchmarking import kv_cache_size_bytes
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import synchronize_device

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    prompt_lengths: tuple[int, ...] = (32, 128, 256)
    generated_lengths: tuple[int, ...] = (16, 64, 128)
    batch_sizes: tuple[int, ...] = (1, 4)
    vocab_size: int = 8192
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    repetitions: int = 5
    warmup_sequences: int = 2
    model_seed: int = 42
    prompt_seed: int = 43
    order_seed: int = 44

    def __post_init__(self):
        for name in ("prompt_lengths", "generated_lengths", "batch_sizes"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            for value in values:
                positive_integer(name, value)
        if min(self.generated_lengths) < 2:
            raise ValueError("generated_lengths must be >= 2 for decode timing")
        for name in (
            "vocab_size",
            "d_model",
            "n_heads",
            "d_ff",
            "n_layers",
            "repetitions",
            "warmup_sequences",
        ):
            positive_integer(name, getattr(self, name))
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("model requires an integral, even head dimension")

    @property
    def model_kwargs(self):
        return {
            name: getattr(self, name)
            for name in ("vocab_size", "d_model", "n_heads", "d_ff", "n_layers")
        }

    @property
    def workloads(self):
        return [
            (p, n, b)
            for p in self.prompt_lengths
            for n in self.generated_lengths
            for b in self.batch_sizes
        ]


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def copy_accounting(
    *, prompt_length, generated_tokens, batch_size, n_layers, d_model, element_size
):
    """Logical cat payload, NOT measured physical traffic or allocator activity.

    q=N-1: prefill selects token one; the last generated token is never forwarded.
    A cat reads both inputs and writes their entire concatenated output.
    """
    for name, value in locals().copy().items():
        positive_integer(name, value)
    q = generated_tokens - 1
    c = kv_cache_size_bytes(
        n_layers=n_layers,
        batch_size=batch_size,
        sequence_length=1,
        d_model=d_model,
        bytes_per_element=element_size,
    )
    old = c * (q * prompt_length + q * (q - 1) // 2)
    payload = c * (q * prompt_length + q * (q + 1) // 2)
    return {
        "decode_forwards": q,
        "kv_cat_calls": 2 * n_layers * q,
        "prefill_logical_cache_bytes": c * prompt_length,
        "final_logical_cache_bytes": c * (prompt_length + q),
        "theoretical_old_prefix_recopied_bytes": old,
        "theoretical_cat_output_bytes": payload,
        "theoretical_cat_read_plus_write_bytes": 2 * payload,
        "theoretical_new_kv_payload_bytes": c * q,
    }


def cache_diagnostics(cache):
    """Observed tensor shapes/bytes; storage capacity is not physical peak memory."""
    tensors = [tensor for layer in cache for tensor in layer]
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return {
        "shapes": [list(t.shape) for t in tensors],
        "logical_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "unique_storage_capacity_bytes": sum(storages.values()),
    }


def memory_snapshot(device):
    """Boundary counters include the whole process; they are not per-cat costs."""
    result = {
        "current_allocated_bytes": None,
        "driver_allocated_bytes": None,
        "recommended_max_bytes": None,
        "peak_allocated_bytes": None,
    }
    if device.type == "mps":
        for key, api in (
            ("current_allocated_bytes", "current_allocated_memory"),
            ("driver_allocated_bytes", "driver_allocated_memory"),
            ("recommended_max_bytes", "recommended_max_memory"),
        ):
            try:
                result[key] = int(getattr(torch.mps, api)())
            except (AttributeError, RuntimeError):
                pass
    return result


def latency_metrics(decode_ms, generated_tokens, batch_size):
    positive_integer("batch_size", batch_size)
    positive_integer("generated_tokens", generated_tokens)
    if generated_tokens < 2 or not math.isfinite(decode_ms) or decode_ms <= 0:
        raise ValueError("positive finite latency and >= 2 generated tokens required")
    q = generated_tokens - 1
    return {
        "decode_total_ms": decode_ms,
        "ms_per_decode_token": decode_ms / q,
        "aggregate_decode_tokens_per_second": 1000 * batch_size * q / decode_ms,
        "per_sequence_decode_tokens_per_second": 1000 * q / decode_ms,
    }


@torch.inference_mode()
def measure_sequence(model, prompt, generated_tokens, device):
    """One real sequence, sync only at prefill/decode boundaries, no per-token reads.

    Caller sets eval mode. Prefill includes first argmax and output initialization.
    Decode includes N-1 forwards, argmax and output-ID cat, as production generation does.
    """
    if model.training:
        raise ValueError("benchmark model must be in eval mode")
    positive_integer("generated_tokens", generated_tokens)
    if generated_tokens < 2:
        raise ValueError("at least two generated tokens required")
    before = memory_snapshot(device)
    synchronize_device(device)
    started = time.perf_counter_ns()
    logits, cache = model(prompt, use_cache=True)
    next_token = greedy_next_token(logits[:, -1, :])
    generated = torch.cat((prompt.clone(), next_token), dim=1)
    del logits
    synchronize_device(device)
    prefill_ms = (time.perf_counter_ns() - started) / 1e6
    prefill_cache = cache_diagnostics(cache)
    after_prefill = memory_snapshot(device)

    synchronize_device(device)
    started = time.perf_counter_ns()
    for _ in range(generated_tokens - 1):
        logits, cache = model(next_token, use_cache=True, kv_cache=cache)
        next_token = greedy_next_token(logits[:, -1, :])
        generated = torch.cat((generated, next_token), dim=1)
        del logits
    synchronize_device(device)
    decode_ms = (time.perf_counter_ns() - started) / 1e6

    diagnostics = {
        "prefill_cache": prefill_cache,
        "final_cache": cache_diagnostics(cache),
        "memory_before_prefill": before,
        "memory_after_prefill": after_prefill,
        "memory_after_decode": memory_snapshot(device),
    }
    metrics = latency_metrics(decode_ms, generated_tokens, prompt.shape[0])
    metrics.update(prefill_ms=prefill_ms, total_latency_ms=prefill_ms + decode_ms)
    return metrics, generated, diagnostics


def summarize(samples):
    """Whole-sequence percentiles; not per-token tail latency."""
    if not samples:
        raise ValueError("samples must not be empty")
    values = sorted(samples)
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("samples must be positive and finite")
    index = 0.95 * (len(values) - 1)
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": values[low] + (values[high] - values[low]) * (index - low),
        "min_ms": values[0],
        "max_ms": values[-1],
        "sample_stddev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def require_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no automatic CPU fallback")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for MPS measurements")
    device = torch.device("mps")
    # Exercise actual dispatch before creating output artifacts.
    if (torch.ones(1, device=device) + 1).cpu().item() != 2:
        raise RuntimeError("MPS operation probe failed")
    synchronize_device(device)
    return device


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def command_output(args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(config, output, device_name="mps"):
    device = require_device(device_name)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.model_seed)
    model = TinyDecoderLM(**config.model_kwargs).to(device=device, dtype=torch.float32).eval()
    generator = torch.Generator(device="cpu").manual_seed(config.prompt_seed)
    # One distinct synthetic prompt for each (P,B), shared across N and repetitions.
    prompts = {
        (p, b): torch.randint(config.vocab_size, (b, p), generator=generator).to(device)
        for p in config.prompt_lengths
        for b in config.batch_sizes
    }
    order = torch.randperm(
        len(config.workloads), generator=torch.Generator().manual_seed(config.order_seed)
    ).tolist()
    workloads = [config.workloads[i] for i in order]
    source_paths = [
        Path(__file__).resolve(),
        *(
            ROOT / "src/transformers_from_scratch" / f"{name}.py"
            for name in (
                "attention",
                "model",
                "generation",
                "rope",
                "layers",
                "inference_benchmarking",
            )
        ),
    ]
    metadata = {
        "schema_version": 1,
        "experiment": "Sequential torch.cat KV-cache growth baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "device": str(device),
        "precision": "fp32",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_git": torch.version.git_version,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "hardware_model": command_output(["sysctl", "-n", "hw.model"]),
        "cpu_model": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "unified_memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "torch_threads": torch.get_num_threads(),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "initial_weights_sha256": hashlib.sha256(
            "".join(name + digest(t) for name, t in model.state_dict().items()).encode()
        ).hexdigest(),
        "prompt_sha256": {f"P{p}_B{b}": digest(t) for (p, b), t in prompts.items()},
        "backend_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
            )
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "execution_order_P_N_B": workloads,
        "timing": "perf_counter_ns; synchronize before/after prefill and entire decode loop; "
        "no per-token synchronization; diagnostic reads outside timers",
        "decode_denominator": "N-1 forwards; prefill selects first of N generated tokens",
        "warmup": "full sequential generations per workload, excluded from samples",
        "memory_limitations": "Boundary counters and storage capacity only; no physical peak, "
        "allocation events, or directly measured cat copy costs",
    }
    write_json(output / "metadata.json", metadata)
    correctness = {"schema_version": 1, "passed": False, "checks": []}
    references = {}
    # All equivalence checks precede benchmark timing/warmup.
    for p, n, b in workloads:
        prompt = prompts[p, b]
        kwargs = {"max_new_tokens": n, "context_length": p + n - 1}
        reference = generate_greedy(model, prompt, **kwargs)
        cached = generate_greedy_cached(model, prompt, **kwargs)
        equal = torch.equal(reference, cached)
        correctness["checks"].append(
            {
                "P": p,
                "N": n,
                "B": b,
                "reference_equal": equal,
                "reference_tokens": reference.cpu().tolist(),
            }
        )
        write_json(output / "correctness.json", correctness)
        if not equal:
            raise RuntimeError(f"cached/uncached greedy mismatch: P={p}, N={n}, B={b}")
        references[p, n, b] = reference.cpu()
        del reference, cached
    rows, summaries = [], []
    for p, n, b in workloads:
        accounting = copy_accounting(
            prompt_length=p,
            generated_tokens=n,
            batch_size=b,
            n_layers=config.n_layers,
            d_model=config.d_model,
            element_size=4,
        )
        for repetition in range(-config.warmup_sequences, config.repetitions):
            metrics, generated, diagnostics = measure_sequence(model, prompts[p, b], n, device)
            if not torch.equal(generated.cpu(), references[p, n, b]):
                raise RuntimeError(f"instrumented greedy mismatch: P={p}, N={n}, B={b}")
            if (
                diagnostics["final_cache"]["logical_bytes"]
                != accounting["final_logical_cache_bytes"]
            ):
                raise RuntimeError("observed final cache differs from theoretical logical size")
            del generated
            if repetition >= 0:
                rows.append({"P": p, "N": n, "B": b, "repetition": repetition, **metrics})
                correctness["checks"].append(
                    {
                        "P": p,
                        "N": n,
                        "B": b,
                        "repetition": repetition,
                        "instrumented_equal": True,
                        **diagnostics,
                    }
                )
        current = [r for r in rows if (r["P"], r["N"], r["B"]) == (p, n, b)]
        stats = summarize([r["decode_total_ms"] for r in current])
        summaries.append(
            {
                "P": p,
                "N": n,
                "B": b,
                **accounting,
                "decode_latency": stats,
                "metrics_from_p50": latency_metrics(stats["p50_ms"], n, b),
                "total_latency": summarize([r["total_latency_ms"] for r in current]),
            }
        )
    correctness["passed"] = True
    write_json(output / "correctness.json", correctness)
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", {"schema_version": 1, "workloads": summaries})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-sequences", type=int, default=2)
    args = parser.parse_args()
    config = Config(repetitions=args.repetitions, warmup_sequences=args.warmup_sequences)
    run_experiment(config, args.output, args.device)
    print(f"Saved cache-growth baseline to {args.output}")


if __name__ == "__main__":
    main()
