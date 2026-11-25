"""Phase 4B: paired fresh-request cat/preallocated KV-cache comparison."""

import argparse
import csv
import hashlib
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.benchmark_cache_growth import (
    ROOT,
    cache_diagnostics,
    command_output,
    copy_accounting,
    digest,
    latency_metrics,
    memory_snapshot,
    require_device,
    summarize,
    write_json,
)
from scripts.benchmark_cache_growth import (
    Config as BaselineConfig,
)
from transformers_from_scratch.generation import greedy_next_token
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.preallocated_cache import PreallocatedKVCache
from transformers_from_scratch.training import synchronize_device

MODES = ("cat", "preallocated")


@dataclass(frozen=True)
class Config(BaselineConfig):
    repetitions: int = 6

    def __post_init__(self):
        super().__post_init__()
        if self.repetitions % 2:
            raise ValueError("repetitions must be even to balance measurement order")


def mode_kwargs(mode, capacity=None):
    if mode not in MODES:
        raise ValueError("invalid cache mode")
    if mode == "cat":
        return {}
    result = {"cache_mode": mode}
    if capacity is not None:
        result["cache_capacity"] = capacity
    return result


def diagnostics(cache):
    if isinstance(cache, PreallocatedKVCache):
        result = cache_diagnostics(cache.logical_layers)
        result.update(length=cache.length, capacity=cache.capacity)
        return result
    result = cache_diagnostics(cache)
    result.update(length=cache[0][0].shape[2], capacity=cache[0][0].shape[2])
    return result


@torch.inference_mode()
def correctness_gate(model, prompt, n):
    """Every-step FP32 logits, greedy IDs, K/V contents, length and stable storage."""
    capacity = prompt.shape[1] + n - 1
    cat_logits, cat = model(prompt, use_cache=True)
    pre_logits, pre = model(prompt, use_cache=True, **mode_kwargs("preallocated", capacity))
    pointers = [t.data_ptr() for pair in pre.layers for t in pair]
    generated = prompt.clone()
    max_error = 0.0
    for step in range(n):
        torch.testing.assert_close(pre_logits, cat_logits, rtol=1e-5, atol=1e-6)
        max_error = max(max_error, (pre_logits - cat_logits).abs().max().item())
        assert pre.length == prompt.shape[1] + step
        for actual, expected in zip(pre.logical_layers, cat, strict=True):
            for a, e in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
        assert pointers == [t.data_ptr() for pair in pre.layers for t in pair]
        token = greedy_next_token(cat_logits[:, -1])
        assert torch.equal(token, greedy_next_token(pre_logits[:, -1]))
        generated = torch.cat((generated, token), dim=1)
        if step < n - 1:
            cat_logits, cat = model(token, use_cache=True, kv_cache=cat)
            pre_logits, pre = model(
                token, use_cache=True, kv_cache=pre, **mode_kwargs("preallocated")
            )
    return generated.cpu(), {"steps_checked": n, "max_abs_logit_error": max_error}


@torch.inference_mode()
def measure_sequence(model, prompt, n, device, mode):
    """Fresh cache each call; allocation is inside prefill; block-boundary sync only."""
    if model.training or n < 2:
        raise ValueError("eval model and N >= 2 required")
    before = memory_snapshot(device)
    synchronize_device(device)
    start = time.perf_counter_ns()
    logits, cache = model(prompt, use_cache=True, **mode_kwargs(mode, prompt.shape[1] + n - 1))
    token = greedy_next_token(logits[:, -1])
    generated = torch.cat((prompt.clone(), token), dim=1)
    del logits
    synchronize_device(device)
    prefill_ms = (time.perf_counter_ns() - start) / 1e6
    prefill_cache = diagnostics(cache)
    after_prefill = memory_snapshot(device)
    synchronize_device(device)
    start = time.perf_counter_ns()
    for _ in range(n - 1):
        logits, cache = model(token, use_cache=True, kv_cache=cache, **mode_kwargs(mode))
        token = greedy_next_token(logits[:, -1])
        generated = torch.cat((generated, token), dim=1)
        del logits
    synchronize_device(device)
    decode_ms = (time.perf_counter_ns() - start) / 1e6
    metrics = latency_metrics(decode_ms, n, prompt.shape[0])
    metrics.update(prefill_ms=prefill_ms, total_latency_ms=prefill_ms + decode_ms)
    diag = dict(
        prefill_cache=prefill_cache,
        final_cache=diagnostics(cache),
        memory_before_prefill=before,
        memory_after_prefill=after_prefill,
        memory_after_decode=memory_snapshot(device),
    )
    return metrics, generated.cpu(), diag


def run_experiment(config, output, device_name="mps"):
    device = require_device(device_name)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.model_seed)
    model = TinyDecoderLM(**config.model_kwargs).to(device).eval()
    generator = torch.Generator().manual_seed(config.prompt_seed)
    prompts = {
        (p, b): torch.randint(config.vocab_size, (b, p), generator=generator).to(device)
        for p in config.prompt_lengths
        for b in config.batch_sizes
    }
    order = torch.randperm(
        len(config.workloads), generator=torch.Generator().manual_seed(config.order_seed)
    ).tolist()
    workloads = [config.workloads[i] for i in order]
    sources = [Path(__file__), ROOT / "scripts/benchmark_cache_growth.py"] + list(
        (ROOT / "src/transformers_from_scratch").glob("*.py")
    )
    metadata = dict(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        experiment="Phase 4B paired preallocated cache",
        config=asdict(config),
        device=str(device),
        precision="fp32",
        pytorch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        git_revision=command_output(["git", "rev-parse", "HEAD"]),
        git_status=command_output(["git", "status", "--porcelain"]),
        torch_threads=torch.get_num_threads(),
        pytorch_git=torch.version.git_version,
        hardware_model=command_output(["sysctl", "-n", "hw.model"]),
        unified_memory_bytes=command_output(["sysctl", "-n", "hw.memsize"]),
        backend_environment={
            key: os.environ.get(key)
            for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
            )
        },
        initial_weights_sha256=digest(
            torch.cat([t.flatten() for t in model.state_dict().values()])
        ),
        prompt_sha256={f"P{p}_B{b}": digest(t) for (p, b), t in prompts.items()},
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        execution_order_P_N_B=workloads,
        timing="Sync before/after prefill and entire decode block; allocation inside prefill; "
        "diagnostics and CPU reads outside timers; no per-token sync",
        decode_denominator="N-1; prefill selects first token; last token is not forwarded",
        warmup="Separate full fresh requests for each mode/workload, excluded from samples",
        overhead="Preallocated: 2L empty buffers, 2L prefill copies, 2L slice writes/decode; "
        "valid-prefix views and all-layer validation each decode. Cat: 2L cats/decode. "
        "Both include growing output-ID cat and argmax. Kernel/allocation events "
        "and physical traffic are not profiled; boundary memory is not peak memory.",
    )
    write_json(output / "metadata.json", metadata)
    checks = {"passed": False, "checks": [], "measurements": []}
    references = {}
    for p, n, b in workloads:
        try:
            references[p, n, b], result = correctness_gate(model, prompts[p, b], n)
            checks["checks"].append(dict(P=p, N=n, B=b, **result))
        finally:
            write_json(output / "correctness.json", checks)
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
        for mode in MODES:
            for _ in range(config.warmup_sequences):
                _, tokens, _ = measure_sequence(model, prompts[p, b], n, device, mode)
                assert torch.equal(tokens, references[p, n, b])
        current = []
        for repetition in range(config.repetitions):
            modes = MODES if repetition % 2 == 0 else MODES[::-1]
            for position, mode in enumerate(modes):
                metrics, tokens, diag = measure_sequence(model, prompts[p, b], n, device, mode)
                assert torch.equal(tokens, references[p, n, b])
                assert (
                    diag["final_cache"]["logical_bytes"] == accounting["final_logical_cache_bytes"]
                )
                assert (
                    diag["final_cache"]["unique_storage_capacity_bytes"]
                    == accounting["final_logical_cache_bytes"]
                )
                row = dict(
                    P=p, N=n, B=b, repetition=repetition, position=position, mode=mode, **metrics
                )
                current.append(row)
                rows.append(row)
                checks["measurements"].append(
                    dict(P=p, N=n, B=b, repetition=repetition, mode=mode, **diag)
                )
        mode_stats = {}
        for mode in MODES:
            selected = [r for r in current if r["mode"] == mode]
            mode_stats[mode] = {
                "prefill_latency": summarize([r["prefill_ms"] for r in selected]),
                "decode_latency": summarize([r["decode_total_ms"] for r in selected]),
                "total_latency": summarize([r["total_latency_ms"] for r in selected]),
                "median_decode_ms_per_token": statistics.median(
                    r["ms_per_decode_token"] for r in selected
                ),
                "aggregate_decode_tokens_per_second": 1000
                * b
                * (n - 1)
                * len(selected)
                / sum(r["decode_total_ms"] for r in selected),
                "aggregate_end_to_end_tokens_per_second": 1000
                * b
                * n
                * len(selected)
                / sum(r["total_latency_ms"] for r in selected),
            }
        paired = {}
        for metric in ("prefill_ms", "decode_total_ms", "total_latency_ms"):
            ratios = []
            for rep in range(config.repetitions):
                pair = {r["mode"]: r[metric] for r in current if r["repetition"] == rep}
                ratios.append(pair["cat"] / pair["preallocated"])
            paired[metric] = dict(
                samples=ratios,
                median=statistics.median(ratios),
                min=min(ratios),
                max=max(ratios),
                sample_stddev=statistics.stdev(ratios),
            )
        summaries.append(
            dict(
                P=p,
                N=n,
                B=b,
                modes=mode_stats,
                paired_speedup=paired,
                final_logical_bytes=accounting["final_logical_cache_bytes"],
                final_reserved_bytes=accounting["final_logical_cache_bytes"],
                theoretical_old_prefix_bytes_eliminated=accounting[
                    "theoretical_old_prefix_recopied_bytes"
                ],
                accounting=accounting,
                operation_counts={
                    "preallocated_kv_storage_allocations_per_request": 2 * config.n_layers,
                    "preallocated_prefill_slice_writes": 2 * config.n_layers,
                    "preallocated_decode_slice_writes": 2 * config.n_layers * (n - 1),
                    "preallocated_decode_kv_storage_allocations": 0,
                    "cat_decode_kv_allocations": 2 * config.n_layers * (n - 1),
                    "note": "Algorithmic counts, not profiled allocator/kernel events. "
                    "Projection, attention, RoPE and output-ID allocations remain.",
                },
            )
        )
        write_json(output / "correctness.json", checks)
    checks["passed"] = True
    write_json(output / "correctness.json", checks)
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "workloads": summaries,
            "caveat": "Theoretical copy elimination is not measured physical traffic savings.",
        },
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(Config(), args.output, args.device)
    print(f"Saved paired comparison to {args.output}")


if __name__ == "__main__":
    main()
