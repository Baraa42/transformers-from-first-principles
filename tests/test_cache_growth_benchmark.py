"""CPU-only verification of sequential-cache instrumentation, not performance results."""

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from transformers_from_scratch.generation import generate_greedy, generate_greedy_cached
from transformers_from_scratch.model import TinyDecoderLM


@pytest.fixture(scope="module")
def bench():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_cache_growth.py"
    spec = importlib.util.spec_from_file_location("cache_growth_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny(bench):
    return bench.Config(
        prompt_lengths=(3,),
        generated_lengths=(2, 5),
        batch_sizes=(1, 2),
        vocab_size=19,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        repetitions=2,
        warmup_sequences=1,
    )


def test_default_workload_matrix_and_model(bench):
    config = bench.Config()
    assert len(config.workloads) == 18
    assert set(config.workloads) == {
        (p, n, b) for p in (32, 128, 256) for n in (16, 64, 128) for b in (1, 4)
    }
    assert config.model_kwargs == dict(
        vocab_size=8192, d_model=128, n_heads=4, d_ff=512, n_layers=4
    )
    with pytest.raises(ValueError):
        bench.Config(generated_lengths=(1,))
    with pytest.raises(ValueError):
        bench.Config(batch_sizes=(1, 1))
    with pytest.raises(ValueError):
        bench.Config(d_model=12)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_copy_accounting_explicit_sum_and_off_by_one(bench, n):
    p, b, layers, width, size = 3, 2, 2, 8, 4
    result = bench.copy_accounting(
        prompt_length=p,
        generated_tokens=n,
        batch_size=b,
        n_layers=layers,
        d_model=width,
        element_size=size,
    )
    q = n - 1
    c = 2 * layers * b * width * size
    assert result["decode_forwards"] == q
    assert result["kv_cat_calls"] == 2 * layers * q
    assert result["prefill_logical_cache_bytes"] == c * p
    assert result["final_logical_cache_bytes"] == c * (p + q)
    assert result["theoretical_old_prefix_recopied_bytes"] == sum(
        c * length for length in range(p, p + q)
    )
    assert result["theoretical_cat_output_bytes"] == sum(
        c * length for length in range(p + 1, p + q + 1)
    )
    assert result["theoretical_cat_read_plus_write_bytes"] == 2 * sum(
        c * length for length in range(p + 1, p + q + 1)
    )
    assert result["theoretical_new_kv_payload_bytes"] == c * q


@pytest.mark.parametrize("batch,n", [(1, 2), (2, 5)])
def test_sequence_shapes_tokens_and_actual_cat_payload(bench, tiny, monkeypatch, batch, n):
    torch.manual_seed(101)
    model = TinyDecoderLM(**tiny.model_kwargs).eval()
    prompt = torch.randint(tiny.vocab_size, (batch, 3))
    kwargs = dict(max_new_tokens=n, context_length=3 + n - 1)
    reference = generate_greedy(model, prompt, **kwargs)
    assert torch.equal(reference, generate_greedy_cached(model, prompt, **kwargs))
    cat = torch.cat
    payloads, old_payloads = [], []

    def observe_cat(tensors, dim=0, **kwargs):
        if dim == 2 and tensors[0].ndim == 4:
            old_payloads.append(tensors[0].numel() * tensors[0].element_size())
            payloads.append(sum(t.numel() * t.element_size() for t in tensors))
        return cat(tensors, dim=dim, **kwargs)

    monkeypatch.setattr(torch, "cat", observe_cat)
    metrics, actual, diag = bench.measure_sequence(model, prompt, n, torch.device("cpu"))
    assert torch.equal(actual, reference)
    assert not model.training
    assert actual.shape == (batch, 3 + n)
    assert diag["prefill_cache"]["shapes"] == [[batch, 2, 3, 4]] * 4
    assert diag["final_cache"]["shapes"] == [[batch, 2, 3 + n - 1, 4]] * 4
    expected = bench.copy_accounting(
        prompt_length=3, generated_tokens=n, batch_size=batch, n_layers=2, d_model=8, element_size=4
    )
    assert len(payloads) == expected["kv_cat_calls"]
    assert sum(payloads) == expected["theoretical_cat_output_bytes"]
    assert sum(old_payloads) == expected["theoretical_old_prefix_recopied_bytes"]
    assert diag["final_cache"]["logical_bytes"] == expected["final_logical_cache_bytes"]
    assert (
        diag["final_cache"]["unique_storage_capacity_bytes"] == diag["final_cache"]["logical_bytes"]
    )
    assert diag["memory_after_decode"]["peak_allocated_bytes"] is None
    assert metrics["decode_total_ms"] > 0


def test_timing_has_only_block_boundaries_and_diagnostics_outside(bench, tiny, monkeypatch):
    model = TinyDecoderLM(**tiny.model_kwargs).eval()
    prompt = torch.ones(1, 3, dtype=torch.long)
    events = []
    clocks = iter([0, 2_000_000, 10_000_000, 16_000_000])

    def clock():
        events.append("timer")
        return next(clocks)

    def memory(device):
        events.append("memory")
        return {}

    monkeypatch.setattr(bench.time, "perf_counter_ns", clock)
    monkeypatch.setattr(bench, "synchronize_device", lambda device: events.append("sync"))
    monkeypatch.setattr(bench, "memory_snapshot", memory)
    handle = model.register_forward_pre_hook(lambda *args: events.append("forward"))
    try:
        metrics, _, _ = bench.measure_sequence(model, prompt, 5, torch.device("cpu"))
    finally:
        handle.remove()
    assert events == [
        "memory",
        "sync",
        "timer",
        "forward",
        "sync",
        "timer",
        "memory",
        "sync",
        "timer",
        *(["forward"] * 4),
        "sync",
        "timer",
        "memory",
    ]
    assert metrics["prefill_ms"] == 2
    assert metrics["decode_total_ms"] == 6
    assert metrics["total_latency_ms"] == 8
    assert metrics["ms_per_decode_token"] == 1.5


def test_statistics_and_batch_denominators(bench):
    result = bench.summarize([8.0, 2.0, 4.0, 6.0])
    assert result["count"] == 4
    assert result["p50_ms"] == 5
    assert result["p95_ms"] == pytest.approx(7.7)
    assert result["sample_stddev_ms"] == pytest.approx(math.sqrt(20 / 3))
    metrics = bench.latency_metrics(6, generated_tokens=4, batch_size=2)
    assert metrics["ms_per_decode_token"] == 2
    assert metrics["aggregate_decode_tokens_per_second"] == 1000
    assert metrics["per_sequence_decode_tokens_per_second"] == 500
    for invalid in ([], [float("nan")], [0.0]):
        with pytest.raises(ValueError):
            bench.summarize(invalid)
    with pytest.raises(ValueError):
        bench.latency_metrics(1, generated_tokens=1, batch_size=1)


def test_cpu_artifacts_and_warmup_exclusion(bench, tiny, tmp_path):
    output = tmp_path / "run"
    summaries = bench.run_experiment(tiny, output, device_name="cpu")
    assert {p.name for p in output.iterdir()} == {
        "metadata.json",
        "correctness.json",
        "samples.csv",
        "summary.json",
    }
    metadata = json.loads((output / "metadata.json").read_text())
    correctness = json.loads((output / "correctness.json").read_text())
    assert metadata["device"] == "cpu"
    assert metadata["precision"] == "fp32"
    assert metadata["decode_denominator"].startswith("N-1")
    assert len(metadata["source_sha256"]) == 7
    assert correctness["passed"]
    assert len(correctness["checks"]) == len(tiny.workloads) * (1 + tiny.repetitions)
    with (output / "samples.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == len(tiny.workloads) * tiny.repetitions
    assert {int(r["repetition"]) for r in samples} == {0, 1}
    for workload in summaries:
        matching = [
            float(r["decode_total_ms"])
            for r in samples
            if tuple(int(r[k]) for k in ("P", "N", "B"))
            == tuple(workload[k] for k in ("P", "N", "B"))
        ]
        assert workload["decode_latency"] == bench.summarize(matching)
    with pytest.raises(FileExistsError):
        bench.run_experiment(tiny, output, device_name="cpu")


def test_correctness_failure_stops_all_measurements(bench, tiny, tmp_path, monkeypatch):
    def corrupted(model, prompt, **kwargs):
        result = generate_greedy_cached(model, prompt, **kwargs).clone()
        result[:, -1] = (result[:, -1] + 1) % tiny.vocab_size
        return result

    monkeypatch.setattr(bench, "generate_greedy_cached", corrupted)
    monkeypatch.setattr(bench, "measure_sequence", lambda *args: pytest.fail("timing started"))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="greedy mismatch"):
        bench.run_experiment(tiny, output, device_name="cpu")
    assert not json.loads((output / "correctness.json").read_text())["passed"]
    assert not (output / "samples.csv").exists()


def test_mps_unavailable_rejected_before_artifacts(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    output = tmp_path / "unavailable"
    with pytest.raises(RuntimeError, match="no automatic CPU fallback"):
        bench.run_experiment(bench.Config(), output)
    assert not output.exists()
