"""Small CPU runs verify pairing/artifacts without running the full benchmark."""

import csv
import importlib
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("scripts.benchmark_preallocated_cache")


def test_matrix_and_pairing(bench):
    assert bench.Config().repetitions == 6
    assert len(bench.Config().workloads) == 18
    with pytest.raises(ValueError):
        bench.Config(repetitions=3)


def test_cpu_run(bench, tmp_path):
    config = bench.Config(
        prompt_lengths=(3,),
        generated_lengths=(4,),
        batch_sizes=(2,),
        vocab_size=19,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        warmup_sequences=1,
    )
    output = tmp_path / "paired"
    summaries = bench.run_experiment(config, output, "cpu")
    checks = json.loads((output / "correctness.json").read_text())
    assert checks["passed"]
    assert checks["checks"][0]["steps_checked"] == 4
    assert len(checks["measurements"]) == 12
    with (output / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    for rep in range(6):
        pair = [r for r in rows if int(r["repetition"]) == rep]
        assert [r["mode"] for r in pair] == list(bench.MODES if rep % 2 == 0 else bench.MODES[::-1])
    summary = summaries[0]
    for metric, distribution in summary["paired_speedup"].items():
        assert len(distribution["samples"]) == 6
        for rep, ratio in enumerate(distribution["samples"]):
            pair = {r["mode"]: float(r[metric]) for r in rows if int(r["repetition"]) == rep}
            assert ratio == pytest.approx(pair["cat"] / pair["preallocated"])
    for measurement in checks["measurements"]:
        pre = measurement["prefill_cache"]
        assert pre["length"] == 3
        assert pre["capacity"] == (6 if measurement["mode"] == "preallocated" else 3)
    with pytest.raises(FileExistsError):
        bench.run_experiment(config, output, "cpu")


def test_gate_precedes_measurement(bench, tmp_path, monkeypatch):
    def fail(*args):
        raise AssertionError("correctness failure")

    monkeypatch.setattr(bench, "correctness_gate", fail)
    monkeypatch.setattr(bench, "measure_sequence", lambda *args: pytest.fail("timing started"))
    config = bench.Config(
        prompt_lengths=(2,),
        generated_lengths=(2,),
        batch_sizes=(1,),
        vocab_size=7,
        d_model=8,
        d_ff=16,
        n_heads=2,
        n_layers=1,
    )
    with pytest.raises(AssertionError, match="correctness failure"):
        bench.run_experiment(config, tmp_path / "fail", "cpu")
    assert not json.loads((tmp_path / "fail/correctness.json").read_text())["passed"]


def test_timing_boundaries(bench, monkeypatch):
    model = bench.TinyDecoderLM(vocab_size=7, d_model=8, d_ff=16, n_heads=2, n_layers=1).eval()
    events = []
    times = iter([0, 2_000_000, 3_000_000, 9_000_000])
    monkeypatch.setattr(
        bench.time, "perf_counter_ns", lambda: (events.append("clock"), next(times))[1]
    )
    monkeypatch.setattr(bench, "synchronize_device", lambda d: events.append("sync"))
    monkeypatch.setattr(bench, "memory_snapshot", lambda d: (events.append("memory"), {})[1])
    hook = model.register_forward_pre_hook(lambda *args: events.append("forward"))
    try:
        metrics, _, _ = bench.measure_sequence(
            model, torch.ones(1, 3, dtype=torch.long), 4, torch.device("cpu"), "preallocated"
        )
    finally:
        hook.remove()
    assert events == [
        "memory",
        "sync",
        "clock",
        "forward",
        "sync",
        "clock",
        "memory",
        "sync",
        "clock",
        "forward",
        "forward",
        "forward",
        "sync",
        "clock",
        "memory",
    ]
    assert metrics["prefill_ms"] == 2
    assert metrics["ms_per_decode_token"] == 2
