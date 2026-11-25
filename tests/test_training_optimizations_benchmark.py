"""Small CPU unit tests; never launches the full MPS benchmark or downloads data."""

import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def bench():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_training_optimizations.py"
    spec = importlib.util.spec_from_file_location("training_optimizations_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Dataclass annotations resolve through the module registry.
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny(bench):
    return bench.Config(
        vocab_size=11,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        batch_size=2,
        context_length=3,
        warmup_steps=1,
        measured_steps=2,
    )


def test_fixed_configuration_variants_and_order(bench):
    config = bench.Config()
    assert config.model_kwargs == {
        "vocab_size": 8192,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
    }
    assert (config.batch_size, config.context_length, config.tokens_per_step) == (16, 128, 2048)
    assert (config.learning_rate, config.weight_decay, config.max_grad_norm) == (3e-4, 0.01, 1.0)
    assert (config.warmup_steps, config.measured_steps, config.repetitions) == (10, 140, 3)
    assert [(v.fused, v.defer_norm, v.defer_loss) for v in bench.VARIANTS.values()] == [
        (False, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ]
    assert Counter(name for name, _ in bench.execution_order()) == dict.fromkeys("ABCD", 3)
    positions = Counter()
    for repetition in range(3):
        row = [name for name, rep in bench.execution_order() if rep == repetition]
        assert set(row) == set("ABCD")
        positions.update({name: position for position, name in enumerate(row, 1)})
    assert positions == {"A": 7, "B": 7, "C": 8, "D": 8}
    with pytest.raises(ValueError, match="Unknown variant"):
        bench.select_variant("E")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"measured_steps": 0},
        {"warmup_steps": -1},
        {"batch_size": True},
        {"d_model": 15},
        {"max_grad_norm": 0},
        {"learning_rate": float("nan")},
        {"repetitions": 5},
    ],
)
def test_invalid_configuration_rejected(bench, kwargs):
    with pytest.raises(ValueError):
        bench.Config(**kwargs)


def test_four_round_order_is_exactly_position_balanced(bench):
    order = bench.execution_order(4)
    assert [
        "".join(name for name, rep in order if rep == repetition) for repetition in range(4)
    ] == ["ABCD", "BCDA", "CDAB", "DABC"]
    assert len(order) == len(set(order)) == 16
    for name in "ABCD":
        positions = [index % 4 for index, (variant, _) in enumerate(order) if variant == name]
        assert sorted(positions) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="three or four"):
        bench.execution_order(5)


def test_four_round_mode_preserves_workload_and_initialization(bench, tiny):
    four = replace(tiny, repetitions=4)
    assert tiny.model_kwargs == four.model_kwargs
    assert bench.digest_tensors(bench.initial_state(tiny).values()) == bench.digest_tensors(
        bench.initial_state(four).values()
    )
    for a, b in zip(bench.make_inputs(tiny), bench.make_inputs(four), strict=True):
        assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))
    config = bench.Config(repetitions=4)
    assert (config.warmup_steps, config.measured_steps) == (10, 140)
    assert len(bench.execution_order(4)) * (config.warmup_steps + config.measured_steps) == 2400
    assert len(bench.execution_order(4)) * config.measured_steps == 2240


def test_four_round_statistics_use_even_median_and_all_raw_steps(bench):
    config = bench.Config(repetitions=4)
    rows = fake_rows(bench, config)
    assert len(rows) == 2400
    assert sum(row["warmup"] for row in rows) == 160
    summary = bench.build_summary(rows, config)
    a, d = summary["variants"]["A"], summary["variants"]["D"]
    assert [run["count"] for run in a["runs"]] == [140] * 4
    assert a["median_run_p50_ms"] == 21.5
    assert a["median_run_p95_ms"] == 21.5
    assert a["run_p50_stddev_ms"] == pytest.approx((5 / 3) ** 0.5)
    assert d["median_run_p50_ms"] == 16.5
    assert d["speedup_vs_A"] == pytest.approx(21.5 / 16.5)
    assert d["latency_reduction_vs_A_percent"] == pytest.approx(100 * 5 / 21.5)
    assert d["tokens_per_second_from_median_run_p50"] == pytest.approx(2048 * 1000 / 16.5)
    assert "Median of 4 run p50s" in summary["aggregation"]


def test_worker_cli_accepts_fourth_repetition_without_running_benchmark(bench, monkeypatch, capsys):
    def fake_worker(config, variant, repetition):
        assert config.repetitions == 4
        assert variant.name == "A"
        assert repetition == 3
        return {"simulated": True}

    monkeypatch.setattr(bench, "run_worker", fake_worker)
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--worker", "A", "--rounds", "4", "--repetition", "3"]
    )
    bench.main()
    assert json.loads(capsys.readouterr().out) == {"simulated": True}
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--worker", "A", "--rounds", "3", "--repetition", "3"]
    )
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_initialization_batches_and_optimizer_flags(bench, tiny):
    state = bench.initial_state(tiny)
    assert bench.digest_tensors(state.values()) == bench.digest_tensors(
        bench.initial_state(tiny).values()
    )
    batches = bench.make_inputs(tiny)
    repeated = bench.make_inputs(tiny)
    for (inputs, targets), (other_inputs, other_targets) in zip(batches, repeated, strict=True):
        assert torch.equal(inputs, other_inputs)
        assert torch.equal(targets, other_targets)
        assert torch.equal(inputs[:, 1:], targets[:, :-1])
    for variant in bench.VARIANTS.values():
        model, optimizer = bench.make_runtime(tiny, variant, state, torch.device("cpu"))
        assert not optimizer.state
        group = optimizer.param_groups[0]
        assert group["fused"] is variant.fused
        assert group["foreach"] is False
        assert (group["lr"], group["weight_decay"]) == (3e-4, 0.01)
        for name, parameter in model.named_parameters():
            assert parameter.dtype == torch.float32
            assert torch.equal(parameter, state[name])


def test_statistics_and_warmup_exclusion(bench, tiny):
    result = bench.timing_statistics([1.0, 2.0, 3.0, 4.0], 2)
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["iqr_ms"] == 1.5
    assert result["median_absolute_deviation_ms"] == 1.0
    assert result["aggregate_tokens_per_second"] == 800
    rows = fake_rows(bench, tiny)
    summary = bench.build_summary(rows, tiny)
    assert summary["variants"]["A"]["median_run_p50_ms"] == 21
    assert summary["variants"]["D"]["speedup_vs_A"] == pytest.approx(21 / 16)
    assert summary["variants"]["D"]["latency_reduction_vs_A_percent"] == pytest.approx(100 * 5 / 21)
    assert summary["variants"]["A"]["run_p50_stddev_ms"] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        bench.build_summary(rows[:-1], tiny)
    for invalid in ([], [0.0], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            bench.timing_statistics(invalid, 2)


def test_correctness_gate_checks_consecutive_updates_and_clipping(bench, tiny):
    report = bench.correctness_gate(tiny, torch.device("cpu"))
    assert report["passed"]
    assert not report["failures"]
    assert set(report["max_absolute_errors"]) == set(bench.TOLERANCES)
    for case in ("inactive", "active"):
        checks = [check for check in report["checks"] if check["case"] == case]
        assert len(checks) == 12
        assert {check["step"] for check in checks} == {1, 2, 3}
        assert all(check["clipping_active"] == (case == "active") for check in checks)
    assert len([check for check in report["checks"] if check["case"] == "workload"]) == 12
    json.dumps(report, allow_nan=False)


def test_correctness_gate_detects_wrong_moment_and_nonfinite_values(bench, tiny, monkeypatch):
    original = bench.optimization_step

    def corrupt(*args, **kwargs):
        loss, norm, snapshot = original(*args, **kwargs)
        if args[3].name == "B":
            moment = next(key for key in snapshot["optimizer_moments"] if key.endswith("exp_avg"))
            snapshot["optimizer_moments"][moment].add_(0.1)
            snapshot["loss"]["loss"].fill_(float("nan"))
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", corrupt)
    report = bench.correctness_gate(tiny, torch.device("cpu"), names=("A", "B"))
    assert not report["passed"]
    assert any("optimizer_moments" in failure for failure in report["failures"])
    assert any("Non-finite" in failure for failure in report["failures"])
    assert report["max_absolute_errors"]["optimizer_moments"] >= 0.09


def test_mps_unavailable_is_rejected_without_cpu_execution(bench, monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        bench.require_mps()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="CPU fallback is prohibited"):
        bench.require_mps()


@pytest.mark.parametrize("name", ["A", "B", "C", "D"])
def test_worker_timing_and_scalar_policies_on_tiny_cpu_fixture(
    bench, tiny, monkeypatch, capsys, name
):
    monkeypatch.setattr(bench, "require_mps", lambda: torch.device("cpu"))
    synchronizations = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronizations.append(True))
    monkeypatch.setattr(bench, "memory_snapshot", lambda: {"peak_allocated_bytes": None})
    clock = iter([0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000])
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(clock))
    original = bench.optimization_step

    def inspect_scalars(*args, **kwargs):
        loss, norm, snapshot = original(*args, **kwargs)
        variant = args[3]
        assert isinstance(loss, torch.Tensor) == variant.defer_loss
        assert isinstance(norm, torch.Tensor) == variant.defer_norm
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", inspect_scalars)
    result = bench.run_worker(tiny, bench.select_variant(name), 0)
    assert len(synchronizations) == 6
    assert [row["step_ms"] for row in result["steps"]] == [10.0, 10.0, 10.0]
    assert [row["warmup"] for row in result["steps"]] == [True, False, False]
    assert result["optimizer_initial_state_entries"] == 0
    assert result["memory_after"]["peak_allocated_bytes"] is None
    assert capsys.readouterr().out == ""


def fake_rows(bench, config):
    rows = []
    for index, (name, repetition) in enumerate(bench.execution_order(config.repetitions)):
        for step in range(config.warmup_steps + config.measured_steps):
            rows.append(
                {
                    "variant": name,
                    "repetition": repetition,
                    "execution_index": index,
                    "step": step + 1,
                    "warmup": step < config.warmup_steps,
                    "step_ms": 999.0
                    if step < config.warmup_steps
                    else {"A": 20, "B": 18, "C": 16, "D": 15}[name] + repetition,
                }
            )
    return rows


@pytest.mark.parametrize("rounds", [3, 4])
def test_artifact_schemas_with_simulated_workers(bench, tiny, tmp_path, monkeypatch, rounds):
    tiny = replace(tiny, repetitions=rounds)
    monkeypatch.setattr(bench, "Config", lambda **kwargs: tiny)
    monkeypatch.setattr(bench, "require_mps", lambda: torch.device("cpu"))
    monkeypatch.setattr(bench, "command_output", lambda args: None)
    monkeypatch.setattr(
        bench,
        "correctness_gate",
        lambda *args: {
            "schema_version": 1,
            "experiment": bench.LABEL,
            "passed": True,
            "failures": [],
        },
    )
    state_hash = bench.digest_tensors(bench.initial_state(tiny).values())
    batch_hash = bench.digest_tensors(value for batch in bench.make_inputs(tiny) for value in batch)

    def fake_worker(command, **kwargs):
        assert command[0] == sys.executable
        assert int(command[command.index("--rounds") + 1]) == rounds
        name = command[command.index("--worker") + 1]
        repetition = int(command[command.index("--repetition") + 1])
        selected = [
            dict(row)
            for row in fake_rows(bench, tiny)
            if row["variant"] == name and row["repetition"] == repetition
        ]
        for row in selected:
            del row["execution_index"]
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "variant": name,
                    "repetition": repetition,
                    "steps": selected,
                    "initial_state_sha256": state_hash,
                    "batches_sha256": batch_hash,
                }
            )
        )

    monkeypatch.setattr(bench.subprocess, "run", fake_worker)
    output = tmp_path / "results"
    bench.run_experiment(output, rounds=rounds)
    assert {path.name for path in output.iterdir()} == {
        "metadata.json",
        "correctness.json",
        "steps.csv",
        "summary.json",
    }
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["schema_version"] == 1
    assert "not an exact historical" in metadata["experiment"]
    assert metadata["precision"] == "fp32"
    assert metadata["device"] == "mps"
    assert set(metadata["variants"]) == set("ABCD")
    assert len(metadata["workers"]) == 4 * rounds
    assert metadata["config"]["repetitions"] == rounds
    assert metadata["expected_workers"] == 4 * rounds
    assert metadata["expected_raw_steps"] == 4 * rounds * 3
    assert metadata["expected_measured_steps"] == 4 * rounds * 2
    assert {(worker["variant"], worker["repetition"]) for worker in metadata["workers"]} == {
        (name, repetition) for name in "ABCD" for repetition in range(rounds)
    }
    assert metadata["batches_sha256"] == batch_hash
    assert (output / "steps.csv").read_text().splitlines()[0] == ",".join(bench.STEP_FIELDS)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["variants"]) == set("ABCD")
    assert all(len(value["runs"]) == rounds for value in summary["variants"].values())
    with pytest.raises(FileExistsError):
        bench.run_experiment(output, rounds=rounds)


def test_gate_failure_stops_before_timed_workers(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "require_mps", lambda: torch.device("cpu"))
    monkeypatch.setattr(bench, "command_output", lambda args: None)
    monkeypatch.setattr(
        bench,
        "correctness_gate",
        lambda *args: {
            "schema_version": 1,
            "passed": False,
            "failures": ["injected numerical mismatch"],
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Timed worker must not start after a failed correctness gate")

    monkeypatch.setattr(bench.subprocess, "run", forbidden)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="no timed workers started"):
        bench.run_experiment(output)
    assert json.loads((output / "correctness.json").read_text())["passed"] is False
    assert not (output / "steps.csv").exists()
    assert not (output / "summary.json").exists()


def arithmetic_fixture(bench, *, parameter=4.0, gradient=0.5):
    config = bench.Config()
    before, grad = torch.tensor([parameter]), torch.tensor([gradient])
    reference = bench.adamw_reference_and_budget(before, grad, {}, config)
    state = {key: reference[key].float() for key in ("exp_avg", "exp_avg_sq")}
    state["step"] = torch.tensor(1.0)
    return config, before, grad, reference, state


def test_rounding_aware_oracle_accepts_neighboring_fp32_endpoints(bench):
    config, before, grad, reference, state = arithmetic_fixture(bench)
    a = reference["parameter"].float()
    c = torch.nextafter(a, torch.full_like(a, float("inf")))
    atol, rtol = bench.TOLERANCES["updates"]
    assert not torch.allclose(c - before, a - before, atol=atol, rtol=rtol)
    for after in (a, c):
        metrics, failures = bench.check_optimizer_arithmetic(before, grad, {}, after, state, config)
        assert not failures
        assert metrics["parameter"]["max_error_budget_ratio"] <= 1


@pytest.mark.parametrize("parameter", [0.0, 1e-12, 0.1, 4.0])
def test_arithmetic_oracle_rejects_wrong_updates_at_small_and_large_parameter_scales(
    bench, parameter
):
    config, before, grad, reference, state = arithmetic_fixture(bench, parameter=parameter)
    wrong = reference["parameter"].float() + 1e-5
    _, failures = bench.check_optimizer_arithmetic(before, grad, {}, wrong, state, config)
    assert any("parameter" in failure for failure in failures)


def test_zero_gradient_weight_decay_is_checked_against_fp64(bench):
    config, before, grad, reference, state = arithmetic_fixture(bench, gradient=0.0)
    assert reference["parameter"].item() == pytest.approx(4.0 * (1 - 3e-4 * 0.01))
    assert not bench.check_optimizer_arithmetic(
        before, grad, {}, reference["parameter"].float(), state, config
    )[1]
    _, failures = bench.check_optimizer_arithmetic(before, grad, {}, before + 1e-4, state, config)
    assert failures


@pytest.mark.parametrize("field", ["exp_avg", "exp_avg_sq"])
def test_arithmetic_oracle_rejects_corrupted_moments(bench, field):
    config, before, grad, reference, state = arithmetic_fixture(bench)
    state[field] = state[field] + 0.01
    _, failures = bench.check_optimizer_arithmetic(
        before, grad, {}, reference["parameter"].float(), state, config
    )
    assert any(field in failure for failure in failures)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_arithmetic_oracle_rejects_nonfinite_parameters(bench, value):
    config, before, grad, _, state = arithmetic_fixture(bench)
    _, failures = bench.check_optimizer_arithmetic(
        before, grad, {}, torch.tensor([value]), state, config
    )
    assert any("Non-finite" in failure for failure in failures)


@pytest.mark.parametrize(
    "category", ["loss", "gradients", "clipped_gradients", "norm", "optimizer_moments"]
)
def test_fused_gate_retains_strict_gradient_clipping_norm_and_moment_checks(
    bench, tiny, monkeypatch, category
):
    original = bench.optimization_step

    def corrupt(*args, **kwargs):
        loss, norm, snapshot = original(*args, **kwargs)
        if args[3].name == "C":
            next(iter(snapshot[category].values())).add_(0.1)
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", corrupt)
    report = bench.correctness_gate(tiny, torch.device("cpu"), names=("A", "C"))
    assert not report["passed"]
    assert any(category in failure for failure in report["failures"])


def test_a_b_keeps_original_strict_update_comparison(bench, tiny, monkeypatch):
    original = bench.optimization_step

    def corrupt(*args, **kwargs):
        loss, norm, snapshot = original(*args, **kwargs)
        if args[3].name == "B":
            next(iter(snapshot["updates"].values())).add_(1e-5)
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", corrupt)
    report = bench.correctness_gate(tiny, torch.device("cpu"), names=("A", "B"))
    assert not report["passed"]
    assert any("updates" in failure for failure in report["failures"])


def test_fused_gate_rejects_materially_wrong_parameter_output(bench, tiny, monkeypatch):
    original = bench.optimization_step

    def corrupt(*args, **kwargs):
        loss, norm, snapshot = original(*args, **kwargs)
        if args[3].name == "C":
            next(iter(snapshot["parameters"].values())).add_(1e-3)
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", corrupt)
    report = bench.correctness_gate(tiny, torch.device("cpu"), names=("A", "C"))
    assert not report["passed"]
    assert any(
        "FP64 arithmetic budget exceeded: parameter" in error for error in report["failures"]
    )


def test_correctness_only_mode_never_starts_workers(bench, tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "Config", lambda **kwargs: tiny)
    monkeypatch.setattr(bench, "require_mps", lambda: torch.device("cpu"))
    monkeypatch.setattr(bench, "command_output", lambda args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Correctness-only mode must not start a worker")

    monkeypatch.setattr(bench.subprocess, "run", forbidden)
    output = tmp_path / "gate"
    bench.run_experiment(output, correctness_only=True)
    report = json.loads((output / "correctness.json").read_text())
    assert report["passed"]
    assert report["arithmetic_policy"]["version"] == 2
    assert report["trajectories"]
    assert {path.name for path in output.iterdir()} == {"metadata.json", "correctness.json"}
    assert (
        json.loads((output / "metadata.json").read_text())["execution_mode"] == "correctness-only"
    )


def test_bad_shapes_and_nonfinite_gradients_fail_closed(bench):
    config, before, grad, reference, state = arithmetic_fixture(bench)
    with pytest.raises(ValueError, match="shape"):
        bench.check_optimizer_arithmetic(
            before, grad, {}, reference["parameter"].float().reshape(1, 1), state, config
        )
    with pytest.raises(ValueError, match="Non-finite"):
        bench.adamw_reference_and_budget(before, grad * float("nan"), {}, config)


def test_arithmetic_gate_rejects_fast_math(bench, tiny, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    with pytest.raises(ValueError, match="fast-math disabled"):
        bench.correctness_gate(tiny, torch.device("cpu"))


def test_independent_ab_drift_is_reported_separately_and_triggers_common_input_replay(
    bench, tiny, monkeypatch
):
    original = bench.optimization_step
    calls = 0

    def simulated_independent_drift(*args, **kwargs):
        nonlocal calls
        loss, norm, snapshot = original(*args, **kwargs)
        if args[3].name == "B":
            calls += 1
            # Each case has 3 common-start B steps followed by 3 independent B steps.
            # Alter only recorded independent-trajectory distances, never common-start results.
            if calls % 6 in (4, 5, 0):
                snapshot["updates"]["token_embedding.weight"].add_(1e-5)
                snapshot["parameters"]["token_embedding.weight"].add_(5e-7)
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", simulated_independent_drift)
    report = bench.correctness_gate(tiny, torch.device("cpu"), names=("A", "B"))
    assert report["passed"]
    assert not report["ab_trajectory_reproducibility_passed"]
    assert len(report["ab_common_input_replays"]) == 3
    assert all(replay["passed"] for replay in report["ab_common_input_replays"])
    assert all(replay["step"] == 1 for replay in report["ab_common_input_replays"])
    rows = [row for row in report["trajectories"] if row["variant"] == "B"]
    assert all(row["reproducibility_failures"] for row in rows)
    assert all(row["max_update_difference"] > 2e-7 for row in rows)
    assert all(row["rms_update_difference"] > 0 for row in rows)
    assert all("max_adam_moment_difference" in row for row in rows)
    json.dumps(report, allow_nan=False)


def test_common_input_ab_replay_still_rejects_wrong_updates(bench, tiny, monkeypatch):
    original = bench.optimizer_replay_snapshot

    def corrupt(config, variant, before, gradients, state, device):
        snapshot = original(config, variant, before, gradients, state, device)
        if variant.name == "B":
            name = "token_embedding.weight"
            snapshot["parameters"][name].add_(5e-7)
            snapshot["updates"][name] = snapshot["parameters"][name] - before[name]
        return snapshot

    monkeypatch.setattr(bench, "optimizer_replay_snapshot", corrupt)
    before = bench.initial_state(tiny)
    gradients = {name: torch.zeros_like(value) for name, value in before.items()}
    old_state = dict.fromkeys(before, {})
    replay = bench.common_input_ab_replay(tiny, before, gradients, old_state, torch.device("cpu"))
    assert not replay["passed"]
    assert any("Tolerance exceeded: updates" in error for error in replay["failures"])


def test_independent_trajectory_nonfinite_updates_still_fail_gate(bench, tiny, monkeypatch):
    original = bench.optimization_step
    calls = 0

    def nonfinite_trajectory(*args, **kwargs):
        nonlocal calls
        loss, norm, snapshot = original(*args, **kwargs)
        if args[3].name == "B":
            calls += 1
            if calls % 6 == 4:
                snapshot["updates"]["token_embedding.weight"].fill_(float("nan"))
        return loss, norm, snapshot

    monkeypatch.setattr(bench, "optimization_step", nonfinite_trajectory)
    report = bench.correctness_gate(tiny, torch.device("cpu"), names=("A", "B"))
    assert not report["passed"]
    assert any("trajectory/" in error and "Non-finite" in error for error in report["failures"])
    json.dumps(report, allow_nan=False)
