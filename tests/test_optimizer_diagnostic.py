"""CPU tests of diagnostic mathematics; never run the MPS performance benchmark."""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_training_optimizers")


def test_one_ulp_endpoint_difference_can_fail_update_check_without_subtraction_error(diagnostic):
    before = torch.tensor([4.0])
    a = before - 3e-4
    c = torch.nextafter(a, torch.full_like(a, float("inf")))
    result = diagnostic.embedding_metrics(before, a, before, c)
    assert result["elements_exceeding_current_update_tolerance"] == 1
    assert result["elements_exceeding_current_parameter_tolerance"] == 0
    assert result["max_subtraction_rounding_error_A"] == 0
    assert result["max_subtraction_rounding_error_C"] == 0
    assert result["max_absolute_update_difference"] == 2**-22
    assert result["max_update_difference_with_float64_subtraction"] == 2**-22
    assert result["non_negligible_update_elements"] == 1


def test_relative_statistics_exclude_negligible_updates_and_report_rms(diagnostic):
    before = torch.zeros(4)
    a = torch.tensor([0.0, 1e-8, 1e-4, 1e-2])
    c = a + torch.tensor([1e-9, 1e-9, 1e-6, 1e-5])
    result = diagnostic.embedding_metrics(before, a, before, c, relative_floor=1e-5)
    assert result["non_negligible_update_elements"] == 2
    expected = ((c.double()[2:] - a.double()[2:]) / a.double()[2:]).abs().max().item()
    assert result["relative_update_difference"]["max"] == expected
    assert result["rms_update_difference"] == pytest.approx(
        (c.double() - a.double()).square().mean().sqrt().item()
    )
    assert result["update_magnitudes_A"]["max"] == a.double().max().item()
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="FP32"):
        diagnostic.embedding_metrics(before.double(), a, before, c)


def test_fp64_adamw_reference_uses_bias_correction_epsilon_and_decoupled_decay(diagnostic):
    config = diagnostic.bench.Config()
    p, gradient = torch.tensor([2.0]), torch.tensor([0.5])
    after, delta, moments = diagnostic.reference_adamw64(p, gradient, {}, config)
    expected = -3e-4 * 0.01 * 2.0 - 3e-4 * 0.5 / (0.5 + 1e-8)
    assert delta.dtype == after.dtype == torch.float64
    assert delta.item() == pytest.approx(expected, rel=1e-12)
    assert after.item() == pytest.approx(2.0 + expected, rel=1e-12)
    assert moments["exp_avg"].item() == pytest.approx(0.05)
    assert moments["exp_avg_sq"].item() == pytest.approx(0.00025)


def test_common_input_replay_restores_existing_moments_and_does_not_mutate_inputs(diagnostic):
    config = diagnostic.bench.Config()
    p, gradient = torch.tensor([4.0, -2.0, 0.1]), torch.tensor([0.3, -0.2, 0.0])
    state = {
        "step": torch.tensor(4.0),
        "exp_avg": torch.full_like(p, 0.05),
        "exp_avg_sq": torch.full_like(p, 0.01),
    }
    saved = copy.deepcopy(state)
    report = diagnostic.common_input_probe(p, gradient, state, config, torch.device("cpu"), 1e-6)
    for name in state:
        assert torch.equal(state[name], saved[name])
    assert torch.equal(p, torch.tensor([4.0, -2.0, 0.1]))
    assert report["zero_gradient_elements"] == 1
    assert set(report["optimizer_moment_max_differences"]) == {"step", "exp_avg", "exp_avg_sq"}
    for name in ("A", "C"):
        assert report[f"{name}_vs_fp64_reference"]["max_parameter_error"] < 1e-6
    json.dumps(report, allow_nan=False)


def test_short_trajectory_diagnostic_preserves_gate_and_covers_clipping(diagnostic, monkeypatch):
    bench = diagnostic.bench
    config = bench.Config(
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
    saved_tolerances = copy.deepcopy(bench.TOLERANCES)

    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostic must not invoke performance workers")

    monkeypatch.setattr(bench, "run_worker", forbidden)
    monkeypatch.setattr(bench, "run_experiment", forbidden)
    report = diagnostic.run_diagnostic(config, torch.device("cpu"), steps=2)
    assert len(report["records"]) == 6
    assert set(report["trajectories"]) == {"inactive", "active", "workload"}
    for record in report["records"]:
        assert "original_gate_failures" in record
        assert "common_input_replay" in record
        if record["case"] != "workload":
            assert (record["gradient_norm_A"] > 1.0) == (record["case"] == "active")
    assert report["decay_only_common_input_probe"]["zero_gradient_elements"] == 88
    assert bench.TOLERANCES == saved_tolerances
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="1..20"):
        diagnostic.run_diagnostic(config, torch.device("cpu"), steps=21)


def test_diagnostic_cli_rejects_unavailable_mps_without_creating_output(
    diagnostic, monkeypatch, tmp_path
):
    def unavailable():
        raise RuntimeError("MPS is unavailable")

    monkeypatch.setattr(diagnostic.bench, "require_mps", unavailable)
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(sys, "argv", ["diagnose_training_optimizers.py", "--output", str(output)])
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        diagnostic.main()
    assert not output.exists()
