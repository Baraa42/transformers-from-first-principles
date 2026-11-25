"""Short A/C numerical diagnostic only: no performance timing or gate modifications."""

import argparse
import hashlib
import json
from pathlib import Path

import benchmark_training_optimizations as bench
import torch

EMBEDDING = "token_embedding.weight"
CASES = {"inactive": 0.01, "active": 100.0, "workload": 1.0}


def maximum(tensor):
    return tensor.abs().max().item() if tensor.numel() else None


def magnitude_statistics(tensor):
    values = tensor.detach().double().abs().flatten()
    if not values.numel():
        return {"median": None, "p95": None, "max": None, "rms": None}
    quantiles = torch.quantile(values, values.new_tensor([0.5, 0.95]))
    return {
        "median": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "max": values.max().item(),
        "rms": values.square().mean().sqrt().item(),
    }


def embedding_metrics(before_a, after_a, before_c, after_c, relative_floor=1e-6):
    """Compare observed FP32 deltas AND exact differences of their rounded endpoints."""
    for value in (before_a, after_a, before_c, after_c):
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError("Metrics require CPU snapshots of FP32 parameters")
        if value.shape != before_a.shape or not torch.isfinite(value).all():
            raise ValueError("Metrics require matching, finite parameters")
    delta_a, delta_c = after_a - before_a, after_c - before_c
    exact_a = after_a.double() - before_a.double()
    exact_c = after_c.double() - before_c.double()
    difference = delta_c.double() - delta_a.double()
    meaningful = exact_a.abs() >= relative_floor
    relative = (exact_c[meaningful] - exact_a[meaningful]).abs() / exact_a[meaningful].abs()
    a, r = bench.TOLERANCES["updates"]
    update_failures = ~torch.isclose(delta_c, delta_a, atol=a, rtol=r)
    a, r = bench.TOLERANCES["parameters"]
    parameter_failures = ~torch.isclose(after_c, after_a, atol=a, rtol=r)
    # ULP spacing of the reference parameter, including exponent-boundary asymmetry.
    up = torch.nextafter(before_a, torch.full_like(before_a, float("inf")))
    down = torch.nextafter(before_a, torch.full_like(before_a, -float("inf")))
    spacing = torch.maximum(
        (up.double() - before_a.double()).abs(), (before_a.double() - down.double()).abs()
    )
    parameter_difference = (after_c.double() - after_a.double()).abs()
    indices = torch.topk(difference.abs().flatten(), min(5, difference.numel())).indices
    worst = []
    for index in indices.tolist():

        def element(value):
            return value.flatten()[index].item()

        worst.append(
            {
                "flat_index": index,
                "before_A": element(before_a),
                "before_C": element(before_c),
                "after_A": element(after_a),
                "after_C": element(after_c),
                "update_A": element(delta_a),
                "update_C": element(delta_c),
                "update_difference": element(difference),
                "parameter_ulp": element(spacing),
                "current_update_check_failed": bool(element(update_failures)),
            }
        )
    return {
        "elements": before_a.numel(),
        "max_absolute_parameter_difference": maximum(parameter_difference),
        "max_absolute_starting_parameter_difference": maximum(
            before_c.double() - before_a.double()
        ),
        "max_absolute_update_difference": maximum(difference),
        "rms_update_difference": difference.square().mean().sqrt().item(),
        "update_magnitudes_A": magnitude_statistics(exact_a),
        "update_magnitudes_C": magnitude_statistics(exact_c),
        "relative_update_floor": relative_floor,
        "non_negligible_update_elements": int(meaningful.sum()),
        "relative_update_difference": magnitude_statistics(relative),
        "elements_exceeding_current_update_tolerance": int(update_failures.sum()),
        "elements_exceeding_current_parameter_tolerance": int(parameter_failures.sum()),
        "max_subtraction_rounding_error_A": maximum(delta_a.double() - exact_a),
        "max_subtraction_rounding_error_C": maximum(delta_c.double() - exact_c),
        "max_update_difference_with_float64_subtraction": maximum(exact_c - exact_a),
        "parameter_difference_in_reference_ulps": magnitude_statistics(
            parameter_difference / spacing
        ),
        "update_difference_in_reference_ulps": magnitude_statistics(difference / spacing),
        "worst_update_elements": worst,
    }


def optimizer_state(model, optimizer):
    parameter = dict(model.named_parameters())[EMBEDDING]
    return {
        name: value.detach().cpu().clone() for name, value in optimizer.state[parameter].items()
    }


def reference_adamw64(parameter, gradient, state, config):
    """One mathematical AdamW step from common FP32 inputs, evaluated on CPU in FP64.

    This is an arithmetic reference, not a bitwise specification for FP32 kernels.
    The gradient is already globally clipped; weight decay is decoupled.
    """
    p, g = parameter.double(), gradient.double()
    m = state.get("exp_avg", torch.zeros_like(parameter)).double()
    v = state.get("exp_avg_sq", torch.zeros_like(parameter)).double()
    step = int(state.get("step", torch.tensor(0)).item()) + 1
    m_next = 0.9 * m + (1 - 0.9) * g
    v_next = 0.999 * v + (1 - 0.999) * g.square()
    adaptive = (m_next / (1 - 0.9**step)) / ((v_next / (1 - 0.999**step)).sqrt() + 1e-8)
    delta = -config.learning_rate * config.weight_decay * p - config.learning_rate * adaptive
    return p + delta, delta, {"exp_avg": m_next, "exp_avg_sq": v_next}


def replay_embedding(parameter, gradient, state, config, device, *, fused):
    """Isolate the optimizer: identical parameter, clipped gradient AND moments for A/C."""
    p = torch.nn.Parameter(parameter.to(device).clone())
    optimizer = torch.optim.AdamW(
        [p],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=fused,
        amsgrad=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    if state:
        # AdamW's unfused, noncapturable step counter lives on CPU; fused uses device state.
        optimizer.state[p] = {
            name: value.to(device if name != "step" or fused else "cpu").clone()
            for name, value in state.items()
        }
    p.grad = gradient.to(device).clone()
    optimizer.step()
    return p.detach().cpu().clone(), {
        name: value.detach().cpu().clone() for name, value in optimizer.state[p].items()
    }


def common_input_probe(parameter, gradient, state, config, device, relative_floor):
    a, moments_a = replay_embedding(parameter, gradient, state, config, device, fused=False)
    c, moments_c = replay_embedding(parameter, gradient, state, config, device, fused=True)
    ideal, ideal_delta, ideal_moments = reference_adamw64(parameter, gradient, state, config)
    metrics = embedding_metrics(parameter, a, parameter, c, relative_floor)
    metrics["optimizer_moment_max_differences"] = {
        key: maximum(moments_c[key].double() - moments_a[key].double()) for key in moments_a
    }
    for name, value, moments in (("A", a, moments_a), ("C", c, moments_c)):
        observed_delta = value.double() - parameter.double()
        metrics[f"{name}_vs_fp64_reference"] = {
            "max_parameter_error": maximum(value.double() - ideal),
            "max_update_error": maximum(observed_delta - ideal_delta),
            "rms_update_error": (observed_delta - ideal_delta).square().mean().sqrt().item(),
            "moment_max_errors": {
                key: maximum(moments[key].double() - expected)
                for key, expected in ideal_moments.items()
            },
        }
    zero = gradient == 0
    metrics["zero_gradient_elements"] = int(zero.sum())
    metrics["max_parameter_difference_on_zero_gradients"] = maximum((c.double() - a.double())[zero])
    return metrics


def run_diagnostic(config, device, *, steps=10, relative_floor=1e-6):
    if not 1 <= steps <= 20 or steps > config.warmup_steps + config.measured_steps:
        raise ValueError("Use 1..20 steps within the available synthetic batches")
    if not 0 < relative_floor < float("inf"):
        raise ValueError("relative_floor must be finite and positive")
    initial, batches = bench.initial_state(config), bench.make_inputs(config)
    records = []
    for case, scale in CASES.items():
        runtimes = {
            name: bench.make_runtime(config, bench.select_variant(name), initial, device)
            for name in ("A", "C")
        }
        for step in range(steps):
            before = {
                name: dict(model.named_parameters())[EMBEDDING].detach().cpu().clone()
                for name, (model, _) in runtimes.items()
            }
            common_state = optimizer_state(*runtimes["A"])
            snapshots = {}
            for name, (model, optimizer) in runtimes.items():
                _, _, snapshots[name] = bench.optimization_step(
                    model,
                    optimizer,
                    batches[step],
                    bench.select_variant(name),
                    config,
                    capture=True,
                    loss_scale=scale,
                )
            a, c = snapshots["A"], snapshots["C"]
            metrics = embedding_metrics(
                before["A"],
                a["parameters"][EMBEDDING],
                before["C"],
                c["parameters"][EMBEDDING],
                relative_floor,
            )
            metrics["optimizer_moment_max_differences"] = {
                key.removeprefix(EMBEDDING + "."): maximum(
                    c["optimizer_moments"][key].double() - value.double()
                )
                for key, value in a["optimizer_moments"].items()
                if key.startswith(EMBEDDING + ".")
            }
            errors, failures = bench.compare_snapshots(c, a)
            records.append(
                {
                    "case": case,
                    "step": step + 1,
                    "loss_scale": scale,
                    "forward_loss_A": a["loss"]["loss"].item(),
                    "forward_loss_C": c["loss"]["loss"].item(),
                    "gradient_norm_A": a["norm"]["norm"].item(),
                    "gradient_norm_C": c["norm"]["norm"].item(),
                    "trajectory": metrics,
                    "original_gate_max_errors": errors,
                    "original_gate_failures": failures,
                    "common_input_replay": common_input_probe(
                        before["A"],
                        a["clipped_gradients"][EMBEDDING],
                        common_state,
                        config,
                        device,
                        relative_floor,
                    ),
                }
            )
        del runtimes
    trends = {}
    for case in CASES:
        selected = [row["trajectory"] for row in records if row["case"] == case]
        values = [row["max_absolute_parameter_difference"] for row in selected]
        trends[case] = {
            "max_parameter_difference_by_step": values,
            "final_parameter_difference": values[-1],
            "max_observed_parameter_difference": max(values),
            "rms_update_difference_by_step": [row["rms_update_difference"] for row in selected],
            "update_tolerance_failure_count_by_step": [
                row["elements_exceeding_current_update_tolerance"] for row in selected
            ],
            "interpretation": "Descriptive trajectory only; no automatic equivalence verdict",
        }
    return {
        "schema_version": 1,
        "diagnostic": "A/C numerical diagnostic; no performance timing",
        "device": str(device),
        "steps_per_case": steps,
        "relative_floor": relative_floor,
        "unchanged_gate_tolerances": bench.TOLERANCES,
        "reference_policy": "CPU FP64 single-step AdamW from common A parameter/gradient/state; "
        "not an independent long-horizon FP64 training trajectory",
        "decay_only_common_input_probe": common_input_probe(
            initial[EMBEDDING],
            torch.zeros_like(initial[EMBEDDING]),
            {},
            config,
            device,
            relative_floor,
        ),
        "records": records,
        "trajectories": trends,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New diagnostic output directory"
    )
    parser.add_argument("--steps", type=int, default=10, choices=range(1, 21))
    parser.add_argument("--relative-floor", type=float, default=1e-6)
    args = parser.parse_args()
    device = bench.require_mps()
    args.output.mkdir(parents=True, exist_ok=False)
    config = bench.Config()
    metadata = bench.collect_metadata(config)
    metadata["diagnostic_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["diagnostic_steps"] = args.steps
    metadata["diagnostic_relative_floor"] = args.relative_floor
    bench.write_json(args.output / "metadata.json", metadata)
    report = run_diagnostic(config, device, steps=args.steps, relative_floor=args.relative_floor)
    bench.write_json(args.output / "diagnostic.json", report)
    print(json.dumps(report["trajectories"], indent=2, allow_nan=False))
    print(f"Detailed diagnostic: {(args.output / 'diagnostic.json').resolve()}")


if __name__ == "__main__":
    main()
