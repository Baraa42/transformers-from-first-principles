"""Controlled synthetic reconstruction, NOT an exact historical benchmark reproduction.

Only this standalone runner selects optimizer backends/scalar-read policies. Production
training is unchanged. Run on MPS from a normal macOS Terminal; no datasets are required.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import causal_lm_loss

ROOT = Path(__file__).resolve().parents[1]
LABEL = "Controlled synthetic reconstruction; not an exact historical reproduction"
SCHEMA_VERSION = 1
# Fixed before collecting results; a mismatch fails the gate, never relaxes these limits.
TOLERANCES = {
    "loss": (1e-6, 1e-5),
    "gradients": (2e-6, 1e-5),
    "norm": (2e-6, 1e-5),
    "clipped_gradients": (2e-6, 1e-5),
    "parameters": (2e-6, 1e-5),
    "updates": (2e-7, 1e-4),
    "optimizer_moments": (2e-6, 1e-5),
}
STEP_FIELDS = ("variant", "repetition", "execution_index", "step", "warmup", "step_ms")


@dataclass(frozen=True)
class Config:
    vocab_size: int = 8192
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    batch_size: int = 16
    context_length: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    measured_steps: int = 140
    repetitions: int = 3
    model_seed: int = 42
    batch_seed: int = 43

    def __post_init__(self):
        for name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "batch_size",
            "context_length",
            "measured_steps",
            "repetitions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "model_seed", "batch_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.repetitions not in (3, 4):
            raise ValueError("The protocol requires three or four repetitions")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("Heads must divide width and have even dimension")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")

    @property
    def model_kwargs(self):
        return {
            name: getattr(self, name)
            for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff")
        }

    @property
    def tokens_per_step(self):
        return self.batch_size * self.context_length


@dataclass(frozen=True)
class Variant:
    name: str
    fused: bool
    defer_norm: bool
    defer_loss: bool


VARIANTS = {
    "A": Variant("A", False, False, False),
    "B": Variant("B", False, True, False),
    "C": Variant("C", True, True, False),
    "D": Variant("D", True, True, True),
}


def select_variant(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}")
    return VARIANTS[name]


def execution_order(repetitions=3):
    # Preserve the original three-round order; the four-round Latin square balances
    # positions exactly (not all predecessor/carryover effects).
    if repetitions not in (3, 4):
        raise ValueError("Execution order requires three or four rounds")
    rows = ("ABCD", "BCDA", "DACB") if repetitions == 3 else ("ABCD", "BCDA", "CDAB", "DABC")
    return [(name, repetition) for repetition, row in enumerate(rows) for name in row]


def require_mps():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK; CPU fallback is prohibited")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this benchmark has no CPU fallback")
    device = torch.device("mps")
    try:
        result = torch.ones(2, device=device).square().sum()
        torch.mps.synchronize()
        if result.cpu().item() != 2:
            raise RuntimeError("Unexpected MPS preflight result")
    except Exception as error:
        raise RuntimeError(f"MPS execution preflight failed: {error}") from error
    return device


def make_inputs(config):
    generator = torch.Generator(device="cpu").manual_seed(config.batch_seed)
    chunks = torch.randint(
        config.vocab_size,
        (config.warmup_steps + config.measured_steps, config.batch_size, config.context_length + 1),
        generator=generator,
    )
    # Contiguous CPU tensors avoid variant-dependent strides or random generation in timing.
    return [(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()) for chunk in chunks]


def initial_state(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.model_seed)
        model = TinyDecoderLM(**config.model_kwargs).to(dtype=torch.float32, device="cpu")
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def make_runtime(config, variant, state, device):
    model = TinyDecoderLM(**config.model_kwargs).to(dtype=torch.float32, device=device)
    model.load_state_dict(state, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=variant.fused,
        capturable=False,
        differentiable=False,
    )
    return model, optimizer


def gradient_snapshot(model):
    return {
        name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
    }


def optimization_step(model, optimizer, batch, variant, config, *, capture=False, loss_scale=1.0):
    """Same clipped update in all variants; only scalar extraction/backend differ."""
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = (value.to(next(model.parameters()).device) for value in batch)
    loss = causal_lm_loss(model(inputs), targets)
    reported_loss = loss.detach() if variant.defer_loss else loss.item()
    (loss * loss_scale).backward()
    snapshot = {}
    if capture:
        snapshot["gradients"] = gradient_snapshot(model)
        before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    reported_norm = norm.detach() if variant.defer_norm else norm.item()
    if capture:
        snapshot["clipped_gradients"] = gradient_snapshot(model)
    optimizer.step()
    if capture:
        snapshot["loss"] = {"loss": torch.as_tensor(reported_loss).detach().cpu().clone()}
        snapshot["norm"] = {"norm": torch.as_tensor(reported_norm).detach().cpu().clone()}
        snapshot["parameters"] = {
            name: value.detach().cpu().clone() for name, value in model.named_parameters()
        }
        snapshot["updates"] = {
            name: value - before[name] for name, value in snapshot["parameters"].items()
        }
        snapshot["optimizer_moments"] = {
            f"{name}.{key}": value.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            for key, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor)
        }
    return reported_loss, reported_norm, snapshot


def compare_snapshots(actual, expected, *, strict_categories=None):
    errors, failures = {}, []
    for category, (atol, rtol) in TOLERANCES.items():
        if actual[category].keys() != expected[category].keys():
            raise ValueError(f"Snapshot keys differ for {category}")
        errors[category] = 0.0
        for name, reference in expected[category].items():
            value = actual[category][name]
            if value.shape != reference.shape:
                raise ValueError(f"Snapshot shapes differ: {category}.{name}")
            if value.dtype != reference.dtype:
                raise ValueError(f"Snapshot dtypes differ: {category}.{name}")
            if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
                failures.append(f"Non-finite tensor: {category}.{name}")
                continue
            errors[category] = max(errors[category], (value - reference).abs().max().item())
            if (strict_categories is None or category in strict_categories) and not torch.allclose(
                value, reference, atol=atol, rtol=rtol
            ):
                failures.append(f"Tolerance exceeded: {category}.{name}")
    return errors, failures


# Arithmetic policy v2: an operation-count error envelope, NOT a fitted absolute tolerance.
# u is FP32 unit roundoff; eta is one FP32 subnormal quantum. Assumptions: finite FP32
# inputs, nonnegative second moments, round-to-nearest arithmetic without fast-math,
# and small integer step counters. Bounds cover both fused and unfused evaluation orders.
# gamma_8 covers moment lerp/product/add and scalar conversion, including cancellation.
# gamma_(2t+4) envelopes t-step scalar powers, constant conversion and bias subtraction.
# Four denominator operations cover division/sqrt/epsilon addition; two cover lr scaling.
# Four parameter operations cover lr*wd, 1-lr*wd, parameter multiplication, final subtraction.
# Intervals propagate cancellation and epsilon sensitivity instead of comparing ULP counts.
ARITHMETIC_POLICY = {
    "version": 2,
    "unit_roundoff": 2.0**-24,
    "subnormal_quantum": 2.0**-149,
    "moment_operations": 8,
    "denominator_operations": 4,
    "learning_rate_operations": 2,
    "parameter_operations": 4,
    "bias_operations": "2 * step + 4",
    "assumptions": "Finite FP32, nonnegative v, small integer steps, no MPS fast-math; "
    "conservative operation-count envelope, not a certified backend error bound",
}


def gamma(operations):
    product = operations * ARITHMETIC_POLICY["unit_roundoff"]
    if product >= 1:
        raise ValueError("Roundoff operation count is outside the policy domain")
    return product / (1 - product)


def interval_divide(low, high, denominator_low, denominator_high):
    if (denominator_low <= 0).any():
        raise ValueError("Unresolved denominator interval; arithmetic check fails closed")
    endpoints = torch.stack(
        (
            low / denominator_low,
            low / denominator_high,
            high / denominator_low,
            high / denominator_high,
        )
    )
    return endpoints.amin(0), endpoints.amax(0)


def adamw_reference_and_budget(parameter, gradient, state, config):
    """Independent CPU FP64 AdamW arithmetic plus a scale-aware FP32 error envelope."""
    if (
        parameter.dtype != torch.float32
        or gradient.dtype != torch.float32
        or parameter.device.type != "cpu"
        or gradient.device.type != "cpu"
    ):
        raise ValueError("Arithmetic oracle requires CPU snapshots of FP32 inputs")
    if parameter.shape != gradient.shape:
        raise ValueError("Parameter/gradient shapes differ")
    if state and state.keys() != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("Incomplete or unsupported optimizer state")
    if any(value.dtype != torch.float32 or value.device.type != "cpu" for value in state.values()):
        raise ValueError("Optimizer state must contain CPU FP32 snapshots")
    p, g = parameter.double(), gradient.double()
    m = state.get("exp_avg", torch.zeros_like(parameter)).double()
    v = state.get("exp_avg_sq", torch.zeros_like(parameter)).double()
    if m.shape != p.shape or v.shape != p.shape:
        raise ValueError("Optimizer moment shapes differ")
    if not all(torch.isfinite(value).all() for value in (p, g, m, v)) or (v < 0).any():
        raise ValueError("Non-finite inputs or negative second moment")
    previous_step = float(state.get("step", torch.tensor(0)).item())
    if not math.isfinite(previous_step) or previous_step < 0 or not previous_step.is_integer():
        raise ValueError("Invalid optimizer step counter")
    step = int(previous_step) + 1
    if step > 1024:
        raise ValueError("Arithmetic policy is limited to short correctness probes")
    eta = ARITHMETIC_POLICY["subnormal_quantum"]
    m_next = 0.9 * m + (1 - 0.9) * g
    v_next = 0.999 * v + (1 - 0.999) * g.square()
    m_error = gamma(8) * (m.abs() + g.abs()) + 8 * eta
    v_error = gamma(8) * (v.abs() + g.square()) + 8 * eta
    c1, c2 = 1 - 0.9**step, 1 - 0.999**step
    c1_error = gamma(2 * step + 4) * (1 + 0.9**step)
    c2_error = gamma(2 * step + 4) * (1 + 0.999**step)
    if c1 <= c1_error or c2 <= c2_error:
        raise ValueError("Unresolved bias-correction interval")
    m_low, m_high = interval_divide(
        m_next - m_error,
        m_next + m_error,
        torch.tensor(c1 - c1_error, dtype=torch.float64),
        torch.tensor(c1 + c1_error, dtype=torch.float64),
    )
    v_low = (v_next - v_error).clamp_min(0) / (c2 + c2_error)
    v_high = (v_next + v_error) / (c2 - c2_error)
    denominator_low, denominator_high = v_low.sqrt() + 1e-8, v_high.sqrt() + 1e-8
    denominator_error = gamma(4) * denominator_high + 4 * eta
    denominator_low = denominator_low - denominator_error
    denominator_high = denominator_high + denominator_error
    adaptive_low, adaptive_high = interval_divide(m_low, m_high, denominator_low, denominator_high)
    adaptive_low, adaptive_high = (
        config.learning_rate * adaptive_low,
        config.learning_rate * adaptive_high,
    )
    lr_error = gamma(2) * torch.maximum(adaptive_low.abs(), adaptive_high.abs()) + 2 * eta
    adaptive = config.learning_rate * (m_next / c1) / ((v_next / c2).sqrt() + 1e-8)
    adaptive_error = (
        torch.maximum((adaptive_low - adaptive).abs(), (adaptive_high - adaptive).abs()) + lr_error
    )
    decay = config.learning_rate * config.weight_decay * p
    delta = -decay - adaptive
    parameter_error = gamma(4) * (p.abs() + decay.abs() + adaptive.abs()) + 4 * eta
    return {
        "parameter": p + delta,
        "delta": delta,
        "parameter_budget": parameter_error + adaptive_error,
        "exp_avg": m_next,
        "exp_avg_sq": v_next,
        "exp_avg_budget": m_error,
        "exp_avg_sq_budget": v_error,
        "step": step,
    }


def check_optimizer_arithmetic(before, gradient, old_state, after, new_state, config):
    reference = adamw_reference_and_budget(before, gradient, old_state, config)
    failures, metrics = [], {}
    for name, actual, expected, budget in (
        ("parameter", after, reference["parameter"], reference["parameter_budget"]),
        ("exp_avg", new_state["exp_avg"], reference["exp_avg"], reference["exp_avg_budget"]),
        (
            "exp_avg_sq",
            new_state["exp_avg_sq"],
            reference["exp_avg_sq"],
            reference["exp_avg_sq_budget"],
        ),
    ):
        if actual.shape != expected.shape or actual.dtype != torch.float32:
            raise ValueError(f"Arithmetic shape/dtype mismatch: {name}")
        if not torch.isfinite(actual).all():
            failures.append(f"Non-finite arithmetic output: {name}")
            continue
        if name == "exp_avg_sq" and (actual < 0).any():
            failures.append("Negative second moment")
        error = (actual.double() - expected).abs()
        if not torch.isfinite(budget).all():
            raise ValueError("Non-finite arithmetic budget")
        exceeded = error > budget
        metrics[name] = {
            "max_absolute_error": error.max().item(),
            "rms_error": error.square().mean().sqrt().item(),
            "max_error_budget_ratio": (error / budget).max().item(),
            "exceeding_elements": int(exceeded.sum()),
            "max_budget": budget.max().item(),
        }
        if exceeded.any():
            failures.append(f"FP64 arithmetic budget exceeded: {name}")
    if (
        new_state["step"].numel() != 1
        or new_state["step"].dtype != torch.float32
        or new_state["step"].item() != reference["step"]
    ):
        failures.append("Incorrect optimizer step counter")
    # Exact subtraction of rounded endpoints: do NOT apply the old tiny-delta tolerance.
    # Its error versus the ideal delta is the same as the endpoint error for common p.
    delta_error = (after.double() - before.double() - reference["delta"]).abs()
    if torch.isfinite(delta_error).all():
        metrics["observed_delta"] = {
            "max_absolute_error": delta_error.max().item(),
            "rms_error": delta_error.square().mean().sqrt().item(),
        }
    return metrics, failures


def named_optimizer_state(model, optimizer):
    return {
        name: {key: value.detach().cpu().clone() for key, value in optimizer.state[p].items()}
        for name, p in model.named_parameters()
    }


def restore_named_optimizer_state(model, optimizer, state, variant):
    for name, parameter in model.named_parameters():
        if state[name]:
            optimizer.state[parameter] = {
                key: value.to(parameter.device if key != "step" or variant.fused else "cpu").clone()
                for key, value in state[name].items()
            }


def arithmetic_snapshot_checks(before, gradients, old_state, snapshot, config):
    metrics, failures = {}, []
    for name, parameter in snapshot["parameters"].items():
        new_state = {
            key: snapshot["optimizer_moments"][name + "." + key]
            for key in ("step", "exp_avg", "exp_avg_sq")
        }
        metrics[name], errors = check_optimizer_arithmetic(
            before[name], gradients[name], old_state[name], parameter, new_state, config
        )
        failures.extend(f"{name}: {error}" for error in errors)
    return metrics, failures


def optimizer_replay_snapshot(config, variant, before, gradients, old_state, device):
    """One optimizer-only step; every supplied parameter, gradient and moment is cloned."""
    model, optimizer = make_runtime(config, variant, before, device)
    restore_named_optimizer_state(model, optimizer, old_state, variant)
    for name, parameter in model.named_parameters():
        parameter.grad = gradients[name].to(device).clone()
    optimizer.step()
    state = named_optimizer_state(model, optimizer)
    parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    return {
        "parameters": parameters,
        "updates": {name: value - before[name] for name, value in parameters.items()},
        "optimizer_moments": {
            name + "." + key: value
            for name, fields in state.items()
            for key, value in fields.items()
        },
    }


def common_input_ab_replay(config, before, gradients, old_state, device):
    snapshots = {
        name: optimizer_replay_snapshot(
            config, select_variant(name), before, gradients, old_state, device
        )
        for name in ("A", "B")
    }
    # No relaxation: original parameter/update/moment tolerances on identical optimizer inputs.
    comparable = {
        name: {category: snapshot.get(category, {}) for category in TOLERANCES}
        for name, snapshot in snapshots.items()
    }
    errors, failures = compare_snapshots(comparable["B"], comparable["A"])
    arithmetic = {}
    for name, snapshot in snapshots.items():
        arithmetic[name], arithmetic_failures = arithmetic_snapshot_checks(
            before, gradients, old_state, snapshot, config
        )
        failures.extend(f"{name}: {error}" for error in arithmetic_failures)
    return {
        "passed": not failures,
        "max_absolute_errors": errors,
        "failures": failures,
        "identical_starting_parameters_gradients_and_moments": True,
        "arithmetic": arithmetic,
    }


def rms_snapshot_difference(actual, expected, category):
    total, count = 0.0, 0
    for name, value in actual[category].items():
        difference = value.double() - expected[category][name].double()
        if not torch.isfinite(difference).all():
            return None
        total += difference.square().sum().item()
        count += difference.numel()
    return math.sqrt(total / count) if count else 0.0


def correctness_gate(config, device, *, names=("A", "B", "C", "D")):
    if not names or names[0] != "A":
        raise ValueError("Correctness gate requires baseline A first")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") == "1":
        raise ValueError("Rounding-aware arithmetic policy requires MPS fast-math disabled")
    state, batches = initial_state(config), make_inputs(config)
    if len(batches) < 3:
        raise ValueError("Correctness gate needs at least three synthetic batches")
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": LABEL,
        "passed": True,
        "arithmetic_policy": ARITHMETIC_POLICY,
        "comparison_policy": "Strict common-start loss/gradient/clipping/moment checks; A/B "
        "retain all original checks. Fused updates use FP64 arithmetic budgets.",
        "trajectory_policy": "Independent trajectories: descriptive drift, "
        "not arithmetic equivalence",
        "trajectory_comparison_policy_version": 3,
        "trajectories": [],
        "ab_trajectory_reproducibility_passed": True,
        "ab_common_input_replays": [],
        "tolerances": {key: {"atol": a, "rtol": r} for key, (a, r) in TOLERANCES.items()},
        "clipping_cases": {"inactive": 0.01, "active": 100.0, "workload": 1.0},
        "loss_scaling_policy": "Correctness-only multipliers; timing always uses 1.0",
        "checks": [],
        "max_absolute_errors": dict.fromkeys(TOLERANCES, 0.0),
        "failures": [],
    }
    for case, scale in report["clipping_cases"].items():
        baseline_model, baseline_optimizer = make_runtime(
            config, select_variant("A"), state, device
        )
        for step in range(3):
            before = {
                name: value.detach().cpu().clone()
                for name, value in baseline_model.state_dict().items()
            }
            old_state = named_optimizer_state(baseline_model, baseline_optimizer)
            reference = None
            for name in names:
                if name == "A":
                    model, optimizer = baseline_model, baseline_optimizer
                else:
                    model, optimizer = make_runtime(config, select_variant(name), before, device)
                    restore_named_optimizer_state(model, optimizer, old_state, select_variant(name))
                _, _, snapshot = optimization_step(
                    model,
                    optimizer,
                    batches[step],
                    select_variant(name),
                    config,
                    capture=True,
                    loss_scale=scale,
                )
                if reference is None:
                    reference = snapshot
                strict = None if name in ("A", "B") else set(TOLERANCES) - {"parameters", "updates"}
                errors, failures = compare_snapshots(snapshot, reference, strict_categories=strict)
                arithmetic, arithmetic_failures = arithmetic_snapshot_checks(
                    before, snapshot["clipped_gradients"], old_state, snapshot, config
                )
                failures.extend(arithmetic_failures)
                # A second, optimizer-only replay uses EXACTLY A's clipped gradients for every
                # backend. No forward-pass or accumulated-state differences enter this check.
                replay_snapshot = optimizer_replay_snapshot(
                    config,
                    select_variant(name),
                    before,
                    reference["clipped_gradients"],
                    old_state,
                    device,
                )
                replay_metrics, replay_failures = arithmetic_snapshot_checks(
                    before, reference["clipped_gradients"], old_state, replay_snapshot, config
                )
                failures.extend("common-input replay: " + error for error in replay_failures)
                norm = snapshot["norm"]["norm"].item()
                active = norm > config.max_grad_norm
                if case != "workload" and active != (case == "active"):
                    failures.append(f"Expected {case} clipping, got pre-clipping norm {norm}")
                for key, value in errors.items():
                    report["max_absolute_errors"][key] = max(
                        report["max_absolute_errors"][key], value
                    )
                report["checks"].append(
                    {
                        "case": case,
                        "step": step + 1,
                        "variant": name,
                        "norm": norm if math.isfinite(norm) else None,
                        "clipping_active": active,
                        "max_absolute_errors": errors,
                        "failures": failures,
                        "actual_step_arithmetic": arithmetic,
                        "common_input_arithmetic": replay_metrics,
                    }
                )
                report["failures"].extend(f"{case}/{step + 1}/{name}: {f}" for f in failures)
        # No resets here: this measures accumulation, separately from the common-start checks.
        runtimes = {
            name: make_runtime(config, select_variant(name), state, device) for name in names
        }
        ab_replayed = False
        for step in range(min(10, len(batches))):
            reference = None
            if "B" in names and not ab_replayed:
                ab_before = {
                    key: value.detach().cpu().clone()
                    for key, value in runtimes["A"][0].state_dict().items()
                }
                ab_old_state = named_optimizer_state(*runtimes["A"])
            for name in names:
                _, _, snapshot = optimization_step(
                    *runtimes[name],
                    batches[step],
                    select_variant(name),
                    config,
                    capture=True,
                    loss_scale=scale,
                )
                if reference is None:
                    reference = snapshot
                # Different independent trajectories no longer have identical optimizer inputs.
                # Shape/dtype/finiteness still gate; distance thresholds describe reproducibility.
                max_errors, errors = compare_snapshots(snapshot, reference, strict_categories=set())
                report["failures"].extend(
                    f"trajectory/{case}/{step + 1}/{name}: {e}" for e in errors
                )
                reproducibility_failures = []
                if name == "B":
                    _, reproducibility_failures = compare_snapshots(snapshot, reference)
                    if reproducibility_failures:
                        report["ab_trajectory_reproducibility_passed"] = False
                        if not ab_replayed:
                            replay = common_input_ab_replay(
                                config,
                                ab_before,
                                reference["clipped_gradients"],
                                ab_old_state,
                                device,
                            )
                            replay.update({"case": case, "step": step + 1})
                            report["ab_common_input_replays"].append(replay)
                            report["failures"].extend(
                                f"common-input-AB/{case}/{step + 1}: {error}"
                                for error in replay["failures"]
                            )
                            ab_replayed = True
                differences = torch.cat(
                    [
                        (value.double() - reference["parameters"][key].double()).flatten()
                        for key, value in snapshot["parameters"].items()
                    ]
                )
                report["trajectories"].append(
                    {
                        "case": case,
                        "step": step + 1,
                        "variant": name,
                        "max_parameter_difference": differences.abs().max().item()
                        if torch.isfinite(differences).all()
                        else None,
                        "rms_parameter_difference": differences.square().mean().sqrt().item()
                        if torch.isfinite(differences).all()
                        else None,
                        "loss_difference": abs(
                            snapshot["loss"]["loss"].item() - reference["loss"]["loss"].item()
                        )
                        if torch.isfinite(snapshot["loss"]["loss"]).all()
                        and torch.isfinite(reference["loss"]["loss"]).all()
                        else None,
                        "arithmetic_verdict": "not inferred from accumulated drift",
                        "max_absolute_errors": max_errors,
                        "max_update_difference": max_errors["updates"],
                        "rms_update_difference": rms_snapshot_difference(
                            snapshot, reference, "updates"
                        ),
                        "max_gradient_difference": max_errors["gradients"],
                        "max_clipped_gradient_difference": max_errors["clipped_gradients"],
                        "max_adam_moment_difference": max_errors["optimizer_moments"],
                        "reproducibility_failures": reproducibility_failures,
                    }
                )
        del runtimes
    report["passed"] = not report["failures"]
    return report


def digest_tensors(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def memory_snapshot():
    values = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        try:
            values[name] = int(getattr(torch.mps, name)())
        except (AttributeError, RuntimeError):
            values[name] = None
    values["peak_allocated_bytes"] = None
    values["peak_note"] = "No peak API used; boundary counters are not peak memory"
    return values


def run_worker(config, variant, repetition):
    device = require_mps()
    state, batches = initial_state(config), make_inputs(config)
    model, optimizer = make_runtime(config, variant, state, device)
    initial_optimizer_entries = len(optimizer.state)
    memory_before = memory_snapshot()
    rows = []
    for step, batch in enumerate(batches):
        torch.mps.synchronize()
        started = time.perf_counter_ns()
        last_loss, last_norm, _ = optimization_step(model, optimizer, batch, variant, config)
        torch.mps.synchronize()
        milliseconds = (time.perf_counter_ns() - started) / 1_000_000
        rows.append(
            {
                "variant": variant.name,
                "repetition": repetition,
                "step": step + 1,
                "warmup": step < config.warmup_steps,
                "step_ms": milliseconds,
            }
        )
    # All host conversions, memory queries, JSON writing and stdout occur outside timers.
    return {
        "variant": variant.name,
        "repetition": repetition,
        "steps": rows,
        "initial_state_sha256": digest_tensors(state.values()),
        "batches_sha256": digest_tensors(value for batch in batches for value in batch),
        "optimizer_initial_state_entries": initial_optimizer_entries,
        "final_loss": float(last_loss),
        "final_norm": float(last_norm),
        "memory_before": memory_before,
        "memory_after": memory_snapshot(),
    }


def timing_statistics(samples, tokens_per_step):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Timings must be non-empty, finite and positive")
    median = float(np.median(samples))
    return {
        "count": len(samples),
        "p50_ms": median,
        "p95_ms": float(np.percentile(samples, 95)),
        "iqr_ms": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
        "median_absolute_deviation_ms": float(np.median(np.abs(np.array(samples) - median))),
        "tokens_per_second_from_p50": tokens_per_step * 1000 / median,
        "aggregate_tokens_per_second": len(samples) * tokens_per_step * 1000 / sum(samples),
    }


def build_summary(rows, config):
    variants = {}
    for name in VARIANTS:
        runs = []
        for repetition in range(config.repetitions):
            selected = [
                row for row in rows if row["variant"] == name and row["repetition"] == repetition
            ]
            if len(selected) != config.warmup_steps + config.measured_steps:
                raise ValueError("Incomplete run")
            if sorted(row["step"] for row in selected) != list(range(1, len(selected) + 1)):
                raise ValueError("Duplicate or missing step indices")
            if any(row["warmup"] != (row["step"] <= config.warmup_steps) for row in selected):
                raise ValueError("Incorrect warmup flags")
            samples = [row["step_ms"] for row in selected if not row["warmup"]]
            if len(samples) != config.measured_steps:
                raise ValueError("Incorrect warmup exclusion")
            runs.append(
                {"repetition": repetition, **timing_statistics(samples, config.tokens_per_step)}
            )
        medians = [run["p50_ms"] for run in runs]
        variants[name] = {
            "runs": runs,
            "median_run_p50_ms": statistics.median(medians),
            "median_run_p95_ms": statistics.median(run["p95_ms"] for run in runs),
            "run_p50_min_ms": min(medians),
            "run_p50_max_ms": max(medians),
            "run_p50_stddev_ms": statistics.stdev(medians),
            "run_p50_iqr_ms": float(np.percentile(medians, 75) - np.percentile(medians, 25)),
        }
    baseline = variants["A"]["median_run_p50_ms"]
    for result in variants.values():
        median = result["median_run_p50_ms"]
        result.update(
            {
                "latency_reduction_vs_A_percent": 100 * (1 - median / baseline),
                "speedup_vs_A": baseline / median,
                "tokens_per_second_from_median_run_p50": config.tokens_per_step * 1000 / median,
            }
        )
        for run, reference in zip(result["runs"], variants["A"]["runs"], strict=True):
            run["latency_reduction_vs_A_percent"] = 100 * (1 - run["p50_ms"] / reference["p50_ms"])
            run["speedup_vs_A"] = reference["p50_ms"] / run["p50_ms"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": LABEL,
        "variants": variants,
        "aggregation": f"Median of {config.repetitions} run p50s; p95s are per-run, not pooled",
    }


def command_output(arguments):
    try:
        return subprocess.check_output(
            arguments, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_metadata(config):
    hardware = {}
    raw = command_output(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"])
    if raw:
        data = json.loads(raw)
        # Deliberately omit serial numbers, UUIDs and all unrestricted environment values.
        hardware = {
            "hardware": [
                {
                    key: item.get(key)
                    for key in (
                        "machine_model",
                        "chip_type",
                        "number_processors",
                        "physical_memory",
                    )
                }
                for item in data.get("SPHardwareDataType", [])
            ],
            "graphics": [
                {key: item.get(key) for key in ("sppci_model", "sppci_cores", "spdisplays_metal")}
                for item in data.get("SPDisplaysDataType", [])
            ],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "precision": "fp32",
        "device": "mps",
        "python": sys.version,
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "pytorch_git": torch.version.git_version,
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "hardware": hardware,
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "backend_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            )
        },
        "variants": {name: asdict(value) for name, value in VARIANTS.items()},
        "optimizer": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "foreach": False,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "initial_state": "fresh/empty for every run",
        },
        "execution_order": execution_order(config.repetitions),
        "order_policy": (
            "Fully position-balanced: ABCD, BCDA, CDAB, DABC; every variant once per position"
            if config.repetitions == 4
            else "Near-balanced: three rounds, position sums A/B/C/D=7/7/8/8"
        ),
        "expected_workers": 4 * config.repetitions,
        "expected_raw_steps": 4
        * config.repetitions
        * (config.warmup_steps + config.measured_steps),
        "expected_measured_steps": 4 * config.repetitions * config.measured_steps,
        "timing": "perf_counter_ns; MPS sync before/after WHOLE step; includes CPU-batch "
        "transfer, zero_grad, forward/loss, selected scalar reads, backward, "
        "clipping and update; no per-component synchronization",
        "warmup_policy": {
            "excluded_updates_per_run": config.warmup_steps,
            "measured_updates_per_run": config.measured_steps,
            "retain_warmup_samples": True,
        },
        "worker_processes": "Fresh Python process for each variant/repetition",
        "historical_limitations": "Synthetic data, current code/hardware, whole-step rather "
        "than historical per-component synchronization",
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(output, *, correctness_only=False, rounds=3):
    config = Config(repetitions=rounds)
    device = require_mps()  # Fail before producing misleading CPU benchmark artifacts.
    output.mkdir(parents=True, exist_ok=False)
    metadata = collect_metadata(config)
    metadata["execution_mode"] = "correctness-only" if correctness_only else "benchmark"
    write_json(output / "metadata.json", metadata)
    try:
        correctness = correctness_gate(config, device)
    except Exception as error:
        correctness = {
            "schema_version": SCHEMA_VERSION,
            "experiment": LABEL,
            "passed": False,
            "failures": [f"{type(error).__name__}: {error}"],
        }
    write_json(output / "correctness.json", correctness)
    if not correctness["passed"]:
        raise RuntimeError(
            "Correctness gate failed; no timed workers started; see correctness.json"
        )
    if correctness_only:
        return
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()  # Release the parent's correctness-gate allocator cache.
    rows, workers = [], []
    expected_state = digest_tensors(initial_state(config).values())
    expected_batches = digest_tensors(value for batch in make_inputs(config) for value in batch)
    metadata.update(
        {
            "initial_state_sha256": expected_state,
            "batches_sha256": expected_batches,
            "workers": workers,
        }
    )
    write_json(output / "metadata.json", metadata)
    with (output / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
        writer.writeheader()
        for index, (name, repetition) in enumerate(execution_order(config.repetitions)):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--worker",
                    name,
                    "--repetition",
                    str(repetition),
                    "--rounds",
                    str(config.repetitions),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
            )
            worker = json.loads(completed.stdout)
            if (
                worker["variant"] != name
                or worker["repetition"] != repetition
                or worker["initial_state_sha256"] != expected_state
                or worker["batches_sha256"] != expected_batches
            ):
                raise RuntimeError("Worker initialization or synthetic batches differ")
            for row in worker.pop("steps"):
                row["execution_index"] = index
                writer.writerow(row)
                rows.append(row)
            handle.flush()
            workers.append(worker)
            write_json(output / "metadata.json", metadata)
    metadata.update(
        {
            "initial_state_sha256": expected_state,
            "batches_sha256": expected_batches,
            "workers": workers,
        }
    )
    write_json(output / "metadata.json", metadata)
    write_json(output / "summary.json", build_summary(rows, config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory (never overwritten)")
    parser.add_argument(
        "--rounds",
        type=int,
        choices=(3, 4),
        default=3,
        help="Three original rounds or four fully position-balanced rounds",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run arithmetic checks and descriptive drift only; never start timed workers",
    )
    parser.add_argument("--worker", choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    parser.add_argument(
        "--repetition", type=int, choices=range(4), default=0, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker and args.correctness_only:
        parser.error("--correctness-only cannot be combined with --worker")
    if args.worker and args.repetition >= args.rounds:
        parser.error("--repetition must be smaller than --rounds")
    if args.worker:
        print(
            json.dumps(
                run_worker(
                    Config(repetitions=args.rounds), select_variant(args.worker), args.repetition
                ),
                allow_nan=False,
            )
        )
    elif args.output is None:
        parser.error("--output is required")
    else:
        run_experiment(args.output, correctness_only=args.correctness_only, rounds=args.rounds)
        mode = "Correctness checks passed" if args.correctness_only else "Completed benchmark"
        print(f"{mode}: {LABEL}\nArtifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
