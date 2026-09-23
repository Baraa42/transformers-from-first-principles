"""Small, explicit helpers for causal-LM training and validation."""

import math
import random
import sys
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for logits ``(B, T, V)`` and next-token targets ``(B, T)``."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("Expected logits (B, T, V) and targets (B, T)")
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits batch/time dimensions must match targets")
    if targets.dtype != torch.long:
        raise ValueError("targets must have torch.long dtype")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def infinite_batches(loader: Iterable[Any]) -> Iterator[Any]:
    """Yield loader batches forever, recreating its iterator after each epoch."""
    while True:
        yield from loader


def evaluate(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int | None = None,
    precision: str = "fp32",
) -> dict[str, float]:
    """Return average validation loss/perplexity and restore the model's mode."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    precision = validate_precision(precision)
    try:
        with torch.inference_mode():
            for batch_index, (input_ids, targets) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                input_ids, targets = input_ids.to(device), targets.to(device)
                with autocast_context(device, precision):
                    loss = causal_lm_loss(model(input_ids), targets)
                n_tokens = targets.numel()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Validation loader produced no batches")
    loss = total_loss / total_tokens
    perplexity = math.exp(loss) if loss < math.log(sys.float_info.max) else math.inf
    return {"loss": loss, "perplexity": perplexity}


def global_grad_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of available gradients for logging only."""
    squared_norm = sum(
        gradient.detach().pow(2).sum().item()
        for parameter in model.parameters()
        if (gradient := parameter.grad) is not None
    )
    return math.sqrt(squared_norm)


def clip_or_measure_grad_norm(model: torch.nn.Module, max_grad_norm: float | None) -> float:
    """Return pre-clipping global norm and clip in place when configured."""
    if max_grad_norm is None:
        return global_grad_norm(model)
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be strictly positive or null")
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()


def validate_grad_accum_steps(value: object) -> int:
    """Validate and return the number of microbatches per optimizer update."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("grad_accum_steps must be an integer >= 1")
    return value


def validate_precision(value: object) -> str:
    """Validate the supported autocast precision names."""
    if value not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    return value


def autocast_context(device: torch.device, precision: str) -> Any:
    """Return FP32 no-op or the requested public PyTorch autocast context."""
    precision = validate_precision(precision)
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{precision} autocast is unsupported on {device.type}") from error


def create_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler | None:
    """Create an enabled public GradScaler for FP16, or no scaler otherwise."""
    precision = validate_precision(precision)
    if precision != "fp16":
        return None
    try:
        scaler = torch.amp.GradScaler(device.type)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"fp16 GradScaler is unsupported on {device.type}") from error
    if not scaler.is_enabled():
        raise ValueError(f"fp16 GradScaler is unsupported on {device.type}")
    return scaler


def validate_precision_state(precision: str, scaler: Any | None) -> None:
    """Reject precision/scaler combinations that cannot occur in a valid run."""
    if (precision == "fp16") != (scaler is not None):
        raise ValueError("fp16 requires a GradScaler; fp32 and bf16 must not use one")


def validate_autocast_support(device: torch.device, precision: str) -> None:
    """Execute a representative operation and verify the requested autocast dtype."""
    precision = validate_precision(precision)
    try:
        model = torch.nn.Linear(2, 2).to(device)
        inputs = torch.ones(2, 2, device=device)
        with autocast_context(device, precision):
            outputs = model(inputs)
        expected_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]
        if outputs.dtype != expected_dtype:
            raise RuntimeError(f"operation produced {outputs.dtype}, not {expected_dtype}")
    except Exception as error:
        raise ValueError(
            f"{precision} autocast is unsupported on {device.type}: {error}"
        ) from error


def validate_grad_scaler_support(device: torch.device) -> None:
    """Execute a real FP16 GradScaler backward/unscale/step/update lifecycle."""
    try:
        model = torch.nn.Linear(2, 2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with autocast_context(device, "fp16"):
            loss = model(torch.ones(2, 2, device=device)).float().square().mean()
        scaler = create_grad_scaler(device, "fp16")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    except Exception as error:
        raise ValueError(f"fp16 GradScaler is unsupported on {device.type}: {error}") from error


def validate_precision_support(device: torch.device, precision: str) -> None:
    """Reject precision modes that fail a real backend capability check."""
    precision = validate_precision(precision)
    try:
        validate_autocast_support(device, precision)
        if precision == "fp16":
            validate_grad_scaler_support(device)
    except ValueError as error:
        raise ValueError(f"{precision} AMP is unsupported on {device.type}: {error}") from error


def complete_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    precision: str,
    scaler: Any | None,
    max_grad_norm: float | None,
) -> tuple[float, bool]:
    """Unscale, measure/clip once, and perform exactly one optimizer attempt."""
    validate_precision_state(precision, scaler)
    if precision == "fp16":
        scaler.unscale_(optimizer)
    grad_norm = clip_or_measure_grad_norm(model, max_grad_norm)
    did_step = optimizer_step(precision=precision, optimizer=optimizer, scaler=scaler)
    return grad_norm, did_step


def optimizer_step(*, precision: str, optimizer: torch.optim.Optimizer, scaler: Any | None) -> bool:
    """Perform one update and return whether it actually occurred."""
    validate_precision_state(precision, scaler)
    if precision != "fp16":
        optimizer.step()
        return True
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return not scaler_step_was_skipped(old_scale, scaler.get_scale())


def scaler_step_was_skipped(old_scale: float, new_scale: float) -> bool:
    """Use public GradScaler scale backoff as the overflow/skip signal."""
    return new_scale < old_scale


def set_seed(seed: int) -> None:
    """Seed common PRNGs; bitwise cross-device determinism is not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work at a wall-clock timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda``, or ``mps`` to an available device."""
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(requested)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a deliberately plain YAML mapping."""
    with Path(path).open() as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    return config
