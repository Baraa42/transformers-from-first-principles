"""Small, explicit helpers for causal-LM training and validation."""

import math
import random
import sys
from collections.abc import Iterable, Iterator
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
) -> dict[str, float]:
    """Return average validation loss/perplexity and restore the model's mode."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.inference_mode():
            for batch_index, (input_ids, targets) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                input_ids, targets = input_ids.to(device), targets.to(device)
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


def set_seed(seed: int) -> None:
    """Seed common PRNGs; bitwise cross-device determinism is not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
