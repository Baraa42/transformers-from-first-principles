"""Checkpoint save/load helpers for explicit single-process training."""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture global PRNG state; DataLoader iterator position is intentionally excluded."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore global PRNG state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    mps_state = state.get("mps")
    if mps_state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(mps_state)


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    vocab_size: int,
    tokenizer_repo: str,
    config: dict[str, Any],
    scaler: Any | None = None,
) -> Path:
    """Atomically save complete model, optimizer, metadata, and RNG state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "vocab_size": vocab_size,
        "tokenizer_repo": tokenizer_repo,
        "config": config,
        "rng_state": capture_rng_state(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    temporary = target.with_name(f"{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    vocab_size: int,
    tokenizer_repo: str,
    config: dict[str, Any],
    scaler: Any | None = None,
) -> Path:
    """Save a standard step-named training checkpoint in ``checkpoint_dir``."""
    path = Path(checkpoint_dir) / f"step-{step:06d}.pt"
    return save_checkpoint(
        path,
        step=step,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        vocab_size=vocab_size,
        tokenizer_repo=tokenizer_repo,
        config=config,
        scaler=scaler,
    )


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_config: dict[str, Any],
    vocab_size: int,
    tokenizer_repo: str,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Validate and restore complete training state, returning checkpoint metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "model_config",
        "vocab_size",
        "tokenizer_repo",
        "rng_state",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is incomplete or legacy; missing: {sorted(missing)}")
    expected = {
        "model_config": model_config,
        "vocab_size": vocab_size,
        "tokenizer_repo": tokenizer_repo,
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(
                f"Incompatible checkpoint {field}: expected {value!r}, "
                f"got {checkpoint.get(field)!r}"
            )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
