import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import causal_lm_loss, evaluate, global_grad_norm


def test_causal_lm_loss_is_finite_scalar_and_differentiable() -> None:
    logits = torch.randn(2, 3, 7, requires_grad=True)
    targets = torch.randint(0, 7, (2, 3))
    loss = causal_lm_loss(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_evaluate_restores_training_mode_and_returns_metrics() -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    loader = DataLoader(
        TensorDataset(torch.randint(0, 10, (4, 3)), torch.randint(0, 10, (4, 3))), 2
    )
    model.train()
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert model.training
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["perplexity"])


def test_global_grad_norm_is_positive_after_backward() -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    loss = causal_lm_loss(model(torch.randint(0, 10, (2, 3))), torch.randint(0, 10, (2, 3)))
    loss.backward()
    assert global_grad_norm(model) > 0
