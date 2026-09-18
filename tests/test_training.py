import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import (
    causal_lm_loss,
    clip_or_measure_grad_norm,
    evaluate,
    global_grad_norm,
)


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


def test_clipping_below_threshold_leaves_gradients_unchanged() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    before = model.weight.grad.clone()

    norm = clip_or_measure_grad_norm(model, max_grad_norm=10.0)

    assert norm == 5.0
    assert torch.equal(model.weight.grad, before)


def test_clipping_above_threshold_scales_all_gradients_by_one_factor() -> None:
    model = torch.nn.Linear(2, 1, bias=True)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    model.bias.grad = torch.tensor([12.0])
    original = [parameter.grad.clone() for parameter in model.parameters()]

    pre_clip_norm = clip_or_measure_grad_norm(model, max_grad_norm=5.0)
    post_clip_norm = global_grad_norm(model)
    scale = 5.0 / 13.0

    assert pre_clip_norm == 13.0
    assert post_clip_norm == pytest.approx(5.0)
    for before, parameter in zip(original, model.parameters(), strict=True):
        assert torch.allclose(parameter.grad, before * scale)


def test_disabled_clipping_preserves_gradients_and_returns_original_norm() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    before = model.weight.grad.clone()

    norm = clip_or_measure_grad_norm(model, max_grad_norm=None)

    assert norm == 5.0
    assert torch.equal(model.weight.grad, before)
