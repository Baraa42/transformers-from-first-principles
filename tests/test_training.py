import copy
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
    validate_grad_accum_steps,
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


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, "2"])
def test_grad_accum_steps_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="grad_accum_steps"):
        validate_grad_accum_steps(value)


def test_gradient_accumulation_matches_one_larger_batch_update() -> None:
    torch.manual_seed(0)
    large_batch_model = torch.nn.Linear(2, 3)
    accumulated_model = copy.deepcopy(large_batch_model)
    large_batch_optimizer = torch.optim.SGD(large_batch_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    targets = torch.tensor([0, 1, 2, 0])

    large_batch_optimizer.zero_grad()
    large_batch_loss = torch.nn.functional.cross_entropy(large_batch_model(inputs), targets)
    large_batch_loss.backward()
    large_batch_optimizer.step()

    accumulated_optimizer.zero_grad()
    for microbatch_inputs, microbatch_targets in zip(
        inputs.chunk(2), targets.chunk(2), strict=True
    ):
        microbatch_loss = torch.nn.functional.cross_entropy(
            accumulated_model(microbatch_inputs), microbatch_targets
        )
        (microbatch_loss / 2).backward()
    accumulated_optimizer.step()

    for large_parameter, accumulated_parameter in zip(
        large_batch_model.parameters(), accumulated_model.parameters(), strict=True
    ):
        assert torch.allclose(large_parameter, accumulated_parameter)


def test_grad_accum_steps_one_matches_an_unscaled_backward() -> None:
    torch.manual_seed(1)
    direct_model = torch.nn.Linear(2, 3)
    accumulation_model = copy.deepcopy(direct_model)
    inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    targets = torch.tensor([1, 2])

    torch.nn.functional.cross_entropy(direct_model(inputs), targets).backward()
    (torch.nn.functional.cross_entropy(accumulation_model(inputs), targets) / 1).backward()

    for direct_parameter, accumulated_parameter in zip(
        direct_model.parameters(), accumulation_model.parameters(), strict=True
    ):
        assert torch.equal(direct_parameter.grad, accumulated_parameter.grad)


def test_gradient_clipping_runs_after_accumulation() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()

    for _ in range(2):
        loss = torch.nn.functional.mse_loss(model(torch.ones(1, 1)), torch.zeros(1, 1))
        (loss / 2).backward()

    pre_clip_norm = clip_or_measure_grad_norm(model, max_grad_norm=1.0)

    assert pre_clip_norm == pytest.approx(2.0)
    assert global_grad_norm(model) == pytest.approx(1.0)
