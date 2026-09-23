import copy
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import transformers_from_scratch.training as training
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import (
    autocast_context,
    causal_lm_loss,
    clip_or_measure_grad_norm,
    create_grad_scaler,
    evaluate,
    global_grad_norm,
    infinite_batches,
    optimizer_step,
    scaler_step_was_skipped,
    set_seed,
    synchronize_device,
    validate_grad_accum_steps,
    validate_precision,
    validate_precision_state,
    validate_precision_support,
)


def test_synchronize_device_cpu_is_a_no_op(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(training.torch.cuda, "synchronize", lambda device: calls.append("cuda"))
    monkeypatch.setattr(training.torch.mps, "synchronize", lambda: calls.append("mps"))

    synchronize_device(torch.device("cpu"))

    assert calls == []


def test_synchronize_device_dispatches_to_cuda(monkeypatch) -> None:
    devices: list[torch.device] = []
    monkeypatch.setattr(training.torch.cuda, "synchronize", devices.append)

    device = torch.device("cuda")
    synchronize_device(device)

    assert devices == [device]


def test_synchronize_device_dispatches_to_mps(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(training.torch.mps, "synchronize", lambda: calls.append("mps"))

    synchronize_device(torch.device("mps"))

    assert calls == ["mps"]


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


class EpochIterable:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield f"epoch-{self.iterations}-first"
        yield f"epoch-{self.iterations}-second"


def test_infinite_batches_starts_a_fresh_iteration_after_each_epoch() -> None:
    loader = EpochIterable()
    batches = infinite_batches(loader)

    assert [next(batches) for _ in range(4)] == [
        "epoch-1-first",
        "epoch-1-second",
        "epoch-2-first",
        "epoch-2-second",
    ]
    assert loader.iterations == 2


def test_evaluate_weights_unequal_batches_by_target_tokens() -> None:
    class LookupLogits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("table", torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]))

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.table[input_ids]

    model = LookupLogits()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[0], [1], [2]]),
            torch.tensor([[0], [1], [0]]),
        ),
        batch_size=2,
        drop_last=False,
    )
    batch_losses: list[float] = []
    batch_tokens: list[int] = []
    for input_ids, targets in loader:
        batch_losses.append(causal_lm_loss(model(input_ids), targets).item())
        batch_tokens.append(targets.numel())

    expected = sum(loss * tokens for loss, tokens in zip(batch_losses, batch_tokens)) / sum(
        batch_tokens
    )
    naive_mean = sum(batch_losses) / len(batch_losses)
    metrics = evaluate(model, loader, torch.device("cpu"))

    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["loss"] != pytest.approx(naive_mean)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_validate_precision_accepts_supported_modes(precision: str) -> None:
    assert validate_precision(precision) == precision


def test_validate_precision_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="precision"):
        validate_precision("float8")


def test_fp32_precision_has_no_scaler_and_keeps_parameters_fp32() -> None:
    device = torch.device("cpu")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    with autocast_context(device, "fp32"):
        loss = model(torch.ones(2, 2)).float().square().mean()
    loss.backward()
    optimizer.step()

    assert create_grad_scaler(device, "fp32") is None
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_bf16_autocast_keeps_parameters_fp32() -> None:
    device = torch.device("cpu")
    model = torch.nn.Linear(2, 2)
    try:
        with autocast_context(device, "bf16"):
            output = model(torch.ones(2, 2))
            loss = output.float().square().mean()
        loss.backward()
    except RuntimeError as error:
        pytest.skip(f"CPU BF16 autocast is unavailable: {error}")

    assert output.dtype == torch.bfloat16
    assert create_grad_scaler(device, "bf16") is None
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_fp16_scaler_unscales_before_clipping_and_keeps_parameters_fp32() -> None:
    device = torch.device("cpu")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        assert create_grad_scaler(device, "fp16") is not None
        scaler = torch.amp.GradScaler(device.type, init_scale=128.0)
        optimizer.zero_grad()
        with autocast_context(device, "fp16"):
            loss = model(torch.ones(2, 2)).float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_or_measure_grad_norm(model, max_grad_norm=1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
    except RuntimeError as error:
        pytest.skip(f"CPU FP16 AMP is unavailable: {error}")

    assert grad_norm >= 0
    assert scaler.get_scale() == old_scale
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_scale_backoff_marks_an_optimizer_attempt_as_skipped() -> None:
    step = 7
    next_step = step if scaler_step_was_skipped(128.0, 64.0) else step + 1

    assert next_step == 7
    assert not scaler_step_was_skipped(128.0, 128.0)


def test_precision_preflight_and_impossible_scaler_states_on_cpu() -> None:
    device = torch.device("cpu")
    validate_precision_support(device, "fp32")
    validate_precision_support(device, "bf16")
    validate_precision_support(device, "fp16")

    with pytest.raises(ValueError, match="fp16 requires"):
        validate_precision_state("fp16", None)
    with pytest.raises(ValueError, match="fp16 requires"):
        validate_precision_state("bf16", torch.amp.GradScaler("cpu"))


def test_real_grad_scaler_overflow_skips_parameter_update() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad()
    non_finite_loss = model(torch.ones(1, 1)).sum() * torch.tensor(float("inf"))
    scaler.scale(non_finite_loss).backward()
    scaler.unscale_(optimizer)
    did_step = optimizer_step(precision="fp16", optimizer=optimizer, scaler=scaler)

    assert not did_step
    for expected, actual in zip(before, model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_optimizer_step_calls_step_then_update_once_for_fp16() -> None:
    events: list[str] = []

    class FakeScaler:
        def get_scale(self) -> float:
            return 8.0

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            events.append("step")

        def update(self) -> None:
            events.append("update")

    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    assert optimizer_step(precision="fp16", optimizer=optimizer, scaler=FakeScaler())
    assert events == ["step", "update"]


def test_fp16_accumulation_unscales_clips_and_steps_once(monkeypatch) -> None:
    events: list[str] = []
    grad_accum_steps = 2

    class ScaledLoss:
        def backward(self) -> None:
            events.append("backward")

    class TrackingScaler:
        def scale(self, loss: torch.Tensor) -> ScaledLoss:
            events.append("scale")
            return ScaledLoss()

        def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
            events.append("unscale")

        def get_scale(self) -> float:
            return 8.0

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            events.append("step")

        def update(self) -> None:
            events.append("update")

    def track_clip(model: torch.nn.Module, max_grad_norm: float | None) -> float:
        events.append("clip")
        return 2.0

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = TrackingScaler()
    optimizer.zero_grad()
    for _ in range(grad_accum_steps):
        loss = model(torch.ones(1, 1)).sum() / grad_accum_steps
        scaler.scale(loss).backward()

    monkeypatch.setattr(training, "clip_or_measure_grad_norm", track_clip)
    grad_norm, did_step = training.complete_optimizer_step(
        model=model,
        optimizer=optimizer,
        precision="fp16",
        scaler=scaler,
        max_grad_norm=1.0,
    )

    assert grad_norm == 2.0
    assert did_step
    assert events == [
        "scale",
        "backward",
        "scale",
        "backward",
        "unscale",
        "clip",
        "step",
        "update",
    ]
    assert events.count("scale") == grad_accum_steps
    assert events.count("unscale") == 1
    assert events.count("clip") == 1
    assert events.count("step") == 1
    assert events.count("update") == 1


def test_precision_preflight_does_not_change_seeded_model_initialization() -> None:
    device = torch.device("cpu")
    validate_precision_support(device, "bf16")
    set_seed(123)
    model_after_preflight = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)

    set_seed(123)
    reference_model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)

    for actual, expected in zip(
        model_after_preflight.parameters(), reference_model.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_fp32_forward_backward_keeps_parameters_fp32() -> None:
    device = torch.device("mps")
    model = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(torch.ones(2, 2, device=device)).float().square().mean().backward()
    optimizer.step()

    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_fp16_is_supported_or_rejected_clearly() -> None:
    device = torch.device("mps")
    try:
        validate_precision_support(device, "fp16")
    except ValueError as error:
        assert "fp16 AMP is unsupported on mps" in str(error)
        return

    model = torch.nn.Linear(2, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    assert create_grad_scaler(device, "fp16") is not None
    scaler = torch.amp.GradScaler("mps", init_scale=128.0)
    optimizer.zero_grad()
    with autocast_context(device, "fp16"):
        output = model(torch.ones(2, 2, device=device))
        loss = output.float().square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    optimizer_step(precision="fp16", optimizer=optimizer, scaler=scaler)

    assert output.dtype == torch.float16
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_bf16_is_supported_or_rejected_clearly() -> None:
    device = torch.device("mps")
    try:
        validate_precision_support(device, "bf16")
    except ValueError as error:
        assert "bf16 AMP is unsupported on mps" in str(error)
        return

    model = torch.nn.Linear(2, 2).to(device)
    with autocast_context(device, "bf16"):
        output = model(torch.ones(2, 2, device=device))

    assert output.dtype == torch.bfloat16
    assert create_grad_scaler(device, "bf16") is None
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
