import random

import numpy as np
import pytest
import torch

from transformers_from_scratch.checkpointing import load_checkpoint, save_checkpoint
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import causal_lm_loss

MODEL_CONFIG = {"d_model": 8, "n_heads": 2, "d_ff": 16, "n_layers": 1}
VOCAB_SIZE = 11
TOKENIZER_REPO = "test/tiny-tokenizer"
CONFIG = {"seed": 123, "data": {}, "model": MODEL_CONFIG, "training": {}}


def make_model_and_optimizer() -> tuple[TinyDecoderLM, torch.optim.AdamW]:
    model = TinyDecoderLM(vocab_size=VOCAB_SIZE, **MODEL_CONFIG)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def train_one_step(model: TinyDecoderLM, optimizer: torch.optim.AdamW) -> None:
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
    optimizer.zero_grad()
    causal_lm_loss(model(inputs), targets).backward()
    optimizer.step()


def save_test_checkpoint(path, model, optimizer, step: int = 20) -> None:
    save_checkpoint(
        path,
        step=step,
        model=model,
        optimizer=optimizer,
        model_config=MODEL_CONFIG,
        vocab_size=VOCAB_SIZE,
        tokenizer_repo=TOKENIZER_REPO,
        config=CONFIG,
    )


def load_test_checkpoint(path, model, optimizer) -> dict:
    return load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        model_config=MODEL_CONFIG,
        vocab_size=VOCAB_SIZE,
        tokenizer_repo=TOKENIZER_REPO,
    )


def test_checkpoint_restores_model_optimizer_and_step(tmp_path) -> None:
    model, optimizer = make_model_and_optimizer()
    train_one_step(model, optimizer)
    path = tmp_path / "step-000020.pt"
    save_test_checkpoint(path, model, optimizer)
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    expected_state = next(iter(optimizer.state.values()))

    restored_model, restored_optimizer = make_model_and_optimizer()
    checkpoint = load_test_checkpoint(path, restored_model, restored_optimizer)

    assert checkpoint["step"] == 20
    assert not path.with_name(f"{path.name}.tmp").exists()
    for expected, actual in zip(expected_parameters, restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    restored_state = next(iter(restored_optimizer.state.values()))
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert key in restored_state
        assert torch.equal(expected_state[key], restored_state[key])


def test_checkpoint_restores_rng_state(tmp_path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer = make_model_and_optimizer()
    path = tmp_path / "rng.pt"
    save_test_checkpoint(path, model, optimizer)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.random(), np.random.rand(), torch.rand(3)

    restored_model, restored_optimizer = make_model_and_optimizer()
    load_test_checkpoint(path, restored_model, restored_optimizer)
    actual = (random.random(), np.random.rand(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_uses_next_global_step_through_target() -> None:
    assert list(range(20 + 1, 30 + 1)) == list(range(21, 31))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_checkpoint_restores_mps_rng_state(tmp_path) -> None:
    torch.mps.manual_seed(19)
    model, optimizer = make_model_and_optimizer()
    path = tmp_path / "mps-rng.pt"
    save_test_checkpoint(path, model, optimizer)
    expected = torch.rand(3, device="mps")
    torch.rand(3, device="mps")

    restored_model, restored_optimizer = make_model_and_optimizer()
    load_test_checkpoint(path, restored_model, restored_optimizer)
    actual = torch.rand(3, device="mps")

    assert torch.equal(actual.cpu(), expected.cpu())
