from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from transformers_from_scratch.serving import ModelRuntime, ServingSettings, create_app


class FakeTokenizer:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, prompt: str) -> SimpleNamespace:
        self.encode_calls += 1
        ids = [1, 2, 3, 4] if prompt == "long" else [1, 2]
        return SimpleNamespace(ids=ids)

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(str(token_id) for token_id in ids if token_id != 7)

    def get_vocab(self) -> dict[str, int]:
        return {"<|endoftext|>": 7}


class FakeCachedModel(torch.nn.Module):
    def __init__(self, next_token: int = 3) -> None:
        super().__init__()
        self.next_token = next_token

    def forward(self, input_ids: torch.Tensor, *, use_cache=False, kv_cache=None):
        logits = torch.zeros((*input_ids.shape, 64), device=input_ids.device)
        logits[..., self.next_token] = 1
        return logits, object()


@pytest.fixture
def app_factory():
    created: list[ModelRuntime] = []
    load_calls = 0

    def make_app(*, next_token: int = 3, context_length: int = 5):
        nonlocal load_calls

        def loader(settings: ServingSettings) -> ModelRuntime:
            nonlocal load_calls
            load_calls += 1
            runtime = ModelRuntime(
                model=FakeCachedModel(next_token),
                tokenizer=FakeTokenizer(),
                device=torch.device("cpu"),
                checkpoint_path=settings.checkpoint_path,
                context_length=context_length,
                vocab_size=64,
                eos_token_id=7,
            )
            created.append(runtime)
            return runtime

        app = create_app(
            ServingSettings(Path("checkpoints/test.pt"), "cpu"),
            runtime_loader=loader,
        )
        return app

    return make_app, created, lambda: load_calls


def test_health_reports_loaded_runtime(app_factory) -> None:
    make_app, _, _ = app_factory
    with TestClient(make_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "device": "cpu",
        "checkpoint": "test.pt",
    }


def test_generate_returns_text_and_actual_counts(app_factory) -> None:
    make_app, _, _ = app_factory
    with TestClient(make_app()) as client:
        response = client.post(
            "/generate",
            json={"prompt": "Once upon a time", "max_new_tokens": 2, "do_sample": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["generated_text"] == "1 2 3 3"
    assert body["prompt_tokens"] == 2
    assert body["generated_tokens"] == 2
    assert body["latency_ms"] >= 0


def test_model_and_tokenizer_are_loaded_once_per_app_lifespan(app_factory) -> None:
    make_app, created, load_calls = app_factory
    with TestClient(make_app()) as client:
        client.get("/health")
        client.post(
            "/generate",
            json={"prompt": "first", "max_new_tokens": 1, "do_sample": False},
        )
        client.post(
            "/generate",
            json={"prompt": "second", "max_new_tokens": 1, "do_sample": False},
        )

    assert load_calls() == 1
    assert len(created) == 1
    assert created[0].tokenizer.encode_calls == 2


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"prompt": "   ", "max_new_tokens": 1}, "prompt"),
        ({"prompt": "ok", "max_new_tokens": 1, "temperature": 0}, "temperature"),
        ({"prompt": "ok", "max_new_tokens": 1, "top_k": 0}, "top_k"),
        ({"prompt": "ok", "max_new_tokens": 1, "top_p": 0}, "top_p"),
        ({"prompt": "ok", "max_new_tokens": 1, "top_p": 1.1}, "top_p"),
        ({"prompt": "ok", "max_new_tokens": 0}, "max_new_tokens"),
        ({"prompt": "ok", "max_new_tokens": 257}, "max_new_tokens"),
    ],
)
def test_request_schema_rejects_invalid_values(app_factory, payload, field: str) -> None:
    make_app, _, _ = app_factory
    with TestClient(make_app()) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 422
    assert field in response.text


def test_generate_rejects_top_k_above_vocabulary(app_factory) -> None:
    make_app, _, _ = app_factory
    with TestClient(make_app()) as client:
        response = client.post(
            "/generate",
            json={"prompt": "ok", "max_new_tokens": 1, "top_k": 65},
        )

    assert response.status_code == 422
    assert "vocabulary size 64" in response.json()["detail"]


def test_generate_rejects_context_overflow(app_factory) -> None:
    make_app, _, _ = app_factory
    with TestClient(make_app()) as client:
        response = client.post(
            "/generate",
            json={"prompt": "long", "max_new_tokens": 3, "do_sample": False},
        )

    assert response.status_code == 422
    assert "require 6 cached positions" in response.json()["detail"]


def test_generate_reports_eos_early_stopping(app_factory) -> None:
    make_app, _, _ = app_factory
    with TestClient(make_app(next_token=7)) as client:
        response = client.post(
            "/generate",
            json={"prompt": "Once", "max_new_tokens": 4, "do_sample": False},
        )

    assert response.status_code == 200
    assert response.json()["generated_tokens"] == 1


def test_greedy_endpoint_is_deterministic(app_factory) -> None:
    make_app, _, _ = app_factory
    payload = {"prompt": "Once", "max_new_tokens": 3, "do_sample": False}
    with TestClient(make_app()) as client:
        first = client.post("/generate", json=payload).json()
        second = client.post("/generate", json=payload).json()

    assert first["generated_text"] == second["generated_text"]
    assert first["generated_tokens"] == second["generated_tokens"]
