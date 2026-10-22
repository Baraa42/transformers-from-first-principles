"""Minimal FastAPI serving layer for the trained TinyDecoderLM checkpoint."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tokenizers import Tokenizer

from .data import load_tokenizer
from .generation import generate_cached
from .model import TinyDecoderLM
from .training import resolve_device, synchronize_device

LOGGER = logging.getLogger(__name__)
DEFAULT_CHECKPOINT = Path("checkpoints/tinystories-2k.pt")
MAX_NEW_TOKENS = 256


class GenerationRequest(BaseModel):
    """Validated single-prompt generation request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    max_new_tokens: int = Field(default=50, gt=0, le=MAX_NEW_TOKENS)
    do_sample: bool = True
    temperature: float = Field(default=0.8, gt=0)
    top_k: int | None = Field(default=40, gt=0)
    top_p: float | None = Field(default=0.9, gt=0, le=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        """Reject blank prompts and remove irrelevant outer whitespace."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        return prompt


class GenerationResponse(BaseModel):
    """Text and measured request-level generation metadata."""

    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Current model lifecycle state."""

    status: str
    model_loaded: bool
    device: str
    checkpoint: str


@dataclass(frozen=True)
class ServingSettings:
    """Process-level model configuration resolved before application startup."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    device: str = "auto"

    @classmethod
    def from_environment(cls) -> ServingSettings:
        """Read optional deployment overrides while preserving useful defaults."""
        return cls(
            checkpoint_path=Path(
                os.environ.get("TRANSFORMER_CHECKPOINT", str(DEFAULT_CHECKPOINT))
            ),
            device=os.environ.get("TRANSFORMER_DEVICE", "auto"),
        )


@dataclass
class ModelRuntime:
    """Objects loaded exactly once and shared by all requests in one process."""

    model: torch.nn.Module
    tokenizer: Tokenizer
    device: torch.device
    checkpoint_path: Path
    context_length: int
    vocab_size: int
    eos_token_id: int | None
    inference_lock: threading.Lock = field(default_factory=threading.Lock)


RuntimeLoader = Callable[[ServingSettings], ModelRuntime]


def _required_checkpoint_field(checkpoint: dict[str, Any], field_name: str) -> Any:
    try:
        return checkpoint[field_name]
    except KeyError as error:
        raise RuntimeError(f"checkpoint is missing required field: {field_name}") from error


def _find_eos_token_id(tokenizer: Tokenizer) -> int | None:
    vocab = tokenizer.get_vocab()
    return next(
        (vocab[token] for token in ("<|endoftext|>", "<eos>", "</s>") if token in vocab),
        None,
    )


def load_model_runtime(settings: ServingSettings) -> ModelRuntime:
    """Load checkpoint metadata, tokenizer, and model once during startup."""
    device = resolve_device(settings.device)
    try:
        checkpoint = torch.load(
            settings.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"checkpoint not found: {settings.checkpoint_path}") from error

    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint root must be a mapping")

    model_config = _required_checkpoint_field(checkpoint, "model_config")
    vocab_size = _required_checkpoint_field(checkpoint, "vocab_size")
    tokenizer_repo = _required_checkpoint_field(checkpoint, "tokenizer_repo")
    checkpoint_config = _required_checkpoint_field(checkpoint, "config")
    try:
        context_length = checkpoint_config["data"]["context_length"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("checkpoint is missing config.data.context_length") from error

    tokenizer = load_tokenizer(tokenizer_repo)
    model = TinyDecoderLM(vocab_size=vocab_size, **model_config)
    model.load_state_dict(_required_checkpoint_field(checkpoint, "model_state_dict"), strict=True)
    model.to(device)
    model.eval()

    return ModelRuntime(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_path=settings.checkpoint_path,
        context_length=context_length,
        vocab_size=vocab_size,
        eos_token_id=_find_eos_token_id(tokenizer),
    )


def _runtime(request: Request) -> ModelRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return runtime


def _encode_and_validate(runtime: ModelRuntime, payload: GenerationRequest) -> list[int]:
    prompt_ids = runtime.tokenizer.encode(payload.prompt).ids
    if not prompt_ids:
        raise HTTPException(status_code=422, detail="prompt must encode to at least one token")
    if payload.top_k is not None and payload.top_k > runtime.vocab_size:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must not exceed vocabulary size {runtime.vocab_size}",
        )

    required_context = len(prompt_ids) + payload.max_new_tokens - 1
    if required_context > runtime.context_length:
        raise HTTPException(
            status_code=422,
            detail=(
                "prompt and generation require "
                f"{required_context} cached positions, but the model supports "
                f"{runtime.context_length}"
            ),
        )
    return prompt_ids


def create_app(
    settings: ServingSettings | None = None,
    *,
    runtime_loader: RuntimeLoader = load_model_runtime,
) -> FastAPI:
    """Build an application whose lifespan owns a single model runtime."""
    settings = settings or ServingSettings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime_loader(settings)
        try:
            yield
        finally:
            app.state.runtime = None

    app = FastAPI(title="Transformer from First Principles", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        runtime = _runtime(request)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            device=str(runtime.device),
            checkpoint=runtime.checkpoint_path.name,
        )

    @app.post("/generate", response_model=GenerationResponse)
    def generate_text(payload: GenerationRequest, request: Request) -> GenerationResponse:
        runtime = _runtime(request)
        prompt_ids = _encode_and_validate(runtime, payload)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)

        started_at = time.perf_counter()
        try:
            with runtime.inference_lock:
                output_ids = generate_cached(
                    runtime.model,
                    input_ids,
                    max_new_tokens=payload.max_new_tokens,
                    context_length=runtime.context_length,
                    do_sample=payload.do_sample,
                    temperature=payload.temperature,
                    top_k=payload.top_k,
                    top_p=payload.top_p,
                    eos_token_id=runtime.eos_token_id,
                )
                synchronize_device(runtime.device)
        except Exception as error:
            LOGGER.exception("generation failed")
            raise HTTPException(status_code=500, detail="inference failed") from error

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        generated_tokens = output_ids.shape[1] - len(prompt_ids)
        generated_text = runtime.tokenizer.decode(
            output_ids[0].detach().cpu().tolist(),
            skip_special_tokens=True,
        )
        return GenerationResponse(
            generated_text=generated_text,
            prompt_tokens=len(prompt_ids),
            generated_tokens=generated_tokens,
            latency_ms=latency_ms,
        )

    return app
