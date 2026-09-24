import pytest
import torch

from transformers_from_scratch.inference_benchmarking import (
    DecodeStepTiming,
    benchmark_uncached_greedy,
    calculate_inference_metrics,
    construct_exact_prompt,
)


class DeterministicModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.vocab_size)
        next_ids = (input_ids + 1) % self.vocab_size
        return logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)


def test_construct_exact_prompt_has_requested_shape_and_valid_ids() -> None:
    prompt = construct_exact_prompt(
        source_token_ids=[1, 2, 3, 4, 5],
        prompt_length=4,
        vocab_size=6,
        device=torch.device("cpu"),
    )

    assert prompt.shape == (1, 4)
    assert prompt.dtype == torch.long
    assert prompt.tolist() == [[1, 2, 3, 4]]
    assert prompt.min().item() >= 0
    assert prompt.max().item() < 6


def test_inference_metric_math_uses_only_subsequent_decode_forwards() -> None:
    result = calculate_inference_metrics(
        prompt_length=32,
        generated_tokens=3,
        prefill_ms=5.0,
        decode_steps=(
            DecodeStepTiming(prefix_length=33, latency_ms=10.0),
            DecodeStepTiming(prefix_length=34, latency_ms=20.0),
        ),
    )

    assert result.decode_total_ms == pytest.approx(30.0)
    assert result.mean_decode_ms == pytest.approx(15.0)
    assert result.tokens_per_sec == pytest.approx(2 / 0.03)
    assert result.total_latency_ms == pytest.approx(35.0)


def test_benchmark_prefill_decode_split_generates_exact_token_count() -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    result, generated = benchmark_uncached_greedy(
        model,
        prompt,
        generated_tokens=4,
        context_length=6,
        device=torch.device("cpu"),
        warmup_forwards=0,
    )

    assert generated.shape == (1, 6)
    assert torch.equal(generated[:, :2], prompt)
    assert len(result.decode_steps) == 3
    assert tuple(step.prefix_length for step in result.decode_steps) == (3, 4, 5)


@pytest.mark.parametrize(
    ("source_token_ids", "prompt_length", "vocab_size", "message"),
    [
        ([1], 0, 2, "prompt_length"),
        ([1], 2, 2, "shorter"),
        ([2], 1, 2, "vocabulary"),
    ],
)
def test_construct_exact_prompt_rejects_invalid_inputs(
    source_token_ids: list[int], prompt_length: int, vocab_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        construct_exact_prompt(
            source_token_ids,
            prompt_length,
            vocab_size,
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("generated_tokens", "context_length", "message"),
    [
        (0, 4, "generated_tokens"),
        (2, 3, "context_length"),
    ],
)
def test_benchmark_rejects_invalid_workload(
    generated_tokens: int, context_length: int, message: str
) -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match=message):
        benchmark_uncached_greedy(
            model,
            prompt,
            generated_tokens=generated_tokens,
            context_length=context_length,
            device=torch.device("cpu"),
            warmup_forwards=0,
        )
