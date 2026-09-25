import pytest
import torch

from transformers_from_scratch.inference_benchmarking import (
    DecodeStepTiming,
    benchmark_uncached_greedy,
    benchmark_uncached_repetitions,
    calculate_inference_metrics,
    construct_exact_prompt,
    decode_window_metrics,
    summarize_inference_results,
)


class DeterministicModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
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


def test_benchmark_accepts_exact_minimum_context_length() -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    result, generated = benchmark_uncached_greedy(
        model,
        prompt,
        generated_tokens=2,
        context_length=3,
        device=torch.device("cpu"),
        warmup_forwards=0,
    )

    assert generated.shape == (1, 4)
    assert len(result.decode_steps) == 1
    assert result.decode_steps[0].prefix_length == 3


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
        (2, 2, "context_length"),
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


def test_decode_window_metrics_compares_first_and_last_windows() -> None:
    steps = tuple(
        DecodeStepTiming(prefix_length=index + 10, latency_ms=float(latency))
        for index, latency in enumerate((1, 2, 3, 4, 5, 6))
    )

    metrics = decode_window_metrics(steps, window_size=2)

    assert metrics.first_mean_ms == pytest.approx(1.5)
    assert metrics.last_mean_ms == pytest.approx(5.5)
    assert metrics.growth_ratio == pytest.approx(5.5 / 1.5)


@pytest.mark.parametrize(
    ("steps", "window_size", "message"),
    [
        ((DecodeStepTiming(1, 1.0),), 0, "window_size"),
        ((DecodeStepTiming(1, 1.0),), 2, "at least"),
        ((DecodeStepTiming(1, 0.0),), 1, "non-zero"),
    ],
)
def test_decode_window_metrics_rejects_invalid_windows(
    steps: tuple[DecodeStepTiming, ...], window_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_window_metrics(steps, window_size=window_size)


def test_inference_summary_uses_medians_across_runs() -> None:
    results = (
        calculate_inference_metrics(
            prompt_length=32,
            generated_tokens=5,
            prefill_ms=1.0,
            decode_steps=tuple(
                DecodeStepTiming(prefix_length=33 + index, latency_ms=latency)
                for index, latency in enumerate((1.0, 1.0, 2.0, 2.0))
            ),
        ),
        calculate_inference_metrics(
            prompt_length=32,
            generated_tokens=5,
            prefill_ms=3.0,
            decode_steps=tuple(
                DecodeStepTiming(prefix_length=33 + index, latency_ms=latency)
                for index, latency in enumerate((2.0, 2.0, 8.0, 8.0))
            ),
        ),
        calculate_inference_metrics(
            prompt_length=32,
            generated_tokens=5,
            prefill_ms=2.0,
            decode_steps=tuple(
                DecodeStepTiming(prefix_length=33 + index, latency_ms=latency)
                for index, latency in enumerate((4.0, 4.0, 12.0, 12.0))
            ),
        ),
    )

    summary = summarize_inference_results(results, window_size=2)

    assert summary.repetitions == 3
    assert summary.median_prefill_ms == pytest.approx(2.0)
    assert summary.median_decode_total_ms == pytest.approx(20.0)
    assert summary.median_mean_decode_ms == pytest.approx(5.0)
    assert summary.median_tokens_per_sec == pytest.approx(200.0)
    assert summary.median_total_latency_ms == pytest.approx(23.0)
    assert summary.median_first_window_ms == pytest.approx(2.0)
    assert summary.median_last_window_ms == pytest.approx(8.0)
    assert summary.median_growth_ratio == pytest.approx(3.0)


def test_repeated_benchmark_resets_prompt_and_preserves_identical_outputs() -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    repeated = benchmark_uncached_repetitions(
        model,
        prompt,
        generated_tokens=4,
        context_length=5,
        device=torch.device("cpu"),
        repetitions=3,
        warmup_forwards=1,
    )

    assert len(repeated.results) == 3
    assert len(repeated.generated_outputs) == 3
    assert all(output.shape == (1, 6) for output in repeated.generated_outputs)
    assert all(
        torch.equal(repeated.generated_outputs[0], output)
        for output in repeated.generated_outputs[1:]
    )
    assert model.forward_calls == 1 + 3 * 4


def test_repeated_benchmark_rejects_zero_repetitions() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        benchmark_uncached_repetitions(
            DeterministicModel(),
            torch.tensor([[1, 2]], dtype=torch.long),
            generated_tokens=2,
            context_length=3,
            device=torch.device("cpu"),
            repetitions=0,
        )
