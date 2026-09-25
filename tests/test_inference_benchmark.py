import pytest
import torch

from transformers_from_scratch.inference_benchmarking import (
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
        self.forward_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forward_lengths.append(input_ids.shape[1])
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


def test_metric_math_uses_decode_block_and_forward_count() -> None:
    result = calculate_inference_metrics(
        prompt_length=32,
        generated_tokens=5,
        prefill_ms=5.0,
        decode_total_ms=40.0,
        first_block_total_ms=12.0,
        last_block_total_ms=20.0,
        window_size=2,
    )

    assert result.decode_total_ms == pytest.approx(40.0)
    assert result.mean_decode_ms == pytest.approx(10.0)
    assert result.tokens_per_sec == pytest.approx(4 / 0.04)
    assert result.total_latency_ms == pytest.approx(45.0)


def test_decode_window_metrics_derive_per_token_latency_from_blocks() -> None:
    metrics = decode_window_metrics(
        first_block_total_ms=12.0,
        last_block_total_ms=20.0,
        window_size=4,
    )

    assert metrics.first_mean_ms == pytest.approx(3.0)
    assert metrics.last_mean_ms == pytest.approx(5.0)
    assert metrics.growth_ratio == pytest.approx(5.0 / 3.0)


@pytest.mark.parametrize(
    ("first_ms", "last_ms", "window_size", "message"),
    [
        (1.0, 1.0, 0, "window_size"),
        (0.0, 1.0, 1, "first block"),
        (1.0, -1.0, 1, "last block"),
    ],
)
def test_decode_window_metrics_reject_invalid_values(
    first_ms: float, last_ms: float, window_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_window_metrics(
            first_block_total_ms=first_ms,
            last_block_total_ms=last_ms,
            window_size=window_size,
        )


def test_benchmark_runs_three_decode_iterations_after_prefill() -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    result, generated = benchmark_uncached_greedy(
        model,
        prompt,
        generated_tokens=4,
        context_length=5,
        device=torch.device("cpu"),
        warmup_forwards=0,
        window_size=1,
    )

    assert generated.shape == (1, 6)
    assert torch.equal(generated[:, :2], prompt)
    assert result.generated_tokens == 4
    # One measured execution and one window execution each run prefill + 3 decode forwards.
    assert model.forward_lengths == [2, 3, 4, 5, 2, 3, 4, 5]


def test_benchmark_accepts_exact_minimum_context_length() -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    result, generated = benchmark_uncached_greedy(
        model,
        prompt,
        generated_tokens=4,
        context_length=5,
        device=torch.device("cpu"),
        warmup_forwards=0,
        window_size=1,
    )

    assert generated.shape == (1, 6)
    assert result.mean_decode_ms >= 0


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
    ("generated_tokens", "context_length", "window_size", "message"),
    [
        (0, 4, 1, "generated_tokens"),
        (4, 4, 1, "context_length"),
        (4, 5, 2, "windows"),
    ],
)
def test_benchmark_rejects_invalid_workload(
    generated_tokens: int, context_length: int, window_size: int, message: str
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
            window_size=window_size,
        )


def make_result(
    *,
    prefill_ms: float,
    decode_total_ms: float,
    first_block_ms: float,
    last_block_ms: float,
):
    return calculate_inference_metrics(
        prompt_length=32,
        generated_tokens=5,
        prefill_ms=prefill_ms,
        decode_total_ms=decode_total_ms,
        first_block_total_ms=first_block_ms,
        last_block_total_ms=last_block_ms,
        window_size=2,
    )


def test_inference_summary_uses_medians_across_runs() -> None:
    results = (
        make_result(prefill_ms=1.0, decode_total_ms=6.0, first_block_ms=2.0, last_block_ms=4.0),
        make_result(prefill_ms=3.0, decode_total_ms=20.0, first_block_ms=4.0, last_block_ms=16.0),
        make_result(prefill_ms=2.0, decode_total_ms=32.0, first_block_ms=8.0, last_block_ms=24.0),
    )

    summary = summarize_inference_results(results)

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
        window_size=1,
    )

    assert len(repeated.results) == 3
    assert len(repeated.generated_outputs) == 3
    assert all(output.shape == (1, 6) for output in repeated.generated_outputs)
    assert all(
        torch.equal(repeated.generated_outputs[0], output)
        for output in repeated.generated_outputs[1:]
    )
    # One warmup total, then two complete four-forward executions per repetition.
    assert len(model.forward_lengths) == 1 + 3 * 2 * 4


def test_repeated_benchmark_rejects_zero_repetitions() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        benchmark_uncached_repetitions(
            DeterministicModel(),
            torch.tensor([[1, 2]], dtype=torch.long),
            generated_tokens=4,
            context_length=5,
            device=torch.device("cpu"),
            repetitions=0,
            window_size=1,
        )
