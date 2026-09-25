"""Benchmark deterministic uncached FP32 autoregressive inference."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from transformers_from_scratch.data import load_tokenizer
from transformers_from_scratch.inference_benchmarking import (
    InferenceBenchmarkSummary,
    RepeatedInferenceBenchmark,
    benchmark_uncached_repetitions,
    construct_exact_prompt,
    summarize_inference_results,
)
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_LENGTHS = (32, 64, 128, 256)
GENERATED_TOKENS = 64
BENCHMARK_CONTEXT_LENGTH = 512
DETERMINISTIC_PROMPT_TEXT = (
    "Once upon a time, a curious child followed a winding path through the quiet forest. "
    "At every turn, the child found a new friend and learned something kind and useful. "
) * 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "tinystories-2k.pt",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--warmup-forwards", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


def load_benchmark_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TinyDecoderLM, Tokenizer]:
    """Load the tokenizer and FP32 model without including setup in measured time."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer(checkpoint["tokenizer_repo"])
    model = TinyDecoderLM(vocab_size=checkpoint["vocab_size"], **checkpoint["model_config"]).to(
        device=device, dtype=torch.float32
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def print_summary(summaries: list[InferenceBenchmarkSummary]) -> None:
    """Print median cross-run latency and early-vs-late decode growth."""
    print()
    print(
        "prompt | reps | prefill_med | decode_ms_tok_med | decode_tok_s_med "
        "| first8_ms | last8_ms | growth | total_med"
    )
    for summary in summaries:
        print(
            f"{summary.prompt_length:>6} | {summary.repetitions:>4} "
            f"| {summary.median_prefill_ms:>11.3f} "
            f"| {summary.median_mean_decode_ms:>17.3f} "
            f"| {summary.median_tokens_per_sec:>16.1f} "
            f"| {summary.median_first_window_ms:>9.3f} "
            f"| {summary.median_last_window_ms:>8.3f} "
            f"| {summary.median_growth_ratio:>6.3f}x "
            f"| {summary.median_total_latency_ms:>9.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.warmup_forwards < 0:
        raise ValueError("warmup_forwards must be non-negative")
    if args.repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    device = resolve_device(args.device)
    model, tokenizer = load_benchmark_model(args.checkpoint, device)
    vocab_size = model.token_embedding.num_embeddings
    if vocab_size != tokenizer.get_vocab_size():
        raise ValueError("checkpoint and tokenizer vocabulary sizes do not match")
    source_token_ids = tokenizer.encode(DETERMINISTIC_PROMPT_TEXT).ids

    print(f"batch_size=1 repetitions={args.repetitions} warmup_forwards={args.warmup_forwards}")
    print(f"device={device.type} precision=fp32 decoding=greedy cache=disabled")
    print(
        f"generated_tokens={GENERATED_TOKENS} benchmark_context_length={BENCHMARK_CONTEXT_LENGTH}"
    )
    print("counting=prefill produces token 1; decode measures 63 subsequent full-prefix forwards")
    print(
        "note=prompt lengths above 128 measure systems/performance scaling, not language-model "
        "quality beyond the training context"
    )

    repeated_workloads: list[RepeatedInferenceBenchmark] = []
    for prompt_length in PROMPT_LENGTHS:
        prompt_ids = construct_exact_prompt(
            source_token_ids,
            prompt_length,
            vocab_size,
            device,
        )
        repeated_workloads.append(
            benchmark_uncached_repetitions(
                model,
                prompt_ids,
                generated_tokens=GENERATED_TOKENS,
                context_length=BENCHMARK_CONTEXT_LENGTH,
                device=device,
                repetitions=args.repetitions,
                warmup_forwards=args.warmup_forwards,
            )
        )

    summaries = [
        summarize_inference_results(workload.results, window_size=8)
        for workload in repeated_workloads
    ]
    print_summary(summaries)


if __name__ == "__main__":
    main()
