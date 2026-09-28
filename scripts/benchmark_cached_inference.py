"""Benchmark cached-versus-uncached inference across prompt lengths."""

import argparse
from pathlib import Path

import torch
from benchmark_inference import (
    DETERMINISTIC_PROMPT_TEXT,
    PROJECT_ROOT,
    load_benchmark_model,
)

from transformers_from_scratch.inference_benchmarking import (
    benchmark_cached_repetitions,
    benchmark_uncached_repetitions,
    construct_exact_prompt,
    summarize_cached_inference_results,
    summarize_inference_results,
)
from transformers_from_scratch.training import resolve_device

PROMPT_LENGTHS = (32, 64, 128, 256)
GENERATED_TOKENS = 64
BENCHMARK_CONTEXT_LENGTH = 512
REPETITIONS = 5
WARMUP_FORWARDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "tinystories-2k.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, tokenizer = load_benchmark_model(args.checkpoint, device)
    vocab_size = model.token_embedding.num_embeddings
    if vocab_size != tokenizer.get_vocab_size():
        raise ValueError("checkpoint and tokenizer vocabulary sizes do not match")

    source_token_ids = tokenizer.encode(DETERMINISTIC_PROMPT_TEXT).ids
    comparisons = []
    for prompt_length in PROMPT_LENGTHS:
        prompt_ids = construct_exact_prompt(
            source_token_ids,
            prompt_length,
            vocab_size,
            device,
        )
        uncached = benchmark_uncached_repetitions(
            model,
            prompt_ids,
            generated_tokens=GENERATED_TOKENS,
            context_length=BENCHMARK_CONTEXT_LENGTH,
            device=device,
            repetitions=REPETITIONS,
            warmup_forwards=WARMUP_FORWARDS,
        )
        cached = benchmark_cached_repetitions(
            model,
            prompt_ids,
            generated_tokens=GENERATED_TOKENS,
            context_length=BENCHMARK_CONTEXT_LENGTH,
            device=device,
            repetitions=REPETITIONS,
            warmup_forwards=WARMUP_FORWARDS,
        )

        for uncached_ids, cached_ids in zip(
            uncached.generated_outputs, cached.generated_outputs, strict=True
        ):
            if not torch.equal(uncached_ids, cached_ids):
                raise RuntimeError(
                    f"cached and uncached generated token IDs differ for prompt {prompt_length}"
                )

        uncached_summary = summarize_inference_results(uncached.results)
        cached_summary = summarize_cached_inference_results(cached.results)
        decode_speedup = (
            uncached_summary.median_decode_total_ms / cached_summary.median_decode_total_ms
        )
        comparisons.append((prompt_length, uncached_summary, cached_summary, decode_speedup))

    print(f"device={device.type}")
    print("precision=fp32 decoding=greedy batch_size=1")
    print(
        f"generated_tokens={GENERATED_TOKENS} repetitions={REPETITIONS} "
        f"warmup_forwards={WARMUP_FORWARDS}"
    )
    print("cache_comparison=uncached vs cached")
    print("correctness=exact token equality required and passed for every repetition")
    print(
        "note=prompt lengths above training context are systems/performance measurements, "
        "not language-model-quality claims"
    )
    print()
    print("prompt | uncached_ms_tok | cached_ms_tok | uncached_tok_s | cached_tok_s | speedup")
    for prompt_length, uncached_summary, cached_summary, decode_speedup in comparisons:
        print(
            f"{prompt_length:>6} | {uncached_summary.median_mean_decode_ms:>15.3f} "
            f"| {cached_summary.median_mean_decode_ms:>13.3f} "
            f"| {uncached_summary.median_tokens_per_sec:>14.1f} "
            f"| {cached_summary.median_tokens_per_sec:>12.1f} "
            f"| {decode_speedup:>6.3f}x"
        )


if __name__ == "__main__":
    main()
