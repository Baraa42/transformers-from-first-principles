"""Benchmark static cached-decode batching at a fixed context length."""

import argparse
from pathlib import Path

from benchmark_inference import (
    DETERMINISTIC_PROMPT_TEXT,
    PROJECT_ROOT,
    load_benchmark_model,
)

from transformers_from_scratch.inference_benchmarking import (
    benchmark_cached_decode_batch,
    construct_exact_prompt,
    kv_cache_size_bytes,
)
from transformers_from_scratch.training import resolve_device

CONTEXT_LENGTH = 128
BATCH_SIZES = (1, 2, 4, 8, 16)
BLOCK_ITERATIONS = 50
REPETITIONS = 5
WARMUP_FORWARDS = 2
BYTES_PER_ELEMENT = 4


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
    single_prompt = construct_exact_prompt(
        source_token_ids,
        CONTEXT_LENGTH,
        vocab_size,
        device,
    )

    rows = []
    for batch_size in BATCH_SIZES:
        prompt_ids = single_prompt.expand(batch_size, -1).clone()
        result = benchmark_cached_decode_batch(
            model,
            prompt_ids,
            device=device,
            repetitions=REPETITIONS,
            warmup_forwards=WARMUP_FORWARDS,
            block_iterations=BLOCK_ITERATIONS,
        )
        cache_bytes = kv_cache_size_bytes(
            n_layers=len(model.blocks),
            batch_size=batch_size,
            sequence_length=CONTEXT_LENGTH,
            d_model=model.token_embedding.embedding_dim,
            bytes_per_element=BYTES_PER_ELEMENT,
        )
        rows.append((result, cache_bytes / 1024**2))

    print(f"device={device.type}")
    print("precision=fp32")
    print(f"context_length={CONTEXT_LENGTH}")
    print(
        f"block_iterations={BLOCK_ITERATIONS} repetitions={REPETITIONS} "
        f"warmup_forwards={WARMUP_FORWARDS}"
    )
    print("reported_latency=synchronized block total / block_iterations")
    print()
    print("B | decode_ms | aggregate_tok_s | per_seq_tok_s | kv_cache_MiB")
    for result, cache_mib in rows:
        print(
            f"{result.batch_size:>2} | {result.median_decode_step_ms:>9.3f} "
            f"| {result.aggregate_tokens_per_sec:>15.1f} "
            f"| {result.per_sequence_tokens_per_sec:>13.1f} "
            f"| {cache_mib:>12.3f}"
        )


if __name__ == "__main__":
    main()
