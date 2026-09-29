"""Benchmark full prefill and one-token cached decode scaling."""

import argparse
from pathlib import Path

from benchmark_inference import (
    DETERMINISTIC_PROMPT_TEXT,
    PROJECT_ROOT,
    load_benchmark_model,
)

from transformers_from_scratch.inference_benchmarking import (
    benchmark_prefill_cached_decode,
    construct_exact_prompt,
    kv_cache_size_bytes,
)
from transformers_from_scratch.training import resolve_device

SEQUENCE_LENGTHS = (32, 64, 128, 256, 512)
BATCH_SIZE = 1
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
    rows = []
    for sequence_length in SEQUENCE_LENGTHS:
        prompt_ids = construct_exact_prompt(
            source_token_ids,
            sequence_length,
            vocab_size,
            device,
        )
        result = benchmark_prefill_cached_decode(
            model,
            prompt_ids,
            device=device,
            repetitions=REPETITIONS,
            warmup_forwards=WARMUP_FORWARDS,
        )
        cache_bytes = kv_cache_size_bytes(
            n_layers=len(model.blocks),
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            d_model=model.token_embedding.embedding_dim,
            bytes_per_element=BYTES_PER_ELEMENT,
        )
        rows.append((result, cache_bytes / 1024**2))

    print(f"device={device.type}")
    print("precision=fp32 batch_size=1")
    print(f"repetitions={REPETITIONS} warmup_forwards={WARMUP_FORWARDS}")
    print(
        "note=sequence lengths above training context are systems/performance measurements, "
        "not language-model-quality claims"
    )
    print()
    print("T | prefill_ms | cached_decode_ms | kv_cache_MiB")
    for result, cache_mib in rows:
        print(
            f"{result.sequence_length:>3} | {result.median_prefill_ms:>10.3f} "
            f"| {result.median_cached_decode_ms:>16.3f} | {cache_mib:>12.3f}"
        )


if __name__ == "__main__":
    main()
