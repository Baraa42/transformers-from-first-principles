"""Benchmark deterministic uncached FP32 autoregressive inference."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from transformers_from_scratch.data import load_tokenizer
from transformers_from_scratch.inference_benchmarking import (
    InferenceBenchmarkResult,
    benchmark_uncached_greedy,
    construct_exact_prompt,
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


def print_summary(results: list[InferenceBenchmarkResult]) -> None:
    """Print prefill and subsequent forward-only decode measurements."""
    print()
    print("prompt | new_tokens | prefill_ms | decode_ms | mean_decode_ms | decode_tok_s | total_ms")
    for result in results:
        print(
            f"{result.prompt_length:>6} | {result.generated_tokens:>10} "
            f"| {result.prefill_ms:>10.3f} | {result.decode_total_ms:>9.3f} "
            f"| {result.mean_decode_ms:>14.3f} | {result.tokens_per_sec:>12.1f} "
            f"| {result.total_latency_ms:>8.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.warmup_forwards < 0:
        raise ValueError("warmup_forwards must be non-negative")

    device = resolve_device(args.device)
    model, tokenizer = load_benchmark_model(args.checkpoint, device)
    vocab_size = model.token_embedding.num_embeddings
    if vocab_size != tokenizer.get_vocab_size():
        raise ValueError("checkpoint and tokenizer vocabulary sizes do not match")
    source_token_ids = tokenizer.encode(DETERMINISTIC_PROMPT_TEXT).ids

    print("batch_size=1")
    print(f"device={device.type} precision=fp32 decoding=greedy cache=disabled")
    print(
        f"generated_tokens={GENERATED_TOKENS} benchmark_context_length={BENCHMARK_CONTEXT_LENGTH}"
    )
    print("counting=prefill produces token 1; decode measures 63 subsequent full-prefix forwards")
    print(
        "note=prompt lengths above 128 measure systems/performance scaling, not language-model "
        "quality beyond the training context"
    )

    results = []
    for prompt_length in PROMPT_LENGTHS:
        prompt_ids = construct_exact_prompt(
            source_token_ids,
            prompt_length,
            vocab_size,
            device,
        )
        result, _ = benchmark_uncached_greedy(
            model,
            prompt_ids,
            generated_tokens=GENERATED_TOKENS,
            context_length=BENCHMARK_CONTEXT_LENGTH,
            device=device,
            warmup_forwards=args.warmup_forwards,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
