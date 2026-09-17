"""Generate text from a saved TinyDecoderLM checkpoint."""

import argparse

import torch

from transformers_from_scratch.data import load_tokenizer
from transformers_from_scratch.generation import generate
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tinystories-2k.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    seed = checkpoint["config"]["seed"] if args.seed is None else args.seed
    set_seed(seed)

    tokenizer = load_tokenizer(checkpoint["tokenizer_repo"])
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("Prompt must encode to at least one token")
    model = TinyDecoderLM(vocab_size=checkpoint["vocab_size"], **checkpoint["model_config"]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    output_ids = generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        max_new_tokens=args.max_new_tokens,
        context_length=checkpoint["config"]["data"]["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
