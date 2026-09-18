# Transformers from First Principles

A minimal, research-friendly project for building a decoder-only Transformer manually in PyTorch. The aim is to understand the architecture and its performance characteristics—not to wrap a high-level Transformer library. Hugging Face Transformer implementations and high-level Transformer libraries are intentionally out of scope.

## Roadmap

### Stage 1 — Transformer from first principles

- Token embeddings
- Causal multi-head self-attention
- RoPE
- Pre-norm decoder blocks
- LayerNorm
- SiLU MLP
- LM head
- Causal LM loss
- Autoregressive generation
- Correctness tests

### Stage 2 — Reproducible training system

- TinyStories data pipeline
- Tokenizer integration
- Dataset/DataLoader
- Training and validation loops
- YAML config, reproducible seeds, and metrics logging

### Stage 3 — Modern LLM components / experiments

- RMSNorm
- SwiGLU
- Weight tying
- Training-dynamics experiments

### Stage 4 — Inference / systems

- KV cache
- Profiling
- Batching
- `torch.compile`

### Stage 4 — Systems experiments

- Mixed precision
- Quantization
- Serving experiments

## Model and training path

```text
Model:
token IDs (B, T)
-> embeddings
-> decoder blocks
-> final norm
-> logits (B, T, V)

Training:
TinyStories
-> tokenizer
-> causal windows
-> DataLoader
-> model
-> cross-entropy loss
-> AdamW
```

The model returns raw logits; loss remains outside `TinyDecoderLM.forward()`.
This initial training-system block deliberately excludes schedulers/warmup,
mixed precision, distributed training, KV cache, and profiling.

## Setup and commands

Install the project and its development tools:

```bash
poetry install
```

Run commands inside Poetry's managed environment:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format .
```

Or activate the Poetry environment for the current shell:

```bash
eval "$(poetry env activate)"
pytest
ruff check .
ruff format .
```

The tiny decoder is implemented and tested in `src/transformers_from_scratch/`.

The training entry point is:

```bash
poetry run python train.py --config configs/tiny.yaml
```

A local 2,000-step TinyStories checkpoint is saved under `checkpoints/` and is
intentionally ignored by Git.

## Generate from a checkpoint

After training, sample from a saved checkpoint with:

```bash
poetry run python generate.py \
  --checkpoint checkpoints/tinystories-2k.pt \
  --prompt "Once upon a time" --max-new-tokens 100 --temperature 0.8 --top-k 50
```

## Checkpoint and resume

Training saves atomic periodic checkpoints in `checkpoints/`; the directory is ignored by Git.

```bash
poetry run python train.py --config configs/tiny.yaml

poetry run python train.py \
  --config configs/tiny.yaml \
  --resume checkpoints/step-001000.pt
```

Resume restores model weights, AdamW optimizer moments, global step, and global RNG
state. It does not restore the active shuffled DataLoader iterator position, so
exact bit-for-bit continuation of batch order is not guaranteed.
