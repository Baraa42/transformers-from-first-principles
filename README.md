# Transformers from First Principles

A minimal, research-friendly project for building a decoder-only Transformer manually in PyTorch. The aim is to understand the architecture and its performance characteristics—not to wrap a high-level Transformer library. Hugging Face Transformer implementations and high-level Transformer libraries are intentionally out of scope.

## Roadmap

### Stage 1 — Tiny decoder

- Token embeddings
- Causal multi-head attention
- RoPE
- RMSNorm
- MLP / SwiGLU
- Decoder block
- Stacked decoder-only model
- LM head
- Autoregressive loss

### Stage 2 — Training and sampling

- Train on a small text dataset
- Generation
- Temperature / top-k / top-p sampling

### Stage 3 — Inference performance

- KV cache
- Inference profiling
- Batching
- `torch.compile`

### Stage 4 — Systems experiments

- Mixed precision
- Quantization
- Systems / serving experiments

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

The first implementation milestone is a tested tiny decoder in `src/transformers_from_scratch/`.
