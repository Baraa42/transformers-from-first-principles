# Transformers from First Principles

A minimal, research-friendly project for building a decoder-only Transformer manually in PyTorch. The aim is to understand the architecture and its performance characteristics—not to wrap a high-level Transformer library. Hugging Face Transformer implementations and high-level Transformer libraries are intentionally out of scope.

## Roadmap

### Stage 1 — Transformer from first principles

- Token embeddings, causal attention, RoPE, decoder blocks, and LM head
- Autoregressive loss and generation

### Stage 2 — Reproducible training

- TinyStories pipeline, checkpoints/resume, gradient clipping, and accumulation

### Stage 3 — Training correctness

- Full training batches, reshuffling epochs, and token-weighted validation

### Stage 4 — Mixed precision

- `precision: fp32 | fp16 | bf16` in the training config
- FP16 uses GradScaler; BF16 and FP16 availability is verified on the selected backend

### Later systems experiments

- KV cache, operator-level profiling, batching, `torch.compile`, quantization, and serving

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
The current training system includes optional mixed precision and opt-in manual timing diagnostics; schedulers/warmup, distributed training, and KV cache remain out of scope.

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

Check the selected backend's actual local AMP capabilities:

```bash
poetry run python scripts/check_precision.py
```

AMP support depends on both the backend and installed PyTorch version; this
command exercises representative autocast and FP16 GradScaler operations.

## Precision benchmark

Run the same 300-step workload in fresh FP32, FP16, and BF16 processes:

```bash
poetry run python scripts/check_precision.py
poetry run python scripts/benchmark_precision.py
```

Raw logs are written to `benchmarks/precision/` and remain local. Reported total
wall time includes the training loop, validation, and checkpoint writes; model,
tokenizer, and dataset setup are excluded.

## Stage 5.1 timing diagnostics

Run the first decomposed FP32 timing experiment with:

```bash
poetry run python train.py --config configs/benchmarks/profile-fp32.yaml
```

Profiling mode synchronizes around each component for diagnostic accuracy, so its
throughput should not be compared directly with the normal precision benchmark.

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
  --checkpoint checkpoints/step-002000.pt \
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

On resume, `optimizer.load_state_dict(...)` restores AdamW parameter-group state, including the saved learning rate and weight decay. For a true resume, those checkpoint settings take precedence over newly specified YAML optimizer values.
