# Transformers from First Principles

A small decoder-only language model implemented directly in PyTorch, with a TinyStories
training pipeline and experiments on training and inference performance. The project
connects model mathematics, numerical correctness, and systems measurement: implement the
operations explicitly, verify their behavior, then test whether an optimization improves
whole-workload latency.

## What was implemented

- Manual multi-head causal attention, RoPE, pre-norm decoder blocks, SiLU MLPs, token
  embeddings, and an untied language-model head; no high-level Transformer implementation.
- Story-level train/validation splitting, BPE tokenization, causal windows, AdamW training,
  gradient accumulation/clipping, checkpoint/resume, and optional FP16/BF16 mixed precision.
- Numerical references for attention, gradients, cached logits, and optimizer arithmetic;
  unit and integration tests for training, generation, and request-local cache state.
- Autoregressive sampling and greedy decoding, KV caching with absolute RoPE offsets,
  prefill/decode diagnostics, static batching, and a small [FastAPI baseline](docs/serving.md).
- Controlled MPS training experiments and an opt-in preallocated KV cache, with raw
  timings, correctness gates, source hashes, and documented measurement boundaries.

## Architecture and training pipeline

The [main configuration](configs/tiny.yaml) uses vocabulary 8,192, width 128, four decoder
layers, four heads, MLP width 512, and **2,888,448 parameters**. Each block applies LayerNorm
before attention and the MLP, with residual connections around both; final normalization
and the LM head produce raw vocabulary logits. See [architecture notes](notes/architecture.md).

TinyStories stories → published BPE tokenizer → within-story next-token windows → shuffled
training batches → cross-entropy → backward/gradient clipping → AdamW → checkpoint.
The default run uses context 128, batch 16, FP32, and 2,000 optimizer steps. Training and
validation stories are selected from disjoint ranges of the upstream training split;
validation loss is target-token weighted. [Data details](docs/data.md) explain duplicate
checks and the limits of split isolation. Resume restores optimizer/RNG state but does not
restore the exact DataLoader shuffle position.

## Key experimental findings

| Experiment | Measured result | Interpretation |
| --- | --- | --- |
| Controlled training, FP32 Apple MPS | Confirmatory A→D whole-step latency fell from **17.881740 to 15.933313 ms**, a **10.896% reduction (1.122x speedup)**. The earlier session measured 11.865%. | Approximately 11% for this synthetic small-model workload. Deferred gradient-norm host reads were the most consistent increment; fused AdamW was less stable, and deferring loss reads added no clear incremental benefit. |
| Preallocated versus `torch.cat` KV cache, MPS | 18 workloads, 216 measured executions, 108 matched pairs; median paired decode speedup **1.012x**, median workload-level speedup **1.003x**, and 60/108 wins. Maximum recorded logit difference: **0**. | Removing repeated historical K/V copying **did not reliably improve decode latency**. `torch.cat` remains the default; preallocation remains explicitly experimental. |

The training result shows why synchronization and optimizer overhead deserve measurement.
The neutral cache result shows why reducing theoretical data movement does not establish a
wall-clock benefit. Neither result establishes model-quality gains, production throughput,
or hardware-general performance.

Read the [controlled training report](docs/training_optimizations.md) and
[raw confirmatory timings](docs/benchmarks/training-optimizations/run-003/steps.csv), plus the
[KV-cache report](docs/preallocated_cache.md) and
[raw paired timings](benchmarks/cache-preallocated/run-001/samples.csv).
The older 19.1% training figure is a historical observation with incomplete evidence,
not the controlled result. Earlier fixed-cache batching measurements are also distinct from
the fresh-request, growing-cache experiment. [Performance notes](docs/performance.md) and
[inference notes](docs/inference.md) retain that context.

## Reproducing the results

Use Python 3.12+ and Poetry, from the repository root. `poetry install` uses the committed
lockfile and installs the package and development tools. Recorded performance runs used
Python 3.12.15 / PyTorch 2.14.0 on Apple MPS; exact per-run environments and flags are in the
reports. CPU tests work without MPS; accelerator-specific tests skip when unavailable.

```bash
poetry install
poetry run pytest -q
poetry run ruff check src tests scripts train.py generate.py serve.py
poetry run ruff format --check src tests scripts train.py generate.py serve.py
```

Train and generate locally; the first training run needs access to the configured Hugging
Face dataset/tokenizer. Checkpoints are generated locally and are not distributed with Git.
`device: auto` selects CUDA, then MPS, then CPU; reduced-precision support is checked on the
selected backend.

```bash
poetry run python train.py --config configs/tiny.yaml
poetry run python train.py --config configs/tiny.yaml --resume checkpoints/step-001000.pt
poetry run python generate.py --checkpoint checkpoints/step-002000.pt \
  --prompt "Once upon a time" --max-new-tokens 50 --temperature 0.8 --top-k 40
```

Resume is optional and continues to the configured final global step. The generation CLI
uses the uncached sampling reference; cached generation is available through the Python
APIs and FastAPI baseline. Load only trusted checkpoints: the training/generation CLI
supports legacy checkpoint deserialization. [Checkpoint evaluation](docs/final_evaluation.md)
records a separate local checkpoint; its filename is not the default training output.

The controlled training and cache benchmarks use synthetic inputs and need no checkpoint
or dataset. On an MPS-capable host, reproduce into **new** output directories:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
poetry run python scripts/benchmark_training_optimizations.py \
  --rounds 4 --output benchmarks/training-optimizations/reproduce-final

env -u PYTORCH_ENABLE_MPS_FALLBACK -u PYTORCH_MPS_FAST_MATH \
  -u PYTORCH_MPS_PREFER_METAL PYTHONPATH=src \
  poetry run python -m scripts.benchmark_preallocated_cache \
  --device mps --output benchmarks/cache-preallocated/reproduce-final
```

For the cache comparison, all three recorded backend flags were unset; inspect the
[environment table](docs/preallocated_cache.md#environment-as-recorded) before reproducing.
The runners refuse existing output directories. Reproduction means repeating the method,
not guaranteeing identical timing. Detailed reports describe warmups, ordering, precision,
correctness gates, and the exact archived training script versions.

An earlier cached/uncached diagnostic accepts an explicitly selected checkpoint:

```bash
poetry run python scripts/benchmark_cached_inference.py --device mps \
  --checkpoint checkpoints/step-002000.pt
```

This diagnostic includes contexts beyond training length; it measures systems behavior,
not long-context quality. [Inference notes](docs/inference.md) cover prefill/decode and
static batching; the [cache-growth report](docs/cache_growth.md) documents Phase 4A.
Committed evidence can be inspected without rerunning any benchmark.

## Repository structure

| Path | Purpose |
| --- | --- |
| [`src/transformers_from_scratch/`](src/transformers_from_scratch/) | Model, attention/RoPE, data, training, checkpointing, generation, and serving |
| [`train.py`](train.py), [`generate.py`](generate.py), [`serve.py`](serve.py) | Training, sampling, and API entry points |
| [`configs/`](configs/), [`tests/`](tests/) | Run configurations and correctness coverage |
| [`scripts/`](scripts/) | Benchmark runners, profiler, data checks, and device diagnostics |
| [`docs/`](docs/), [`notes/`](notes/) | Experimental reports and architecture notes |
| [`docs/benchmarks/`](docs/benchmarks/) | Archived controlled training evidence, source snapshots, and integrity manifests |
| [`benchmarks/cache-growth/run-001/`](benchmarks/cache-growth/run-001/), [`benchmarks/cache-preallocated/run-001/`](benchmarks/cache-preallocated/run-001/) | Explicitly tracked raw cache evidence; other generated benchmark outputs are ignored |

## Limitations and future directions

This is a small research implementation, not a production LLM or a production serving
system. Measurements are specific to the recorded Apple MPS workload and sessions; sample
counts are modest, and no statistical-significance or cross-hardware claim is made.
Logical copy accounting is not measured physical memory traffic, and boundary memory
counters are not peak usage. Dataset/tokenizer identifiers do not pin immutable upstream
revisions, and short numerical checks do not establish long-horizon model quality.

Continuous batching, paged/evicting caches, compilation, quantization, and larger-model or
cross-hardware evaluation are possible future directions, not implemented or validated
outcomes. The project’s interview overview is in [Project Summary](docs/project_summary.md).
