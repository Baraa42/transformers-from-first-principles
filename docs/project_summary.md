# Project summary for ML research and systems interviews

## 30-second explanation

I implemented a small decoder-only Transformer directly in PyTorch and built its training
and inference systems around it: TinyStories data processing, checkpoint/resume, mixed
precision, autoregressive generation, and request-local KV caching. I used numerical
references and controlled Apple MPS experiments to separate correctness from performance.
A confirmatory training experiment reduced whole-step latency by about 11%, while replacing
cache concatenation with preallocation gave no reliable decode speedup. The project shows
how I test an optimization hypothesis and retain the evidence even when the result is neutral.

## Technical two-minute explanation

The configured model has 2.89 million parameters: four pre-norm decoder blocks, width 128,
four attention heads, RoPE on queries and keys, SiLU MLPs, and untied embeddings/output
weights. Attention, masking, and cache handling are explicit PyTorch operations rather
than a high-level Transformer implementation. TinyStories stories are split before
within-story next-token windowing, then trained with AdamW, accumulation, clipping,
optional mixed precision, and token-weighted validation. Checkpoints restore model,
optimizer, RNG, and optional scaler state; exact DataLoader position is not restored.

For training performance, I examined synchronization and optimizer overhead, then compared
four variants from identical weights and synthetic batches. The confirmatory run used
fresh workers, balanced execution positions, ten warmup updates and 140 measured updates
per worker. Deferring gradient-norm host reads, using fused AdamW, and deferring loss reads
together reduced the median of worker p50 latencies from 17.882 to 15.933 ms. Common-start
checks and an independent CPU FP64 AdamW oracle checked arithmetic separately from
short independent training trajectories. The final loss-read increment was inconclusive,
so I do not attribute the full gain to each component or claim long-horizon equivalence.

For inference, each layer caches already-rotated keys and values. Single-token decode
uses cache length as the absolute RoPE offset. I retained `torch.cat` as the reference and
added an explicit preallocated mode with separate logical length/capacity, bounded slice
writes, whole-cache validation before append, and stale-handle rejection. Independent
requests receive independent storage. The benchmark warmed both modes and alternated
measurement order across six matched repetitions per workload. It used real growing
sequences and synchronized prefill/decode blocks, with allocation inside prefill time.
Across 108 pairs, median decode speedup was 1.012x and only 60 pairs favored preallocation;
recorded logits were exactly equal. Reducing theoretical copy volume did not reliably
improve MPS latency, so the simpler reference remains the default.

## Three engineering lessons

1. **Correctness gates and timing answer different questions.** Compare common inputs,
   per-layer state and numerical references before timing; report trajectory drift
   separately from one-step arithmetic. Matching logged loss alone is not sufficient.
2. **Measure synchronization at the workload boundary.** Host scalar reads can force
   completion of queued accelerator work. Component-level barriers perturb execution;
   an operator's CPU time does not establish its device-kernel cost.
3. **Reduced algorithmic work is a hypothesis about latency.** Copy elimination,
   preallocation and batching change storage or utilization, but fixed overhead,
   layout and backend behavior can dominate a tiny model. A neutral result still
   informs the default implementation choice.

## Three decisions and trade-offs

1. **Explicit tensor operations and pre-norm blocks.** A manual attention path makes
   masking, tensor shapes and RoPE offsets inspectable and independently testable.
   It prioritizes understanding and correctness over fused production-kernel throughput.
2. **Request-local cache state with an opt-in mutable buffer.** Tuple caches keep the
   reference simple. Preallocation avoids historical prefix recopy but adds capacity
   checks, reserved memory, in-place ownership and stale-handle constraints. Branching
   requires fresh prefill; the measured outcome did not justify changing the default.
3. **Synthetic performance workloads separated from data evaluation.** Fixed inputs and
   fresh workers isolate implementation differences without dataset/checkpoint variability.
   They cannot establish real-data throughput, model quality or production latency.
   The full configured-subset checkpoint evaluation is reported separately.

## Numerical results and boundaries

| Result | Evidence and caveat |
| --- | --- |
| 2,888,448 parameters; context 128, batch 16, FP32 default | [Configuration](../configs/tiny.yaml) and [architecture](../notes/architecture.md). Constructor defaults are smaller inspection settings. |
| Training A→D: 17.881740→15.933313 ms, 10.896% reduction, 1.122x speedup | Confirmatory run-003: 16 workers, 2,240 measured updates; median of four worker p50s per variant, not pooled steps. Earlier run-002: 11.865%. [Report](training_optimizations.md), [CSV](benchmarks/training-optimizations/run-003/steps.csv). |
| Largest historical copy payload removed: 663,764,992 bytes; final logical cache 6,275,072 bytes | P=256, N=128, B=4. The copy figure is theoretical old-prefix payload, not measured physical traffic or time saved. [Baseline](cache_growth.md), [comparison](preallocated_cache.md). |
| Cat/preallocated: 18/18 workloads passed; maximum recorded absolute logit difference 0 | Six paired repetitions per workload: 216 measured executions and 108 pairs. [Correctness artifact](../benchmarks/cache-preallocated/run-001/correctness.json). |
| Median paired decode speedup 1.012x; median workload median 1.003x; 60/108 wins | B=1 median 0.986x; B=4 median 1.021x. No reliable overall improvement; one MPS session, modest sample count. [Raw timings](../benchmarks/cache-preallocated/run-001/samples.csv). |
| Local checkpoint validation loss 3.232746; perplexity 25.349173 | Historical full evaluation of the configured 1,000-story validation subset, 8,980,736 target tokens. Not an independent test set; checkpoint is not distributed. [Evaluation provenance](final_evaluation.md). |

The controlled training headline is approximately 11%, not the older documentation-only
19.1% figure. Historical fixed-cache batching timings are not end-to-end generation
throughput and must not be combined with the growing-cache experiment. Recorded environment
flags differ between cache phases; use the paired Phase 4B data for the cache comparison.
No result establishes cross-hardware generalization, production scalability, long-context
quality, or improved model quality from systems changes.

## Code and reports to discuss

- Model mechanics: [attention](../src/transformers_from_scratch/attention.py),
  [RoPE](../src/transformers_from_scratch/rope.py),
  [decoder blocks](../src/transformers_from_scratch/layers.py),
  [model](../src/transformers_from_scratch/model.py), and
  [independent numerical tests](../tests/test_numerical_references.py).
- Training correctness: [data pipeline](data.md),
  [training loop](../train.py), [checkpoint handling](../src/transformers_from_scratch/checkpointing.py),
  and [optimizer arithmetic policy](optimizer_correctness.md).
- Controlled performance: [training methodology/results](training_optimizations.md),
  [source/evidence manifest](benchmarks/training-optimizations/manifest.json), and
  [historical diagnostics](performance.md).
- Inference: [generation](../src/transformers_from_scratch/generation.py),
  [preallocated storage](../src/transformers_from_scratch/preallocated_cache.py),
  [paired runner](../scripts/benchmark_preallocated_cache.py),
  [inference notes](inference.md), and [minimal API](serving.md).
