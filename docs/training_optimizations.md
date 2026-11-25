# Controlled training optimization benchmark

The confirmatory MPS experiment measured **10.896% lower whole-step latency (1.122x
speedup)** from A to D. An earlier session measured 11.865%. The defensible headline is
**approximately 11%** for this synthetic FP32 workload. This is a controlled reconstruction,
not an exact reproduction of the historical 19.1% claim.

## Workload and environment

Both sessions ran on an Apple M5 Pro MacBook Pro (`Mac17,8`), 18 CPU cores, 20 GPU
cores, and 48 GB unified memory; macOS 26.5.2 arm64, Python 3.12.15, PyTorch 2.14.0
(build `08187d9e0fba026dc8217405802ab5381dc88d90`), six PyTorch CPU threads.
MPS fallback and MPS fast math were disabled. Full environment and worker metadata are
preserved in the linked artifacts below.

`TinyDecoderLM` has vocabulary 8,192, width 128, four layers, four heads, FFN width 512,
and 2,888,448 parameters. Batch size is 16, context length 128 (2,048 tokens/update),
precision FP32. The existing language-model loss is used. AdamW uses learning rate
3e-4, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8, `foreach=False`, and global
norm clipping at 1.0; one batch per update. Model seed 42 and batch seed 43 are fixed.
Synthetic CPU batches are generated before timing. Every worker starts from identical
weights and fresh, empty optimizer state. No dataset, tokenizer, or checkpoint is needed.

## Variants

| Variant | AdamW | Clipped gradient-norm scalar | Loss scalar |
| --- | --- | --- | --- |
| A | Non-fused | `.item()` each step | `.item()` each step |
| B | Non-fused | Deferred | `.item()` each step |
| C | Fused | Deferred | `.item()` each step |
| D | Fused | Deferred | Deferred |

All variants execute `clip_grad_norm_`; A reconstructs the actual clipped training path,
not the standalone `global_grad_norm()` loop. Here “deferred” means intermediate host
reads are omitted and final metrics are extracted after the loop. This does not simulate
periodic production logging. Model, clipping, optimizer hyperparameters, and input batches
are otherwise identical. Implementation: [benchmark script](../scripts/benchmark_training_optimizations.py).

## Correctness before timing

The MPS gate passed in both sessions before any timing workers launched. A failure aborts
worker execution. It checks three consecutive common-start updates in active and inactive
clipping cases, restoring A's starting weights and optimizer state for each comparison.
Loss, gradients, gradient norms, clipped gradients, and moments retain strict comparisons;
A/B also retain strict parameter and update-delta comparisons. Shapes, dtypes, counters,
and finite values are checked.

Each optimizer is also replayed from identical parameters, clipped gradients and moments
against an independent CPU FP64 AdamW oracle. Fused endpoint errors use an elementwise
FP32 rounding budget incorporating parameter scale, weight decay, moment arithmetic and
denominator sensitivity, rather than a larger global update-delta tolerance. See the
[arithmetic policy](optimizer_correctness.md) for equations and assumptions.

Ten-step independent trajectories report parameter max/RMS differences and loss divergence
separately from arithmetic correctness. A/B reproducibility thresholds trigger strict
common-input replay if exceeded; neither successful session needed replay. Each report
contains 36 common-start comparisons and 120 trajectory records. In run-003 the maximum
common-start loss difference was zero, gradient difference 9.31e-10, parameter/update
difference 4.77e-7, and moment difference 4.66e-10. Passing these short checks does not
certify long-horizon training quality or bitwise trajectory identity.

## Experimental design and timing

Each variant/repetition uses a fresh Python worker with 10 excluded warmup updates and
140 measured updates. Warmup measurements remain in CSV with an explicit flag.
Run-002 used three rounds (`ABCD`, `BCDA`, `DACB`): 12 workers, 1,680 measured steps.
Run-003 used four rounds (`ABCD`, `BCDA`, `CDAB`, `DABC`): **16 workers, 2,240 measured
steps**. Each variant appears exactly once in each execution position in run-003.
This balances position, not all possible predecessor/carryover effects.

`perf_counter_ns` starts after MPS synchronization and stops after the final MPS
synchronization. Whole-step latency includes CPU-batch transfer, zeroing gradients,
forward/loss, selected scalar reads, backward, clipping and optimizer update. There are
no component-level timing barriers. Row recording, final metric extraction, memory queries,
artifact writing and console output occur outside measured intervals; no profiler runs.

For each worker, p50 and p95 use linear-interpolated percentiles of its 140 measured
latencies. The primary statistic is the median of worker p50s; the tail summary is the
median of worker p95s, **not pooled p95**. Variability below is sample standard deviation
of worker p50s. Tokens/s = 2,048 / primary latency in seconds. Relative reduction is
`100 * (1 - candidate / baseline)`; speedup is `baseline / candidate`. JSON also retains
per-worker IQR, median absolute deviation, ranges and aggregate throughput. These are
whole-step observations, not sums of component medians.

## Results

| Variant | Run-002 p50 (ms) | Run-003 p50 (ms) | Run-003 p95 summary (ms) | Run-003 p50 SD (ms) | Run-003 tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 17.681500 | 17.881740 | 18.922374 | 0.180562 | 114,530 |
| B | 16.493750 | 16.322031 | 17.828998 | 0.166192 | 125,475 |
| C | 15.883896 | 15.987073 | 16.787443 | 0.358604 | 128,103 |
| D | 15.583542 | 15.933313 | 18.686347 | 0.752520 | 128,536 |

| Comparison | Run-002 latency reduction | Run-003 latency reduction |
| --- | ---: | ---: |
| A → B | 6.717% | 8.722% |
| B → C | 3.697% | 2.052% |
| C → D | 1.891% | 0.336% |
| A → D | 11.865% (1.135x) | 10.896% (1.122x) |

A→B improves consistently at the experiment-summary level, not in every individual round.
B→C is less stable; C was slower than B in the first run-003 round. C→D's incremental
benefit is inconclusive: only 0.054 ms in run-003, with D slower in two of four rounds.
**C has better p95 behavior than D by the run-003 tail summary**, although not in every
repetition. D's greater variability argues against claiming it is universally preferable.
Run-002 A/B p50 standard deviations were 1.409/1.242 ms, motivating confirmation.

## Evidence and provenance

| Session | Metadata | Raw timings | Summary statistics | Correctness |
| --- | --- | --- | --- | --- |
| Run-002 | [JSON](benchmarks/training-optimizations/run-002/metadata.json) | [CSV](benchmarks/training-optimizations/run-002/steps.csv) | [JSON](benchmarks/training-optimizations/run-002/summary.json) | [Summary](benchmarks/training-optimizations/run-002/correctness-summary.json), [full JSON.gz](benchmarks/training-optimizations/run-002/correctness.json.gz) |
| Run-003 | [JSON](benchmarks/training-optimizations/run-003/metadata.json) | [CSV](benchmarks/training-optimizations/run-003/steps.csv) | [JSON](benchmarks/training-optimizations/run-003/summary.json) | [Summary](benchmarks/training-optimizations/run-003/correctness-summary.json), [full JSON.gz](benchmarks/training-optimizations/run-003/correctness.json.gz) |

Both runs record Git revision `9d35357ad7f6d98027470ee4a27419c99149c5f5` with a dirty
working tree: the benchmark was uncommitted. Exact source snapshots therefore establish
provenance beyond the Git revision:

- Run-002 SHA-256: `d0b0723fcf722882f0829e13b09230c96246ee92587291f473738f22f0d5146f`
- Run-003 SHA-256: `e84ececd6abc51fcbaa1c4c19e1bcd7a2922def7596b8b9ea8c262e2edfb26d2`

The [manifest](benchmarks/training-optimizations/manifest.json) inventories file sizes and
SHA-256 hashes, including both byte-exact [source snapshots](benchmarks/training-optimizations/source-snapshots/).
Snapshots have `.py.txt` extensions to prevent automatic formatter changes. Metadata,
CSV and summary copies preserve original bytes; full correctness JSON is compressed
losslessly and its uncompressed SHA-256 is recorded. Correctness summaries are navigation
exports, not replacements for complete traces. Initialization and synthetic-batch hashes
match across sessions and are recorded in metadata. All original ignored artifacts,
including failed gates and diagnostics, remain unchanged under
`benchmarks/training-optimizations/`.

## Reproducible commands

From the repository root in normal macOS Terminal, with the recorded Poetry environment
and an MPS-capable PyTorch build, use new output directories:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
poetry run python -B scripts/benchmark_training_optimizations.py \
  --correctness-only --rounds 4 \
  --output benchmarks/training-optimizations/verify-new

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
poetry run python -B scripts/benchmark_training_optimizations.py \
  --rounds 4 --output benchmarks/training-optimizations/reproduce-new
```

These commands are documentation, not additional measurements. The script rejects missing
MPS rather than falling back to CPU. `--rounds 3` selects the earlier schedule. For the
exact historical script, restore its archived bytes to the same script path in a separate
checkout before execution; running a snapshot from its archive directory changes path
resolution. Preserve the matching dependency lockfile and inspect recorded environment
metadata; seeds alone do not ensure backend determinism or identical timing.

## Limitations and historical comparability

These are synthetic, small-model, FP32, MPS-specific results. They establish neither
real-data training throughput nor training quality, general GPU behavior, or peak throughput.
Memory counters are boundary snapshots; true peak memory was not measured. Only three/four
repetitions were collected; no statistical significance or confidence interval is claimed.

Ten warmup steps may not fully stabilize compilation, dispatch, power or temperature.
Run-002's first A p50 was 20.064 ms versus 17.681/17.569 later; run-003's first C/D p50s
were 16.680/17.327 ms versus later values around 16 ms. Warm-up/stabilization, thermal
conditions, background activity and execution order are possible explanations, not proven
causes. Position balance reduces one bias but cannot remove these effects. All measured
samples are retained without post-hoc exclusions.

The historical 20.699→16.745 ms figures imply 19.1%, but the final value has no retained
raw step series (the baseline log remains at `benchmarks/scaling/context128-batch16.log`).
Historical runs used TinyStories/DataLoader inputs and synchronized
component profiling; these runs use deterministic synthetic batches and whole-step timing.
Exact historical hardware/source provenance is incomplete. Present the old values as
historical documented observations, and the approximately 11% result as the independently
recorded controlled reconstruction. See [historical performance notes](performance.md).
