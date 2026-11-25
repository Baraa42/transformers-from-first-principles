# Phase 4B: preallocated request-local KV cache

Removing repeated historical K/V copying did **not reliably improve MPS decoding
latency** in Phase 4B run-001. Keep `torch.cat` as the default reference; retain
preallocation as an explicit experimental option. No production behavior is changed
by this report.

The default attention, block, and torch.cat cache implementation remain the
reference. Opt in using `cache_mode="preallocated"` on `TinyDecoderLM.forward`,
`generate_cached`, or `generate_greedy_cached`. Training and default inference
calls retain their previous behavior.

## Measured MPS result: run-001

The [raw samples](../benchmarks/cache-preallocated/run-001/samples.csv),
[summary](../benchmarks/cache-preallocated/run-001/summary.json),
[correctness checks](../benchmarks/cache-preallocated/run-001/correctness.json), and
[metadata](../benchmarks/cache-preallocated/run-001/metadata.json) are the evidence.
The run started at `2026-11-25T12:30:20.687354+00:00` on Apple MPS, FP32,
`Mac17,8`, 51,539,607,552 bytes of unified memory, macOS 26.5.2 arm64,
Python 3.12.15, PyTorch 2.14.0, and six Torch CPU threads. PyTorch Git revision:
`08187d9e0fba026dc8217405802ab5381dc88d90`.

All 18 workloads passed the stepwise gate. The maximum absolute cat/preallocated
logit difference recorded across those checks was **0.0**. The gate also checked
exact greedy tokens, per-layer K/V contents, logical lengths and stable buffer
addresses. All 216 measured executions passed token and cache-size assertions.

Speedup is `cat latency / preallocated latency`; values above 1 favor preallocation.
Each matched pair shares workload, input and repetition. Workload speedup is the
median of its six paired ratios, rather than a ratio of separate latency medians.

| Measure | Observed value |
| --- | ---: |
| Workloads / measured executions / matched pairs | 18 / 216 / 108 |
| Median of all 108 paired decode speedups | 1.012323x |
| Median of 18 workload-level paired decode medians | 1.003129x |
| Pairs with faster preallocated decode | 60 / 108 |
| Batch 1 median paired decode speedup | 0.986119x |
| Batch 4 median paired decode speedup | 1.020610x |
| Range of individual paired decode speedups | 0.651584–1.586886x |
| Median paired prefill speedup | 0.957925x |
| Median paired total-latency speedup | 1.007206x |

Prefill includes allocation and population of preallocated buffers. Its median
paired ratio favors cat. Small decode differences, mixed wins across workloads
and batches, and the wide paired range provide **no robust overall performance
improvement**. This is one session on a tiny model with synthetic prompts;
it does not establish a statistically significant benefit or identify a dominant
kernel cost. Eliminating logical prefix recopies alone did not yield a reliable
MPS latency improvement. The observations do not establish why: kernel dispatch,
validation, layout, allocation, attention and measurement variability remain
possible contributors, not measured causal explanations.

### Workload evidence

Latency columns are each mode's median decode ms/token over six repetitions,
using `N-1` decode forwards. The speedup column is the median paired ratio and
therefore need not equal the ratio of the two displayed latency medians.

| P | N | B | Cat ms/token | Preallocated ms/token | Paired decode speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 1 | 3.260 | 3.300 | 0.990x |
| 32 | 16 | 4 | 3.369 | 3.289 | 1.017x |
| 32 | 64 | 1 | 3.048 | 3.041 | 0.985x |
| 32 | 64 | 4 | 3.077 | 3.006 | 1.017x |
| 32 | 128 | 1 | 2.807 | 3.055 | 0.970x |
| 32 | 128 | 4 | 2.996 | 2.998 | 1.003x |
| 128 | 16 | 1 | 2.988 | 2.857 | 1.051x |
| 128 | 16 | 4 | 2.826 | 2.752 | 1.021x |
| 128 | 64 | 1 | 3.240 | 3.153 | 1.003x |
| 128 | 64 | 4 | 3.547 | 3.267 | 1.023x |
| 128 | 128 | 1 | 2.612 | 2.751 | 0.940x |
| 128 | 128 | 4 | 3.198 | 2.930 | 1.113x |
| 256 | 16 | 1 | 2.905 | 3.281 | 0.923x |
| 256 | 16 | 4 | 3.391 | 3.170 | 1.094x |
| 256 | 64 | 1 | 2.946 | 2.967 | 1.002x |
| 256 | 64 | 4 | 3.010 | 3.035 | 0.980x |
| 256 | 128 | 1 | 2.654 | 2.571 | 1.032x |
| 256 | 128 | 4 | 2.675 | 2.627 | 1.001x |

Per-workload prefill/decode/total distributions, aggregate decode throughput,
end-to-end throughput and all six speedup samples remain in the linked summary;
no single pooled throughput is presented as representative of this mixed matrix.

### Cache bytes and memory evidence

For `P=256, N=128, B=4`, both modes end with 6,275,072 logical and reserved
cache bytes. Preallocation reserves that capacity during prefill, when only
4,194,304 bytes are logically valid; cat's prefill storage equals its logical
size. The old-prefix payload no longer recopied during decode is theoretically
663,764,992 bytes. This is logical copy accounting, **not measured physical
traffic savings**, and does not imply a latency saving.

Across all recorded measurement boundaries, MPS `current_allocated_bytes` ranged
from 11,570,432 to 17,858,048; `driver_allocated_bytes` ranged from 1,141,571,584
to 1,149,960,192; `recommended_max_bytes` was 40,200,896,512 throughout.
`peak_allocated_bytes` was `null` (unavailable). These are whole-process boundary
counters, not peak usage or isolated cache/kernel allocation measurements.

### Environment as recorded

These are exact JSON values from each run's metadata. `null` means the variable
was absent from the benchmark process environment; it does not mean an explicit
`"0"`, nor does it establish the backend's effective default setting.

| Variable | Phase 4A run-001 | Phase 4B run-001 |
| --- | --- | --- |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `"0"` | `null` |
| `PYTORCH_MPS_FAST_MATH` | `"0"` | `null` |
| `PYTORCH_MPS_PREFER_METAL` | `null` | `null` |

See [Phase 4A metadata](../benchmarks/cache-growth/run-001/metadata.json).
The run-001 metadata records only these three backend environment variables;
other shell variables must not be inferred from a suggested launch command.
Phase 4A and Phase 4B are separate sessions with different recorded flags.
The performance conclusion uses the within-Phase-4B matched pairs, not a ratio
between historical Phase 4A medians and Phase 4B medians.

### Evidence integrity and provenance

The [integrity record](benchmarks/cache-comparison-run-001-integrity.json) records
SHA-256 checksums of all eight original Phase 4A/4B run files and the independently
recomputed numerical aggregates. The original run directories are retained
unchanged. Raw artifacts currently live under the Git-ignored `benchmarks/`
tree; include those eight exact files explicitly if committing the report so its
raw-evidence links remain available in a fresh checkout.

All 17 Phase 4B recorded source hashes match the current source files. Of the
seven Phase 4A source hashes, five match current files; `model.py` and
`generation.py` match their versions at recorded Git revision
`7dbc19d0dfa690505c7244edf91e747474059ba8`, before the opt-in cache additions.
A Git revision alone is insufficient to reproduce Phase 4B: its metadata records
an uncommitted working tree, and the source hashes identify the measured files.
Both runs' initial-weight hashes and all six prompt hashes were regenerated
from their respective recorded algorithms, dimensions and seeds and matched.

The audit recomputed pair ordering and counts, every paired ratio and its
summary statistics, mode latency statistics and throughput, the `N-1` timing
denominator, cache shapes/bytes and theoretical old-prefix payloads. It also
verified Phase 4A's 18 reference checks, 90 measured executions and decode
medians. Checksums detect subsequent changes; they are not independent evidence
of historical authenticity or a substitute for a second benchmark session.

## Experimental cache API

```python
with torch.inference_mode():
    logits, cache = model(prompt, use_cache=True, cache_mode="preallocated",
                          cache_capacity=P + N - 1)
    logits, cache = model(next_token, use_cache=True, cache_mode="preallocated",
                          kv_cache=cache)
```

Generation allocates for `P + N - 1`, because prefill selects the first output
token and the final generated token is never forwarded. Explicit model callers
may reserve extra capacity. K/V are projected identically to the reference;
rotated keys and values are copied into independently allocated contiguous
buffers per layer. Only `:cache.length` participates in attention. Decode uses
the logical length as the absolute RoPE offset and writes only the new slot.
`cache.logical_layers` exposes prefix views; `cache.layers` exposes reserved
buffers, whose unused tails are uninitialized and must never be read.

Caches are owned by one request and model. Append mutates storage and returns a
new prefix handle, invalidating the old handle for decode. Independent requests
or branches require new prefill. No model/global cache is stored. Validation of
all layers precedes writes, including length/capacity, layer count, shape, batch,
dtype, device, contiguity, duplicate storage, ownership and overflow. Cache
buffers and prefix views are intentionally aliased within one request; callers
must not mutate them. Autograd and autocast are explicitly rejected on this
opt-in path; use explicit model dtype and no_grad/inference_mode.

## Correctness and comparison

The tests compare every step against cat and uncached forwards: FP32 logits
(`rtol=1e-5, atol=1e-6`), exact greedy tokens, each layer's K/V contents, explicit
first-layer RoPE offsets, logical lengths, stable storage pointers, overflow,
invalid metadata/buffers, stale-handle rejection and independent requests.
Unused tails are poisoned with NaNs to detect attention beyond logical length.
CPU and available MPS devices use the same tests. Tests also verify that only
prefill allocates the reserved K/V buffers and decode never invokes a K/V cat.

Run from the repository root:

```sh
poetry run pytest -q tests/test_preallocated_cache.py tests/test_preallocated_cache_benchmark.py
poetry run pytest -q
poetry run ruff check src tests scripts
```

The completed run used this benchmark entry point (shown for provenance;
no new benchmark was run while finalizing this report):

```sh
PYTHONPATH=src poetry run python -m scripts.benchmark_preallocated_cache \
  --device mps --output benchmarks/cache-preallocated/run-001
```

The runner refuses to overwrite an existing output directory. Do not rerun into
the preserved `run-001` directories. Phase 4A files and harness are preserved.
The matrix matches Phase 4A: P=32/128/256, N=16/64/128, B=1/4, the same FP32
model dimensions and seeds. Each workload warms each mode independently with
two full fresh requests, then performs six paired repetitions, alternating
cat/preallocated and preallocated/cat. This produces 216 measured requests.
Every workload passes the stepwise correctness gate before any measurement.
Timed warmups and measured sequences are also checked against the gate's tokens.

Timing synchronizes only around prefill and the entire N-1 step decode block.
Allocation and prefill buffer population remain inside prefill/total latency.
Argmax and growing output-ID concatenation are included in both implementations.
Diagnostics, memory counters and CPU token reads occur outside timers.

Artifacts contain raw samples, execution order, configuration, source/weight/
prompt hashes, environment metadata, correctness diagnostics and summaries.
Summaries report separate prefill/decode/total distributions, median decode
ms/token, all six paired speedup ratios (cat/preallocated) and their spread,
aggregate decode and end-to-end throughput, logical/reserved cache bytes,
available MPS process memory counters, and theoretical old-prefix bytes
eliminated. Reserved storage is visible at prefill, when it exceeds logical
storage; with exact capacity they coincide at the final decode step.

Preallocation adds 2L buffer allocations and 2L slice copies at prefill. Decode
uses 2L slice writes instead of 2L allocating cats; prefix views, validation,
projections, RoPE, attention and output-ID allocations remain. Algorithmic
operation counts are reported, but allocator/kernel events are not profiled.
Noncontiguous valid-prefix views can also affect backend attention performance.
MPS counters are process-wide boundary readings, not peak measurements or
per-operation allocation counts. Theoretical old-prefix bytes eliminated are
logical payload accounting, **not measured physical traffic savings**. No
speedup is assumed from that reduction; use the paired distributions to assess
performance and measurement noise.


## Finalization validation

The documentation-only finalization passed the full test suite: 255 passed,
10 skipped, with one existing Starlette/httpx deprecation warning. MPS was
unavailable in the validation runtime; the completed MPS run above is preserved
raw evidence, not a new measurement from these checks. Ruff passed for `src`,
`tests`, and `scripts`. All README/report relative links resolved, all eight
original artifact checksums remained unchanged, and no production source,
benchmark runner, or test source was modified during finalization. No new
benchmark was launched, and nothing was committed or pushed.
