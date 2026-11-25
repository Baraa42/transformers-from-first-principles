# KV-cache growth: baseline and instrumentation

Phase 4A measures realistic sequential decoding with the existing `torch.cat` cache.
It does not implement preallocation or establish a measured cache-copy bottleneck. No MPS
results have been collected for this experiment in the sandbox.

## Implementation audit

- [SelfAttention.forward](../src/transformers_from_scratch/attention.py) projects Q/K/V,
  reshapes to `(B,H,T,d_head)`, rotates Q/K with RoPE, and stores rotated K and unrotated V.
  With a cache, input must be one token; the previous cache length is its RoPE offset.
  Lines 90–91 concatenate both old prefixes and new K/V along sequence dimension 2.
  Each cat creates a fresh output containing the whole concatenation; there is no spare
  capacity or in-place append. Old prefix tensors are not mutated. There are two cat
  operations per layer per decode forward. The allocator may reuse released blocks:
  these operations do not prove that each cat requires a fresh operating-system allocation.
- [TinyDecoderLM.forward](../src/transformers_from_scratch/model.py) validates matching
  cache lengths and passes one tuple to each layer (lines 57–88). The previous model cache
  remains live during its forward while replacement layer caches are constructed. Temporary
  projections, attention scores, logits, and old/new storage coexist; final logical cache
  size is therefore neither instantaneous live memory nor physical peak allocation.
- [generate_cached](../src/transformers_from_scratch/generation.py), lines 150–167, prefills
  once, selects token one, then feeds back one token at a time. Generating `N` tokens
  requires `N-1` growth forwards. The final token is returned without entering the cache.
  `generate_greedy_cached` delegates here; `generate_greedy` is the uncached argmax reference.
  Generated token IDs themselves also grow via cat. EOS checks can synchronize, but this
  benchmark uses greedy decoding with no EOS stopping.
- [benchmark_cached_greedy](../src/transformers_from_scratch/inference_benchmarking.py)
  already measures growing sequential decode for batch one. The existing prefill/decode
  and static-batching helpers instead repeat one forward from an unchanged baseline cache
  (lines 712–723 and 786–793), so they do not measure a sustained growth trajectory.
  Existing [cached/uncached](../scripts/benchmark_cached_inference.py) and
  [batching](../scripts/benchmark_decode_batching.py) scripts load a checkpoint/tokenizer.
  The shared [loader](../scripts/benchmark_inference.py) uses unrestricted checkpoint
  deserialization and tokenizer loading; this experiment does not call it or access network
  data. Previous benchmarks are unchanged.
- Existing [attention](../tests/test_attention.py) and [model](../tests/test_model.py) tests
  verify prefix preservation, shapes and cached/full outputs. [Generation tests](../tests/test_generation.py)
  verify batched exact tokens, context boundaries and one-token decode inputs.
  [Independent numerical tests](../tests/test_numerical_references.py), particularly
  `test_cached_logits_and_layer_positions_against_independent_decoder`, compare per-layer
  K/V, positions and logits against a separate mathematical decoder reference.
  [Benchmark tests](../tests/test_inference_benchmark.py) already check fixed-cache reuse
  and the `N-1` timing denominator. These paths are reused, not replaced.

## Theoretical payload accounting

Let prompt length be `P`, number of generated tokens `N >= 1`, batch size `B`, layer
count `L`, width `d_model`, and element size `s` bytes. Let `q=N-1` and
`C=2*B*L*d_model*s` bytes per cached position, counting both K and V across all layers.
Standard multi-head attention uses `H*d_head=d_model`; heads cancel from the byte count.
At growth forward `i=1..q`, old length is `P+i-1` and new length is `P+i`.

| Quantity | Logical byte estimate |
| --- | --- |
| Initial prefill cache | `C*P` |
| Final cache | `C*(P+q)` |
| Old prefixes recopied cumulatively | `C*(q*P + q*(q-1)/2)` |
| Cumulative cat output payload (old + new) | `C*(q*P + q*(q+1)/2)` |
| Idealized cat reads plus writes | `2*C*(q*P + q*(q+1)/2)` |
| Newly produced K/V payload | `C*q` |
| K/V cat operation count | `2*L*q` |

The read-plus-write estimate assumes each logical element is read/written once by cat.
It is not measured DRAM traffic: layout, caches, kernels, allocator reuse and backend
implementation affect physical behavior. It excludes projection/RoPE, attention, prefill
writes, token-ID concatenation, allocator bookkeeping and other intermediates.
Old-prefix recopy grows as `O(B*L*d_model*s*(P*N + N^2))`, while retained logical cache
size is `O(B*L*d_model*s*(P+N))`. `N=1` gives no growth copies.

For FP32, `L=4`, `d_model=128`, the largest matrix case (`P=256,N=128,B=4`) has a
final logical cache of 5.984375 MiB, cumulative cat output payload of 635 MiB, and
idealized read-plus-write volume of 1,270 MiB. These are algebraic estimates, not timings.

## Focused experiment

[benchmark_cache_growth.py](../scripts/benchmark_cache_growth.py) uses the production
model and attention with random initialized FP32 weights: vocabulary 8,192, width 128,
four layers, four heads, FFN 512. It reuses the production argmax selector, greedy cached
and uncached references, device synchronization helper and logical cache-size helper.
The benchmark loop is instrumented separately so it can inspect the final runtime cache.
Reference token comparisons share production components and are integration checks;
independent mathematical coverage remains in the existing numerical tests.

The full Cartesian matrix is `P={32,128,256}`, `N={16,64,128}`, `B={1,4}`: 18 workloads.
Weights use seed 42, synthetic prompts seed 43, and workload order seed 44. Each `(P,B)`
prompt is shared across generated lengths and repetitions. Inputs are created and moved
to device outside timing. There is no checkpoint, dataset or training. Each workload gets
two full-sequence warmups and five measured sequential generations, each restarting from
its prompt. All 18 reference comparisons run before warmups/timing. Each measured output
must also exactly match the uncached reference or the experiment fails.

Prefill timing includes the prompt forward, first argmax and initial output-ID cat.
Decode timing includes all `q` sequential cached forwards, argmax selections and token-ID
cats. Timers use `perf_counter_ns` with device synchronization only at each block boundary;
there is no synchronization, host scalar read, logging, profiling or memory inspection
inside the per-token loop. Unused logits are released by the runner. Model setup, references,
diagnostics and artifact writing are excluded. `total_latency_ms` sums prefill and decode
blocks, excluding the diagnostic gap between them.

Decode ms/token is `decode_ms/q`, the latency of one batched decode iteration.
Aggregate throughput is `B*q*1000/decode_ms`; per-sequence throughput omits `B`.
These metrics exclude token one, which is produced by prefill. Summary p50/p95 describe
whole-sequence decode durations across five repetitions, not individual-token tails.
P95 uses linear interpolation; sample standard deviation and ranges show variability.

Artifacts in a fresh output directory:

- `metadata.json`: configuration, seeds, order, environment, selected backend flags,
  Git revision/status, source hashes, initial-weight/prompt hashes and timing definitions.
- `correctness.json`: reference tokens, exact equality results, observed initial/final
  cache shapes, logical tensor bytes, unique storage capacity and memory boundary counters.
- `samples.csv`: every measured prefill/decode duration and derived rate; warmups excluded.
- `summary.json`: per-workload p50/p95, variability, rates, final cache size and theoretical
  copy volumes. Copy estimates are explicitly named `theoretical_*`.

MPS diagnostics use available current/driver allocation counters and recommended memory.
Unavailable values and true peak allocation are null; storage capacity is tensor-visible
storage, not allocator-rounded physical memory. No operator allocation events or isolated
cat duration are measured. Boundary counters include model/temporary/allocator state and
cannot identify transient peaks. This is a single-process baseline; order randomization
and warmups do not eliminate thermal or stabilization effects.

## Local MPS command (not executed here)

From the repository root in normal macOS Terminal:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
poetry run python -B scripts/benchmark_cache_growth.py \
  --device mps --repetitions 5 --warmup-sequences 2 \
  --output benchmarks/cache-growth/run-001
```

The command rejects unavailable MPS and enabled CPU fallback, probes an actual MPS
operation and refuses to overwrite an existing output directory. Explicit CPU mode exists
for tiny verification only; CPU results are labeled as such. No full MPS benchmark was run
during implementation.

## Evidence required before preallocation

First inspect exact-token checks, raw sequential latencies, repetition variability, scaling
with `P/N/B`, final cache sizes and available memory diagnostics. Confirm warmup stability
and repeat selected workloads in a separate session if needed. Increasing latency can
also come from longer attention, dispatch, argmax and token-ID growth; a correlation with
theoretical copied bytes does not isolate cat overhead.

A later controlled cat/preallocated comparison must preserve tokens, RoPE offsets, layer
association, precision, weights, prompts, timing boundaries and output-ID bookkeeping.
Only an independently verified A/B measurement can establish a material latency improvement.
An optional separate allocation trace or isolated copy diagnostic can support attribution,
but its instrumentation must stay outside the authoritative latency experiment. Do not
infer a speedup or percentage of time spent copying from this baseline alone.
