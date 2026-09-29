# Training Performance Notes

These results are empirical measurements for one specific environment and workload. They
are not universal claims about FP16, BF16, PyTorch, or Apple MPS.

- Backend: Apple MPS
- PyTorch: 2.14.0
- Model: `d_model=128`, `n_heads=4`, `d_ff=512`, `n_layers=4`
- Tokenizer: TinyStories BPE8192

## Methodology

The precision benchmark ran the same 300-step training workload in fresh FP32, FP16, and
BF16 processes. The synchronized component profiles used explicit device synchronization
around each measured section, excluded the first 10 successful optimizer steps, and
reported medians and p95 values over the remaining steps.

The batch/context experiment used FP32 synchronized profiling for 150 successful optimizer
steps per workload, again excluding 10 warmup steps. Its derived throughput is:

```text
tokens_per_step = batch_size * context_length
median_tokens_per_sec = tokens_per_step / median_step_seconds
```

Synchronized profiling intentionally perturbs normal asynchronous execution. These timings
are useful for comparisons and decomposition, not as peak-throughput measurements.

Commands used to reproduce the benchmark families:

```bash
poetry run python scripts/benchmark_precision.py
poetry run python train.py --config configs/benchmarks/profile-fp32.yaml
poetry run python scripts/benchmark_scaling.py
```

## Precision benchmark

| Precision | Median tokens/s | vs FP32 | Wall time | Final val loss |
| --------- | --------------: | ------: | --------: | -------------: |
| FP32      |         117,790 |   1.00x |   5.587 s |         4.9455 |
| FP16      |          91,006 |   0.77x |   8.072 s |         4.9454 |
| BF16      |         120,609 |   1.02x |   6.474 s |         4.9454 |

FP16 recorded zero overflows and finished with a GradScaler scale of 65,536.

All three modes followed essentially identical loss trajectories. FP16 was about 23% slower
in steady-state throughput, and the absence of overflows indicates that this slowdown was
not caused by numerical instability. BF16 was about 2% faster than FP32 in steady state,
but its short-run total wall time was worse because startup and AMP overhead mattered at
this scale. Mixed precision is therefore a hardware- and workload-specific optimization,
not an automatic speedup.

## Synchronized component profiling

### FP32 baseline

The baseline profile used `context=128` and `batch=16`.

| Component       | Median ms |  Share |
| --------------- | --------: | -----: |
| zero_grad       |     0.054 |  0.28% |
| batch_fetch     |     0.187 |  0.96% |
| host_to_device  |     0.400 |  2.06% |
| forward_loss    |     4.721 | 24.34% |
| backward        |     7.288 | 37.57% |
| grad_processing |     3.812 | 19.65% |
| optimizer_step  |     2.937 | 15.14% |

```text
step_time_median_ms = 19.771
step_time_p95_ms = 20.539
```

Backward is the largest individual component. Forward plus backward account for about 62%
of measured step cost, while gradient processing plus the optimizer account for about 35%.
Data fetching, transfer, and gradient zeroing together contribute only about 3%, so this
workload is not materially DataLoader-bound.

### Why FP16 was slower

```text
FP32 step median = 19.771 ms
FP16 step median = 24.971 ms
```

| Component       | FP32 ms | FP16 ms |
| --------------- | ------: | ------: |
| forward_loss    |   4.721 |   5.128 |
| backward        |   7.288 |   6.734 |
| grad_processing |   3.812 |   8.553 |
| optimizer_step  |   2.937 |   3.414 |

FP16 backward was slightly faster, but gradient processing increased from about 3.8 ms to
8.6 ms. The measured section includes GradScaler unscale/non-finite checking and the
existing clipping/global-norm work. That increase explains most of the observed FP16
slowdown for this small model, without establishing stronger causality than the component
measurements support.

### BF16 comparison

```text
BF16 median step = 19.366 ms
FP32 median step = 19.771 ms
```

The main BF16 component medians were:

```text
forward_loss    = 4.996 ms
backward        = 6.607 ms
grad_processing = 3.835 ms
optimizer_step  = 2.940 ms
```

BF16 avoids GradScaler overhead. Its backward pass was modestly faster, while gradient
processing and optimizer costs stayed close to FP32. The resulting steady-state step time
improved slightly, although startup and AMP overhead still made the very short BF16 run
slower in total wall time.

## Batch and context scaling

The baseline for relative throughput is `context=128`, `batch=16`.

| Context | Batch | Tokens/step | Median step ms | Median tokens/s | vs baseline |
| ------: | ----: | ----------: | -------------: | --------------: | ----------: |
|      64 |    16 |       1,024 |         14.578 |          70,243 |       0.71x |
|     128 |     8 |       1,024 |         14.840 |          69,003 |       0.70x |
|     128 |    16 |       2,048 |         20.699 |          98,942 |       1.00x |
|     128 |    32 |       4,096 |         30.567 |         134,001 |       1.35x |
|     256 |    16 |       4,096 |         34.806 |         117,681 |       1.19x |

### Batch scaling

At fixed `context=128`, throughput increased from about 69k tokens/s at batch 8 to 99k at
batch 16 and 134k at batch 32. Moving from batch 16 to 32 doubled tokens per step while
latency rose by only about 48%, showing that larger batches materially improved MPS
utilization.

Gradient processing and optimizer-step costs remained almost constant because they depend
primarily on model parameter count rather than examples per batch. Larger batches amortized
these fixed parameter-side costs over more tokens, which is a major reason small batches
were inefficient for this tiny model.

### Context scaling

At fixed `batch=16`, throughput increased from about 70k tokens/s at context 64 to 99k at
128 and 118k at 256. Attention still contains an approximately quadratic `O(T^2)` score
matrix; these measurements do not imply linear attention. Through 256 tokens, improved
utilization and amortization outweighed the additional attention cost.

### Equal-token comparisons

For 1,024 tokens per step:

```text
T=64,  B=16 -> 14.578 ms
T=128, B=8  -> 14.840 ms
```

The costs were nearly identical.

For 4,096 tokens per step:

```text
T=128, B=32 -> 30.567 ms
T=256, B=16 -> 34.806 ms
```

The longer-context workload was approximately 14% slower for the same number of processed
tokens. Sequence-length-dependent attention cost is beginning to become visible at this
larger workload.

## Overall diagnosis

For this tiny Transformer on Apple MPS, the baseline workload under-utilizes the
accelerator. Backward is the largest individual compute component, but parameter-side costs
such as global gradient processing and AdamW are substantial at small workloads. Increasing
batch size improves throughput significantly by amortizing these fixed costs. Longer
contexts also improve utilization through 256 tokens, although equal-token comparisons show
the quadratic attention penalty beginning to emerge. FP16 is counterproductive for this
workload because GradScaler/unscale overhead dominates its compute savings; BF16 is the
cleaner mixed-precision mode and provides a small steady-state improvement.

## Stage 5.3 — Operator profiling and profiler-guided optimization

The operator-level baseline used FP32, context length 128, batch size 16, five warmup
steps, and 15 profiled steps. On Apple MPS, `torch.profiler` exposed CPU-side operator
activity rather than direct MPS kernel timing.

> CPU profiler attribution on MPS should not be interpreted as direct device-kernel
> execution time. Synchronization or waiting can be charged to the CPU operation that
> forces previously queued device work to complete.

### Scalar-synchronization progression

#### Initial implementation

The original profiler run reported:

```text
aten::_local_scalar_dense
810 calls across 15 profiled steps
```

That is 54 scalar reads per step:

```text
810 / 15 = 54
```

The source was the original global-gradient-norm implementation, which performed a host
scalar extraction for every gradient-bearing parameter, effectively calling:

```python
gradient.detach().pow(2).sum().item()
```

inside a loop over model parameters. On MPS, each repeated `.item()` created a
device-to-host synchronization boundary.

#### After moving global-norm reduction on-device

After reducing the global gradient norm on-device, the profile reported:

```text
aten::_local_scalar_dense
30 calls across 15 profiled steps
134.443 ms self CPU
55.52% self CPU
```

Thirty calls over 15 steps is exactly two scalar reads per training step. They corresponded
to:

```python
loss.item()
clip_grad_norm_(...).item()
```

#### After deferring scalar reads to logging boundaries

After device-to-host scalar conversion was deferred to logging boundaries,
`aten::_local_scalar_dense` disappeared from the profiled hot loop. The complete progression
was:

```text
810 -> 30 -> 0
```

The large CPU time attributed to `_local_scalar_dense` did not mean scalar conversion itself
was expensive. These reads forced previously queued MPS work to complete, so the reported
CPU time was mostly synchronization wait. As noted above, the CPU-side profiler does not
expose direct MPS kernel timing.

### Fused AdamW

The Stage 5.3 optimization sequence used a fresh controlled baseline (20.699 ms); it should
not be numerically conflated with the earlier Stage 5.1 component-profile run (19.771 ms).

The original synchronized FP32 baseline and the result after selecting fused AdamW on MPS
were:

| Metric          |  Original | Fused AdamW |
| --------------- | --------: | ----------: |
| median step     | 20.699 ms |   18.068 ms |
| p95 step        | 21.986 ms |   18.852 ms |
| optimizer_step  |  3.154 ms |    1.326 ms |
| grad_processing |  3.890 ms |    3.793 ms |
| forward_loss    |  4.971 ms |    4.721 ms |
| backward        |  7.600 ms |    7.240 ms |
| wall time       |   3.431 s |     2.977 s |

Measured changes were approximately:

```text
optimizer_step: 3.154 -> 1.326 ms  (~58% reduction)
median step:    20.699 -> 18.068 ms (~12.7% reduction)
wall time:       3.431 -> 2.977 s   (~13.2% reduction)
```

Optimizer cost was a substantial fixed overhead for this tiny model, and fused AdamW
materially reduced it. Other major components remained broadly similar. Identical logged
losses showed that training behavior was preserved, although not every residual
run-to-run difference should be attributed to AdamW alone.

### Deferred host scalar reads

The next optimization stopped calling `loss.item()` every training step, retained detached
loss tensors only when logging was due, kept the gradient norm as a device scalar, and
converted loss and norm to Python only at logging boundaries. Gradient clipping continued
to run on every optimizer step.

After this change, `aten::_local_scalar_dense` was no longer among the dominant
profiled-loop operators. The top CPU-side entry became:

```text
aten::copy_
30 calls across 15 profiled steps
```

That call count is consistent with two host-to-device input transfers per step. Its large
CPU attribution should not be interpreted as true transfer latency: after removing earlier
scalar synchronization, waiting can move to a later operation that becomes the next
completion boundary. The synchronized Stage 5.1 measurements continued to show that
host-to-device transfer was only a small fraction of step time.

### Final synchronized baseline

After fused AdamW and deferred host scalar reads, the synchronized FP32 baseline measured:

| Metric          |     Final |
| --------------- | --------: |
| median step     | 16.745 ms |
| p95 step        | 17.166 ms |
| wall time       |   2.755 s |
| zero_grad       |  0.024 ms |
| batch_fetch     |  0.128 ms |
| host_to_device  |  0.348 ms |
| forward_loss    |  4.417 ms |
| backward        |  6.974 ms |
| grad_processing |  3.590 ms |
| optimizer_step  |  1.215 ms |

Logged throughput was:

```text
step 50  = 109,783 tok/s
step 100 = 121,379 tok/s
step 150 = 114,845 tok/s
```

Training losses remained:

```text
step 50  train_loss=6.4355
step 100 train_loss=5.4658
step 150 train_loss=5.0278
```

Validation remained:

```text
step 100 val_loss=5.8570
val_perplexity=349.66
```

### Optimization summary

| Metric          |  Original | Fused AdamW |     Final |
| --------------- | --------: | ----------: | --------: |
| Median step     | 20.699 ms |   18.068 ms | 16.745 ms |
| p95 step        | 21.986 ms |   18.852 ms | 17.166 ms |
| Optimizer       |  3.154 ms |    1.326 ms |  1.215 ms |
| Grad processing |  3.890 ms |    3.793 ms |  3.590 ms |
| Wall time       |   3.431 s |     2.977 s |   2.755 s |

From the original to the final baseline, median synchronized step time fell from 20.699 ms
to 16.745 ms, an approximate 19.1% reduction. Wall time fell from 3.431 s to 2.755 s,
approximately 19.7%, while optimizer-step time fell from 3.154 ms to 1.215 ms,
approximately 61.5%.

## Final Stage 5 systems diagnosis

The initial tiny-model workload under-utilized Apple MPS and carried substantial fixed
parameter-side overhead. Batch scaling improved utilization by amortizing those costs.
Operator profiling then exposed CPU synchronization and optimizer overhead that were not
obvious from coarse timing alone. Switching to fused AdamW materially reduced optimizer
cost, and deferring unnecessary device-to-host scalar reads yielded an additional
end-to-end gain. Across the full Stage 5 optimization sequence, median synchronized step
time improved by roughly 19% while training metrics remained unchanged.

On asynchronous accelerators, CPU profiler attribution must be interpreted carefully:
operations such as scalar reads or copies may appear expensive because they become
synchronization boundaries for previously queued device work.

## Stage 6.4 — Prefill vs Cached Decode

Stage 6.4 measured full-prompt prefill and one-token cached decode independently on Apple
MPS in FP32 with batch size 1. To reduce the fixed synchronization and timer overhead that
distorted the earlier per-operation measurements, each sample timed 50 identical forwards
inside one synchronized block. The reported value is:

```text
median(
    synchronized block total / 50 identical forwards
)
```

The benchmark used five repetitions for each sequence length:

```text
device=mps
precision=fp32
batch_size=1
block_iterations=50
repetitions=5
```

| T   | Prefill ms | Cached decode ms | KV cache MiB |
| --: | ---------: | ---------------: | -----------: |
|  32 |      1.053 |            1.012 |        0.125 |
|  64 |      1.009 |            0.939 |        0.250 |
| 128 |      0.989 |            0.964 |        0.500 |
| 256 |      1.253 |            1.402 |        1.000 |
| 512 |      1.308 |            1.095 |        2.000 |

### Prefill

Prefill processes the entire prompt in parallel. For an input shaped `(B, T)`, the
attention tensors have shapes:

```text
Q, K, V:          (B, H, T, d_head)
attention scores: (B, H, T, T)
```

Constructing and applying the attention score matrix requires approximately `O(T^2)`
work. This statement applies to the attention portion of the model, not the entire
Transformer: projections, MLPs, normalization, and other operations have different
sequence-length scaling.

### Cached decode

For one new token with a cache containing `T` previous positions, the relevant shapes are:

```text
input:             (B, 1)
Q_new:             (B, H, 1, d_head)
K_cache, V_cache:  (B, H, T, d_head)
attention scores:  (B, H, 1, T + 1)
```

Only the new token's Q, K, and V projections are computed; the preceding K/V tensors are
reused. Attention work for each decoded token is therefore approximately `O(T)`, rather
than recomputing attention over the full prefix. Generation still decodes tokens
sequentially because each new token depends on the preceding output.

### KV-cache memory

For a standard multi-head cache, the number of stored K/V elements is:

```text
KV elements = 2 * L * B * T * d_model
```

The corresponding storage is:

```text
KV memory = 2 * L * B * T * d_model * bytes_per_element
```

For this four-layer, `d_model=128` model in FP32, the theoretical cache sizes are:

```text
T=32  -> 0.125 MiB
T=64  -> 0.250 MiB
T=128 -> 0.500 MiB
T=256 -> 1.000 MiB
T=512 -> 2.000 MiB
```

KV-cache memory scales linearly with context length.

### Measurement interpretation

Cached decode remains roughly around 1 ms over the measured range, while prefill begins
increasing somewhat at the larger sequence lengths. Neither curve cleanly exposes its
theoretical asymptotic scaling. This is not contradictory: the benchmark uses a very small
four-layer, `d_model=128`, batch-1 model on MPS. Fixed overhead, kernel dispatch,
utilization, and non-attention operations remain a large fraction of the total runtime.

The algorithmic complexity difference is real, but it only becomes clearly visible in
wall-clock latency once the sequence-dependent work becomes a significant fraction of
total runtime. These measurements do not isolate a specific hardware bottleneck.

The earlier Stage 6.3 end-to-end cached-versus-uncached comparison reported:

```text
P=32   speedup 1.037x
P=64   speedup 1.039x
P=128  speedup 1.064x
P=256  speedup 0.980x
```

Cached and uncached greedy outputs matched exactly, establishing correctness. The timings
showed no material realized KV-cache speedup for this tiny MPS workload. That is a
workload-specific result, not evidence that KV caching is ineffective in general.

### Systems summary

Prefill:

- processes prompt tokens in parallel;
- has attention-score work that grows quadratically with sequence length;
- is the compute-oriented phase of inference.

Decode:

- processes one token at a time;
- reuses K/V tensors from previous tokens;
- has attention work that grows linearly with cache length per token;
- is inherently sequential across generated tokens;
- has KV-cache memory that grows linearly with context length.

Real serving systems must manage both latency and throughput. That trade-off motivates the
next stage on inference batching.

## Stage 6.5 — Static Decode Batching

This benchmark held context length at 128, precision at FP32, and the device at MPS while
varying only the static batch size across 1, 2, 4, 8, and 16. Each cached-decode step used:

```text
input:              (B, 1)
KV cache per layer: (B, H, T, d_head)
T = 128
```

Timing used five repetitions with two warmup forwards. Each reported latency is:

```text
median(
    synchronized block total / 50 identical cached-decode forwards
)
```

Every forward within a measured block received the same baseline cache, so cache length
remained fixed at 128 rather than growing between iterations.

| B  | Decode ms | Aggregate tok/s | Per-sequence tok/s | KV cache MiB |
| -: | --------: | --------------: | -----------------: | -----------: |
|  1 |     0.859 |          1164.3 |             1164.3 |        0.500 |
|  2 |     0.860 |          2324.8 |             1162.4 |        1.000 |
|  4 |     0.884 |          4524.6 |             1131.1 |        2.000 |
|  8 |     0.876 |          9129.0 |             1141.1 |        4.000 |
| 16 |     0.876 |         18260.1 |             1141.3 |        8.000 |

### Latency and throughput metrics

One batched forward produces one new token for every sequence in the batch. Decode-step
latency is the wall-clock duration of that forward. At `B=8`, a step takes approximately
0.876 ms and produces eight tokens.

Aggregate throughput measures total output across all sequences:

```text
aggregate tok/s = B / step_time_seconds
```

At `B=8`, this is approximately 9,129 tok/s. Per-sequence throughput approximates the
generation rate experienced by each sequence:

```text
per-sequence tok/s = 1 / step_time_seconds
```

At `B=8`, this is approximately 1,141 tok/s per sequence.

The theoretical KV-cache storage remains:

```text
KV memory = 2 * L * B * T * d_model * bytes_per_element
```

At fixed `T`, KV-cache memory therefore grows linearly with batch size: from 0.5 MiB at
`B=1` to 1, 2, 4, and 8 MiB at batch sizes 2, 4, 8, and 16 respectively.

### Measurement interpretation

Batch size increased by 16x while measured decode-step latency remained roughly
0.86–0.88 ms. Aggregate throughput consequently increased almost proportionally:

```text
B=1  -> ~1.16k tok/s
B=2  -> ~2.32k tok/s
B=4  -> ~4.52k tok/s
B=8  -> ~9.13k tok/s
B=16 -> ~18.26k tok/s
```

Per-sequence throughput stayed around 1.1k tok/s. For this tiny model and workload, larger
static batches therefore improved aggregate throughput dramatically with almost no
measured per-sequence latency penalty. This is evidence that batch-1 decode underutilizes
the MPS device for this workload; the exact scaling should not be generalized to larger
models or other hardware.

Static batching improves accelerator utilization by processing multiple independent
sequences in one forward, but it increases KV-cache memory. The practical trade-off is:

```text
higher batch size
-> higher aggregate throughput
-> more KV-cache memory
-> potentially higher latency once hardware saturation is reached
```

This experiment did not reach a saturation point by `B=16`. It does not establish a
production-optimal batch size.

### Static versus continuous batching

Static batching collects a fixed group of requests, runs them together, and keeps batch
membership fixed. This benchmark models that simplified situation.

Real serving systems often require dynamic or continuous batching because requests arrive
at different times, prompt lengths vary, sequences finish at different times, and keeping
batch slots occupied can improve throughput. Continuous batching is not implemented or
measured here.

### Progression from Stages 6.3 and 6.4

Stage 6.3 established exact greedy-output equality between cached and uncached decoding,
but KV caching did not materially improve latency for batch-1 tiny-model MPS decode. Stage
6.4 separated prefill from cached decode and measured linear KV-memory growth with context
length. Stage 6.5 then exposed unused device capacity through static batching, delivering
near-linear aggregate-throughput scaling through `B=16` in this experiment.

The systems lesson is workload-specific: an optimization can have little benefit at batch
1 yet become valuable in a serving regime where many requests are processed concurrently.
