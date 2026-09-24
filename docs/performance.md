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
