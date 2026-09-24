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

### Initial operator-profiler finding

The initial profile reported:

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

This did not mean scalar conversion itself required approximately 9 ms of computation per
step. Instead, these device-to-host reads acted as synchronization boundaries, so their CPU
attribution included time spent waiting for earlier queued MPS work to complete.

### Fused AdamW

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
