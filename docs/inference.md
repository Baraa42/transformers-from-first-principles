# Transformer Inference from First Principles

This chapter explains the decoder-only Transformer inference path implemented in this
project: naive autoregressive generation, per-layer KV caching, RoPE position handling,
prefill versus decode, and static batching. It also summarizes what the measurements show
for the project's small four-layer model on Apple MPS. Detailed benchmark methodology and
the broader training-performance work are in [Performance Notes](performance.md).

## Naive autoregressive decoding

Autoregressive generation begins with a prompt and predicts one token at a time. Without a
cache, every step sends the complete generated prefix through the model:

```python
logits = model(generated_prefix)
next_token = argmax(logits[:, -1, :])
```

For a prompt of length `P`, successive model inputs have lengths:

```text
P, P + 1, P + 2, ...
```

This path is simple and useful as a correctness reference, but it repeats work. Each new
full-prefix forward recomputes old tokens' Q/K/V projections, attention interactions, and
hidden-state transformations. Under causal masking, a token cannot attend to future
positions. Appending a token therefore cannot change representations already computed for
earlier positions. In evaluation mode, recomputing those positions produces the same old
states rather than new information.

## Why KV caching works

To compute the output for one new token, each attention layer needs its new query, key, and
value:

```text
Q_new, K_new, V_new
```

The new query must also attend to the keys and values produced by all previous tokens:

```text
K_old, V_old
```

Those old K/V tensors do not change, so the model can retain and reuse them. Old queries
are unnecessary because cached decoding computes an output only for the new position; it
does not recompute outputs for previous positions. Attention-score matrices are also not
cached. They are query-dependent intermediate results and can be much larger than the K/V
state needed for subsequent steps.

Each layer stores a pair:

```text
(K, V)

K shape = (B, H, T, d_head)
V shape = (B, H, T, d_head)
```

The full model cache is a tuple containing one pair per decoder layer:

```text
(
    (K_layer_0, V_layer_0),
    ...,
    (K_layer_L-1, V_layer_L-1),
)
```

Every layer needs its own cache because it operates on different hidden states and has
distinct K/V projection matrices.

## Cached-decode tensor shapes

Suppose the cache already contains `T` positions. For a single new token, one attention
layer receives and produces:

```text
input:              (B, 1, d_model)

Q_new:              (B, H, 1, d_head)
K_new, V_new:       (B, H, 1, d_head)

K_cache, V_cache:   (B, H, T, d_head)
updated K/V:         (B, H, T + 1, d_head)

attention scores:   (B, H, 1, T + 1)
```

The new K/V tensors are concatenated with the cached tensors along the sequence dimension.
The new query then attends across the combined keys and values. By contrast, an uncached
length-`T` full-prefix forward constructs an attention-score tensor shaped:

```text
(B, H, T, T)
```

The cache changes the attention problem for a decode step from every query against every
key to one new query against the retained prefix.

## Causal masking during cached decode

Prompt prefill still processes many positions simultaneously, so it needs the normal
causal mask: query position `i` must not attend to key positions greater than `i`.

One-token cached decode is simpler. A new query at absolute position `T` receives keys only
for positions `0 ... T`. Every available key is a valid past or current position, and no
future key exists, so the usual `T x T` causal mask is unnecessary for that step.

This simplification is specific to one-token decode. If several new positions were decoded
as a chunk, masking would still be required among those new positions so that an earlier
position in the chunk could not attend to a later one.

## RoPE positions and cache offsets

Rotary position embeddings encode absolute positions into queries and keys before
attention. During prompt prefill, a prompt of length `P` uses positions:

```text
0 ... P - 1
```

During cached decode, the input tensor may contain only one token, but that token is not at
position zero. If the existing cache length is `P`, the token must be rotated at absolute
position `P`:

```python
apply_rope(q_new, offset=P)
apply_rope(k_new, offset=P)
```

Cached keys are stored after RoPE has been applied. Old cached keys must not be rotated
again. The central correctness invariant is:

```text
RoPE(single token, offset=P)
==
that token's RoPE result at position P in a full sequence
```

Tests enforce this equivalence for both individual tokens and multi-token chunks.

## Explicit cache state in the model API

The ordinary model path remains unchanged:

```python
logits = model(ids)
```

Prefill requests the initial cache explicitly:

```python
logits, cache = model(
    prompt_ids,
    use_cache=True,
)
```

Each decode step passes the current cache and receives the updated one:

```python
logits, cache = model(
    next_token,
    use_cache=True,
    kv_cache=cache,
)
```

The cache is runtime request state rather than mutable `self.cache` state inside the model.
This keeps the model itself stateless, prevents stale data from leaking between requests,
makes cache behavior straightforward to test, supports multiple concurrent request states,
and provides a natural basis for batching and serving systems.

## Prefill versus decode

Prefill and decode exercise the same model in different computational regimes:

| Phase   | Input             | Attention scores | Attention work | Execution pattern |
| ------- | ----------------- | ---------------- | -------------- | ----------------- |
| Prefill | Full prompt       | `T x T`          | approximately `O(T^2)` | Prompt tokens processed in parallel |
| Decode  | One token + cache | `1 x T`          | approximately `O(T)` per token | Sequential across generated tokens |

These complexity statements describe attention primarily. Projections, MLPs,
normalization, and other model operations have different scaling. Prefill exposes
parallelism across the prompt, while decode is inherently sequential across generated
tokens because token `n + 1` depends on token `n`.

## KV-cache memory

For standard multi-head attention, `H * d_head = d_model`. Across `L` layers, batch size
`B`, and cache length `T`, the cache contains:

```text
KV elements = 2 * L * B * T * d_model
KV bytes    = 2 * L * B * T * d_model * bytes_per_element
```

For this project's `L=4`, `d_model=128`, FP32 model at `B=1`:

```text
T=32  -> 0.125 MiB
T=64  -> 0.250 MiB
T=128 -> 0.500 MiB
T=256 -> 1.000 MiB
T=512 -> 2.000 MiB
```

KV memory is linear in the number of layers, batch size, context length, and model width.
Caching trades this persistent memory for less repeated decode computation.

## Correctness and batch-1 performance

Cached decoding was validated against the uncached reference by requiring exact equality
of every greedily generated token ID. Once correctness was established, the measured
decode speedups were:

| Prompt | Cached speedup |
| -----: | -------------: |
|     32 |         1.037x |
|     64 |         1.039x |
|    128 |         1.064x |
|    256 |         0.980x |

For this tiny batch-1 MPS workload, KV caching did not materially improve wall-clock
latency. This does not mean KV caching is ineffective generally. Here, the reduced
sequence-dependent compute remained small relative to fixed and backend overhead for a
very small model and workload. The benchmark does not isolate a more specific bottleneck.

## Prefill and cached-decode measurements

To reduce distortion from per-operation synchronization, each latency sample was measured
as:

```text
median(
    synchronized block total / 50 identical forwards
)
```

At batch size 1 in FP32 on MPS:

| T   | Prefill ms | Cached decode ms | KV cache MiB |
| --: | ---------: | ---------------: | -----------: |
|  32 |      1.053 |            1.012 |        0.125 |
|  64 |      1.009 |            0.939 |        0.250 |
| 128 |      0.989 |            0.964 |        0.500 |
| 256 |      1.253 |            1.402 |        1.000 |
| 512 |      1.308 |            1.095 |        2.000 |

The latency curves are noisy and should not be used to infer clean asymptotic behavior for
this tiny model. Fixed overhead, utilization, and non-attention work remain substantial.
The clearest measured scaling result is the linear growth of KV-cache memory with context
length.

## Static decode batching

The static batching experiment held context length at 128 and varied batch size. Every
timed forward used the same fixed-length baseline cache and produced one token per active
sequence:

| B  | Decode ms | Aggregate tok/s | Per-sequence tok/s | KV cache MiB |
| -: | --------: | --------------: | -----------------: | -----------: |
|  1 |     0.859 |          1164.3 |             1164.3 |        0.500 |
|  2 |     0.860 |          2324.8 |             1162.4 |        1.000 |
|  4 |     0.884 |          4524.6 |             1131.1 |        2.000 |
|  8 |     0.876 |          9129.0 |             1141.1 |        4.000 |
| 16 |     0.876 |         18260.1 |             1141.3 |        8.000 |

The two throughput metrics are:

```text
aggregate tok/s    = B / step_time_seconds
per-sequence tok/s = 1 / step_time_seconds
```

Batch size increased 16x while decode-step latency remained approximately 0.86–0.88 ms.
Aggregate throughput scaled nearly linearly, per-sequence throughput stayed around 1.1k
tok/s, and KV memory grew linearly. For this workload, that is strong evidence that
batch-1 decode underutilized the MPS device. The exact scaling is specific to this model,
backend, and benchmark setup.

## Static versus continuous batching

Static batching processes a fixed group of requests together and keeps membership
unchanged. It is a useful controlled experiment, but real traffic is dynamic: requests
arrive at different times, prompts have different lengths, and generated sequences finish
at different times.

Continuous or dynamic batching conceptually inserts and removes active sequences while
keeping useful device work available. It must coordinate variable sequence lengths and
per-request cache state. This repository does not implement continuous batching or a
serving scheduler.

## What a production serving system adds

A production system must extend the basic inference loop with concerns such as:

- continuous batching and request scheduling;
- variable-length sequences, padding, or ragged/paged representations;
- KV-cache allocation, eviction, and memory-fragmentation management;
- admission control and latency service-level objectives;
- throughput optimization under changing traffic;
- lower-precision caches or architectural variants such as MQA/GQA to reduce memory.

These are conceptual next steps, not features present in this repository.

## Interview-ready takeaways

- KV caching removes redundant K/V projection and attention-input computation for previous
  tokens.
- Cache per-layer K/V tensors, not old queries or attention-score matrices.
- Prefill processes prompt tokens in parallel; autoregressive decode is sequential across
  generated tokens.
- Cached decode uses a `1 x T` attention-score row instead of recomputing a `T x T` matrix.
- RoPE positions must continue from the cache length, and cached keys remain already
  rotated.
- KV-cache memory grows linearly with layers, batch size, context length, and model width.
- Batch-1 algorithmic savings may not translate directly into wall-clock latency when
  fixed overhead dominates.
- Batching can dramatically improve aggregate throughput when the device is underutilized,
  at the cost of additional KV-cache memory.

For exact commands, measurement controls, and the full performance record, see
[Performance Notes](performance.md).
