# Architecture notes

The implemented model is a decoder-only, pre-norm Transformer. This page describes the
current configuration; [Session 04](session_04_tiny_decoder.md) preserves the earlier
forward-pass derivation and its smaller inspection defaults.

| Tensor / operation | Shape or behavior |
| --- | --- |
| Input token IDs | `(B, T)` |
| Embedding / block hidden state | `(B, T, D)` |
| Q, K, V after head reshape | `(B, H, T, D/H)` |
| Attention scores | `(B, H, T, T)` for prefill; `(B, H, 1, cached_length+1)` for decode |
| MLP | `D → F → D`, SiLU, with biases |
| Vocabulary logits | `(B, T, V)`, no output softmax |

Each block computes `x + Attention(LayerNorm(x))`, then
`x + MLP(LayerNorm(x))`. Queries and keys use adjacent-pair RoPE rotations;
values remain unrotated. Attention projections are bias-free. A final LayerNorm
precedes the bias-free LM head; token embeddings and head weights are untied.

The [training configuration](../configs/tiny.yaml) has `D=128`, `F=512`, `H=4`,
`L=4`, and the configured tokenizer supplies `V=8192`. The total parameter count is
`2*V*D + L*(4*D*D + 2*D*F + F + 5*D) + 2*D = 2,888,448`.
Constructor defaults are smaller inspection settings, not the benchmark model.

See [attention](../src/transformers_from_scratch/attention.py),
[blocks](../src/transformers_from_scratch/layers.py), and
[model](../src/transformers_from_scratch/model.py) for the implementation;
[numerical reference tests](../tests/test_numerical_references.py) for independent checks;
and [inference notes](../docs/inference.md) for masking, absolute RoPE offsets and
`2*L*B*T*D*bytes_per_element` logical KV-cache memory. That byte formula is not a
measurement of peak memory or physical traffic.
