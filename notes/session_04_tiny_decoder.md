# Session 04: Tiny Decoder

This session implements the forward pass only. Training, loss computation, and
generation intentionally begin in the next session.

## Tensor dimensions

```text
B       = batch size
n       = sequence length
V       = vocabulary size
d_model = Transformer hidden/token representation width
H       = number of attention heads
d_head  = d_model / H
d_ff    = expanded MLP hidden width
```

The default inspection configuration is `B=2`, `n=5`, `V=100`, `d_model=16`,
`H=4`, `d_head=4`, `d_ff=64`, and `n_layers=3`.

## Multi-head attention

```text
(B, n, d_model)
-> (B, H, n, d_head)
-> scores (B, H, n, n)
-> attention output (B, H, n, d_head)
-> concat (B, n, d_model)
```

Each score row is masked above its diagonal, so a position can only attend to
itself and earlier positions.

## RoPE

RoPE can be viewed as:

```math
q_p' = R_p q_p
k_p' = R_p k_p
```

`R_p` is a deterministic rotation based on position. The implementation rotates
adjacent feature pairs independently in 2D instead of constructing a full
matrix. The useful identity is:

```math
(R_i q_i)^T(R_j k_j) = q_i^T R_{j-i} k_j
```

Thus attention is sensitive to relative position.

## Decoder block

```math
x_1 = x + Attention(Norm(x))
```

```math
x_2 = x_1 + MLP(Norm(x_1))
```

Attention communicates between tokens; the MLP transforms features within each
token. Norm operates over the last feature dimension and residual connections
combine an old representation with a learned update.

## Current end-to-end path

```text
token IDs
-> embedding vectors
-> repeated pre-norm decoder blocks
-> final norm
-> vocabulary logits
```

The next session will begin with autoregressive next-token loss, a
forward/backward pass, gradient inspection, causal-mask verification, and a
minimal generation loop.
