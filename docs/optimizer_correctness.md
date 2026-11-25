# Optimizer arithmetic correctness policy (v2)

This policy applies only to the standalone synthetic training-optimization benchmark.
It does not change production training, optimizer implementations, or the original
`TOLERANCES` dictionary. Existing result and diagnostic artifacts remain historical evidence.

## Separate arithmetic from trajectory drift

For each of three consecutive baseline updates in each clipping case, restore every
variant from A's identical pre-step weights and Adam state. Retain the original strict
comparisons for forward loss, gradients, pre-clipping norm, clipped gradients and moments.
A/B also retain the original parameter and update-delta comparisons.

Check each backend against an independent CPU FP64 AdamW step from its exact FP32
inputs. Additionally replay each optimizer from **exactly the same** A parameters,
clipped gradients and moments: this isolates the optimizer from forward-pass variation.
The oracle checks every parameter tensor, not just embeddings. Step counters, shapes,
dtypes, finiteness and nonnegative second moments are mandatory.

Independent trajectories then run for up to ten updates. Report maximum/RMS parameter
differences and loss divergence at every step. These distances are descriptive for fused
all independent trajectories, including A/B, not substitutes for common-input arithmetic
checks. The original A/B trajectory thresholds are retained as a separate reproducibility
result, not applied as one-step arithmetic gates to already-diverged states. At the first
A/B reproducibility failure in each clipping case, replay both optimizers from A's exact
pre-step parameters, clipped gradients and moments. That replay must pass the original
strict parameter/update/moment tolerances and the unchanged FP64 arithmetic policy.
Common-start A/B forward/backward checks remain strict. All trajectories must remain
finite and preserve tensor shapes and dtypes. A passing arithmetic result does not imply
independent trajectory reproducibility.

## Error budget

Use FP32 unit roundoff `u = 2^-24`, subnormal quantum `eta = 2^-149`, and
`gamma(n) = n*u/(1-n*u)`. The CPU FP64 oracle computes decoupled weight decay,
first/second moments, bias correction, epsilon placement and the ideal parameter update.

Moment error envelopes are:

```
E_m = gamma(8) * (abs(m_old) + abs(g)) + 8*eta
E_v = gamma(8) * (abs(v_old) + g*g) + 8*eta
```

Eight operations conservatively cover scalar representation and alternate product/add
or lerp evaluation orders, including cancellation. For `c = 1 - beta^t`, use
`E_c = gamma(2*t+4) * (1 + abs(beta^t))`, accounting for scalar conversion,
power evaluation and subtraction. Propagate these intervals through bias correction,
square root and epsilon addition; expand denominator arithmetic by `gamma(4)` and
learning-rate scaling by `gamma(2)`. Nonpositive/unresolved denominator intervals fail closed.

Let `a` be the ideal adaptive step and `d = lr*weight_decay*p`. The parameter-write
envelope is `gamma(4)*(abs(p)+abs(d)+abs(a)) + 4*eta`, covering the decay coefficient
product/subtraction, parameter scaling and final subtraction. Add the propagated adaptive
step error. Compare each backend's endpoint directly with the FP64 oracle under this
elementwise budget; verify moments under their own envelopes as well.

This uses both parameter and update scales and denominator sensitivity. It does not use
a global absolute tolerance fitted to a failed run, and does not depend on maximum ULP
counts near zero. Subtracting already rounded endpoints in FP64 cannot undo optimizer
rounding: the common-start observed-delta error equals its parameter endpoint error
apart from FP64 arithmetic.

## Scope and limitations

The operation counts are conservative engineering envelopes under standard round-to-nearest
FP32 arithmetic, not certified bounds for every Metal implementation of power/square root.
Reject MPS fast-math. Overflow, negative second moments, unresolved intervals and observed
budget violations require investigation, not automatic tolerance expansion. Subnormal
flush-to-zero behavior outside the stated model may cause a legitimate backend to fail.
Near a cancellation or very small denominator, the propagated envelope can be wider;
per-tensor maximum budget/error ratios are recorded for review.

A passing arithmetic gate does not certify long-horizon training equivalence, model quality,
or the historical performance claim. Run the MPS correctness-only mode before collecting
performance measurements.
