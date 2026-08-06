# Calibration-aware adaptive clipping

Adaptive clipping is an opt-in GPTQ range search. It selects one clipping factor per output row and quantization
group before the group is quantized. The default objective is `gptq_error`, which evaluates candidate ranges with
the same calibration-dependent, sequential error correction used by GPTQ.

## GPTQ objective

For calibration activations `X` with `N` samples, GPTQ forms

```text
H = (2 / N) X^T X
H_lambda = H + lambda I
U = chol(H_lambda^-1, upper=True)
```

`lambda` is the damping selected by the normal GPTQ damping path. For a clipping candidate `c`, the quantizer first
derives its fixed per-row scale and zero point from the range clipped by `c`. It then simulates each column in the
group in GPTQ order:

```text
Q_j^(c) = quantize(W_j^(c); scale_c, zero_c)
E_j^(c) = (W_j^(c) - Q_j^(c)) / U_jj
W_[j:]^(c) = W_[j:]^(c) - E_j^(c) outer U_[j, j:]
L_c = 0.5 * sum_j ||E_j^(c)||_2^2
```

The winning candidate is the one with the smallest `L_c` independently for each output row. The simulation starts
from the current group weights, so corrections from preceding GPTQ groups are already included. Exact adaptive
clipping bounds the outer GPTQ block to at most one quantization group; the block-level update therefore reaches the
next group before its candidates are evaluated. Its selected range is then consumed by the normal eager, CPU-block,
or Triton GPTQ quantization path.

This is not a weight-only score. Later quantized columns depend on earlier errors through `U`, and `U` comes from
the collected calibration activations. If `Delta = W_original - Q`, the simulated recursion gives

```text
Delta = E U
0.5 * Delta H_lambda Delta^T = 0.5 * ||E||_2^2
```

For the undamped calibration Hessian, the quadratic is also the mean squared output perturbation:

```text
0.5 * Delta H Delta^T = (1 / N) ||X Delta^T||_2^2
```

## Objective modes and failure behavior

```text
+----------------+-------------------------+----------------------------+------------------------------------+
| metric         | calibration dependency  | correction propagation     | missing/invalid geometry           |
+----------------+-------------------------+----------------------------+------------------------------------+
| gptq_error     | full damped inverse     | exact GPTQ group recursion | use full unclipped range           |
| hessian_diag   | raw Hessian diagonal    | diagonal approximation     | use full unclipped range           |
| mse            | none                    | none                       | continue weight-only search        |
+----------------+-------------------------+----------------------------+------------------------------------+
```

There is deliberately no implicit fallback from `gptq_error` or `hessian_diag` to `mse`. Such a fallback would make
quantization quality depend on whether calibration geometry happened to be available while presenting the same
configuration to the user. If exact geometry is unavailable, candidate `1.0` is selected even when it was omitted
from the configured candidate list.

`gptq_error` is supported by dynamic grouped dense GPTQ, including group-bounded batched search, tail groups, CPU
block fallback, and eager execution. Static groups are constructed before inverse factorization, so exact clipping
safely uses the unclipped range there. `hessian_diag` and `mse` remain explicit compatibility/experimental objectives.

## Configuration and usability

Adaptive clipping remains opt-in, preserving the PR 215 behavior:

```python
QuantizeConfig(
    bits=4,
    group_size=128,
    adaptive_clipping={
        "enabled": True,
        "metric": "gptq_error",
        "candidates": [0.99, 0.995, 0.999, 1.0],
    },
)
```

Omitting `adaptive_clipping` disables it. Users who explicitly want the old weight-only search can request
`metric="mse"`; users who want the cheaper activation-aware approximation can request `metric="hessian_diag"`.
Candidate evaluation is chunked and uses views of the already-live inverse factor, so it does not retain per-group
factor copies. The exact objective needs one candidate-by-row-by-group working buffer and small per-column scratch.

## Validation contract

Tests compare the implementation against an independent scalar GPTQ recursion and verify the dense identities above.
Coverage includes 2/3/4/8-bit quantization, symmetric and asymmetric ranges, per-row and batched groups, uniform
Hessian scaling, calibration-dependent candidate changes, correlated/rank-deficient/ill-conditioned/dead-feature
geometry, zeros, constants, tiny values, Gaussian/asymmetric/outlier distributions, and NaN/Inf sanitization.
