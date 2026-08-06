# Adaptive damping mathematics

## Accuracy contract

GPTQ minimizes calibration-output error by using activation statistics to propagate each column's quantization
residual into the unquantized columns. Adaptive damping must therefore adapt the activation Hessian regularization;
it must not be a weight-only rule, and its safe default must not replace GPTQ's error correction with a heuristic.

For a matrix layer with calibration rows `X` and `N` observed rows, GPTQModel accumulates

```text
H = (2 / N) X^T X.
```

`GPTQ.add_batch()` accumulates each batch's `X^T X` in FP32, and `materialize_global_hessian()` applies `2 / N`.
Consequently, different calibration activations can select different damping for identical weights.

## Adaptive regularization

Let `m = mean(diag(H))` and let `lambda_max(H)` be the configured spectral estimate. The safe default computes

```text
r = lambda_max(H) / m
p(H) = clamp(p0 * r^alpha, p_min, p_max)
lambda = p(H) * m
H_lambda = H + lambda I.
```

Here `p0` is `base_percdamp` and `alpha` is `spectral_alpha`. The dimensionless ratio `r` makes `p(H)` invariant
to a uniform rescaling of the calibration activations. Isotropic activation covariance has `r = 1`; correlated or
low-rank calibration data has `r > 1` and receives stronger regularization, subject to the configured bounds.

Invalid, non-finite, or non-positive statistics fall back to `base_percdamp`. Singular and dead-feature Hessians are
handled by the existing positive-definiteness floor and Cholesky retry path before inversion.

## Canonical GPTQ error correction

After damping, GPTQModel constructs the upper Cholesky factor of the inverse:

```text
U = chol(H_lambda^-1, upper=True).
```

For column `j`, the quantization residual and remaining-weight update are exactly

```text
E_j = (W_j - Q_j) / U_jj
W[:, j:] = W[:, j:] - E_j outer U[j, j:].
```

The corresponding block loss contribution is

```text
L = 0.5 * ||E||_F^2.
```

This is the GPTQ error-correction path. Adaptive damping changes `H_lambda` and therefore `U`, so it is directly
conditioned on calibration activations while retaining the canonical correction coefficient.

## Experimental controls

The following controls are disabled by default:

| Control | Why it is not part of the safe default |
|---|---|
| `module_prior_enabled` | Module names are metadata, not calibration or GPTQ error evidence. |
| `group_size_prior_enabled` | Group size is a quantizer layout choice, not an observed error statistic. |
| `online_feedback_enabled` | It applies under/over-relaxation to `E_j`, changing canonical GPTQ propagation. |
| `group_error_use_hessian_weighting` | `E_j` already includes inverse-Hessian geometry; raw `diag(H)` weighting can double count activation importance. |

When explicitly enabled, online feedback measures a completed group's raw canonical correction loss
`L_g = 0.5 * ||E_g||_F^2`, maintains an EMA target `T_g`, and applies a bounded scale to the next group:

```text
T_g = beta T_(g-1) + (1 - beta) L_g
a_g = clamp((T_g / L_g)^gamma, a_min, a_max)
E_next = a_g E_next.
```

If a group-size prior is also explicitly enabled, its factor divides `a_g`. These modes are retained for controlled
experiments, but their results must be compared against the safe calibration-only default.

## Validation matrix

| Region | Ideal/normal cases | Boundary/corner cases | Required invariant |
|---|---|---|---|
| Hessian accumulation | Orthonormal and random calibration rows | Multiple batches, zero/dead features | Exact `(2 / N) X^T X` |
| Spectral adaptation | Isotropic and correlated covariance | Rank deficient, ill conditioned, tiny/large scale | Finite bounded damping; scale invariance |
| GPTQ correction | Dense FP32 correction oracle | Zero residual, group tails, 2/3/4/8-bit groups | Exact `E_j` and outer-product update |
| Optional feedback | Low/equal/high group loss | Zero, infinity, extreme ratios | Raw loss recovery and configured clamps |
| Ordering/integration | Natural order | `desc_act`, group-aware ordering, static groups | Finite outputs and restored column order |

The focused tests live in `tests/test_adaptive_damping.py`. GPU parametrizations exercise the same equations when a
GPU is available; the FP32 CPU tests provide deterministic dense-reference coverage independent of GPU availability.
