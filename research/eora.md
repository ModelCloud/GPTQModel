# EoRA and deployed-output recovery

## Sources

- Liu et al., [EoRA: Training-free Compensation for Compressed LLM with
  Eigenspace Low-Rank Approximation, v1](https://arxiv.org/html/2410.21271v1).
  The [current abstract](https://arxiv.org/abs/2410.21271) uses “Fine-tuning-free”.
- [Official NVIDIA implementation](https://github.com/NVlabs/EoRA).
- [QVQ EoRA source at the audited revision](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/eora/eora.py).

## Source finding

EoRA constructs additive low-rank compensation from a compression residual,
weighted by the eigenspace of calibration inputs. This prioritizes directions
that matter to layer outputs instead of applying unweighted SVD to the residual.
The basic solve requires no gradient training; subsequent fine-tuning is a
separate option. It supplies correction factors, not NVFP4 or KV-cache scales.
[Paper, §§2–3](https://arxiv.org/html/2410.21271v1)

## Repository evidence: standard EoRA

In `gptqmodel/eora/eora.py`, `eora_process_input` accumulates input second
moments. `eora_compute_lora` receives `w_wq_delta` and the calibration matrix.
The eigensolve or Cholesky path weights the residual, performs SVD, and maps the
factors back. The covariance scaling matrix is an offline fitting tool, not a
runtime activation quantizer scale.

In column-vector notation, with `W[N,K]`, `A[r,K]`, `B[N,r]`, the idealized
objective is

$$
\min_{A,B}\;\mathbb E_x\left\|(W-\widehat W)x-BAx\right\|_2^2.
$$

This describes the objective, not a guarantee that finite-precision or randomized
SVD attains the global optimum in every run. Numerical-rank truncation and
exported factor dtype affect the actual result.

## Existing QVQ output-residual research

QVQ already has a broader reference fitter:
[`native_recovery.py`](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/scripts/p32_twenty/native_recovery.py).

- `capture_residual` executes teacher and native callbacks on their input
  dtype/device, then subtracts their outputs on CPU in FP64.
- `fit_output_residual` solves reduced-rank output reconstruction in the
  retained activation singular space. It does not merely truncate
  `pinv(X) @ residual`.
- That file uses row-major `X[M,K]`, `A[K,r]`, `B[r,N]` and `(X @ A) @ B`;
  do not transpose its factors according to the column-vector notation above.

The [native recovery report](../docs/experiments/p32-twenty/NATIVE_RECOVERY.md)
describes actual W4A16/BF16 runtime residual fitting, including kernel rounding,
with separate calibration/evaluation captures. It also records held-out failures.
The [fused recovery report](../docs/experiments/p32-twenty/FUSED_RECOVERY_MODEL.md)
documents a bounded opt-in path and its limits. These reports do not establish
production NVFP4 W4A4 recovery or a universal rank-8 benefit.

## Proposed W4A4 use

For an idealized main GEMM using `x4 = Q_NVFP4(x)`:

$$
Wx-\widehat W x_4
=(W-\widehat W)x+\widehat W(x-x_4).
$$

A weight-only residual does not explicitly target the activation-error term.
Replacing the covariance with A4 covariance alone does not fix that target mismatch.

Instead capture `R = Y_teacher - Y_deployed` and fit

$$
\min_{A,B}\left\|R-BAz\right\|_F^2,
$$

using column batches here. `z` must be the correction branch's actual input,
including its transform and precision. It may be a higher-precision activation
before A4 rounding or the decoded A4 operand, depending on the runtime contract.
Keeping the same transformed coordinate system does not imply keeping the same
precision; record both.

Reuse the reference fitter, then validate the exported runtime factors and
addition order. A CPU solve does not certify the GPU path. Calibrate
[NVFP4 scales](nvfp4-hybrid-ptq.md) first, bind the factors to the deployed payload
and scale policy, and compare correction off/on after reload.

A static low-rank map can recover only the component representable from `z`;
it cannot reconstruct arbitrary information destroyed by input-dependent rounding.
Recovery quality, rank, storage and launch cost therefore require held-out
measurements. Changes to scales, weights, transforms or precision invalidate an
assumption that previously fitted factors remain optimal.
