---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-torch-oracle-numerics
description: Validate changed quantization math, packed formats, and inference kernels against a Torch oracle; adjudicate near-tie mismatches before rejecting optimized arithmetic.
---

# Torch oracle for numerical changes

Apply this skill when adding or changing quantization math, a weight packer,
or an inference kernel, regardless of backend or performance intent. Build
deterministic A/B unit tests that feed the same inputs and configuration to the
changed path and a separate Torch reference. The oracle must perform its own
arithmetic; do not call the changed kernel or reuse its intermediate outputs as
expected results.

## Acceptance limits

- **Quantization:** Packed codes, indices, and other discrete outputs must match
  exactly except for adjudicated floating-point rounding ties described below.
  Floating quantization outputs must satisfy both relative and absolute
  tolerances of `1e-6` or tighter away from those ties. For scale-dependent matrix intermediates such
  as Hessians, apply the `1e-6` limit to normalized matrix error
  `||actual - oracle|| / ||oracle||`, and check for nonfinite values. Use a
  float64 Torch oracle for accumulated matrices when float32 accumulation
  error would consume the allowed tolerance.
- **Inference:** Floating outputs may use relative and absolute tolerances up to
  `2e-3`. Discrete outputs, including selected token IDs, must match exactly
  when the test fixes sampling and tie behavior.

## Quantization boundary cases

For every changed packer or quantizer, test values exactly at and one
representable input step on each side of relevant rounding thresholds. Also
cover clamp/saturation endpoints, positive and negative values, zero (including
signed zero when its packed bits matter), and scale or zero-point selection ties
when the format has them. Exercise casts used by supported input dtypes. Random
inputs are supplemental; they cannot replace boundary cases.

Compare complete packed bytes or codes with the independent Torch oracle.
Reproduce any *specified* format precision and tie rule exactly: a packer with
defined bytes cannot claim a rounding exception. For iterative quantizers that
do not specify bitwise equivalence to Torch, investigate each code mismatch
before rejecting a path or replacing GPU arithmetic with a slower Torch-matching
operation. Trace the pre-rounding value, scale, zero point, and preceding
updates; measure its distance in code units from the half-step. Adjudicate
suspected ties with independent higher-precision arithmetic on the same source
values and Hessian, including their actual stored precision. A one-ULP change
may flip a code at a true half-step without indicating an inaccurate kernel.
Check that mismatches away from proven ties still meet the exact-code and
`1e-6` floating limits. Keep explicit threshold-neighbor and endpoint tests;
do not turn a broad numerical tolerance into a blanket code exception.
Report total mismatches, adjudicated tie mismatches, maximum output drift,
maximum floating drift away from ties, and the reference precision in the PR.

Use representative sizes and supported dtypes. Synchronize asynchronous
backends before reading results or timing them. Run the tests on the target
hardware. Report the tested shapes, dtypes, tolerances, observed errors, and
benchmark method in the pull request.

For MLX weight quantizers, packers, and inference kernels, include every
projection in `tests/qwen38_27b_shapes.py`: full-attention Q/K/V/O, fused
linear-attention QKV/O, and MLP gate/up/down. Use the checkpoint's bfloat16
dtype where the kernel supports it. Compare quantization against the independent
Torch oracle over the complete output, including every packed byte or an
individually adjudicated quantizer tie. Compare
inference outputs using the limits above. Benchmark every applicable projection
shape against the corresponding main-branch path with inputs already resident;
report per-shape timings and the aggregation method. Keep small boundary tests
in addition to this model-scale matrix. If a projection or dtype cannot run,
name the specific limitation and cover the closest supported configuration.

If an optimized path exceeds a limit, investigate the numerical cause before
changing it. Fix genuine arithmetic or format errors and rerun the A/B tests.
Record unresolved mismatches as merge blockers; retain a faster accurate path
when only proven natural rounding ties differ, with the evidence above.
