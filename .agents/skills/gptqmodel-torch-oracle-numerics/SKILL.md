---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-torch-oracle-numerics
description: Validate changed quantization math, packed formats, and inference kernels against an independent Torch oracle, including rounding boundaries.
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
  exactly. Floating quantization outputs must satisfy both relative and absolute
  tolerances of `1e-6` or tighter. For scale-dependent matrix intermediates such
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

Compare complete packed bytes or codes exactly with the independent Torch
oracle. Reproduce the format's specified intermediate precision in that oracle
(for example, float64 arithmetic applied to float32 operands) so reference
rounding error does not hide a boundary mismatch. Report the exact comparison
and the boundary cases in the pull request.

Use representative sizes and supported dtypes. Synchronize asynchronous
backends before reading results or timing them. Run the tests on the target
hardware. Report the tested shapes, dtypes, tolerances, observed errors, and
benchmark method in the pull request.

For MLX weight quantizers, packers, and inference kernels, include every
projection in `tests/qwen38_27b_shapes.py`: full-attention Q/K/V/O, fused
linear-attention QKV/O, and MLP gate/up/down. Use the checkpoint's bfloat16
dtype where the kernel supports it. Compare quantization against the independent
Torch oracle over the complete output, including every packed byte. Compare
inference outputs using the limits above. Benchmark every applicable projection
shape against the corresponding main-branch path with inputs already resident;
report per-shape timings and the aggregation method. Keep small boundary tests
in addition to this model-scale matrix. If a projection or dtype cannot run,
name the specific limitation and cover the closest supported configuration.

If an optimized path exceeds a limit, fix the arithmetic and rerun the A/B
tests before treating its speedup as validated. Record any unresolved mismatch
as a merge blocker; do not loosen the limit merely to make the test pass.
