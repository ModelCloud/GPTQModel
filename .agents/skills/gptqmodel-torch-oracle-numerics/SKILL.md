---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-torch-oracle-numerics
description: Validate faster quantization math and inference kernels against an independent Torch oracle before accepting performance claims or regressions.
---

# Torch oracle for numerical changes

Apply this skill when adding or optimizing a quantization computation or an
inference kernel, regardless of backend. Build deterministic A/B unit tests
that feed the same inputs and configuration to the changed path and a separate
Torch reference. The oracle must perform its own arithmetic; do not call the
changed kernel or reuse its intermediate outputs as expected results.

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

Use representative sizes, supported dtypes, and at least one edge case that
could change the numerical result. Synchronize asynchronous backends before
reading results or timing them. Run the tests on the target hardware. Report
the tested shapes, dtypes, tolerances, observed errors, and benchmark method in
the pull request.

If an optimized path exceeds a limit, fix the arithmetic and rerun the A/B
tests before treating its speedup as validated. Record any unresolved mismatch
as a merge blocker; do not loosen the limit merely to make the test pass.
