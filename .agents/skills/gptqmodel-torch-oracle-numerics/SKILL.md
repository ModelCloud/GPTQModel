---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-torch-oracle-numerics
description: Validate quantization math, packed formats, and inference kernels against a Torch oracle; enforce quantized persistent weight storage for inference implementations, dispatch, and cache changes; adjudicate near-tie mismatches.
---

# Torch oracle for numerical changes

Apply this skill when adding or changing quantization math, a weight packer,
an inference kernel, inference dispatch, or runtime weight storage, regardless
of backend or performance intent. Also apply it when auditing existing kernels. Build
deterministic A/B unit tests that feed the same inputs and configuration to the
changed path and a separate Torch reference. The oracle must perform its own
arithmetic; do not call the changed kernel or reuse its intermediate outputs as
expected results.

## Persistent inference weight storage

The repository-wide storage rule in `AGENTS.md` is an acceptance gate for all
quantization methods and all inference backends/devices. After an operation
finishes consuming temporary weights, no inference-owned reference may retain
a dense, dequantized, or expanded copy for reuse. Apply the gate to optimized
and fallback paths, eager execution, JIT/compiled execution, warmup, and
optional configurations. Disabling a prohibited cache by default is not a fix;
environment flags and public opt-in APIs are not exceptions.

Judge storage by representation and lifetime, not names or dtypes. Native
quantized weights, packed alternate layouts, scales, zero points, format
codebooks, and indexing metadata may persist. Full-layer unpacked INT4 codes
stored as INT8, or reconstructed FP16/BF16/FP32 weights, may not. This rule is
about expanded weight copies, not ordinary activation/KV caches or weights
already stored in their native quantized format.

Keep dequantized scratch local to its consuming computation, preferably in
bounded tiles. Release references once all consuming work completes, including
asynchronous work. Do not retain it in module attributes, buffers, containers,
closures, global/thread-local caches, compiled graph constants, or backend
objects. `persistent=False`, detaching a tensor, offloading it to CPU, and
changing its precision do not change its lifetime. Allocator-reserved free
memory is not a live weight cache; do not add per-forward allocator flushes.
Explicit export/dequantization APIs and independent test oracles may return
dense tensors to callers without installing them as inference caches.

For changed inference/storage paths:

- Inspect persistent state after the first and repeated forwards, including
  each supported fallback and opt-in path. Add targeted regression coverage
  that demonstrates no dense/expanded weight is retained; do not only check a
  cache's name or serialization behavior.
- Verify allowed packed caches invalidate on weight/scale mutation or
  replacement, checkpoint loading, and device/dtype moves. Exercise the shared
  explicit-clear contract for inference tensors without version counters.
- Report retained weight/cache bytes and temporary dequantization storage
  alongside accuracy and speed. Distinguish unchanged checkpoint storage,
  additional packed caches, and transient scratch. A historical dense-cache
  benchmark may be shown only as a clearly labeled comparison, never as an
  eligible production optimization.
- Remove violations encountered in the implementation under review. Record
  any remaining audited violations with file references; do not claim that
  documentation alone establishes compliance for every existing kernel.

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

For every MLX inference kernel, test FP16 and BF16 activations separately.
Assert `output.dtype == input.dtype` for each supported activation dtype and
compare both outputs against an independent Torch oracle. Keep the FP32
arithmetic comparison separate from the visible-output comparison. Also compare
the visible output with the Torch oracle rounded to the same dtype, so kernel
error can be distinguished from mandatory output rounding. Report both error
measurements; keep `rtol=atol=2e-3` or tighter and do not hide raw-oracle drift
caused by low-precision output rounding.

Cover each execution path used by the kernel, including native and dense
fallback paths when both exist. If FP16 or BF16 is unsupported, test its
explicit rejection and document the limitation instead of omitting that dtype.

If an optimized path exceeds a limit, investigate the numerical cause before
changing it. Fix genuine arithmetic or format errors and rerun the A/B tests.
Record unresolved mismatches as merge blockers; retain a faster accurate path
when only proven natural rounding ties differ, with the evidence above.
