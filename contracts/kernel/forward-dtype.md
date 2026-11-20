<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Kernel / module contract: forward activation dtype

Applies to every quantized linear kernel and module: `BaseQuantLinear`
subclasses in `gptqmodel/nn_modules/qlinear/`, the native extensions they
launch, and the forward-time helpers they call. It also applies to any module
that loads weights and then serves `forward`.

## The rule

1. **One activation dtype per module.** The first `forward` call on a loaded
   module fixes its activation dtype. Every later `forward` on that module
   receives the same dtype. A model never switches its activation dtype during
   a run.
2. **Output dtype equals input dtype.** A linear layer returns its output in
   the input dtype. The only exceptions are modules whose documented job
   produces a wider output, such as a routing or transform step that returns
   FP32. Such a module must declare that output dtype explicitly, and it must
   stay the same across calls.
3. **A different dtype on a later call is a bug.** It is a bug in the caller,
   not something the kernel should handle. The kernel must raise an error. It
   must not silently cast, rebuild, re-convert, or keep a second variant.

To change dtype, start a new run: reload the model, or build a new module.
Changing the dtype of a live module is not supported.

## Violations: code that must not exist

- Caches, dicts or entries keyed by activation dtype that hold more than one
  activation-dtype variant of a module's state, such as "wide" and "narrow"
  entries.
- Logic that detects `x.dtype` differing from an earlier call and then
  invalidates, rebuilds or re-derives state.
- Converting stored tensors on every call for the sole reason of keeping a
  later call in a different dtype correct. These are often called
  "call-local" or `_*_for_compute` copies.
- Tests that call one module instance with two activation dtypes and
  expect both calls to succeed, match, or leave storage unchanged.
- Issues and PRs whose reproduction requires such a switch, for example
  "after a BF16 forward, a later FP16 forward drifts". These are closed as
  invalid.

## What stays allowed

- **Storage dtype ≠ activation dtype.** A checkpoint may store scales, zeros
  or bias in FP16 or FP32 while the model runs in BF16. That is a single dtype
  for the whole run, and the kernel must handle it correctly.
- **One-time conversion to the locked dtype.** The module may convert a
  stored operand to the activation dtype once, at `post_init` or on the first
  forward, and reuse that conversion afterwards, for example Marlin's scale
  cast.
- **Keeping wide storage for single-dtype reasons.** A module may keep an
  operand wider than the activation dtype, but only for a reason that holds
  within a single-dtype run, such as:
  - adding a bias exactly in a wide epilogue
  - range-safe decoding
  - re-saving a checkpoint bit-exactly

  The justification must not be "a later call might use another dtype".
- **Different modules, different dtypes.** Two separately loaded models, or
  two module instances, may use different dtypes.

## Enforcement

`BaseQuantLinear` records the dtype of the first `forward` input and raises a
`RuntimeError` when a later call passes a different dtype (the
"forward dtype lock"). Kernels must not bypass or reset the lock. Kernel code
may assume the locked dtype, so it needs no per-dtype branching after the
first call.

## Testing

- Cover each supported activation dtype on its **own** module instance.
  Build or load the module again for each dtype.
- Assert `output.dtype == input.dtype`, or the declared output dtype for the
  documented wide-output modules.
- Test storage stability across repeated forwards in the **same** dtype.
- The only allowed cross-dtype test on one instance asserts that the second
  dtype raises the forward-dtype-lock error.

## History

PR #5863 / issue #5793 (Marlin), issue #5812 (GGUF Triton) and the AWQ "call-local
scales" change from #4178 treated a BF16 call followed by an FP16 call
on the same module as a real workload. That workload does not occur. This
contract was added to stop that class of work.
