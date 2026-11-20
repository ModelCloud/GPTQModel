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

4. **Weight dtypes are frozen after the first forward.** A module may
   transform its weights, scales, zeros and bias to suit its kernel. It may do
   this in `post_init`, or on its first forward. Once a forward has run,
   nothing may change the dtype of the module's parameters or buffers. This
   applies to the module itself, to modeling code, and to any higher-level
   API. Examples: `module.to(other_dtype)`, `.half()`, `.bfloat16()`,
   `.float()`, `model.to(dtype)`, or reassigning a weight in another dtype.
   Any of these is a bug in the caller, and `BaseQuantLinear` raises. Moving
   a module to another device keeps the dtype, so it is still allowed.

To change dtype, start a new run: reload the model, or build a new module.
Changing the dtype of a module that has already run a forward is not
supported. Choose the dtype at load time, or cast before the first forward.

## Violations: code that must not exist

- Caches, dicts or entries keyed by activation dtype that hold more than one
  activation-dtype variant of a module's state, such as "wide" and "narrow"
  entries.
- Logic that detects `x.dtype` differing from an earlier call and then
  invalidates, rebuilds or re-derives state.
- Logic that re-derives a kernel layout after a post-forward weight dtype
  change, for example `_apply` restoring a snapshot after `layer.to(bf16)`.
  The dtype change itself is the bug. Device-move handling stays.
- Converting stored tensors on every call for the sole reason of keeping a
  later call in a different dtype correct. These are often called
  "call-local" or `_*_for_compute` copies.
- Tests that call one module instance with two activation dtypes and
  expect both calls to succeed, match, or leave storage unchanged.
- Tests that run a forward, change the module's dtype, and expect the module
  to keep working.
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
- **Dtype casts before the first forward.** Loading at a dtype, or calling
  `model.to(dtype)` before any forward, is fine.
- **Device moves at any time**, for example CPU → XPU. A kernel may need to
  re-derive a device-specific layout after a move.
- **Keeping wide storage for single-dtype reasons.** A module may keep an
  operand wider than the activation dtype, but only for a reason that holds
  within a single-dtype run, such as:
  - adding a bias exactly in a wide epilogue
  - range-safe decoding
  - re-saving a checkpoint bit-exactly

  The justification must not be "a later call might use another dtype".
- **Different modules, different dtypes.** Two separately loaded models, or
  two module instances, may use different dtypes.
- **Quantization replaces modules; it does not mutate them.** Quantizing
  produces weights in a new target form from the source weights. It does this
  by swapping the source module (for example an `nn.Linear`) for a new kernel
  module. The new module has never run a forward, so it starts with no lock,
  and its first forward fixes its dtype. Calibration forwards on source
  modules are outside this contract. If a quantization path (`requantize`,
  conversion, post-quant replay) needs different weights or dtypes on a
  kernel module that has already run a forward, it must build a replacement
  module. It must not cast or rewrite the existing one.

## Enforcement

`BaseQuantLinear` records the dtype of the first `forward` input (the
"forward dtype lock"). It raises a `RuntimeError` in two cases:

- a later call passes a different input dtype;
- after the lock is set, a `Module._apply` (`to`, `half`, `float`, and so on)
  would change the dtype of any of its parameters or buffers. The check runs
  before that module is modified. On a whole-model cast, modules that come
  earlier in the traversal may already have been cast. The model is invalid
  after this error.

Kernels must not bypass or reset the lock. Kernel code may assume the locked
dtype and frozen weight dtypes, so it needs no per-dtype branching or
dtype-move invalidation after the first call.

## Testing

- Cover each supported activation dtype on its **own** module instance.
  Build or load the module again for each dtype.
- Assert `output.dtype == input.dtype`, or the declared output dtype for the
  documented wide-output modules.
- Test storage stability across repeated forwards in the **same** dtype.
- Test storage stability after a device move or a same-dtype `to()`.
- The only allowed cross-dtype test on one instance asserts that the error is
  raised. That covers both a second input dtype and a dtype change after the
  first forward.

## History

PR #5863 / issue #5793 (Marlin), issue #5812 (GGUF Triton) and the AWQ "call-local
scales" change from #4178 treated a BF16 call followed by an FP16 call
on the same module as a real workload. That workload does not occur. This
contract was added to stop that class of work.
