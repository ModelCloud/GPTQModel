<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Kernel / module contract: forward dtype

A model runs in one dtype. This contract is about what that means for two kinds
of module:

- **Kernel modules** serve inference forwards from prepared weights. They
  include every `BaseQuantLinear`, the EXL3 linears, and fused projection
  executors.
- **Source modules** are the modules of a model while it is being quantized:
  the linears being quantized, and the norms, previous linears, routers and
  gates that quantization folds into. Calibration capture stand-ins (AUTO_NF12,
  native FP8/FP4) are kernel modules.

## K. Kernel modules

- **K1. One input dtype.** The first eager forward fixes the module's input
  dtype. Every later forward uses that dtype. The output dtype equals the
  input dtype, unless the module documents a fixed wider output (for example an
  FP32 routing step).
- **K2. Frozen weight dtypes.** A module may transform its weights, scales,
  zeros and bias for its kernel in `post_init` or on its first forward. After
  the first forward, no parameter or buffer may change dtype, whatever API is
  used: `.to(dtype)`, `.half()`, `.float()`, `model.to(dtype)`, re-packing,
  buffer re-registration, or assignment. Values and layout may still change in
  place if the module decodes the same weights afterwards, for example qzero
  format conversion, fusion views, offload/restore, or moving to another
  device.
- **K3. Replace, don't mutate.** Different weights or a different dtype means a
  new module. A new module starts unlocked.

Breaking K1 or K2 is a bug in the caller. The module raises instead of
adapting. Choose the dtype at load time, or cast before the first forward.

## Q. Quantization stage

- **Q1. One activation dtype per execution.** Every source module receives and
  returns the model's activation dtype in calibration, search and replay. Wider
  arithmetic stays inside a forward, or on separate copies.

  The one exception is a replay that emulates the deployed runtime. W4A
  replay carries FP32 operands between norm and GEMM because the deployed
  W4A stream does. It must use that carrier consistently for every replay
  pass, and its teacher pass uses the model dtype.
- **Q2. Values may change, dtypes may not.** Between forwards, quantization
  may rewrite source weight values in place:
  - GPTQ-family replay weights (`module.weight.data = wq`);
  - AWQ and SmoothQuant scale folding and clipping;
  - rotation;
  - replacing the weight Parameter object.

  Every rewrite keeps the tensor's existing dtype and shape. Results computed
  wider are cast to the live tensor's dtype before they are written.
- **Q3. Train and search on copies.** Optimization in FP32 or under autocast
  runs on separate instances only: deep copies, standalone optimizer modules,
  or `functional_call`. It never casts the live modules.
- **Q4. Stand-ins are kernels.** A capture stand-in follows K. At hand-off it
  is replaced by a dense source in the source dtype, and it is never forwarded
  again.
- **Q5. Packing ends a source module.** After a source module's last forward,
  the packer may load an exact wider reconstruction into it (`pack_wq`, #5721)
  to recover integer codes. The module is then replaced by a newly built,
  unlocked kernel module. Any whole-model dtype cast happens before the first
  kernel forward.

## What this rules out

- Caches keyed by activation dtype that hold more than one variant, and
  invalidation or rebuilds triggered by a new input dtype.
- `_apply` logic that re-derives a layout after a post-forward dtype cast.
- Per-call conversions kept only so a later call in another dtype stays
  correct.
- Tests that call one module with two dtypes, or that cast a module after a
  forward, and expect it to keep working.
- Issues and PRs whose reproduction needs any of the above. Close them and
  link this contract.

Storing scales, zeros or bias in a dtype other than the activation dtype is
fine, and so is keeping an operand wide for a single-dtype reason (an exact
wide bias epilogue, range-safe decode, bit-exact re-save). The reason must
never be "a later call might use another dtype".

## Enforcement

`ForwardDtypeLockMixin` (`gptqmodel/nn_modules/forward_dtype_lock.py`) is a
base of `BaseQuantLinear`, `ExllamaV3Linear` and `ExllamaV3TorchLinear`. It
wraps every subclass's `forward`, `_apply` and pack entry points:

- The first eager forward records `_forward_dtype_lock`. A later forward in
  another dtype raises. While `torch.compile` is tracing, the lock is checked
  but never recorded, so no recompile is forced.
- Once the lock is set, `_apply` raises if it would change the dtype of any
  tensor in the subtree. The check runs before any tensor changes dtype.
- `pack`, `pack_block`, `pack_gpu`, `pack_original`, `hf_pack` and
  `set_embedding_output_dtype` refuse to run on a locked module.
- A fused projection group applies its source modules' locks before it builds
  or reuses an executor.

On a whole-model cast, modules earlier in the traversal may already have been
cast when a locked module raises; the model is invalid after that error. The
source-module rules (Q) are not enforced in code; review against them.

## Testing

- Cover each activation dtype on its own module instance.
- Assert `output.dtype == input.dtype`, or the documented wider output.
- Check storage across repeated same-dtype forwards, device moves, and
  same-dtype `to()`.
- On one instance, the only cross-dtype test allowed is one that asserts the
  error.

## History

PR #5863, issues #5793 (Marlin) and #5812 (GGUF Triton), and the AWQ
"call-local scales" change from #4178 all treated a BF16 call followed by an
FP16 call on one module as a real workload. That workload does not happen.
The repository audit for this contract also found these violations:

- staged GSQ export ran `model.half()` after packed blocks had run forwards,
  turning BF16 values beyond FP16's range into `inf`;
- Paro layer scope cast the live decoder layer to FP32 for training, then cast
  it back to the linear's dtype.

Both are fixed.
