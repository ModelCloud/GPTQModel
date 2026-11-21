<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Model / container contract: re-fusion

Some checkpoints store several logical projections in one tensor: an expert's
fused `gate_up_weight`, a fused `qkv_proj`, a fused `gate_up_proj`. **Defusion**
splits such a container into the modules a quantizer can plan and calibrate
per projection. **Re-fusion** rebuilds the container from those modules, so a
layer the plan skipped is saved in the layout its checkpoint declaration
(`experts_formats`, `quantization_config`, ...) describes.

A fused tensor describes exactly one quantization. Re-fusion is therefore a
merge of quantizations, not just of shapes.

## R. Re-fusion

- **R1. One quantization.** Every module merged into one container must carry
  the identical quantization signature: same format (`nvfp4`, `mxfp8`, ...),
  same bit width, same weight and scale dtypes, same block size (values per
  scale), same global-scale kind, and for kernel modules the same `bits`,
  `group_size`, `sym`, `desc_act` and pack dtypes. A container never mixes
  formats, bit widths, block sizes or storage dtypes, and a checkpoint
  declaration never claims a format that only some of its tensors use.
- **R2. Own payload only.** A container is rebuilt only from the payload it was
  defused from. Quantized kernel modules (`BaseQuantLinear` and friends) are
  never merged back into a packed container: their storage is a different
  layout, and concatenating it would silently change the checkpoint's format.
  A quantized layer stays per-module and loses its packed declaration.
- **R3. Refuse, don't repair.** When R1 or R2 does not hold, re-fusion raises
  `RefusionContractError` **before creating any tensor**, naming the container
  and the first pair of disagreeing modules. It never downcasts, re-quantizes,
  pads or drops a module to make the merge fit, and a rejected container keeps
  its defused state so the caller cannot save a half-fused layer.

## Violations

- Concatenating leaves whose `fmt`, bit width, scale dtype or block size differ.
- Re-fusing a layer where any leaf is already a quantized kernel module, or
  where some leaves were quantized and others were not.
- Writing a packed declaration (`experts_formats[layer] = "nvfp4"`) for a layer
  whose saved tensors are per-module quantized linears.
- Re-fusing a layer quantized with different effective configs because
  `dynamic` rules selected only some of its modules.

## Allowed

- Re-fusing untouched leaves: they are views of one payload, so their
  signatures are identical by construction and the round trip is bit-exact.
- Keeping a quantized layer per-module, with its packed declaration cleared.
- Fusing **quantized** modules through the explicit post-quantize fusion
  planner, which rebinds equal-config kernel storage and checks the configs
  itself; that is a runtime fusion, not a checkpoint re-fusion.

## Enforcement

- `gptqmodel.utils.refusion.quantization_signature(module)` derives the
  signature from what the module actually holds: packed payload attributes
  (`packed_weight`, `packed_scales`, `packed_global_scale`, `fmt`), kernel
  payload (`qweight`, `scales`, `bits`, `group_size`, ...), or a dense weight.
- `gptqmodel.utils.refusion.assert_refusable(target, modules, allow=("packed",))`
  enforces R1 and R2 and returns the shared signature.
- `Inkling2LoopPackedExperts.refuse(target)` calls `assert_refusable()` before
  it builds `gate_up_weight` / `down_weight`; `declare_expert_storage()` passes
  the layer path so the error names the layer.

## Tests

`tests/test_inkling2_loop_packed_requant.py` must cover:

- identical leaves re-fuse bit-exactly (`test_packed_experts_refuse_restores_...`);
- a leaf whose format/bit width differs is rejected before mutation
  (`test_refusion_rejects_a_leaf_with_a_different_quantization`);
- a quantized leaf is rejected, naming the module and the contract
  (`test_refusion_rejects_a_quantized_leaf`);
- leaves that agree with each other but not with the container's declared
  format are rejected (`test_refusion_rejects_leaves_that_disagree_with_the_container_declaration`);
- a partially quantized layer is never declared as packed
  (`test_partially_quantized_expert_layer_is_reported_not_saved_mixed`);
- signature classification for packed, dense and kernel modules
  (`test_refusion_signature_separates_packed_dense_and_kernel_modules`).
