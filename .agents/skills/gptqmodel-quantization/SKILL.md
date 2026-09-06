---
name: gptqmodel-quantization
description: Implement, review, or debug GPT-QModel quantization algorithms, calibration processors, quantization formats, packing, serialization, protocol integration, and GPTQ/AWQ/QQQ/FP8/GGUF/EXL3/ParoQuant/RTN/bitsandbytes behavior. Use when a change affects quantized values or metadata rather than only a kernel implementation.
---

# GPT-QModel quantization

Treat the algorithm, serialized format, and inference backend as separate contracts. A model that quantizes successfully is not complete until its metadata round-trips, its packed tensors reload, and a compatible backend produces bounded error.

## Scientific findings for PTQ recovery

For supported quantization methods and enhancements, recovery, EoRA,
NVFP4 activation/KV scales, QTIP/YAQA or P32 design choices,
start with [research/README.md](../../../research/README.md) and use
[$qvq-ptq-recovery-research](../qvq-ptq-recovery-research/SKILL.md).
Keep scale calibration, weight rounding, lossless packing and additive recovery
as separate contracts; cite the relevant research note in design decisions.

## Classify the change

Decide which boundaries move before editing:

1. **Algorithm**: calibration statistics, scale/zero search, Hessian use, clipping, ordering, or reconstruction.
2. **Configuration**: public fields, aliases, validation, defaults, or serialization.
3. **Format**: tensor layout, packing, on-disk names, or method/format mapping.
4. **Lifecycle**: processor selection, calibration-free execution, module replacement, or shared state.
5. **Integration**: protocol compilation, serving engines, backend compatibility, or model loading.

Read [references/method-map.md](references/method-map.md) for the current dispatch map and test layers.
Read [references/quantization_packing.md](references/quantization_packing.md) completely whenever work touches integer
code reconstruction, clipping, zero-points, packing, unpacking, serialization, or cross-packer parity.
Use `$gptqmodel-contiguous-memory` when a kernel or quantizer silently falls back or produces wrong results because a tensor is non-contiguous.

## Trace the existing contract

Inspect the nearest implementation in this order:

- `gptqmodel/quantization/config.py`: `METHOD`, `FORMAT`, mappings, concrete config, validation, and `from_quant_config`/`to_dict` behavior.
- `gptqmodel/models/base.py`: public quantization entry point and processor dispatch.
- `gptqmodel/looper/`: calibration and weight-only processors.
- `gptqmodel/quantization/protocol.py`: public protocol support, when the method is exposed there.
- `gptqmodel/nn_modules/qlinear/`: pack/load expectations and supported backends.
- The closest tests under `tests/qcfg/`, `tests/protocol/`, and method-specific `tests/test_*.py` files.

Use the nearest complete method as a structural template, but preserve the new method's math and metadata semantics. Do not infer a quantization method solely from a format when the format is ambiguous.

## Implement the full path

1. Add or update the enums and method/format mappings only when the public contract changes.
2. Define a concrete configuration with explicit defaults, validation, aliases, and stable serialization. Test both dictionary and saved-config round trips.
3. Route the lifecycle to the appropriate processor:
   - calibration algorithms use a dedicated processor or a deliberately shared processor contract;
   - weight-only methods must work with `calibration_dataset=None`;
   - shared activation, Hessian, or restore caches must be bounded and released after use.
4. Keep the mathematical reference path readable. Establish quantize/dequantize behavior before changing a packed kernel.
5. Add packing and loading support without silently reinterpreting GPTQ and AWQ layouts. Check `sym`, zero-point convention, `desc_act`, group size, dynamic overrides, and pack dtype explicitly.
6. Update protocol and serving integrations only if the new method is meant to be public there. Unsupported integrations should fail clearly rather than fall through to a similar format.

## Validate in layers

Run the smallest applicable checks first:

1. Config validation, aliases, and serialization round trip.
2. Tensor-level quantize/dequantize comparison against a dense BF16, FP16, or FP32 reference, including degenerate groups and non-divisible shapes where supported.
3. Processor behavior on a tiny module: expected tensors, finite values, cleanup, and deterministic behavior under a fixed seed.
4. Pack, save, reload, and inference equivalence through an actual supported backend.
5. A small model quantize/save/load/generate test and, for model-affecting math, an evaluation comparison against the dense baseline.
6. Serving-engine checks for GPTQ or AWQ only when that integration is in scope.

Report the exact method, format, bits, group size, symmetry, activation ordering, dtype, device, and backend with every numerical result. Never use a successful save as the sole correctness signal.

## Reference: Hadamard rotation for outlier suppression

Low-bit quantization (especially int4) is highly sensitive to outliers because the scale is often set by the largest absolute value. A single extreme value widens the spacing between every representable level, which increases relative error for the many small values.

Hadamard rotation is a pre-quantization orthogonal transform that mixes values so that extreme values are distributed across many less-extreme dimensions rather than isolated in a few channels:

- It is an orthogonal rotation, so `Q @ Q.T = I`. Applying the rotation before quantization and its inverse after quantization cancels out in exact arithmetic; the benefit appears only when values are rounded in between.
- Example effect from Jessie Dong's experiment (8 artificial ±25 outliers in a 4096-length vector, symmetric int4):
  - largest absolute value: 25.0 → 5.5
  - int4 step size: 3.571 → 0.781
  - relative error: 0.598 → 0.152 (≈3.9× reduction)
  - RMS stayed at 1.493; the rotation did not remove information, it moved the same information across more channels.
- The Hadamard matrix uses only `+1`/`-1` entries, so the transform can be done with additions and subtractions. The fast Walsh-Hadamard transform runs in `O(n log n)` instead of `O(n²)` and does not require building the full matrix.
- In GPT-QModel, `BaseQuantizeConfig.rotation` accepts `"hadamard"` or `"random"` and the rotation logic lives in `gptqmodel/quantization/rotation/`. The Hadamard path is adapted from QuaRot and can use the vendored fast-hadamard-transform CUDA kernel.
- Activations may need online rotation when the transform cannot be folded into the weights. That overhead can erode the speedup of quantization; fused kernels (e.g. HadaCore, the native `qvq_hadamard_*` kernels) reduce this cost.
- Caveats:
  - Outlier suppression is fundamentally limited by the geometry of the input vector. If several outlier channels land in the same small block, the rotation only spreads them inside that block. Recent work such as PeRQ rearranges channels first so large values are split across different blocks.
  - Rotations are not a substitute for calibration-aware scale/zero search; they are a pre-conditioning step that changes the quantization problem geometry.

Source: Jessie Dong, *one strange way to make 4-bit inference more accurate is to mix the model’s values together before rounding them*, 2026-08-20, https://x.com/jessiedong_/status/2090308407123402875?s=20.

## See also

- [Curated GPU performance engineering resources](references/wafer-gpu-perf-resources.md) — External reading list for quantization and low precision from wafer-ai's performance engineering index.
