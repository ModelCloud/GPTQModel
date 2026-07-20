---
name: gptqmodel-quantization
description: Implement, review, or debug GPT-QModel quantization algorithms, calibration processors, quantization formats, packing, serialization, protocol integration, and GPTQ/AWQ/QQQ/FP8/GGUF/EXL3/ParoQuant/RTN/bitsandbytes behavior. Use when a change affects quantized values or metadata rather than only a kernel implementation.
---

# GPT-QModel quantization

Treat the algorithm, serialized format, and inference backend as separate contracts. A model that quantizes successfully is not complete until its metadata round-trips, its packed tensors reload, and a compatible backend produces bounded error.

## Classify the change

Decide which boundaries move before editing:

1. **Algorithm**: calibration statistics, scale/zero search, Hessian use, clipping, ordering, or reconstruction.
2. **Configuration**: public fields, aliases, validation, defaults, or serialization.
3. **Format**: tensor layout, packing, on-disk names, or method/format mapping.
4. **Lifecycle**: processor selection, calibration-free execution, module replacement, or shared state.
5. **Integration**: protocol compilation, serving engines, backend compatibility, or model loading.

Read [references/method-map.md](references/method-map.md) for the current dispatch map and test layers.

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
