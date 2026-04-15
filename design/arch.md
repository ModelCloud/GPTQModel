# W4A4 QQQ Prototype Architecture

Date: 2026-04-15
Prototype commit: `4a2b4076`

## Goal

Add a working `W4A4` path to GPTQModel for local `sm_80` A100-class GPUs using the existing QQQ weight format as the base runtime.

The immediate goal is functional end-to-end support:

- quantize with `activation_bits=4`
- save and reload a checkpoint that preserves that setting
- run generation on CUDA
- benchmark the runtime against the existing `W4A8` QQQ path

The immediate goal is not a final fast kernel. This prototype is a correctness and integration milestone.

## Current Design

The implementation extends the existing QQQ flow rather than creating a brand-new quantization format.

### 1. Config and loader plumbing

- `QQQConfig` now carries `activation_bits` with supported values `{4, 8}`.
- The config serializes `activation_bits` into `quantize_config.json` and model `config.json`.
- The QQQ processor passes `activation_bits` into quant module construction so load and quantize both build the same module shape.

## 2. Python module behavior

`QQQLinear` now supports two activation modes:

- `A8`: existing per-token dynamic quantization to `int8`
- `A4`: new per-token dynamic quantization to signed `int4`, followed by packing 8 values into one `int32`

Runtime helpers:

- `pack_signed_int4(x)`: packs the last dimension from signed nibble codes into `int32`
- `mul(...)`: existing `W4A8` QQQ path
- `mul_w4a4(...)`: new prototype path wired to a new torch op

## 3. CUDA path

The new op is `qqq_w4a4_gemm`.

Current behavior:

1. Receive packed activation tensor `A` as `int32`.
2. Unpack `A` on device into a temporary `int8` tensor.
3. Reuse the existing QQQ GEMM path with the unpacked tensor.
4. Keep the existing QQQ weight packing, scales, reduce buffer, workspace, and output path.

This means the prototype is logically `W4A4`, but physically executes as:

`packed A4 -> unpack to A8 temp -> existing W4A8 QQQ GEMM`

That is why it works without a brand-new tensor-core kernel, and also why it is not yet faster.

## 4. Quantization semantics

Weights:

- remain in QQQ `4-bit` packed format
- keep existing per-channel and per-group scale handling

Activations:

- are quantized dynamically at runtime
- use per-token scale
- use signed symmetric range `[-8, 7]`
- are packed into `int32` for transport into the prototype op

This is GPTQ-like storage behavior for weights, but not literal GPTQ checkpoint semantics for activations. Activations are transient runtime values and need runtime-produced scale metadata.

## 5. Save and load contract

The saved checkpoint now records:

- `method=qqq`
- `format=qqq`
- `bits=4`
- `activation_bits=4`

This allows:

- quantize on one run
- save a W4A4-tagged checkpoint
- reload with `BACKEND.QQQ`
- execute the W4A4 runtime path after load

## File Map

Primary files involved in the prototype:

- `gptqmodel/quantization/config.py`
- `gptqmodel/looper/qqq_processor.py`
- `gptqmodel/nn_modules/qlinear/qqq.py`
- `gptqmodel/utils/qqq.py`
- `gptqmodel_ext/qqq/qqq.cpp`
- `gptqmodel_ext/qqq/qqq_gemm.h`
- `gptqmodel_ext/qqq/qqq_gemm.cu`
- `tests/test_qqq_jit.py`
- `tests/benchmark/qqq_w4a4_llama32_smoke.py`

## Validation Summary

Validated on 2026-04-15 using:

- model: `/monster/data/model/Llama-3.2-1B-Instruct`
- GPUs: PCI-order `8` and `9`
- hardware class: local `sm_80` A100 OEM boards

Observed outcomes:

- quantization completed successfully
- saved checkpoint reloaded successfully
- generation completed successfully
- unit tests passed
- smoke benchmark ran on both GPUs

Prefill benchmark results:

- GPU `8`: `A8 41.65 ms`, `A4 56.70 ms`, `1.36x` slowdown
- GPU `9`: `A8 42.80 ms`, `A4 56.52 ms`, `1.32x` slowdown

## Known Limitations

- The current `W4A4` path is not fused.
- A4 activations are unpacked back to `int8` before the matmul.
- The path adds temporary activation storage and unpack overhead.
- Current measured performance is worse than `W4A8`.
- Accuracy has only been smoke-tested, not evaluated on a real benchmark suite.

## Next Architecture Step

To make `W4A4` worthwhile on A100, the next step is a true fused kernel that:

- consumes packed `W4` and packed `A4` directly
- dequantizes in registers or shared memory
- avoids materializing an unpacked `int8` activation tensor
- reuses QQQ scale semantics where they remain valid
- preserves QQQ integration points so save/load format changes stay minimal

The likely shape is closer to a new Marlin-like or QQQ-derived tensor-core kernel than to a small patch on top of the current bridge implementation.
