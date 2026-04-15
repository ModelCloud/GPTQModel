# W4A4 QQQ Prototype Architecture

Date: 2026-04-15
Base prototype commit: `4a2b4076`
Current status: `working packed-A4 in-kernel staging prototype`

## Goal

Add a working `W4A4` path to GPTQModel for local `sm_80` A100-class GPUs using the existing QQQ weight format as the base runtime.

The current milestone is stronger than the first bridge prototype:

- quantize with `activation_bits=4`
- save and reload a checkpoint that preserves that setting
- run generation on CUDA
- benchmark the runtime against the existing `W4A8` QQQ path
- avoid a separate dense activation dequant kernel and temporary `A8` activation tensor

The current milestone is still not the final fast kernel. The MMA path is still the existing int8 QQQ path.

## Current Design

The implementation extends the existing QQQ flow rather than creating a brand-new quantization format.

### 1. Config and loader plumbing

- `QQQConfig` carries `activation_bits` with supported values `{4, 8}`.
- The config serializes `activation_bits` into `quantize_config.json` and model `config.json`.
- The QQQ processor passes `activation_bits` into quant module construction so load and quantize both build the same module shape.

## 2. Python module behavior

`QQQLinear` supports two activation modes:

- `A8`: existing per-token dynamic quantization to `int8`
- `A4`: per-token dynamic quantization to signed `int4`, followed by packing 8 values into one `int32`

Runtime helpers:

- `pack_signed_int4(x)`: packs the last dimension from signed nibble codes into `int32`
- `mul(...)`: existing `W4A8` QQQ path
- `mul_w4a4(...)`: packed-A4 path wired to a dedicated torch op

## 3. CUDA path

The new op is `qqq_w4a4_gemm`.

Current behavior:

1. Receive packed activation tensor `A` as `int32`.
2. Launch a dedicated QQQ translation unit for `W4A4`.
3. Inside the GEMM kernel, read packed `A4` tiles directly from global memory.
4. Unpack those tiles directly into the shared-memory `A` staging buffer that the existing QQQ MMA path already consumes.
5. Continue with the existing `W4` dequant flow, shared-memory pipeline, and int8 tensor-core matmul.

This means the current prototype is logically:

`packed A4 global -> unpack into shared A tile inside GEMM -> existing W4/int8 MMA path`

Important distinction from the first prototype:

- there is no standalone unpack kernel
- there is no materialized dense `A8` activation tensor
- activation unpack now happens in the kernel’s tile-fetch path while the rest of the QQQ pipeline continues staging weights and scales

## 4. Quantization semantics

Weights:

- remain in QQQ `4-bit` packed format
- keep existing per-channel and per-group scale handling

Activations:

- are quantized dynamically at runtime
- use per-token scale
- use signed symmetric range `[-8, 7]`
- are packed into `int32` for transport into the `qqq_w4a4_gemm` op

This is GPTQ-like storage behavior for weights, but not literal GPTQ checkpoint semantics for activations. Activations are transient runtime values and need runtime-produced scale metadata.

## 5. Save and load contract

The saved checkpoint records:

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

Primary files involved in the current prototype:

- `gptqmodel/quantization/config.py`
- `gptqmodel/looper/qqq_processor.py`
- `gptqmodel/nn_modules/qlinear/qqq.py`
- `gptqmodel/utils/qqq.py`
- `gptqmodel_ext/qqq/qqq.cpp`
- `gptqmodel_ext/qqq/qqq_gemm.h`
- `gptqmodel_ext/qqq/qqq_gemm.cu`
- `gptqmodel_ext/qqq/qqq_gemm_w4a4.cu`
- `tests/test_qqq_jit.py`
- `tests/benchmark/qqq_w4a4_llama32_smoke.py`

## Validation Summary

Validated on 2026-04-15 using:

- model: `/monster/data/model/Llama-3.2-1B-Instruct`
- checkpoint: `/tmp/llama3_2_1b_instruct_qqq_w4a4_gpu8`
- GPUs: PCI-order `8` and `9`
- hardware class: local `sm_80` A100 OEM boards

Observed outcomes:

- checkpoint reload succeeded
- CUDA generation completed successfully on GPUs `8` and `9`
- packed-A4 kernel completed prefill without illegal memory access
- a focused zero-weight CUDA smoke passed for both `A8` and `A4`
- end-to-end smoke benchmark ran in parallel, one process per GPU

Prefill benchmark results from the in-kernel packed-A4 path:

- GPU `8`: `A8 40.75 ms`, `A4 55.20 ms`, `1.35x` slowdown
- GPU `9`: `A8 41.95 ms`, `A4 58.94 ms`, `1.40x` slowdown

Generation smoke results:

- GPU `8`: `32` tokens in `2.146 s`
- GPU `9`: `32` tokens in `2.259 s`

## Known Limitations

- The current `W4A4` path is fused at the staging level, not at the tensor-core primitive level.
- A4 activations are unpacked inside the GEMM kernel, but the MMA path still consumes int8 fragments.
- Packed-A4 loads are not yet using a fully async copy path comparable to the existing `A8` cp.async flow.
- Current measured performance is still worse than `W4A8`.
- Accuracy has only been smoke-tested, not evaluated on a real benchmark suite.

## Next Architecture Step

To make `W4A4` worthwhile on A100, the next step is a more native fused kernel that:

- keeps packed `W4` and packed `A4` in the tile pipeline longer
- minimizes or eliminates per-tile unpack overhead on the critical path
- preserves the no-temporary-activation-tensor property
- reuses QQQ scale semantics where they remain valid
- preserves QQQ integration points so save/load format changes stay minimal

The likely shape is still closer to a new Marlin-like or QQQ-derived tensor-core kernel than to a small patch on top of the current prototype.
