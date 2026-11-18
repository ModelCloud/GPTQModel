# Hardware, formats, and backends

[Documentation index](README.md) · [Project overview](../README.md)

Choose a method, checkpoint format, and runtime together. The tables below are
reference inventories, not a promise that every combination works on every device.

## Runtime integrations

| Feature      | GPT-QModel | Transformers | vLLM | SGLang | Lora Training |
|---------------------------|------------|---|---|---|---------------|
| GPTQ                      | ✅          | ✅ | ✅ | ✅ | ✅             |
| AWQ                       | ✅          | ✅ | ✅ | ✅ | ✅             |
| Paro                 | ✅          | x | x | x | ✅             |
| GGUF                      | ✅          | x | x | x | x             |
| FP8                       | ✅          | x | x | x | x             |
| Exllama V3 / EXL3         | ✅          | x | x | x | x             |
| EoRA                      | ✅          | ✅ | ✅ | ✅ | x             |
| Group Aware Act Reordering | ✅          | ✅ | ✅ | ✅ | ✅             |
| QQQ                       | ✅          | x | x | x | x             |
| VPTQ                      | ✅          | x | x | x | x             |
| MACaQ (MXFP4 W4A4)        | ✅          | x | x | x | x             |
| NF12                      | ✅          | x | x | x | x             |
| BitsAndBytes              | ✅          | x | x | x | x             |
| Rotation                  | ✅          | x | x | x | x             |
| GPTAQ                     | ✅          | ✅ | ✅ | ✅ | ✅             |
| FOEM                      | ✅          | ✅ | ✅ | ✅ | ✅             |
| GSQ (GPTQ, AWQ, RTN, QQQ, Paro) | ✅ | Existing format† | Existing format† | Existing format† | Existing format† |

`GGUF`, `EXL3`, `Paro`, `VPTQ`, `MACaQ`, and `NF12` are currently native GPT-QModel quantization/runtime paths. FP8 is load-only: GPT-QModel reads production FP8 checkpoints (transformers `FineGrainedFP8`, DeepSeek/Qwen block scales, NVIDIA ModelOpt, MXFP8, NVFP8) and no longer writes its own FP8 layout, so there is no `gptqmodel_fp8` alias. SGLang loading is limited to `METHOD.GPTQ` with `FORMAT.GPTQ`, `FORMAT.GPTQ_V2`, or `FORMAT.MARLIN`, and `METHOD.AWQ` with `FORMAT.GEMM` or `FORMAT.MARLIN`.

† GSQ preserves the initializer's checkpoint format; it does not add a runtime format.

See [runtime loading](https://github.com/ModelCloud/GPTQModel/wiki/Loading-and-Inference),
[prompt normalization](runtime-prompt-input-normalization.md), and
[external-runtime device mapping](external-runtime-device-mapping.md) for integration constraints.

## Mixed precision and fused modules

Use [`dynamic` rules](https://github.com/ModelCloud/GPTQModel/wiki/QuantizeConfig-Reference#dynamic-per-layer-overrides)
to override individual modules or exclude them from quantization. For vLLM and
SGLang, a merged module and all its shards must have one compatible precision:
routed experts merge into `experts`, gate/up projections into `gate_up_proj`,
and Q/K/V projections into `qkv_proj`.

A rule that excludes only `down_proj` inside an expert can produce a checkpoint
that loads natively but fails in an external engine. Match the merged expert
module and all its children instead:

```python
dynamic = {
    r"-:.*\.20\.mlp\.experts(\.|$)": {},
    r"+:.*\.18\.mlp\.experts(\.|$)": {"bits": 4, "group_size": 32},
}
```

For sibling projections, use an alternation such as
`(gate_proj|up_proj|gate_up_proj)`. GPT-QModel warns about incompatible rules
before quantization. Keep the default `fallback` enabled for external-engine
checkpoints: with `fallback=None`, unvisited experts retain their source dtype,
which can mix precisions inside a merged module.

## Methods, formats, and backends

| Quant Method | Formats | Backends / Kernels |
| --- | --- | --- |
| `METHOD.GPTQ` CPU | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2` | `BACKEND.GPTQ_TORCH_CPU_AVX512` |
| `METHOD.AWQ` CPU | `FORMAT.GEMM` | `BACKEND.AWQ_TORCH_CPU_AVX512` |
| `METHOD.GPTQ` Tenstorrent | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2` | `BACKEND.GPTQ_TT_XLA`, `BACKEND.GPTQ_TTNN`, `BACKEND.GPTQ_TT_BF8` (opt-in `bf8_b` conversion) |
| `METHOD.GPTQ` | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2`, `FORMAT.GPTQ_P`, `FORMAT.MARLIN`, `FORMAT.BITBLAS` | `FORMAT.GPTQ`: `BACKEND.GPTQ_TORCH_ATEN`, `BACKEND.MACHETE`, `BACKEND.MARLIN`, `BACKEND.GPTQ_EXLLAMA_V2`, `BACKEND.GPTQ_TORCH_FUSED`, `BACKEND.GPTQ_TRITON`, `BACKEND.GPTQ_BITBLAS`, `BACKEND.GPTQ_TORCH`, `BACKEND.GPTQ_TORCH_INT8`<br>`FORMAT.GPTQ_V2`: `BACKEND.GPTQ_TORCH_ATEN`, `BACKEND.GPTQ_EXLLAMA_V2`, `BACKEND.GPTQ_TORCH_FUSED`, `BACKEND.GPTQ_TRITON`, `BACKEND.GPTQ_BITBLAS`, `BACKEND.GPTQ_TORCH`, `BACKEND.GPTQ_TORCH_INT8`<br>`FORMAT.GPTQ_P`: `BACKEND.GPTQ_TORCH`<br>`FORMAT.MARLIN`: `BACKEND.MARLIN`<br>`FORMAT.BITBLAS`: `BACKEND.GPTQ_BITBLAS` |
| `METHOD.AWQ` | `FORMAT.GEMM`, `FORMAT.GEMV`, `FORMAT.GEMV_FAST`, `FORMAT.LLM_AWQ`, `FORMAT.MARLIN`, `FORMAT.BITBLAS` | `FORMAT.GEMM`: `BACKEND.AWQ_TORCH_ATEN`, `BACKEND.MACHETE`, `BACKEND.MARLIN`, `BACKEND.AWQ_EXLLAMA_V2`, `BACKEND.AWQ_GEMM`, `BACKEND.AWQ_GEMM_TRITON`, `BACKEND.AWQ_TORCH_FUSED`, `BACKEND.AWQ_TORCH`, `BACKEND.AWQ_BITBLAS`<br>`FORMAT.GEMV`: `BACKEND.AWQ_GEMV`, `BACKEND.AWQ_GEMV_TRITON`, `BACKEND.AWQ_TORCH` (CPU in-session reader)<br>`FORMAT.GEMV_FAST`: `BACKEND.AWQ_GEMV_FAST`, `BACKEND.AWQ_TORCH` (CPU in-session reader)<br>`FORMAT.LLM_AWQ`: `BACKEND.AWQ_GEMV_FAST`, `BACKEND.AWQ_TORCH` (CPU in-session reader)<br>`FORMAT.MARLIN`: `BACKEND.MACHETE`, `BACKEND.MARLIN`<br>`FORMAT.BITBLAS`: `BACKEND.AWQ_BITBLAS` |
| `METHOD.GPTQ` / `METHOD.AWQ` Swordfish | `FORMAT.GPTQ` / `FORMAT.GEMM` | `BACKEND.SWORDFISH` |
| `METHOD.GPTQ` / `METHOD.AWQ` AllSpark | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2` / `FORMAT.GEMM` | `BACKEND.ALLSPARK` (per-channel 8-bit) |
| `METHOD.PARO` | `FORMAT.PARO` | `BACKEND.PARO_CUDA`, `BACKEND.PARO_TRITON` |
| `METHOD.QQQ` | `FORMAT.QQQ` | `BACKEND.QQQ`, `BACKEND.QQQ_TORCH` |
| `METHOD.VPTQ` | `FORMAT.VPTQ` | `BACKEND.VPTQ_CUDA`, `BACKEND.VPTQ_TORCH` |
| `METHOD.MACAQ` | `FORMAT.MXFP4_MACAQ` | `BACKEND.MACAQ_TRITON`, `BACKEND.MACAQ_TORCH` |
| `METHOD.NF12` (`nf12`) | `FORMAT.NF12` (`nf12`) | `BACKEND.NF12` (`nf12`) |
| `METHOD.GGUF` | `FORMAT.GGUF` | `BACKEND.GGUF_TRITON`, `BACKEND.GGUF_CPP_CUDA`, `BACKEND.GGUF_CPP_CPU`, `BACKEND.GGUF_TORCH` |
| `METHOD.FP8` | `FORMAT.FP8` | `BACKEND.FP8_TORCH` |
| `METHOD.BITSANDBYTES` | `FORMAT.BITSANDBYTES` | `BACKEND.BITSANDBYTES` |
| `METHOD.EXL3` | `FORMAT.EXL3` | `BACKEND.EXL3_EXLLAMA_V3`, `BACKEND.EXL3_TORCH` |

Backend aliases and dispatch rules are covered in the
[backend reference](https://github.com/ModelCloud/GPTQModel/wiki/Kernel-Backends-Reference).
See also [CPU dispatch](kernel-dispatch-cpu.md), [MLX](mlx-runtime.md),
[Tenstorrent](tenstorrent.md), and [packed INT4 transformations](torch_fused_int4_transformations.md).

## Hardware and operating systems

| Platform | Device |  | Optimized Arch | Kernels |
|---|---|---|---|---|
| 🐧 Linux | NVIDIA GPU | ✅ | `Turing+` (`sm_75+`); Marlin, NF12 and block-float CUDA `sm_80+`; backend-specific limits apply | Machete, Marlin, Swordfish, AllSpark, Exllama V3 / EXL3, Exllama V2, AWQ GEMM/GEMV, Paro CUDA/Triton, MACaQ Triton/Torch, GGUF CUDA/Triton, QQQ, VPTQ, NF12, `bf4_b`/`bf8_b` CUDA, BitBLAS, Triton, BitsAndBytes, Torch |
| 🐧 Linux | AMD GPU | ✅ | `ROCm 10.1+`, `gfx950` (CDNA4), `gfx1200/1201/1250` (RDNA4); backend-specific limits apply | AWQ Triton GEMM/GEMV, Paro Triton, MACaQ Triton (`gfx1201`) / Torch, GGUF Triton, BitsAndBytes 4-bit Triton, QQQ, NF12 HIP (RDNA4), AllSpark (gfx1200/1201), `bf4_b`/`bf8_b` HIP, FP8 Torch, Torch |
| 🐧 Linux | Tenstorrent | ✅ | `Blackhole p300c`, `DEVICE.XLA` | Packed GPTQ via Torch/XLA/PJRT or native TTNN; opt-in `bf8_b` TTNN; multi-card execution |
| 🐧 Linux | Huawei Ascend NPU | ✅ | `Ascend 910B`, `torch-npu` / `CANN` | Native Torch kernels for GPTQ, AWQ, Paro, GGUF, QQQ, and EXL3 |
| 🐧 Linux | Intel XPU | ✅ | `Arc`, `Datacenter Max` | TorchFused, TorchFusedAWQ, FP8 Torch, Torch |
| 🐧 Linux | Intel/AMD CPU | ✅ | x86-64; AVX2, AVX512-BF16, or AMX for applicable optimized paths | GPTQ/AWQ AVX512-BF16, TorchFused, TorchFusedAWQ, TorchAten int4, GPTQ INT8 (8-bit sources), GGUF C++, BitsAndBytes, VPTQ, NF12 Torch, `ttbfp` block floats, Torch |
| 🍎 macOS | GPU (Metal) / CPU | ✅ | `Apple Silicon`, `M1+` | Native MLX quantization and packed inference (GPTQ, AWQ, Paro, QQQ, GGUF, FP8, MACaQ, NF12, EXL3); MPS/Torch, FP8 Torch, NF12 Torch |
| 🪟 Windows | GPU (NVIDIA) / CPU | ✅ | `NVIDIA` | Torch |

Availability depends on each backend's architecture, dtype, shape, and runtime
requirements. A platform row does not imply that every listed kernel runs on
every device in that family. Tenstorrent hardware inference and CPU block-float
execution are separate paths; CPU block-float kernels do not require TTNN or PJRT.

ROCm requires version 10.1 or newer and a matching PyTorch build. See
[RDNA4 support](rocm-rdna4.md), [ROCm defaults](rocm-default-environment.md),
and [JIT toolkit discovery](rocm-jit-toolkit-paths.md) for restrictions and setup.
