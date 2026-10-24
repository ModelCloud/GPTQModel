<!-- SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai -->
<!-- SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Contact: qubitium@modelcloud.ai, x.com/qubitium -->

<p align=center>
<div align=center>
<img src="https://github.com/user-attachments/assets/ab70eb1e-06e7-4dc9-83e5-bd562e1a78b2" width=500>
</div>
<h1 align="center">GPT-QModel ⚡</h1>
</p>
<p align="center"><strong>An extensible platform for LLM quantization, validation, and deployment.</strong><br>GPTQ, AWQ, ParoQuant, GGUF, FP8, EXL3, QQQ, and more—across NVIDIA CUDA, AMD ROCm, Huawei Ascend, Intel XPU, and CPU, with Transformers, vLLM, and SGLang.</p>
<p align="center">
    <a href="https://github.com/ModelCloud/GPTQModel/releases" style="text-decoration:none;"><img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/GPTQModel.svg"></a>
    <a href="https://pypi.org/project/gptqmodel/" style="text-decoration:none;"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/gptqmodel"></a>
    <a href="https://pepy.tech/projects/gptqmodel" style="text-decoration:none;"><img src="https://static.pepy.tech/badge/gptqmodel" alt="PyPI Downloads"></a>
    <a href="https://github.com/ModelCloud/GPTQModel/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/gptqmodel"></a>
    <a href="https://huggingface.co/modelcloud/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ModelCloud-%23ff8811.svg"></a>
    <a href="https://huggingface.co/models?search=gptq">
        <img alt="Huggingface - Models" src="https://img.shields.io/badge/🤗_6.7K_gptq_models-8A2BE2">
    </a>
    <a href="https://huggingface.co/models?search=awq">
        <img alt="Huggingface - Models" src="https://img.shields.io/badge/🤗_8.2K_awq_models-8A2BE2">
    </a>
</p>

## Latest News 🗞️🚀

* **7.7.0**:
  * **CPU inference:** Added the shared AVX512-BF16 `int4_qzero_group_scaled_cpu` kernel for GPTQ/AWQ group sizes 16/32/64/128/256/512 and prioritized it ahead of ATen when supported. Added packed/low-precision CPU decode paths for QQQ, VPTQ, EXL3, FP8, NVFP4, and BitsAndBytes, plus oneDNN BF16x3 accumulation and shape-aware BitsAndBytes INT8 dispatch. GPTQ/AWQ measured up to 2.02x faster than ATen; BitsAndBytes INT8 reached 1.28x at M=64.
  * **GGUF:** Added fused CPU Q4_0/Q8_0/Q4_K/Q5_K/Q6_K forward with FP32 accumulation under oneDNN BF16x3, fused bias, per-shape autotuning, and no retained dense weights. K-quant fused paths measured up to 1.92x faster than dense at M=7.
  * **Quantized storage:** Removed persistent dense, dequantized, expanded, and INT4-to-INT8 inference caches across GPTQ/AWQ, QQQ, EXL3, VPTQ, FP8, FP4, GGUF, BitsAndBytes, and INT8. Only packed weights, scales, zero points, and metadata persist; temporary dequantization uses bounded tiles.
  * **Quantization quality:** Added token-normalized GPTQ loss reporting, on-device embedding diagonal flooring and VPTQ Hessian damping, and QQQ vLLM compatibility/CUDA saturation fixes.
  * **Models and runtimes:** Added LLaVA quantization support, fused VPTQ CUDA inference, and hardened remote LoRA adapter loading.
  * **Docs and tooling:** Added the GPTQ FSDP/PEFT LoRA example, multi-GPU MoE quantization guide, and independent Torch-oracle accuracy and CPU speed benchmarks.
  * **Dependencies:** Refreshed runtime and optional dependency floors to current stable releases, including PyTorch 2.14.1, Transformers 5.20.2, Datasets 5.1.0, and Optimum 2.3.0. vLLM/SGLang extras retain their upstream framework pins and should be installed in dedicated environments.

* **7.6.0**:
  * **Model support:** Added Jamba/Jamba2, SmolVLM, Granite MoE, InternVL, EXAONE 4.5, Qwen3-VL MoE, Qwen4-Exp text, OLMoE, ZDTaichu 5.0, Agnes 3.0 Flash, and Xing4.0 quantization support, plus MiMo V2.6 Flash-RL and Pro-RL checkpoint support.
  * **Apple silicon:** Added native MLX GPTQ/AWQ quantization and an EXL3 weight-quantization pipeline. Expanded Metal quantization and packed inference kernels for ParoQuant, GGUF, FP8, QQQ, and BitsAndBytes, with GPTAQ/FOEM/RTN acceleration, broader GPTQ/AWQ bit-width and group-size support, Marlin/BitBLAS checkpoint loading, and FP16/BF16 output preservation.
  * **GPTQ performance:** Added fused CUDA block updates for 2–8-bit quantization, faster Hessian collection and factor reuse, lower peak Hessian memory, and faster Cholesky damping recovery.
  * **Experimental W4A:** Added Llama decoder activation streams using FP8 on NVIDIA SM89+ or NVFP4 on GB10 (SM121), while preserving native packed GPTQ INT4 checkpoints. See the [W4A guide](docs/gptq-w4a-gb10.md) for supported configurations and measured quality.
  * **Source formats:** Added automatic decoding of native FP8/NVFP4 and OCP MXFP4/MXFP8 source checkpoints for requantization.
  * **Runtime and integration:** Added stable [`hf_` kernel selection, packing, and repacking APIs](docs/hf_packing_api.md), improved AWQ Triton scheduling, and added opt-in [Marlin scratch-buffer reuse](docs/marlin_scratch.md). Improved SGLang format validation, dtype handling, and generation parameter compatibility.
  * **Reliability:** Fixed model-tree isolation and dynamic-rule caching/serialization, offloaded tensor device/dtype restoration, fused DeepSeek expert loading, Qwen3.5-MoE multimodal calibration, Gemma4 calibration KV memory retention, ParoQuant positional replay, GLM NoPE configuration handling, and GPTQ damping recovery after inverse-factor failures.

* 09/15/2026 [7.5.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.5.0): ✨ Added K2 Horizon, Qwen-Drive, Nanbeige, and Diffusion Gemma quantization support. Improved GPTQ/AWQ/ParoQuant reliability with device-aware kernel selection, safer CUDA dispatch and empty-input handling, partial packing-word support, correct calibration masks, model-specific Paro layer replay, and hardened Machete runtime caching. Loading and saving now validate GPTQ `qweight` and offload metadata, while InternVL image preprocessing keeps `torchvision` optional. Quantization finalization is faster and startup version reporting is more accurate.
* 09/07/2026 [7.4.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.4.0): 🎉 Added resumable quantization checkpoints, shared-input Hessian deduplication, `lm_head` and embedding requantization, and updated native GGUF support. Added GLM-5.3-Flash, Apertus 1.5, and XHToken `ouro` / `spark2_5` quantization support, plus quantization, JIT cache, and Triton compatibility fixes.
* 08/31/2026 [7.3.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.6): ✨ Added HunyuanOCR, NVIDIA LocateAnything-3B, and Qwen3.8-Flash-Next quantization support. Added tile-misaligned GPTQ/AWQ Marlin support, reduced QQQ packing memory, and improved JIT extension cache reuse.
* 08/25/2026 [7.3.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.5): ✨ Added `lm_head` and embedding quantization lifecycle support; Unlimited-OCR, DeepSeek V3.2, Mage-VL, Muse Glimmer, OLMo 3, and SmolLM3 model support.
* 08/19/2026 [7.3.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.4): 🚀🔥⚡ Added the `Swordfish` Blackwell GPTQ/AWQ kernel, planar GPTQ checkpoint formats, native MPS quantization, and Cohere Compass / A.X-K2 model support, with quantization performance and reliability improvements.
Older releases and development notes: [changelog archive](docs/changelog/archive.md).

## A unified quantization platform

GPT-QModel provides a consistent API for calibration, quantization, quality evaluation, model conversion, and accelerated inference. It supports GPTQ, AWQ, ParoQuant, QQQ, GGUF, FP8, EXL3, GPTAQ, EoRA, GAR, and FOEM across supported hardware and runtime integrations.

Its method, format, backend, and kernel layers are modular: method-specific controls remain available where needed, while implementations share the same model lifecycle. This architecture supports mixed and model-specific quantization workflows today and provides a clear integration path for additional methods, formats, kernels, and accelerators.

`TorchLinear` never retains dense/dequantized weights between inference calls.
The legacy `GPTQ_TORCH_CACHE_WEIGHTS` / `GPTQ_TORCH_LOOKAHEAD` flags and
`enable_weight_cache(True)` / `enable_lookahead(True)` APIs now warn and leave
these features disabled. CUDA streaming uses per-forward tiles, not persistent
dequantized workspaces. Only zero-point and group-index metadata is cached;
call `clear_weight_cache()` after manually mutating inference-mode tensors
without version counters. Normal tensor mutations, checkpoint loads, and
device/dtype moves invalidate this metadata automatically.

`BACKEND.GPTQ_TORCH_INT8` (including its `TORCH_INT8` alias) accepts only
8-bit GPTQ sources. Converting 2/4-bit weights into a persistent 8-bit runtime
representation is no longer supported. `AWQ_TORCH_INT8` / `TORCH_INT8_AWQ`
are retired and fail with migration guidance. Use `BACKEND.AUTO` to select a
compatible packed kernel or `BACKEND.TORCH` for a portable fallback. Native
8-bit GPTQ still uses the existing per-channel INT8 requantizer; it is not a
lossless repacking of arbitrary grouped/asymmetric GPTQ weights.

## Quantization Support 🛠️

Start with the **[quantization guide and end-to-end recipes](docs/quantization-guide.md)**
to choose weight and activation precision, prepare calibration data, quantize,
save, reload, and validate a model. It also maps advanced features to their guides.

Every quantization method shares a common lifecycle—calibrate, quantize, validate, save, load, and serve—while retaining method-specific controls where they matter. New methods and kernels plug into that lifecycle instead of creating another disconnected toolchain.

| Feature      | GPT-QModel | Transformers | vLLM | SGLang | Lora Training |
|---------------------------|------------|---|---|---|---------------|
| GPTQ                      | ✅          | ✅ | ✅ | ✅ | ✅             | 
| AWQ                       | ✅          | ✅ | ✅ | ✅ | ✅             |
| ParoQuant                 | ✅          | x | x | x | ✅             |
| GGUF                      | ✅          | x | x | x | x             |
| FP8                       | ✅          | x | x | x | x             |
| Exllama V3 / EXL3         | ✅          | x | x | x | x             |
| EoRA                      | ✅          | ✅ | ✅ | ✅ | x             | 
| Group Aware Act Reordering | ✅          | ✅ | ✅ | ✅ | ✅             |
| QQQ                       | ✅          | x | x | x | x             | 
| VPTQ                      | ✅          | x | x | x | x             |
| Rotation                  | ✅          | x | x | x | x             |  
| GPTAQ                     | ✅          | ✅ | ✅ | ✅ | ✅             |
| FOEM                      | ✅          | ✅ | ✅ | ✅ | ✅             |

`GGUF`, `FP8`, `EXL3`, `ParoQuant`, and `VPTQ` are currently native GPT-QModel quantization/runtime paths. SGLang loading is limited to `METHOD.GPTQ` with `FORMAT.GPTQ`, `FORMAT.GPTQ_V2`, or `FORMAT.MARLIN`, and `METHOD.AWQ` with `FORMAT.GEMM` or `FORMAT.MARLIN`.

SGLang accepts the common engine aliases `tensor_parallel_size` → `tp_size`, `gpu_memory_utilization` → `mem_fraction_static`, `max_model_len` → `context_length`, `seed` → `random_seed`, and `enforce_eager` → `disable_cuda_graph`. An explicit legal `dtype` is preserved; deprecated `torch_dtype` is normalized to SGLang's string dtype names. AWQ `FORMAT.GEMV_FAST` and `FORMAT.LLM_AWQ` still require `torch.float16`, but those formats are not in SGLang's supported-format list.

### Quant Method / Format / Backend Matrix 📋

Canonical backend names are shown below. Method-specific aliases are only accepted where explicitly implemented by that quant method.

| Quant Method | Formats | Backends / Kernels |
| --- | --- | --- |
| `METHOD.GPTQ` CPU | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2` | `BACKEND.GPTQ_TORCH_CPU_AVX512` |
| `METHOD.AWQ` CPU | `FORMAT.GEMM` | `BACKEND.AWQ_TORCH_CPU_AVX512` |
| `METHOD.GPTQ` | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2`, `FORMAT.MARLIN`, `FORMAT.BITBLAS` | `FORMAT.GPTQ`: `BACKEND.GPTQ_TORCH_ATEN`, `BACKEND.GPTQ_MACHETE`, `BACKEND.GPTQ_MARLIN`, `BACKEND.GPTQ_EXLLAMA_V2`, `BACKEND.GPTQ_TORCH_FUSED`, `BACKEND.GPTQ_TRITON`, `BACKEND.GPTQ_BITBLAS`, `BACKEND.GPTQ_TORCH`, `BACKEND.GPTQ_TORCH_INT8`<br>`FORMAT.GPTQ_V2`: `BACKEND.GPTQ_TORCH_ATEN`, `BACKEND.GPTQ_EXLLAMA_V2`, `BACKEND.GPTQ_TORCH_FUSED`, `BACKEND.GPTQ_TRITON`, `BACKEND.GPTQ_BITBLAS`, `BACKEND.GPTQ_TORCH`, `BACKEND.GPTQ_TORCH_INT8`<br>`FORMAT.MARLIN`: `BACKEND.GPTQ_MARLIN`<br>`FORMAT.BITBLAS`: `BACKEND.GPTQ_BITBLAS` |
| `METHOD.AWQ` | `FORMAT.GEMM`, `FORMAT.GEMV`, `FORMAT.GEMV_FAST`, `FORMAT.LLM_AWQ`, `FORMAT.MARLIN`, `FORMAT.BITBLAS` | `FORMAT.GEMM`: `BACKEND.AWQ_TORCH_ATEN`, `BACKEND.AWQ_MACHETE`, `BACKEND.AWQ_MARLIN`, `BACKEND.AWQ_EXLLAMA_V2`, `BACKEND.AWQ_GEMM`, `BACKEND.AWQ_GEMM_TRITON`, `BACKEND.AWQ_TORCH_FUSED`, `BACKEND.AWQ_TORCH`, `BACKEND.AWQ_BITBLAS`<br>`FORMAT.GEMV`: `BACKEND.AWQ_GEMV`<br>`FORMAT.GEMV_FAST`: `BACKEND.AWQ_GEMV_FAST`<br>`FORMAT.LLM_AWQ`: `BACKEND.AWQ_GEMV_FAST`<br>`FORMAT.MARLIN`: `BACKEND.AWQ_MACHETE`, `BACKEND.AWQ_MARLIN`<br>`FORMAT.BITBLAS`: `BACKEND.AWQ_BITBLAS` |
| `METHOD.PARO` | `FORMAT.PAROQUANT` | `BACKEND.PAROQUANT_CUDA`, `BACKEND.PAROQUANT_TRITON` |
| `METHOD.QQQ` | `FORMAT.QQQ` | `BACKEND.QQQ`, `BACKEND.QQQ_TORCH` |
| `METHOD.VPTQ` | `FORMAT.VPTQ` | `BACKEND.VPTQ_CUDA`, `BACKEND.VPTQ_TORCH` |
| `METHOD.GGUF` | `FORMAT.GGUF` | `BACKEND.GGUF_TRITON`, `BACKEND.GGUF_CPP_CUDA`, `BACKEND.GGUF_CPP_CPU`, `BACKEND.GGUF_TORCH` |
| `METHOD.FP8` | `FORMAT.FP8` | `BACKEND.FP8_TORCH` |
| `METHOD.BITSANDBYTES` | `FORMAT.BITSANDBYTES` | `BACKEND.BITSANDBYTES` |
| `METHOD.EXL3` | `FORMAT.EXL3` | `BACKEND.EXL3_EXLLAMA_V3`, `BACKEND.EXL3_TORCH` |

`BACKEND.GPTQ_TORCH_CPU_AVX512` and `BACKEND.AWQ_TORCH_CPU_AVX512` share the
same AVX512-BF16 int4 GEMM on Linux x86-64 hosts with AVX512-BF16,
AVX512-VBMI, and OS AVX512 state enabled. GPTQ and AWQ use separate fused
checkpoint-to-kernel packers but no separate compute kernels. `AUTO` selects
the AVX512 path ahead of the ATen int4pack fallback when the layer shape
satisfies the kernel's 32-row and group-size alignment requirements. Supported
group sizes are 16, 32, 64, 128, 256, and 512.

`BACKEND.QQQ_TORCH` uses the same packed INT4 CPU layout with a QQQ-specific
integer GEMM that preserves round-to-nearest-even and INT8 saturation for
group size 128, plus the signed per-channel layout for group size -1.
Small batches use AVX512 integer dot products; larger batches use oneDNN
integer GEMM with temporary 128-by-32 weight tiles when available. Runtime
caches retain only packed INT4 codes and scales. Unsupported hosts use the
Torch fallback without retaining dequantized weights. Both FP16 and BF16
activations keep their output dtype.

`BACKEND.EXL3_TORCH` uses oneDNN BF16x3 FP32 matmul on AVX512-BF16 hosts for
GEMMs large enough to amortize dispatch. Forward decodes at most 512 output
columns at a time and never retains a dense dequantized weight;
`get_weight_tensor()` materializes one on demand for export and tests. Set
`GPTQMODEL_EXL3_CPU_BF16X3=0` to disable the BF16x3 path.

`BACKEND.VLLM`, `BACKEND.SGLANG`, and `BACKEND.MLX` are external runtime backends and are not part of the native kernel matrix above.

Marlin uses `GPTQMODEL_MARLIN_USE_FP32` (default: enabled) to control fp32 accumulation.

On ROCm, quantization runs with torch's rocBLAS backend instead of hipBLASLt: hipBLASLt
loads one Tensile library per (dtype, layout) pair on first use, and those one-time loads
dominate a cold quantize while rocBLAS covers the same math. The override is scoped to the
quantization phase and the previous backend is restored afterwards. Set
`GPTQMODEL_ROCM_QUANT_BLAS=hipblaslt` to keep torch's default backend.

## Features ✨
* ✨ Native integration with HF [Transformers](https://github.com/huggingface/transformers), [Optimum](https://github.com/huggingface/optimum), and [Peft](https://github.com/huggingface/peft)
* ✨ Stable `hf_` [packing/repacking API](docs/hf_packing_api.md) for layer-wise kernel feasibility, packing, and repacking used by external integrators such as AutoRound.
* 🚀 [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) inference integration for quantized models. SGLang supports GPTQ `FORMAT.GPTQ`/`FORMAT.GPTQ_V2`/`FORMAT.MARLIN` and AWQ `FORMAT.GEMM`/`FORMAT.MARLIN`.
* ✨ GPTQ, AWQ, ParoQuant, QQQ, VPTQ, GGUF, FP8, EXL3, GPTAQ, and FOEM quantization support.
* ✨ All current GGUF quantized tensor types are supported end to end for CPU quantization and inference: every `Q*_0`/`Q*_1`, `Q2_K`-`Q8_K`, `IQ1_S`-`IQ4_XS`, `TQ1_0`/`TQ2_0`, `MXFP4`, `NVFP4`, `Q1_0`, `Q2_0`, and the `Q1_0_g128` sign-only variant. Install the bindings with `pip install gptqmodel[gguf]` (`gguf>=0.19.0`, `llama-cpp-python>=0.3.36`). CPU inference uses llama.cpp's native SIMD kernels (`BACKEND.GGUF_CPP_CPU`) with a per-shape autotuned Torch fallback. Native ggml dot kernels also quantize activations, so outputs can differ from weight-only Torch inference; set `GPTQMODEL_GGUF_CPP_NATIVE_MATMUL=0` to keep the Torch path. `Q8_1`/`Q8_K` always use Torch, including when native execution is forced, since upstream ggml exposes no matmul for them. CPU dequantization reconstructs weights in FP32 before casting to FP16/BF16. See the [GGUF compatibility notes](docs/prism-bonsai-gguf.md).
* 🚀 Routing-aware MoE quantization controls for extreme activation bias via `Moe.Routing` and/or `FailSafe`.
* 🚀 Data Parallelism for 80%+ quantization speed reduction with Multi-GPU.
* 🚀 Optimized for Python >= 3.13t (free threading) with lock-free threading.
* ✨ Linux, macOS, Windows platform support for CUDA (NVIDIA), NPU (Huawei Ascend), XPU (Intel), ROCm (AMD), MPS (Apple Silicon), CPU (Intel/AMD/Apple Silicon).
* ✨ `Dynamic` per-module mixed quantization control: each layer/module can have a unique quantization config or be excluded from quantization. 
* 🚀 Intel Torch 2.8 fused kernel support for XPU [`Arc` + `Datacenter Max`] and CPU [`avx`, `amx`].
* 🚀 Python 3.13.3t (free-threading, GIL disabled) support for multi-GPU accelerated quantization for MoE models and multi-core CPU boost for packing.
* ✨ Asymmetric `Sym=False` support. 
* ✨ `lm_head` module quant inference support for further VRAM reduction.
* 🚀 [Microsoft/BITBLAS](https://github.com/microsoft/BitBLAS) optimized tile based inference.
* 💯 CI unit-test coverage for all supported models and kernels including post-quantization quality regression.

## Organizations and projects using GPT-QModel 🌐

Selected public references where teams or companies explicitly mention GPT-QModel in documentation, integration notes, or quantized model usage. This is not an exhaustive customer list.

* <img src="https://cdn.simpleicons.org/huggingface/FFD21E" alt="Hugging Face logo" height="14"> Hugging Face
* <img src="https://cdn.simpleicons.org/intel/0071C5" alt="Intel logo" height="14"> Intel
* <img src="https://cdn.simpleicons.org/nvidia/76B900" alt="NVIDIA logo" height="14"> NVIDIA
* <img src="https://cdn.simpleicons.org/alibabacloud/FF6A00" alt="Alibaba Cloud logo" height="14"> Alibaba Cloud


## Quality: GPTQ 4-bit evaluation 🏆
🤗 [ModelCloud quantized Vortex models on HF](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2)

<img src=https://github.com/user-attachments/assets/c1b89394-f8f6-44e5-9949-bef15a124723 width="51%"> <img src=https://github.com/user-attachments/assets/23901236-10c5-4435-ac2f-06cf2e097f1e width="47%">

## Model Support 🤖

The table mirrors every explicit registration in `gptqmodel.models.auto.MODEL_MAP`, including text-model and backward-compatibility aliases. Qwen 3.5 registrations require Transformers 5.2.0 or newer.

<!-- model-types:start -->
| Model family | Registered Transformers `model_type` values |
|---|---|
| A.X-K2 | `axk2` |
| AfMoE / Trinity | `afmoe` |
| Agnes 3.0 Flash | `agnes` |
| Apertus 1 / 1.5 | `apertus`, `apertus1p5`, `apertus1p5_text` |
| Baichuan | `baichuan` |
| Bailing MoE / Hybrid (LING / RING) | `bailing_moe`, `bailing_hybrid` |
| Bloom | `bloom` |
| Brumby | `brumby` |
| ChatGLM | `chatglm` |
| CodeGen | `codegen` |
| Cohere 1 / 2 / 2 MoE / Compass (North Mini / Micro Vision) | `cohere`, `cohere2`, `cohere2_moe`, `cohere_compass` |
| DBRX / DBRX Converted | `dbrx`, `dbrx_converted` |
| DeciLM | `deci` |
| DeepSeek V2 / V3 / V3.2 / V4 / VL / VL2 / OCR2 | `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4`, `deepseek_vl`, `deepseek_vl_v2`, `deepseek_ocr2` |
| DiffusionGemma | `diffusion_gemma` |
| Dots1 | `dots1` |
| Dream | `dream` |
| ERNIE 4.5 / MoE / VL MoE | `ernie4_5`, `ernie4_5_moe`, `ernie4_5_moe_vl`, `ernie4_5_vl_moe` |
| EXAONE 3 / 4 / 4.5 | `exaone`, `exaone4`, `exaone4_5` |
| Falcon / Falcon H1 / Falcon Mamba | `falcon`, `falcon_h1`, `falcon_mamba` |
| Gemma 1-4 / 3n / Unified | `gemma`, `gemma2`, `gemma3`, `gemma3_text`, `gemma3n`, `gemma3n_text`, `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` |
| GLM / GLM4 / GLM4V / GLM5 / OCR / ASR | `glm`, `glm4`, `glm4_moe`, `glm4_moe_lite`, `glm4v`, `glm4v_moe`, `glm4v_moe_text`, `glm5_next`, `glm_moe_dsa`, `glm_ocr`, `glmasr` |
| GPT-2 | `gpt2` |
| GPT BigCode | `gpt_bigcode` |
| GPT-Neo / GPT-NeoX | `gpt_neo`, `gpt_neox` |
| GPT-OSS | `gpt_oss` |
| GPT-J | `gptj` |
| Granite / Granite MoE / Granite MoE Hybrid | `granite`, `granitemoe`, `granitemoehybrid` |
| GRIN-MoE | `grinmoe` |
| HRM | `hrm_text` |
| Hunyuan V1 / VL / OCR | `hunyuan_v1_dense`, `hunyuan_v1_moe`, `hunyuan_vl` |
| HY-V3 | `hy_v3` |
| Hymba | `hymba` |
| Inkling | `inkling_mm_model` |
| Instella | `instella` |
| Intern S1 / S2 Preview | `interns1`, `intern_s2_preview` |
| InternLM 1 / 2 / 2.5 | `internlm`, `internlm2` |
| InternVL 3 / InternVL Chat | `internvl`, `internvl_chat` |
| Jamba / Jamba2 | `jamba` |
| K2-Horizon (Dense / MoVA) |
| Kimi K2 / K2.5 | `kimi_k2`, `kimi_k25` |
| Klear | `klear` |
| Laguna | `laguna` |
| LFM2 / LFM2 MoE / LFM2-VL | `lfm2`, `lfm2_moe`, `lfm2_vl` |
| LLaDA2 MoE | `llada2_moe` |
| Llama 1-4 / TinyLlama / Nemotron Ultra | `llama`, `llama4`, `llama4_text` |
| Llama 3.2 VL | `mllama`, `mllama_text_model` |
| LLaVA-1.5 | `llava` |
| FastVLM / LLaVA-Qwen2 | `llava_qwen2` |
| LocateAnything | `locateanything` |
| LongCat Flash | `longcat_flash` |
| LongLLaMA | `longllama` |
| Mage-VL | `mage_vl` |
| Marin | `marin` |
| MiMo / MiMo V2 | `mimo`, `mimo_v2` |
| MiniCPM / MiniCPM3 / MiniCPM-O / MiniCPM-V | `minicpm`, `minicpm3`, `minicpmo`, `minicpmv`, `minicpmv4_6` |
| MiniMax M2 / M3-VL | `minimax`, `minimax_m2`, `minimax_m3_vl` |
| Mistral / Mistral3 / Ministral3 | `mistral`, `mistral3`, `ministral3` |
| Mixtral | `mixtral` |
| MobileLLM | `mobilellm` |
| MOSS | `moss` |
| MPT | `mpt` |
| Muse Glimmer | `muse_glimmer` |
| Nanbeige 4.2 | `nanbeige` |
| Nemotron NAS / H / H Puzzle / Omni / Labs Diffusion | `nemotron-nas`, `nemotron_h`, `nemotron_h_puzzle`, `nemotronh_nano_omni_reasoning_v3`, `nemotron_labs_diffusion` |
| OLMo 2 / 3 / OLMoE | `olmo2`, `olmo3`, `olmoe` |
| OpenVLA | `openvla` |
| OPT | `opt` |
| Ouro | `ouro` |
| Ovis 1.6 / 2 / 2.5 / 2.6 MoE / 2.6 Next | `ovis`, `ovis2`, `ovis2_5`, `ovis2_6_moe`, `ovis2_6_next` |
| PanGu-α | `gpt_pangu` |
| Phi 1-4 / Phi MoE | `phi`, `phi3`, `phi4mm`, `phimoe` |
| Qwen 1-4 / 3.5 / 3.6 / 3.8 / MoE / Next | `qwen`, `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen4_exp`, `qwen4_exp_text` |
| Qwen 2 / 2.5 / 3 VL / 3 VL MoE | `qwen2_vl`, `qwen2_vl_text`, `qwen2_5_vl`, `qwen2_5_vl_text`, `qwen3_vl`, `qwen3_vl_moe` |
| Qwen 2.5 / 3 Omni | `qwen2_5_omni`, `qwen3_omni_moe` |
| Qwen2-Audio | `qwen2_audio` |
| Qwen-Drive 1.0 | `qwen_drive` |
| RefinedWeb | `refinedWeb`, `refinedWebModel` |
| Seed-OSS | `seed_oss` |
| SmolLM3 | `smollm3` |
| SmolVLM | `smolvlm` |
| Solar Open / Open 2 | `solar_open`, `solar_open2` |
| Spark 2.5 | `spark2_5` |
| StableLM | `stablelm`, `stablelm_epoch` |
| StarCoder2 | `starcoder2` |
| TeleChat2 | `telechat` |
| Unlimited-OCR | `unlimited-ocr` |
| Voxtral | `voxtral` |
| [Xing4.0 (29B-A4B)](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | `xing4_0` |
| XVERSE | `xverse` |
| Yi | `yi` |
| Zamba / Zamba2 | `zamba`, `zamba2` |
| ZDTaichu 5.0 | `zdtaichu5_0` |
<!-- model-types:end -->

Qwen2.5-Omni supports quantizing its text output head with `lm_head=True`.
The head is `thinker.lm_head`; GPTQ defaults it to 8-bit, group size 32.
To use a 4-bit head, configure an explicit dynamic override:

```python
from gptqmodel import QuantizeConfig

quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    lm_head=True,
    dynamic={r"thinker\.lm_head$": {"bits": 4, "group_size": 32, "desc_act": False}},
)
```

This option quantizes the thinker's text head; the talker and audio decoder
retain their original weights. Tied output-head weights must be untied first.

## Platform and HW Support 🖥️

GPT-QModel is validated on Linux, macOS, and Windows 11:

| Platform | Device |  | Optimized Arch | Kernels |
|---|---|---|---|---|
| 🐧 Linux | NVIDIA GPU | ✅ | `Turing+` (`sm_75+`) | Machete, Marlin, Exllama V3 / EXL3, Exllama V2, AWQ GEMM/GEMV, ParoQuant CUDA/Triton, GGUF CUDA/Triton, QQQ, VPTQ, BitBLAS, Triton, BitsAndBytes, Torch |
| 🐧 Linux | AMD GPU | ✅ | `7900XT+`, `ROCm 6.2+` | Exllama V2, AWQ Triton GEMM, AWQ GEMV, QQQ, FP8 Torch, Torch (gfx950 fused GPTQ GEMV) |
| 🐧 Linux | Huawei Ascend NPU | ✅ | `Ascend 910B`, `torch-npu` / `CANN` | Native Torch kernels for GPTQ, AWQ, ParoQuant, GGUF, QQQ, and EXL3 |
| 🐧 Linux | Intel XPU | ✅ | `Arc`, `Datacenter Max` | TorchFused, TorchFusedAWQ, FP8 Torch, Torch |
| 🐧 Linux | Intel/AMD CPU | ✅ | `avx`, `amx` | TorchFused, TorchFusedAWQ, TorchAten int4, TorchInt8, GGUF C++, BitsAndBytes, VPTQ, Torch |
| 🍎 macOS | GPU (Metal) / CPU | ✅ | `Apple Silicon`, `M1+` | Torch, FP8 Torch, MLX via conversion |
| 🪟 Windows | GPU (NVIDIA) / CPU | ✅ | `NVIDIA` | Torch |

ROCm pip SDK installations that expose `_rocm_sdk_core` as `ROCM_HOME` need the
matching `rocm-sdk-devel` package and `rocm-sdk init` before building JIT kernels.
On gfx950, the Torch GPTQ backend uses a fused Triton GEMV for single-token
2/4/8-bit inference and falls back to the existing path for other shapes. Set
`GPTQ_TORCH_ROCM_GEMV=0` to disable the fused path.

`Marlin` and JIT CUDA kernels now support NVIDIA `Turing+` (`sm_75+`) GPUs.
Huawei Ascend NPU support uses native Torch kernels through `torch-npu` / `CANN`.

### Native MLX quantization on Apple silicon

Install the optional MLX dependency and quantize a dense model directly to an
MLX-LM checkpoint:

```bash
pip install 'gptqmodel[mlx]'
```

```python
from gptqmodel import GPTQModel

GPTQModel.quantize_mlx(
    "Qwen/Qwen3-0.6B",
    "./qwen3-mlx-gptq",
    method="gptq",  # or "awq"
    bits=4,
    group_size=64,
    num_samples=128,
)
```

GPTQ uses MLX for calibration and weight updates, with one fused Metal dispatch
per weight group. MLX currently performs the Cholesky factorization on its CPU
stream. AWQ uses MLX-LM's native scale and clip search. The output uses MLX-LM's
quantized model format and can be loaded with `mlx_lm.load`.
Pass an MLX token array as `calibration_data` to use your own calibration set.


## Install 💾

### PIP/UV 💿

```bash
# vLLM and SGLang publish their own Torch/Transformers pins. Install one engine
# in a dedicated environment aligned to that engine's release.
# Example optional module: pip install -v "gptqmodel[bitblas]"
pip install -v gptqmodel
uv pip install -v gptqmodel
```

The package depends on `ninja` for first-use JIT kernel compilation.

### Install from source 🛠️

```bash
# clone repo
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# python3-dev is required for some source installs
apt install python3-dev

# pip: install from source
# Example optional module: pip install -v ".[bitblas]"
pip install -v .
```

### Inference 🔮
Three-line API to use `GPT-QModel` for GPTQ model inference:

```py
from gptqmodel import GPTQModel

model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")
result = model.generate("Uncovering deep insights begins with")[0] # tokens
print(model.tokenizer.decode(result)) # string output
```

On Apple Silicon, install the optional MLX runtime with `pip install "gptqmodel[mlx]"`.
`GPTQModel.load(..., backend=BACKEND.AUTO)` selects MLX Metal for supported 4-bit
GPTQ and AWQ GEMM checkpoints when the model architecture is supported by MLX-LM.
The converter transfers compatible packed weights directly to MLX's quantized
matmul layout. Supported affine group sizes are 32, 64, and 128. Explicit backend
choices remain available.

```py
from gptqmodel import GPTQModel

model = GPTQModel.load("ModelCloud/Qwen2.5-0.5B-Instruct-gptqmodel-w4a16")
print(model.generate(prompt="The capital of France is", max_tokens=16))
```

To use models from [ModelScope](https://www.modelscope.cn/) instead of HuggingFace Hub, set an environment variable:
```shell
export GPTQMODEL_USE_MODELSCOPE=True
```

### FP32 accumulation toggle 🔢

Some AWQ and ParoQuant CUDA/Triton kernels support an fp32 accumulation mode to reduce numerical drift during fused quantized matmul. This setting defaults to `True` because accuracy is prioritized over speed.

```shell
# default behavior: higher accuracy, slightly lower speed on some kernels
export GPTQMODEL_FP32_ACCUM=1

# optional speed-first mode for some kernels
export GPTQMODEL_FP32_ACCUM=0
```

### JIT kernel cache and multi-process quantization ⚙️

JIT-compiled kernels are cached at `~/.cache/gptqmodel/torch_extensions` by default. Multiple processes on one host may safely share the cache: builds are serialized with a cross-process file lock that the OS releases automatically if a process dies.

Machete's pinned CUTLASS checkout and generated CUDA sources are kept in the
versioned user cache (`~/.cache/gptqmodel`, or `GPTQMODEL_CACHE_DIR`; when set,
`XDG_CACHE_HOME/gptqmodel` is used). Set `GPTQMODEL_CUTLASS_DIR` to use an
already-installed, read-only CUTLASS 4.7.1 checkout. `GPTQMODEL_OFFLINE=1`
disables downloads and requires that checkout or a verified cache hit already
exist. For environments where compilation is not allowed, point
`GPTQMODEL_MACHETE_PRECOMPILED_LIBRARY` at a compatible Machete shared library;
an invalid or missing explicit library is reported as an error and does not
fall back to JIT. The generated source cache can be populated ahead of time by
prewarming the extension:

```shell
python -c "from gptqmodel import extension; extension.load('machete')"
```

```shell
# optional: relocate the kernel cache (e.g. one cache per process)
export GPTQMODEL_TORCH_EXTENSIONS_DIR=/path/to/cache

# optional: max seconds to wait for another process's in-flight build before
# falling back to non-JIT paths. Default: 600 or 5x the kernel's compile
# baseline, whichever is larger.
export GPTQMODEL_TORCH_OPS_LOCK_TIMEOUT=600
```

Notes:
* This is a runtime toggle. It does not change model weights or saved checkpoints.
* It mainly affects some fused AWQ and ParoQuant CUDA/Triton kernels. Dense/dequantize fallback paths are mostly unaffected.
* `1` is recommended for regression testing and quality-sensitive evaluation. `0` may be useful when chasing a small latency win and the quality tradeoff is acceptable.

### OpenAI API compatible endpoint 🌐
```py
# load model using above inference guide first
model.serve(host="0.0.0.0",port="12345")
```

### Quantization 🔧
Example: quantizing an LLM with `GPT-QModel`:

```py
from datasets import load_dataset
from gptqmodel import GPTQConfig, GPTQModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = GPTQConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match GPU/VRAM specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)
```

#### Calibration grid workspaces

GPTQ/QQQ MSE shrink searches and AWQ clipping searches evaluate candidates in
chunks. Their temporary workspace targets can be adjusted per
`QuantizeConfig` via `MSEConfig` on the `mse` field (a bare float, e.g.
`mse=2.4`, still works and only sets the search exponent `norm`):

```py
from gptqmodel import GPTQConfig, MSEConfig

quant_config = GPTQConfig(
    bits=4,
    group_size=128,
    mse=MSEConfig(norm=2.4, grid_budget="64M", clip_grid_budget="64M"),
)
```

| `MSEConfig` field | Search | CPU default | GPU default |
|---|---|---:|---:|
| `norm` | MSE objective exponent (`0` disables the search) | — | — |
| `grid_budget` | GPTQ/QQQ MSE shrink grid | `16M` | `256M` |
| `clip_grid_budget` | AWQ clipping grid | `16M` | `256M` |

Budget values are byte counts (ints) or strings with a `K`/`M`/`G` postfix
(e.g. `"256M"`, `"1G"`).

Set either field to `0` to evaluate one candidate at a time; AWQ also uses
one-token product tiles. These are chunk-sizing targets, not limits on total
process memory: AWQ sizes its scratch buffers independently, and a single
candidate, full-token error buffer, or minimum product tile can exceed the
target. These settings affect calibration scratch only, not stored quantized
weights.

#### Checkpoint and Resume Long Quantization Jobs 💾

Checkpointing commits completed transformer layers so an interrupted
quantization job can reload the original model and continue without repeating
those layers. Use an explicit checkpoint path that survives process restarts:

```py
from gptqmodel import CheckpointConfig, CheckpointStopped, GPTQConfig, GPTQModel

quant_config = GPTQConfig(
    bits=4,
    group_size=128,
    offload_to_disk=True,
)
model = GPTQModel.load(model_id, quant_config)

try:
    model.quantize(
        calibration_dataset,
        batch_size=1,
        checkpoint=CheckpointConfig(
            path="checkpoints/Llama-3.2-1B-Instruct-gptqmodel-4bit",
            resume="auto",
            interval="layer:1",
            keep_last=2,
        ),
    )
except CheckpointStopped:
    # Run the script again with the same source, calibration, config, and path.
    raise SystemExit(75)

model.save(quant_path)
```

Ctrl+C and `SIGTERM` finish the current layer, publish a safe checkpoint, and
raise `CheckpointStopped`. A hard kill resumes from the last published layer;
the unfinished layer is repeated. Resume validation requires the same source
weights, calibration data, quantization settings, software versions, and device
topology. Current checkpoint support is limited to Llama and Qwen3 MoE model
types, requires `offload_to_disk=True`, and does not yet support dynamic
exclusions, rotation, or embedding/`lm_head` quantization.

See **[Quantization checkpointing and resume](checkpoint.md)** for resume
policies, supported methods, safe-stop behavior, storage requirements, recovery
procedures, and operational gotchas.

#### Other Quantization Formats 📦

`QuantizeConfig` remains the broad factory. The concrete config classes are now `GPTQConfig`, `AWQConfig`, `ParoConfig`, `QQQConfig`, `VPTQConfig`, `RTNConfig`, `GGUFConfig`, `FP8Config`, `BitsAndBytesConfig`, and `EXL3Config`.

`GPTQ`, `AWQ`, `ParoQuant`, `VPTQ`, and `EXL3` are calibration-based. `GGUF` and `FP8` are weight-only and should be quantized with `calibration=None`.

##### QQQ Example: Llama 3.2 1B Instruct

QQQ supports 4-bit symmetric weights with `group_size=-1` or `group_size=128`. Use `desc_act=False`: the packed QQQ runtime stores contiguous group scales and does not support GPTQ-style activation reordering. MSE scale search (`mse=2.4`) is the recommended quality recipe.

```py
import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QQQConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-QQQ-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(256))["text"]

quant_config = QQQConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    mse=2.4,
)

model = GPTQModel.load(model_id, quant_config, dtype=torch.float16)
model.quantize(
    calibration_dataset,
    batch_size=1,
    calibration_concat_size=2048,
)
model.save(quant_path)
```

Experimental W4A streams on NVIDIA GB10 (SM121) keep GPTQ INT4 weights and quantize the *activations*: E4M3 FP8 with a per-token scale, or E2M1 NVFP4 with E4M3 block scales. Selected Llama decoder operators pass encoded activations and scales across module and layer boundaries instead of decoding to BF16. See the **[W4A guide](docs/gptq-w4a-gb10.md)** for the policy reference, and the [accuracy appendix](docs/gptq-w4a-accuracy-notes.md) for measured quality.

##### Preprocessors 🧹

`preprocessors=[...]` adds optional module-weight preparation steps before quantization or repacking. They are available on `GPTQConfig`, `AWQConfig`, `ParoConfig`, `RTNConfig`, `GGUFConfig`, `FP8Config`, and `BitsAndBytesConfig`.

- `SmootherConfig`: apply weight smoothing before quantization.
- `AutoModuleDecoderConfig`: decode FP8/FP4 source modules to a dense `target_dtype` before downstream quantization or repacking.
- `TensorParallelPadderConfig`: opt-in tensor-parallel padding metadata for TP-aligned packing.

```py
import torch
from gptqmodel import GGUFConfig, GPTQConfig
from gptqmodel.quantization import (
    AutoModuleDecoderConfig,
    SmoothMAD,
    SmootherConfig,
    TensorParallelPadderConfig,
)

gptq_cfg = GPTQConfig(
    bits=4,
    group_size=128,
    preprocessors=[
        SmootherConfig(smooth=SmoothMAD(k=2.0)),
        AutoModuleDecoderConfig(target_dtype=torch.bfloat16),
        TensorParallelPadderConfig(),
    ],
)

gguf_cfg = GGUFConfig(
    bits=4,
    format="q_k_m",
    preprocessors=[
        AutoModuleDecoderConfig(target_dtype=torch.bfloat16),
        TensorParallelPadderConfig(),
    ],
)
```

##### GGUF Example: Llama 3.2 1B Instruct

```py
from gptqmodel import BACKEND, GGUFConfig, GPTQModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-GGUF-Q4_K_M"

qcfg = GGUFConfig(
    bits=4,
    format="q_k_m",
)

model = GPTQModel.load(model_id, qcfg)
model.quantize(calibration=None, backend=BACKEND.GGUF_TORCH)
model.save(quant_path)
```

##### FP8 Example: Llama 3.2 1B Instruct

```py
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import FP8Config

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-FP8-E4M3"

qcfg = FP8Config(
    format="float8_e4m3fn",  # or "float8_e5m2"
    bits=8,
    weight_scale_method="row",
)

model = GPTQModel.load(model_id, qcfg)
model.quantize(calibration=None, backend=BACKEND.GPTQ_TORCH)
model.save(quant_path)
```

##### Exllama V3 / EXL3 Example: Llama 3.2 1B Instruct

```py
from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import EXL3Config

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-EXL3"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

qcfg = EXL3Config(
    bits=4.0,        # target average bits-per-weight
    head_bits=6.0,   # optional higher bitrate for attention heads / sensitive tensors
    codebook="mcg",  # one of: mcg, mul1, 3inst
)

model = GPTQModel.load(model_id, qcfg)
model.quantize(calibration_dataset, batch_size=1, backend=BACKEND.EXL3_EXLLAMA_V3)
model.save(quant_path)
```

##### VPTQ Example: Llama 3.2 1B Instruct

VPTQ learns vector codebooks from calibration data with GPTQ-style error
compensation; Hessians are accumulated online, so no pre-generated Hessian
files are needed. See the **[VPTQ guide](docs/vptq.md)** for the full
configuration reference.

```py
from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import VPTQConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-VPTQ"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

qcfg = VPTQConfig(
    vector_lens=[-1, 8],         # 8-weight vectors, no outlier codebook
    num_centroids=[-1, 4096],    # 4096-entry main codebook
    num_res_centroids=[-1, 16],  # 16-entry residual codebook
)

model = GPTQModel.load(model_id, qcfg)
model.quantize(calibration_dataset, batch_size=1, backend=BACKEND.VPTQ_TORCH)
model.save(quant_path)
```

##### W4A Example: Llama 3.2 1B Instruct (GB10)

W4A quantizes activations while leaving the saved GPTQ checkpoint in its native
INT32-packed INT4 layout. It requires NVIDIA GB10 (SM121), a Llama decoder
model, `bits=4`, `group_size=128`, `sym=True`, `desc_act=False`,
`pack_dtype="int32"`, `lm_head=False`, and Triton.

```py
import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import QuantizeConfig

model_id = "unsloth/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-W4A"

qcfg = QuantizeConfig(
    bits=4, group_size=128, sym=True, desc_act=False,
    pack_dtype="int32", rotation="hadamard", offload_to_disk=False,
    activation={
        "mode": "w4a_nvfp4",
        "recipe": "least_squares_grid",
        "attention": {"mode": "w4afp8"},
    },
)

model = GPTQModel.load(model_id, qcfg, device="cuda", dtype=torch.bfloat16)
model.quantize(calibration, backend=BACKEND.GPTQ_W4A_NVFP4)
model.save(quant_path)

loaded = GPTQModel.load(quant_path, backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
```

`activation` accepts a bare mode string as shorthand for that stream
(`activation="w4afp8"` or `activation="w4a_nvfp4"`). The dict form adds
`recipe`, `attention`, `mlp`, and `global_scales`; see the
**[activation policy reference](docs/gptq-w4a-gb10.md#activation-policy-reference)**
for every field and its validation rules.

Two configurations are worth calling out. `activation="w4afp8"` is the highest
accuracy option and matches W4A16 within noise. The balanced NVFP4-MLP /
FP8-attention config above is the accepted W4A4 operating point; it costs about
2.3 percentage points against W4A16 on full GSM8K Platinum, which is the
measured price of the 4-bit activation grid on this model.

#### MoE Quantization 🧩

Some MoE (mixture of experts) models have extremely uneven/biased routing (distribution of tokens) to the `experts` causing some expert modules to receive close-to-zero activated tokens, thus failing to complete calibration-based quantization (GPTQ/AWQ).
To better quantize these heavily biased `MoE` routed modules, GPT-QModel exposes 3 controls:

* `Moe.Routing = ExpertsRoutingOverride`: Manually override the `num_experts_per_tok` used for model `routing` math, i.e., if a model only routes 4 experts per token out of 48 total experts, you can set this equal to 24 for 50% routing or 48 for 100% routing.
`ExpertsRoutingOverride` requires the model exposes `num_experts_per_tok` or equivalent configuration control.
* `Moe.Routing = ExpertsRoutingBypass`: Brute-force and bypass all `routing` math so all `experts` receive `all` activated tokens. This is akin to `ExpertsRoutingOverride.num_experts_per_tok` set to total number of experts. 
`ExpertsRoutingBypass` is enabled/tested for some models and, due to the lifecycle complexity, it needs to be validated for every model.
* `FailSafe`: This is `enabled` by `default` and is a naive weight-only quantization technique using simple (naive) quantization methods such as `nearest` with optional `smoothing`. 
There are various `FailSafeStrategy` options, along with `SmoothMethod` options, to complement this feature. `FailSafe` does not require `activations` but has higher quantization error loss than normally activated GPTQ/AWQ. It is fast and applicable for all MoE models.

`FailSafe` can be combined with `ExpertsRoutingOverride`. There is no single best way to quantize MoE, and we recommend users to test all three methods.

### Quantized Inference 🔍
```py
# test post-quant inference
model = GPTQModel.load(quant_path)
result = model.generate("Uncovering deep insights begins with")[0] # tokens
print(model.tokenizer.decode(result)) # string output
```

### EoRA Accuracy Recovery: Enhanced Post-Quant Error Recovery via Lora 🎯

GPT-QModel supports EoRA, a LoRA method developed by Nvidia that can further improve the accuracy of the quantized model.
```py
# EoRa is currently only validated for GPTQ
# higher rank improves accuracy at the cost of VRAM usage
# suggestion: test rank 64 and 32 before 128 or 256 as latter may overfit while increasing memory usage
eora = Lora(
  # for eora generation, path is adapter save path; for load, it is loading path
  path=f"{quant_path}/eora_rank32", 
  rank=32,
)

# provide a previously GPTQ-quantized model path
GPTQModel.adapter.generate(
  adapter=eora,
  model_id_or_path=model_id,
  quantized_model_id_or_path=quant_path,
  calibration_dataset=calibration_dataset,
  calibration_dataset_concat_size=0,
)

# post-eora inference
model = GPTQModel.load(
  model_id_or_path=quant_path,
  adapter=eora
)

tokens = model.generate("Capital of France is")[0]
result = model.tokenizer.decode(tokens)

print(f"Result: {result}")
# For more details on EoRA, please see docs/eora/
# Please use the benchmark tools in later part of this README to evaluate EoRA effectiveness
```

### Remote LoRA Adapters 🔒

Adapter paths stored in `quantize_config.json` are confined to the checkpoint's own storage (including same-repository Hugging Face cache blobs); metadata cannot authorize Hub downloads. Loading an adapter hosted on the Hub requires an explicit opt-in with an immutable commit pin and a SHA-256 digest of `adapter_model.safetensors`:

```py
model = GPTQModel.load(
  model_id_or_path=quant_path,
  adapter=Lora(path="org/adapter-repo", rank=32),
  allow_remote_adapter=True,
  adapter_revision="<40-char Hub commit sha>",
  adapter_sha256="<sha256 of adapter_model.safetensors>",
)
```

Manifest adapter directories must be checkpoint-relative and cannot contain directory symlinks.
Adapter-file symlinks are rejected except for the standard Hugging Face cache links into the
same repository's `blobs` directory. This confinement also applies when using
`QuantizeConfig.from_pretrained()` and initializing its adapter directly.
Explicit local `adapter=` arguments remain available for adapters stored elsewhere.
For pinned Hugging Face blob URLs, the config and weights are loaded from the URL's
containing folder at the supplied revision.

### PEFT LoRA Training and Adapter Loading 🤗

GPT-QModel GPTQ/AWQ checkpoints can be fine-tuned with PEFT LoRA and loaded through `PeftModel`. Use a current PEFT release with `optimum>=2.3.0`; older combinations may warn that the quantized layer type is unsupported. For training, select the method-specific trainable backend (`gptq_torch` for GPTQ, `torch_awq` for AWQ); default inference kernels do not propagate gradients.

PEFT cannot merge LoRA deltas into packed GPTQ/AWQ weights. Keep the adapter separate for quantized inference, or merge it into the original full-precision model and then quantize the merged model. See [docs/peft.md](docs/peft.md) for tested training, loading, and merge examples.

### How to Add Support for a New Model 🛠️

Read the [`gptqmodel/models/llama.py`](https://github.com/ModelCloud/GPTQModel/blob/5627f5ffeb3f19b1a2a97e3b6de6fbe668b0dc42/gptqmodel/models/llama.py) code which explains in detail via comments how the model support is defined. Use it as a guide for PRs to add new models. Most models follow the same pattern.

#### Shared-input metadata (`:in=<tag>`) 🔗

Modules that consume the *same* activation tensor (e.g. `q_proj`/`k_proj`/`v_proj` after `input_layernorm`) produce identical GPTQ Hessians (`H = XᵀX`), so the Hessian only needs to be collected once per group. `BaseQModel.shared_input_plan(model_config, quantize_config)` derives these groups from `module_tree`:

- Default: every quantizable leaf is its own singleton group. Subset digits (`:0`) describe execution/quantization order, not tensor identity, so they are never used to infer sharing.
- Opt in with `:in=<tag>`: sibling leaves (same parent) with the same tag share an input, e.g. `"q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x"` or `"gate_proj:0:in=x", "up_proj:0:in=x"`. Tags are scoped per parent. Different tags never share (MLA: `"q_b_proj:1:in=q_a", "kv_b_proj:1:in=kv_a"` read different latents).
- A leaf repeated across `module_tree` variants must carry identical flags; conflicts raise at plan time.
- `:!` / `:?` leaves and `:in=` tags never change the emitted subset blocks or quantization order.
- Runtime dedup is per subset block: the looper captures one block at a time and elects the first group member in that block as leader; the other members in the *same* block skip Hessian capture and adopt a private copy of the leader's `H`. A tag whose members sit in different blocks (e.g. `in_proj_qkv:0` / `in_proj_z:1`) is still validated by the probe but deduplicates nothing (`SharedInputGroup.dedup_followers`, `SharedInputPlan.dedup_count` reflect this).
- Tags are inert until the definition lists the `model_type` in its own `shared_input_verified_model_types` (not inherited). Unlisted model types (including Llama-clone subclasses that inherit `module_tree`) get singleton plans and never skip capture; `tests/module_tree/test_shared_input_cpu_forward.py` enforces that every listed type has a real-forward case.

Only add `:in=` tags after verifying them against a real (tiny, CPU) model with `gptqmodel.models.shared_input.probe_shared_inputs(layer, plan, forward)`; it hooks every planned module, runs `forward`, and reports groups whose inputs differ (`mismatches`), identical inputs that were not declared (`undeclared`), planned modules that do not exist (`missing_modules`) and groups that never ran (`unverified`, e.g. un-routed experts). `report.ok` is strict (`fully_verified`); use `has_errors` when un-routed experts are expected. See `tests/module_tree/test_shared_input*.py` for the covered definitions.

### Pair with Evaluation for post-quantization LLM Benchmarks 📊

GPT-QModel evaluation is integrated into [Evalution](https://github.com/ModelCloud/Evalution), a benchmarking toolkit with 153 benchmark suites.
Use Evalution to measure post-quantization accuracy recovery instead of relying only on narrow regression language-model metrics.

```
# install Evalution
pip install Evalution
```

Example: running `gsm8k_platinum` through Evalution's native GPT-QModel engine.

```py
import evalution as eval

run = (
    eval.GPTQModel(
        backend="marlin",
        device="cuda:0",
    )
    .model(eval.Model(path="ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"))
    .run(eval.benchmarks.gsm8k_platinum(apply_chat_template=True, batch_size=16))
)

print(run.to_dict()["tests"][0]["metrics"])

```
### Dynamic Quantization (Per Module QuantizeConfig Override) ⚙️

`QuantizeConfig.dynamic` is a dynamic control that allows specific matching `modules` to be skipped for quantization (negative matching)
or have a unique `[bits, group_size, sym, desc_act, mse, pack_dtype]` property override per matching `module` vs base `QuantizeConfig` (positive match with override). 

Sample `QuantizeConfig.dynamic` usage:

```py
dynamic = { 
    # `.*\.` matches the layers_node prefix 
    # layer index starts at 0 
    
    # positive match: layer 19, gate module 
    r"+:.*\.18\..*gate.*": {"bits": 4, "group_size": 32},  
    
    # positive match: layer 20, gate module (prefix defaults to positive if missing)
    r".*\.19\..*gate.*": {"bits": 8, "group_size": 64},  
    
    # negative match: skip layer 21, gate module
    r"-:.*\.20\..*gate.*": {}, 
    
    # negative match: skip all down modules for all layers.
    r"-:.*down.*": {},  
 } 

```

#### Fused Modules in Inference Engines (vLLM / SGLang) 🧩

Inference engines merge modules before they resolve the quantization config, and each merged module carries one precision and one
quantization config:

* every routed expert of one decoder layer is merged into one module (`<prefix>.experts`),
* `gate_proj` + `up_proj` are merged into `gate_up_proj`, and `q_proj` + `k_proj` + `v_proj` into `qkv_proj`.

A rule that matches just part of a merged module (for example
`-:.*down.*`, which matches `...experts.<id>.down_proj` but not the merged `...experts` module) therefore produces a checkpoint that
loads in GPT-QModel but fails in vLLM/SGLang, because those engines mix float and quantized shards of the same merged module. GPT-QModel
detects this before quantization starts and warns, naming the offending rules and a pattern that covers both. A rule must match the merged
module **and** its shards: routed experts live below their merged module, so one pattern on `...mlp.experts` covers everything, while
`gate_up_proj`/`qkv_proj` are siblings of their shards and need an alternation (for example `(gate_proj|up_proj|gate_up_proj)`).

```py
dynamic = {
    # skip the whole merged routed-expert module of layer 20
    r"-:.*\.20\.mlp\.experts(\.|$)": {},

    # give the whole merged routed-expert module of layer 18 one config
    r"+:.*\.18\.mlp\.experts(\.|$)": {"bits": 4, "group_size": 32},
}
```

Experts that never receive calibration traffic are a special case: with `fallback` disabled (`fallback=None`) GPT-QModel skips them and
they keep their source dtype. Those engines load a merged expert module with a single precision, so keep `fallback` enabled (the default)
when the checkpoint must load in vLLM/SGLang.

### Group Aware Reordering (GAR) 🔄

Group Aware Reordering (GAR) is an enhanced activation reordering scheme developed by Intel to improve the accuracy of quantized models without incurring additional inference overhead. Unlike traditional activation reordering, GAR restricts permutations to within individual groups or rearrangements of entire groups. This ensures each group's associated scales and zero-points remain efficiently accessible during inference, thereby avoiding any inference-time overhead.

How to enable GAR:

Set the `act_group_aware` parameter to `True` and disable the default activation reordering by setting `desc_act` to `False` in your `QuantizeConfig`. For example:

```python
quant_config = QuantizeConfig(bits=4, group_size=128, act_group_aware=True)
```


### Experimental Features 🧪

#### Norm tuning

After quantization, `requantize()` can tune the parameters of normalization
modules tagged by the model definition while leaving packed quantized tensors
unchanged. Pass an unquantized teacher model (or its Hugging Face path) with the
`norm` argument:

```py
from gptqmodel import BACKEND, GPTQModel, NormConfig

model = GPTQModel.load("path/to/saved-quantized-model", backend=BACKEND.TORCH)
model.requantize(
    calibration=calibration,
    norm=NormConfig(teacher="Qwen/Qwen3-0.6B", steps=400),
)
```

`NormConfig.types` selects which normalization families are tuned and defaults
to `(NormType.RMSNORM,)`. Pass
`NormType.LAYERNORM` (or both) to also tune LayerNorm gains and biases:

```py
from gptqmodel import BACKEND, GPTQModel, NormConfig, NormType

model.requantize(
    calibration=calibration,
    norm=NormConfig(teacher="teacher-model", types=(NormType.RMSNORM, NormType.LAYERNORM)),
)
```

Both normalization styles are covered by the underlying method: the
normalization parameters are distilled against the teacher's next-token
distribution. Gain parameterizations are detected per module, so
Gemma-style `1 + weight` gains are handled without extra configuration.

This feature requires a backend with autograd support through the quantized
linears, such as `BACKEND.TORCH`. After fresh `quantize()`, save and reload
before tuning: the live quantization model may still contain offloaded/meta
or inference tensors, or checkpoint-format GPTQ-v1 zero points. These states
are rejected before tuning rather than trained with incorrectly decoded weights.

For GPTQ-v1 embedding requantization, use two separate passes:

```py
from gptqmodel import QuantizeEmbed

model.requantize(calibration=calibration, embed_quant_mode=QuantizeEmbed.OUTPUT)
model.save("embedding-requantized")
model = GPTQModel.load("embedding-requantized", backend=BACKEND.TORCH)
model.requantize(calibration=calibration, norm=NormConfig(teacher="teacher-model"))
model.save("norm-tuned")
```

A combined embedding/Norm call remains supported for training-capable GPTQ-v2
models.

Use `NormConfig` with the `norm=` keyword; tuning statistics are returned under
the `"norm"` result key. Model definitions mark eligible modules with `:norm`.

#### Using GPTAQ (Experimental, not MoE compatible, and results may not be better than original) ⚗️

Enable GPTAQ quantization by setting `gptaq = GPTAQConfig(...)`.
```py
# GPTAQ is experimental, is not MoE compatible, and requires 2-4x more VRAM.
# It may not improve quality relative to GPTQ; evaluate it for the target model and workload.
# If one GPU OOMs, set CUDA_VISIBLE_DEVICES=0,1; GPTQModel will use the second GPU automatically.
quant_config = QuantizeConfig(bits=4, group_size=128, gptaq=GPTAQConfig(alpha=0.25, device="auto"))
```

#### Using FOEM 🧮

FOEM (First-order error matters) adds first-order error compensation for GPTQ-style quantization. Enable FOEM by setting `foem = FOEMConfig(...)`.
```py
# FOEM default hyperparameters are alpha=0.0 and beta=0.2
quant_config = QuantizeConfig(bits=4, group_size=128, foem=FOEMConfig(alpha=0.0, beta=0.2, device="auto"))
```
### Migrating from AutoGPTQ and AutoAWQ 🔄

GPT-QModel supports GPTQ and AWQ workflows that integrate with HF Transformers, Optimum, and PEFT. Existing inference integrations can generally be retained; verify configuration compatibility when migrating.

For model quantization, there are some config changes for AutoAWQ:

* AutoAWQ: `version` property is now `format`. `zero_point` is now `sym` (Symmetric Quantization): `sym = True` is equivalent to `zero_point = False`

Models quantized by GPT-QModel are inference compatible with HF Transformers (minus `dynamic`), vLLM, and SGLang. 

## Attributions 📚

* GPTQ: IST-DASLab, main-author: Elias Frantar, arXiv:2210.17323
* AWQ: main-authors: Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song
* ParoQuant: Z-Lab, main-authors: Yesheng Liang, Haisheng Chen, Song Han, and Zhijian Liu. [Official implementation](https://github.com/z-lab/paroquant), [Paper](https://openreview.net/forum?id=1USeVjsKau)
* EoRA: Nvidia, main-author: Shih-Yang Liu, arXiv preprint arXiv:2410.21271.
* GAR: Intel, main-author: T Gafni, A Karnieli, Y Hanani, [Paper](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Gafni_Dual_Precision_Quantization_for_Efficient_and_Accurate_Deep_Neural_Networks_CVPRW_2025_paper.html)
* GPTAQ: Yale Intelligent Computing Lab, main-author: Yuhang Li, arXiv:2504.02692.
* Fast Hadamard Transform: [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform), by Tri Dao, vendored into `gptqmodel_ext/hadamard/` under the BSD-3-Clause license. A copy of the license is included in `gptqmodel_ext/hadamard/LICENSE`.
* Swordfish Kernel: Blackwell (`>= sm100`) GPTQ/AWQ kernel from [AlpinDale](https://x.com/AlpinDale). [Paper](https://blog.alpindale.net/posts/swordfish/)
* QQQ: Meituan, main-author Ying Zhang, arXiv:2406.09904
* VPTQ: Microsoft, main-authors Yifei Liu et al., arXiv:2409.17066. [Official implementation](https://github.com/microsoft/VPTQ)
* Norm Tweaking: Liang Li, Qingyuan Li, Bo Zhang, and Xiangxiang Chu, [paper](https://arxiv.org/abs/2309.02784)
* FOEM: Zheng, Xingyu and Qin, Haotong and Li, Yuye and Chu, Haoran and Wang, Jiakai and Guo, Jinyang and Magno, Michele and Liu, Xianglong [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40123)

## Citations 📖

```bibtex
# GPT-QModel
@misc{qubitium2024gptqmodel,
  author = {ModelCloud.ai and qubitium@modelcloud.ai},
  title = {GPT-QModel},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/modelcloud/gptqmodel}},
  note = {Contact: qubitium@modelcloud.ai},
  year = {2024},
}

# GPTQ
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
  
}

# AWQ
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

# ParoQuant
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}

# GGUF / llama.cpp
@misc{ggerganov2023gguf,
  author = {Georgi Gerganov and ggml-org contributors},
  title = {llama.cpp and the GGUF model format},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ggml-org/llama.cpp}},
  note = {Canonical GGUF implementation and format reference; see also \url{https://github.com/ggml-org/llama.cpp/wiki/dev-notes}},
  year = {2023}
}

# EoRA
@article{liu2024eora,
  title={EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation},
  author={Liu, Shih-Yang and Yang, Huck and Wang, Chien-Yi and Fung, Nai Chit and Yin, Hongxu and Sakr, Charbel and Muralidharan, Saurav and Cheng, Kwang-Ting and Kautz, Jan and Wang, Yu-Chiang Frank and others},
  journal={arXiv preprint arXiv:2410.21271},
  year={2024}
}

# GPTAQ
@article{li2025gptaq,
  title={GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration}, 
  author={Yuhang Li and Ruokai Yin and Donghyun Lee and Shiting Xiao and Priyadarshini Panda},
  journal={arXiv preprint arXiv:2504.02692},
  year={2025}
}

# FOEM
@inproceedings{zheng2026first,
  title={First-order error matters: Accurate compensation for quantized large language models},
  author={Zheng, Xingyu and Qin, Haotong and Li, Yuye and Chu, Haoran and Wang, Jiakai and Guo, Jinyang and Magno, Michele and Liu, Xianglong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={34},
  pages={28883--28891},
  year={2026}
}

# QQQ 
@article{zhang2024qqq,
      title={QQQ: Quality Quattuor-Bit Quantization for Large Language Models}, 
      author={Ying Zhang and Peng Zhang and Mincong Huang and Jingyang Xiang and Yujie Wang and Chao Wang and Yineng Zhang and Lei Yu and Chuan Liu and Wei Lin},
      journal={arXiv preprint arXiv:2406.09904},
      year={2024}
}

# VPTQ
@inproceedings{liu2024vptq,
  title={{VPTQ}: Extreme Low-bit Vector Post-Training Quantization for Large Language Models},
  author={Liu, Yifei and Wen, Jicheng and Wang, Yang and Ye, Shengyu and Zhang, Li Lyna and Cao, Ting and Li, Cheng and Yang, Mao},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}

# Norm Tweaking
@misc{li2023normtweakinghighperformancelowbit,
  title={Norm Tweaking: High-performance Low-bit Quantization of Large Language Models},
  author={Liang Li and Qingyuan Li and Bo Zhang and Xiangxiang Chu},
  year={2023},
  eprint={2309.02784},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2309.02784}
}

# ExLlama V3 / EXL3
@misc{turboderp2026exllamav3,
  author = {turboderp and exllamav3 contributors},
  title = {ExLlamaV3 and the EXL3 quantization format},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/turboderp-org/exllamav3}},
  note = {Project repository and EXL3 format documentation: \url{https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md}},
  year = {2026}
}

# Group Aware Reordering (GAR)
@article{gar,
  title={Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference, CVPRW 2025.},
  author={T. Gafni, A. Karnieli, Y. Hanani},
  journal={arXiv preprint arXiv:2505.14638},
  year={2025}
}

# Marlin Kernel
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}

# Swordfish Kernel
@misc{alpindale2026swordfish,
  author = {AlpinDale},
  title = {Swordfish: A Weight-Quantized {GEMM} Family for {NVIDIA} Blackwell},
  howpublished = {\url{https://blog.alpindale.net/posts/swordfish/}},
  year = {2026}
}

# Machete Kernel
@misc{vllm2024machete,
  author = {vLLM contributors},
  title = {Machete: Hopper-optimized mixed-precision {GEMM} kernels},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vllm-project/vllm/tree/main/csrc/quantization/machete}},
  note = {CUTLASS-based mixed-precision kernel implementation},
  year = {2024}
}

```

## Operational notes 🗒️

For MoE quantization across multiple GPUs, see the [multi-GPU quantization guide](docs/multi_gpu_quantization.md). Setting `CUDA_VISIBLE_DEVICES` exposes GPUs to the process; MoE device placement and free-threaded parallel quantization also need to be configured.

### Limit log level 🔇

`GPT-QModel` uses a shared `LogBar` logger. Set the level once near process startup:

```python
from logbar import LogBar

LogBar.shared().setLevel("WARNING")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Apply Triton nogil patch early in multi-package scripts 🩹

If your script imports multiple Triton users (for example `gptqmodel`, `vllm`, and `sglang`), apply the patch at the very top before other Triton-related imports:

```python
from gptqmodel import TritonPatch

# Mitigate Triton kernel-cache crashes under nogil/free-threading Python 3.13+, where the cache storage is not thread-safe.
TritonPatch.apply()
```

## License 📜

GPT-QModel is licensed under the Apache-2.0 license. The optional Swordfish kernel sources vendored under `gptqmodel_ext/swordfish/` are licensed under the AGPL-3.0-or-later license and are only compiled/linked at runtime via JIT. A copy of the Swordfish license is included in `gptqmodel_ext/swordfish/licenses/LICENSE` and `licenses/SWORDFISH`.
