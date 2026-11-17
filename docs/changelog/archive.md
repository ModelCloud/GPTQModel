# Changelog archive

Release highlights for 7.7.0 and earlier are kept here so the project README can focus on the current platform and recent releases. Full release artifacts remain available on the [GitHub releases page](https://github.com/ModelCloud/GPTQModel/releases).

## 2026 platform releases

* **7.7.0**:
  * **CPU inference:** Added the shared AVX512-BF16 `int4_qzero_group_scaled_cpu` kernel for GPTQ/AWQ group sizes 16/32/64/128/256/512 and prioritized it ahead of ATen when supported. Added packed/low-precision CPU decode paths for QQQ, VPTQ, EXL3, FP8, NVFP4, and BitsAndBytes, plus oneDNN BF16x3 accumulation and shape-aware BitsAndBytes INT8 dispatch. GPTQ/AWQ measured up to 2.02x faster than ATen; BitsAndBytes INT8 reached 1.28x at M=64.
  * **GGUF:** Added fused CPU Q4_0/Q8_0/Q4_K/Q5_K/Q6_K forward with FP32 accumulation under oneDNN BF16x3, fused bias, per-shape autotuning, and no retained dense weights. K-quant fused paths measured up to 1.92x faster than dense at M=7.
  * **Quantized storage:** Removed persistent dense, dequantized, expanded, and INT4-to-INT8 inference caches across GPTQ/AWQ, QQQ, EXL3, VPTQ, FP8, FP4, GGUF, BitsAndBytes, and INT8. Only packed weights, scales, zero points, and metadata persist; temporary dequantization uses bounded tiles.
  * **Quantization quality:** Added token-normalized GPTQ loss reporting, on-device embedding diagonal flooring and VPTQ Hessian damping, and QQQ vLLM compatibility/CUDA saturation fixes.
  * **Models and runtimes:** Added LLaVA quantization support, fused VPTQ CUDA inference, and hardened remote LoRA adapter loading.
  * **Docs and tooling:** Added the GPTQ FSDP/PEFT LoRA example, multi-GPU MoE quantization guide, and independent Torch-oracle accuracy and CPU speed benchmarks.
  * **Dependencies:** Refreshed runtime and optional dependency floors to current stable releases, including PyTorch 2.14.1, Transformers 5.20.2, Datasets 5.1.0, and Optimum 2.3.0. vLLM/SGLang extras retain their upstream framework pins and should be installed in dedicated environments.

* **7.6.0**:
  * **Model support:** Added RecursiveSI Flint, Jamba/Jamba2, SmolVLM, Granite MoE, InternVL, EXAONE 4.5, Qwen3-VL MoE, Qwen4-Exp text, OLMoE, ZDTaichu 5.0, Agnes 3.0 Flash, and Xing4.0 quantization support, plus MiMo V2.6 Flash-RL and Pro-RL checkpoint support.
  * **Apple silicon:** Added native MLX GPTQ/AWQ quantization and an EXL3 weight-quantization pipeline. Expanded Metal quantization and packed inference kernels for Paro, GGUF, FP8, QQQ, and BitsAndBytes, with GPTAQ/FOEM/RTN acceleration, broader GPTQ/AWQ bit-width and group-size support, Marlin/BitBLAS checkpoint loading, and FP16/BF16 output preservation.
  * **GPTQ performance:** Added fused CUDA block updates for 2–8-bit quantization, faster Hessian collection and factor reuse, lower peak Hessian memory, and faster Cholesky damping recovery.
  * **Experimental W4A:** Added Llama decoder activation streams using FP8 on NVIDIA SM89+ or NVFP4 on GB10 (SM121), while preserving native packed GPTQ INT4 checkpoints. See the [W4A guide](../gptq-w4a-gb10.md) for supported configurations and measured quality.
  * **Source formats:** Added automatic decoding of native FP8/NVFP4 and OCP MXFP4/MXFP8 source checkpoints for requantization.
  * **Runtime and integration:** Added stable [`hf_` kernel selection, packing, and repacking APIs](../hf_packing_api.md), improved AWQ Triton scheduling, and added opt-in [Marlin scratch-buffer reuse](../marlin_scratch.md). Improved SGLang format validation, dtype handling, and generation parameter compatibility.
  * **Reliability:** Fixed model-tree isolation and dynamic-rule caching/serialization, offloaded tensor device/dtype restoration, fused DeepSeek expert loading, Qwen3.5-MoE multimodal calibration, Gemma4 calibration KV memory retention, Paro positional replay, GLM NoPE configuration handling, and GPTQ damping recovery after inverse-factor failures.

* 09/15/2026 [7.5.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.5.0): ✨ Added K2 Horizon, Qwen-Drive, Nanbeige, and Diffusion Gemma quantization support. Improved GPTQ/AWQ/Paro reliability with device-aware kernel selection, safer CUDA dispatch and empty-input handling, partial packing-word support, correct calibration masks, model-specific Paro layer replay, and hardened Machete runtime caching. Loading and saving now validate GPTQ `qweight` and offload metadata, while InternVL image preprocessing keeps `torchvision` optional. Quantization finalization is faster and startup version reporting is more accurate.
* 09/07/2026 [7.4.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.4.0): 🎉 Added resumable quantization checkpoints, shared-input Hessian deduplication, `lm_head` and embedding requantization, and updated native GGUF support. Added GLM-5.3-Flash, Apertus 1.5, and XHToken `ouro` / `spark2_5` quantization support, plus quantization, JIT cache, and Triton compatibility fixes.
* 08/31/2026 [7.3.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.6): ✨ Added HunyuanOCR, NVIDIA LocateAnything-3B, and Qwen3.8-Flash-Next quantization support. Added tile-misaligned GPTQ/AWQ Marlin support, reduced QQQ packing memory, and improved JIT extension cache reuse.
* 08/25/2026 [7.3.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.5): ✨ Added `lm_head` and embedding quantization lifecycle support; Unlimited-OCR, DeepSeek V3.2, Mage-VL, Muse Glimmer, OLMo 3, and SmolLM3 model support.
* 08/19/2026 [7.3.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.4): 🚀🔥⚡ Added the `Swordfish` Blackwell GPTQ/AWQ kernel, planar GPTQ checkpoint formats, native MPS quantization, and Cohere Compass / A.X-K2 model support, with quantization performance and reliability improvements.
* **07/25/2026 [7.3.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.2)** — Added quantized embedding inference and Solar Open, Solar Open 2, Intern S2 Preview, Inkling, and Poolside Laguna S 2.1 model support.
* **07/20/2026 [7.3.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.3.1)** — Added DeepSeek VL/VL2/OCR2 and Nemotron H Puzzle model support, Windows ExllamaV2 compatibility, and ModelOpt NVFP4 dequantization.
* **07/03/2026 [7.2.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.2.0)** — Added LFM2/LFM2-VL, MiniMax M3 VL, Cohere2 MoE, Gemma4 Unified, and text-only multimodal model definitions.
* **06/08/2026 [7.1.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.1.0)** — Added Laguna, ERNIE 4.5 VL MoE, Ling 2.6 Flash, Nemotron 3 Nano Omni, GLM4V MoE, Zamba/Zamba2, MiniCPM-V 4.6, DeepSeek V4, MiMo V2, Ovis 2.5/2.6, Intern S1, Nemotron Labs Diffusion, and Hunyuan V1 model support.
* **04/28/2026 [7.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v7.0.0)** — Added Huawei Ascend NPU support through native torch kernels for GPTQ, AWQ, Paro, GGUF, QQQ, and EXL3. CUDA kernels are now JIT-compiled, reducing wheel size and building only the kernels in use; Marlin supports NVIDIA Turing+ GPUs, with new GLM 5/5.1, InternVL Chat, Gemma3n, GLM-OCR, GLM-ASR, and Falcon Mamba model support.
* **04/02/2026 [6.0.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v6.0.3)** — Added Paro, GGUF, FP8, EXL3, and FOEM. Added PrismML/Bonsai 1bit model quantization (inference only), faster Paro/AWQ kernels, Paro optimization scope control, and Gemma4, MiniCPM-O, MiniCPM-V, and GLM4 MoE Lite support.
* **03/19/2026 [5.8.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.8.0)** — Added Transformers 5.3.0 support with Defuser auto-defusing, Qwen 3.5 support, fast HF CPU kernels for GPTQ/AWQ, and experimental GPTQ INT8 CPU kernels.

## 2026 and 2025

* **5.7.0** — Added `MoE.Routing` controls, unified AWQ symmetry configuration, and Qwen3/Exaone compatibility fixes.
* **5.6.2 / 5.6.0** — Improved install and multi-architecture compatibility; added AMX, AVX2, and AVX512 CPU kernels and broader Transformers, PEFT, and Optimum support.
* **5.4.0 / 5.2.0** — Added Intel CPU/XPU AWQ kernels, AWQ `torch.compile` support, balanced VRAM strategy, and more MoE support.
* **5.0.0** — Introduced data-parallel MoE quantization, disk offload by default, accelerated packing, and production AWQ support.
* **4.2.5 / 4.2.0 / 4.1.0 / 4.0.0** — Added `act_group_aware`, FailSafe controls, GAR, Python free-threading, and support for Qwen3, Llama 4, GPT-OSS, Gemma3, and other model families.
* **3.0.0 / 2.2.0 / 2.1.0 / 2.0.0** — Added GPTAQ experiments, QQQ, staged quantization internals, GSM8K Platinum and MMLU-Pro evaluation, and expanded multimodal/model support.
* **1.9.0 through 1.5.0** — Improved tokenizer and `lm_head` handling, flexible packing, buffered forward, compilation, MLX support, AMD ROCm, OpenAI-compatible serving, and multimodal quantization.

## 2024

* **1.4.5 through 1.0.0** — Added Windows and Apple Silicon support, visual-language model calibration, EvalPlus integration, dynamic quantization controls, Intel XPU support, new loader/save APIs, and PyPI distribution improvements.
* **0.9.11 through 0.9.0** — Added per-module dynamic quantization, Marlin and BitBLAS improvements, Llama 3.1 and Gemma 2 support, vLLM/SGLang runtime integration, AutoRound export, ExLlama compatibility, and faster batched calibration.

## Development notes

The pre-5.0 development notes documented the transition to data-parallel MoE quantization, Python 3.13 free-threading, disk offload, AWQ/Marlin support, and the model-definition refactor that enabled future quantization formats.
