<p align=center>
<div align=center>
<img src="https://github.com/user-attachments/assets/ab70eb1e-06e7-4dc9-83e5-bd562e1a78b2" width=500>
</div>
<h1 align="center">GPT-QModel</h1>
</p>
<p align="center">LLM model quantization (compression) toolkit with hw acceleration support for NVIDIA CUDA, AMD ROCm, Intel XPU, and Intel/AMD/Apple CPUs via HF, vLLM, and SGLang.</p>
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

## Latest News
* 04/23/2026 6.1.0-dev `main`: ✨ Added `gemma3n`、`GLM-OCR`、`GLM-ASR` and `falcon_mamba` model support.
* 04/16/2026 [6.1.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v6.1.0): 🚀🔥⚡ CUDA kernels are now fully JIT-compiled, shrinking the wheel by about 300x and building only what you use; Marlin now supports NVIDIA `Turing+` GPUs, Machete kernel validation now covers supported GPUs, `GLM 5/5.1` joins the lineup, and LazyTurtle plus AWQ / multi-GPU MoE fixes make large-model quantization easier, lighter, and smoother.
* 04/03/2026 [6.0.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v6.0.3): 🎉 New quantization methods: `ParoQuant`, `GGUF`, `FP8`, `EXL3`, and `FOEM: First-Order Error Matters`. Added PrismML/Bonsai 1bit model quantization (inference only), faster ParoQuant/AWQ kernels, ParoQuant `optimization scope` control: `module` (Paro Lite) or `layer` (Paro reference), plus `Gemma4`, `MiniCPM-O`, `MiniCPM-V`, and `GLM4 MoE Lite` model support.
* 03/19/2026 [5.8.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.8.0): ✨HF Transformers 5.3.0 support with auto-defusing of `fused` models via pypi pkg: [Defuser](https://github.com/ModelCloud/Defuser). Qwen 3.5 family support added. New fast HF `cpu` kernels for GPTQ/AWQ added. Experimental INT8 `cpu` kernel added for GPTQ. 

<details>

<summary>Archived News</summary>
* 02/09/2026 [5.7.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.7.0): ✨New `MoE.Routing` config with `Bypass` and `Override` options to allow multiple brute-force MoE routing controls for higher quality quantization of MoE experts. Combined with `FailSafeStrategy`, GPT-QModel now has three separate control settings for efficient MoE expert quantization.
`AWQ` `qcfg.zero_point` property has been merged with a unified `sym` symmetry property; `zero_point=True` is now `sym=False`.
Fixed `AWQ` `sym=True` packing/inference and quantization compatibility with some Qwen3 models. Exaone 4.0 support.

* 12/31/2025 5.7.0-dev: ✨New `FailSafe` config and `FailSafeStrategy`, auto-enabled by default, to address uneven routing of MoE experts resulting in quantization issues for some MoE modules. `Smooth` operations are introduced to `FailSafeStrategy` to reduce the impact of outliers in `FailSafe` quantization using `RTN` by default. Different `FailSafeStrategy` and `Smoothers` can be selected. `Threshold` to activate `FailSafe` can also be customized. 
New Voxtral and Glm-4v model support, plus audio dataset calibration for Qwen2-Omni. `AWQ` compatibility fix for `GLM 4.5-Air`.

* 12/17/2025 [5.6.2-12 Patch](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.6.12): Fixed `uv` compatibility. Both `uv` and `pip` installs will now show UI progress for external wheel/dependency downloads. Fixed `macOS` and `AWQMarlin` kernel loading import regressions. Resolved most `multi-arch` compile issues on `Ubuntu`, `Arch`, `RedHat` and other distros. Fixed `multi-arch` build issues and `Tritonv2` kernel launch bug on multi-GPUs. Fixed 3-bit Triton GPTQ kernel dequant/inference and `license` property compatibility issue with latest pip/setuptools.
* 12/9/2025 [5.6.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.6.0): ✨New `HF Kernel` for CPU optimized for `AMX`, `AVX2` and `AVX512`. Auto module tree for auto-model support. Added `AfMoE` and `Dots1` model support. Fixed pre-layer pass quantization speed regression. Improved HF Transformers, PEFT and Optimum support for both GPTQ and AWQ. Fixed many AWQ compatibility bugs and regressions.
* 11/9/2025 [5.4.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.4.0): ✨New Intel CPU and XPU hardware-optimized AWQ `TorchFusedAWQ` kernel. Torch Fused kernels now compatible with `torch.compile`. Fixed AWQ MoE model compatibility and reduced VRAM usage.
* 11/3/2025 [5.2.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.2.0): ✨MiniMax M2 support with [ModelCloud BF16 M2 Model](https://huggingface.co/ModelCloud/MiniMax-M2-BF16). New `VramStrategy.Balanced` quantization property for reduced memory usage for large MoE on multi-3090 (24GB) devices. ✨Marin model. New AWQ Torch reference kernel. Fixed AWQ Marlin kernel for bf16. Fixed GLM 4.5/4.6 MoE missing `mtp` layers on model save (HF bug). Modular refactor. 🎉AWQ support out of beta with full feature support including multi-GPU quant and MoE VRAM saving. ✨Brumby (attention free) model support. ✨IBM Granite Nano support. New `calibration_concat_separator` config option.

* 10/24/2025 [5.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.0.0): 🎉 Data-parallel quant support for `MoE` models on multi-GPU using `nogil` Python. `offload_to_disk` support enabled by 
default to massively reduce `CPU` RAM usage. New `Intel` and `AMD` CPU hardware-accelerated `TorchFused` kernel. Packing stage is now 4x faster and now inlined with quantization. `VRAM` pressure for large models reduced during quantization.
`act_group_aware` is  16k+ times faster and now the default when `desc_act=False` for higher quality recovery without inference penalty of `desc_act=True`. New beta quality `AWQ` support with full `gemm`, 
`gemm_fast`, `marlin` kernel support. `LFM`, `Ling`, `Qwen3 Omni` model support. 
`Bitblas` kernel updated to support Bitblas `0.1.0.post1` release.
Quantization is now faster with reduced VRAM usage. Enhanced logging support with `LogBar`.

* 09/16/2025 [4.2.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v4.2.5): `hyb_act` renamed to `act_group_aware`. Removed finicky `torch` import within `setup.py`. Packing bug fix and prebuilt PyTorch 2.8 wheels. 
* 09/12/2025 [4.2.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v4.2.0): ✨ New Models Support: Qwen3-Next, Apertus, Kimi K2, Klear, FastLLM, Nemotron H. New `fail_safe` `boolean` toggle to `.quantize()` to patch-fix non-activated `MoE` modules due to highly uneven MoE model training. Fixed LavaQwen2 compatibility. Patch-fixed GIL=0 CUDA error for multi-GPU. Fixed compatibility with autoround + new transformers. 

* 09/04/2025 [4.1.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v4.1.0): ✨ Meituan LongCat Flash Chat, Llama 4, GPT-OSS (BF16), and GLM-4.5-Air support. New experimental `mock_quantization` config to skip complex computational code paths during quantization to accelerate model quant testing. 
* 08/21/2025 [4.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v4.0.0): 🎉 New Group Aware Reordering (GAR) support. New models support: Bytedance Seed-OSS, Baidu Ernie, Huawei PanGu, Gemma3, Xiaomi Mimo, Qwen 3/MoE, Falcon H1, GPT-Neo. Memory leak and multiple model compatibility fixes related to Transformers >= 4.54. Python >= 3.13t free-threading support added with near N x GPU linear scaling for quantization of MoE models and also linear N x CPU core scaling of packing stage. Early access PyTorch 2.8 fused-ops on Intel XPU for up to 50% speedup.

* 10/17/2025 5.0.0-dev `main`: 👀: EoRA now multi-GPU compatible. Fixed both quality stability in multi-GPU quantization and VRAM usage. New LFM and Ling models support.
* 09/30/2025 5.0.0-dev `main`: 👀: New Data Parallel + Multi-GPU + Python 3.13T (PYTHON_GIL=0) equals 80%+ overall quant time reduction of large MoE models vs v4.2.5. 
* 09/29/2025 5.0.0-dev `main`: 🎉 New Qwen3 Omni model support. AWQ Marlin kernel integrated + many disk offload, threading, and memory usage fixes. 
* 09/24/2025 5.0.0-dev `main`: 🎉 Up to 90% CPU memory saving for large MoE models with faster/inline packing! 26% quant time reduction for Qwen3 MoE! AWQ Marlin kernel added. AWQ Gemm loading bug fixes. `act_group_aware` now faster and auto enabled for GPTQ when `desc_act` is False for higher quality recovery. 
* 09/19/2025 5.0.0-dev `main`: 👀 CPU memory saving of ~73.5% during quantization stage with new `offload_to_disk` quantization config property defaults to `True`. 
* 09/18/2025 5.0.0-dev `main`: 🎉 AWQ quantization support! Complete refactor and simplification of model definitions in preparation for future quantization formats.
* 08/19/2025 4.0.0-dev `main`: Fixed quantization memory usage due to some models' incorrect application of `config.use_cache` during inference. Fixed `Transformers` >= 4.54.0 compatibility which changed layer forward return signature for some models. 
* 08/18/2025 4.0.0-dev `main`: GPT-Neo model support. Memory leak fix in error capture (stack trace) and fixed `lm_head` quantization compatibility for many models.
* 07/31/2025 4.0.0-dev `main`: New Group Aware Reordering (GAR) support and preliminary PyTorch 2.8 fused-ops for Intel XPU for up to 50% speedup. 
* 07/03/2025 4.0.0-dev `main`: New Baidu Ernie and Huawei PanGu model support.
* 07/02/2025 4.0.0-dev `main`: Gemma3 4B model compatibility fix.
* 05/29/2025 4.0.0-dev `main`: Falcon H1 model support. Fixed Transformers `4.52+` compatibility with Qwen 2.5 VL models.
* 05/19/2025 4.0.0-dev `main`: Qwen 2.5 Omni model support. 
* 05/05/2025 4.0.0-dev `main`: Python 3.13t free-threading support added with near N x GPU linear scaling for quantization of MoE models and also linear N x CPU core scaling of packing stage. 
* 04/29/2025 3.1.0-dev (Now 4.) `main`: Xiaomi Mimo model support. Qwen 3 and 3 MoE model support. New arg for `quantize(..., calibration_dataset_min_length=10)` to filter out bad calibration data that exists in public dataset (wikitext). 
* 04/13/2025 [3.0.0](https://github.com/ModelCloud/Model/releases/tag/v3.0.0): 🎉 New experimental ` v2` quantization option for improved model quantization accuracy validated by `GSM8K_PLATINUM` [benchmarks](https://github.com/ModelCloud/Model#quantization-using-gptq-v2) vs original `gptq`. New `Phi4-MultiModal` model support. New Nvidia Nemotron-Ultra model support. New `Dream` model support. New experimental `multi-GPU` quantization support. Reduced VRAM usage. Faster quantization.
* 04/2/2025 [2.2.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v2.2.0): New `Qwen 2.5 VL` model support. New `samples` log column during quantization to track module activation in MoE models. `Loss` log column now color-coded to highlight modules that are friendly/resistant to quantization. Progress (per-step) stats during quantization now streamed to log file. Auto `bfloat16` dtype loading for models based on model config. Fixed kernel compile for PyTorch/ROCm. Slightly faster quantization and auto-resolve some low-level OOM issues for smaller VRAM GPUs. 
* 03/12/2025 [2.1.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v2.1.0): ✨ New `QQQ` quantization method and inference support!
New Google `Gemma 3` zero-day model support.
New Alibaba `Ovis 2` VL model support. 
New AMD `Instella` zero-day model support. New `GSM8K Platinum` and `MMLU-Pro` benchmarking support.
Peft Lora training with GPT-QModel is now 30%+ faster on all GPU and IPEX devices.
Auto-detect MoE modules not activated during quantization due to insufficient calibration data. 
`ROCm` `setup.py` compatibility fixes. `Optimum` and `Peft` compatibility fixes.
Fixed `Peft` `bfloat16` training. 
* 03/03/2025 [2.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v2.0.0): 🎉 `GPTQ` quantization internals are now broken into multiple stages (processes) for feature expansion. 
Synced `Marlin` kernel inference quality fix from upstream. Added reduced-precision Marlin accumulation mode via environment control (`GPTQMODEL_MARLIN_USE_FP32=0` disables it, default is enabled).
`ModelScope` support added. Logging and CLI progress bar output has been revamped with sticky bottom progress.
Fixed `generation_config.json` save and load. Fixed Transformers v4.49.0 compatibility. Fixed compatibility of models without `bos`. Fixed `group_size=-1` and `bits=3` packing regression. 
Fixed Qwen 2.5 MoE regressions. 
Added CI tests to track regression in kernel inference quality and sweep all bits/group_sizes. Delegate logging/progress bar to [LogBar](https://github.com/modelcloud/logbar) package.
Fixed ROCm version auto-detection in `setup` install.
* 02/12/2025 [1.9.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.9.0): ⚡ Offload `tokenizer` fixes to [Toke(n)icer](https://github.com/modelcloud/tokenicer) package. Optimized `lm_head` quant time and VRAM usage.
  Optimized `DeepSeek v3/R1` model quant VRAM usage. Fixed `Optimum` compatibility regression in `v1.8.1`. 3x speed-up for `Torch` kernel when using PyTorch >= 2.5.0 with `model.optimize()`. New `calibration_dataset_concat_size` option to enable calibration data `concat` mode to mimic original GPTQ data packing strategy which may improve quant speed and accuracy for datasets like `wikitext2`. 
* 02/08/2025 [1.8.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.8.1): ⚡ `DeepSeek v3/R1` model support. New flexible weight `packing`: allow quantized weights to be packed to `[int32, int16, int8]` dtypes. 
`Triton` and `Torch` kernels support full range of new `QuantizeConfig.pack_dtype`. 
New `auto_gc: bool` control in `quantize()` which can reduce quantization time for small model with no chance of OOM. 
New `buffered_fwd: bool` control in `model.quantize()`. Over 50% quantization speed-up for visual (vl) models.  
Fixed `bits=3` packing and `group_size=-1` regression in v1.7.4.
* 01/26/2025 [1.7.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.4): New `compile()` API for ~4-8% inference TPS improvement. Faster `pack()` for post-quantization model save. `Triton` kernel validated for Intel/`XPU` when Intel Triton packages are installed. Fixed Transformers (bug) downcasting tokenizer class on save. 
* 01/20/2025 [1.7.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.3): New Telechat2 (China Telecom) and PhiMoE model support. Fixed `lm_head` weights duplicated in post-quantize save() for models with tied-embedding. 
* 01/19/2025 [1.7.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.2): Effective BPW (bits per weight) will now be logged during `load()`. Reduce loading time on Intel Arc A770/B580 `XPU` by 3.3x. Reduce memory usage in MLX conversion and fix Marlin kernel auto-select not checking CUDA compute version. 
* 01/17/2025 [1.7.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.0): 👀 ✨ `backend.MLX` added for runtime-conversion and execution of GPTQ models on Apple's `MLX` framework on Apple Silicon (M1+). ✨ `lm_head` quantization now fully supported by GPT-QModel without external pkg dependency.
* 01/07/2025 [1.6.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.6.1): 🎉 New OpenAI API compatible endpoint via `model.serve(host, port)`. Auto-enable flash-attention2 for inference. Fixed `sym=False` loading regression. 
* 01/06/2025 [1.6.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.6.0): ⚡25% faster quantization. 35% reduction in VRAM usage vs v1.5. 👀 AMD ROCm (6.2+) support added and validated for 7900XT+ GPU. Auto-tokenizer loader via `load()` API. For most models you no longer need to manually init a tokenizer for both inference and quantization.
* 01/01/2025 [1.5.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.5.1): 🎉 2025! Added `QuantizeConfig.device` to clearly define which device is used for quantization: default = `auto`. Non-quantized models are always loaded on CPU by-default and each layer is moved to `QuantizeConfig.device` during quantization to minimize VRAM usage. Compatibility fixes for `attn_implementation_autoset` in latest transformers. 

* 12/23/2024 [1.5.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.5.0): Multi-modal (image-to-text) optimized quantization support has been added for Qwen 2-VL and Ovis 1.6-VL. Previous image-to-text model quantizations did not use image calibration data, resulting in less than optimal post-quantization results. Version 1.5.0 is the first release to provide a stable path for multi-modal quantization: only text layers are quantized.
* 12/19/2024 [1.4.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.5): Windows 11 support added/validated. Ovis VL model support with image dataset calibration. Fixed `dynamic` loading. Reduced quantization VRAM usage.
* 12/15/2024 [1.4.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.2): macOS `GPU` (Metal) and `CPU` (M+) support added/validated for inference and quantization. Cohere 2 model support added.
* 12/13/2024 [1.4.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.1): Added Qwen2-VL model support. `mse` quantization control exposed in `QuantizeConfig`. Monkey patch `patch_vllm()` and `patch_hf()` API added to allow Transformers/Optimum/PEFT and vLLM to correctly load GPT-QModel quantized models while upstream PRs are in pending status.
* 12/10/2024 [1.4.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.0) `EvalPlus` harness integration merged upstream. We now support both the legacy evaluation harness and `EvalPlus`. Added pure torch `Torch` kernel. Refactored `Cuda` kernel to be `DynamicCuda` kernel. `Triton` kernel now auto-padded for max model support. `Dynamic` quantization now supports both positive `+:`:default, and `-:` negative matching which allows matched modules to be skipped entirely for quantization. Fixed auto-`Marlin` kernel selection. Added auto-kernel fallback for unsupported kernel/module pairs. Lots of internal refactor and cleanup in preparation for transformers/optimum/peft upstream PR merge. Deprecated the saving of `Marlin` weight format since `Marlin` supports auto conversion of `gptq` format to `Marlin` during runtime. 

* 11/29/2024 [1.3.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.3.1) Olmo2 model support. Intel XPU acceleration via IPEX. Model sharding Transformer compatibility fix due to API deprecation in HF. Removed triton dependency. Triton kernel now optionally dependent on triton package. 

* 11/26/2024 [1.3.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.3.0) Zero-Day Hymba model support. Removed `tqdm` and `rogue` dependency. 
* 11/24/2024 [1.2.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.2.3) HF GLM model support. ClearML logging integration. Use `device-smi` and replace `gputil` + `psutil` dependencies. Fixed model unit tests. 

* 11/11/2024 🚀 [1.2.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.2.1) Meta MobileLLM model support added. legacy evaluation integration merged upstream. Intel/IPEX CPU inference merged replacing QBits (deprecated). Auto-fix/patch ChatGLM-3/GLM-4 compatibility with latest transformers. New `.load()` and `.save()` API. 

* 10/29/2024 🚀 [1.1.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.1.0) IBM Granite model support. Full auto-buildless wheel install from PyPI. Reduce max CPU memory usage by >20% during quantization. 100% CI model/feature coverage. 

* 10/12/2024 ✨ [1.0.9](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.9) Move AutoRound to optional and fix pip install regression in v1.0.8.

* 10/11/2024 ✨ [1.0.8](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.8) Add wheel for Python 3.12 and CUDA 11.8.
* 10/08/2024 ✨ [1.0.7](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.7) Fixed Marlin (faster) kernel was not auto-selected for some models.

* 09/26/2024 ✨ [1.0.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.6) Fixed Llama 3.2 vision quantized loader.
* 09/26/2024 ✨ [1.0.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.5) Partial Llama 3.2 Vision model support (mllama): only text-layer quantization layers are supported for now.

* 09/26/2024 ✨ [1.0.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.4) Integrated Liger Kernel support for ~1/2 memory reduction on some models during quantization. Added control toggle to disable parallel packing. 
* 09/18/2024 ✨ [1.0.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.3) Added Microsoft GRIN-MoE and MiniCPM3 support.
* 08/16/2024 ✨ [1.0.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.2) Support Intel/AutoRound v0.3, prebuilt whl packages, and PyPI release. 
* 08/14/2024 ✨ [1.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.0) 40% faster `packing`, fixed Python 3.9 compatibility, added evaluation API. 
* 08/10/2024 🚀 [0.9.11](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.11) Added LG EXAONE 3.0 model support. New `dynamic` per layer/module flexible quantization where each layer/module may have different bits/params. Added proper sharding support to `backend.BITBLAS`. Auto-heal quantization errors due to small damp values. 
* 07/31/2024 🚀 [0.9.10](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.10) Ported vllm/nm `gptq_marlin` inference kernel with expanded bits (8bits), group_size (64,32), and desc_act support for all GPTQ models with `FORMAT.GPTQ`. Auto-calculate auto-round nsamples/seglen parameters based on calibration dataset. Fixed save_quantized() called on pre-quantized models with non-supported backends. HF transformers dependency updated to ensure Llama 3.1 fixes are correctly applied to both quant and inference.
* 07/25/2024 🚀 [0.9.9](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.9): Added Llama-3.1 support, Gemma2 27B quant inference support via vLLM, auto pad_token normalization, fixed auto-round quant compatibility for vLLM/SGLang, and more.  
* 07/13/2024 🚀 [0.9.8](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.8):
Run quantized models directly using GPT-QModel with fast `vLLM` or `SGLang` backend! Both vLLM and SGLang are optimized for dynamic batching inference for maximum `TPS` (check usage under examples). Marlin backend also
got full end-to-end in/out features padding to enhance current/future model compatibility.
* 07/08/2024 🚀 [0.9.7](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.7): InternLM 2.5 model support added.
* 07/08/2024 🚀 [0.9.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.6): [Intel/AutoRound](https://github.com/intel/auto-round) QUANT_METHOD support added for a potentially higher quality quantization with `lm_head` module quantization support for even more VRAM reduction: format export to `FORMAT.GPTQ` for max inference compatibility.
* 07/05/2024 🚀 [0.9.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.5): CUDA kernels have been fully deprecated in favor of Exllama(v1/v2)/Marlin/Triton.
* 07/03/2024 🚀 [0.9.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.4): HF Transformers integration added and bug fixed Gemma 2 support.
* 07/02/2024 🚀 [0.9.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.3): Added Gemma 2 support, faster quality/benchmark calculations on GPU, and more code/arg refactor.
* 06/30/2024 🚀 [0.9.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.2): Added auto-padding of model in/out-features for exllama and exllama v2. 
Fixed quantization of OPT and DeepSeek V2-Lite models. Fixed inference for DeepSeek V2-Lite.
* 06/29/2024 🚀 [0.9.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.1): With 3 new models (DeepSeek-V2, DeepSeek-V2-Lite, DBRX Converted), BITBLAS new format/kernel, proper batching of calibration dataset resulting > 50% quantization speedup, security hash check of loaded model weights, tons of refactor/usability improvements, bug fixes, and much more.
* 06/20/2924 ✨ [0.9.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.0): Thanks for all the work from ModelCloud team and the open-source ML community for their contributions!
</details>

## Special Notes: 

PrismAI/Bonsai inference sample script. GPT-QModel loads Prism/Bonsai GGUF checkpoints through its native GGUF loading path and internal GGUF runtime shim. No external `gguf` PyPI package is required.

```py
• from gptqmodel import GPTQModel

  model = GPTQModel.load("prism-ml/Bonsai-1.7B-gguf")
  # or: model = GPTQModel.load("prism-ml/Bonsai-1.7B-gguf", profile="low_memory")

  tokens = model.generate(
      "Who wrote Romeo and Juliet?",
      max_new_tokens=128,
  )[0]

  print(model.tokenizer.decode(tokens, skip_special_tokens=True))
  ```

## What is GPT-QModel?
GPT-QModel is a production-ready LLM model compression/quantization toolkit with hw-accelerated inference support for both CPU/GPU via HF Transformers, vLLM, and SGLang.

GPT-QModel currently supports GPTQ, AWQ, ParoQuant, QQQ, GGUF, FP8, EXL3, GPTAQ, EoRa, GAR and FOEM, with more quantization methods and enhancements planned. 

## Quantization Support

GPT-QModel is a modular design supporting multiple quantization methods and feature extensions.

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
| Rotation                  | ✅          | x | x | x | x             |  
| GPTAQ                     | ✅          | ✅ | ✅ | ✅ | ✅             |
| FOEM                      | ✅          | ✅ | ✅ | ✅ | ✅             |

`GGUF`, `FP8`, `EXL3`, and `ParoQuant` are currently native GPT-QModel quantization/runtime paths. `vLLM` and `SGLang` integration currently targets `GPTQ` and `AWQ`.

### Quant Method / Format / Backend Matrix

Canonical backend names are shown below. Legacy aliases such as `BACKEND.TORCH`, `BACKEND.MARLIN`, `BACKEND.GEMM`, and `BACKEND.PARO` are still accepted and normalized to the matching canonical backend for the selected quant method.

| Quant Method | Formats | Backends / Kernels |
| --- | --- | --- |
| `METHOD.GPTQ` | `FORMAT.GPTQ`, `FORMAT.GPTQ_V2`, `FORMAT.MARLIN`, `FORMAT.BITBLAS` | `FORMAT.GPTQ`: `BACKEND.GPTQ_TORCH_ATEN`, `BACKEND.GPTQ_MACHETE`, `BACKEND.GPTQ_MARLIN`, `BACKEND.GPTQ_EXLLAMA_V2`, `BACKEND.GPTQ_TORCH_FUSED`, `BACKEND.GPTQ_TRITON`, `BACKEND.GPTQ_BITBLAS`, `BACKEND.GPTQ_TORCH`, `BACKEND.GPTQ_TORCH_INT8`<br>`FORMAT.GPTQ_V2`: `BACKEND.GPTQ_TORCH_ATEN`, `BACKEND.GPTQ_EXLLAMA_V2`, `BACKEND.GPTQ_TORCH_FUSED`, `BACKEND.GPTQ_TRITON`, `BACKEND.GPTQ_BITBLAS`, `BACKEND.GPTQ_TORCH`, `BACKEND.GPTQ_TORCH_INT8`<br>`FORMAT.MARLIN`: `BACKEND.GPTQ_MARLIN`<br>`FORMAT.BITBLAS`: `BACKEND.GPTQ_BITBLAS` |
| `METHOD.AWQ` | `FORMAT.GEMM`, `FORMAT.GEMV`, `FORMAT.GEMV_FAST`, `FORMAT.LLM_AWQ`, `FORMAT.MARLIN`, `FORMAT.BITBLAS` | `FORMAT.GEMM`: `BACKEND.AWQ_TORCH_ATEN`, `BACKEND.AWQ_MACHETE`, `BACKEND.AWQ_MARLIN`, `BACKEND.AWQ_EXLLAMA_V2`, `BACKEND.AWQ_GEMM`, `BACKEND.AWQ_GEMM_TRITON`, `BACKEND.AWQ_TORCH_FUSED`, `BACKEND.AWQ_TORCH`, `BACKEND.AWQ_TORCH_INT8`, `BACKEND.AWQ_BITBLAS`<br>`FORMAT.GEMV`: `BACKEND.AWQ_GEMV`<br>`FORMAT.GEMV_FAST`: `BACKEND.AWQ_GEMV_FAST`<br>`FORMAT.LLM_AWQ`: `BACKEND.AWQ_GEMV_FAST`<br>`FORMAT.MARLIN`: `BACKEND.AWQ_MACHETE`, `BACKEND.AWQ_MARLIN`<br>`FORMAT.BITBLAS`: `BACKEND.AWQ_BITBLAS` |
| `METHOD.PARO` | `FORMAT.PAROQUANT` | `BACKEND.PAROQUANT_CUDA`, `BACKEND.PAROQUANT_TRITON` |
| `METHOD.QQQ` | `FORMAT.QQQ` | `BACKEND.QQQ` |
| `METHOD.GGUF` | `FORMAT.GGUF` | `BACKEND.GGUF_TRITON`, `BACKEND.GGUF_CPP_CUDA`, `BACKEND.GGUF_CPP_CPU`, `BACKEND.GGUF_TORCH` |
| `METHOD.FP8` | `FORMAT.FP8` | `BACKEND.FP8_TORCH` |
| `METHOD.BITSANDBYTES` | `FORMAT.BITSANDBYTES` | `BACKEND.BITSANDBYTES` |
| `METHOD.EXL3` | `FORMAT.EXL3` | `BACKEND.EXL3_EXLLAMA_V3`, `BACKEND.EXL3_TORCH` |

`BACKEND.VLLM`, `BACKEND.SGLANG`, and `BACKEND.MLX` are external runtime backends and are not part of the native kernel matrix above.

Marlin uses `GPTQMODEL_MARLIN_USE_FP32` (default: enabled) to control fp32 accumulation.

## Features
* ✨ Native integration with HF [Transformers](https://github.com/huggingface/transformers), [Optimum](https://github.com/huggingface/optimum), and [Peft](https://github.com/huggingface/peft)
* 🚀 [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) inference integration for quantized models with format = `FORMAT.[GPTQ/AWQ]`
* ✨ GPTQ, AWQ, ParoQuant, QQQ, GGUF, FP8, EXL3, GPTAQ, and FOEM quantization support.
* ✨ Prism Bonsai `Q1_0_g128` GGUF checkpoints can be loaded for post-quantized inference through the normal `model_id_or_path` argument. GPT-QModel normalizes the GGUF artifact internally for HF Transformers via its native GGUF runtime, and does not support Prism Bonsai quantization or export.
* 🚀 Quantize MoE models with ease even with extreme routing activation bias via `Moe.Routing` and/or `FailSafe`.
* 🚀 Data Parallelism for 80%+ quantization speed reduction with Multi-GPU.
* 🚀 Optimized for Python >= 3.13t (free threading) with lock-free threading.
* ✨ Linux, macOS, Windows platform support for CUDA (NVIDIA), XPU (Intel), ROCm (AMD), MPS (Apple Silicon), CPU (Intel/AMD/Apple Silicon).
* ✨ `Dynamic` per-module mixed quantization control: each layer/module can have a unique quantization config or be excluded from quantization. 
* 🚀 Intel Torch 2.8 fused kernel support for XPU [`Arc` + `Datacenter Max`] and CPU [`avx`, `amx`].
* 🚀 Python 3.13.3t (free-threading, GIL disabled) support for multi-GPU accelerated quantization for MoE models and multi-core CPU boost for packing.
* ✨ Asymmetric `Sym=False` support. 
* ✨ `lm_head` module quant inference support for further VRAM reduction.
* 🚀 [Microsoft/BITBLAS](https://github.com/microsoft/BitBLAS) optimized tile based inference.
* 💯 CI unit-test coverage for all supported models and kernels including post-quantization quality regression.

## Who's Using GPT-QModel?

Selected public references where teams or companies explicitly mention GPT-QModel in documentation, integration notes, or quantized model usage. This is not an exhaustive customer list.

* <img src="https://cdn.simpleicons.org/huggingface/FFD21E" alt="Hugging Face logo" height="14"> Hugging Face
* <img src="https://cdn.simpleicons.org/intel/0071C5" alt="Intel logo" height="14"> Intel
* <img src="https://cdn.simpleicons.org/nvidia/76B900" alt="NVIDIA logo" height="14"> NVIDIA
* <img src="https://cdn.simpleicons.org/alibabacloud/FF6A00" alt="Alibaba Cloud logo" height="14"> Alibaba Cloud


## Quality: GPTQ 4bit can match native BF16:
🤗 [ModelCloud quantized Vortex models on HF](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2)

<img src=https://github.com/user-attachments/assets/c1b89394-f8f6-44e5-9949-bef15a124723 width="51%"> <img src=https://github.com/user-attachments/assets/23901236-10c5-4435-ac2f-06cf2e097f1e width="47%">

## Model Support  
| Model             |   |               |   |                        |   |                |   |                     |   |
|-------------------|---|---------------|---|------------------------|---|----------------|---|---------------------|---|
| Apertus           | ✅ | EXAONE 3/4    | ✅ | Dots1                  | ✅ | Mistral3       | ✅ | Qwen 2/3/3.5 (Next/MoE) | ✅ |
| Baichuan          | ✅ | Falcon (H1 / Mamba) | ✅ | InternLM 1/2/2.5 | ✅ | Mixtral        | ✅ | Qwen 2/2.5/3 VL     | ✅ |
| Bloom             | ✅ | FastVLM       | ✅ | Kimi K2                | ✅ | MobileLLM      | ✅ | Qwen 2.5/3 Omni     | ✅ |
| ChatGLM           | ✅ | Gemma 1-4 / 3n | ✅ | Klear                 | ✅ | MOSS           | ✅ | RefinedWeb          | ✅ |
| CodeGen           | ✅ | GPTBigCode    | ✅ | LING/RING              | ✅ | MPT            | ✅ | StableLM            | ✅ |
| Cohere 1-2        | ✅ | GPT-Neo / NeoX | ✅ | Llama 1-3.3           | ✅ | Nemotron H     | ✅ | StarCoder2          | ✅ |
| DBRX Converted    | ✅ | GPT-2         | ✅ | Llama 3.2 VL           | ✅ | Nemotron Ultra | ✅ | TeleChat2           | ✅ |
| Deci              | ✅ | GPT-J         | ✅ | Llama 4                | ✅ | OPT            | ✅ | Trinity             | ✅ |
| DeepSeek-V2/V3/R1 | ✅ | GPT-OSS       | ✅ | LongCat Flash          | ✅ | OLMo2 / LLaDA2 | ✅ | Yi                  | ✅ |
| DeepSeek-V2-Lite  | ✅ | Granite / Granite MoE | ✅ | LongLLaMA       | ✅ | Ovis 1.6/2     | ✅ | Seed-OSS            | ✅ |
| Dream             | ✅ | GRIN-MoE      | ✅ | Instella               | ✅ | Phi 1-4        | ✅ | Voxtral             | ✅ |
| ERNIE 4.5 / 4.5 MoE | ✅ | GLM 4/4V/5/5.1/OCR/ASR | ✅ | GLM4 MoE / Lite   | ✅ | MiniCPM 3/O/V  | ✅ | PanGu-α             | ✅ |
| XVERSE            | ✅ | Brumby        | ✅ | Hymba                  | ✅ | Mistral                      | ✅ | Qwen 1/2/3/3.5     | ✅ |
| MiniMax M2        | ✅ | AfMoE         | ✅ | Bailing-MoE            | ✅ | LFM2-MoE       | ✅ | Marin               | ✅ |

Prism Bonsai GGUF checkpoints are supported for inference only through GPT-QModel's native GGUF path and internal GGUF runtime. Bonsai checkpoints load through the normal model path or repo argument and do not require the external `gguf` package. Prism model quantization is not included.


## Platform and HW Support 

GPT-QModel is validated on Linux, macOS, and Windows 11:

| Platform        | Device        |     |  Optimized Arch          | Kernels                                       |
|-----------------|---------------| --- | ------------ |-----------------------------------------------| 
| 🐧 Linux           | NVIDIA GPU    | ✅       | `Turing+` | Marlin, Exllama V2, Exllama V1, Triton, Torch |
| 🐧 Linux | AMD GPU     | ✅             |   `7900XT+`,  `ROCm 6.2+` | Exllama V2, Exllama V1, Torch                 |
| 🐧 Linux | Intel XPU     | ✅             |   `Arc`, `Datacenter Max` | TorchFused, TorchFusedAWQ, Torch              |
| 🐧 Linux           | Intel/AMD CPU | ✅          | `avx`, `amx` | TorchFused, TorchFusedAWQ, Torch                           |
| 🍎 macOS | GPU (Metal) / CPU          | ✅             |   `Apple Silicon`, `M1+` | Torch, MLX via conversion                     |
| 🪟 Windows | GPU (NVIDIA) / CPU       | ✅             |   `NVIDIA`  | Torch                                         |

`Marlin` and JIT CUDA kernels now support NVIDIA `Turing+` (`sm_75+`) GPUs.


## Install

### PIP/UV 

```bash
# You can install optional modules like autoround, ipex, vllm, sglang, bitblas.
# Example: pip install -v gptqmodel[vllm,sglang,bitblas]
pip install -v gptqmodel
uv pip install -v gptqmodel
```

The package depends on `ninja` for first-use JIT kernel compilation.

### Install from source

```bash
# clone repo
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# python3-dev is required for some source installs
apt install python3-dev

# pip: install from source
# You can install optional modules like  vllm, sglang, bitblas.
# Example: pip install -v .[vllm,sglang,bitblas]
pip install -v .
```

### Inference
Three-line API to use `GPT-QModel` for GPTQ model inference:

```py
from gptqmodel import GPTQModel

model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")
result = model.generate("Uncovering deep insights begins with")[0] # tokens
print(model.tokenizer.decode(result)) # string output
```

To use models from [ModelScope](https://www.modelscope.cn/) instead of HuggingFace Hub, set an environment variable:
```shell
export GPTQMODEL_USE_MODELSCOPE=True
```

### FP32 accumulation toggle

Some AWQ and ParoQuant CUDA/Triton kernels support an fp32 accumulation mode to reduce numerical drift during fused quantized matmul. This setting defaults to `True` because accuracy is prioritized over speed.

```shell
# default behavior: higher accuracy, slightly lower speed on some kernels
export GPTQMODEL_FP32_ACCUM=1

# optional speed-first mode for some kernels
export GPTQMODEL_FP32_ACCUM=0
```

Notes:
* This is a runtime toggle. It does not change model weights or saved checkpoints.
* It mainly affects some fused AWQ and ParoQuant CUDA/Triton kernels. Dense/dequantize fallback paths are mostly unaffected.
* `1` is recommended for regression testing and quality-sensitive evaluation. `0` may be useful when chasing a small latency win and the quality tradeoff is acceptable.

### OpenAI API compatible endpoint
```py
# load model using above inference guide first
model.serve(host="0.0.0.0",port="12345")
```

### Quantization
Basic example of using `GPT-QModel` to quantize an LLM model:

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

#### Other Quantization Formats

`QuantizeConfig` remains the broad factory. The concrete config classes are now `GPTQConfig`, `AWQConfig`, `ParoConfig`, `QQQConfig`, `RTNConfig`, `GGUFConfig`, `FP8Config`, `BitsAndBytesConfig`, and `EXL3Config`.

`GPTQ`, `AWQ`, `ParoQuant`, and `EXL3` are calibration-based. `GGUF` and `FP8` are weight-only and should be quantized with `calibration=None`.

##### Preprocessors

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
from gptqmodel import BACKEND, FP8Config, GPTQModel

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
from gptqmodel import BACKEND, EXL3Config, GPTQModel

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

#### MoE Quantization

Some MoE (mixture of experts) models have extremely uneven/biased routing (distribution of tokens) to the `experts` causing some expert modules to receive close-to-zero activated tokens, thus failing to complete calibration-based quantization (GPTQ/AWQ).
To better quantize these heavily biased `MoE` routed modules, GPT-QModel exposes 3 controls:

* `Moe.Routing = ExpertsRoutingOverride`: Manually override the `num_experts_per_tok` used for model `routing` math, i.e., if a model only routes 4 experts per token out of 48 total experts, you can set this equal to 24 for 50% routing or 48 for 100% routing.
`ExpertsRoutingOverride` requires the model exposes `num_experts_per_tok` or equivalent configuration control.
* `Moe.Routing = ExpertsRoutingBypass`: Brute-force and bypass all `routing` math so all `experts` receive `all` activated tokens. This is akin to `ExpertsRoutingOverride.num_experts_per_tok` set to total number of experts. 
`ExpertsRoutingBypass` is enabled/tested for some models and, due to the lifecycle complexity, it needs to be validated for every model.
* `FailSafe`: This is `enabled` by `default` and is a naive weight-only quantization technique using simple (naive) quantization methods such as `nearest` with optional `smoothing`. 
There are various `FailSafeStrategy` options, along with `SmoothMethod` options, to complement this feature. `FailSafe` does not require `activations` but has higher quantization error loss than normally activated GPTQ/AWQ. It is fast and applicable for all MoE models.

`FailSafe` can be combined with `ExpertsRoutingOverride`. There is no single best way to quantize MoE, and we recommend users to test all three methods.

### Quantized Inference
```py
# test post-quant inference
model = GPTQModel.load(quant_path)
result = model.generate("Uncovering deep insights begins with")[0] # tokens
print(model.tokenizer.decode(result)) # string output
```

### EoRA Accuracy Recovery: Enhanced Post-Quant Error Recovery via Lora

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

### How to Add Support for a New Model

Read the [`gptqmodel/models/llama.py`](https://github.com/ModelCloud/GPTQModel/blob/5627f5ffeb3f19b1a2a97e3b6de6fbe668b0dc42/gptqmodel/models/llama.py) code which explains in detail via comments how the model support is defined. Use it as a guide for PRs to add new models. Most models follow the same pattern.

### Pair with Evaluation for post-quantization LLM Benchmarks

GPT-QModel evaluation is integrated into [Evalution](https://github.com/ModelCloud/Evalution), a modern benchmarking toolkit with 150+ of the world's most widely used benchmark suites.
We highly recommend using Evalution to measure post-quant accuracy recovery after quantization instead of relying on narrow regression-only language-model metrics.

```
# install Evalution
pip install Evalution
```

Below is a short example running `gsm8k_platinum` through Evalution's native GPT-QModel engine.

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
### Dynamic Quantization (Per Module QuantizeConfig Override)

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
    
    # negative match: skip all down modules for all layers
    r"-:.*down.*": {},  
 } 

```

### Group Aware Reordering (GAR)

Group Aware Reordering (GAR) is an enhanced activation reordering scheme developed by Intel to improve the accuracy of quantized models without incurring additional inference overhead. Unlike traditional activation reordering, GAR restricts permutations to within individual groups or rearrangements of entire groups. This ensures each group's associated scales and zero-points remain efficiently accessible during inference, thereby avoiding any inference-time overhead.

How to enable GAR:

Set the `act_group_aware` parameter to `True` and disable the default activation reordering by setting `desc_act` to `False` in your `QuantizeConfig`. For example:

```python
quant_config = QuantizeConfig(bits=4, group_size=128, act_group_aware=True)
```


### Experimental Features

#### Using GPTAQ (Experimental, not MoE compatible, and results may not be better than original)

Enable GPTAQ quantization by setting `gptaq = GPTAQConfig(...)`.
```py
# Note GPTAQ is currently experimental, not MoE compatible, and requires 2-4x more VRAM to execute
# We have many reports of GPTAQ not working better or exceeding GPTQ so please use for testing only
# If OOM on 1 GPU, please set CUDA_VISIBLE_DEVICES=0,1 to 2 GPUs and gptqmodel will auto use second GPU
quant_config = QuantizeConfig(bits=4, group_size=128, gptaq=GPTAQConfig(alpha=0.25, device="auto"))
```

#### Using FOEM

FOEM (First-order error matters) adds first-order error compensation for GPTQ-style quantization. Enable FOEM by setting `foem = FOEMConfig(...)`.
```py
# FOEM default hyperparameters are alpha=0.0 and beta=0.2
quant_config = QuantizeConfig(bits=4, group_size=128, foem=FOEMConfig(alpha=0.0, beta=0.2, device="auto"))
```
### Migrating from AutoGPTQ and AutoAWQ:

GPT-QModel has fully supplanted AutoGPTQ and AutoAWQ for HF Transformers/Optimum/Peft integration. Model inference has drop-in support with zero changes. 

For model quantization, there are some config changes for AutoAWQ:

* AutoAWQ: `version` property is now `format`. `zero_point` is now `sym` (Symmetric Quantization): `sym = True` is equivalent to `zero_point = False`

Models quantized by GPT-QModel are inference compatible with HF Transformers (minus `dynamic`), vLLM, and SGLang. 

## Attributions:

* GPTQ: IST-DASLab, main-author: Elias Frantar, arXiv:2210.17323
* AWQ: main-authors: Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song
* ParoQuant: Z-Lab, main-authors: Yesheng Liang, Haisheng Chen, Song Han, and Zhijian Liu. [Official implementation](https://github.com/z-lab/paroquant), [Paper](https://openreview.net/forum?id=1USeVjsKau)
* EoRA: Nvidia, main-author: Shih-Yang Liu, arXiv preprint arXiv:2410.21271.
* GAR: Intel, main-author: T Gafni, A Karnieli, Y Hanani, [Paper](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Gafni_Dual_Precision_Quantization_for_Efficient_and_Accurate_Deep_Neural_Networks_CVPRW_2025_paper.html)
* GPTAQ: Yale Intelligent Computing Lab, main-author: Yuhang Li, arXiv:2504.02692.
* QQQ: Meituan, main-author Ying Zhang, arXiv:2406.09904
* FOEM: Zheng, Xingyu and Qin, Haotong and Li, Yuye and Chu, Haoran and Wang, Jiakai and Guo, Jinyang and Magno, Michele and Liu, Xianglong [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40123)

## Citations:

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

```

## Quick Notes

### Limit log level

`GPT-QModel` uses a shared `LogBar` logger. Set the level once near process startup:

```python
from logbar import LogBar

LogBar.shared().setLevel("WARNING")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Apply Triton nogil patch early in multi-package scripts

If your script imports multiple Triton users (for example `gptqmodel`, `vllm`, and `sglang`), apply the patch at the very top before other Triton-related imports:

```python
from gptqmodel import TritonPatch

# Fix Triton crashing under nogil/free-threading Python 3.13+ where the kernel cache storage in Triton is not thread-safe
TritonPatch.apply()
```
