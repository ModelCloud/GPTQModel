<h1 align="center">GPTQModel</h1>
<p align="center">Production ready LLM model compression/quantization toolkit with accelerated inference support for both cpu/gpu via HF, vLLM, and SGLang.</p>
<p align="center">
    <a href="https://github.com/ModelCloud/GPTQModel/releases" style="text-decoration:none;"><img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/GPTQModel.svg"></a>
    <a href="https://pypi.org/project/gptqmodel/" style="text-decoration:none;"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/gptqmodel"></a>
    <a href="https://pepy.tech/projects/gptqmodel" style="text-decoration:none;"><img src="https://static.pepy.tech/badge/gptqmodel" alt="PyPI Downloads"></a>
    <a href="https://github.com/ModelCloud/GPTQModel/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/gptqmodel"></a>
    <a href="https://huggingface.co/modelcloud/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ModelCloud-%23ff8811.svg"></a>
</p>
  
## News
* 01/20/2025 [1.7.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.3): New Telechat2 (China Telecom) and PhiMoE model support. Fixed `lm_head` weights duplicated in post-quantize save() for models with tied-embedding. 
* 01/19/2025 [1.7.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.2): Effective BPW (bits per weight) will now be logged during `load()`. Reduce loading time on Intel Arc A770/B580 `XPU` by 3.3x. Reduce memory usage in MLX conversion and fix Marlin kernel auto-select not checking CUDA compute version. 
* 01/17/2025 [1.7.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.7.0): 👀 ✨ `backend.MLX` added for runtime-conversion and execution of GPTQ models on Apple's `MLX` framework on Apple Silicon (M1+). Exports of `gptq` models to `mlx` also now possible. We have added `mlx` exported models to [huggingface.co/ModelCloud](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2). ✨ `lm_head` quantization now fully support by GPTQModel without external pkg dependency. 
* 01/07/2025 [1.6.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.6.1): 🎉 New OpenAI api compatible end-point via `model.serve(host, port)`. Auto-enable flash-attention2 for inference.  Fixed `sym=False` loading regression. 
* 01/06/2025 [1.6.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.6.0): ⚡25% faster quantization. 35% reduction in vram usage vs v1.5. 👀 AMD ROCm (6.2+) support added and validated for 7900XT+ GPU. Auto-tokenizer loader via `load()` api. For most models you no longer need to manually init a tokenizer for both inference and quantization.
* 01/01/2025 [1.5.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.5.1): 🎉 2025! Added `QuantizeConfig.device` to clearly define which device is used for quantization: default = `auto`. Non-quantized models are always loaded on cpu by-default and each layer is moved to `QuantizeConfig.device` during quantization to minimize vram usage. Compatibility fixes for `attn_implementation_autoset` in latest transformers. 
* 12/23/2024 [1.5.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.5.0): Multi-modal (image-to-text) optimized quantization support has been added for Qwen 2-VL and Ovis 1.6-VL. Previous image-to-text model quantizations did not use image calibration data, resulting in less than optimal post-quantization results. Version 1.5.0 is the first release to provide a stable path for multi-modal quantization: only text layers are quantized.
* 12/19/2024 [1.4.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.5): Windows 11 support added/validated. Ovis VL model support with image dataset calibration. Fixed `dynamic` loading. Reduced quantization vram usage. 
* 12/15/2024 [1.4.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.2): MacOS `gpu` (Metal) and `cpu` (M+) support added/validated for inference and quantization. Cohere 2 model support added. 
* 12/13/2024 [1.4.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.1): Added Qwen2-VL model support. `mse` quantization control exposed in `QuantizeConfig`. Monkey patch `patch_vllm()` and `patch_hf()` api added to allow Transformers/Optimum/PEFT and vLLM to correctly loaded GPTQModel quantized models while upstream PRs are in pending status. 


<details>
    
<summary>Archived News</summary>
* 12/10/2024 [1.4.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.4.0) `EvalPlus` harness integration merged upstream. We now support both `lm-eval` and `EvalPlus`. Added pure torch `Torch` kernel. Refactored `Cuda` kernel to be `DynamicCuda` kernel. `Triton` kernel now auto-padded for max model support. `Dynamic` quantization now supports both positive `+:`:default, and `-:` negative matching which allows matched modules to be skipped entirely for quantization. Fixed auto-`Marlin` kerenl selection. Added auto-kernel fallback for unsupported kernel/module pairs. Lots of internal refractor and cleanup in-preparation for transformers/optimum/peft upstream PR merge. Deprecated the saving of `Marlin` weight format since `Marlin` supports auto conversion of `gptq` format to `Marlin` during runtime. 

* 11/29/2024 [1.3.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.3.1) Olmo2 model support. Intel XPU acceleration via IPEX. Model sharding Transformer compat fix due to api deprecation in HF. Removed triton dependency. Triton kernel now optionally dependent on triton pkg. 

* 11/26/2024 [1.3.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.3.0) Zero-Day Hymba model support. Removed `tqdm` and `rogue` dependency. 
* 11/24/2024 [1.2.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.2.3) HF GLM model support. ClearML logging integration. Use `device-smi` and replace `gputil` + `psutil` depends. Fixed model unit tests. 

* 11/11/2024 🚀 [1.2.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.2.1) Meta MobileLLM model support added. `lm-eval[gptqmodel]` integration merged upstream. Intel/IPEX cpu inference merged replacing QBits (deprecated). Auto-fix/patch ChatGLM-3/GLM-4 compat with latest transformers. New `.load()` and `.save()` api. 

* 10/29/2024 🚀 [1.1.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.1.0) IBM Granite model support. Full auto-buildless wheel install from pypi. Reduce max cpu memory usage by >20% during quantization. 100% CI model/feature coverage. 

* 10/12/2024 ✨ [1.0.9](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.9) Move AutoRound to optional and fix pip install regression in v1.0.8.

* 10/11/2024 ✨ [1.0.8](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.8) Add wheel for python 3.12 and cuda 11.8.
* 10/08/2024 ✨ [1.0.7](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.7) Fixed marlin (faster) kernel was not auto-selected for some models.

* 09/26/2024 ✨ [1.0.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.6) Fixed quantized Llama 3.2 vision quantized loader.
* 09/26/2024 ✨ [1.0.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.5) Partial Llama 3.2 Vision model support (mllama): only text-layer quantization layers are supported for now.

* 09/26/2024 ✨ [1.0.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.4) Integrated Liger Kernel support for ~1/2 memory reduction on some models during quantization. Added control toggle disable parallel packing. 
* 09/18/2024 ✨ [1.0.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.3) Added Microsoft GRIN-MoE and MiniCPM3 support.
* 08/16/2024 ✨ [1.0.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.2) Support Intel/AutoRound v0.3, pre-built whl packages, and PyPI release. 
* 08/14/2024 ✨ [1.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.0) 40% faster `packing`, Fixed Python 3.9 compat, added `lm_eval` api. 
* 08/10/2024 🚀 [0.9.11](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.11) Added LG EXAONE 3.0 model support. New `dynamic` per layer/module flexible quantization where each layer/module may have different bits/params. Added proper sharding support to `backend.BITBLAS`. Auto-heal quantization errors due to small damp values. 
* 07/31/2024 🚀 [0.9.10](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.10) Ported vllm/nm `gptq_marlin` inference kernel with expanded bits (8bits), group_size (64,32), and desc_act support for all GPTQ models with `FORMAT.GPTQ`. Auto calculate auto-round nsamples/seglen parameters based on calibration dataset. Fixed save_quantized() called on pre-quantized models with non-supported backends. HF transformers depend updated to ensure Llama 3.1 fixes are correctly applied to both quant and inference.
* 07/25/2024 🚀 [0.9.9](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.9): Added Llama-3.1 support, Gemma2 27B quant inference support via vLLM, auto pad_token normalization, fixed auto-round quant compat for vLLM/SGLang, and more.  
* 07/13/2024 🚀 [0.9.8](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.8):
Run quantized models directly using GPTQModel using fast `vLLM` or `SGLang` backend! Both vLLM and SGLang are optimized for dyanamic batching inference for maximum `TPS` (check usage under examples). Marlin backend also
got full end-to-end in/out features padding to enhance current/future model compatibility.
* 07/08/2024 🚀 [0.9.7](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.7): InternLM 2.5 model support added.
* 07/08/2024 🚀 [0.9.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.6): [Intel/AutoRound](https://github.com/intel/auto-round) QUANT_METHOD support added for a potentially higher quality quantization with `lm_head` module quantization support for even more vram reduction: format export to `FORMAT.GPTQ` for max inference compatibility.
* 07/05/2024 🚀 [0.9.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.5): Cuda kernels have been fully deprecated in favor of Exllama(v1/v2)/Marlin/Triton.
* 07/03/2024 🚀 [0.9.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.4): HF Transformers integration added and bug fixed Gemma 2 support.
* 07/02/2024 🚀 [0.9.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.3): Added Gemma 2 support, faster PPL calculations on gpu, and more code/arg refractor.
* 06/30/2024 🚀 [0.9.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.2): Added auto-padding of model in/out-features for exllama and exllama v2. 
Fixed quantization of OPT and DeepSeek V2-Lite models. Fixed inference for DeepSeek V2-Lite.
* 06/29/2024 🚀 [0.9.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.1): With 3 new models (DeepSeek-V2, DeepSeek-V2-Lite, DBRX Converted), BITBLAS new format/kernel, proper batching of calibration dataset resulting > 50% quantization speedup, security hash check of loaded model weights, tons of refractor/usability improvements, bugs fixes and much more.
* 06/20/2924 ✨ [0.9.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.0): Thanks for all the work from ModelCloud team and the opensource ML community for their contributions!
</details>

## What is GPTQModel?

GPTQModel started out as a major refractor (fork) of AutoGPTQ but has now morphed into a full-stand-in replacement with cleaner api, up-to-date model support, faster inference, higher quality quants and a pledge that ModelCloud, together with the open-source ML community, will take every effort to bring the library up-to-date with latest advancements and model support.

Public tests/papers and ModelCloud's internal tests have shown that GPTQ is on-par and/or exceeds other 4bit quantization methods in terms of both quality recovery and production-level inference speed for token latency and rps. GPTQ has the optimal blend of quality and inference speed you need in a real-world production deployment. 

## Features
* 🚀 Extensive model support for: `Ovis VL`, `Llama 1-3.3`, `Qwen2-VL`, `Olmo2`, `Hymba`, `GLM`, `IBM Granite`, `Llama 3.2 Vision`, `MiniCPM3`, `GRIN-Moe`, `Phi 1-4`, `EXAONE 3.0`, `InternLM 2.5`, `Gemma 2`, `DeepSeek-V2`, `DeepSeek-V2-Lite`, `ChatGLM`, `MiniCPM`, `Qwen2MoE`, `DBRX`.
* ✨ Linux, MacOS, Windows platform quantization and accelerated inference support for CUDA (Nvidia), XPU (Intel), ROCm (AMD), MPS (Apple Silicon), CPU (Intel/AMD/Apple Silicon).
* 💯 100% CI unit-test coverage for all supported models and kernels including post-quantization quality regression.
* ✨ `Dynamic` mixed quantization control on a per-module basis. Each layer/module can have a unique quantization config or be excluded from quantization all together. 
* 🚀 [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) inference integration for quantized model where format = `FORMAT.GPTQ` 
* 🚀 [Intel/IPEX](https://github.com/intel/intel-extension-for-pytorch) hardware accelerated quantization/inference for CPU [`avx`, `amx`, `xmx`] and Intel GPU [`Arc` + `Datacenter Max`]. 
* 🚀 [Microsoft/BITBLAS](https://github.com/microsoft/BitBLAS) format + dynamically compiled inference.
* ✨ [Intel/AutoRound](https://github.com/intel/auto-round) alternative gptq-inference compatible quantization method.
* ✨ Asymmetric `Sym=False` support. 
* ✨ `lm_head` module quant inference support for further VRAM reduction (auto-round only). 
* ✨ Model weights sharding support with optional hash check of model weights on load.
* 🚀 40% faster `packing` stage in quantization (Llama 3.1 8B). 50% faster PPL calculations (OPT).

## Quality: GPTQ 4bit (5.0 bpw) can match BF16:
🤗 [ModelCloud quantized Vortex models on HF](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2)

![image](https://github.com/user-attachments/assets/713e00e4-d991-4917-aca1-0af35fe853c6)

## Model Support  
| Model            |    |                |    |                  |    |            |    |    |   |
|------------------|----|----------------|----|------------------|----|------------|----|----|---|
| Baichuan         | ✅  | Falcon         | ✅  | Llama 1-3.3      | ✅  | OLMo2      | ✅ | Yi | ✅ |
| Bloom            | ✅  | Gemma 2        | ✅ | Llama 3.2 VL | ✅ | Ovis 1.6   | ✅ |   XVERSE | ✅  |
| ChatGLM          | ✅ | GPTBigCod      | ✅  | LongLLaMA        | ✅  | Phi 1-4    | ✅ |    |   |
| CodeGen          | ✅  | GPTNeoX        | ✅  | MiniCPM3         | ✅  | Qwen       | ✅  |    |   |
| Cohere 1-2       | ✅  | GPT-2          | ✅  | Mistral          | ✅  | Qwen2 MoE   | ✅ |    |   |
| DBRX Converted   | ✅ | GPT-J          | ✅  | Mixtral          | ✅  | Qwen2 VL    | ✅ |    |   |
| Deci             | ✅  | Granite        | ✅ | MobileLLM        | ✅ | RefinedWeb | ✅  |    |   |
| DeepSeek-V2      | ✅ | GRIN-MoE       | ✅ | MOSS             | ✅  | StableLM   | ✅  |    |   |
| DeepSeek-V2-Lite | ✅ | Hymba          | ✅ | MPT              | ✅  | StarCoder2 | ✅  |    |   |
| EXAONE 3.0       | ✅ | InternLM 1/2.5 | ✅ | OPT              | ✅  | TeleChat2 | ✅  |    |   |

## Platform and HW Support 

GPTQModel is validated for Linux, MacOS, and Windows 11:

| Platform        | Device        |     |  Optimized Arch              |  Kernels |
|-----------------|---------------| --- | -------------- | -------------- | 
| 🐧 Linux           | Nvidia GPU    | ✅       | `Ampere+` | Marlin, Exllama V2, Exallma V1, Triton, DyanamicCuda, Torch |
| 🐧 Linux | Intel XPU     | ✅             |   `Arc`, `Datacenter Max` | IPEX, Torch |
| 🐧 Linux | AMD GPU     | ✅             |   `7900XT+`,  `ROCm 6.2+` | Exllama V2, Exallma V1, DyanamicCuda, Torch |
| 🐧 Linux           | Intel/AMD CPU | ✅          | `avx`, `amx`, `xmx` | IPEX, Torch |
| 🍎 MacOS | GPU (Metal) / CPU          | ✅             |   `Apple Silicon`, `M1+` | Torch, MLX via conversion |
| 🪟 Windows | GPU (Nvidia) / CPU       | ✅             |   `Nvidia`  | DynamicCuda, Torch  |


## Install

### PIP/UV 

```bash
# You can install optional modules like autoround, ipex, vllm, sglang, bitblas, and ipex.
# Example: pip install -v --no-build-isolation gptqmodel[vllm,sglang,bitblas,ipex,auto_round]
pip install -v gptqmodel --no-build-isolation
uv pip install -v gptqmodel --no-build-isolation
```

### Install from source

```bash
# clone repo
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# pip: compile and install
# You can install optional modules like autoround, ipex, vllm, sglang, bitblas, and ipex.
# Example: pip install -v --no-build-isolation .[vllm,sglang,bitblas,ipex,auto_round]
pip install -v . --no-build-isolation
```

### Inference
Two line api to use `GPTQModel` for gptq model inference:

```py
from gptqmodel import GPTQModel

model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")
result = model.generate("Uncovering deep insights begins with")[0]
```


### OpenAI API compatible end-point
```py
# load model using above inference guide first
model.serve(host="0.0.0.0",port="12345")
```

### Quantization
Basic example of using `GPTQModel` to quantize a llm model:

```py
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

# test post-quant inference
model = GPTQModel.load(quant_path)
result = model.generate("Uncovering deep insights begins with")[0]
```

For more advanced features of model quantization, please reference to [this script](https://github.com/ModelCloud/GPTQModel/blob/main/examples/quantization/basic_usage_wikitext2.py)

### How to Add Support for a New Model

Read the [`gptqmodel/models/llama.py`](https://github.com/ModelCloud/GPTQModel/blob/5627f5ffeb3f19b1a2a97e3b6de6fbe668b0dc42/gptqmodel/models/llama.py) code which explains in detail via comments how the model support is defined. Use it as guide to PR for to new models. Most models follow the same pattern.

### Evaluation and Quality Benchmarks

GPTQModel inference is integrated into both [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [evalplus](https://github.com/evalplus/evalplus)  
We highly recommend avoid using `ppl` and use `lm-eval`/`evalplus` to validate post-quantization model quality. `ppl` should only be used for regression tests and is not a good indicator of model output quality.  

```
# gptqmodel is integrated into lm-eval >= v0.4.7
pip install lm-eval>=0.4.7
```

```
# gptqmodel is integrated into evalplus[main]
pip install -U "evalplus @ git+https://github.com/evalplus/evalplus"
```

Below is a basic sample using `GPTQModel.eval` API

```py
from gptqmodel import GPTQModel
from gptqmodel.utils import EVAL

model_id = "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

# Use `lm-eval` as framework to evaluate the model
lm_eval_results = GPTQModel.eval(model_id, framework=EVAL.LM_EVAL, tasks=[EVAL.LM_EVAL.ARC_CHALLENGE], output_file='lm-eval_result.json')

# Use `evalplus` as framework to evaluate the model
evalplus_results = GPTQModel.eval(model_id, framework=EVAL.EVALPLUS, tasks=[EVAL.EVALPLUS.HUMAN], output_file='evalplus_result.json')
```

## Citation
```
@misc{gptqmodel,
    author = {ModelCloud.ai and qubitium@modelcloud.ai},
    title = {GPTQModel},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/modelcloud/gptqmodel}},
    note = {Contact: qubitium@modelcloud.ai}
}

@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}

@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}
```
