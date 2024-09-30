<h1 align="center">GPTQModel</h1>
<p align="center">GPTQ based LLM model compression/quantization library.</p>
<p align="center">
    <a href="https://github.com/ModelCloud/GPTQModel/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/GPTQModel.svg">
    </a>
    <a href="https://pypi.org/project/gptqmodel/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/gptqmodel">
    </a>
</p>

## News
* 09/26/2024 ✨ [v1.0.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.6) Fixed quantized Llama 3.2 vision quantized loader
* 09/26/2024 ✨ [v1.0.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.5) Partial Llama 3.2 Vision model support (mllama): only text-layer quantization layers are supported for now.
* 09/26/2024 ✨ [v1.0.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.4) Integrated Liger Kernel support for ~1/2 memory reduction on some models during quantization. Added control toggle disable parallel packing. 
* 09/18/2024 ✨ [v1.0.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.3) Added Microsoft GRIN-MoE and MiniCPM3 support.
* 08/16/2024 ✨ [v1.0.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.2) Support Intel/AutoRound v0.3, pre-built whl packages, and PyPI release. 
* 08/14/2024 ✨✨ [v1.0.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v1.0.0) 40% faster `packing`, Fixed Python 3.9 compat, added `lm_eval` api. 
* 08/10/2024 🚀🚀🚀 [v0.9.11](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.11) Added LG EXAONE 3.0 model support. New `dynamic` per layer/module flexible quantization where each layer/module may have different bits/params. Added proper sharding support to `backend.BITBLAS`. Auto-heal quantization errors due to small damp values. 
* 07/31/2024 🚀🚀 [v0.9.10](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.10) Ported vllm/nm `gptq_marlin` inference kernel with expanded bits (8bits), group_size (64,32), and desc_act support for all GPTQ models with `FORMAT.GPTQ`. Auto calculate auto-round nsamples/seglen parameters based on calibration dataset. Fixed save_quantized() called on pre-quantized models with non-supported backends. HF transformers depend updated to ensure Llama 3.1 fixes are correctly applied to both quant and inference.
* 07/25/2024 🚀 [v0.9.9](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.9): Added Llama-3.1 support, Gemma2 27B quant inference support via vLLM, auto pad_token normalization, fixed auto-round quant compat for vLLM/SGLang, and more.  
* 07/13/2024 🚀🚀🚀 [v0.9.8](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.8):
Run quantized models directly using GPTQModel using fast `vLLM` or `SGLang` backend! Both vLLM and SGLang are optimized for dyanamic batching inference for maximum `TPS` (check usage under examples). Marlin backend also
got full end-to-end in/out features padding to enhance current/future model compatibility.
<details>
    
<summary>Archived News:</summary>

* 07/08/2024 🚀 [v0.9.7](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.7): InternLM 2.5 model support added.

* 07/08/2024 🚀🚀 [v0.9.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.6): [Intel/AutoRound](https://github.com/intel/auto-round) QUANT_METHOD support added for a potentially higher quality quantization with `lm_head` module quantization support for even more vram reduction: format export to `FORMAT.GPTQ` for max inference compatibility.

* 07/05/2024 🚀🚀 [v0.9.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.5): [Intel/QBits](https://github.com/intel/intel-extension-for-transformers) support added for [2,3,4,8] bit quantization/inference on CPU. Cuda kernels have been fully deprecated in favor of Exllama(v1/v2)/Marlin/Triton.

* 07/03/2024 🚀 [v0.9.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.4): HF Transformers integration added and bug fixed Gemma 2 support.

* 07/02/2024 🚀 [v0.9.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.3): Added Gemma 2 support, faster PPL calculations on gpu, and more code/arg refractor.

* 06/30/2024 🚀 [v0.9.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.2): Added auto-padding of model in/out-features for exllama and exllama v2. 
Fixed quantization of OPT and DeepSeek V2-Lite models. Fixed inference for DeepSeek V2-Lite.

* 06/29/2024 🚀🚀🚀 [v0.9.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.1): With 3 new models (DeepSeek-V2, DeepSeek-V2-Lite, DBRX Converted), BITBLAS new format/kernel, proper batching of calibration dataset resulting > 50% quantization speedup, security hash check of loaded model weights, tons of refractor/usability improvements, bugs fixes and much more.

* 06/20/2924 ✨ GPTQModel [v0.9.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.0): Thanks for all the work from ModelCloud team and the opensource ML community for their contributions!
</details>

## Mission Statement

We want GPTQModel to be highly focused on GPTQ based quantization and target inference compatibility with HF Transformers, vLLM, and SGLang. 

## How is GPTQModel different from AutoGPTQ?

GPTQModel started out as a refractored fork of AutoGTQP and but has now morphed into a full-stand-in replacement with clean api, up-to-date model support, faster inference, faster quantization, higher quality quants and a pledge from the ModelCloud team and that we, along with the open-source ML community, will take every effort to bring the library up-to-date with latest advancements, model support, and bug fixes. 

We will backport bug fixes to AutoGPTQ on a case-by-case basis.

## Major Changes (Advantages) vs AutoGPTQ
* 🚀🚀🚀🚀 Extensive model support for: `EXAONE 3.0`, `InternLM 2.5`, `Gemma 2`, `DeepSeek-V2`, `DeepSeek-V2-Lite`, `ChatGLM`, `MiniCPM`, `Phi-3`, `Qwen2MoE`, `DBRX` (Converted).
* 🚀🚀 vLLM inference integration for quantized model where format = `FORMAT.GPTQ` 
* 🚀🚀 SGLang inference integration for quantized model where format = `FORMAT.GPTQ` 
* 🚀 [Intel/AutoRound](https://github.com/intel/auto-round) QUANT_METHOD support added for a potentially higher quality quantization with `lm_head` module quantization support for even more vram reduction: format export to `FORMAT.GPTQ` for max inference compatibility.
* 🚀 [Intel/QBits](https://github.com/intel/intel-extension-for-transformers) support added for [2,3,4,8] bit quantization/inference on CPU.
* 🚀 [BITBLAS](https://github.com/microsoft/BitBLAS) format/inference support from Microsoft
* 🚀`Sym=False` Support. AutoGPTQ has unusable `sym=false`. (Re-quant required)
* 🚀`lm_head` module quant inference support for further VRAM reduction. 
* 🚀 Faster quantization: More than 50% faster for TinyLlama + 4090 with batching and large calibration dataset.
* 🚀 Better quality quants as measured by PPL. (Test config: defaults + `sym=True` + `FORMAT.GPTQ`, TinyLlama)
* 🚀 Model weights sharding support
* 🚀 Security: hash check of model weights on load
* 🚀 Over 50% faster PPL calculations (OPT)
* 🚀 Over 40% faster `packing` stage in quantization (Llama 3.1 8B)
* ✨ Alert users of sub-optimal calibration data. Most new users get this part horribly wrong.
* ✨ Increased compatibility with newest models with auto-padding of in/out-features for [ Exllama, Exllama V2 ] backends.
* 👾 Removed non-working, partially working, or fully deprecated features: Peft, ROCM, AWQ Gemm inference, Triton v1 (replaced by v2), Fused Attention (Replaced by Marlin/Exllama).
* 👾 <del>Fixed packing Performance regression on high core-count systems.</del> Backported to AutoGPTQ
* 👾 <del>Fixed crash on H100.</del> Backported to AutoGPTQ
* ✨ 10s of thousands of lines of refactor/cleanup.
* ✨ Over 8+ overly complex api args removed/merged into simple human-readable args. 
* ✨ Added CI workflow for validation of future PRs and prevent code regressions.
* ✨ Added perplexity unit-test to prevent against model quant quality regressions.
* 👾 De-bloated 271K lines of which 250K was caused by a single dataset used only by an example. 
* 👾 De-bloat the number of args presented in public `.from_quantized()`/`.from_pretrained()` api
* ✨ Shorter and more concise public api/internal vars. No need to mimic HF style for verbose class names. 
* ✨ Everything that did not pass unit-tests have been removed from repo.

## Model Support ( 🚀 GPTQModel only ) 
[Ready to deply quantized models](https://hf.co/ModelCloud)
  
| Model            |     |                       |     |           |     |            |     |     |
| ---------------- | --- | --------------------- | --- | --------- | --- | ---------- | --- | --- |
| Baichuan         | ✅   | EXAONE 3.0            | 🚀  | Llama     | ✅   | Phi/Phi-3  | 🚀  |     |
| Bloom            | ✅   | Falon                 | ✅   | LongLLaMA | ✅   | Qwen       | ✅   |     |
| ChatGLM          | 🚀  | Gemma 2               | 🚀  | MiniCPM   | 🚀  | Qwen2MoE   | 🚀  |     |
| CodeGen          | ✅   | GPTBigCod             | ✅   | MiniCPM3  | 🚀  | RefinedWeb | ✅   |     |
| Cohere           | ✅   | GPTNeoX               | ✅   | Mistral   | ✅   | StableLM   | ✅   |     |
| DBRX Converted   | 🚀  | GPT-2                 | ✅   | Mixtral   | ✅   | StarCoder2 | ✅   |     |
| Deci             | ✅   | GPT-J                 | ✅   | MOSS      | ✅   | XVERSE     | ✅   |     |
| DeepSeek-V2      | 🚀  | GRIN-MoE              | 🚀  | MPT       | ✅   | Yi         | ✅   |     |
| DeepSeek-V2-Lite | 🚀  | InternLM 1/2.5 | 🚀  | OPT       | ✅   |            |     |     |

## Compatiblity 

We aim for 100% compatibility with models quanted by AutoGPTQ <= 0.7.1 and will consider syncing future compatibilty on a case-by-case basis. 

## Platform/GPU Requirements

GPTQModel is currently Linux only and requires CUDA capability >= 6.0 Nvidia GPU. 

WSL on Windows should work as well. 

ROCM/AMD support will be re-added in a future version after everything on ROCM has been validated. Only fully validated features will be re-added from the original AutoGPTQ repo. 

## Install

### PIP 

```bash
# Include any specific modules needed using brackets. Example: pip install gptqmodel[sglang,vllm,bitblas] --no-build-isolation
pip install gptqmodel --no-build-isolation
```

### Install from source

```bash
# clone repo
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# compile and install
# You can optionally include specific modules like vllm, sglang, or bitblas by adding them in brackets. Example: pip install -vvv --no-build-isolation .[vllm,sglang,bitblas]
pip install -vvv --no-build-isolation .

# If you have `uv` package version 0.1.16 or higher, you can use `uv pip` for potentially better dependency management
# Include modules as needed: uv pip install -vvv --no-build-isolation .[vllm,sglang,bitblas]
uv pip install -vvv --no-build-isolation .
```

### Script installation  
```bash
# You can pass modules as arguments, e.g., --vllm --sglang --bitblas. Example: bash install.sh --vllm --sglang --bitblas
bash install.sh
```



### Quantization and Inference

> warning: this is just a showcase of the usage of basic apis in GPTQModel, which uses only one sample to quantize a much small model, quality of quantized model using such little samples may not good.

Below is an example for the simplest use of `gptqmodel` to quantize a model and inference after quantization:

```py
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

pretrained_model_dir = "facebook/opt-125m"
quant_output_dir = "opt-125m-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
calibration_dataset = [
    tokenizer(
        "The world is a wonderful place full of beauty and love."
    )
]

quant_config = QuantizeConfig(
    bits=4,  # 4-bit
    group_size=128,  # 128 is good balance between quality and performance
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = GPTQModel.from_pretrained(pretrained_model_dir, quant_config)

# quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(calibration_dataset)

# save quantized model
model.save_quantized(quant_output_dir)

# load quantized model to the first GPU
model = GPTQModel.from_quantized(quant_output_dir)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0]))
```

For more advanced features of model quantization, please reference to [this script](https://github.com/ModelCloud/GPTQModel/blob/main/examples/quantization/basic_usage_wikitext2.py)

### How to Add Support for a New Model

Read the [`gptqmodel/models/llama.py`](https://github.com/ModelCloud/GPTQModel/blob/5627f5ffeb3f19b1a2a97e3b6de6fbe668b0dc42/gptqmodel/models/llama.py) code which explains in detail via comments how the model support is defined. Use it as guide to PR for to new models. Most models follow the same pattern.

### Evaluation on Downstream Tasks

You can use tasks defined in `gptqmodel.eval_tasks` to evaluate model's performance on specific down-stream task before and after quantization.

The predefined tasks support all causal-language-models implemented in [🤗 transformers](https://github.com/huggingface/transformers) and in this project.

<details>

<summary>Below is an example to evaluate `EleutherAI/gpt-j-6b` on sequence-classification task using `cardiffnlp/tweet_sentiment_multilingual` dataset:</summary>

```python
from functools import partial

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.eval_tasks import SequenceClassificationTask

MODEL = "EleutherAI/gpt-j-6b"
DATASET = "cardiffnlp/tweet_sentiment_multilingual"
TEMPLATE = "Question:What's the sentiment of the given text? Choices are {labels}.\nText: {text}\nAnswer:"
ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
LABELS = list(ID2LABEL.values())


def ds_refactor_fn(samples):
    text_data = samples["text"]
    label_data = samples["label"]

    new_samples = {"prompt": [], "label": []}
    for text, label in zip(text_data, label_data):
        prompt = TEMPLATE.format(labels=LABELS, text=text)
        new_samples["prompt"].append(prompt)
        new_samples["label"].append(ID2LABEL[label])

    return new_samples


#  model = AutoModelForCausalLM.from_pretrained(MODEL).eval().half().to("cuda:0")
model = GPTQModel.from_pretrained(MODEL, QuantizeConfig())
tokenizer = AutoTokenizer.from_pretrained(MODEL)

task = SequenceClassificationTask(
    model=model,
    tokenizer=tokenizer,
    classes=LABELS,
    data_name_or_path=DATASET,
    prompt_col_name="prompt",
    label_col_name="label",
    **{
        "num_samples": 1000,  # how many samples will be sampled to evaluation
        "sample_max_len": 1024,  # max tokens for each sample
        "block_max_len": 2048,  # max tokens for each data block
        # function to load dataset, one must only accept data_name_or_path as input
        # and return datasets.Dataset
        "load_fn": partial(datasets.load_dataset, name="english"),
        # function to preprocess dataset, which is used for datasets.Dataset.map,
        # must return Dict[str, list] with only two keys: [prompt_col_name, label_col_name]
        "preprocess_fn": ds_refactor_fn,
        # truncate label when sample's length exceed sample_max_len
        "truncate_prompt": False
    }
)

# note that max_new_tokens will be automatically specified internally based on given classes
print(task.run())

# self-consistency
print(
    task.run(
        generation_config=GenerationConfig(
            num_beams=3,
            num_return_sequences=3,
            do_sample=True
        )
    )
)
```

</details>

## Learn More

[tutorials](docs/tutorial) provide step-by-step guidance to integrate `gptqmodel` with your own project and some best practice principles.

[examples](examples/README.md) provide plenty of example scripts to use `gptqmodel` in different ways.

## Supported Evaluation Tasks

Currently, `gptqmodel` supports: `LanguageModelingTask`, `SequenceClassificationTask` and `TextSummarizationTask`; more Tasks will come soon!

### Which kernel is used by default?

GPTQModel will use Marlin, Exllama v2, Triton kernels in that order for maximum inference performance.

# Acknowledgements

* **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh**: for creating [GPTQ](https://github.com/IST-DASLab/gptq) and [Marlin](https://github.com/IST-DASLab/marlin).
* **PanQiWei**: for creation of [AutoGPTQ](https://github.com/autogptq/AutoGPTQ) which this project code is based upon.
* **FXMarty**: for maintaining and support of [AutoGPTQ](https://github.com/autogptq/AutoGPTQ).
* **Qwopqwop200**: for quantization code used in this project adapted from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda).
* **Turboderp**: for releasing [Exllama v1](https://github.com/turboderp/exllama) and [Exllama v2](https://github.com/turboderp/exllamav2) kernels adapted for use in this project.
* **FpgaMiner**: for [GPTQ-Triton](https://github.com/fpgaminer/GPTQ-triton) kernels used in [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda) which is adapted into this project.
