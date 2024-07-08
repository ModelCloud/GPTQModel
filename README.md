<h1 align="center">GPTQModel</h1>
<p align="center">An easy-to-use LLM quantization and inference toolkit based on GPTQ algorithm (weight-only quantization).</p>
<p align="center">
    <a href="https://github.com/ModelCloud/GPTQModel/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/GPTQModel.svg">
    </a>
    <a href="https://pypi.org/project/model-gptq/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/gptq-model">
    </a>
</p>

## News
* 07/08/2024 🚀🚀 [v0.9.6](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.6): [Intel/AutoRound](https://github.com/intel/auto-round) QUANT_METHOD support added for a potentially higher quality quantization with `lm_head` module quantization support for even more vram reduction: format export to `FORMAT.GPTQ` for max inference compatibility.

* 07/05/2024 🚀🚀 [v0.9.5](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.5): [Intel/QBits](https://github.com/intel/intel-extension-for-transformers) support added for [2,3,4,8] bit quantization/inference on CPU. Cuda kernels have been fully deprecated in favor of Exllama(v1/v2)/Marlin/Triton.

* 07/03/2024 🚀 [v0.9.4](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.4): HF Transformers integration added and bug fixed Gemma 2 support.

* 07/02/2024 🚀 [v0.9.3](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.3): Added Gemma 2 support, faster PPL calculations on gpu, and more code/arg refractor.

* 06/30/2024 🚀 [v0.9.2](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.2): Added auto-padding of model in/out-features for exllama and exllama v2. 
Fixed quantization of OPT and DeepSeek V2-Lite models. Fixed inference for DeepSeek V2-Lite.

* 06/29/2024 🚀🚀🚀 [v0.9.1](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.1): With 3 new models (DeepSeek-V2, DeepSeek-V2-Lite, DBRX Converted), BITBLAS new format/kernel, proper batching of calibration dataset resulting > 50% quantization speedup, security hash check of loaded model weights, tons of refractor/usability improvements, bugs fixes and much more.

* 06/20/2924 ✨ GPTQModel [v0.9.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v0.9.0): Thanks for all the work from ModelCloud team and the opensource ML community for their contributions!

## Mission Statement

We want GPTQModel to be highly focused on GPTQ based quantization and target inference compatibility with HF Transformers, vLLM, and SGLang. 

## How is GPTQModel different from AutoGPTQ?

GPTQModel is an opinionated fork/refactor of AutoGPTQ with latest bug fixes, more model support, faster quant inference, faster quantization, better quants (as measured in PPL) and a pledge from the ModelCloud team and that we, along with the open-source ML community, will take every effort to bring the library up-to-date with latest advancements, model support, and bug fixes.

We will backport bug fixes to AutoGPTQ on a case-by-case basis.

## Major Changes (Advantages) vs AutoGPTQ

* 🚀 Added `Gemma 2` Model Support
* 🚀 Added `DeepSeek-V2` Model Support
* 🚀 Added `DeepSeek-V2-Lite` Model Support
* 🚀 Added `ChatGLM` Model Support
* 🚀 Added `MiniCPM` Model Support
* 🚀 Added `Phi-3` Model Support
* 🚀 Added `Qwen2MoE` Model Support
* 🚀 Added `DBRX` Model Support (Converted Model)
* 🚀 [Intel/AutoRound](https://github.com/intel/auto-round) QUANT_METHOD support added for a potentially higher quality quantization with `lm_head` module quantization support for even more vram reduction: format export to `FORMAT.GPTQ` for max inference compatibility.
* 🚀 [Intel/QBits](https://github.com/intel/intel-extension-for-transformers) support added for [2,3,4,8] bit quantization/inference on CPU.
* 🚀 [BITBLAS](https://github.com/microsoft/BitBLAS) format/inference support from Microsoft
* 🚀`Sym=False` Support. AutoGPTQ has unusable `sym=false`. (Re-quant required)
* 🚀`lm_head` module quant inference support for further VRAM reduction. 
* 🚀 Faster quantization: More than 50% faster for TinyLlama + 4090 with batching and large calibration dataset.
* 🚀 Better quality quants as measured by PPL. (Test config: defaults + `sym=True` + `FORMAT.GPTQ`, TinyLlama)
* 🚀 Model weights sharding support
* 🚀 Security: hash check of model weights on load
* 🚀 Over 50% faster PPL calculations (OPT model)
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

## Roadmap (Target Date: July 2024):

* <del>`lm_head` quantization support by integrating with Intel/AutoRound.</del> Completed
* Customizable callback in Per-Layer quantization.
* <del>Add Qbits (cpu inference) support from Intel/Qbits.</del> Completed
* Add back ROCM/AMD support once everything is validated.
* Store quant loss stat and apply diffs to new quant for quality control.
* Add Tests for every single supported model.


## Model Support ( 🚀 GPTQModel only )

| Model          |    |                  |    |           |    |            |    |
|----------------|----|------------------|----|-----------|----|------------|----|
| Baichuan       | ✅  | DeepSeek-V2-Lite | 🚀 | Llama     | ✅  | Phi/Phi-3  | 🚀 |
| Bloom          | ✅  | Falon            | ✅  | LongLLaMA | ✅  | Qwen       | ✅  |
| ChatGLM        | 🚀 | Gemma 2          |  🚀  | MiniCPM   | 🚀 | Qwen2MoE   | 🚀 |
| CodeGen        | ✅  | GPTBigCod        | ✅  | Mistral   | ✅  | RefinedWeb | ✅  |
| Cohere         | ✅  | GPTNeoX          | ✅  | Mixtral   | ✅  | StableLM   | ✅  |
| DBRX Converted | 🚀 | GPT-2            | ✅  | MOSS      | ✅  | StarCoder2 | ✅  |
| Deci           | ✅  | GPT-J            | ✅  | MPT       | ✅  | XVERSE     | ✅  |
| DeepSeek-V2    | 🚀 | InternLM         | ✅  | OPT       | ✅  | Yi         | ✅  |

## Compatiblity 

We aim for 100% compatibility with models quanted by AutoGPTQ <= 0.7.1 and will consider syncing future compatibilty on a case-by-case basis. 

## Platform/GPU Requirements

GPTQModel is currently Linux only and requires CUDA capability >= 6.0 Nvidia GPU. 

WSL on Windows should work as well. 

ROCM/AMD support will be re-added in a future version after everything on ROCM has been validated. Only fully validated features will be re-added from the original AutoGPTQ repo. 

## Install

### Install from source

```bash
# clone repo
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# compile and install
pip install -vvv --no-build-isolation .

# If you have `uv` package version 0.1.16 or higher, you can use `uv pip` for potentially better dependency management
uv pip install -vvv --no-build-isolation .
```

### Script installation  
```bash
bash install.sh
```

### PIP (PENDING RELEASE) 

```bash
pip install gptq-model --no-build-isolation
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

For more advanced features of model quantization, please reference to [this script](examples/quantization/quant_with_alpaca.py)

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
