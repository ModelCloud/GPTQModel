<h1 align="center">AutoGPTQ-NEXT</h1>
<p align="center">An easy-to-use LLM quantization and inference toolkit based on GPTQ algorithm (weight-only quantization).</p>
<p align="center">
    <a href="https://github.com/Qubitium/AutoGPTQ/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/Qubitium/AutoGPTQ.svg">
    </a>
    <a href="https://pypi.org/project/auto-gptq/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/auto-gptq-next">
    </a>
</p>

## News

- 2024-06-XX - (News)   🤗 PENDING

## How is AutoGPTQ-NEXT different from AutoGPTQ?

AutoGPTQ-NEXT is an opinionated fork/refractor of AugtoGPTQ with latest bug fixes applied, new features, better/latest model support, and a pledge from the ModelCloud.ai team and that we, along with the open-source ML community, will take every effort to bring the library up-to-date with latest advancements, model support, and bug fixes.

## Mission Statement

We want AutoGPTQ-NEXT to be highy focused on GPTQ based quantization and target inference compatibility with HF Transformers, vLLM, and SGLang. 

## Major Changes vs AutoGPTQ

* `Sym=False` Support. AutoGPTQ main has broken `sym=false`.
* `lm_head` module quantized inference support for further vram reduction.
* ChatGLM Model Support.
* Better defaults resulting in faster inference.
* Better default PPL with tweaked internal code (Result may vary depending on calibration set and gpu usage).
* Alert users of sub-optimal calibration data. Almost all new-users get this part horribly wrong.
* Removed non-working, partially working, or fully deprecated features: Peft, ROCM, AWQ Gemm inference, Triton v1 (replaced by v2), Fused Attention (Replaced by Marlin/Exllama).
* Fixed Packing Performance regression on high core-count systems.
* Thousands of lines of refractor/cleanup.
* Debloated 271K lines of which 250K was caused by a single dataset used only by an example. Move dataset to HF.
* Shorter and more concise public api/internal vars. No need to mimic HF style for verbose class names. 
* Complete tests with every feature and model tested. Everything that does not pass tests will be removed from repo. We want quality over quantity.

## Roadmap (Target Date: July 2024):

* DBRX support.
* `lm_head` quantization support by integrating with Intel/Autoround.
* Customizable callback in Per-Layer quantization.
* Add Qbits (cpu inference) support from Intel/Qbits.
* Add back ROCM/AMD support once verything is validated.
* Store quant loss stat and apply diffs to new quant for quality control.
* Add CI workflow for PRs.
* Add Tests for every single supported model.


## Model Support 

| Model            |    |              |    |              |    |                  |    |
|------------------|----|--------------|----|--------------|----|------------------|----|
| baichuan         | ✅ | gpt_bigcode  | ✅ | mixtral     | ✅ | RefinedWebModel  | ✅ |
| bloom            | ✅ | gpt_neox     | ✅ | moss        | ✅ | stablelm_epoch   | ✅ |
| chatglm          | ✅ | gpt2         | ✅ | mpt         | ✅ | starcoder2       | ✅ |
| codegen          | ✅ | gptj         | ✅ | opt         | ✅ | xverse           | ✅ |
| cohere           | ✅ | internlm     | ✅ | phi         | ✅ | Yi               | ✅ |
| deci             | ✅ | llama        | ✅ | qwen        | ✅ |                  |    |
| falcon           | ✅ | longllama    | ✅ | qwen2       | ✅ |                  |    |
| gemma            | ✅ | mistral      | ✅ | RefinedWeb  | ✅ |                  |    |


## Platform Support
AutoGPTQ-NEXT is currently Linux only and requires Torch/Cuda capable GPU from NVIDIA. WSL on Windows should work as well. ROCM/AMD support will be re-added in a furture version after everything on ROCM has been validated. Only fully validated features will be re-added from the original AutoGPTQ repo. 

## Install

AutoGPTQ-NEXT is available for Linux only. You can install the latest stable release of AutoGPTQ from pip with pre-built wheels:

| CUDA version | Installation                                                                                      | Built against PyTorch |
|-------------------|---------------------------------------------------------------------------------------------------|-----------------------|
| CUDA 12.1         | `pip install auto-gptq-next --no-build-isolation`                                                                            | 2.3.1+cu121           |


AutoGPTQ-NEXT does not support [Maxwell or lower](https://qiita.com/uyuni/items/733a93b975b524f89f46) GPUs.

### Install from source

Clone repo:
```bash
git clone https://github.com/Qubitium/AutoGPTQ-NEXT.git && cd AutoGPTQ-NEXT
```

Compile:
```bash
pip install -vvv --no-build-isolation -e .
```

### Quantization and Inference

> warning: this is just a showcase of the usage of basic apis in AutoGPTQ-NEXT, which uses only one sample to quantize a much small model, quality of quantized model using such little samples may not good.

Below is an example for the simplest use of `auto_gptq_next` to quantize a model and inference after quantization:

```py
from transformers import AutoTokenizer
from auto_gptq_next import AutoGPTQNext, QuantizeConfig

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
model = AutoGPTQNext.from_pretrained(pretrained_model_dir, quant_config)

# quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(calibration_dataset)

# save quantized model
model.save_quantized(quant_output_dir)

# load quantized model to the first GPU
model = AutoGPTQNext.from_quantized(quant_output_dir, device="cuda:0")

# download quantized model from Hugging Face Hub and load to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0")

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq_next is", return_tensors="pt").to(model.device))[0]))

# or you can also use pipeline
# pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline("auto-gptq is")[0]["generated_text"])
```

For more advanced features of model quantization, please reference to [this script](examples/quantization/quant_with_alpaca.py)

### How to Add Support for a New Model

Read the `auto_gptq_next/models/llama.py` code which explains in detail via comments how the model support is defined. Use it as guide to PR for to new models. Most models follow the same pattern.

### Evaluation on Downstream Tasks

You can use tasks defined in `auto_gptq_next.eval_tasks` to evaluate model's performance on specific down-stream task before and after quantization.

The predefined tasks support all causal-language-models implemented in [🤗 transformers](https://github.com/huggingface/transformers) and in this project.

<details>

<summary>Below is an example to evaluate `EleutherAI/gpt-j-6b` on sequence-classification task using `cardiffnlp/tweet_sentiment_multilingual` dataset:</summary>

```python
from functools import partial

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from auto_gptq_next import AutoGPTQNext, QuantizeConfig
from auto_gptq_next.eval_tasks import SequenceClassificationTask

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
model = AutoGPTQNext.from_pretrained(MODEL, QuantizeConfig())
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

[tutorials](docs/tutorial) provide step-by-step guidance to integrate `auto_gptq_next` with your own project and some best practice principles.

[examples](examples/README.md) provide plenty of example scripts to use `auto_gptq_next` in different ways.

## Supported Evaluation Tasks

Currently, `auto_gptq_next` supports: `LanguageModelingTask`, `SequenceClassificationTask` and `TextSummarizationTask`; more Tasks will come soon!

### Which kernel is used by default?

AutoGPTQ-NEXT will use Marlin, Exllama v2, Triton/CUDA kernels in that order for maximum inference performance.

## Acknowledgement

* **PanQiWei** and **FXMarty** for their creation and support of AutoGPTQ of which this project is based upon.
* **Elias Frantar**, **Saleh Ashkboos**, **Torsten Hoefler** and **Dan Alistarh** for **GPTQ**/**Marlin** algorithm and [code](https://github.com/IST-DASLab/gptq) [Marlin kernel](https://github.com/IST-DASLab/marlin).
* **qwopqwop200**, for quantization code used in this project adapted from [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda).
* **turboderp**, for releasing [Exllama](https://github.com/turboderp/exllama) and [Exllama v2](https://github.com/turboderp/exllamav2) kernels adapted for use in this project.
* **FPGAMiner**, for triton kernels used in GPTQ-for-LLaMa which is adapted into this project.
