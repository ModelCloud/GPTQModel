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
<p align="center"><strong>An extensible platform for LLM quantization, validation, and deployment.</strong><br>Compress, evaluate, and run models across supported GPUs, accelerators, and CPUs.</p>
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

[Documentation](https://github.com/ModelCloud/GPTQModel/wiki) · [Quantization guide](docs/quantization-guide.md) · [Reference index](docs/README.md) · [Releases](https://github.com/ModelCloud/GPTQModel/releases)

## Latest News 🗞️🚀

**7.11.0**

- **MACaQ:** Native MXFP4 W4A4 quantization with CUDA, ROCm, and Apple silicon inference. [Guide](docs/macaq.md)
- **NF12:** Lossless BF16 weight compression and native Metal inference. [NF12](https://github.com/ModelCloud/GPTQModel/wiki/NF12) · [MLX](docs/mlx-runtime.md)
- **Tenstorrent:** Expanded hardware inference, block-float quantization, and long-context Llama serving. [Guide](docs/tenstorrent.md)
- **Quantization:** Resumable distributed EXL3, broader mixed-bit allocation, and SmoothQuant preprocessing. [EXL3](docs/exl3-durable-execution.md) · [SmoothQuant](docs/protocol-actions.md)
- **Inference:** Opt-in projection fusion and faster quantized inference. [Fusion guide](docs/fusion-planner.md)
- **Compat:** Major calibration, checkpoint, model-loading, multi-GPU, ROCm, and security fixes. [Full notes](docs/changelog/recent.md#7110)

| Recent release | Highlights |
| --- | --- |
| [7.10.0](docs/changelog/recent.md#7100) | Ricursive Ripple support, faster decode, checkpoint resharding, and multi-GPU fixes. |
| [7.9.0](docs/changelog/recent.md#790) | Kernel tuning, MLX improvements, new models, and numerical fixes. |
| [7.8.0](docs/changelog/recent.md#780) | GSQ/RCO training and allocation, expanded model support, and compatibility fixes. |

[Detailed release notes](docs/changelog/recent.md) · [Older changelog](docs/changelog/archive.md)

## A unified quantization platform

GPT-QModel provides one workflow for calibration, quantization, evaluation,
conversion, and accelerated inference.

- **Methods:** GPTQ, AWQ, RTN, ParoQuant, MACaQ, QQQ, GGUF, FP8, EXL3, VPTQ, NF12, and BitsAndBytes.
- **Control:** Mixed precision, per-module exclusions, activation quantization, and staged quantization protocols.
- **Scale:** Multi-GPU calibration, memory offload, and resumable quantization.
- **Quality:** GSQ refinement, EoRA recovery, LoRA training, and post-quantization evaluation.
- **Deployment:** Native inference plus compatible Transformers, vLLM, SGLang, and MLX integrations.

Start with the [quantization guide](docs/quantization-guide.md).
External runtimes support only specific method/format combinations; check the
[integration and backend tables](docs/support-matrix.md) before exporting.

## Platform and HW Support 🖥️

| Platform | Hardware | Runtime overview |
| --- | --- | --- |
| Linux | NVIDIA GPUs | CUDA kernels; Turing or newer, with backend-specific minimums |
| Linux | AMD GPUs | ROCm 10.1+; architecture-specific Torch/Triton/HIP paths |
| Linux | Tenstorrent | Torch/XLA/PJRT and native TTNN |
| Linux | Huawei Ascend | Native Torch through `torch-npu` / CANN |
| Linux | Intel XPU | Torch and TorchFused |
| Linux | Intel/AMD CPUs | Torch and optimized x86 kernels |
| macOS | Apple silicon | Native MLX/Metal quantization and inference; MPS/Torch |
| Windows 11 | NVIDIA GPUs / CPU | Torch |

Support depends on the chosen method, backend, dtype, and device architecture.
See the [full hardware and kernel matrix](docs/support-matrix.md#hardware-and-operating-systems),
[ROCm guide](docs/rocm-rdna4.md), [Tenstorrent guide](docs/tenstorrent.md),
and [Apple silicon guide](docs/mlx-runtime.md).

Tenstorrent-compatible block-float formats also run on CPU without Tenstorrent
hardware. See the [CPU block-float guide](docs/tt-blockfloat-cpu.md).

## Model Support 🤖

Dense, MoE, and multimodal families include Llama, Qwen, DeepSeek, Gemma,
GLM, Mistral, and many more. See the [complete model registry](docs/supported-models.md)
for registered model types and model-specific limitations.

[Browse quantized models on Hugging Face](https://huggingface.co/modelcloud/)

## Install 💾

Use a Python environment with a compatible PyTorch build for your hardware:

```bash
pip install gptqmodel
# Or: uv pip install gptqmodel
```

PyTorch **2.13.x or >= 2.15.0** is required; known-vulnerable versions are
blocked by the [runtime safety check](docs/torch-runtime-safety-gate.md).
Compiled package backends build and cache extensions on first use (JIT),
so install the compiler and CUDA/ROCm toolkit required by your backend.

See [installation and optional extras](https://github.com/ModelCloud/GPTQModel/wiki/Installation-and-Quickstart)
for source installs and platform setup. Install vLLM or SGLang in a separate
environment aligned with that engine's dependency pins.

## Quick start

### Inference

```python
from gptqmodel import GPTQModel

model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")
tokens = model.generate("Uncovering deep insights begins with", max_new_tokens=64)[0]
print(model.tokenizer.decode(tokens))
```

[Loading and inference](https://github.com/ModelCloud/GPTQModel/wiki/Loading-and-Inference)
covers backend selection, generation, and adapters. Only enable
`trust_remote_code=True` for checkpoint code you trust.

### Quantization

A minimal GPTQ example; use representative calibration data for your workload:

```python
from datasets import load_dataset
from gptqmodel import GPTQConfig, GPTQModel

dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
    streaming=True,
)
calibration = [row["text"] for row in dataset.take(256)]

model = GPTQModel.load(
    "Qwen/Qwen3-0.6B", GPTQConfig(bits=4, group_size=128),
)
model.quantize(calibration, batch_size=1)
model.save("Qwen3-0.6B-GPTQ-4bit")
```

The [end-to-end guide](docs/quantization-guide.md) covers other methods,
calibration, saving, reloading, and validation.
For Apple silicon, use the [native MLX workflow](docs/mlx-runtime.md).

## Guides and reference

| Task | Documentation |
| --- | --- |
| Choose precision, format, and backend | [Quantization recipes](docs/quantization-guide.md) · [Support matrix](docs/support-matrix.md) |
| Configure mixed precision and advanced options | [Configuration reference](https://github.com/ModelCloud/GPTQModel/wiki/QuantizeConfig-Reference) · [Protocols](docs/quantization_protocol.md) |
| Quantize large or MoE models | [Multi-GPU and memory](docs/multi_gpu_quantization.md) · [MoE and multimodal](https://github.com/ModelCloud/GPTQModel/wiki/Multimodal-and-MoE-Models) |
| Resume a job | [Checkpoint and resume](checkpoint.md) · [Distributed EXL3](docs/exl3-durable-execution.md) |
| Recover accuracy or fine-tune | [EoRA](docs/eora/README.md) · [PEFT/LoRA](docs/peft.md) · [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ) |
| Evaluate quality and performance | [Evaluation](https://github.com/ModelCloud/GPTQModel/wiki/Evaluation) · [Benchmark methodology](docs/benchmark-methodology.md) |
| Serve an OpenAI-compatible API | [Serving guide](https://github.com/ModelCloud/GPTQModel/wiki/Serving) · [Security](docs/openai-server-exposure.md) |
| Add a model or inspect its plan | [Module-plan guide](docs/module-plan-guide.md) · [Developer guide](https://github.com/ModelCloud/GPTQModel/wiki/Developer-Guide) |
| Diagnose runtime issues | [Troubleshooting](https://github.com/ModelCloud/GPTQModel/wiki/Troubleshooting-and-FAQ) · [Concurrency](docs/concurrency.md) |

Kernel contracts, implementation flows, and focused technical notes are in the
[documentation index](docs/README.md), not repeated on this page.

## Quality: GPTQ 4-bit evaluation 🏆

[ModelCloud quantized Vortex models](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2)

<img src=https://github.com/user-attachments/assets/c1b89394-f8f6-44e5-9949-bef15a124723 width="51%"> <img src=https://github.com/user-attachments/assets/23901236-10c5-4435-ac2f-06cf2e097f1e width="47%">

## Organizations and projects using GPT-QModel 🌐

Selected public references include Hugging Face, Intel, NVIDIA, and Alibaba Cloud.
This is not an exhaustive customer list.

## Attributions and citations

See [attributions and BibTeX citations](docs/citations.md) for the algorithms,
kernels, and projects used by GPT-QModel.

## License 📜

GPT-QModel is licensed under [Apache-2.0](LICENSE). The optional, JIT-loaded
Swordfish kernel is [AGPL-3.0-or-later](licenses/SWORDFISH).
