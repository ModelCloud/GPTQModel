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

<p align="center">
    <a href="https://github.com/ModelCloud/GPTQModel/wiki">Documentation</a> ·
    <a href="https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md">Quantization guide</a> ·
    <a href="https://github.com/ModelCloud/GPTQModel/blob/main/docs/README.md">Reference index</a> ·
    <a href="https://github.com/ModelCloud/GPTQModel/releases">Releases</a>
</p>

## Latest News 🗞️🚀

**7.11.0**

- **[MACaQ](https://github.com/ModelCloud/GPTQModel/blob/main/docs/macaq.md):** Native MXFP4 W4A4 quantization for low-precision weights and activations, with CUDA, ROCm, and Apple silicon inference. Includes packed checkpoints and a portable Torch fallback.
- **[NF12](https://github.com/ModelCloud/GPTQModel/wiki/NF12):** Lossless 12-bit compression of BF16 weights reduces source-weight storage during calibration without changing the restored values. Packed inference now includes native Metal support alongside CUDA, HIP, and Torch paths. [MLX](https://github.com/ModelCloud/GPTQModel/blob/main/docs/mlx-runtime.md)
- **Tenstorrent:** Packed GPTQ inference through Torch/XLA and TTNN, multi-card execution, and block-float quantization. Long-context Llama serving adds paged caching and chunked prefill; compatible block-float checkpoints also run on CUDA, ROCm, and CPU. [Guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/tenstorrent.md) · [Serving](https://github.com/ModelCloud/GPTQModel/blob/main/docs/tt-llama-paged-serving.md)
- **Quantization:** Resumable [distributed EXL3](https://github.com/ModelCloud/GPTQModel/blob/main/docs/exl3-durable-execution.md) jobs, broader mixed-bit allocation across [AWQ](https://github.com/ModelCloud/GPTQModel/wiki/AWQ)/[EXL3](https://github.com/ModelCloud/GPTQModel/wiki/EXL3)/[GGUF](https://github.com/ModelCloud/GPTQModel/wiki/GGUF), and [SmoothQuant](https://github.com/ModelCloud/GPTQModel/blob/main/docs/protocol-actions.md) preprocessing improve large-model workflows and quality control. [Quantizer calibration flows](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantizer-moe.md#4-hessian-failure-and-zero-sample-fallback-chain) document the shared Hessian-failure and zero-sample fallback contracts.
- **Inference:** Opt-in attention and MLP projection fusion, faster quantized embedding lookups, and improved small-batch inference. Native backend selection and packed-weight storage are preserved. [Fusion guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/fusion-planner.md)
- **Compat:** Major fixes across dense/MoE calibration, multimodal replay, checkpoint round trips, Transformers model loading, multi-GPU memory use, and ROCm execution. Unsupported combinations fail clearly rather than loading incomplete weights. [Full notes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/recent.md#7110)
- **Security:** Runtime checks for vulnerable PyTorch versions, explicit checkpoint-code trust, stronger checkpoint validation, and bounded server/media requests. [Runtime safety](https://github.com/ModelCloud/GPTQModel/blob/main/docs/torch-runtime-safety-gate.md) · [Serving security](https://github.com/ModelCloud/GPTQModel/blob/main/docs/openai-server-exposure.md)

| Recent release | Highlights |
| --- | --- |
| [7.10.0](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/recent.md#7100) | Ricursive Ripple support, faster decode, checkpoint resharding, and multi-GPU fixes. |
| [7.9.0](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/recent.md#790) | Kernel tuning, MLX improvements, new models, and numerical fixes. |
| [7.8.0](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/recent.md#780) | [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ)/[RCO](https://github.com/ModelCloud/GPTQModel/wiki/GSQ#non-uniform-bit-allocation-rco) training and allocation, expanded model support, and compatibility fixes. |

[Detailed release notes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/recent.md) · [Older changelog](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/archive.md)

## A unified quantization platform 🧩

GPT-QModel brings calibration, quantization, evaluation, conversion, and accelerated
inference into one workflow. Start from a pretrained model, choose a compression
recipe, validate its quality, and save a checkpoint for native inference or a
compatible deployment engine.

### Methods and compression 🎛️

| Method | What it provides |
| --- | --- |
| **[GPTQ](https://github.com/ModelCloud/GPTQModel/wiki/GPTQ)** | Calibration-driven, low-bit weight quantization with mixed precision and broad checkpoint/runtime support. |
| **[AWQ](https://github.com/ModelCloud/GPTQModel/wiki/AWQ)** | Activation-aware weight quantization, including dense and MoE workflows. |
| **[RTN](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md#rtn-w4a16-without-calibration)** | Round-to-nearest weight quantization when a calibration-free baseline is useful. |
| **[ParoQuant](https://github.com/ModelCloud/GPTQModel/wiki/ParoQuant)** | Low-bit weight quantization with learned rotations for accuracy recovery. [Affine fitting and optimization controls](docs/paroquant-optimization.md). |
| **[MACaQ](https://github.com/ModelCloud/GPTQModel/blob/main/docs/macaq.md)** | Calibrated MXFP4 weights and activations, with native W4A4 inference and a W4A16 option. |
| **[QQQ](https://github.com/ModelCloud/GPTQModel/wiki/QQQ)** | INT4 weights with dynamic INT8 activation quantization for W4A8 inference. |
| **[GGUF](https://github.com/ModelCloud/GPTQModel/wiki/GGUF)** | Block-quantized checkpoints for GGUF-oriented workflows and native inference. |
| **[FP8](https://github.com/ModelCloud/GPTQModel/wiki/FP8-and-NVFP4)** | Floating-point weight quantization with scaled E4M3/E5M2 formats. Native inference is weight-only W8A16; [W8A8 is an explicit per-token opt-in](docs/fp8-weight-only-dispatch.md). |
| **[EXL3](https://github.com/ModelCloud/GPTQModel/wiki/EXL3)** | Variable-rate compression with target average bits per weight and mixed-tier allocation. |
| **[VPTQ](https://github.com/ModelCloud/GPTQModel/wiki/VPTQ)** | Vector-codebook quantization for low-bit model compression. |
| **[BitsAndBytes](https://github.com/ModelCloud/GPTQModel/wiki/BitsAndBytes)** | NF4/FP4 and INT8 quantization options. |
| **[NF12](https://github.com/ModelCloud/GPTQModel/wiki/NF12)** | Lossless BF16 weight compression, distinct from lossy low-bit quantization. |

### Control, scale, and quality ⚙️

- **Choose where to spend precision.** Mix bit widths, keep sensitive modules
  unquantized, and use activation-quantization or preprocessing policies where
  supported. Quantization protocols let you organize multi-stage recipes.
- **Keep floating activation searches finite.** NVFP4 block-scale ranking
  normalizes candidate error by an exact power of two so large finite
  activations cannot overflow the comparison and silently return zeros; see
  [NVFP4 candidate ranking](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gptq-w4a-gb10.md#candidate-ranking-and-the-finite-scale-contract).
- **Fit larger models.** Multi-GPU calibration, CPU/disk offload, and checkpointed
  execution help manage memory and recover interrupted jobs. Distributed EXL3
  workflows support durable replay and resume.
- **Recover accuracy.** Use [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ) refinement, [EoRA](https://github.com/ModelCloud/GPTQModel/wiki/EoRA) low-rank compensation, [rotation](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#rotations),
  [smoothing](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#weight-smoothing), and [norm tuning](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#post-quantization-norm-tuning) for supported recipes. [LoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md) training and adapter
  loading support further task adaptation. [Rotation model support](https://github.com/ModelCloud/GPTQModel/blob/main/docs/rotation-model-support.md) lists the exact definitions that opt in. AWQ GSQ uses the
  [packed-code scoring contract](docs/gsq-awq-code-recovery.md) to evaluate exported candidates.
- **Trust the fold.** AWQ divides a scale only into the norm that actually feeds
  a quantized block, including multimodal wrappers and hybrid layer shapes; see
  the [AWQ scaling groups](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-scaling-groups.md) contract. Scale search keeps
  routing in the original-input coordinates, validates replay coverage, and
  restores temporary weights on failure; see [search safety](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-scaling-groups.md#scale-search-safety).
- **Reject unrepresentable AWQ pack scales.** The native GEMM/GEMV packers
  validate the actual FP16 storage cast before replacing module state; see
  [AWQ pack-time scale representability](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-pack-scale-representability.md).
- **Measure before deploying.** Evaluate perplexity and downstream tasks, compare
  against a non-quantized baseline, and test prompts representative of your users.
  Compression settings trade model size, calibration cost, speed, and quality.
- **Preserve causal replay.** Bloom AWQ keeps captured attention context and
  rejects unsafe residual folds; see [Bloom replay safety](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-bloom-replay.md).
- **Replay each calibration document separately.** AWQ token-row replay rebuilds
  a block-diagonal per-document mask and keeps each row's original position;
  additive masks treat `0` as attended. See
  [AWQ calibration aggregation](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-calibration-aggregation.md).
- **Trim padding before the position budget.** Over-long calibration rows are
  measured on their real tokens, not their padded width, so a left-padded row
  cannot lose real tokens to its own padding. See
  [calibration row trimming](https://github.com/ModelCloud/GPTQModel/blob/main/docs/calibration-row-trimming.md).

### Inference and deployment 🚢

Run compatible checkpoints directly with GPT-QModel, integrate with Transformers,
or deploy supported GPTQ/AWQ formats through vLLM and SGLang. Apple silicon has
native MLX quantization and packed inference. Optional projection fusion can
combine supported attention and MLP operations without changing the saved format.

The [model computation DSL design proposal](https://github.com/ModelCloud/GPTQModel/blob/main/docs/model-computation-dsl.md)
describes a shared executable representation for quantization and inference,
with typed tensor interfaces, request-wide resource/state mappings, exact per-layer computation and a path to JIT
compilation independent of Transformers modeling code. This is a design draft;
the compiler and executor are not implemented yet.

[EoRA](https://github.com/ModelCloud/GPTQModel/wiki/EoRA)/[LoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md) adapters, mixed-precision checkpoints, and external engines have their
own compatibility limits; support for a method does not imply support in every engine.

Start with the [quantization guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md).
External runtimes support only specific method/format combinations; check the
[integration and backend tables](https://github.com/ModelCloud/GPTQModel/blob/main/docs/support-matrix.md) before exporting.

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
See the [full hardware and kernel matrix](https://github.com/ModelCloud/GPTQModel/blob/main/docs/support-matrix.md#hardware-and-operating-systems),
[ROCm guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/rocm-rdna4.md), [Tenstorrent guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/tenstorrent.md),
and [Apple silicon guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/mlx-runtime.md).

Tenstorrent-compatible block-float formats also run on CPU without Tenstorrent
hardware. See the [CPU block-float guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/tt-blockfloat-cpu.md).

## Model Support 🤖

Support spans dense, mixture-of-experts (MoE), vision-language, and audio-language
families. The compact list below keeps model coverage visible; the
[complete registry](https://github.com/ModelCloud/GPTQModel/blob/main/docs/supported-models.md) lists exact model-type aliases and
model-specific limitations.

| Families | Supported models and variants |
| --- | --- |
| A | A.X-K2; AfMoE / Trinity; Agnes 3.0 Flash; Apertus 1 / 1.5 |
| B–C | Baichuan; Bailing MoE / Hybrid (LING / RING); Bloom; Brumby; ChatGLM; CodeGen; Cohere 1 / 2 / 2 MoE / Compass (North Mini / Micro Vision) |
| D | DBRX / DBRX Converted; DeciLM; DeepSeek V2 / V3 / V3.2 / V4 / VL / VL2 / OCR2; DiffusionGemma; Dots1 / Dots3 Note; Dream |
| E–F | ERNIE 4.5 / MoE / VL MoE; EXAONE 3 / 4 / 4.5; Falcon / Falcon H1 / Falcon Mamba; FastVLM / LLaVA-Qwen2 |
| G | Gemma 1–4 / 3n / Unified; GLM / GLM4 / GLM4V / GLM5 / OCR / ASR; GPT-2; GPT BigCode; GPT-Neo / GPT-NeoX; GPT-OSS; GPT-J; Granite / Granite SWA / MoE / MoE SWA / MoE Hybrid; GRIN-MoE |
| H | Helium; HRM; Hunyuan V1 / VL / OCR; HY-V3; Hymba |
| I–J | Inkling; Instella; Intern S1 / S2 Preview; InternLM 1 / 2 / 2.5; InternVL 3 / InternVL Chat; Jamba / Jamba2 |
| K | K2-Horizon (Dense / MoVA); Kimi K2 / K2.5 / K3 / Linear; Klear; KRAFTON Raon-1 mini |
| L | Laguna; LFM2 / LFM2 MoE / LFM2-VL; LFM3 / LFM3 Omni / LFM3-VL; LLaDA2 MoE; Llama 1–4 / TinyLlama / Nemotron Ultra; Llama 3.2 VL; LLaVA-1.5; LocateAnything; LongCat Flash; LongLLaMA |
| M | Mage-VL; Marin; MiMo / MiMo V2 / MiMo V2 Flash; MiniCPM / MiniCPM3 / MiniCPM-O / MiniCPM-V; MiniMax M2 / M3-VL; Mistral / Mistral3 / Ministral3; Mixtral; MobileLLM; MOSS; MPT; Muse Glimmer |
| N | Nanbeige 4.2; Nemotron NAS / H / H Puzzle / Omni / Labs Diffusion |
| O | OLMo / OLMo 2 / 3 / Hybrid / OLMoE; OpenVLA; OPT; Ouro; Ovis 1.6 / 2 / 2.5 / 2.6 MoE / 2.6 Next |
| P | PanGu-α; Phi 1–4 / Phi MoE |
| Q | Qwen 1–4 / 3.5 / 3.6 / 3.8 / MoE / Next; Qwen 2 / 2.5 / 3 VL / 3 VL MoE; Qwen 2.5 / 3 Omni; Qwen2-Audio; Qwen-Drive 1.0 |
| R | RecursiveSI Flint; RefinedWeb; Ricursive Ripple |
| S | Seed-OSS; SmolLM3; SmolVLM; Solar Open / Open 2; Spark 2.5; StableLM; StarCoder2 |
| T–V | TeleChat2; Unlimited-OCR; Voxtral |
| X–Z | Xing4.0 (29B-A4B); XVERSE; Yi; Zamba / Zamba2; ZDTaichu 5.0 |

For multimodal checkpoints, support may target the language decoder rather than
every vision/audio component. Kimi K3 / Linear requires the dedicated native-source
calibration workflow; Qwen 3.5 requires Transformers 5.2.0 or newer. Check the
[model notes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/supported-models.md) before starting a large run.

[Browse quantized models on Hugging Face](https://huggingface.co/modelcloud/)

## Install 💾

Use a Python environment with a compatible PyTorch build for your hardware:

```bash
pip install gptqmodel
# Or: uv pip install gptqmodel
```

PyTorch **2.13.x or >= 2.15.0** is required; known-vulnerable versions are
blocked by the [runtime safety check](https://github.com/ModelCloud/GPTQModel/blob/main/docs/torch-runtime-safety-gate.md).
Compiled package backends build and cache extensions on first use (JIT),
so install the compiler and CUDA/ROCm toolkit required by your backend.

See [installation and optional extras](https://github.com/ModelCloud/GPTQModel/wiki/Installation-and-Quickstart)
for source installs and platform setup. Install vLLM or SGLang in a separate
environment aligned with that engine's dependency pins.

## Quick start 🚀

### Inference 💬

Load an existing quantized checkpoint and generate text. GPT-QModel selects a
compatible installed backend for the checkpoint and your device; you can choose
an explicit backend when tuning a deployment.

```python
from gptqmodel import GPTQModel

model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")
tokens = model.generate("Uncovering deep insights begins with", max_new_tokens=64)[0]
print(model.tokenizer.decode(tokens))
```

[Loading and inference](https://github.com/ModelCloud/GPTQModel/wiki/Loading-and-Inference)
covers backend selection, generation, and adapters. Only enable
`trust_remote_code=True` for checkpoint code you trust.

### Quantization 🗜️

This example creates a 4-bit GPTQ checkpoint using 256 streamed C4 samples.
Use calibration text representative of your workload; the dataset and precision
settings affect the quality of the resulting model.

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

The [end-to-end guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md) covers other methods,
calibration, saving, reloading, and validation.
For Apple silicon, use the [native MLX workflow](https://github.com/ModelCloud/GPTQModel/blob/main/docs/mlx-runtime.md).

### What to tune next 🔧

- **Quality versus size:** Start from a supported recipe, then compare bit widths
  and group sizes. Use mixed precision or exclusions for sensitive layers rather
  than assuming one configuration suits the whole model.
- **Calibration:** Choose representative text and enough examples to cover your
  use case. MoE and multimodal models may need specialized data preparation.
- **Memory and recovery:** Configure multi-GPU placement and offload before a
  large run, and use checkpoint/resume where supported.
- **Deployment:** Reload the saved checkpoint, evaluate it, and verify its format
  in the intended engine. A successful native load is not an external-engine
  compatibility guarantee.

See the [configuration reference](https://github.com/ModelCloud/GPTQModel/wiki/QuantizeConfig-Reference)
and [quantization recipes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md) for the available controls.

## Guides and reference 🧭

| Task | Documentation |
| --- | --- |
| Choose precision, format, and backend | [Quantization recipes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md) · [Support matrix](https://github.com/ModelCloud/GPTQModel/blob/main/docs/support-matrix.md) |
| Configure mixed precision and advanced options | [Configuration reference](https://github.com/ModelCloud/GPTQModel/wiki/QuantizeConfig-Reference) · [Protocols](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization_protocol.md) |
| Quantize large or MoE models | [Multi-GPU and memory](https://github.com/ModelCloud/GPTQModel/blob/main/docs/multi_gpu_quantization.md) · [MoE and multimodal](https://github.com/ModelCloud/GPTQModel/wiki/Multimodal-and-MoE-Models) |
| Resume a job | [Checkpoint and resume](https://github.com/ModelCloud/GPTQModel/blob/main/checkpoint.md) · [Distributed EXL3](https://github.com/ModelCloud/GPTQModel/blob/main/docs/exl3-durable-execution.md) |
| Recover accuracy or fine-tune | [EoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/eora/README.md) · [PEFT/LoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md) · [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ) · [RCO](https://github.com/ModelCloud/GPTQModel/wiki/GSQ#non-uniform-bit-allocation-rco) |
| Evaluate quality and performance | [Evaluation](https://github.com/ModelCloud/GPTQModel/wiki/Evaluation) · [Benchmark methodology](https://github.com/ModelCloud/GPTQModel/blob/main/docs/benchmark-methodology.md) |
| Serve an OpenAI-compatible API | [Serving guide](https://github.com/ModelCloud/GPTQModel/wiki/Serving) · [Security](https://github.com/ModelCloud/GPTQModel/blob/main/docs/openai-server-exposure.md) |
| Add a model or inspect its plan | [Module-plan guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/module-plan-guide.md) · [Leaf coverage](https://github.com/ModelCloud/GPTQModel/blob/main/docs/module-tree-leaf-coverage.md) · [Developer guide](https://github.com/ModelCloud/GPTQModel/wiki/Developer-Guide) |
| Load legacy remote-code models on Transformers 5 | [Transformers 5 compatibility](https://github.com/ModelCloud/GPTQModel/blob/main/docs/transformers-5-compat.md#k-native-loads-run-the-pre-load-hook-one-legacy-generate-adapter-issue-4121-items-2-3) |
| Diagnose runtime issues | [Troubleshooting](https://github.com/ModelCloud/GPTQModel/wiki/Troubleshooting-and-FAQ) · [Concurrency](https://github.com/ModelCloud/GPTQModel/blob/main/docs/concurrency.md) · [Thread caps](https://github.com/ModelCloud/GPTQModel/blob/main/docs/thread-caps.md) |
| Reuse cross-layer KV (Hymba) | [Shared KV cache contract](https://github.com/ModelCloud/GPTQModel/blob/main/docs/looper-shared-kv-cache.md) |

Kernel contracts, implementation flows, and focused technical notes are in the
[documentation index](https://github.com/ModelCloud/GPTQModel/blob/main/docs/README.md), not repeated on this page.

## Quality: GPTQ 4-bit evaluation 🏆

Quantization quality depends on the source model, recipe, and calibration data.
The Vortex collection provides GPTQ examples; the charts below illustrate selected
evaluations, not a guarantee for every supported model or method.

[ModelCloud quantized Vortex models](https://huggingface.co/collections/ModelCloud/vortex-673743382af0a52b2a8b9fe2)

<img src=https://github.com/user-attachments/assets/c1b89394-f8f6-44e5-9949-bef15a124723 width="51%"> <img src=https://github.com/user-attachments/assets/23901236-10c5-4435-ac2f-06cf2e097f1e width="47%">

## Organizations and projects using GPT-QModel 🌐

Selected public references include Hugging Face, Intel, NVIDIA, Alibaba Cloud,
Fujitsu, and IBM Research.
This is not an exhaustive customer list.

## Contributors and sponsors 🤝

### Contributions welcome 🛠️

We welcome all contributors, including people working with human-guided AI
agents. Code, model and hardware support, documentation, bug reports, reviews,
and testing all help move GPT-QModel forward. See the
[developer guide](https://github.com/ModelCloud/GPTQModel/wiki/Developer-Guide)
to get started.

### Support ongoing development 💛

Supporting a wide range of hardware and models requires continuous CI testing,
accuracy validation, and compatibility work. That takes substantial compute and
AI-agent tokens. Contributions of compute, model/API credits, or financial
support of any size help sustain project maintenance and future development.

Interested in sponsoring? [Message Qubitium on X](https://x.com/Qubitium).
If you find GPT-QModel useful, sending even a couple of dollars through X is
also appreciated. Ongoing sponsors will be credited on this README page.

## License 📜

GPT-QModel is licensed under [Apache-2.0](https://github.com/ModelCloud/GPTQModel/blob/main/LICENSE). The optional, JIT-loaded
Swordfish kernel is [AGPL-3.0-or-later](https://github.com/ModelCloud/GPTQModel/blob/main/licenses/SWORDFISH).

## Attributions and citations 📚

GPT-QModel builds on the research and open-source projects below. Please cite
GPT-QModel and the methods you use; full author lists, BibTeX, and attribution
details are available in the [citation reference](https://github.com/ModelCloud/GPTQModel/blob/main/docs/citations.md).

<details>
<summary>Cite GPT-QModel (BibTeX)</summary>

```bibtex
@misc{qubitium2024gptqmodel,
  author = {ModelCloud.ai and qubitium@modelcloud.ai},
  title = {GPT-QModel},
  year = {2024},
  howpublished = {\url{https://github.com/ModelCloud/GPTQModel}}
}
```

</details>

| Work | Authors / project | Paper or reference |
| --- | --- | --- |
| GPT-QModel | ModelCloud.ai and contributors | [Project](https://github.com/ModelCloud/GPTQModel) |
| [GPTQ](https://github.com/ModelCloud/GPTQModel/wiki/GPTQ) | Elias Frantar et al., IST-DASLab | [Paper](https://arxiv.org/abs/2210.17323) |
| [AWQ](https://github.com/ModelCloud/GPTQModel/wiki/AWQ) | Ji Lin et al. | [Paper](https://arxiv.org/abs/2306.00978) |
| [ParoQuant](https://github.com/ModelCloud/GPTQModel/wiki/ParoQuant) | Yesheng Liang et al., Z-Lab | [Paper](https://openreview.net/forum?id=1USeVjsKau) · [Code](https://github.com/z-lab/paroquant) |
| [MACaQ](https://github.com/ModelCloud/GPTQModel/blob/main/docs/macaq.md) | Lorenz K. Muller et al., Huawei CSL | [Paper](https://arxiv.org/abs/2611.10649) · [Code](https://github.com/huawei-csl/MACaQ) |
| [QQQ](https://github.com/ModelCloud/GPTQModel/wiki/QQQ) | Ying Zhang et al., Meituan | [Paper](https://arxiv.org/abs/2406.09904) |
| [VPTQ](https://github.com/ModelCloud/GPTQModel/wiki/VPTQ) | Yifei Liu et al., Microsoft | [Paper](https://arxiv.org/abs/2409.17066) · [Code](https://github.com/microsoft/VPTQ) |
| [GGUF](https://github.com/ModelCloud/GPTQModel/wiki/GGUF) / llama.cpp | Georgi Gerganov and ggml-org contributors | [Project](https://github.com/ggml-org/llama.cpp) |
| [ExLlama V3 / EXL3](https://github.com/ModelCloud/GPTQModel/wiki/EXL3) | turboderp and contributors | [Project](https://github.com/turboderp-org/exllamav3) |
| [NF12 / NyoomFloat12](https://github.com/ModelCloud/GPTQModel/wiki/NF12) | Liberman et al., Together AI | [Research and codec](https://github.com/dphnAI/nf12) |
| [EoRA](https://github.com/ModelCloud/GPTQModel/wiki/EoRA) | Shih-Yang Liu et al., NVIDIA | [Paper](https://arxiv.org/abs/2410.21271) |
| [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ) | Alireza Dadgarnia et al., IST-DASLab | [Paper](https://arxiv.org/abs/2604.18556) · [Code](https://github.com/IST-DASLab/GSQ) |
| [RCO](https://github.com/ModelCloud/GPTQModel/wiki/GSQ#non-uniform-bit-allocation-rco) | Michael Helcig and Dan Alistarh, IST-DASLab | [Paper](https://arxiv.org/abs/2605.00649) · [Code](https://github.com/IST-DASLab/RCO) |
| [GPTAQ](https://github.com/ModelCloud/GPTQModel/wiki/GPTQ) | Yuhang Li et al., Yale Intelligent Computing Lab | [Paper](https://arxiv.org/abs/2504.02692) |
| [GAR](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#group-aware-reordering-gar) | T. Gafni, A. Karnieli, and Y. Hanani, Intel | [Paper](https://arxiv.org/abs/2505.14638) |
| [FOEM](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#foem-and-gptaq-correction) | Xingyu Zheng et al. | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40123) |
| [Norm Tweaking](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#post-quantization-norm-tuning) | Liang Li, Qingyuan Li, Bo Zhang, and Xiangxiang Chu | [Paper](https://arxiv.org/abs/2309.02784) |
| [Fast Hadamard Transform](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#hadamard-dimensions-and-implementation-selection) | Tri Dao, Dao-AILab | [Code](https://github.com/Dao-AILab/fast-hadamard-transform) |
| [Marlin](https://github.com/ModelCloud/GPTQModel/wiki/Kernel-Backends-Reference#marlin-and-caller-owned-scratch) | Elias Frantar et al. | [Paper](https://arxiv.org/abs/2408.11743) |
| [Machete](https://github.com/ModelCloud/GPTQModel/wiki/Kernel-Backends-Reference#machete) | vLLM contributors | [Code](https://github.com/vllm-project/vllm/tree/main/csrc/quantization/machete) |
| [Swordfish](https://github.com/ModelCloud/GPTQModel/wiki/Kernel-Backends-Reference#swordfish) | AlpinDale | [Article](https://blog.alpindale.net/posts/swordfish/) |
