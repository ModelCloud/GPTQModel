# Changelog archive

Release highlights before 5.8.0 are kept here so the project README can focus on the current platform and recent releases. Full release artifacts remain available on the [GitHub releases page](https://github.com/ModelCloud/GPTQModel/releases).

## 2026 and 2025

* **5.7.0** — Added `MoE.Routing` controls, unified AWQ symmetry configuration, and Qwen3/Exaone compatibility fixes.
* **5.6.2 / 5.6.0** — Improved install and multi-architecture compatibility; added AMX, AVX2, and AVX512 CPU kernels and broader Transformers, PEFT, and Optimum support.
* **5.4.0 / 5.2.0** — Added Intel CPU/XPU AWQ kernels, AWQ `torch.compile` support, balanced VRAM strategy, and more MoE support.
* **5.0.0** — Introduced data-parallel MoE quantization, disk offload by default, accelerated packing, and production AWQ support.
* **4.2.5 / 4.2.0 / 4.1.0 / 4.0.0** — Added `act_group_aware`, FailSafe controls, GAR, Python free-threading, and support for Qwen3, Llama 4, GPT-OSS, Gemma3, and other model families.
* **3.0.0 / 2.2.0 / 2.1.0 / 2.0.0** — Added GPTQ v2 experiments, QQQ, staged quantization internals, GSM8K Platinum and MMLU-Pro evaluation, and expanded multimodal/model support.
* **1.9.0 through 1.5.0** — Improved tokenizer and `lm_head` handling, flexible packing, buffered forward, compilation, MLX support, AMD ROCm, OpenAI-compatible serving, and multimodal quantization.

## 2024

* **1.4.5 through 1.0.0** — Added Windows and Apple Silicon support, visual-language model calibration, EvalPlus integration, dynamic quantization controls, Intel XPU support, new loader/save APIs, and PyPI distribution improvements.
* **0.9.11 through 0.9.0** — Added per-module dynamic quantization, Marlin and BitBLAS improvements, Llama 3.1 and Gemma 2 support, vLLM/SGLang runtime integration, AutoRound export, ExLlama compatibility, and faster batched calibration.

## Development notes

The pre-5.0 development notes documented the transition to data-parallel MoE quantization, Python 3.13 free-threading, disk offload, AWQ/Marlin support, and the model-definition refactor that enabled future quantization formats.
