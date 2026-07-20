# Upstream agent-skill audit

Audit date: 2026-07-20.

This repository's local skills adapt applicable workflow ideas from Intel AutoRound, vLLM, and SGLang. They are not wholesale copies. All three audited repositories use Apache-2.0 licensing, matching GPT-QModel.

## Pinned sources

| Project | Audited commit | Agent material reviewed | Port decision |
| --- | --- | --- | --- |
| Intel AutoRound | [`06c33eae41c9cc1bc1f2521ceb45295731622c89`](https://github.com/intel/auto-round/tree/06c33eae41c9cc1bc1f2521ceb45295731622c89) | Root `AGENTS.md`; skills for adding LLMs, export formats, inference backends, and quantization datatypes | Adapted architecture diagnosis, explicit registration, quant-function contracts, and pack/save/load validation into the quantization, backend, and model-support skills. Diffusion- and VLM-only steps were excluded unless they generalize to a GPT-QModel model adapter. |
| vLLM | [`4ec199b66a791070348f3baf847b3a873c48cdd9`](https://github.com/vllm-project/vllm/tree/4ec199b66a791070348f3baf847b3a873c48cdd9) | Root `AGENTS.md`; Buildkite failure skill | Adapted the test-first, nearest-helper, behavior-assertion, separate-benchmark, and model-evaluation guidance. No vLLM quantization or hardware `SKILL.md` existed at the pinned revision; the Buildkite-specific workflow was not ported. |
| SGLang | [`91b210f7b06cf28ccd3273633835cdc6ddfe9be5`](https://github.com/sgl-project/sglang/tree/91b210f7b06cf28ccd3273633835cdc6ddfe9be5) | Skills for JIT kernels, compiled SGL kernels, CUDA crash debugging, trace generation, and profiler analysis | Adapted kernel-path selection, thin validated wrappers, correctness-plus-benchmark requirements, architecture gates, staged CUDA debugging, stage-separated capture, and source-backed profiler triage. Serving-only operations and diffusion ModelOpt quantization were excluded. |

The directly applicable source files were:

- AutoRound: [`adapt-new-llm`](https://github.com/intel/auto-round/blob/06c33eae41c9cc1bc1f2521ceb45295731622c89/.claude/skills/adapt-new-llm/SKILL.md), [`add-export-format`](https://github.com/intel/auto-round/blob/06c33eae41c9cc1bc1f2521ceb45295731622c89/.claude/skills/add-export-format/SKILL.md), [`add-inference-backend`](https://github.com/intel/auto-round/blob/06c33eae41c9cc1bc1f2521ceb45295731622c89/.claude/skills/add-inference-backend/SKILL.md), and [`add-quantization-datatype`](https://github.com/intel/auto-round/blob/06c33eae41c9cc1bc1f2521ceb45295731622c89/.claude/skills/add-quantization-datatype/SKILL.md).
- vLLM: [root `AGENTS.md`](https://github.com/vllm-project/vllm/blob/4ec199b66a791070348f3baf847b3a873c48cdd9/AGENTS.md). Its only root Claude skill at this revision was Buildkite-specific and outside this port's scope.
- SGLang: [`add-jit-kernel`](https://github.com/sgl-project/sglang/blob/91b210f7b06cf28ccd3273633835cdc6ddfe9be5/.claude/skills/add-jit-kernel/SKILL.md), [`add-sgl-kernel`](https://github.com/sgl-project/sglang/blob/91b210f7b06cf28ccd3273633835cdc6ddfe9be5/.claude/skills/add-sgl-kernel/SKILL.md), [`debug-cuda-crash`](https://github.com/sgl-project/sglang/blob/91b210f7b06cf28ccd3273633835cdc6ddfe9be5/.claude/skills/debug-cuda-crash/SKILL.md), [`generate-profile`](https://github.com/sgl-project/sglang/blob/91b210f7b06cf28ccd3273633835cdc6ddfe9be5/.claude/skills/generate-profile/SKILL.md), and [`llm-torch-profiler-analysis`](https://github.com/sgl-project/sglang/blob/91b210f7b06cf28ccd3273633835cdc6ddfe9be5/.claude/skills/llm-torch-profiler-analysis/SKILL.md).

The profiling port deliberately omits upstream fixed host paths, dated reference-run claims, framework-specific helper scripts that do not exist here, and process-name-wide kill commands. It retains representative workload capture, accuracy gating, prefill/decode separation, graph-off source mapping versus graph-on formal evidence, and deterministic source-backed analysis.

## GPT-QModel-specific result

The port is organized under `.agents/skills/`:

- `gptqmodel-quantization`: algorithm, calibration, format, packing, serialization, and GPTQ/AWQ-family validation.
- `gptqmodel-backends`: declarative quantized-linear capabilities, selection priority, availability, and fallback.
- `gptqmodel-cuda-kernels`: Triton versus CUDA/C++ JIT choice, extension registration, correctness, debugging, and benchmarking.
- `gptqmodel-gpu-profiling`: bounded trace capture, stage separation, kernel/overlap/fusion triage, and artifact reporting.
- `gptqmodel-ampere-kernels`: `sm_80` tuning and validation for A100-class devices.
- `gptqmodel-hopper-kernels`: `sm_90`/`sm_90a` boundaries and H100-only validation requirements.
- `gptqmodel-model-support`: model-family diagnosis, module trees, MoE lifecycle, registration, and end-to-end tests.

The workflows use GPT-QModel's actual integration points: `gptqmodel/quantization/config.py`, processors in `gptqmodel/looper/`, quantized-linear discovery in `gptqmodel/utils/importer.py`, the JIT registry in `gptqmodel/extension.py`, and model registration in `gptqmodel/models/auto.py`.

## Hardware snapshot and portability rule

The audit host reported eight PCI-ordered `PG506-230/232` devices, each with compute capability 8.0 and approximately 96 GiB of memory. PyTorch reported 124 SMs per visible device. This is a dated observation, not a device-index contract. Every hardware workflow must probe the live process and derive launch assumptions from properties rather than embedding the snapshot.

No H100 was present during this port. Hopper guidance can be statically reviewed or compiled where toolchains permit, but runtime correctness and performance claims require an actual Hopper run. Generic Hopper code should target `sm_90`; use `sm_90a` only when intentionally depending on architecture-accelerated features and keep that path isolated from portable fallbacks.
