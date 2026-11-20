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
<p align="center"><strong>An extensible platform for LLM quantization.</strong><br>Compress, evaluate, and run models across supported GPUs, accelerators, and CPUs.</p>
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

**7.11.4**

- **Quantization:** SmoothMSE weights its shrink search by input saliency (the Hessian diagonal for GPTQ/QQQ fallbacks, weight magnitude for calibration-free RTN) and keeps the unclipped range when clipping would increase that weighted error; QQQ's GSQ calibration now uses the same FP32 activation scales as inference. AWQ samples routed-expert activations uniformly across the calibration set instead of favoring later batches. Paro retains mixed-shape batches, and SenseNova-Loop calibrates on every pass through its repeated layers.
- **Quantization memory:** AWQ and Paro limit how many activation rows they retain. AWQ reuses its expert sampling buffers and scales experts in place across GPUs, while EXL3 spreads expert Hessians across the configured GPU pool. [ROCm allocator defaults](docs/rocm-default-environment.md) reduce peak reserved VRAM for wide MACaQ layers.
- **Inference:** Marlin [speeds up large-batch prefill](docs/marlin-hybrid-schedule.md) and fixes GPTQ/AWQ output errors under GPU contention and MXFP8/MXFP4 execution. Portable QQQ and VPTQ Torch kernels now support compiled generation. Fixed crashes when loading another model during compiled inference and improved EXL3 BF16 execution and Paro dispatch for long prefills.
- **Compat:** Fixed Transformers `device_map="auto"` when GPTQ, RTN, or AWQ layers spill to CPU or disk. Checkpoint loads preserve required FP32 tensors, and re-saves preserve [quantized embeddings and output heads](docs/embedding-checkpoint-shard-replacement.md). Marlin now requires [NVIDIA Ampere or newer](docs/support-matrix.md).

**7.11.2**

- **Inference:** `optimize()` now compiles the forward pass used by generation, and deleting compiled models releases VRAM. [Supported CPU kernels](docs/kernel-compile-matrix.md) compile with Inductor without graph breaks. [Torch compile fallback](docs/torch-compile-fallback.md) covers the first-use compile lock and dynamo failure classification, and the [generation compile policy](docs/generation-compile-policy.md) explains when static-cache auto-compile is suppressed. EXL3 uses native kernels for BF16, and [Paro](docs/paro.md) selects Triton on ROCm.
- **Models:** Added quantization for [SenseNova-Loop 9B](docs/supported-models.md) and [Decision Transformer](docs/decision-transformer-calibration.md), which calibrates on trajectories without a tokenizer. Fixed checkpoint round trips for Qwen2.5-Omni, Qwen3-Omni, and InternVL.
- **Quantization:** GPTQ and EXL3 use less GPU memory. NF12 and MACaQ can resume from checkpoints, and models stay runnable after [failed or cancelled runs](docs/cancelled-run-contract.md).
- **Calibration:** Fixed token positions and masks in padded and multimodal batches. Chat datasets retain tool and document context, and AWQ still scales MoE experts that received tokens when others received none.
- **GGUF:** Added ggml 1.0 support, including reading and dequantizing FP8 E4M3 tensors. The `gguf` extra requires `llama-cpp-python>=0.3.37`.
- **Compat:** Fixed GPTQ and AWQ CPU offload in Transformers and added CPU readers for AWQ GEMV_FAST and LLM_AWQ. ParoQuant is now Paro; existing checkpoints still load.

*API note:* Use [`CalibrationConfig`](docs/quantization-guide.md) for calibration options. [`GPTQModel.export()` and public MLX-LM conversion](docs/checkpoint-interop.md#the-gptqmodelexport-and-mlx-conversion-apis-were-removed) were removed.

**7.11.0**

- **[MACaQ](https://github.com/ModelCloud/GPTQModel/blob/main/docs/macaq.md):** Native MXFP4 W4A4 quantization for low-precision weights and activations, with CUDA, ROCm, and Apple silicon inference. Includes packed checkpoints and a portable Torch fallback.
- **[NF12](https://github.com/ModelCloud/GPTQModel/wiki/NF12):** Lossless 12-bit compression of BF16 weights reduces source-weight storage during calibration without changing the restored values. Packed inference now includes native Metal support alongside CUDA, HIP, and Torch paths. [MLX](https://github.com/ModelCloud/GPTQModel/blob/main/docs/mlx-runtime.md) · [MoE capture contract](https://github.com/ModelCloud/GPTQModel/blob/main/docs/nf12-moe-capture-contract.md)
- **Tenstorrent:** Packed GPTQ inference through Torch/XLA and TTNN, multi-card execution, and block-float quantization. Long-context Llama serving adds paged caching and chunked prefill; compatible block-float checkpoints also run on CUDA, ROCm, and CPU. [Guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/tenstorrent.md) · [Serving](https://github.com/ModelCloud/GPTQModel/blob/main/docs/tt-llama-paged-serving.md) · [Device ownership](https://github.com/ModelCloud/GPTQModel/blob/main/docs/ttnn-device-ownership.md)
- **Quantization:** Resumable [distributed EXL3](https://github.com/ModelCloud/GPTQModel/blob/main/docs/exl3-durable-execution.md) jobs ([projection publication recovery](https://github.com/ModelCloud/GPTQModel/blob/main/docs/exl3-projection-publication-recovery.md)), broader mixed-bit allocation across [AWQ](https://github.com/ModelCloud/GPTQModel/wiki/AWQ)/[EXL3](https://github.com/ModelCloud/GPTQModel/wiki/EXL3)/[GGUF](https://github.com/ModelCloud/GPTQModel/wiki/GGUF), and [SmoothQuant](https://github.com/ModelCloud/GPTQModel/blob/main/docs/protocol-actions.md) preprocessing improve large-model workflows and quality control. [Quantizer calibration flows](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantizer-moe.md#4-hessian-failure-and-zero-sample-fallback-chain) document the shared Hessian-failure and zero-sample fallback contracts. [MoE routing override and bypass](https://github.com/ModelCloud/GPTQModel/blob/main/docs/moe-routing-override.md) defines when each mode is honoured and when it fails closed, and its [AWQ bypass replay kwargs contract](https://github.com/ModelCloud/GPTQModel/blob/main/docs/moe-routing-override.md#8-awq-replay-kwargs-for-bypass-routing) keeps instance-level replay wrappers from widening a standard MoE block's signature. [Shared-input Hessian dedup](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gptq-hessian-accumulation.md#shared-input-dedup-for-calibrated-quantizers) covers QQQ/VPTQ/MACaQ/EXL3 and GPTAQ/FOEM, and the [`desc_act` sink-row cap](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gptq-hessian-accumulation.md#desc_act-sink-row-cap) keeps its row-norm bound on device. Optional [calibration attention-sink exclusion](https://github.com/ModelCloud/GPTQModel/blob/main/docs/calibration-attention-sinks.md) drops BOS-token rows from the calibration statistics. Embedding-endpoint quantization is [requantize-only on a loaded checkpoint](docs/embed-endpoint-quantize.md); a fresh `quantize()` refuses the unsupported modes before calibration.
- **Inference:** Opt-in attention and MLP projection fusion, faster quantized embedding lookups, and improved small-batch inference. Native backend selection and packed-weight storage are preserved. [Fusion guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/fusion-planner.md)
- **Configuration:** Typed [backend configurations](https://github.com/ModelCloud/GPTQModel/blob/main/docs/backend-selection.md) unify backend selection and tuning, with explicit numerical and worker controls instead of environment switches.
- **Calibration:** Broader [dataset input support](https://github.com/ModelCloud/GPTQModel/blob/main/docs/vlm-calibration-input-normalization.md), document-isolated [AWQ replay](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-calibration-aggregation.md), and opt-in [attention-sink exclusion](https://github.com/ModelCloud/GPTQModel/blob/main/docs/quantization-guide.md#gptq-w8a16-or-another-integer-weight-width) improve calibration coverage and control. [Calibration layer replay](https://github.com/ModelCloud/GPTQModel/blob/main/docs/calibration-replay.md) documents the per-definition replay contract, including PanGu-α's top-query layer.
- **Memory:** Reduced packed-weight duplication, bounded EXL3 scratch, and shorter-lived [inference workspaces](https://github.com/ModelCloud/GPTQModel/blob/main/docs/kernel-scratch-lifecycle.md) lower memory use. Fused-expert MoE loads keep an excluded GPU unclaimed during [Defuser conversion](docs/multi_gpu_quantization.md#fused-expert-conversion-device-scope), and AWQ folds each expert on its own device in a [balanced MoE pool](docs/multi_gpu_quantization.md#awq-scale-groups-on-a-balanced-moe-pool)., and compiled forwards stay on the model's device through [compiled forward device scope](docs/multi_gpu_quantization.md#compiled-forward-device-scope).
- **Models:** Added OLMo Hybrid, Granite SWA/MoE SWA, and Dots3 Note quantization, plus broader decoder-weight coverage and Transformers 5.22 compatibility. [Model notes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/supported-models.md) · [Decision Transformer calibration](https://github.com/ModelCloud/GPTQModel/blob/main/docs/decision-transformer-calibration.md)
- **Compat:** Major fixes across dense/MoE calibration, multimodal replay, checkpoint round trips, Transformers model loading, multi-GPU memory use, and ROCm execution. Unsupported combinations fail clearly rather than loading incomplete weights. [Full notes](https://github.com/ModelCloud/GPTQModel/blob/main/docs/changelog/recent.md#7110) · [Cancelled-run model state](https://github.com/ModelCloud/GPTQModel/blob/main/docs/cancelled-run-contract.md)
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
| **[Paro](https://github.com/ModelCloud/GPTQModel/wiki/ParoQuant)** | Low-bit weight quantization with learned rotations for accuracy recovery. [Affine fitting and optimization controls](docs/paro-optimization.md). [Fused rotation backward](docs/paro.md#fused-rotation-backward). |
| **[MACaQ](https://github.com/ModelCloud/GPTQModel/blob/main/docs/macaq.md)** | Calibrated MXFP4 weights and activations, with native W4A4 inference and a W4A16 option. |
| **[QQQ](https://github.com/ModelCloud/GPTQModel/wiki/QQQ)** | INT4 weights with dynamic INT8 activation quantization for W4A8 inference. |
| **[GGUF](https://github.com/ModelCloud/GPTQModel/wiki/GGUF)** | Block-quantized checkpoints for GGUF-oriented workflows and native inference. [Bits and subtype resolution](docs/gguf-config-resolution.md). [Torch fused dispatch](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gguf-torch-fused-dispatch.md). [CPU zero-copy packed weights](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gguf-kernel-selection.md#cpu-zero-copy-packed-weight). [mmap pages after the aligned copy](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gguf-kernel-selection.md#mmap-pages-after-the-aligned-copy). [Checkpoint load policy](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gguf-load-policy.md). [Q8_1 auxiliary-field range](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gguf-q8-1-auxiliary-field-range.md). |
| **[FP8](https://github.com/ModelCloud/GPTQModel/wiki/FP8-and-NVFP4)** | Floating-point weight quantization with scaled E4M3/E5M2 formats. Native inference is weight-only W8A16; [W8A8 is an explicit per-token opt-in](docs/fp8-weight-only-dispatch.md). [Packed-weight load and move lifecycle](docs/fp8-packed-weight-lifecycle.md). [Lean decode epilogue](docs/fp8-lean-decode-epilogue.md). [Block-scale decode](docs/fp8-block-scale-decode.md).  NVFP4 decode keeps the [effective-weight range](docs/fp4-effective-weight-range.md) instead of rounding the reconstructed weight to the activation dtype. |
| **[EXL3](https://github.com/ModelCloud/GPTQModel/wiki/EXL3)** | Variable-rate compression with target average bits per weight and mixed-tier allocation. [BF16 activation dispatch and the fused pre-scale](docs/exl3-wide-activation-dispatch.md). |
| **[VPTQ](https://github.com/ModelCloud/GPTQModel/wiki/VPTQ)** | Vector-codebook quantization for low-bit model compression. [Runtime kernel selection](docs/vptq.md#inference-and-checkpoint-compatibility). |
| **[BitsAndBytes](https://github.com/ModelCloud/GPTQModel/wiki/BitsAndBytes)** | NF4/FP4 and INT8 quantization options. |
| **[NF12](https://github.com/ModelCloud/GPTQModel/wiki/NF12)** | Lossless BF16 weight compression, distinct from lossy low-bit quantization. |

Quantized kernels keep only packed weights resident; per-call QQQ and EXL3 kernel scratch, one quantization stream and one tile-scratch pair per device, per-device plan lifetimes and the single GGUF CUDA weight copy are documented in [kernel scratch lifecycle](docs/kernel-scratch-lifecycle.md). CPU layer packing quantizes and packs each bounded input block in one shift/reduce pass; see [CPU block packing](docs/cpu-block-packing.md). Compiled CPU int4pack forwards decode the live runtime layout for grad-requiring calls; see [compiled CPU int4pack gradient fallback](docs/cpu-int4pack-compiled-grad.md).

EXL3 supports packed 1.5, 2.5 and 3.5 bits per weight with the `mul1` codebook. [Native half-bit EXL3 rates](docs/exl3-runtime-weights.md#native-mul1-half-bit-rates)

### Control, scale, and quality ⚙️

- **Choose where to spend precision.** Mix bit widths, keep sensitive modules
  unquantized, and use activation-quantization or preprocessing policies where
  supported. Quantization protocols let you organize multi-stage recipes.
- **Select embedding endpoints explicitly.** `QuantizeEmbedConfig` rejects an
  unknown `embed_quant_mode` or a non-`bool` `embed_only` before any layer
  replay instead of silently quantizing nothing; see
  [embedding quantization config validation](docs/embedding-quantization-config.md).
- **Keep floating activation searches finite.** NVFP4 block-scale ranking
  normalizes candidate error by an exact power of two so large finite
  activations cannot overflow the comparison and silently return zeros; see
  [NVFP4 candidate ranking](https://github.com/ModelCloud/GPTQModel/blob/main/docs/gptq-w4a-gb10.md#candidate-ranking-and-the-finite-scale-contract).
- **Fit larger models.** Multi-GPU calibration, CPU/disk offload, and checkpointed
  execution help manage memory and recover interrupted jobs. [Disk-offload run
  directories](https://github.com/ModelCloud/GPTQModel/blob/main/docs/offload-run-directories.md)
  are per-run scratch space and are removed at teardown or failure, and the
  [in-session model](https://github.com/ModelCloud/GPTQModel/blob/main/docs/in-session-model-after-offload.md)
  is fully materialized when `quantize()` returns;
  [reference offload](https://github.com/ModelCloud/GPTQModel/blob/main/docs/safetensors-reference-offload.md)
  keeps untouched base modules as source-checkpoint references instead of
  second copies, and releases pinned staging memory. Quantized MoE reloads park
  fused expert projections on `meta` until the packed weights are read
  ([meta fused-expert shell](docs/quantized-moe-reload-meta-shell.md)).
  Distributed EXL3 workflows support durable replay and resume.
- **Recover accuracy.** Use [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ) refinement, [EoRA](https://github.com/ModelCloud/GPTQModel/wiki/EoRA) low-rank compensation, [rotation](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#rotations),
  [smoothing](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#weight-smoothing) (see the [smoother applicability contract](docs/smoother-applicability.md)), and [norm tuning](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#post-quantization-norm-tuning) for supported recipes. [LoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md) training and adapter
  loading support further task adaptation; exported EoRA `rank_pattern` keys follow the [adapter rank-pattern contract](docs/adapter-rank-pattern-matching.md) so PEFT builds the trained rank per module. [Rotation model support](https://github.com/ModelCloud/GPTQModel/blob/main/docs/rotation-model-support.md) lists the exact definitions that opt in. AWQ GSQ uses the
  [packed-code scoring contract](docs/gsq-awq-code-recovery.md) to evaluate exported candidates.
  QQQ inference and GSQ calibration share one [activation INT8 grid contract](docs/qqq-activation-grid.md). The fused relaxation kernel keeps the [eager precision contract](docs/gsq-fused-relaxation-precision.md)
  for BF16 and FP32 logits, and per-linear refinement searches the exported
  codes and refits scales under the [scalar refinement
  contract](docs/gsq-scalar-refinement.md).
- **Trust the fold.** AWQ divides a scale only into the norm that actually feeds
  a quantized block, including multimodal wrappers and hybrid layer shapes; see
  the [AWQ scaling groups](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-scaling-groups.md) contract. Scale search keeps
  routing in the original-input coordinates, validates replay coverage, and
  restores temporary weights on failure; see [search safety](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-scaling-groups.md#scale-search-safety).
- **Reject unrepresentable AWQ pack scales.** The native GEMM/GEMV packers
  validate the actual FP16 storage cast before replacing module state; see
  [AWQ pack-time scale representability](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-pack-scale-representability.md).
- **Decode AWQ weights in range.** The portable Torch AWQ forward picks its
  decode dtype from the stored scale range, so a wide or tiny checkpoint scale
  no longer becomes `inf`/`0` before the product; see
  [AWQ Torch forward decode range](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-torch-decode-range.md).
- **Keep the AWQ extraction table out of a capture.** A four-bit dequant that
  first runs inside `torch.cuda.graph(...)` records its offset table instead of
  computing it, so the table is published only from eager calls; an abandoned
  capture can no longer leave later eager dequant decoding with zeros. See
  [capture-safe extraction offsets](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-dequantize-tiling.md#capture-safe-extraction-offsets).
- **Keep the AWQ bias out of the scale dtype.** A non-quantized bias is added
  after the reduction, so packing keeps its source precision instead of casting
  it to FP16; see
  [AWQ pack-time bias storage](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-pack-bias-representability.md).
- **Keep the Marlin bias in the checkpoint.** The native Marlin kernel adds the
  bias to its FP32 accumulator, so the buffer keeps its checkpoint width and
  order and the padded, permuted operand is derived per launch; see
  [Marlin bias epilogue and storage](https://github.com/ModelCloud/GPTQModel/blob/main/docs/marlin-bias-epilogue.md).
- **Measure before deploying.** Evaluate perplexity and downstream tasks, compare
  against a non-quantized baseline, and test prompts representative of your users.
  Compression settings trade model size, calibration cost, speed, and quality.
- **Preserve causal replay.** Bloom AWQ keeps captured attention context and
  rejects unsafe residual folds; see [Bloom replay safety](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-bloom-replay.md).
- **Capture masks a model passes positionally.** GPT-2 hands its causal mask to
  each block as a positional argument; reading only keyword arguments dropped it
  and replayed every layer without causality. See
  [positional replay mask capture](https://github.com/ModelCloud/GPTQModel/blob/main/docs/positional-replay-mask-capture.md).
- **Classify additive masks before dropping padding.** An all-zero extended mask
  means "no padding", not "all padding"; layer replay, user calibration and AWQ
  share one keep-mask classifier. See
  [attention-mask keep classification](https://github.com/ModelCloud/GPTQModel/blob/main/docs/attention-mask-keep-classification.md).
- **Replay each calibration document separately.** AWQ token-row replay rebuilds
  a block-diagonal per-document mask and keeps each row's original position;
  additive masks treat `0` as attended. See
  [AWQ calibration aggregation](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-calibration-aggregation.md).
- **Resolve the rotary label from the attention.** AWQ scale search rebuilds
  `position_embeddings` with the attention's own rope label and keeps a
  per-rope-type mapping intact; see
  [AWQ rotary replay selection](https://github.com/ModelCloud/GPTQModel/blob/main/docs/awq-rotary-replay.md).
- **Trim padding before the position budget.** Over-long calibration rows are
  measured on their real tokens, not their padded width, so a left-padded row
  cannot lose real tokens to its own padding. See
  [calibration row trimming](https://github.com/ModelCloud/GPTQModel/blob/main/docs/calibration-row-trimming.md).
- **Pack before filtering when concatenating.** With `concat_size` set, the
  per-row minimum-length filter and position trim run on the packed windows
  instead, so short documents are combined rather than discarded and a long
  document is not cut before it is windowed. See
  [calibration row trimming](https://github.com/ModelCloud/GPTQModel/blob/main/docs/calibration-row-trimming.md#concat_size-reverses-the-order-pack-then-filter).
- **Keep VLM calibration rows intact.** Pre-tokenized rows (`input_ids`) take the
  base text path instead of a chat template, columnar mappings are unpacked, and
  conversations reject `CalibrationConfig.concat_size`. See
  [VLM calibration input normalization](https://github.com/ModelCloud/GPTQModel/blob/main/docs/vlm-calibration-input-normalization.md).

### Inference and deployment 🚢

Run compatible checkpoints directly with GPT-QModel, integrate with Transformers
(import `gptqmodel` first so packed MoE experts load through the
[native bridge](https://github.com/ModelCloud/GPTQModel/blob/main/docs/hf-interop.md)),
or deploy supported GPTQ/AWQ formats through vLLM and SGLang. Apple silicon has
native MLX quantization and packed inference. Optional projection fusion can
combine supported attention and MLP operations without changing the saved format.

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
| D | DBRX / DBRX Converted; DeciLM; DeepSeek V2 / V3 / V3.2 / V4 / VL / VL2 / OCR2; [DeepSeek V4.1 Flash](docs/deepseek-v41-module-inventory.md); DiffusionGemma; Dots1 / Dots3 Note; Dream |
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
so install the compiler and CUDA/ROCm toolkit required by your backend. HIP
builds stage their sources under the cache; see
[HIP JIT source staging](https://github.com/ModelCloud/GPTQModel/blob/main/docs/hip-jit-source-staging.md).

See [installation and optional extras](https://github.com/ModelCloud/GPTQModel/wiki/Installation-and-Quickstart)
for source installs and platform setup. Install vLLM or SGLang in a separate
environment aligned with that engine's dependency pins.

The native GGUF C++ backends live in the `gguf` extra
(`llama-cpp-python>=0.3.37`, which vendors ggml 1.0). `GGUF_CPP` reads the
loaded release through `ggml_version()`; ggml 1.0 adds the `F8_E4M3` tensor
type, which the native GGUF reader can dequantize.

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
`trust_remote_code=True` for checkpoint code you trust. For model types the
installed Transformers implements natively, the native class is used unless the
checkpoint declares a remote-only layout; see
[`docs/trust-remote-code-isolation.md`](docs/trust-remote-code-isolation.md#native-override-of-explicit-trust).
Generation settings preserved during quantized checkpoint re-save are described
in [`docs/generation-config-round-trip.md`](docs/generation-config-round-trip.md).
Re-saving a loaded fused-expert MoE defuses the export shell and fails closed on
dropped quantization payloads; see
[`docs/quantized-moe-resave-defuse.md`](docs/quantized-moe-resave-defuse.md).
`export(..., "hf")` picks its dense decoder from the checkpoint's declared
`quant_method` and refuses methods that have none (Paro, MACaQ, GGUF, QQQ, NF12,
VPTQ, EXL3) instead of decoding them with another layout's decoder; see
[`docs/export-hf-format-detection.md`](docs/export-hf-format-detection.md).
In-session QQQ kernel rehoming and cache initialization are documented in
[docs/qqq-kernel-selection.md](docs/qqq-kernel-selection.md).
QQQ's shared-memory staging of the per-group scales and the reduction-scratch
ordering are documented in
[docs/qqq-group-scale-staging.md](docs/qqq-group-scale-staging.md).
Quantized modules load their scales and other quantization buffers at the dtype
the checkpoint stores, whatever model dtype you request; see
[checkpoint buffer dtype](docs/checkpoint-and-runtime-contracts.md#4-checkpoint-buffer-dtype).

### Quantization 🗜️

This example creates a 4-bit GPTQ checkpoint using 256 streamed C4 samples.
Use calibration text representative of your workload; the dataset and precision
settings affect the quality of the resulting model.

```python
from datasets import load_dataset
from gptqmodel import CalibrationConfig, GPTQConfig, GPTQModel

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
model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))
model.save("Qwen3-0.6B-GPTQ-4bit")
```

`quantize()` takes every calibration choice in one `CalibrationConfig`: `dataset`,
`concat_size`, `sort` (`"desc"`), `batch_size` (`1`), `min_length` (`10`),
`concat_separator`, `exclude_attention_sinks` (`False`) and the EoRA
`adapter_dataset`. See
[the calibration API](https://github.com/ModelCloud/GPTQModel/blob/main/docs/calibration-config.md).

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
  large run, and use checkpoint/resume where supported. A CPU-offloaded
  placement also changes the autograd context `generate()` uses; see
  [generate on a CPU-offloaded model](docs/generate-cpu-offload-autograd.md).
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
| Select and tune execution backends | [Typed backend configuration](https://github.com/ModelCloud/GPTQModel/blob/main/docs/backend-selection.md), [kernel and thread reference](https://github.com/ModelCloud/GPTQModel/blob/main/docs/kernel-config.md) |
| Resume a job | [Checkpoint and resume](https://github.com/ModelCloud/GPTQModel/blob/main/checkpoint.md) · [Distributed EXL3](https://github.com/ModelCloud/GPTQModel/blob/main/docs/exl3-durable-execution.md) |
| Export a dense or MLX checkpoint | [Dense dequantization](https://github.com/ModelCloud/GPTQModel/blob/main/docs/checkpoint-interop.md#7-dense-dequantization-with-dequantize_model) · [MLX conversion](https://github.com/ModelCloud/GPTQModel/blob/main/docs/mlx-runtime.md) |
| Check a checkpoint's key layout on reload | [Checkpoint key layout](https://github.com/ModelCloud/GPTQModel/blob/main/docs/checkpoint-interop.md#13-checkpoint-key-layout-source-renames-are-idempotent) |
| Recover accuracy or fine-tune | [EoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/eora/README.md) · [PEFT/LoRA](https://github.com/ModelCloud/GPTQModel/blob/main/docs/peft.md) · [GSQ](https://github.com/ModelCloud/GPTQModel/wiki/GSQ) · [RCO](https://github.com/ModelCloud/GPTQModel/wiki/GSQ#non-uniform-bit-allocation-rco) |
| Evaluate quality and performance | [Evaluation](https://github.com/ModelCloud/GPTQModel/wiki/Evaluation) · [Benchmark methodology](https://github.com/ModelCloud/GPTQModel/blob/main/docs/benchmark-methodology.md) |
| Serve an OpenAI-compatible API | [Serving guide](https://github.com/ModelCloud/GPTQModel/wiki/Serving) · [Security](https://github.com/ModelCloud/GPTQModel/blob/main/docs/openai-server-exposure.md) |
| Add a model or inspect its plan | [Module-plan guide](https://github.com/ModelCloud/GPTQModel/blob/main/docs/module-plan-guide.md) · [Tree resolution](https://github.com/ModelCloud/GPTQModel/blob/main/docs/model-tree-resolution.md) · [Leaf coverage](https://github.com/ModelCloud/GPTQModel/blob/main/docs/module-tree-leaf-coverage.md) · [Developer guide](https://github.com/ModelCloud/GPTQModel/wiki/Developer-Guide) |
| Load legacy remote-code models on Transformers 5 | [Transformers 5 compatibility](https://github.com/ModelCloud/GPTQModel/blob/main/docs/transformers-5-compat.md#k-native-loads-run-the-pre-load-hook-one-legacy-generate-adapter-issue-4121-items-2-3) |
| Diagnose runtime issues | [Troubleshooting](https://github.com/ModelCloud/GPTQModel/wiki/Troubleshooting-and-FAQ) · [Concurrency](https://github.com/ModelCloud/GPTQModel/blob/main/docs/concurrency.md) · [Thread caps](https://github.com/ModelCloud/GPTQModel/blob/main/docs/thread-caps.md) · [Device pool intra-op budget](https://github.com/ModelCloud/GPTQModel/blob/main/docs/device-thread-pool-intraop-budget.md) |
| Validate the CUDA CI and source release runtime | [CI runtime and publication gates](docs/ci-runtime.md) |
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
to get started. AI agents must follow the
[issue-claim security gate](docs/issue-claim-security-screen.md) before claiming work.

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
| [Paro](https://github.com/ModelCloud/GPTQModel/wiki/ParoQuant) | Yesheng Liang et al., Z-Lab | [Paper](https://openreview.net/forum?id=1USeVjsKau) · [Code](https://github.com/z-lab/paroquant) |
| [MACaQ](https://github.com/ModelCloud/GPTQModel/blob/main/docs/macaq.md) | Lorenz K. Muller et al., Huawei CSL | [Paper](https://arxiv.org/abs/2611.10649) · [Code](https://github.com/huawei-csl/MACaQ) |
| [QQQ](https://github.com/ModelCloud/GPTQModel/wiki/QQQ) | Ying Zhang et al., Meituan | [Paper](https://arxiv.org/abs/2406.09904) |
| [VPTQ](https://github.com/ModelCloud/GPTQModel/wiki/VPTQ) | Yifei Liu et al., Microsoft | [Paper](https://arxiv.org/abs/2409.17066) · [Code](https://github.com/microsoft/VPTQ) |
| [GGUF](https://github.com/ModelCloud/GPTQModel/wiki/GGUF) / llama.cpp | Georgi Gerganov and ggml-org contributors | [Project](https://github.com/ggml-org/llama.cpp) |
| [ExLlama V3 / EXL3](https://github.com/ModelCloud/GPTQModel/wiki/EXL3) | turboderp and contributors | [Project](https://github.com/turboderp-org/exllamav3) · [Native CUDA dispatch](docs/exl3-native-decode.md) |
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
