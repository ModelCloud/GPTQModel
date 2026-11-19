<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Quantization guide and end-to-end recipes

This guide follows the current GPT-QModel API. Start with the
[installation instructions](../README.md#install-), then choose a recipe below.
The examples use a supported causal language model on one CUDA GPU. Model
downloads require network access; gated models also require Hugging Face access.
Use an unquantized model ID or local checkpoint as the source.

## Choose weight and activation precision

`W4A16` means 4-bit weights and 16-bit activations. `bits` controls weight
quantization; `dtype=torch.float16` or `torch.bfloat16` controls the model dtype.
Neither `desc_act` (activation ordering) nor AWQ's activation-aware calibration
means that activations are stored in INT4. Scales, zero points, codebooks, and
unquantized modules add storage beyond the nominal weight bit width.

| Method / recipe | Stored weights | Activation / compute path | Calibration |
| --- | --- | --- | --- |
| GPTQ | Integer 2, 3, 4, 5, 6, 7, or 8 bits | Normally FP16/BF16; kernel support varies by bit width | Required |
| AWQ | Integer weights; start with 4 bits | Normally FP16/BF16; optimized kernels commonly require 4-bit weights | Required |
| RTN | Rounded integer weights exported in GPTQ or AWQ layout | Normally FP16/BF16, according to the destination kernel | Not required |
| Paro | INT4 with learned rotations | FP16/BF16 | Required; includes optimization |
| QQQ | Symmetric INT4, group size 128 or -1 | Dynamic INT8 activation quantization inside the linear operation; FP16/BF16 inputs and outputs | Required |
| GGUF | Format-specific blocks, e.g. Q4_0 or Q4_K_M | Native kernels consume floating-point activations | Not required |
| FP8 | E4M3 or E5M2 weights with scales | The native Torch path consumes floating-point activations; this config alone does not select W8A8 | Not required |
| BitsAndBytes 4-bit | NF4 or FP4 | Floating-point activations | Not required |
| BitsAndBytes 8-bit | INT8 | Native INT8 matmul quantizes activations dynamically; fallback arithmetic can differ | Not required |
| EXL3 | Variable-rate codebooks; `bits` is a target average bits per weight | Floating-point model activations | Required |
| VPTQ | Vector-codebook indices plus codebooks | Floating-point model activations; effective bits per weight depend on vector and codebook sizes | Required |
| MACaQ | OCP MXFP4 (E2M1) with one E8M0 scale per 32 values | Native MXFP4 W4A4 (`tl.dot_scaled`) on CUDA; bf16 W4A16 on the portable Torch path | Required |
| Experimental GPTQ W4A | Packed GPTQ INT4 | Explicit FP8 or NVFP4 activation policy for selected Llama decoder operations | Required for fresh quantization |

This is a recipe map, not a promise that every method, bit width, dtype, and
device combination works. Check the [method/format/backend matrix](support-matrix.md#methods-formats-and-backends)
and [hardware table](support-matrix.md#hardware-and-operating-systems). `BACKEND.AUTO`
selects a compatible installed kernel; it cannot make an unsupported combination
valid. The portable Torch kernels are useful for establishing a working baseline.
An FP8 **source checkpoint** decoded for requantization is also distinct from
choosing an FP8 **output recipe**.

## Quantize, save, reload, and generate

Copy this complete example into a Python file. It produces a GPTQ W4A16
checkpoint. To use another recipe, replace the `qcfg` and `backend` assignments
with one of the blocks in the next section; keep the rest of the workflow.

```python
import gc

import torch
from datasets import load_dataset
from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization import GPTQConfig

model_id = "Qwen/Qwen3-0.6B"
quant_path = "Qwen3-0.6B-GPTQ-W4A16"
load_dtype = torch.float16
qcfg = GPTQConfig(bits=4, group_size=128)
backend = BACKEND.TORCH

# Calibration-based recipes need representative text. Streaming avoids
# downloading the entire C4 shard. Weight-only recipes skip this step.
calibration = None  # weight-only recipes call `model.quantize()` without one
if qcfg.requires_calibration_dataset():
    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
        streaming=True,
    )
    calibration = CalibrationConfig(
        dataset=[row["text"] for row in dataset.take(256)],
        batch_size=1,
    )

model = GPTQModel.load(model_id, qcfg, device="cuda:0", dtype=torch.float16)
model.quantize(calibration, backend=backend)
model.save(quant_path)

# Release the first model before loading the saved checkpoint.
del model
gc.collect()
torch.cuda.empty_cache()

loaded = GPTQModel.load(
    quant_path, device="cuda:0", dtype=load_dtype, backend=backend,
)
tokens = loaded.generate(
    "The capital of France is", max_new_tokens=16, do_sample=False,
)[0]
print(loaded.tokenizer.decode(tokens, skip_special_tokens=True))
```

Give each recipe a different `quant_path`. After validating the saved model,
try `backend=BACKEND.AUTO` when reloading to select a faster installed kernel.
The explicit Torch backend above favors portability over throughput.

The 256 samples are a starting point, not a quality guarantee. Replace C4 with
data representative of the target language, domain, and prompt distribution.
Calibration accepts text strings or tokenized input dictionaries; multimodal
models need their model-specific processor and inputs. Increase calibration
coverage before increasing `batch_size`; batch size controls memory and speed,
not the number of examples. Too-short samples may be filtered out. Keep a
separate evaluation set, and record the dataset revision, sample selection,
model revision, dtype, and configuration for reproducibility.

## Reshard a checkpoint

`gptqmodel.reshard()` rewrites an existing safetensors checkpoint into a new
shard layout without loading or dequantizing any weights. The default
`ShardLayout.PER_LAYER_WITH_SINGLE_GLOBAL` writes one directory per transformer
layer and exactly one root `global.safetensors` for embeddings, lm-head, norms,
and every other non-layer module. Every layer stack found in the checkpoint is
sharded separately, including vision transformer blocks; numerically indexed
non-layer modules such as `mlp1.0` stay in the global shard.

```python
from gptqmodel import ShardConfig, ShardLayout, reshard

reshard(
    "Qwen3-0.6B-GPTQ-W4A16",
    "Qwen3-0.6B-GPTQ-W4A16-per-layer",
    ShardConfig(layout=ShardLayout.PER_LAYER_WITH_SINGLE_GLOBAL),
)
```

The output uses `model.layers.0/layer.safetensors`, `model.layers.1/layer.safetensors`,
and so on, plus a root `global.safetensors` and `model.safetensors.index.json`.
The global shard is not named `model.safetensors` because
`transformers.from_pretrained` prefers that single-file name over the index and
would load only the global shard. `max_shard_size` splits oversized layer
directories but never the single global shard. Set
`ShardConfig(layer_prefixes=["model.layers"])` when the checkpoint has no
`config.json` or uses a custom architecture. Pass `overwrite=True` to replace an
existing non-empty destination.

Pass a specific `.safetensors.index.json` path to choose its shards, or a
standalone `.safetensors` path to choose only that file. A path to an indexed
shard selects the complete checkpoint. See
[reshard source selection](reshard-source-selection.md) for the rules.

Resharding reads only safetensors headers, copies payloads byte-for-byte through
an 8 MiB buffer, and keeps at most one source shard open at a time. It does not
mmap source weights, so peak host RAM stays bounded even for multi-hundred-GB
checkpoints. Config, tokenizer, processor, and nested auxiliary files are copied
to the destination; the destination must be outside the source checkpoint.

## Recipe substitutions

Each block below replaces the configuration and backend in the complete example.
Weight-only configurations automatically take the `calibration=None` branch.
These are starting configurations, not identical quality or speed targets.

### GPTQ W8A16 or another integer weight width

```python
from gptqmodel.quantization import GPTQConfig

qcfg = GPTQConfig(bits=8, group_size=128)
backend = BACKEND.TORCH
```

The Torch path also supports `bits=2`, `3`, `4`, `5`, `6`, and `7`. Optimized
backends support narrower subsets. Lower bit widths usually need more careful
quality evaluation. Smaller groups increase scale storage. `sym=False` selects
asymmetric weights where the chosen kernel supports them. `desc_act=True`
enables GPTQ activation ordering and disables the default group-aware ordering;
do not explicitly enable both `desc_act` and `act_group_aware`.

Disabling the group-aware ordering also disables its per-group scale
protection, so on models whose tokenizer emits a BOS token `desc_act=True`
makes attention-sink damage much worse: the few sink rows own the largest
Hessian diagonal, get ordered first with a tiny `Hinv[i, i]`, and their
quantization error is amplified into every column correlated with the sink.
On Llama-3.1-8B that saturated `layers.1.mlp.down_proj` and produced garbage
perplexity (up to ~12000) on wikitext windows whose first token after BOS is a
delimiter (issue #4742). To bound that, `desc_act=True` now caps each
calibration row's norm at `min(p99.9, N x median)` before forming `X^T X`,
where `N` is `HessianConfig.sink_row_cap_median_multiple` (default `4.0`).
The cap keeps the sink direction but stops it from dominating the Hessian:

```python
from gptqmodel.quantization.config import GPTQConfig, HessianConfig

qcfg = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    hessian=HessianConfig(sink_row_cap_median_multiple=4.0),  # default; 0 or None disables
)
```

The cap is a no-op whenever `act_group_aware` is active, so the default GPTQ
path is bit-for-bit unchanged. It is a mitigation, not a guarantee: still
validate the checkpoint per model, and consider dropping the sink rows
outright with `CalibrationConfig(exclude_attention_sinks=True)` below.

Tokenizers that prepend a BOS token make it an attention sink: its activation
norm can exceed the text tokens by hundreds of times, so a few rows dominate the
input Hessian and the quantizer stops optimizing for text tokens (issue #4393).
`CalibrationConfig(exclude_attention_sinks=True)` drops every attended BOS token (the
leading one and each document-boundary BOS of concatenated calibration) from
every calibration method's statistics (GPTQ, GPTAQ, FOEM, QQQ, MACaQ, EXL3, AWQ) while still running it through the forward:

```python
model.quantize(CalibrationConfig(dataset=texts, exclude_attention_sinks=True))
```

The default is `False`, which keeps the historical statistics bit for bit. A
tokenizer without a BOS token (Qwen2.5/Qwen3) has no sink to drop, so the option
changes nothing for it (issue #4459). VPTQ, Paro and staged GSQ reject the
option. The effect is model-dependent, so validate it per model. See
[calibration attention-sink exclusion](calibration-attention-sinks.md) for the
exact token rule, the flow, and measurements.

### AWQ W4A16

```python
from gptqmodel.quantization import AWQConfig

qcfg = AWQConfig(bits=4, group_size=128, sym=False)
backend = BACKEND.AWQ_TORCH
```

This writes the default AWQ GEMM format. Calibration is still required.
Other AWQ layouts have different runtime and dtype restrictions; changing only
the file's format label does not convert the packed tensors.

`FORMAT.GEMV` checkpoints are packed with the native CUDA GEMV kernel on CUDA,
with `AWQ_GEMV_TRITON` on ROCm, and with the portable Torch reader
(`BACKEND.AWQ_TORCH`) on a CPU-only host. The CPU reader is also what
`quantize()` re-homes the in-session model to after the packing loop leaves the
layers on CPU, so `quantize()` -> `save()` completes on every host and still
writes the requested GEMV checkpoint (issue #4554).

`FORMAT.GEMV_FAST` and `FORMAT.LLM_AWQ` follow the same rule: the native
CUDA GEMV_FAST kernel packs on CUDA, and the portable Torch readers
(`AwqGEMVFastTorchLinear`, `LLMAwqTorchLinear`, `BACKEND.AWQ_TORCH`) pack on
every host without that extension - a CPU-only host (#4736) and ROCm, where
hipcc cannot build it (#5109) - and serve as the in-session kernel, so
`quantize()` -> `save()` keeps the calibration and writes the requested
interleaved checkpoint. On CUDA an AUTO load still selects the native kernel;
on CPU and ROCm the same AUTO load resolves the portable reader, because these
two layouts have no Triton kernel.

### RTN W4A16 without calibration

```python
from gptqmodel.quantization import RTNConfig

qcfg = RTNConfig(bits=4, group_size=128)
backend = BACKEND.TORCH
```

RTN skips Hessian collection and error compensation. Its default export is a
GPTQ-layout checkpoint, but its quality need not match calibrated GPTQ.

### QQQ W4A8

```python
from gptqmodel.quantization import QQQConfig

qcfg = QQQConfig(bits=4, group_size=128, sym=True, desc_act=False, mse=2.4)
backend = BACKEND.QQQ
```

Use a compatible NVIDIA GPU and CUDA build for `BACKEND.QQQ`;
`BACKEND.QQQ_TORCH` provides the native Torch alternative. QQQ also supports
`group_size=-1`. Activation ordering is unsupported. INT8 activation operands
inside QQQ do not change the model's public FP16/BF16 input/output dtype.

### FP8 weights (load-only)

FP8 is a production checkpoint format, not a GPT-QModel quantization recipe.
GPT-QModel no longer quantizes or exports FP8 weights: `FP8Config` exists to
describe a checkpoint you load, and `GPTQModel.quantize()` refuses it with a
`ValueError`. Load production FP8 checkpoints with `GPTQModel.load` (or
`transformers` directly):

- `transformers` `FineGrainedFP8` and DeepSeek/Qwen block scales, which store
  `amax / 448` under `weight_scale_inv` and multiply it;
- NVIDIA ModelOpt per-tensor scales;
- MXFP8 and NVFP8 on hardware that supports them.

A loaded checkpoint may use `float8_e4m3fn` or `float8_e5m2`. Legacy
GPT-QModel checkpoints that store an inverse scale still load when they
declare `weight_scale_semantics="inverse"`; that spelling is accepted on load
and never written. The legacy `quant_method: "gptqmodel_fp8"` name is also
accepted on load and normalized to `fp8`; it is never written again.

`TorchFP8Linear` is **weight-only (W8A16)**: the packed FP8 weight tile is
dequantized for the current matmul and activations keep their FP16/BF16 input
dtype. The GPU `torch._scaled_mm` path is not used by default. It is reachable
only through the explicit
`TorchFP8Linear.set_activation_quant_method("per_token")` opt-in, which makes
the layer **W8A8** with one activation scale per token. There is no per-tensor
activation mode, because it would make a row's output depend on its batch
neighbours. See [FP8 weight-only dispatch](fp8-weight-only-dispatch.md).

### GGUF block quantization

```python
from gptqmodel.quantization import GGUFConfig

qcfg = GGUFConfig(bits=4, format="q_k_m")
backend = BACKEND.GGUF_TORCH
```

For Q4_0, use `GGUFConfig(bits=4, format="q_0")`. GGUF formats define their own
block layout; do not carry GPTQ's `group_size` into this config. The example
uses GPT-QModel's native save/load workflow. Consult the
[GGUF compatibility notes](prism-bonsai-gguf.md) for tensor types and interoperability.

### BitsAndBytes NF4, FP4, or INT8

```python
import torch
from gptqmodel.quantization import BitsAndBytesConfig

qcfg = BitsAndBytesConfig(bits=4, format="nf4", compress_statistics=False)
backend = BACKEND.BITSANDBYTES
load_dtype = torch.float32
```

Install `bitsandbytes` in the same Python environment. For FP4 use
`BitsAndBytesConfig(bits=4, format="fp4", compress_statistics=False)`; for INT8 use
`BitsAndBytesConfig(bits=8)`. NF4 and FP4 are different weight codebooks from
GPTQ INT4 and from the experimental NVFP4 activation policy.

Keep the `load_dtype` override when inserting this block. The current native
loader can cast BitsAndBytes scale tensors to FP16, although BitsAndBytes
requires FP32 scales. Compressed 4-bit statistics can also cause a
`weight_quant_state` size mismatch on reload. This example avoids those paths
by leaving statistics uncompressed and reloading in FP32. Its reloaded model
therefore uses FP32 activations, rather than the W4A16 workflow above.

### Paro W4A16

```python
from gptqmodel.quantization import ParoConfig

qcfg = ParoConfig(bits=4, group_size=128)
backend = BACKEND.PARO_CUDA
```

Paro learns rotations during calibration. Budget for optimization time
and activation memory; see [optimization scopes and checkpointing](paro.md).
The CUDA recipe requires a compatible GPU and build toolchain.

### EXL3 variable bitrate

```python
from gptqmodel.quantization import EXL3Config

qcfg = EXL3Config(bits=4.0, codebook="mcg")
backend = BACKEND.EXL3_EXLLAMA_V3
```

`bits` is a target average bitrate, not a uniform integer field width.
Quantization requires CUDA/HIP and the ExLlama V3 integration and its build
dependencies, including the Python package `pydantic`. After saving,
`BACKEND.EXL3_TORCH` is an alternative for inference;
it is not an EXL3 quantization backend.

### VPTQ vector codebooks

```python
from gptqmodel.quantization import VPTQConfig

qcfg = VPTQConfig(
    vector_lens=[-1, 8],
    num_centroids=[-1, 4096],
    num_res_centroids=[-1, 16],
)
backend = BACKEND.VPTQ_TORCH
```

This assigns 12 main index bits plus 4 residual index bits per 8-weight vector,
or nominally 2 bits per weight before codebook overhead. Read the
[VPTQ guide](vptq.md) for calibration coverage, codebook sizing, and runtime choices.

### MACaQ MXFP4 W4A4

```python
from gptqmodel.quantization import MacaqConfig

qcfg = MacaqConfig(
    n_flip_iters=256,
    had_block=32,
    pre_alpha=0.5,
    activation_precision="w4a4",
)
backend = BACKEND.MACAQ_TRITON
```

MACaQ stores nibble-packed E2M1 weights with one E8M0 scale per 32 values and
applies the same D1 balance plus block-Hadamard transform to activations at
runtime. Use `BACKEND.MACAQ_TRITON` on CUDA for the native block-scaled GEMM;
`BACKEND.MACAQ_TORCH` is the portable path (CPU, ROCm, XPU, MPS). Set
`activation_precision="w4a16"` to keep bf16 activations. See the
[MACaQ guide](macaq.md) for the recipe, kernel scope, and validation numbers.

### Experimental GPTQ INT4 with FP8 or NVFP4 activations

These require a separate workflow: explicit `activation` policy, supported Llama
decoder structure, packed INT4 layout, and compatible hardware. FP8 W4A requires
NVIDIA SM89+; NVFP4 requires GB10 / SM121. See the
[W4A workflow and activation policy reference](gptq-w4a-gb10.md)
and [accuracy measurements](gptq-w4a-accuracy-notes.md). Do not substitute
`bits=8` or `FP8Config` to enable activation quantization. These policies do not
imply that attention arithmetic or the KV cache is also entirely FP8/FP4.

## Validate quality and deploy

A successful `generate()` checks the save/load path; it does not establish
accuracy. Evaluate the original unquantized model and each quantized checkpoint
on the same held-out tasks, rendered prompts, token IDs, generation settings,
and dtype. Use the [evaluation example](https://github.com/ModelCloud/GPTQModel/wiki/Evaluation#evalution-the-current-benchmark-path)
to run task benchmarks. Compare memory use and latency separately from quality.

Before deploying to vLLM, SGLang, or Transformers, check that runtime's supported
method, format, dtype, and hardware. Native GPT-QModel support does not imply
external-engine support. Use the [integration matrix](support-matrix.md#runtime-integrations)
and keep an engine's pinned dependencies in a separate environment.

## Explore the remaining features

| Goal | Configuration / guide |
| --- | --- |
| Use several GPUs or control calibration memory | [Multi-GPU quantization](multi_gpu_quantization.md) |
| Resume an interrupted job | [Quantization checkpoints](../checkpoint.md) |
| Mix weight precisions or exclude modules | [`dynamic` rules and fused-module constraints](support-matrix.md#mixed-precision-and-fused-modules) |
| Describe quantization as staged rules | [Quantization protocol](quantization_protocol.md) |
| Requantize FP8/FP4 source modules or smooth weights | [Preprocessors](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#preprocessor-pipeline) |
| Improve GPTQ error compensation | [GPTAQ](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#foem-and-gptaq-correction) and [FOEM](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#foem-and-gptaq-correction); these modify the solve, not activation precision |
| Recover quality with adapters or train LoRA | [EoRA](eora/README.md) and [PEFT](peft.md) |
| Handle sparse MoE calibration coverage | [MoE routing and fallback controls](https://github.com/ModelCloud/GPTQModel/wiki/Multimodal-and-MoE-Models#moe-configuration-and-examples) |
| Tune normalization after quantization | [Norm tuning](https://github.com/ModelCloud/GPTQModel/wiki/Rotation-Smoothing-and-Norm-Tuning#post-quantization-norm-tuning) |
| Quantize on Apple silicon | [Native MLX workflow](mlx-runtime.md) |
| Select, pack, or repack individual layer kernels | [HF packing API](hf_packing_api.md) |
| Change the on-disk shard layout | [Reshard a checkpoint](#reshard-a-checkpoint) |

If quantization runs out of memory, start with `batch_size=1`, shorter calibration
sequences, and the memory controls in the multi-GPU guide. If no kernel accepts a
checkpoint, inspect its method, format, bits, group size, symmetry, dtype, and
device together. Start with the matching Torch backend where available; merely
renaming the backend or editing checkpoint metadata cannot convert weights.
