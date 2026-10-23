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
| ParoQuant | INT4 with learned rotations | FP16/BF16 | Required; includes optimization |
| QQQ | Symmetric INT4, group size 128 or -1 | Dynamic INT8 activation quantization inside the linear operation; FP16/BF16 inputs and outputs | Required |
| GGUF | Format-specific blocks, e.g. Q4_0 or Q4_K_M | Native kernels consume floating-point activations | Not required |
| FP8 | E4M3 or E5M2 weights with scales | The native Torch path consumes floating-point activations; this config alone does not select W8A8 | Not required |
| BitsAndBytes 4-bit | NF4 or FP4 | Floating-point activations | Not required |
| BitsAndBytes 8-bit | INT8 | Native INT8 matmul quantizes activations dynamically; fallback arithmetic can differ | Not required |
| EXL3 | Variable-rate codebooks; `bits` is a target average bits per weight | Floating-point model activations | Required |
| VPTQ | Vector-codebook indices plus codebooks | Floating-point model activations; effective bits per weight depend on vector and codebook sizes | Required |
| Experimental GPTQ W4A | Packed GPTQ INT4 | Explicit FP8 or NVFP4 activation policy for selected Llama decoder operations | Required for fresh quantization |

This is a recipe map, not a promise that every method, bit width, dtype, and
device combination works. Check the [method/format/backend matrix](../README.md#quant-method--format--backend-matrix-)
and [hardware table](../README.md#platform-and-hw-support-). `BACKEND.AUTO`
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
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GPTQConfig

model_id = "Qwen/Qwen3-0.6B"
quant_path = "Qwen3-0.6B-GPTQ-W4A16"
load_dtype = torch.float16
qcfg = GPTQConfig(bits=4, group_size=128)
backend = BACKEND.TORCH

# Calibration-based recipes need representative text. Streaming avoids
# downloading the entire C4 shard. Weight-only recipes skip this step.
calibration = None
if qcfg.requires_calibration_dataset():
    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
        streaming=True,
    )
    calibration = [row["text"] for row in dataset.take(256)]

model = GPTQModel.load(model_id, qcfg, device="cuda:0", dtype=torch.float16)
model.quantize(calibration, batch_size=1, backend=backend)
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

### AWQ W4A16

```python
from gptqmodel.quantization import AWQConfig

qcfg = AWQConfig(bits=4, group_size=128, sym=False)
backend = BACKEND.AWQ_TORCH
```

This writes the default AWQ GEMM format. Calibration is still required.
Other AWQ layouts have different runtime and dtype restrictions; changing only
the file's format label does not convert the packed tensors.

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

### FP8 weights

```python
from gptqmodel.quantization import FP8Config

qcfg = FP8Config(bits=8, format="float8_e4m3fn", weight_scale_method="row")
backend = BACKEND.FP8_TORCH
```

Use `format="float8_e5m2"` for E5M2. This recipe quantizes weights without
calibration; it does not enable the experimental GPTQ FP8 activation stream.

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

### ParoQuant W4A16

```python
from gptqmodel.quantization import ParoConfig

qcfg = ParoConfig(bits=4, group_size=128)
backend = BACKEND.PAROQUANT_CUDA
```

ParoQuant learns rotations during calibration. Budget for optimization time
and activation memory; see [optimization scopes and checkpointing](paroquant.md).
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
and dtype. Use the [evaluation example](../README.md#pair-with-evaluation-for-post-quantization-llm-benchmarks-)
to run task benchmarks. Compare memory use and latency separately from quality.

Before deploying to vLLM, SGLang, or Transformers, check that runtime's supported
method, format, dtype, and hardware. Native GPT-QModel support does not imply
external-engine support. Use the [integration matrix](../README.md#quantization-support-)
and keep an engine's pinned dependencies in a separate environment.

## Explore the remaining features

| Goal | Configuration / guide |
| --- | --- |
| Use several GPUs or control calibration memory | [Multi-GPU quantization](multi_gpu_quantization.md) |
| Resume an interrupted job | [Quantization checkpoints](../checkpoint.md) |
| Mix weight precisions or exclude modules | [`dynamic` rules and fused-module constraints](../README.md#dynamic-quantization-per-module-quantizeconfig-override-) |
| Describe quantization as staged rules | [Quantization protocol](quantization_protocol.md) |
| Requantize FP8/FP4 source modules or smooth weights | [Preprocessors](../README.md#preprocessors-) |
| Improve GPTQ error compensation | [GPTAQ](../README.md#using-gptaq-experimental-not-moe-compatible-and-results-may-not-be-better-than-original-) and [FOEM](../README.md#using-foem-); these modify the solve, not activation precision |
| Recover quality with adapters or train LoRA | [EoRA](eora/README.md) and [PEFT](peft.md) |
| Handle sparse MoE calibration coverage | [MoE routing and fallback controls](../README.md#moe-quantization-) |
| Tune normalization after quantization | [Norm tuning](../README.md#norm-tuning) |
| Quantize on Apple silicon | [Native MLX workflow](../README.md#native-mlx-quantization-on-apple-silicon) |
| Select, pack, or repack individual layer kernels | [HF packing API](hf_packing_api.md) |

If quantization runs out of memory, start with `batch_size=1`, shorter calibration
sequences, and the memory controls in the multi-GPU guide. If no kernel accepts a
checkpoint, inspect its method, format, bits, group size, symmetry, dtype, and
device together. Start with the matching Torch backend where available; merely
renaming the backend or editing checkpoint metadata cannot convert weights.
