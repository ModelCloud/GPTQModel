<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# VPTQ (Vector Post-Training Quantization)

VPTQ compresses each weight matrix with learned vector codebooks: weights are
split into sub-vectors, every sub-vector is replaced by the index of its
nearest codebook entry, and an optional residual codebook refines the result.
GPT-QModel implements the recipe from
[microsoft/VPTQ](https://github.com/microsoft/VPTQ)
([paper](https://arxiv.org/abs/2409.17066)) as a Torch-only quantization and
inference path (`METHOD.VPTQ` / `FORMAT.VPTQ` / `BACKEND.VPTQ_TORCH`).

The pipeline is:

1. Collect the per-module Hessian `H = E[x^T x]` from calibration data.
2. Optionally re-order input channels by descending Hessian diagonal
   (`enable_perm`) and normalize per-channel ranges (`enable_norm`).
3. Learn a per-group codebook with weighted k-means over weight sub-vectors.
4. Run a GPTQ-style sequential pass that compensates each column's vector
   quantization error with the inverse Hessian.
5. Optionally learn a residual codebook over the remaining error and repeat
   the pass with a two-stage (main + residual) lookup.

## Do I need to pre-generate Hessians?

No. GPT-QModel accumulates the Hessian online while it streams the calibration
dataset through the model, exactly like GPTQ. You only need to pass a
calibration dataset to `model.quantize(...)`; no offline Hessian files are
required. Quantization without calibration is not supported for VPTQ and
raises a `ValueError`.

## Calibration coverage and fallback

Modules that never receive calibration traffic (for example MoE experts that
are never routed to) cannot produce a Hessian. By default GPT-QModel prunes
those modules from VPTQ quantization and keeps them dense. If you configure
`fallback` (see `Fallback`/`FallbackStrategy`), low-coverage modules are kept
and quantized with the Hessian-free fallback instead: plain vector
quantization with uniform codebook weights and no Hessian error compensation.
This is the VPTQ analogue of the RTN fallback used by GPTQ, and it is also
applied automatically when a module's Hessian is unusable (all-zero or
non-finite), so a never-invoked module is never silently zeroed.

## Example: Llama 3.2 1B Instruct

```py
from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import VPTQConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-VPTQ"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

qcfg = VPTQConfig(
    vector_lens=[-1, 8],        # [-1, 8] = 8-weight vectors, no outlier codebook
    num_centroids=[-1, 4096],   # 4096 main codebook entries
    num_res_centroids=[-1, 16], # 16-entry residual codebook
)

model = GPTQModel.load(model_id, qcfg)
model.quantize(calibration_dataset, batch_size=1, backend=BACKEND.VPTQ_TORCH)
model.save(quant_path)
```

The default configuration stores 12 main bits plus 4 residual bits per
8-weight vector, i.e. a nominal 2 bits-per-weight label. The derived label is
reported as `config.bits` and can be tuned by changing the codebook sizes
(`num_centroids`, `num_res_centroids`) and `vector_lens`.

## Configuration reference

| Field | Default | Meaning |
| --- | --- | --- |
| `vector_lens` | `[-1, 8]` | `[outlier_vector_len, vector_len]`; `-1` disables the outlier codebook. `vector_len` is the number of weights per sub-vector. |
| `num_centroids` | `[-1, 4096]` | `[outlier_num_centroids, num_centroids]`; each entry must be a power of two, `-1` disables. |
| `num_res_centroids` | `[-1, 16]` | `[outlier_res_centroids, num_res_centroids]`; residual codebook sizes, `-1` disables the residual stage. |
| `npercent` | `0.0` | Percentage of input channels assigned to the outlier codebook. Requires `vector_lens[0] > 0`. |
| `group_size` | `-1` | Number of input channels per codebook. `-1` lets `group_num` decide. |
| `group_num` | `-1` | Number of codebooks. `-1` lets `group_size` decide; cannot be combined with `group_size`. |
| `kmeans_mode` | `"hessian"` | Sample weighting for codebook learning: `"hessian"` (Hessian-diagonal weighted, the reference default) or `""` (uniform). |
| `kmeans_iters` | `100` | Maximum Lloyd iterations per codebook. |
| `kmeans_tol` | `1e-5` | Relative centroid-movement tolerance for early stop. |
| `kmeans_seed` | `0` | Seed for deterministic centroid initialization. |
| `enable_norm` | `False` | Per-channel min/max normalization before vector quantization. |
| `norm_dim` | `0` | Normalization axis: `0` (columns) or `1` (rows). |
| `enable_perm` | `False` | Re-order input channels by descending Hessian diagonal before quantization. |
| `block_size` | `128` | GPTQ error-propagation block size. |
| `percdamp` | `0.01` | Hessian damping percentage, applied once to the mean-normalized Hessian by the error-propagation pass (`percdamp=0` adds no damping). |

`bits` is derived from the codebook configuration and is not a storage
parameter; passing an explicit `bits` only logs the derived value. Per-layer
`dynamic` overrides are not supported for VPTQ.

Configuration constraints:

- `num_centroids[1]` must be a power of two of at least `2`, and `npercent > 0`
  additionally requires `vector_lens[0] > 0` and `num_centroids[0]` of at least
  `2`. A one-entry codebook needs zero index bits and cannot be packed.
- `num_res_centroids[1]` must be `-1` (disabled) or a power of two of at least
  `2`. A one-entry residual codebook needs zero index bits and is rejected
  rather than silently disabling residual refinement.
- The main and residual index widths together may not exceed 32 bits, the width
  of one packed word.
- `num_res_centroids[0]` must stay `-1`: the runtime has no residual codebook
  for the outlier group.
- `group_num` may not exceed the input channels available for grouped
  quantization (after the outlier split when the outlier codebook is enabled);
  a larger value would leave a codebook group without any channel and is
  rejected before quantization starts.
- With `group_num`, channels left over by the division are added to the outlier
  set while the outlier codebook is enabled. Without an outlier codebook the
  remainder stays in the last group, and the last group's zero padding is
  dropped when the weight is reconstructed. The same holds for a trailing
  partial group when `group_size` is set explicitly.

## Supported modules

VPTQ quantizes `nn.Linear` and `transformers.Conv1D` layers. Convolution
modules (`nn.Conv1d`, `nn.Conv2d`) are rejected: the runtime reconstructs a
dense linear weight and cannot preserve convolutional semantics. `transformers.Conv1D`
stores its weight as `(in_features, out_features)` and is transposed into the
same `F.linear` layout as a linear layer.

LoRA/EoRA adapters are supported: when a model is loaded with an adapter, the
kernel adds the adapter projection to the dequantized linear output.

## Quality and codebook size

Reconstruction quality is driven by the codebook sizes and the vector length.
The default (`4096` main + `16` residual centroids over 8-weight vectors,
2 bits per weight) is a compact configuration; quality degrades sharply if the
codebooks are shrunk much below it. The reference implementation's
higher-quality ~3-bit recipe is:

```py
qcfg = VPTQConfig(
    vector_lens=[-1, 8],
    num_centroids=[-1, 65536],
    num_res_centroids=[-1, 256],
    enable_perm=True,
    enable_norm=True,
    kmeans_mode="hessian",
)
```

`enable_perm` re-orders input channels by Hessian importance and `enable_norm`
normalizes per-channel ranges before vector quantization; both are optional
and trade extra quantization time for accuracy.

## Inference and checkpoint compatibility

Checkpoints saved by GPT-QModel contain `FORMAT.VPTQ` and can be loaded with
`GPTQModel.load(quant_path, backend=BACKEND.VPTQ_TORCH)`. The runtime keeps the
codebooks and packed indices in the checkpoint and dequantizes on the fly
(weight-only, no activation quantization). FP16, BF16, and FP32 inference are
supported on CPU, CUDA, and other Torch devices.

VPTQ checkpoints are not understood by Transformers, vLLM, or SGLang, and the
checkpoint/resume feature is not available for VPTQ quantizations.
