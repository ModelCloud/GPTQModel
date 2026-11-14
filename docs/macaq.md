<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MACaQ (MXFP4 W4A4)

GPT-QModel includes a native port of **MACaQ**, Huawei CSL's model-agnostic
calibrated quantizer for low-precision weights and activations.

- Upstream: <https://github.com/huawei-csl/MACaQ> (Apache License 2.0)
- Paper: [MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision
  Weights and Activations](https://arxiv.org/abs/2611.10649), Muller et al., 2026
- Ported files: `gptqmodel/quantization/macaq.py`,
  `gptqmodel/utils/macaq_triton.py`,
  `gptqmodel/nn_modules/qlinear/macaq.py`, and
  `gptqmodel/looper/macaq_processor.py`

## What it does

MACaQ quantizes each Linear weight to OCP **MXFP4**: E2M1 values with one E8M0
scale per 32-element block. At inference the same transform is applied to the
activation, so both GEMM operands are 4-bit and the kernel is a native
block-scaled E2M1 matmul.

The per-Linear recipe is:

1. **D1 activation-aware diagonal pre-scale.** With the calibration Hessian
   `H = sum(x^T x)`, compute
   `p_j = w_std_j^alpha / act_rms_j^(1 - alpha)`, where
   `act_rms_j = sqrt(H_jj / T)`. Fold `p` into the input and `1/p` into the
   weight; it cancels in the GEMM.
2. **Block-Hadamard rotation.** Right-multiply the scaled weight by a
   block-diagonal Hadamard matrix whose block is the MXFP4 group (32 by
   default). Rotating `H` by the same transform gives the metric used below.
3. **OCP MXFP4 round-to-nearest.** Per 32-value block, set the E8M0 exponent to
   `floor(log2(amax)) - 2 + offset` and round each value to the nearest E2M1
   level.
4. **Hessian-weighted exponent screen.** Evaluate a small set of exponent
   offsets (default `(1, 0, -1)`) and keep, per `(row, block)`, the one with the
   lowest diagonal-Hessian-weighted squared error.
5. **Greedy flip refinement.** With the exponents frozen, repeatedly apply the
   best single-level E2M1 move per row. For a code change `c`, the objective
   changes by `c^2 H_jj - 2 c g_j` with `g = H (w_target - w_current)`.

The packed artifact is `W_packed` (`uint8`, two E2M1 nibbles per byte),
`W_scales` (E8M0 bytes, one per 32 values), the pre-scale vector, and the shared
block-Hadamard. The runtime reconstructs only the current tile; no dense
dequantized weight cache is retained between calls.

## Relationship to GPTQ

MACaQ shares GPTQ's calibration input (`H = sum(x^T x)`) and per-Linear
lifecycle, but it is **not** a GPTQ variant:

| | GPTQ | MACaQ |
|---|---|---|
| Weight grid | affine INT4 group scales/zeros | E2M1 + E8M0 per-32 block scale |
| Error compensation | sequential inverse-Hessian column updates | rotation + exponent screen + greedy local flips |
| Activation quantization | none (W4A16) | MXFP4 W4A4 |
| Extra transform | optional act ordering | D1 diagonal + block-Hadamard |
| Storage | int32-packed affine weights | nibble-packed E2M1 + E8M0 bytes |

Because the stored grid, the solve, and the runtime are different, the
implementation is a standalone `Macaq` quantizer and `MacaqConfig` rather than a
`GPTQ` subclass. It reuses the shared Hessian-capture and processor
infrastructure.

## Usage

```python
import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import MacaqConfig

model = GPTQModel.load(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantize_config=MacaqConfig(
        n_flip_iters=256,
        had_block=32,
        pre_alpha=0.5,
        activation_precision="w4a4",
    ),
    device="cuda:0",
    dtype=torch.bfloat16,
)
model.quantize(calibration_texts, batch_size=1, backend=BACKEND.AUTO)
model.save("./llama-3.2-1b-macaq")

quantized = GPTQModel.load("./llama-3.2-1b-macaq", backend=BACKEND.MACAQ_TRITON)
```

`activation_precision="w4a16"` keeps bf16 activations and stores only the
4-bit weights. The portable Torch path (`BACKEND.MACAQ_TORCH`) runs on CPU,
ROCm, XPU, and MPS; `BACKEND.MACAQ_TRITON` uses the native `tl.dot_scaled`
E2M1 GEMM on CUDA.

### Configuration

| Field | Default | Meaning |
|---|---:|---|
| `bits` | 4 | MXFP4 only |
| `block_size` / `group_size` | 32 | OCP MXFP4 scale group |
| `had_block` | 32 | block-Hadamard width (power of two, >= 32) |
| `do_flip` | `True` | enable greedy level refinement |
| `n_flip_iters` | 256 | maximum flip sweeps |
| `fast_flip` | `True` | active-set sweep (bit-identical to the dense sweep) |
| `ridge` | 1.0 | uniform ridge added to the flip Hessian diagonal |
| `pre_mode` | `"balance"` | `"none"` or D1 activation-aware balancing |
| `pre_alpha` | 0.5 | D1 exponent (0.5 symmetric, 1.0 weight-std only) |
| `exp_screen` | `True` | enable the H-weighted exponent screen |
| `exp_offsets` | `(1, 0, -1)` | exponent offsets to screen |
| `activation_precision` | `"w4a4"` | `"w4a4"` or `"w4a16"` |

## Validation (Llama-3.2-1B-Instruct)

Calibration: 128 Wikitext-2 training chunks (about 115k tokens), 128 flip
sweeps, 32-wide blocks, `pre_alpha=0.5`, Triton W4A4 runtime. Perplexity on 16
contiguous Wikitext-2 test chunks of 512 tokens:

| Model | Perplexity | Delta |
|---|---:|---:|
| BF16 | 20.50 | — |
| MACaQ W4A16 (same weights) | 21.67 | +5.7% |
| MACaQ W4A4 (Triton) | 24.28 | +18.4% |

These numbers are a smoke/regression baseline, not a tuned recipe: the upstream
paper uses substantially more calibration data (millions of tokens) and reports
W4A4 perplexity much closer to BF16 on larger models. W4A16 isolates the weight
error from the activation error and is the better choice when activation
accuracy matters more than the W4A4 speedup.

## Port scope

This port covers the native GPT-QModel safetensors path and the dense kernels
that path needs:

- the Torch quantizer (D1, block-Hadamard, MXFP4 RTN, exponent screen, flip
  refinement) in `gptqmodel/quantization/macaq.py`
- the Triton activation quantizer, fused rotate+quantize kernel, native
  `tl.dot_scaled` W4A4 GEMM, and W4A16 dequant GEMM in
  `gptqmodel/utils/macaq_triton.py`
- the portable Torch and CUDA module paths in
  `gptqmodel/nn_modules/qlinear/macaq.py`

The upstream repository's MoE-specific kernels, W4A8 FP8 path, cuDNN/CUTLASS
backend, Diffusers/PyTorch TTS pipelines, GGUF writer/refinement tooling, and
the vLLM plugin are not ported here. They remain available upstream; use the
Huawei CSL repository when you need those deployment targets.

## Credits

The algorithm, reference recipe, and kernel structure are from Huawei CSL's
MACaQ project. The port keeps the upstream numerical conventions and is
distributed under the same Apache-2.0 terms; see `CREDITS.md` and the file
headers for the per-file attribution.

```bibtex
@misc{muller2026macaq,
  title         = {{MACaQ}: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations},
  author        = {Muller, Lorenz K. and Bich, Philippe and Boretti, Chiara and Chang, Hyun-Min and Zhuang, Jiawei and Cavigelli, Lukas},
  year          = {2026},
  eprint        = {2611.10649},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.10649}
}
```
