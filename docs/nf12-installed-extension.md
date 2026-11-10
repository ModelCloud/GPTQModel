<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NF12 vendored runtime

GPT-QModel ships the lightweight `gptqmodel_nf12` Python API and the vendored
CUDA sources in one wheel. Inference callers can encode a BF16 matrix once and
keep only packed NF12 storage. The legacy `gptqmodel.utils.nf12_ad` entry points
delegate to the same package; callers no longer need an installed `nf12` wheel.

```mermaid
flowchart TD
    A["gptqmodel_nf12.encode or linear"] --> B["_library.load"]
    B --> C{"Complete nf12 Torch ops registered?"}
    C -- yes --> D["Reuse registered operators"]
    C -- no --> E["_jit.library_directory"]
    E --> F{"NVIDIA CUDA and SM80+ with build tools?"}
    F -- yes --> G["Build vendored sources in source-keyed cache"]
    G --> H["Load codec then Torch extension"]
    H --> I["Run native BF16 linear, embedding, or MoE"]
    D --> I
    F -- no --> J["Raise a specific runtime error"]
    J --> K["Legacy nf12_ad.available returns false"]
    classDef failure fill:#f9d8d8,stroke:#b42318,color:#111;
    class J,K failure;
```

`gptqmodel_nf12.linear_fp32` selects a separate Triton kernel for FP32
activations. It decodes bounded weight fragments in registers. Selected output
rows use a short-K tiled kernel; small full projections use the general packed
projection. Large input batches use a call-local decode and cuBLAS projection;
the temporary dense weight is released after that call. No path retains a
dense weight between calls. `NF12Weight.narrow_rows`
views a 16-row-aligned packed range and shares the original escape table.
`linear_bf16_fp32` uses packed BF16 Tensor Core tiles with FP32 output for
short-K multirow heads; its other shapes use the FP32 packed dispatch.
On SM80, BF16 prefill projections with `N` of 384 or 512, `K` of 5120,
and `128 <= M < 1024` decode once into a call-local BF16 buffer and use
cuBLAS. The previous fused large-M route decoded the same weights for each
input tile. The decoded buffer is released when the call returns.

```mermaid
flowchart LR
    A["BF16 checkpoint matrix"] --> B["gptqmodel_nf12.encode"]
    B --> C["NF12Weight packed plus escapes"]
    C --> D{"Activation and operation"}
    D -- "BF16 linear" --> E["Native nf12.linear"]
    E -- "SM80 medium-width prefill" --> P["Call-local BF16 decode plus GEMM"]
    D -- "Integer IDs" --> F["Native nf12.embedding"]
    D -- "FP32 small-row projection" --> G["linear_fp32 packed Triton"]
    D -- "FP32 prefill projection" --> L["Temporary decode plus cuBLAS"]
    D -- "BF16 input, FP32 score" --> M["Packed Tensor Core tile"]
    D -- "BF16 MoE" --> H["Native fused_moe"]
    A --> I{"N multiple 16 and K multiple 64?"}
    I -- no --> J["Encoding rejects shape"]
    classDef failure fill:#f9d8d8,stroke:#b42318,color:#111;
    class J failure;
```

| Previous mechanism | Current mechanism |
| --- | --- |
| `gptqmodel/utils/nf12_ad.py:173` tried `_load_installed` before `_build_and_load` at `:187` | `gptqmodel/utils/nf12_ad.py:25` delegates to `gptqmodel_nf12.load` |
| An external `nf12` wheel could own the operator registration | `gptqmodel_nf12/_library.py:39` loads GPT-QModel's vendored CUDA source build, or reuses a complete registered operator set |
| Legacy CUDA runtime exposed BF16 `linear` at `gptqmodel/utils/nf12_ad.py:318` | `gptqmodel_nf12/api.py:131` provides BF16 linear and `:142` adds `linear_fp32` for FP32 routing and score projections |

The native CUDA API supports NVIDIA SM80 or newer. GPT-QModel's separate HIP
runtime and CPU codec remain the fallbacks for the existing quantizer and
quantized linear layer. The lightweight Python API rejects ROCm explicitly so
the HIP runtime is not mistaken for NVIDIA CUDA.

The exact BF16 format requires a two-dimensional matrix with `N % 16 == 0` and
`K % 64 == 0`. Norm vectors, biases, and unaligned matrices remain BF16 in
applications that use this API. The source wheel includes the native CUDA
sources, but a first use without prebuilt binaries requires nvcc, CMake, Ninja,
and a C++ compiler; a missing toolchain fails before inference starts.

Synthetic SM80 tests use independent dense Torch oracles for BF16 linear,
embedding, FP32 projection, selected rows, and MoE. FP32 scores may differ in
the last bits from cuBLAS because reduction order changes. Serving applications
must qualify routing and token decisions at model level before enabling the new
dispatch.

The paired CUDA graph benchmark in `scripts/bench_nf12_lite_bf16.py` compares
the BF16 NF12 path with `torch.nn.functional.linear` on the same BF16 inputs and
original dense BF16 weights. It encodes before timing and replays 32 calls per
graph in ABBA order. On this host's NVIDIA PG506-230 (SM80), PyTorch
2.15.0+cu132, and GIL disabled, only 3 of 14 Lite-sized probes cleared the
requested 1.15× gate. The indexer key, Markov, and vocabulary decode shapes
passed at 1.616×, 1.191×, and 1.390×. The indexer score and ratio-one
compressor decode shapes reached 1.087× and 1.044× after the SM80 compressor
tuning. The optimized compressor prefill shape `(N,K,M)=(512,5120,512)`
improved from 0.388× to 0.815×, but
still missed the gate. Gate-shaped BF16 probes are diagnostics only: Lite's
actual MoE gates use FP32 activations and scores. A BF16 gate cannot replace
that path without separate model-level decision qualification. The benchmark
supports `--strict` to return nonzero when any shape misses the speed target.

For the 124-SM PG506-230, the six-row 512×5120 compressor decode uses five
CTA K ranks and eight in-CTA K parts. A one-super-atom-per-warp decoder serves
the same SM80 weight shape for call-local prefill. The direct decoder A/B graph
comparison was 31.056 µs (two atoms) versus 30.320 µs (one atom) for the
whole prefill projection. The new synthetic paired run measured 8.336/8.704
µs NF12/dense at M=6 and 29.856/24.320 µs at M=512; clocks and neighboring
load differed from the prior run, so compare within each pair. The real
`layers.20.attn.compressor.wkv.weight` has 412 escape groups and measured
1.034× at M=6 and 0.812× at M=512. Its prefill output was bitwise equal to
dense BF16; its six-row output passed the dense and sampled FP64 tolerances.
Neither path meets the earlier 1.15× speed target.

### Real compressor prefill gate across eight GPUs (2026-11-10)

The requested target for compressor BF16 prefill `(N,K,M)=(512,5120,512)` is
now **at least 1.05×** dense BF16 on the same GPU. This prefill-specific target
does not change the earlier 1.15× diagnostic gate for other shapes.
[`scripts/bench_nf12_real_compressor_prefill.py`](../scripts/bench_nf12_real_compressor_prefill.py)
loads the released layer-20 ratio-one compressor weight, encodes it once, and
compares the full NF12 call with `F.linear` on identical BF16 activations. It
uses 32 calls per CUDA graph and eight ABBA rounds per GPU with TF32 disabled.
The [compact result](nf12-real-compressor-prefill-8gpu-20261110.json) records
the weight hash, runtime, per-GPU times, output bits, and top-8 decisions.

All eight GPUs missed 1.05×: speedups ranged from **0.8117× to 0.8191×**,
with a median of **0.8161×**. NF12 restored every BF16 weight bit, and its
prefill output matched dense BF16 bit-for-bit on all eight runs. The new gate
requires roughly a 22% reduction in current NF12 latency at this shape.

An experimental fused-MMA sweep split tile and K configurations across the
eight GPUs. Nsight Compute showed the earlier 64×128 tile launched only 64
blocks on a 124-SM GPU, with 182 registers per thread and 49 KiB dynamic shared
memory per block. A smaller 64×64 tile improved occupancy but still measured
37.79 µs NF12 and 29.15 µs with BF16 tiled weights against 20.99 µs dense
BF16 on the same GPU. Its output also changed one top-1 row in the fixed
fixture. These are diagnostic prototypes, not a qualified dispatch change.
The production NF12 setting remains disabled.
