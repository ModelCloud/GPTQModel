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

```mermaid
flowchart LR
    A["BF16 checkpoint matrix"] --> B["gptqmodel_nf12.encode"]
    B --> C["NF12Weight packed plus escapes"]
    C --> D{"Activation and operation"}
    D -- "BF16 linear" --> E["Native nf12.linear"]
    D -- "Integer IDs" --> F["Native nf12.embedding"]
    D -- "FP32 small-row projection" --> G["linear_fp32 packed Triton"]
    D -- "FP32 prefill projection" --> L["Temporary decode plus cuBLAS"]
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
