<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GGUF C++ bias epilogue

The CPU and CUDA GGML backends return a quantized matrix product to
`GGUFCppKernel` or `GGUFCudaKernel`. A stored FP32 bias must be added before a
low-precision activation result is rounded. This flow was corrected in the
GGUF C++ bias epilogue PR.

```mermaid
flowchart TD
    A["GGUF C++ CPU or CUDA forward"] --> B["Run selected GGML matmul"]
    B --> C{"Native output is FP16 or BF16 and bias exists?"}
    C -- Yes --> D["Temporarily decode packed weight to FP32"]
    D --> E["FP32 matmul plus FP32 bias"]
    E --> F["Round once to activation dtype"]
    C -- No --> G["Add bias in native output dtype"]
    G --> F
    H["CPU Torch fallback"] --> I["Pass bias into fused or dense matmul"]
    I --> F
    J["Old flow"] --> K["Round native or Torch product before bias"]
    K --> L["Overflow or cancellation error"]
    classDef failure fill:#fdd,stroke:#c33
    class L failure
```

| Path | Before | After |
| --- | --- | --- |
| GGML FP32 output | Converted product to FP16/BF16 before adding bias | Adds original bias in FP32, then rounds output once |
| GGML low-precision output | Added bias to an already rounded product | Recomputes this biased case with a temporary FP32 decoded weight |
| CPU Torch fallback | Called fused/dense matmul without bias, then added it after rounding | Passes bias into the existing wide epilogue |
| CPU autotuning | Timed the product without bias | Times the complete biased path, including rare fallback |

The packed Q8_0 regression checks a 100000 product canceled by a -100000 FP32
bias against an independent FP64 decode and matmul. It covers GGML FP32 and
low-precision outputs and the CPU Torch fallback. The decoded FP32 weight is
temporary; the module retains only packed checkpoint buffers.
