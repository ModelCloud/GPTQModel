# Kernel input contiguity and launch contracts

Quantized inference kernels fall into two groups. Native and ctypes kernels
(BitBLAS, EXL3 `had_r_128`/`hgemm`/`bc`, Machete, the Paro fused rotation)
receive a raw `data_ptr()` and walk rows with a fixed `K` stride. Triton kernels
compute their own dense offsets from the same assumption. Both groups require a
packed operand, and both must launch in the CUDA context that owns that operand.

The canonical non-contiguous activation is `hidden_states[:, -1:, :]` with
batch > 1: every batched decode step produces it. `reshape(-1, K)` succeeds on
that slice and returns a strided view, so nothing raises until the kernel reads
the wrong rows (or a Triton launch rejects the pointer).

This page documents the contracts introduced by the kernel-correctness fix for
issue #4042. The flows below describe the code in that change.

## 1. Activation packing before a packed-row kernel

```mermaid
flowchart TD
    A["x: [B, 1, K] decode slice (strided)"] --> B["reshape(-1, K) returns a strided view"]
    B --> C{"backend"}
    C -- BitBLAS --> D["_invoke_bitblas_matmul passes data_ptr() to the prebuilt CUDA library"]
    C -- EXL3 --> E["had_r_128 / hgemm / bc read packed rows"]
    C -- Machete --> F["machete_mm reads packed rows"]
    C -- Paro --> G["apply_paro_rotation launch config"]
    D --> X["silently wrong rows (rel err 1.30)"]
    E --> Y["silently wrong rows (rel err 1.36)"]
    F --> Z["silently wrong rows"]
    G --> W["RuntimeError: expects contiguous floating-point input"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y,Z,W failure;
```

```mermaid
flowchart TD
    A["x: [B, 1, K] decode slice (strided)"] --> B["reshape(-1, K).contiguous()"]
    B --> C{"backend"}
    C -- BitBLAS --> D["_invoke_bitblas_matmul: enter the module device, then data_ptr()"]
    C -- EXL3 --> E["had_r_128 / hgemm / bc + TORCH_CHECK(is_contiguous)"]
    C -- Machete --> F["machete_mm"]
    C -- Paro --> G["fused rotation"]
    D --> H["correct output"]
    E --> H
    F --> H
    G --> H
```

| Kernel path | Before | Current |
| --- | --- | --- |
| BitBLAS fp16 | `bitblas.py:865` reshaped and passed the strided view's `data_ptr()` (`bitblas.py:889-899`). | `bitblas.py:875` packs the operand; the dtype cast above already copies, so this is a no-op for bf16. |
| EXL3 fp16 | `exl3.py:135` used `x.view(-1, in_features)` and the kernels assumed packed rows. | `exl3.py:139` packs; `hadamard.cu`, `hgemm.cu`, and `libtorch/linear.cpp` now reject a strided operand. |
| Machete | `machete.py:314` / `machete_awq.py:212` passed the strided view. | `machete.py:317` / `machete_awq.py:215` pack first. |
| Paro | `paro.py:593` fed the strided view to the fused rotation. | `paro.py:595` packs first. |

The native checks are deliberate: a future caller that forgets to pack now
fails closed instead of returning wrong values.

## 2. Device binding for Triton and ggml launches

A `device_map` shards one checkpoint over several GPUs. Triton launches into the
process's *current* CUDA context, so a layer whose operands live on `cuda:1`
must enter that device first.

```mermaid
flowchart TD
    A["GGUFTritonKernel.forward(x on cuda:1)"] --> B["_launch / _launch_with_meta"]
    B --> C{"with torch.cuda.device(x.device)"}
    C --> D["kernel launch in the operand's context"]
    D --> E["cast and restore kernels use the same guard"]
    E --> F["result on cuda:1"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    B -. "before: current device cuda:0" .-> X["ValueError: Pointer argument cannot be accessed from Triton"]
    class X failure;
```

The ggml CUDA bridge has two extra contracts. It runs its graph on its own
stream, and its backend calls `cudaSetDevice`.

```mermaid
sequenceDiagram
    participant Caller
    participant Torch
    participant GGML
    Caller->>Torch: enqueue producer work on the current stream
    Torch->>Torch: current_stream(x.device).synchronize()
    Torch->>GGML: ggml_backend_tensor_set(x.data_ptr())
    GGML->>GGML: graph_compute + synchronize on the ggml stream
    Torch->>Torch: restore the previous current device
    Torch-->>Caller: output tensor
```

| Path | Before | Current |
| --- | --- | --- |
| GGUF Triton | `_launch` / `_launch_with_meta` and the cast/restore launches had no device guard. | `gguf_triton.py:1004`, `gguf_triton.py:1027`, and the two `forward` launches enter `torch.cuda.device(x.device)`. |
| ggml current device | A forward on `cuda:1` left `torch.cuda.current_device()` at 1. | `gguf_cpp.py:799` saves and restores the current device. |
| ggml stream order | The input copy could read memory before the caller's stream produced it. | `gguf_cpp.py:708` synchronizes the caller's stream before the copy. |

## 3. 64-bit row offsets and capped grids

`row * K` and `row * N` were 32-bit. They wrap once `rows * K` (or `rows * N`)
exceeds 2^31, so rows past the boundary read or write the wrong memory. The
row index is now cast before the multiply.

```mermaid
flowchart LR
    A["row = tl.program_id(0)"] --> B["int32 row * K"]
    B --> C{"rows * K > 2^31?"}
    C -- No --> D["correct"]
    C -- Yes --> X["wrapped address: wrong rows"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X failure;
```

| Site | Before | Current |
| --- | --- | --- |
| QQQ `_quantize_rows` | `X + row * K` | `qqq_triton.py:15` casts the row. |
| W4A `_token_fp8_quant` | `X + row * K` | `w4a_triton.py` casts the row. |
| W4A `_grouped_fp8_gemm` | int32 row offsets | `w4a_triton.py` uses a 64-bit row index. |
| GGUF cast kernel | `row * K` | `gguf_triton.py:52` casts the row. |
| GGUF Q4_K/Q6_K fused matmuls | `offs_m * stride_xm` | `gguf_triton.py:416` and siblings cast `offs_m`. |
| NVFP4 pack/decode | int32 row and element offsets | `w4a_nvfp4_triton.py` casts them. |
| EXL3 Hadamard | `gridDim.y * 128 * blockIdx.x` in unsigned 32-bit | `hadamard.cu` uses `size_t`. |

Two launch limits are handled separately. `launch_linear_f32` and
`launch_linear_fused` in `tt_blockfloat_cuda.cu` map rows to `gridDim.y`, which
is capped at 65535; they now split large batches into row chunks and shift the
input/output pointers. The chunking preserves the weight operands, which are
indexed by column.

## 4. Empty operands

An empty batch used to reach a zero-sized grid or an ambiguous `reshape`.

```mermaid
flowchart TD
    A["x.numel() == 0"] --> B{"where"}
    B -- native Hadamard --> C["return empty output before the launch"]
    B -- EXL3 had_r_128/dual --> G["return empty output before the launch"]
    B -- LinearEXL3.forward --> H["return empty y before torch_mode"]
    B -- Torch butterfly --> D["return empty_like before reshape"]
    B -- apply_online_hadamard --> E["return x unchanged"]
    B -- NF12Linear --> F["empty_linear_output before the rotation"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    C -. "before: CUDA error invalid argument" .-> X["crash"]
    G -. "before: CUDA error invalid argument" .-> X
    D -. "before: RuntimeError ambiguous reshape" .-> X
    class X failure;
```

The native early return is in every binding in `gptqmodel_ext/hadamard/hadamard.cpp`,
in EXL3's `had_r_128`/`had_r_128_dual` (`gptqmodel_ext/exllamav3/quant/hadamard.cu`),
and in `LinearEXL3.forward` (`gptqmodel/exllamav3/modules/quant/exl3.py`);
the Torch fallback and `apply_online_hadamard` are in
`gptqmodel/quantization/rotation/hadamard_utils.py`; `NF12Linear._forward_impl`
checks `input_rows(x) == 0` before rotating.

## 5. Per-token Triton recompiles

The token count is a runtime property of a forward, not a property of the
compiled kernel. Declaring it `tl.constexpr` (or `gl.constexpr`) compiled a new
variant for every new batch size.

```mermaid
flowchart TD
    A["forward with M tokens"] --> B{"M declared constexpr?"}
    B -- Yes --> C["compile a new kernel variant for this M"]
    C --> D["40-340 ms first call"]
    B -- No --> E["reuse one variant"]
    E --> F["2.3 ms warm"]
```

| Kernel | Before | Current |
| --- | --- | --- |
| GGUF wide-output restore | `N` and `ELEMENTS` constexpr | runtime ints with `do_not_specialize` (`gguf_triton.py:68`) |
| W4A `_grouped_fp8_gemm` | `M` constexpr | runtime `M` with `do_not_specialize` (`w4a_triton.py`) |
| NVFP4 pack/decode/accumulate | `M` and `PADDED_M` constexpr | runtime ints with `do_not_specialize` (`w4a_nvfp4_triton.py`) |

## Contracts for new kernels

- A kernel that takes a pointer or computes dense row offsets requires a
  contiguous operand. Pack at the module boundary and keep a
  `TORCH_CHECK(is_contiguous())` in the native entry point.
- A Triton or ggml launch enters the device that owns its operands, and a ggml
  call restores the process's previous current device.
- Row and element offsets are 64-bit when the product can exceed 2^31.
- `gridDim.y` is capped at 65535; chunk rows instead of mapping them 1:1.
- A zero-element operand short-circuits before any launch or reshape.
- A token count is a runtime argument, never a compile-time constant.

The regression coverage lives in `tests/test_hadamard_empty_input.py`
(including EXL3's `had_r_128` and the `LinearEXL3.forward` short-circuit),
`tests/kernels/test_kernel_input_contiguity.py`,
`tests/kernels/test_gguf_triton_multi_gpu.py`,
`tests/kernels/test_triton_int32_offsets.py`, and
`tests/kernels/test_triton_per_m_recompile.py`. The EXL3 Hadamard wrap and the
block-float `gridDim.y` limit are pinned by
`tests/kernels/test_exllamav3_kernel.py` and `tests/test_tt_blockfloat_cuda.py`.

Introduced in #4042.
