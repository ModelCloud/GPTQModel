# GGUF Triton packed metadata cache

`GGUFTritonKernel` decodes compact scales, minima, and bit layouts from packed
GGUF `qweight` blocks once per device and format. The decoded metadata remains
packed or block-sized; a full dequantized layer is never retained. The cache
must be rebuilt whenever the source `qweight` changes.

## Cache flow

```mermaid
flowchart TD
    A["GGUFTritonKernel._forward_triton"] --> B["_get_triton_cache for device and qtype"]
    B --> C["tensor_cache_key of qweight"]
    C --> D{"Stored key matches?"}
    D -- "Yes" --> E["Reuse compact metadata"]
    D -- "No" --> F["_build_triton_cache decodes packed blocks"]
    F --> G["Replace old cache entry"]
    E --> H["Launch GGUF Triton kernel"]
    G --> H
    I["In-place qweight write"] --> J{"Inference tensor?"}
    J -- "No" --> K["PyTorch version increments"]
    K --> C
    J -- "Yes" --> L{"Caller cleared cache?"}
    L -- "Yes" --> F
    L -- "No" --> X["Stale metadata until clear_weight_cache"]
    classDef failure fill:#fce8e6,stroke:#a33,color:#111
    class X failure
```

`tensor_cache_key` includes tensor identity, version, storage pointer, device,
dtype, shape, and inference status. Normal in-place writes increment the
version; buffer replacement changes the identity. The base quantized-linear
class clears derived caches on checkpoint load and device/dtype moves. Packing
and `GGUFTritonKernel.clear_weight_cache` also clear this cache. Inference
tensors have no version counter, so callers that mutate them inside
`torch.inference_mode()` must call `clear_weight_cache()` afterward.

| Mechanism | Before | After |
| --- | --- | --- |
| Cache hit | `gguf_triton.py:1832-1839` compared only `qweight.data_ptr()`; an in-place write kept the same pointer and reused old metadata | `_get_triton_cache` compares `tensor_cache_key(qweight)` and rebuilds when a versioned weight changes |
| Cache entry | `gguf_triton.py:1745-1829` stored the source pointer in each format-specific metadata dictionary | `_get_triton_cache` stamps one key on every rebuilt entry, independent of format |
| Inference tensor write | Pointer did not change, so stale metadata remained | The documented explicit `clear_weight_cache()` contract invalidates metadata |

The old line references are from the implementation before this cache
correction. Tests replace packed Q1_0, Q1_0_g128, Q4_K, Q5_K, and Q6_K
weights in place, then compare the fused GPU result with an independent NumPy
GGUF decoder and FP64 Torch dot product. They also cover buffer replacement,
checkpoint load, device move, repeated cache hits, and the explicit-clear
contract for inference tensors.

On an NVIDIA GB10 (SM 12.1), PyTorch 2.15.0+cu132, and Triton 3.9.0,
Q4_K with K=N=1024 retained 655,360 bytes (0.625 MiB) of compact metadata
after cache construction. Cold construction peaked 0.703 MiB above the
packed-weight baseline; the additional expanded-weight cache remained 0
bytes. With resident FP16 inputs, five warmups, and the median of seven sets
of 30 CUDA-event timed forwards, the old pointer-only cache-hit logic and
the new tensor-key logic measured:

| Rows | Pointer-only hit | Tensor-key hit | Hot peak allocator delta |
| --- | ---: | ---: | ---: |
| 1 | 0.0817 ms | 0.0846 ms | 0.002 MiB |
| 16 | 0.0817 ms | 0.0847 ms | 0.031 MiB |

The benchmark used the same fused Q4_K kernel for both paths and changed
only the cache-hit check. No weight was updated during timing.

This cache contract was corrected in
[PR #3664](https://github.com/ModelCloud/GPTQModel/pull/3664).

## Q4_K and Q5_K scale range

The packed `d` and `dmin` values are FP16, but their products with 6-bit
subscales can exceed FP16 range. A later subtraction can still leave a finite
weight. Some completed weights also exceed FP16 even though a small activation
gives a finite output. `_build_triton_cache` now uses the shared
`_k_scale_overflows` guard to keep the compact metadata arrays in FP32 for
such blocks. The fused Q4_K and Q5_K kernels reconstruct the weight in that
dtype and use an IEEE FP32 dot. Ordinary packed weights keep FP16 metadata
and the existing FP16 dot path.

```mermaid
flowchart TD
    A["_get_triton_cache detects a changed packed weight"] --> B["_build_triton_cache for Q4_K or Q5_K"]
    B --> C["_k_scale_overflows checks packed d and dmin"]
    C --> D{"FP16 intermediate can exceed range?"}
    D -- "Yes" --> E["Build FP32 scale and minimum metadata"]
    D -- "No" --> F["Build FP16 scale and minimum metadata"]
    E --> G["Fused kernel reconstructs weight and dots in FP32"]
    F --> H["Fused kernel reconstructs weight and dots in FP16"]
    G --> J["Accumulate output in FP32"]
    H --> J
    K["In-place packed weight update"] --> A
    B -. "Old FP16-only path" .-> X["Infinite cached minimum and wrong output"]
    classDef failure fill:#fce8e6,stroke:#a33,color:#111
    class X failure
```

| Mechanism | Before | After |
| --- | --- | --- |
| K scale metadata | `gguf_triton.py:1796-1826` multiplied `d` and `dmin` by subscales in FP16, so Q5_K could cache an infinite minimum | `_build_triton_cache` keeps the compact scale/minimum arrays in FP32 when `_k_scale_overflows` flags the packed weight |
| Weight reconstruction | `gguf_triton.py:702,801` used FP16 scale/minimum values before the Q4_K/Q5_K dot product | The fused kernels reconstruct and dot in FP32 for flagged blocks, so even a completed weight above FP16 range can contribute to a finite output |

The old line references describe the code before this range correction. On
the NVIDIA GB10, a crafted Q5_K block with `d=74`, 6-bit scale and minimum
`63`, and an encoded maximum code returned `-inf` before the correction even
though an independent NumPy GGUF decoder and FP64 Torch dot product gave a
finite `58.5625`. The corrected Q4_K and Q5_K results both gave `58.5625`.
The GPU regression covers FP16/BF16 inputs, one and eight rows, bias,
repeatability, completed weights above FP16 range, direct wrapper calls,
and packed-weight replacement. For the 64×256 case, the wide
scale and minimum arrays add 2,048 bytes to the compact cache; no expanded
weight is retained.

With normal packed weights at K=N=1024, resident FP16 inputs, five warmups,
and the median of seven sets of 20 CUDA-event timed calls, the hot path was:

| Format, rows | Before | After | Peak allocator delta | Retained compact cache |
| --- | ---: | ---: | ---: | ---: |
| Q4_K, 1 | 0.0872 ms | 0.0868 ms | 0.002 MiB | 655,360 bytes |
| Q4_K, 16 | 0.0866 ms | 0.0868 ms | 0.031 MiB | 655,360 bytes |
| Q5_K, 1 | 0.0913 ms | 0.0914 ms | 0.002 MiB | 786,432 bytes |
| Q5_K, 16 | 0.0920 ms | 0.0915 ms | 0.031 MiB | 786,432 bytes |

The FP32 arrays are selected only for packed blocks whose intermediate range
requires them. These measurements cover the ordinary FP16 metadata path.
This Q4_K/Q5_K range correction was introduced by
[PR #3668](https://github.com/ModelCloud/GPTQModel/pull/3668).

## Q6_K signed scale range

Q6_K stores a signed 8-bit subscale for each 16-element group and a block
scale `d` in FP16. Its decoded code is in `[-32, 31]`. A packed block with
`d=1000` and subscale `127` has a finite effective scale of `127000`, beyond
FP16 range. The previous cache rounded that scale to infinity, so even a zero
code produced `0 × inf = NaN`. Nonzero codes can have weights beyond FP16
while a small FP16 activation still gives a finite output.

The same wide-weight case also affected `GGUFTorchLinear`: its dense and
chunked CUDA paths returned infinity for a finite dot product. They now use
the cached Q6_K bound and FP32 arithmetic only when needed, as shown in the
[K-scale overflow cache flow](gguf-k-scale-overflow-cache.md#finite-output-cuda-fallback-for-k-quants).

`_build_triton_cache` computes Q6_K effective scales in FP32. It keeps the
compact scale array in FP32 when the largest possible signed-code product for
any group exceeds FP16 range; otherwise it converts the array to FP16. The
fused Q6_K kernel reconstructs and dots in IEEE FP32 only for the wide array.

```mermaid
flowchart TD
    A["Packed Q6_K qweight"] --> B["_build_triton_cache computes d times signed subscale in FP32"]
    B --> C{"Any absolute effective scale times 32 exceeds FP16 range?"}
    C -- "Yes" --> D["Keep compact FP32 scale array"]
    C -- "No" --> E["Store compact FP16 scale array"]
    D --> F["_gguf_q6_k_fused_matmul_kernel_impl uses IEEE FP32 dot"]
    E --> G["_gguf_q6_k_fused_matmul_kernel_impl uses FP16 dot"]
    F --> H["Accumulate output in FP32"]
    G --> H
    I["Changed qweight"] --> J["_get_triton_cache rebuilds scale array"]
    J --> B
    B -. "Old FP16 multiplication" .-> X["Infinite scale and NaN for zero code"]
    classDef failure fill:#fce8e6,stroke:#a33,color:#111
    class X failure
```

| Mechanism | Before | After |
| --- | --- | --- |
| Q6_K cache build | `gguf_triton.py:1835-1837` multiplied FP16 `d` by signed subscales and could cache infinity | `_build_triton_cache` multiplies in FP32, then selects FP16 or FP32 for the compact scale array |
| Q6_K fused dot | `gguf_triton.py:906-907` reconstructed the weight and dotted in FP16 | `_gguf_q6_k_fused_matmul_kernel_impl` reconstructs and dots in FP32 for wide scales |

On an NVIDIA GB10, the independent NumPy GGUF decoder and FP64 Torch dot
product give zero for code `0` and `496.09375` for code `1` with the block
above and FP16 input `2**-16`; the latter rounds to `496` in FP16. The
previous fused path returned NaN for code `0`.
GPU tests cover codes `0`, `1`, `-32`, and `31`; positive and negative
subscales; FP16 and BF16; one and eight rows; bias; direct wrappers; ordinary
metadata; and packed-weight replacement. The 64×256 wide scale array adds
2,048 bytes to the compact cache and retains no expanded weights. For
K=N=1024, the wide cache uses 1,048,576 bytes versus 917,504 bytes for the
ordinary FP16 path. Resident-input M=1/M=16 hot forwards measured
0.0884/0.0881 ms for the wide path; ordinary old/new measurements stayed
within timing noise (M=1 0.0877/0.0872 ms, M=16 0.0875/0.0867 ms). The
benchmark used five warmups and median CUDA-event timing of seven groups of
20 forwards; peak allocator deltas were unchanged.

This Q6_K range correction was introduced in
[PR #3670](https://github.com/ModelCloud/GPTQModel/pull/3670).
