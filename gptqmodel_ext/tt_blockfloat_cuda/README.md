# Tenstorrent block-float emulation on CUDA

`ttnn` stores `bf4_b` / `bf8_b` weights as blocks of 16 values that share one
exponent. That is native storage on Tensix, so a checkpoint written that way
needs no conversion to run on the hardware — but validating one, or measuring a
model before the silicon arrives, needs a device that can execute it.

These kernels reproduce that storage exactly on an NVIDIA GPU, so a `bf4_b`
checkpoint produces the same numbers here as it will on Tensix. They are the
CUDA counterpart of `gptqmodel_ext/tt_blockfloat_hip`; the host contract in
`tt_blockfloat_cuda.h` is identical to the HIP header, and only the runtime
types differ.

The bit layout is defined once, in [`hw/dtypes.md`](../../hw/dtypes.md) section
8, and the quantiser that produces it lives in
`gptqmodel/utils/tt_blockfloat_quant.py`. This file does not restate either.
What matters for a kernel is that the exponent is a **signed `int8`** — negative
for blocks below 1.0, unlike the hardware's biased `uint8` exponent field — and
that the grid has 15 codes for `bf4_b` and 255 for `bf8_b`, not 16 and 256.

`blockfloat_quantize` reproduces `quantize_blockfloat` bit for bit, including
its three-candidate exponent search (`base`, `base - 1`, `base + 1`) and its
strict `<` tie-break. `blockfloat_decompose` is the exact inverse of
`blockfloat_dequant` and reports failure for any block that is not already on
the grid.

## The GEMM

`blockfloat_linear` has two implementations behind one entry point.

**Prefill** expands the packed weights once into a bf16/f16 scratch buffer and
hands the GEMM to `cublasGemmEx` with fp32 accumulate. A hand-written
128x128x32 `mma.m16n8k16` tile was measured at roughly half of what cuBLAS
reaches on the same shape (63 vs 146 TFLOP/s at `m = 1024`), and competing with
cuBLAS is not the interesting part of this problem.

That scratch is owned by the call: allocated from a private per-device memory
pool and freed on the calling stream, so two overlapping streams never share a
decoded weight and nothing expanded survives the forward pass. The private pool
is what makes it affordable — the default pool returns every freed block to the
driver, which puts a fresh ~34 MB allocation at ~1.9 ms, more than the GEMM it
feeds, while a pool that keeps its blocks makes the same alloc/free pair ~12 us.
It keeps *reserved* memory, not live allocations: `cudaMemPoolAttrUsedMemCurrent`
returns to zero once the stream drains.

cuBLAS is resolved with `dlopen`/`dlsym` rather than a link-time `-lcublas`.
pip CUDA wheels ship `libcublas.so.13` with no `libcublas.so` symlink, so
`-lcublas` fails to link there, and the soname follows the CUDA major. Every
process that loads this extension already has the library mapped through torch.

**Fallback** — if cuBLAS cannot be resolved, a fused dequant + `mma.m16n8k16`
kernel runs instead (128x128x32 tile, 8 warps, `cp.async` double buffering,
register-prefetched dequantise for the weight tile). It is correct but slower; it
exists so the extension still works on a bare toolkit.

`fp32` activations take a separate SIMT path and are not the performance target.

The dequantiser is at the memory roofline: 26 us for a 4096x4096 `bf4_b` unpack
(1285 GB/s), against 31 us for a plain 33.5 MB fill on the same part.

## Measured

PG506-230 (A100-class, sm_80), CUDA 13.3, `n = k = 4096`, weights quantised to
`bf4_b`. The GPU was shared with another job throughout, which swings absolute
timings by up to 3x between runs; the table is the best of six across two
sessions, and relative behaviour was stable.

| tokens | `blockfloat_linear` µs | TFLOP/s |
| -----: | ---------------------: | ------: |
|      1 |                     48 |     0.7 |
|     16 |                     52 |    10.3 |
|     32 |                     46 |    23.3 |
|     64 |                     47 |    45.7 |
|    128 |                     64 |    67.1 |
|    256 |                     93 |    92.4 |
|    512 |                    145 |   118.5 |
|   1024 |                    253 |   135.8 |

The shape of this table is the point: below ~64 tokens the cost is dominated by
reading and expanding the weights once, so time is nearly flat; above that the
GEMM dominates and throughput climbs toward what cuBLAS reaches on bf16.

## Why there is no `bf2_b`

Blackhole's LLK can pack `Bfp2_b` (`pack_as_bfp2_tiles` in
`tt-metalium/bfloat2.hpp`), so the p300c silicon can hold it, but nothing above
the kernel exposes it: `tt::tt_metal::DataType` has no `BFLOAT2_B`, and
`ttnn.types` offers only `bfloat16`, `bfloat8_b` and `bfloat4_b`. With no
consumer in the stack and no oracle to validate a kernel against, there is
nothing to emulate. Two bits are one sign and one magnitude, so a 16-value block
would hold only `{-1, 0, +1} x 2**exponent` in any case.

`bits == 2` is therefore rejected rather than silently quantising to a grid
nothing can read back.

## Building and testing

Through the library (JIT, cached under `~/.cache/gptqmodel/torch_extensions`):

```python
from gptqmodel.utils import tt_blockfloat_cuda as bf_cuda

codes, exponents = bf_cuda.quantize(weight_fp32, bits=4)
out = bf_cuda.linear(x, codes, exponents, bits=4, bias=bias)
```

Standalone, without torch, for the kernel-level checks:

```sh
nvcc -O3 -std=c++17 -arch=sm_80 -Xcompiler -fPIC -shared \
    tt_blockfloat_cuda.cu -I . -o libtt_blockfloat_cuda.so
```

`pytest tests/test_tt_blockfloat_cuda.py` compares quantise/decompose bit for
bit against the torch reference and the GEMM against a float32 matmul.
