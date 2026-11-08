# NF12-AD HIP port (AMD RDNA4)

Decode kernel and Torch bindings that run the **NF12-AD** (`nf12_ad`)
quantization format on AMD RDNA4 GPUs — `gfx1200` / `gfx1201` / `gfx1250`
(Radeon AI PRO R9700, RX 9000 series).

## Attribution

* **Format and reference codec:** *"NyoomFloat12: Accelerating LLM Inference
  via Lossless 12-bit Weight Compression"* by Liberman, Fang, Zhang, Dao, and
  Fu — **Together AI** (2026). <https://www.together.ai/>
* **CUDA kernels, tiled layout, and reference implementation:**
  [dphnAI/nf12](https://github.com/dphnAI/nf12) by **AlpinDale**, Apache-2.0.
* The HIP port in this directory is a GPT-QModel contribution (ModelCloud.ai).
  It is a reimplementation of the same on-disk layout, not a transliteration:
  the tiling, escape coding, and decode expressions follow AlpinDale's CUDA
  sources, but the ISA-level primitives were re-derived for AMD (see below).

The format is specified in [`../nf12/docs/SPEC.md`](../nf12/docs/SPEC.md).
A checkpoint written by the CUDA kernels loads unchanged on an AMD host and
vice versa — both consume the identical byte stream.

## Two routes

RDNA4 has no `mma.sync.m16n8k16`, no `wgmma`, no `tcgen05`, and no TMA. Porting
the CUDA fused MMA kernel verbatim would mean inventing a different accumulation
order, which breaks the invariant in SPEC.md §6.2 (`nf12_ad` must reproduce the
`bf16t` twin for a given configuration). This port therefore has two routes,
both of which decode NF12 → **exact BF16** first, so neither introduces new
numerics.

### 1. Fused small-M kernel (`m <= 16`)

`nf12_ad_linear_fused` never materialises the dense weight. Each CTA owns one
16-row output block and splits K across its eight warps; the decoded super-atom
stays in registers and is reduced against the activation tile with
`v_dot2_f32_bf16` (two columns per instruction), then the per-warp partials are
summed in fixed warp order in shared memory. Every token in flight reuses the
same decoded super-atom, so the packed weight is read once per launch.

This is the route that matters: decode-then-GEMM moves
`packed + dense_write + dense_read` = **88 MiB** for a 4096² matrix, while a
BF16 GEMM needs **32 MiB**. At small M the dense copy dominates, so nothing
that materialises the weight can reach BF16 parity. Reading only the packed
weight is a 25% traffic saving over BF16, which is where the wins come from.

### 2. Row-sliced decode-then-GEMM (`m > 16`)

Above 16 tokens the fused kernel's scalar-ish reduction loses to rocBLAS, so
the port decodes the weight and multiplies. Per SPEC.md §6.5 the matrix is
decoded in **row slices** (escape indices are absolute, so every slice decodes
with the full table). Slices are sized to at least 1024 rows — smaller slices
make the GEMM launch-bound and lose more than the cache locality buys.

**Trade-off, stated plainly:** route 1 wins 13 of 24 measured shapes, best
2.74× at `m=1` on `4096x11008`. It loses at `m <= 16` on the *smallest*
matrices (`2048x2048`, `4096x4096`), where the weight is L2-resident for both
paths and rocBLAS's GEMM is already the bottleneck, and it loses at `m >= 64`,
where re-decoding per launch cannot amortise against a pure GEMM. Those
shapes favour a real matrix-core kernel; RDNA4 exposes `wmma_bf16_16x16x16`,
which is the obvious next step and is not implemented here.

The fused path is not bit-identical to rocBLAS by construction — different
accumulation order — but it agrees with the BF16 twin to 4e-5 RMS relative
(`|err| <= 2e-3 + 2e-3*|ref|`, zero violations). Numbers are reproducible with
`scripts/validate_nf12_ad.py --backend hip`.

### Fused-kernel notes

* **`__builtin_amdgcn_fdot2` is FP16, not BF16.** It compiles and runs, and
  produces silently wrong results on BF16 data. The BF16 packed dot is
  `__builtin_amdgcn_fdot2_f32_bf16`, which lowers to `v_dot2_f32_bf16`.
* **Bias joins the FP32 reduction, not the BF16 result.** Adding bias to the
  already-rounded output double-rounds and measured 1.00 ULP / 1.7e-3 RMS
  relative error; folding it into the accumulator before the single BF16
  rounding gives 0.64 ULP / 1.7e-3.
* **Empty K slices must be zeroed.** When K is short a warp's slice can be
  empty; the reduction still reads its slot, so the shared array is cleared
  before use (an uninitialised read produced NaN).

## Porting notes (NVIDIA → AMD)

All three of these were verified on `gfx1201` hardware, not assumed:

1. **`__byte_perm` → `v_perm_b32` is not a transliteration.**
   NVIDIA numbers a 4-bit nibble per output byte over the concatenation
   `[x.b0..x.b3, y.b0..y.b3]`. AMD uses one *full byte* per output byte, where
   `0..3` selects `src1` (the **second** argument) and `4..7` selects `src0`
   (the **first**) — reversed relative to NVIDIA. `nv_sel_to_amd()` in
   `nf12_ad_format_hip.cuh` converts NVIDIA selectors at compile time, so every
   call site still lowers to a single `v_perm_b32` with a literal selector.
2. **`hip_bfloat16` is not a valid type in a hipified `.hip` TU.** torch's
   hipify rewrites the sources before compiling, and the rewritten file knows
   only `__hip_bfloat16` (from `<hip/hip_bf16.h>`); a bare `hip_bfloat16`
   silently degrades to `int`. The device code therefore uses
   `__hip_bfloat16`, and the host TU passes the output buffer as `void*` so no
   bf16 spelling crosses the hipify boundary.
3. **`__shfl_xor` and wave32.** The quad-transpose shuffle uses xor masks 1 and
   2, which stay inside a 32-lane wave, so it behaves identically to the CUDA
   version.

## Self-test

`nf12_ad_selftest` launches a real kernel that stamps the sentinel `0x4E463132`
("NF12"). It must be a kernel launch: hipcc leaves `__CUDA_ARCH__` undefined,
so a stale architecture guard would emit an empty kernel and a host-side
memcpy would not detect it. `prewarm_nf12_ad_hip_extension()` fails loudly if
the sentinel is missing.

## Building

Nothing is shipped precompiled — no `.so` in the repo or the wheel. The
extension is JIT-built from these sources for the visible gfx target on first
use, cached under
`~/.cache/gptqmodel/torch_extensions/nf12_ad_hip/<hash>/`.

```python
from gptqmodel.utils import nf12_ad_hip
assert nf12_ad_hip.prewarm_nf12_ad_hip_extension()
```

Environment knobs:

| Variable | Meaning |
| --- | --- |
| `GPTQMODEL_NF12_AD_HIP_BUILD_ROOT` | Override the build/cache directory. |
| `GPTQMODEL_NF12_AD_HIP_FORCE_REBUILD` | Rebuild even when the cache matches. |
| `GPTQMODEL_EXT_VERBOSE` | Show the full hipcc command line. |

## Files

| File | Contents |
| --- | --- |
| `nf12_ad_format_hip.cuh` | Format geometry, `nv_sel_to_amd`/`prmt`, `decode_bulk`/`decode_group`, tiling helpers. |
| `nf12_ad_decode.hip` | `nf12_ad_decode` (NF12 → exact BF16), `nf12_ad_linear_fused` (small-M fused decode+GEMM), and `nf12_ad_selftest`. |
| `nf12_ad_torch.cpp` | Host-only Torch bindings: `decode`, `linear`, `selftest`, `arch_info`. |

`nf12_ad_torch.cpp` must stay free of torch types in the `.hip` translation
unit: the ROCm SDK here ships no thrust, so the two TUs are strictly split.
