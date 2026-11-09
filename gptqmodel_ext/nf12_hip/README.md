# NF12-AD HIP port (AMD RDNA4)

Decode, encode and fused matrix-core kernels plus Torch bindings that run the
**NF12-AD** (`nf12_ad`) quantization format on AMD RDNA4 GPUs — `gfx1200` /
`gfx1201` / `gfx1250` (Radeon AI PRO R9700, RX 9000 series).

Validated on **ROCm SDK 10.1.0 / HIP 7.16.26385** with
**PyTorch 2.16.0.dev20261025+rocm10.1**, free-threaded Python 3.14 and
Radeon AI PRO R9700 (`gfx1201`). `hipconfig --version` reports
`7.16.26385-0000000`; `torch.version.hip` reports `7.16.26385`.
The HIP component version is distinct from the ROCm release.

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

## Three routes

RDNA4 has no `mma.sync.m16n8k16`, no `wgmma`, no `tcgen05`, and no TMA. Porting
the CUDA fused MMA kernel verbatim would mean inventing a different accumulation
order, which breaks the invariant in SPEC.md §6.2 (`nf12_ad` must reproduce the
`bf16t` twin for a given configuration). Every route below therefore decodes
NF12 → **exact BF16** first, so none of them introduces new numerics.

| tokens | route | why |
| --- | --- | --- |
| `m <= 15` | `nf12_ad_linear_fused` | bandwidth-bound; decoding once and reusing it beats a GEMM |
| `16 <= m <= 256` | `nf12_ad_linear_wmma` | matrix cores amortise the decode across token tiles |
| `m > 256`, or no `wmma-128b-insts` | row-sliced decode-then-rocBLAS | a real GEMM wins once decode can be amortised |

### 1. Fused small-M kernel (`m <= 15`)

`nf12_ad_linear_fused` never materialises the dense weight. Each CTA owns one
16-row output block and splits K across its eight warps; the decoded super-atom
stays in registers and is reduced against the activation tile with
`v_dot2_f32_bf16` (two columns per instruction), then the per-warp partials are
summed in fixed warp order in shared memory. Every token in flight reuses the
same decoded super-atom, so the packed weight is read once per launch.

This is the route that matters most for decode: materialising the weight would
move `packed + dense_write + dense_read` = **88 MiB** for a 4096² matrix, where
a BF16 GEMM needs **32 MiB**. Reading only the packed weight is a 25% traffic
saving over BF16, which is where the wins come from.

### 2. Matrix-core fused kernel (`16 <= m <= 256`)

`nf12_ad_linear_wmma` decodes each 16×64 super-atom **once** into shared memory
and feeds `v_wmma_f32_16x16x16_bf16_w32_gfx12` for every token tile in flight,
so the packed bytes are read exactly once and the dense weight never exists.
Accumulation is FP32 in the matrix core and bias joins the accumulator before
the single BF16 rounding.

**Fragment layout.** `v_wmma_f32_16x16x16_bf16_w32_gfx12` takes 8 BF16 A, 8
BF16 B and 8 FP32 C/D per lane. The map was determined empirically on gfx1201
(one-hot probes, then asserted against a CPU reference) and is:

```
A reg (l,i) = A[ l % 16 ][ kb + i + 8 * (l / 16) ]
B reg (l,i) = B[ i + 8 * (l / 16) ][ colbase + l % 16 ]
D reg (l,i) = D[ i + 8 * (l / 16) ][ colbase + l % 16 ]
```

With `B = W^T`, `B reg (l,i) = W[ colbase + l%16 ][ kb + i + 8*(l/16) ]`, i.e.
the **same** 16-byte row fragment as A. That is why the decoded super-atom is
staged row-major: one 16-byte load then serves as B for every token tile.

**Tiles per CTA (`kMt`) is a template parameter** chosen from `m`, so `m=32`
runs a 2-tile kernel instead of paying for 8 tiles of matrix-core work. A
runtime `n_tiles` loop defeats `#pragma unroll`, spills `acc[]` to local memory
and measured 3–5× slower, which is why this is a template and not a parameter.

**Split-K** over `grid.z` uses disjoint K slices and separate FP32 partial
planes; a small reduce kernel sums them and rounds to BF16 exactly once, so the
result is identical to a single-pass FP32 accumulation followed by one rounding.
Atomics measured 2.6–3.8× slower. Each call obtains its partial workspace from
PyTorch's stream-aware caching allocator; directly calling `hipMalloc`/`hipFree`
per launch had cost +110 µs at m=32 and erased the win.
The current split planner uses CTA count, `m`, and K-block count. It selects
eight splits for very narrow launches, two for very wide launches, and can use
four for wide, short-K launches at small `m`; other shapes may remain unsplit.

**Double buffering** is enabled only for `kMt <= 6`. Decoding the next
super-atom into the idle buffer while the busy one is consumed removes the
barrier between the store and the first matrix-core read. It is not free: the
second buffer costs 16 KiB of LDS and pushes VGPRs from 67–167 to a flat
169–188, which halves resident CTAs per CU. At `kMt=8` that made m=128 regress
184 → 289 µs, so the two variants are selected by tile count at compile time.

### 3. Row-sliced decode-then-GEMM (`m > 256`)

Per SPEC.md §6.5 the matrix is decoded in **row slices** (escape indices are
absolute, so every slice decodes with the full table). Slices are sized to at
least 1024 rows — smaller slices make the GEMM launch-bound and lose more than
the cache locality buys.

## Encoding

`nf12_ad_encode.hip` ports the CUDA analyze/count/scan/write pipeline, with a
hand-written block-wide exclusive scan because ROCm ships no CUB. It is
**byte-identical** to `nf12_codec.encode`: same `exp_base`, same packed bytes,
same escape order. Encoding a 4096² matrix drops from 809 ms to 1.4 ms, and
11008×4096 from 1309 ms to 21.4 ms, which removes the CPU codec from the
quantization critical path. A forced `exp_base` still routes through the codec.

`scripts/validate_nf12_ad.py` calls `hip.encode()` and compares packed bytes,
escapes and the decode round-trip against the reference codec, so the native
encoder is gated rather than assumed.

## Measured performance

The measurements below are from #3644 on Radeon AI PRO R9700 (`gfx1201`, 32
CUs, wave32), before the per-call split-K workspace change in #3635. They are
the median of the best of several trials, compared against
`torch.nn.functional.linear` on the dense BF16 weight. Reproduce with
`scripts/validate_nf12_ad.py --backend hip`.

### Read the cache geometry before reading the table

This part has 8 MiB of L2 **and** a 64 MiB memory-side last-level cache that
AMD markets as Infinity Cache (LLVM/DRM call it MALL). `amd-smi static --cache`
reports it as `L3 = 65536 KB`, and a measured hot-reread sweep is flat at
~1.4 TB/s to 65 MiB, then falls to 1190 GB/s at 68 MiB and 607 GB/s at 96 MiB,
i.e. a cliff at ~65–66 MiB.

Two consequences:

* A BF16 weight of 32 MiB (4096²) is **fully cache-resident** and rocBLAS
  serves it at ~1.4 TB/s. NF12 saves only 1.33× of that traffic while paying
  decode cost, so it cannot win in that regime.
* `torch.cuda.get_device_properties().L2_cache_size` returns 8 MiB. That is the
  real L2 and is **not** a bug, but it is not the outermost cache either. HIP
  exposes no LLC field; `gptqmodel.rocm_llc` back-ports
  [pytorch#203965](https://github.com/pytorch/pytorch/pull/203965) so
  `last_level_cache_size` reports 64 MiB and Inductor stops sizing its flush
  buffer from 8 MiB.

So the same kernel wins or loses depending on whether the weight fits in the
LLC. Both regimes are reported below.

### Warm / cold on the canonical CUDA benchmark grid

Same `k x n` rows as `bench/common.py`, so the HIP numbers line up with the
CUDA ones. Each cell is **warm / cold**: *warm* reuses one weight in a loop,
*cold* cycles several so the working set exceeds 96 MiB — which is what a
real multi-layer model does.

| k × n | m=1 | m=8 | m=16 | m=32 | m=64 | m=128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096×4096 | 1.20 / 1.26 | 0.94 / 1.17 | 1.01 / 1.13 | 0.79 / 1.04 | 0.68 / 0.96 | 0.82 / 0.92 |
| 4096×14336 | 1.34 / 1.34 | 1.29 / 1.29 | 1.23 / 1.23 | 1.17 / 1.16 | 1.15 / 1.15 | 0.86 / 0.77 |
| 14336×4096 | 1.34 / 1.33 | 1.30 / 1.30 | 1.32 / 1.32 | 1.25 / 1.24 | 1.20 / 1.21 | 1.08 / 1.08 |
| 4096×28672 | 1.34 / 1.34 | 1.31 / 1.31 | 1.14 / 1.14 | 1.07 / 1.06 | 0.96 / 0.96 | 0.75 / 0.67 |
| 5120×25600 | 1.38 / 1.38 | 1.36 / 1.37 | 1.21 / 1.21 | 1.14 / 1.14 | 1.04 / 1.04 | 0.79 / 0.75 |
| 25600×5120 | 1.33 / 1.33 | 1.30 / 1.31 | 1.35 / 1.35 | 1.35 / 1.34 | 1.21 / 1.20 | 1.18 / 1.11 |
| 8192×8192 | 1.69 / 1.62 | 1.27 / 1.27 | 1.30 / 1.29 | 1.29 / 1.28 | 1.25 / 1.24 | 0.80 / 0.73 |
| 8192×57344 | 1.40 / 1.41 | 1.32 / 1.32 | 1.22 / 1.22 | 1.21 / 1.21 | 1.15 / 1.14 | 0.77 / 0.74 |
| 2048×8192 | 1.39 / 1.36 | 1.02 / 1.22 | 1.02 / 1.23 | 0.93 / 1.19 | 0.73 / 1.11 | 0.64 / 0.74 |
| 896×4864 | 1.70 / 1.82 | 0.96 / 1.06 | 1.26 / 1.25 | 1.20 / 1.65 | 0.80 / 1.39 | 0.77 / 1.33 |

Accuracy on every row: `rms_rel ≈ 1.66e-3` (1.64–1.67e-3), the BF16
output-rounding floor. The fused linear has **0 violations** of
`|err| <= 2e-3 + 2e-3*|ref|`.

The win is bandwidth: NF12 reads 25% fewer weight bytes, so it pays off from
`m=8` upward on weights large enough that the packed read dominates —
1.14–1.35x at `m=16..64` for the 14336/25600/57344-wide shapes.

### Interpreting the measurements

* **Wins where the weight must stream from HBM.** 14336x4096 and 25600x5120 are
  real MLP shapes and the packed form is 25% smaller than BF16; NF12 wins
  1.17–1.35x warm and 1.20–1.35x cold across `m = 16..64`. At those points the
  kernel runs at ~640 GB/s, the measured HBM read ceiling (633 GB/s), so the
  win is bandwidth, not luck.
* **Wins cold where BF16 fits in the 64 MiB LLC.** At 4096x4096 the BF16 weight
  is 32 MiB and cache-resident at ~1.4 TB/s, so warm it is a 0.87x at m=16.
  Cold, where rotation evicts it, NF12 is 1.09x at m=16 and 0.69-1.07x above
  that. 2048x8192 and 896x4864 are at or above parity at m=16 (1.02x / 1.14x
  warm, 1.23x / 1.09x cold).
* **`m = 16` is now at parity on all three small shapes** (1.02x / 0.99x /
  1.25x warm, 1.12x / 1.23x / 1.25x cold). Two knobs are derived from the
  launch shape: double buffering is dropped where its 18 KiB of extra shared
  memory is what caps resident CTAs (worth 1.20x on 4096x4096 at m=16), and
  split-K goes four ways up to m=32 but not beyond, because from four token
  tiles the accumulator already caps a CU at three CTAs.
* **`m = 128` is now 0.82x on 4096x4096 and 0.77x warm / 1.33x cold on
  896x4864.** At `kMt=8` the eight accumulators are 64 VGPRs on their own and
  the kernel fits about one CTA per CU; halving the tile count and splitting M
  across CTAs doubles the packed read but restores occupancy, which pays where
  the second read is cheap (long K, or a narrow N whose doubled grid is still
  ~2.5 waves). `2048x8192` is excluded: 64 column blocks means splitting M
  makes 128 CTAs *and* reads the weight twice (58.6 -> 85.2 us).
* **The remaining gap is `m >= 32` on the three small shapes** (0.64-1.20x
  warm). Eight token tiles per CTA halve the CTA count and the extra packed
  reads only pay back once the weight leaves the cache.
* **Loses above `m = 256`.** There the CTA already covers the whole token
  block, the kernel becomes matrix-core/decode bound rather than bandwidth
  bound, and rocBLAS's larger-tile GEMM wins; the dispatcher hands larger M to
  the decode-then-GEMM route.

Reproduce the table with `bench/hip_linear.py` in
[dphnAI/nf12](https://github.com/dphnAI/nf12); `scripts/validate_nf12_ad.py
--backend hip` is the accuracy gate and the shape-coverage check.

No end-to-end model speedup is claimed. These are single-operator numbers.

## Porting notes (NVIDIA → AMD)

All of these were verified on `gfx1201` hardware, not assumed:

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
4. **`__gfx1201__` is defined only during the device pass.** A host-side guard
   on it silently reported "unsupported" and the wmma route was never taken.
   The architecture test is now a runtime `hipGetDeviceProperties().gcnArchName`
   check. For the same reason, guards must not change a *declaration* between
   the host and device passes — different mangled names fail the launch with
   "Cannot find Symbol" — so `NF12_AD_HAS_WMMA` gates only executable bodies.
5. **`__builtin_amdgcn_fdot2` is FP16, not BF16.** It compiles and runs, and
   produces silently wrong results on BF16 data. The BF16 packed dot is
   `__builtin_amdgcn_fdot2_f32_bf16`, which lowers to `v_dot2_f32_bf16`.
   That builtin needs the `dot12-insts` target feature, which only
   `gfx1200`/`gfx1201` provide. `gfx1250` has no such instruction, so the
   kernel uses an exact BF16-to-FP32 FMA fallback there; both targets compile
   from the same source, and a family-wide `PYTORCH_ROCM_ARCH` build succeeds.
6. **Bias joins the FP32 reduction, not the BF16 result.** Adding bias to the
   already-rounded output double-rounds and measured 1.00 ULP / 1.7e-3 RMS
   relative error; folding it into the accumulator before the single BF16
   rounding gives 0.64 ULP / 1.7e-3.

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
| `GPTQMODEL_NF12_AD_HIP_FORCE_DOT_FALLBACK` | Force the gfx1250 portable-dot path on gfx1200/gfx1201 for oracle validation. |
| `GPTQMODEL_EXT_VERBOSE` | Show the full hipcc command line. |
| `GPTQMODEL_NO_ROCM_LLC` | Skip the `gptqmodel.rocm_llc` back-port. |

## Files

| File | Contents |
| --- | --- |
| `nf12_ad_format_hip.cuh` | Format geometry, `nv_sel_to_amd`/`prmt`, `decode_bulk`/`decode_group`, tiling and wmma fragment helpers. |
| `nf12_ad_decode.hip` | `nf12_ad_decode` (NF12 → exact BF16), `nf12_ad_linear_fused` (small-M fused decode+GEMM), and `nf12_ad_selftest`. |
| `nf12_ad_encode.hip` | Native analyze/count/scan/write encoder, byte-identical to the Torch codec. |
| `nf12_ad_wmma.hip` | Matrix-core fused path for `16 <= m <= 256`, including split-K, its reduce kernel, and the bf16t twin. |
| `nf12_ad_moe.hip` | `moe_align` (stable counting sort), `moe_combine`, `moe_silu_mul`, `moe_decode`, `moe_grouped_operands`. |
| `nf12_ad_pack_bf16t.hip` | BF16 → bf16t twin packing (SPEC.md §4.4). |
| `nf12_ad_torch.cpp` | Host-only Torch bindings: `encode`, `decode`, `linear`, `selftest`, `arch_info`. |

`nf12_ad_torch.cpp` must stay free of torch types in the `.hip` translation
unit: the ROCm SDK here ships no thrust, so the two TUs are strictly split.

## Syncing with upstream nf12

These files are a mechanical copy of `dphnAI/nf12`'s `csrc/hip/*`, so a fix
lands once and is copied, not reimplemented. The transform is:

| Step | Rule |
| --- | --- |
| Headers | `#include "format_hip.cuh"` → `#include "nf12_ad_format_hip.cuh"` |
| Symbols | `nf12_hip_` → `nf12_ad_`, except `nf12_hip_linear_wmma_workspace_bytes` → `nf12_ad_wmma_workspace_bytes` (the name `nf12_ad_torch.cpp` and the docs already use) |
| Attribution | Prepend `// SPDX-FileCopyrightText: 2026 ModelCloud.ai`; drop the upstream `Adapted from GPT-QModel` line |
| Geometry | `nf12_ad_format_hip.cuh` drops `#include "nf12/format.cuh"` and the `static_assert`s that compare against it |
| Duplicates | `nf12_ad_decode.hip` drops the local `warp_super_atom`/`grid_blocks`, which the format header now owns |

Two constraints the upstream sources already satisfy, and that a sync must not
break:

* Every `kernel<<<...>>>` launch must be a plain identifier or template-id.
  torch's hipify rewrites launches by matching an identifier, so a launch
  through an indexed expression is silently mangled.
* `nf12_ad_decode.hip` and `nf12_ad_wmma.hip` must not include a public header
  that drags in CUDA-compat aliases; the queries the launchers need are
  forward-declared in the file.

MoE and the bf16t twin are vendored for parity but not exposed through
`torch.ops.nf12_ad_hip`: GPT-QModel quantizes MoE expert linears individually,
so there is no fused-MoE consumer, and the C ABI is the integration point.
`_SOURCE_FILES` in `gptqmodel/utils/nf12_ad_hip.py` derives the compiled source
list from itself, so a new file is either built or not staged at all.
