# NF12-AD HIP port (AMD RDNA4)

Decode, encode and fused matrix-core kernels plus Torch bindings that run the
**NF12-AD** (`nf12_ad`) quantization format on AMD RDNA4 GPUs — `gfx1200` /
`gfx1201` / `gfx1250` (Radeon AI PRO R9700, RX 9000 series).

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
| `m <= 16` | `nf12_ad_linear_fused` | bandwidth-bound; decoding once and reusing it beats a GEMM |
| `17 <= m <= 128` | `nf12_ad_linear_wmma` | matrix cores amortise the decode across token tiles |
| `m > 128`, or no `wmma-128b-insts` | row-sliced decode-then-rocBLAS | a real GEMM wins once decode can be amortised |

### 1. Fused small-M kernel (`m <= 16`)

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

### 2. Matrix-core fused kernel (`17 <= m <= 128`)

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
Atomics measured 2.6–3.8× slower. The partial workspace is cached per process
because per-call `hipMalloc`/`hipFree` cost +110 µs at m=32 and erased the win.
Split count is picked from the **CTA count**, not `m`: splitting only pays below
~1.2 waves (32 CTAs on 32 CUs) and loses at N=11008 (86 CTAs).

**Double buffering** is enabled only for `kMt <= 6`. Decoding the next
super-atom into the idle buffer while the busy one is consumed removes the
barrier between the store and the first matrix-core read. It is not free: the
second buffer costs 16 KiB of LDS and pushes VGPRs from 67–167 to a flat
169–188, which halves resident CTAs per CU. At `kMt=8` that made m=128 regress
184 → 289 µs, so the two variants are selected by tile count at compile time.

### 3. Row-sliced decode-then-GEMM (`m > 128`)

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

All numbers are Radeon AI PRO R9700 (`gfx1201`, 32 CUs, wave32), median of the
best of several trials, compared against `torch.nn.functional.linear` on the
dense BF16 weight. Reproduce with `scripts/validate_nf12_ad.py --backend hip`.

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

### Warm: one weight reused in a loop (best case, single layer)

| N | M | nf12 µs | bf16 µs | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 17 | 41.2 | 22.9 | 0.56× |
| 4096 | 32 | 54.8 | 22.9 | 0.42× |
| 4096 | 64 | 61.9 | 26.1 | 0.42× |
| 4096 | 96 | 57.5 | 37.1 | 0.65× |
| 4096 | 128 | 74.3 | 47.3 | 0.64× |
| 11008 | 17 | 103.0 | 148.5 | **1.44×** |
| 11008 | 32 | 103.2 | 148.4 | **1.44×** |
| 11008 | 64 | 108.2 | 152.4 | **1.41×** |
| 11008 | 96 | 125.7 | 164.0 | **1.30×** |
| 11008 | 128 | 184.4 | 168.5 | 0.91× |

### Cold: several weights cycled, working set > 96 MiB

This is the regime a real multi-layer model runs in — by the time a layer is
revisited, the others have evicted it.

| N | M | nf12 µs | bf16 µs | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 17 | 54.4 | 57.7 | **1.06×** |
| 4096 | 32 | 69.3 | 57.8 | 0.84× |
| 4096 | 64 | 77.8 | 59.6 | 0.77× |
| 4096 | 96 | 78.5 | 62.3 | 0.79× |
| 4096 | 128 | 100.7 | 62.9 | 0.62× |
| 11008 | 17 | 126.8 | 149.3 | **1.18×** |
| 11008 | 32 | 127.9 | 150.3 | **1.18×** |
| 11008 | 64 | 133.5 | 158.2 | **1.18×** |
| 11008 | 96 | 141.1 | 163.2 | **1.16×** |
| 11008 | 128 | 202.8 | 167.7 | 0.83× |

### What this means, stated plainly

* **Wins where the weight must stream from HBM.** 11008×4096 is a real MLP
  shape and the packed form is 64.5 MiB against 86 MiB of BF16; NF12 wins
  1.30–1.44× warm and 1.16–1.18× cold across the whole `m = 17..96` band. At
  `m = 17..64` the kernel runs at ~640 GB/s, which is the measured HBM read
  ceiling (633 GB/s), so that win is bandwidth, not luck.
* **Loses where BF16 fits in the 64 MiB LLC.** At 4096² the BF16 weight is
  32 MiB and cache-resident at ~1.4 TB/s, so the 1.33× traffic saving cannot
  pay for the decode. Cold, where rotation evicts it, the two are close
  (1.06× at m=17, 0.62–0.84× above that).
* **Loses above `m = 96`.** At m=128 the CTA already covers 128 tokens, the
  kernel becomes matrix-core/decode bound rather than bandwidth bound, and
  rocBLAS's larger-tile GEMM wins. `nf12_ad_wmma_max_m()` returns 128 for that
  reason and the dispatcher hands larger M to the decode-then-GEMM route.

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
| `GPTQMODEL_EXT_VERBOSE` | Show the full hipcc command line. |
| `GPTQMODEL_NO_ROCM_LLC` | Skip the `gptqmodel.rocm_llc` back-port. |

## Files

| File | Contents |
| --- | --- |
| `nf12_ad_format_hip.cuh` | Format geometry, `nv_sel_to_amd`/`prmt`, `decode_bulk`/`decode_group`, tiling and wmma fragment helpers. |
| `nf12_ad_decode.hip` | `nf12_ad_decode` (NF12 → exact BF16), `nf12_ad_linear_fused` (small-M fused decode+GEMM), and `nf12_ad_selftest`. |
| `nf12_ad_encode.hip` | Native analyze/count/scan/write encoder, byte-identical to the Torch codec. |
| `nf12_ad_wmma.hip` | Matrix-core fused path for `17 <= m <= 128`, including split-K and its reduce kernel. |
| `nf12_ad_torch.cpp` | Host-only Torch bindings: `encode`, `decode`, `linear`, `selftest`, `arch_info`. |

`nf12_ad_torch.cpp` must stay free of torch types in the `.hip` translation
unit: the ROCm SDK here ships no thrust, so the two TUs are strictly split.
