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

## Why decode-then-GEMM

RDNA4 has no `mma.sync.m16n8k16`, no `wgmma`, no `tcgen05`, and no TMA. Porting
the CUDA fused MMA kernel would mean inventing a different accumulation order,
which breaks the invariant in SPEC.md §6.2 (`nf12_ad` must reproduce the `bf16t`
twin for a given configuration).

Instead this port decodes NF12 → **exact BF16** with the same super-atom walk
the CUDA decoder uses and hands the result to rocBLAS (`torch.matmul`). That is
the decode-then-GEMM route of SPEC.md §6.5. Because decode is bit-exact, the
result equals the same BF16 GEMM on exactly-decoded weights — no new numerics
are introduced.

**Trade-off, stated plainly:** this path is *decode*-bound and does not beat
BF16 `torch.matmul` for every shape. It wins where the 25% smaller weight
footprint dominates (small-M, large-N/K) and loses where rocBLAS's BF16 GEMM is
already compute-bound. Measured numbers are in the PR and reproducible with
`scripts/validate_nf12_ad.py --backend hip`.

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
| `nf12_ad_decode.hip` | `nf12_ad_decode` (NF12 → exact BF16) and `nf12_ad_selftest`. |
| `nf12_ad_torch.cpp` | Host-only Torch bindings: `decode`, `linear`, `selftest`, `arch_info`. |

`nf12_ad_torch.cpp` must stay free of torch types in the `.hip` translation
unit: the ROCm SDK here ships no thrust, so the two TUs are strictly split.
