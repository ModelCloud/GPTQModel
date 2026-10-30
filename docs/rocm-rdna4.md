<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AMD RDNA4 (gfx1200 / gfx1201 / gfx1250) support

This page describes GPT-QModel behaviour on AMD RDNA4 consumer/workstation GPUs
(for example the Radeon AI PRO R9700, `gfx1201`). The CDNA baseline is
`gfx950` (MI350X/MI355X) and `gfx942`; its tuned values are unchanged.

> **Verification status.** Measured on 4x AMD Radeon AI PRO R9700 (`gfx1201`,
> HIP 7.16.26385, torch 2.16.0.dev20261029+rocm10.1) and 2x RTX 4090
> (`sm_89`, CUDA 13.2, torch 2.15.0+cu132). Kernel numerics, the fused/split
> crossover, `v_wmma` lowering, quantization stages and P2P bandwidth below are
> **measured on that hardware**; rows without a measurement say so explicitly.
> No `gfx1200` or `gfx1250` SKU was available, so those entries share the
> `gfx1201` table by family and are **not** independently measured.

## Architecture families

Architecture detection lives in `gptqmodel/utils/arch.py`:

| Function | Meaning |
|---|---|
| `normalize_arch(name)` | strip `:sramecc+` / `:xnack-` style suffixes, lower-case |
| `rocm_arch(device)` | cached `gcnArchName` for a device, `None` off ROCm |
| `is_rocm_fused_arch(device)` | `gfx950` and `gfx120*` / `gfx125*` only |
| `arch_class(device)` | `cdna4` / `rdna4` / `rdna3` / `other` |
| `wave_size(device)` | 32 on RDNA, 64 on CDNA |
| `gate_limits(device)` | `(small_m_max, max_in_features)` for the fused GPTQ kernels |
| `gemm_fused_dtypes(device)` | activation dtypes the fused small-M GEMM is a measured win for |

`torch.cuda.get_device_capability()` is never used to classify AMD hardware; on
ROCm it returns an unrelated numeric pair.

RDNA4 is wave32 (the Triton AMD backend uses `warp_size = 32` for
`gfx_major >= 10`), while the historically tuned CDNA kernels assume wave64
scheduling. The fused GPTQ kernels therefore keep a per-architecture launch
table instead of a single module-level constant set.

## Backend support matrix

| Backend | gfx1200 | gfx1201 | gfx1250 | Notes |
|---|---|---|---|---|
| GPTQ Torch (fused GEMV / small-M GEMM / dequant) | ✅ (family table) | ✅ measured | ✅ (family table) | Per-arch launch table: RDNA4 GEMV `block_n=16/num_warps=4`, GEMM `(32, 32)/num_warps=4`, gate `(16, 17408)`; `tl.dot` lowers to `v_wmma_f32_16x16x16_f16`; **fused small-M GEMM is FP16-only** (BF16 goes through dequantize + matmul), GEMV keeps both dtypes |
| GPTQ Triton (`triton_utils/kernels.py`) | ✅ | ✅ | ✅ | portable Triton ops |
| AWQ Triton GEMM / GEMV | ✅ | ✅ | ✅ | capability-based dispatch removed for ROCm |
| QQQ | ✅ (`VPTQ`/QQQ Torch + Triton where declared) | ✅ | ✅ | numerics regression still required on hardware |
| ParoQuant Triton | ✅ explicit `PAROQUANT_TRITON` only | ✅ | ✅ | portable Triton ops; AUTO keeps the Torch `ParoLinear` kernel |
| GGUF | ✅ via `GGUF_TORCH` (`GGUF_TRITON` is CUDA-only) | ✅ | ✅ | AUTO selects the Torch path on ROCm |
| FP8 Torch | ✅ (family table) | ✅ measured | ✅ (family table) | `_scaled_mm` accepts `float8_e4m3fn`/`e5m2` and rejects the `fnuz` encodings on gfx1201; a rejected dtype is logged once and falls back to dequantize + matmul |
| VPTQ | ✅ via `VPTQ_TORCH` (`vptq_cuda` is CUDA-only and rejects ROCm explicitly) | ✅ | ✅ | |
| EXL3 | ✅ `EXL3_TORCH` only | ✅ | ✅ | tensor-core GEMM tables are `nullptr` on ROCm; AUTO selects `EXL3_TORCH` |
| Marlin / Machete / Swordfish / BitBLAS / W4A-NVFP4 | ❌ | ❌ | ❌ | explicitly unsupported on ROCm; not optimized |

### hipBLASLt is present but untuned

The ROCm SDK ships Tensile libraries for `gfx1200`, `gfx1201`, and `gfx1250`
(check `hipblaslt/library/`). The claim that "RDNA4 has no hipBLASLt library" is
incorrect; the libraries exist but are not tuned for RDNA4. Cold hipBLASLt
library loading dominates a cold quantize on CDNA, so quantization still selects
rocBLAS by default. hipBLASLt is available only as an explicit experiment:

```bash
GPTQMODEL_ROCM_QUANT_BLAS=hipblaslt python quantize.py
```

`gptqmodel.utils.rocm.quantize_tuning(device)` returns
`{"blas_backend", "amp", "prewarm_ops"}`; `amp` stays disabled on ROCm
(including RDNA4) until autotune overhead is measured on RDNA4.

### ParoQuant Triton

`ParoQuantTritonLinear` now declares `DEVICE.ROCM` alongside `DEVICE.CUDA`,
which resolves the contradiction with the HIP `libdevice` branch in
`quantization/paroquant/modules/triton/sym_ste.py`. The decision is based on
inspection, not hardware: the runtime GEMM/GEMV kernels use only portable
Triton (`tl.load`/`tl.store`, shifts, `tl.interleave`, `tl.dot`) with no
CUDA-only intrinsic, inline PTX, or libdevice call, and the launch path uses
`torch.cuda.device`, which HIP also provides.

`SUPPORTS_FORMATS[FORMAT.PAROQUANT] = 0` means `PAROQUANT_TRITON` is never
chosen by `AUTO`; selecting it explicitly is required on every platform.

**Verified on gfx1201.** Both Triton kernels (`decode` and `prefill` launch
configs) were A/B'd against an independent FP64 Torch oracle that unpacks the
4-bit codes and applies `(code - zero) * scale` itself, for `group_size` 32/64/128
in the 256x256 to 4096x4096 shape range, `M` in {1, 2, 8, 16, 32, 64}, and FP16
and BF16 activations:

| Check | Result |
|---|---|
| `paroquant_dequantize_triton` vs FP64 oracle | max abs err 7.5e-09, max rel err 6.0e-08 (FP32 output = exact) |
| `decode` / `prefill` GEMM, FP16 | arithmetic error 2.8e-04 to 3.1e-04 relative to an FP64 truth; output dtype == input dtype |
| `decode` / `prefill` GEMM, BF16 | arithmetic error 2.2e-03 to 2.4e-03 relative; output dtype == input dtype |

The BF16 relative error is dominated by the BF16 output rounding, not the
kernel: against the output rounded once to BF16 the max absolute error is
4.8e-02 for a value range whose half-ULP is 5.8e-02 (< 1 ULP), and the FP16 case
is 5.4e-03 against a 7.2e-03 half-ULP (0.75 ULP). The fused ParoQuant rotation
(`gptqmodel_ext/paroquant`) also loads and runs on gfx1201; its error versus an
FP64 no-rounding oracle is smaller than the pure-Torch reference path's
(FP16 3.7e-04 vs 4.4e-04 at K=1024, BF16 3.4e-03 vs 3.6e-03).

## Measured RDNA4 numbers (gfx1201, R9700)

Fused (product kernel) vs `dequantize_weight() + matmul`, FP16, 4-bit,
group_size 128, median of 30 iterations after warm-up:

| K | N | M | fused us | split us | speedup |
|---|---|---|---|---|---|
| 2048 | 2048 | 8 | 23.8 | 43.9 | 1.85x |
| 4096 | 4096 | 8 | 69.6 | 94.6 | 1.36x |
| 5120 | 5120 | 8 | 111.0 | 108.8 | 1.01x |
| 6144 | 6144 | 8 | 134.0 | 246.4 | 1.84x |
| 8192 | 2048 | 8 | 84.7 | 146.6 | 1.73x |
| 11008 | 4096 | 8 | 121.1 | 370.5 | 3.06x |
| 14336 | 4096 | 8 | 179.1 | 482.3 | 2.69x |
| 17408 | 5120 | 8 | 331.1 | 686.2 | 2.07x |
| 6144 | 6144 | 128 | 396.6 | 269.2 | 0.68x (split wins) |

The fused path therefore covers `max_in_features=17408` on RDNA4 while
`small_m_max` stays 16 (M>=32 was 0.98x at K=4864/N=896 and M>=64 regressed).

Launch-table sweep (K=4096, N=4096, M=8, us): `(16,16)` 86.3, `(32,16)` 87.5,
`(16,32)` 85.4, `(32,32)` 69.6, `(16,64)` 146.6, `(32,64)` 186.6. Across all
swept shapes `(32,32)` had the lowest worst-case regret (1.138x) versus
`(16,16)`'s 1.365x. `.amdgcn` for the GEMM kernel contains
`v_wmma_f32_16x16x16_f16`; the GEMV kernel has no `tl.dot` and correctly
contains none.

GEMV tile sweep (us, K=4096/N=4096, M=1): `(bn=8,nw=4)` 43.5,
`(bn=16,nw=4)` 41.8, `(bn=16,nw=8)` 42.2, `(bn=32,nw=4)` 65.4. `(16, 4)` also
won at K=2048/3584/5120; `(16, 8)` was best at K=8192/N=2048 (44.1 vs 45.6).

Quantization stage timings (`scripts/benchmark_rdna4_quant.py`,
Qwen2.5-0.5B-Instruct, 4-bit, group_size 128, 32x128 calibration tokens,
BLAS backend `rocblas` throughout; single process, one R9700):

| Method | model load (s) | quantize (s) | save (s) | total (s) | peak VRAM (MiB) |
|---|---|---|---|---|---|
| GPTQ | 0.47 | 32.39 | 0.31 | 33.62 | 413.8 |
| AWQ | 0.41 | 47.84 | 0.35 | 48.90 | 354.2 |
| QQQ | 0.41 | 17.58 | 0.46 | 18.82 | 88.0 |
| ParoQuant | 0.53 | 194.61 | 0.41 | 195.84 | 298.1 |

ParoQuant's quantize stage includes the JIT build/load of the fused rotation
extension, so it is a cold-cache number. These are wall-clock measurements of a
single run, not a tuned or repeated benchmark.

### Inference numerics (fused GPTQ kernels, gfx1201)

`tests/kernels/test_gptq_rocm_gemv.py` (20 passed) and
`tests/kernels/test_gptq_rocm_gemm.py` (43 passed) compare the fused kernels
against an independent FP64 Torch oracle at `rtol=atol=2e-3` for 2/4/8-bit,
group sizes 32/64/128, and several K/N shapes; both assert the output dtype
equals the input dtype. Dense-weight residency is checked separately: a reloaded
4-bit checkpoint exposes packed `qweight`/`scales`/`qzeros` only, no dequantized
copy is retained after forward, and `GPTQ_TORCH_CACHE_WEIGHTS` stays unset.

### Activation dtype gate for the fused small-M GEMM (gfx1201)

`scripts/benchmark_rdna4_inference.py` (4-bit, group_size 128, median speedup of
fused module forward vs `dequantize + matmul`, M=2..16). The BF16 column is the
**pre-gate** measurement, i.e. what the fused kernel did when it was still
dispatched; it is the evidence for adding the gate, not the post-gate result
(post-gate BF16 small-M dispatch count is 0 and speedup is ~1.00x):

| K | N | FP16 (median, range) | BF16 (median, range) |
|---|---|---|---|
| 2048 | 2048 | 1.16x (1.11-1.52) | 0.90x (0.72-0.96) |
| 4096 | 4096 | 1.60x (1.43-1.68) | 0.78x (0.43-1.35) |
| 5120 | 8192 | 2.51x (2.10-2.53) | 1.17x (1.03-1.42) |
| 8192 | 2048 | 1.68x (1.52-1.83) | 0.70x (0.61-0.85) |

Band medians (pre-gate): FP16 small-M 1.64x / BF16 0.85x; decode 2.37x vs 3.01x; large-M
(64/128) 0.99x vs 0.98x (no fused dispatch, as designed).

BF16 loses to the split path at the shipped RDNA4 launch tile `(32, 32)` across
the swept shapes, so `torch.py` gates the fused small-M GEMM to the dtypes
returned by `gemm_fused_dtypes()`: `{float16}` on RDNA4, `{float16, bfloat16}`
on CDNA4 (unchanged). GEMV (M=1) is faster in both dtypes on gfx1201 (FP16
1.64x-3.67x, BF16 1.40x-2.84x) and is deliberately not dtype-gated.

The gate is applied in `_can_use_rocm_gemm()`; gated-off shapes take the exact
dequantize + matmul path an unregistered architecture takes, so correctness is
unaffected. `tests/kernels/test_gptq_rocm_gemm.py` asserts this by reproducing
the visible output from an independent decode of the packed weights and
checking dispatch against `gemm_fused_dtypes`.

A per-tile sweep suggests the BF16 loss is a tile-choice effect rather than a
kernel limitation, but that sweep crashed the GPU (`HSA_STATUS_ERROR_MEMORY_FAULT`)
and is not a trustworthy basis for shipping a second launch table; retuning BF16
on RDNA4 is left as future work.

## Known limitations

- **EXL3:** the tensor-core GEMM implementation has no ROCm kernel (all entries
  are `nullptr` and the loader already raises a clear `TORCH_CHECK`). Loading
  EXL3 on ROCm uses `EXL3_TORCH`; AUTO now does the same instead of selecting
  `EXL3_EXLLAMA_V3`.
- **hipBLASLt:** libraries are shipped but not tuned for RDNA4.
- **Marlin / Machete / Swordfish / BitBLAS / W4A-NVFP4:** no ROCm support.
- `tl.dot` may silently lower to FMA instead of WMMA on AMD targets
  (`get_min_dot_size()` returns `(1, 1, 1)`), so a fused kernel's speed must be
  proven from its `.amdgcn` (`v_wmma`) and measured timings, never assumed.
- **RDNA4 BF16 small-M GEMM:** the fused kernel is not a uniform win over
  dequantize + matmul in BF16 on gfx1201 (0.43x-1.42x, median 0.85x, at the
  shipped `(32, 32)` tile), so it is disabled for that dtype. BF16 inference
  still works, it just uses the split path; M=1 GEMV and large-M matmul are
  unaffected. Retuning the tile may recover it (see above), but is unmeasured.
- **RDNA4 support is measured on gfx1201 only.** `gfx1200` / `gfx1250` share the
  same family launch table but have not been run on this hardware.

## P2P (multi-GPU) on ROCm and NVIDIA

`gptqmodel/utils/p2p.py` probes the directed peer matrix
(`torch.cuda.can_device_access_peer`), enables peer access per direction through
ctypes (`hipDeviceEnablePeerAccess`, or `cuCtxEnablePeerAccess` on CUDA), and
classifies every direction as:

| State | Meaning |
|---|---|
| `enabled` | capability true and measured bandwidth passed the gate |
| `unsupported` | capability false, or enablement failed |
| `degraded` | capability true but measured bandwidth below the gate — forced onto host staging |

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GPTQMODEL_P2P` | `auto` | `0` disables P2P, `1` forces the enablement attempt, `auto` uses the capability probe |
| `GPTQMODEL_P2P_MIN_GBPS` | `5.0` | absolute bandwidth floor below which a link is `degraded` |
| `GPTQMODEL_P2P_MIN_RATIO` | `1.0` | direct copy must be at least this multiple of pinned host staging, else `degraded` |
| `GPTQMODEL_P2P_PROBE_MIB` | `16` | size of the bandwidth probe copy |

Peer access is enabled automatically (best effort, one summary log line) before
`clone_module_for_devices()`, from `WeightOnlyLooper`, and for multi-device
`device_map` loads. A `strict=False` failure never raises.

**Measured peer bandwidth.** Both test pairs report capability `True` in both
directions but are **slower than host staging**, so both are correctly
classified `degraded` and forced onto staging:

`scripts/benchmark_multi_gpu_p2p.py --mib 128 --iters 20`:

| Pair | peer probe GB/s | pinned-host GB/s | ratio | state |
|---|---|---|---|---|
| 2x R9700 (gfx1201), 0->1 | 46.83 | 55.32 | 0.981 | degraded |
| 2x R9700 (gfx1201), 1->0 | 46.66 | 55.32 | 0.981 | degraded |
| 2x RTX 4090 (sm_89), 0->1 | 25.84 | 26.31 | 0.998 | degraded |
| 2x RTX 4090 (sm_89), 1->0 | 25.84 | 26.31 | 0.998 | degraded |

With the fallback in effect, forcing P2P on/off gives the same throughput
(54.25 vs 54.29 GB/s on gfx1201, 26.2568 vs 26.2549 GB/s on the 4090s), which is
the expected result of staging through pinned host memory.

This is why the absolute floor alone is insufficient: a low GB/s threshold
(let alone the capability bit) would have declared these links usable.

The probe times each iteration individually with a host clock and uses the
median: CUDA-event pairs under-reported a 0.35 ms copy as 300+ ms on gfx1201,
and a single batched loop reported 13 GB/s where the steady state is 50 GB/s.

**RTX 4090 note.** Consumer Ada cards commonly report
`can_device_access_peer == True` while direct throughput is poor; they are
therefore gated by measured bandwidth, not by `sm_89` or the capability bit.
`P2P != NCCL P2P`: the `gloo`/`nccl` distributed path in
`gsq_distributed.py` is separate and is validated independently.

## Verification commands

Run these on the target hardware; without a GPU they must be treated as
unverified. Use the absolute interpreters from the task environment.

```bash
ROCM_PY=/root/miniconda3/envs/gp_314t_rocm/bin/python
CUDA_PY=/root/miniconda3/envs/gp_314t/bin/python
REPO=/root/projects/GPTQModel

# CPU-only (mock) coverage, safe on any host
cd $REPO && $ROCM_PY -m pytest tests/test_arch.py tests/test_p2p.py -p no:cacheprovider

# RDNA4 kernels + numerics (requires gfx120*)
cd $REPO && $ROCM_PY -m pytest tests/kernels/test_gptq_rocm_gemv.py tests/kernels/test_gptq_rocm_gemm.py -p no:cacheprovider

# Benchmarks
cd $REPO && $ROCM_PY scripts/benchmark_rdna4_quant.py --model <local-model> --method gptq --bits 4
cd $REPO && $ROCM_PY scripts/benchmark_rdna4_inference.py --model <local-model>
cd $REPO && $ROCM_PY scripts/benchmark_multi_gpu_p2p.py --sku 4090

# 4090 (CUDA) P2P + NCCL, separately
cd $REPO && $CUDA_PY -m pytest tests/test_p2p.py tests/test_gsq_distributed.py -p no:cacheprovider
cd $REPO && $CUDA_PY scripts/benchmark_multi_gpu_p2p.py --sku 4090
```
