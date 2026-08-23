# QVQ end-to-end nsys profile — Llama-3.2-1B-Instruct (W2 `qvq_v2b2_p32`, YAQA)

Date: 2026-08-23. Branch `perf/qvq-nsys-profile` (base `0888a695`). No kernel or quantization code was modified;
all attribution comes from NVTX ranges injected at the Python op-resolver boundary by
`scripts/profile_qvq_quantize_nsys.py` and from `nsys stats` reports on the captured `.nsys-rep`.

## TL;DR — ranked conclusion

1. **`qvq_v2_segment_grid_kernel<Shift=4, Banks=2, SegmentSteps=16, FuseBoundary=1, MidpointOnly=0, __half, uint8>`
   is the single kernel to optimise next.** It runs 1,454,336 times, totals **1083.4 s = 61.2 % of all GPU kernel
   time and 55.9 % of the 1938.9 s end-to-end wall** of `qvq_quantize.main` (avg 0.745 ms, max 4.43 ms per launch).
   Its dominant call site is the `viterbi_v2_segment_family_grid_trusted` op (102,304 calls, 987.6 s of kernel time =
   55.8 % GPU / **50.9 % of wall**; 10 kernel launches per op call: 8× grid + 1× finalize + 1× codebook-norm pack),
   invoked from the YAQA v2b2 block-family selection / re-selection path in `gptqmodel/quantization/qvq.py`
   (`_qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()` at lines 2971 and 4703/4713). The same kernel
   (127.1 s) also backs `viterbi_v2_segment_tail_trusted` (39,744 calls, 134.5 s = 7.6 % GPU).
   A 4–10× speed-up of this kernel alone would remove 0.49–0.56 of the wall clock (1939 s → ≈1100–1000 s)
   for this model; anything beyond that is bounded by the remaining 44 %.
2. **`qvq_viterbi_kernel<4, 2, __half, uint8>`** (the plain L16 recurrence) is second: 128,800 launches,
   **313.5 s = 17.7 % GPU / 16.2 % of wall**, reached through two call sites — `viterbi_tail_trusted` (43,920 calls,
   175.8 s kernel, avg 4.0 ms/call) and `viterbi` via `qvq_cuda_viterbi()` (40,960 calls, 137.7 s, avg 3.4 ms/call).
3. YAQA feedback GEMMs (`cutlass_80_simt_sgemm_grouped_128x128`, cuBLAS `gemmSN_NN`, `ampere_sgemm_128x128_nt`)
   inside `yaqa_feedback` / `yaqa_feedback_update` account for 146.3 s = 8.3 % GPU. The Sketch-B preparation stage
   (`stage.prepare_yaqa`, 135.6 s wall, 7.0 %) is ordinary fp32 cuBLAS (`ampere_sgemm_128x128_tn` 62.9 s).
4. **The workload is GPU-bound, not host-bound**: GPU is busy 99.8 % of `qvq_quantize.main`, kernel time is 91.3 % of
   wall, memcpy+memset total 21.65 s (1.1 %). `cuda_kern_exec_sum` shows an average **queue wait of 283 ms** before
   each `qvq_v2_segment_grid_kernel` executes (the host runs far ahead and blocks in `cudaLaunchKernel`, avg 108 µs
   over 9.19 M launches — that is back-pressure from a full launch queue, not launch overhead). There is no CPU
   Viterbi fallback: zero `qvq_cpu.*` ranges were recorded.
5. Variants that exist in `qvq_cuda.py:97-108` but are **never invoked by the real W2 v2b2 YAQA workload** (they are only
   exercised by `scripts/benchmark_qvq_viterbi.py` / unit tests): `viterbi_trusted`, `viterbi_v4`, `viterbi_banked`,
   `viterbi_v2_segment_banked`, `viterbi_v2_segment_g`, `viterbi_v2_segment_grid`, `viterbi_v2_segment_grid_trusted`,
   `viterbi_v2_segment_midpoint_trusted`, plus `gemv`, `gemv_v4`, `hadamard` (inference/codec paths, not touched by
   quantization; the RHT is done with torch ops inside `rht_*` telemetry phases). Optimising any of them would not move
   this workload.

Target for the next kernel task: the `<4,2,16,fused-boundary>` instantiation of `qvq_v2_segment_grid_kernel` in
`gptqmodel_ext/qvq/qvq_viterbi_cuda.cu:1301`, with `qvq_v2_segment_grid_finalize_kernel` (37.7 s, 2.1 %) as the
obvious fusion partner. Everything else is ≤ 18 % each.

## Workload

| item | value |
|---|---|
| model | `/monster/data/model/Llama-3.2-1B-Instruct` (16 decoder layers × 7 linears = 112 modules, all quantized) |
| config | `--format qvq_v2b2_p32 --bits 2 --rounding yaqa --bank-count 2` (same knobs as `configs/qwen3_8b_qvq_w2_acceptance.json`, but 128-row streams) |
| calibration | `/monster/data/model/dataset/nm-calibration/llm.parquet` rows [0,128) |
| YAQA Sketch-B | same parquet rows [512,640), `--yaqa-minimum-sequences 128` |
| device | NVIDIA PG506-230 (Hopper-class, 98 GB), CUDA 13.3 toolkit, torch 2.13.0+cu132, nsys 2026.4.1 |
| end-to-end wall (`qvq_quantize.main`, incl. model load, calibration tokenisation, prep, 16 layers, save) | **1938.9 s** |
| per decoder layer (`StageLayer ... wall_clock`) | 111.1–117.0 s, 16 layers ≈ 1792 s (92.4 %) |
| quantized checkpoint | `/root/qvq_prof/llama32_1b_full` (not committed) |

## How it was captured

```bash
scripts/setup_qvq_profile_env.sh                 # one-off env (see "Environment recipe")
scripts/profile_qvq_quantize_nsys.sh llama32_1b_full \
  --model /monster/data/model/Llama-3.2-1B-Instruct --output /root/qvq_prof/llama32_1b_full \
  --format qvq_v2b2_p32 --bits 2 --rounding yaqa --bank-count 2 \
  --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet --calibration-row-start 0 --calibration-rows 128 \
  --yaqa-dataset /monster/data/model/dataset/nm-calibration/llm.parquet --yaqa-row-start 512 --yaqa-rows 128 \
  --yaqa-minimum-sequences 128
# quick run: add --max-layers 2
```

The wrapper runs

```
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=false --sample=none --cpuctxsw=none \
  --output artifacts/nsys/reps/llama32_1b_full python scripts/profile_qvq_quantize_nsys.py --attribution-json ... <qvq_quantize.py args>
```

and then, for each report, `nsys stats --report <R> --format csv --force-export=true --output artifacts/nsys/llama32_1b_full artifacts/nsys/reps/llama32_1b_full.nsys-rep`
with `R ∈ {cuda_gpu_kern_sum, cuda_gpu_sum, cuda_api_sum, cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, nvtx_sum, nvtx_pushpop_sum, nvtx_gpu_proj_sum, nvtx_kern_sum, cuda_kern_exec_sum, osrt_sum}`.
The CSVs are committed under `artifacts/nsys/llama32_1b_full_*.csv` together with the wrapper's host-side
`llama32_1b_full_host_attribution.json`; the tables below are produced by
`python scripts/summarize_qvq_nsys_stats.py artifacts/nsys llama32_1b_full`.

The `.nsys-rep` is **906 MB** (+ 419 MB `.sqlite`) and is gitignored; it lives at
`/root/QvQ-wt/nsys-profile/artifacts/nsys/reps/llama32_1b_full.nsys-rep` on the profiling box.

NVTX ranges (all injected by monkeypatching module attributes, nothing in `gptqmodel/` changed):

* `qvq_cuda.<op>` — around the `torch.ops` callable returned by every `gptqmodel.utils.qvq_cuda._qvq_cuda_<op>_op()`
  resolver (the 17 `required_ops`), so each kernel-variant launch site is attributed individually.
* `qvq_api.<fn>` — around `qvq_cuda_viterbi`, `qvq_cuda_viterbi_banked`, `qvq_cuda_viterbi_v2_segment_banked`,
  `_qvq_cuda_viterbi_trusted`, `qvq_cuda_gemv`, `qvq_cuda_hadamard` (these nest the `qvq_cuda.*` range plus the Python
  validation/reductions around it; `qvq_api.qvq_cuda_viterbi` ⊃ `qvq_cuda.viterbi`, do not add them).
* `qvq_cpu.<op>` / `qvq_api.qvq_cpu_viterbi*` — CPU fallbacks (none fired).
* `stage.<prepare_yaqa|preprocess|process|submodule_finalize|finalize>:<module>` — `QVQProcessor` per-module stages;
  `qvq_quantize.main` — whole harness.

### Totals

| metric | value |
|---|---|
| `qvq_quantize.main` wall (host, incl. load/save) | 1938.9 s |
| total CUDA kernel time (`cuda_gpu_kern_sum`) | 1769.97 s |
| total CUDA GPU time incl. memops (`cuda_gpu_sum`) | 1791.63 s |
| total memcpy/memset time (`cuda_gpu_mem_time_sum`) | 21.65 s |
| kernel time / wall | 91.3 % |

### Viterbi / codec variants invoked by the real workload

Kernel time is the sum of CUDA kernels that executed inside the variant's NVTX range (`nvtx_kern_sum`); `% GPU` is relative to total kernel time. `host avg/max` are host-side durations of the op call (launch + any sync inside the op) from the Python wrapper.

| NVTX range (variant) | calls | kernel time (s) | % GPU kernel time | kernel avg / call (ms) | kernel launches | host avg (ms) | host max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qvq_cuda.viterbi_v2_segment_family_grid_trusted` | 102304 | 987.56 | 55.8 | 9.653 | 1023040 | 3.757 | 111.6 |
| `qvq_cuda.viterbi_tail_trusted` | 43920 | 176.30 | 10.0 | 4.014 | 263520 | 0.307 | 11.6 |
| `qvq_api.qvq_cuda_viterbi` | 40960 | 142.52 | 8.1 | 3.479 | 1474560 | 1.110 | 202.3 |
| `qvq_cuda.viterbi` | 40960 | 140.08 | 7.9 | 3.420 | 737280 | 0.520 | 60.8 |
| `qvq_cuda.viterbi_v2_segment_tail_trusted` | 39744 | 134.46 | 7.6 | 3.383 | 874368 | 2.140 | 32.8 |
| `qvq_cuda.yaqa_feedback` | 61344 | 101.19 | 5.7 | 1.650 | 184032 | 0.130 | 25.0 |
| `qvq_cuda.yaqa_feedback_update` | 61344 | 45.16 | 2.6 | 0.736 | 184032 | 0.602 | 29.9 |

Registered `qvq_cuda` ops never invoked by this workload: `viterbi_trusted`, `viterbi_v4`, `viterbi_banked`, `viterbi_v2_segment_banked`, `viterbi_v2_segment_g`, `viterbi_v2_segment_grid`, `viterbi_v2_segment_grid_trusted`, `viterbi_v2_segment_midpoint_trusted`, `gemv`, `gemv_v4`, `hadamard`

### Kernels inside each variant range

- `qvq_cuda.viterbi_v2_segment_family_grid_trusted`: `qvq_v2_segment_grid_kernel<(int)4, (int)2, (int)16, (bool)1, (bool)0, __half, unsigned char>` 956.35 s ×818432; `qvq_v2_segment_grid_finalize_kernel<(int)4, (int)2, (int)16, (bool)0, __half, unsigned char>` 30.76 s ×102304; `qvq_half2_codebook_norm_pack_kernel` 0.45 s ×102304
- `qvq_cuda.viterbi_tail_trusted`: `qvq_viterbi_kernel<(int)4, (int)2, __half, unsigned char>` 175.79 s ×87840; `qvq_half2_codebook_norm_pack_kernel` 0.25 s ×87840; `at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::AUnaryFun` 0.14 s ×43696; `at::native::roll_cuda_kernel<float>` 0.11 s ×43920
- `qvq_api.qvq_cuda_viterbi`: `qvq_viterbi_kernel<(int)4, (int)2, __half, unsigned char>` 137.71 s ×40960; `at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::na` 1.28 s ×204800; `at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::Half, at::native::func_wrapper_t<c10::Ha` 0.68 s ×81920; `at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::` 0.58 s ×81920
- `qvq_cuda.viterbi`: `qvq_viterbi_kernel<(int)4, (int)2, __half, unsigned char>` 137.71 s ×40960; `at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::na` 0.70 s ×122880; `at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::Half, at::native::func_wrapper_t<c10::Ha` 0.34 s ×40960; `at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::` 0.29 s ×40960
- `qvq_cuda.viterbi_v2_segment_tail_trusted`: `qvq_v2_segment_grid_kernel<(int)4, (int)2, (int)16, (bool)1, (bool)0, __half, unsigned char>` 127.07 s ×635904; `qvq_v2_segment_grid_finalize_kernel<(int)4, (int)2, (int)16, (bool)0, __half, unsigned char>` 6.98 s ×79488; `qvq_half2_codebook_norm_pack_kernel` 0.22 s ×79488; `at::native::roll_cuda_kernel<float>` 0.10 s ×39744
- `qvq_cuda.yaqa_feedback`: `cutlass::Kernel2<cutlass_80_simt_sgemm_grouped_128x128_8x4_align1>` 100.83 s ×61344; `qvq_yaqa_feedback_epilogue_kernel` 0.22 s ×61344; `qvq_yaqa_grouped_pointers_kernel` 0.14 s ×61344
- `qvq_cuda.yaqa_feedback_update`: `gemmSN_NN_kernel<float, (int)128, (int)2, (int)4, (int)8, (int)4, (int)4, (bool)0, cublasGemvTensorBatched<con` 24.66 s ×61216; `ampere_sgemm_128x128_nt` 20.38 s ×61248; `qvq_yaqa_update_grouped_pointers_kernel(float *, float *, const float *, const float *, const float *, float *` 0.12 s ×61344; `ampere_sgemm_32x128_nt` 0.00 s ×96

### Top-10 CUDA kernels overall (`cuda_gpu_kern_sum`)

| # | kernel | total (s) | % | instances | avg (ms) | max (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `qvq_v2_segment_grid_kernel<(int)4, (int)2, (int)16, (bool)1, (bool)0, __half, unsigned char>` | 1083.41 | 61.2 | 1454336 | 0.745 | 4.430 |
| 2 | `qvq_viterbi_kernel<(int)4, (int)2, __half, unsigned char>` | 313.50 | 17.7 | 128800 | 2.434 | 7.394 |
| 3 | `cutlass::Kernel2<cutlass_80_simt_sgemm_grouped_128x128_8x4_align1>` | 100.83 | 5.7 | 61344 | 1.644 | 6.140 |
| 4 | `ampere_sgemm_128x128_nt` | 68.28 | 3.9 | 64832 | 1.053 | 101.438 |
| 5 | `ampere_sgemm_128x128_tn` | 62.90 | 3.6 | 1792 | 35.102 | 103.866 |
| 6 | `qvq_v2_segment_grid_finalize_kernel<(int)4, (int)2, (int)16, (bool)0, __half, unsigned char>` | 37.74 | 2.1 | 181792 | 0.208 | 0.973 |
| 7 | `gemmSN_NN_kernel<float, (int)128, (int)2, (int)4, (int)8, (int)4, (int)4, (bool)0, cublasGemvTensorBatched<con` | 24.66 | 1.4 | 61216 | 0.403 | 1.024 |
| 8 | `ampere_sgemm_64x32_sliced1x4_nt` | 13.69 | 0.8 | 23360 | 0.586 | 0.907 |
| 9 | `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>` | 5.55 | 0.3 | 35184 | 0.158 | 13.699 |
| 10 | `ampere_sgemm_128x64_nt` | 5.24 | 0.3 | 2160 | 2.427 | 6.770 |

### Per-stage GPU projection (`nvtx_gpu_proj_sum`, union of GPU activity under the range)

| stage range | instances | projected GPU time (s) | range wall (s) | GPU busy % of range |
|---|---:|---:|---:|---:|
| `qvq_quantize.main` | 1 | 1934.95 | 1938.89 | 99.8 |
| `stage.process` | 112 | 1756.43 | 1760.78 | 99.8 |
| `stage.prepare_yaqa` | 1 | 135.43 | 135.64 | 99.8 |

### Memory transfers (`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`)

| operation | count | total time (s) | avg (us) | max (ms) | total size (MB) |
|---|---:|---:|---:|---:|---:|
| [CUDA memcpy Device-to-Host] | 581818 | 10.80 | 18.6 | 226.433 | 22504.924 |
| [CUDA memcpy Host-to-Device] | 66918 | 9.45 | 141.2 | 540.957 | 36323.559 |
| [CUDA memcpy Device-to-Device] | 384352 | 1.00 | 2.6 | 0.257 | 303642.412 |
| [CUDA memset] | 339720 | 0.41 | 1.2 | 0.077 | 29.718 |

### CUDA API (host side, `cuda_api_sum`)

| API | calls | total (s) | avg (us) | max (ms) |
|---|---:|---:|---:|---:|
| `cudaLaunchKernel` | 9193860 | 997.15 | 108.5 | 5994.766 |
| `cudaEventRecordWithFlags` | 919586 | 275.69 | 299.8 | 72.323 |
| `cudaStreamSynchronize` | 586832 | 262.79 | 447.8 | 1832.654 |
| `cudaMemcpyAsync` | 1033088 | 110.17 | 106.6 | 542.442 |
| `cudaDeviceSynchronize` | 14549 | 7.06 | 485.3 | 12.660 |
| `cudaMemsetAsync` | 339720 | 6.77 | 19.9 | 102.708 |
| `cudaEventCreateWithFlags` | 919028 | 4.32 | 4.7 | 45.764 |
| `cuLaunchKernel` | 65611 | 3.45 | 52.6 | 102.742 |
| `cuKernelGetName` | 9193860 | 3.01 | 0.3 | 6.015 |
| `cudaEventQuery` | 920126 | 2.80 | 3.0 | 7.882 |

### Host-side stage timings (profile wrapper, `*_host_attribution.json`)

| range | calls | host total (s) | host avg (ms) | host max (ms) |
|---|---:|---:|---:|---:|
| `stage.process` | 112 | 1760.78 | 15721.212 | 33498.6 |
| `qvq_cuda.viterbi_v2_segment_family_grid_trusted` | 102304 | 384.35 | 3.757 | 111.6 |
| `stage.prepare_yaqa` | 1 | 135.64 | 135638.434 | 135638.4 |
| `qvq_cuda.viterbi_v2_segment_tail_trusted` | 39744 | 85.04 | 2.140 | 32.8 |
| `qvq_api.qvq_cuda_viterbi` | 40960 | 45.46 | 1.110 | 202.3 |
| `qvq_cuda.yaqa_feedback_update` | 61344 | 36.94 | 0.602 | 29.9 |
| `qvq_cuda.viterbi` | 40960 | 21.30 | 0.520 | 60.8 |
| `qvq_cuda.viterbi_tail_trusted` | 43920 | 13.49 | 0.307 | 11.6 |
| `stage.submodule_finalize` | 112 | 8.35 | 74.554 | 145.4 |
| `qvq_cuda.yaqa_feedback` | 61344 | 7.97 | 0.130 | 25.0 |
| `stage.preprocess` | 112 | 0.18 | 1.596 | 6.0 |
| `stage.finalize` | 1 | 0.00 | 0.278 | 0.3 |
| `stage.prepare_module_granular_replay` | 1 | 0.00 | 0.010 | 0.0 |


### Reading the tables

* **Variant table.** "kernel time" is exact GPU kernel execution time attributed by `nvtx_kern_sum` to kernels launched
  inside the range; "calls" is the NVTX instance count (`nvtx_pushpop_sum`), which matches the wrapper's host-side call
  counters in `llama32_1b_full_host_attribution.json` exactly. `% of wall` = kernel time / 1938.9 s:
  `viterbi_v2_segment_family_grid_trusted` 50.9 %, `viterbi_tail_trusted` 9.1 %, `viterbi` 7.2 %,
  `viterbi_v2_segment_tail_trusted` 6.9 %, `yaqa_feedback` 5.2 %, `yaqa_feedback_update` 2.3 %. Sum of all
  `qvq_cuda.*` ranges = 1584.8 s = 89.5 % of kernel time; the remaining 115.8 s outside any `qvq_*` range is fp32 cuBLAS
  (`ampere_sgemm_128x128_tn` 62.9 s in `stage.prepare_yaqa`, `ampere_sgemm_128x128_nt` 47.9 s — Hessian / RHT /
  factorisation torch ops inside `stage.process`) plus ~5 s of small elementwise/reduce kernels.
* **Host gaps.** `stage.process` is 1760.8 s wall with 1756.4 s projected GPU activity → 4.4 s (0.25 %) of GPU-idle gaps
  inside module quantization across 112 modules. `cudaStreamSynchronize` is called 586,832× for 262.8 s of host time,
  but since the GPU stays busy those syncs are the host waiting on a deep queue rather than stalls that idle the GPU.
  Host-side `max` per op call (e.g. 202 ms for `qvq_api.qvq_cuda_viterbi`, 112 ms for the family-grid op) are first-call /
  queue-full outliers, not kernel time. The `osrt_sum` top entries are `sem_wait` / `sem_clockwait` / `poll` /
  `pthread_cond_wait` from the torch and nsys worker threads (tens of thousands of seconds summed across threads) and
  are not on the critical path.
* **Memory.** 581,818 D2H copies (22.5 GB, 10.8 s) and 66,918 H2D (36.3 GB, 9.45 s): 1.1 % of wall. The D2H count
  (~5.2k per module) comes from scalar/loss read-backs in the YAQA anti-diagonal loop; worth batching eventually but
  not the bottleneck. D2D 303 GB / 1.0 s is `.contiguous()` / `clone()` traffic on the weight tiles.
* **Launch rate.** 9.19 M `cudaLaunchKernel` in 1939 s (4.7k/s) with `qvq_v2_segment_grid_kernel` averaging only 0.745 ms:
  after the kernel gets 4–10× faster, launch throughput *will* become the next ceiling (≈0.1–0.2 ms per launch),
  so the follow-up kernel task should also cut launches per op call (8 grid launches + finalize per
  `viterbi_v2_segment_family_grid_trusted` call today).

## Environment recipe (negative results included)

`scripts/setup_qvq_profile_env.sh` reproduces the env; the non-obvious parts:

* System Python is 3.14 and there is no torch. `uv python install 3.12` + `uv venv --python 3.12 .venv-qvq-profile`.
* torch is installed from `https://download.pytorch.org/whl/cu130` (resolved to `torch 2.13.0+cu132`), then
  `requirements.txt` + `pytest`, then `uv pip install --no-build-isolation --no-deps -e .`.
* **Failure 1:** the box's `/usr/local/cuda` (13.3) has `nvcc` but none of the math-library headers, so the JIT build of
  `gptqmodel_ext/qvq/*.cu` dies with `fatal error: cusparse.h: No such file or directory` (torch's
  `ATen/cuda/CUDAContextLight.h` includes it).
* **Failure 2:** adding the whole `nvidia/cu13/include` directory from the pip wheels to `CPATH` fails with
  `#error "CUDA compiler and CUDA toolkit headers are incompatible"` (13.2 wheel headers vs. 13.3 nvcc CCCL).
* **Fix:** the script symlinks *only* the headers missing from `/usr/local/cuda/include` (cublas*, cusparse*, cusolver*,
  curand*, cufft*, cufile …, 40 files) into `.venv-qvq-profile/cuda-shim-include` and exports
  `CPATH=<shim> PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda`. JIT build of the QVQ ops then takes ~160 s.
* The harness computes a different extension build hash than a bare `prewarm_qvq_cuda()` call, so the *first*
  `qvq_quantize.py` run rebuilds once (visible as a 156 s first `qvq_cuda_viterbi` call in a smoke run); the profiled
  run used the warm cache (the `qvq_api.qvq_cuda_viterbi` host max of 202 ms confirms no rebuild).
* `nsys` OS-runtime tracing + CUDA tracing on 9.2 M launches made the post-processing (`nsys stats --force-export`) take
  ~25 min for the 906 MB report; the quantization itself ran at the same per-layer speed as un-profiled
  (111–117 s/layer vs 112 s in the un-profiled smoke run), so profiler overhead on the measured numbers is negligible.

## Qwen3-8B (first 4 layers) — comparison

Same config and datasets, `--max-layers 4` (`/monster/data/model/Qwen3-8B`, 4 × 7 = 28 modules; 36 decoder layers
in total). Captured with `scripts/profile_qvq_quantize_nsys.sh qwen3_8b_layers4 --max-layers 4 ...`; stats in
`artifacts/nsys/qwen3_8b_layers4_*.csv`, report at `artifacts/nsys/reps/qwen3_8b_layers4.nsys-rep` (359 MB, untracked).

| metric | Llama-3.2-1B (16/16 layers) | Qwen3-8B (4/36 layers) |
|---|---:|---:|
| `qvq_quantize.main` wall | 1938.9 s | 1519.8 s |
| `stage.prepare_yaqa` (Sketch-B, full model) | 135.6 s | 175.4 s |
| per decoder layer | 111–117 s | 325–328 s (→ ≈ 3.3 h for all 36) |
| GPU busy % of wall | 99.8 % | 99.4 % |
| `qvq_v2_segment_grid_kernel<4,2,16,fused>` | 1083.4 s, 61.2 % GPU, 55.9 % wall | 802.7 s, 58.0 % GPU, 52.8 % wall |
| `viterbi_v2_segment_family_grid_trusted` op | 102,304 calls, 987.6 s, 9.65 ms/call | 60,360 calls, 827.9 s, 13.7 ms/call |
| `qvq_viterbi_kernel<4,2>` | 313.5 s, 17.7 % | 198.1 s, 14.3 % |
| YAQA feedback GEMMs (`yaqa_feedback` + `_update`) | 146.3 s, 8.3 % | 169.1 s, 12.2 % |
| memcpy + memset | 21.65 s (1.1 %) | 21.64 s (1.4 %) |

| NVTX range (variant) | calls | kernel time (s) | % GPU kernel time | kernel avg / call (ms) | kernel launches | host avg (ms) | host max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qvq_cuda.viterbi_v2_segment_family_grid_trusted` | 60360 | 827.87 | 59.9 | 13.716 | 603600 | 5.684 | 754.6 |
| `qvq_cuda.yaqa_feedback` | 37832 | 108.17 | 7.8 | 2.859 | 113496 | 0.183 | 56.2 |
| `qvq_cuda.viterbi_tail_trusted` | 18916 | 103.59 | 7.5 | 5.476 | 113496 | 0.192 | 9.8 |
| `qvq_api.qvq_cuda_viterbi` | 22528 | 97.65 | 7.1 | 4.335 | 811008 | 1.152 | 209.1 |
| `qvq_cuda.viterbi` | 22528 | 96.19 | 7.0 | 4.270 | 405504 | 0.539 | 71.7 |
| `qvq_cuda.yaqa_feedback_update` | 37832 | 60.92 | 4.4 | 1.610 | 113496 | 0.858 | 16.0 |

| # | kernel | total (s) | % | instances | avg (ms) | max (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `qvq_v2_segment_grid_kernel<(int)4, (int)2, (int)16, (bool)1, (bool)0, __half, unsigned char>` | 802.71 | 58.0 | 482880 | 1.662 | 4.390 |
| 2 | `qvq_viterbi_kernel<(int)4, (int)2, __half, unsigned char>` | 198.12 | 14.3 | 60360 | 3.282 | 7.294 |
| 3 | `cutlass::Kernel2<cutlass_80_simt_sgemm_grouped_128x128_8x4_align1>` | 107.93 | 7.8 | 37832 | 2.853 | 15.504 |
| 4 | `ampere_sgemm_128x128_nt` | 90.17 | 6.5 | 38752 | 2.327 | 454.492 |
| 5 | `ampere_sgemm_128x128_tn` | 76.92 | 5.6 | 456 | 168.694 | 471.608 |
| 6 | `gemmSN_NN_kernel<float, (int)128, (int)2, (int)4, (int)8, (int)4, (int)4, (bool)0, cublasGemvTensorBatched<con` | 31.65 | 2.3 | 37688 | 0.840 | 3.078 |
| 7 | `qvq_v2_segment_grid_finalize_kernel<(int)4, (int)2, (int)16, (bool)0, __half, unsigned char>` | 24.89 | 1.8 | 60360 | 0.412 | 0.958 |
| 8 | `ampere_sgemm_128x128_nn` | 5.16 | 0.4 | 168 | 30.718 | 60.764 |
| 9 | `ampere_sgemm_128x64_nt` | 4.21 | 0.3 | 3560 | 1.182 | 11.166 |
| 10 | `ampere_sgemm_128x64_nn` | 2.58 | 0.2 | 232 | 11.115 | 21.286 |

Differences worth noting: on the wider Qwen3 modules (4096×12288 MLP, 4096×8192 QKV) each family-grid op call carries
more tiles (13.7 vs 9.65 ms/call) so the kernel is an even more dominant share of `stage.process`; the
`viterbi_v2_segment_tail_trusted` path was not taken at all for Qwen3's shapes; the plain fp32 cuBLAS share
(`ampere_sgemm_128x128_{nt,tn}`, 167 s = 12 %) grows with hidden size. The ranked conclusion is unchanged:
**optimise `qvq_v2_segment_grid_kernel<4,2,16,fused-boundary>` first (52.8–55.9 % of end-to-end wall on both models),
then `qvq_viterbi_kernel` (14–18 %).**
