# QVQ end-to-end nsys re-profile — post round-2 optimizations (PR #8 + PR #11)

Date: 2026-08-24. Branch `perf/qvq-nsys-reprofile`, base `origin/main @ fcd679e8` (includes the PR #7 NVTX
profiler harness, the PR #8 fused `qvq_fused_w2_family_grid_kernel`, the PR #11 round-2 squared-difference /
weight-fold optimizations, and the PR #12 test fixes). Baseline for all comparisons:
`docs/qvq_nsys_profile_llama32_1b.md` (PR #7, pre-optimization main @ `0888a695`). Same models, datasets,
config, device, and capture method as the baseline; the only intentional env addition is
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6` (see "Environment").

## TL;DR — ranked conclusion

**The optimization landed.** End-to-end `qvq_quantize` on the full Llama-3.2-1B is **1938.9 s → 1284.7 s
(1.51×)**; Qwen3-8B first-4-layers is **1519.8 s → 984.6 s (1.54×)**, per decoder layer 325–328 s → 194–200 s
(1.65×). The workload is still GPU-bound (GPU busy 99.7 % / 99.5 % of wall), so kernel time remains the only
lever. The family-grid range that was 61 % of GPU kernel time / 56 % of wall is now the fused kernel at
**36 % of GPU kernel time / 31–32 % of wall** on both models.

Ranked next targets (Amdahl ceilings on the Llama full-model wall of 1284.7 s; Qwen fractions in parentheses):

1. **`qvq_viterbi_kernel<4, 2, __half, uint8>` — the plain L16 recurrence — is now the top *actionable*
   target.** 312.4 s = 27.9 % of kernel time = **24.3 % of wall** (Qwen: 22.7 % / 20.2 %). It was untouched by
   rounds 1–2 (313.5 s baseline → 312.4 s now) and its call sites are unchanged: `viterbi_tail_trusted`
   (87,840 launches, 175.2 s) and `viterbi` via `qvq_cuda_viterbi()` (40,960 launches, 137.3 s). The same
   treatment that fused the family grid (persistent CTA, emission sharing between the two PGC16 banks,
   decision-equivalent arithmetic) has a realistic 2× here → **saves ~12 % of wall (→ ~1.14×)**; infinite
   ceiling 1.32×.
2. **YAQA feedback GEMMs → tensor cores.** `cutlass_80_simt_sgemm_grouped_128x128_align1` (97.6 s) +
   `gemmSN_NN`/`ampere_sgemm_128x128_nt` inside `yaqa_feedback_update` (43.7 s) = 141.4 s = 12.6 % of kernel
   time = **11.0 % of wall** — and a *larger* share on the bigger model (Qwen: 18.7 % of kernel time,
   16.5 % of wall, growing with hidden size). These are fp32 **SIMT** (non-tensor-core) kernels; moving the
   grouped feedback GEMM to TF32/tensor-core CUTLASS (~3× on these shapes) → **saves ~7 % of wall (→ ~1.08×)
   on Llama, ~1.12× on Qwen**. Lowest-risk item on this list: it is a kernel-selection/precision change, not
   a new algorithm.
3. **`qvq_fused_w2_family_grid_kernel` — still the single largest kernel, but plateaued.** 402.6 s = 36.0 %
   of kernel time = **31.3 % of wall** (Qwen: 36.2 % / 32.1 %). Rounds 1–2 already took it 9.65 → 3.74 ms per
   family-grid op call (2.58× cumulative, matching the predicted 2.59×); it is issue/math-pipe-bound at 50 %
   occupancy with the op order pinned by decision equivalence, and the recorded dead ends (fp64 min keys,
   ping-pong frontiers, packed IMNMX keys, smem emission tables, fminf+predicated select) mean the next step
   is **algorithmic or Hopper DPX**, not more micro-optimization. Infinite ceiling 1.46×; a realistic further
   1.3× saves ~7 % of wall (→ ~1.08×). High effort — rank it below 1–2 despite the bigger fraction.
4. **Sketch-B preparation GEMMs (`stage.prepare_yaqa`).** `ampere_sgemm_128x128_tn` 63.0 s = 4.9 % of wall
   (Qwen: 77.0 s, **7.8 %** — grows with model size; the stage runs over the *full* 36-layer model even in
   the 4-layer capture). Plain fp32 cuBLAS; TF32 → ~1.03–1.05× end-to-end.
5. **Residual un-fused `qvq_v2_segment_grid_kernel` under `viterbi_v2_segment_tail_trusted`.** 55.8 s = 4.3 %
   of wall on Llama (Qwen: 1.7 s, negligible — its shapes never take this path). 364,032 launches at avg
   0.145 ms: these are the batch < 40 calls the PR #8 gate routes to the reference path, plus small
   family-grid tiles. Lifting the gate (or batching small calls) is cheap to try but capped at ~1.03×.

Not targets: memcpy/memset (19.4 s, 1.5 % of wall), host gaps (GPU idle 0.3–0.5 % of `stage.process`), CPU
fallbacks (zero `qvq_cpu.*` ranges again). Launch pressure eased with the launch-count drop (9.19 M → 8.39 M
`cudaLaunchKernel`, avg 108 → 75 µs); at ~4.7 k launches/s it becomes a ceiling only after items 1–3 land.

**Top-3 ops by kernel time, and do the models agree?** Llama: `viterbi_v2_segment_family_grid_trusted`
(34.2 % of kernel time), `viterbi_tail_trusted` (15.7 %), `viterbi` (12.7 %). Qwen: the same #1
(37.2 %), then `viterbi_tail_trusted` and `yaqa_feedback` tied (11.9 % each) ahead of `viterbi` (11.2 %) —
i.e. **both models agree on the #1 op and on `qvq_viterbi_kernel` as the top actionable kernel; Qwen ranks
the YAQA GEMMs higher** (wider modules → bigger GEMM share), which is why target 2 outranks target 3 there.

## Workload

Identical knobs to the baseline captures (`--format qvq_v2b2_p32 --bits 2 --rounding yaqa --bank-count 2`,
calibration `llm.parquet` rows [0,128), YAQA Sketch-B rows [512,640), `--yaqa-minimum-sequences 128`),
device NVIDIA PG506-230 (98 GB), CUDA 13.3 toolkit, torch 2.13.0+cu132, nsys 2026.4.1.

```bash
scripts/setup_qvq_profile_env.sh   # one-off env; then an unprofiled --layers 1 warm run (see Environment)
scripts/profile_qvq_quantize_nsys.sh llama32_1b_full_post_r2 \
  --model /monster/data/model/Llama-3.2-1B-Instruct --output /root/qvq_prof/llama32_1b_full_post_r2 \
  --format qvq_v2b2_p32 --bits 2 --rounding yaqa --bank-count 2 \
  --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet --calibration-row-start 0 --calibration-rows 128 \
  --yaqa-dataset /monster/data/model/dataset/nm-calibration/llm.parquet --yaqa-row-start 512 --yaqa-rows 128 \
  --yaqa-minimum-sequences 128
scripts/profile_qvq_quantize_nsys.sh qwen3_8b_layers4_post_r2 --model /monster/data/model/Qwen3-8B \
  --output /root/qvq_prof/qwen3_8b_layers4_post_r2 --max-layers 4 <same remaining args>
```

CSV summary sets (same 11 `nsys stats` reports as the baseline) are committed as
`artifacts/nsys/llama32_1b_full_post_r2_*.csv` and `artifacts/nsys/qwen3_8b_layers4_post_r2_*.csv` with the
wrappers' `*_host_attribution.json`; tables below are `scripts/summarize_qvq_nsys_stats.py` output. The
`.nsys-rep` files are gitignored (`artifacts/nsys/reps/` on the profiling box).

## Llama-3.2-1B full model — post-r2 vs PR #7 baseline

### Totals

| metric | baseline (PR #7) | post-r2 | change |
|---|---:|---:|---:|
| `qvq_quantize.main` wall (incl. load, calibration, prep, 16 layers, save) | 1938.9 s | **1284.7 s** | **1.51×** |
| per decoder layer (`StageLayer` wall) | 111.1–117.0 s | 70.2–72.5 s | 1.58× |
| total CUDA kernel time | 1770.0 s | 1119.1 s | 1.58× |
| kernel time / wall | 91.3 % | 87.1 % | — |
| GPU busy % of `qvq_quantize.main` | 99.8 % | 99.7 % | GPU-bound both |
| memcpy + memset | 21.7 s (1.1 %) | 19.4 s (1.5 %) | unchanged |
| `cudaLaunchKernel` calls / avg | 9.19 M / 108 µs | 8.39 M / 75 µs | queue pressure eased |

The realized 1.51× matches the prediction from the PR #11 2-layer A/B (252.4 → 171.7 s = 1.47×). A second,
independent full-model run in this session (before the allocator env was added) measured 1299.7 s wall —
within 1.2 % of the committed capture, so the number is stable and the allocator setting is not a confound.

### Per-op attribution (NVTX `qvq_cuda.*` ranges)

`% kern` = share of total CUDA kernel time (1119.1 s); `% wall` = share of the 1284.7 s wall. Baseline
columns from the PR #7 full-model capture.

| op (NVTX range) | calls | kernel s (base → now) | % kern (base → now) | % wall (base → now) | ms/call (base → now) |
|---|---:|---:|---:|---:|---:|
| `viterbi_v2_segment_family_grid_trusted` | 102,304 | 987.6 → **382.3** | 55.8 → **34.2** | 50.9 → **29.8** | 9.65 → 3.74 (**2.58×**) |
| `viterbi_tail_trusted` | 43,920 | 176.3 → 175.7 | 10.0 → 15.7 | 9.1 → 13.7 | 4.01 → 4.00 (1.00×) |
| `viterbi` (via `qvq_cuda_viterbi`) | 40,960 | 140.1 → 139.6 | 7.9 → 12.5 | 7.2 → 10.9 | 3.42 → 3.41 (1.00×) |
| `viterbi_v2_segment_tail_trusted` | 39,744 | 134.5 → 95.0 | 7.6 → 8.5 | 6.9 → 7.4 | 3.38 → 2.39 (1.42×) |
| `yaqa_feedback` | 61,344 | 101.2 → 98.0 | 5.7 → 8.8 | 5.2 → 7.6 | 1.65 → 1.60 |
| `yaqa_feedback_update` | 61,344 | 45.2 → 43.8 | 2.6 → 3.9 | 2.3 → 3.4 | 0.74 → 0.71 |

Op coverage is unchanged from the baseline: all 17 `required_ops` carry an NVTX range, the same 6 are invoked
by this W2 v2b2 YAQA workload, and the same 11 (incl. `gemv`, `gemv_v4`, `hadamard`) record zero instances.

### Top CUDA kernels (`cuda_gpu_kern_sum`)

| # | kernel | total (s) | % kern | % wall | instances | avg (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `qvq_fused_w2_family_grid_kernel<(bool)0>` *(new, PR #8/#11)* | 402.64 | 36.0 | 31.3 | 133,792 | 3.009 |
| 2 | `qvq_viterbi_kernel<4, 2, __half, uint8>` | 312.44 | 27.9 | 24.3 | 128,800 | 2.426 |
| 3 | `cutlass_80_simt_sgemm_grouped_128x128_align1` (yaqa_feedback) | 97.60 | 8.7 | 7.6 | 61,344 | 1.591 |
| 4 | `ampere_sgemm_128x128_nt` | 67.67 | 6.0 | 5.3 | 64,832 | 1.044 |
| 5 | `ampere_sgemm_128x128_tn` (Sketch-B prep) | 62.99 | 5.6 | 4.9 | 1,792 | 35.152 |
| 6 | `qvq_v2_segment_grid_kernel<4,2,16,fused>` *(old kernel, small-batch path)* | 55.75 | 5.0 | 4.3 | 384,000 | 0.145 |
| 7 | `gemmSN_NN_kernel` (yaqa_feedback_update) | 23.98 | 2.1 | 1.9 | 61,216 | 0.392 |
| 8 | `ampere_sgemm_64x32_sliced1x4_nt` | 13.40 | 1.2 | 1.0 | 23,360 | 0.574 |
| 9 | `qvq_fused_w2_traceback_kernel` | 12.91 | 1.2 | 1.0 | 133,792 | 0.096 |

Baseline #1 (`qvq_v2_segment_grid_kernel<4,2,16,fused>`, 1083.4 s = 61.2 % kern / 55.9 % wall) is replaced by
the fused kernel + traceback + residual old-kernel launches totalling 471.3 s = 42.1 % kern / 36.7 % wall —
a 2.30× reduction of that family. Baseline #2 `qvq_viterbi_kernel` (313.5 s) is byte-for-byte the same work
(312.4 s) and rises from 17.7 % → 27.9 % of kernel time purely by denominator shrink. The old
`qvq_v2_segment_grid_finalize_kernel` (37.7 s baseline) survives only in the small-batch path (3.6 s).

Inside `viterbi_v2_segment_family_grid_trusted` the kernel mix is now: fused kernel 367.6 s ×99,808 +
traceback 10.1 s + old grid kernel 2.9 s ×19,968 (the batch < 40 gate) + family-mask detect 1.3 s.
`viterbi_v2_segment_tail_trusted` still runs mostly on the old grid kernel (52.8 s ×364,032, avg 0.145 ms)
with the fused kernel taking its larger calls (35.0 s ×33,984) — that split is ranked target 5.

## Qwen3-8B, first 4 of 36 layers — second data point

| metric | baseline (PR #7) | post-r2 | change |
|---|---:|---:|---:|
| `qvq_quantize.main` wall | 1519.8 s | **984.6 s** | **1.54×** |
| per decoder layer | 325–328 s | 194.4–199.8 s | 1.65× (→ ≈ 2.0 h for all 36 layers, was ≈ 3.3 h) |
| `stage.prepare_yaqa` (full-model Sketch-B) | 175.4 s | 175.7 s | unchanged (untouched by r1/r2) |
| GPU busy % of wall | 99.4 % | 99.5 % | GPU-bound |
| total CUDA kernel time | 1383.9 s | 874.4 s | 1.58× |
| family-grid op (`viterbi_v2_segment_family_grid_trusted`) | 827.9 s, 59.9 % kern, 13.72 ms/call | **325.0 s, 37.2 % kern, 5.39 ms/call** | 2.55× per call |
| `qvq_viterbi_kernel<4,2>` | 198.1 s, 14.3 % kern | 198.6 s, 22.7 % kern / 20.2 % wall | absolute unchanged |
| YAQA feedback GEMM kernels (`yaqa_feedback` + `_update` ranges) | 169.1 s, 12.2 % kern | 162.9 s, 18.7 % kern / 16.5 % wall | now #2 GPU consumer |
| Sketch-B `ampere_sgemm_128x128_tn` | 76.9 s | 77.0 s, 7.8 % wall | grows with model size |
| memcpy + memset | 21.6 s (1.4 %) | 20.5 s (2.1 %) | unchanged |

Top kernels: fused family grid 316.2 s (36.2 % kern / 32.1 % wall), `qvq_viterbi_kernel` 198.6 s (22.7 %),
grouped SIMT sgemm 103.6 s (11.9 %), `ampere_sgemm_128x128_nt` 89.3 s (10.2 %), `_tn` 77.0 s (8.8 %),
`gemmSN_NN` 30.8 s (3.5 %). As in the baseline, `viterbi_v2_segment_tail_trusted` is never invoked for
Qwen3's shapes, and the old grid kernel appears only marginally (1.7 s ×11,648 — the batch < 40 gate).
**The two models agree on the ranking**; Qwen weights targets 2 and 4 (the fp32 GEMM items) more heavily
because its 4096-wide modules make every GEMM larger while the Viterbi state space stays fixed.

## Environment

`scripts/setup_qvq_profile_env.sh` unchanged (uv Python 3.12 venv, torch 2.13.0+cu132 from the cu130 index,
editable install `--no-build-isolation --no-deps`, cuda-shim-include for the missing math-lib headers; JIT
build 155 s against this checkout). Notes specific to this session:

* The harness-hash JIT variant was warmed with an unprofiled `--layers 1` run at
  `GPTQMODEL_QVQ_NVCC_THREADS=8` (same value for the captures) so nvcc never runs under nsys.
* **Both captures set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`.**
  Without it, the Qwen3-8B full-model Sketch-B backward hit CUDA-allocator OOM-retry warnings
  (`memory allocation failed ... free: 740 MB`) whose flush-and-retry path can slow execution badly;
  with expandable segments the warnings disappear. Effect on the Llama numbers is ≤ 1.2 % (1299.7 s
  measured without it, 1284.7 s with).
* Capture-history caveat: the first full-model Llama capture completed (wall 1299.7 s, exit 0) but its
  626 MB `.qdstrm` was orphaned when the wrapper process tree was externally killed during nsys collection;
  `nsys import` refuses the truncated stream (`IncompleteFileException`), so the capture was rerun end to
  end. The committed CSVs come from the clean rerun (1284.7 s).

## Gates

* `pytest tests/test_qvq.py tests/test_qvq_cuda.py` at this tip, in the profiling venv:
  **1850 passed, 129 skipped, 0 failed** (110 s) — the PR #12 state plus its new regression test.
* No source changes outside `docs/` + `artifacts/` (profiling scripts untouched).
