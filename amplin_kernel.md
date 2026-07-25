
## 2026-07-25: Add M=1-16 N64 `splitk4` pipe2 interleaved kernel and refreshed routing table

Added `mma_lane_m16_n64_splitk4_pipe2_interleaved` (4 warps, 128 threads, 16 KiB shared partials) to bridge the gap between `splitk8` and `splitk24` on the dense projection shapes that still lose to raw Marlin. The kernel reuses the existing generic `amplin_mma_lane_mN_n64_splitkX_pipe2_interleaved_body` with `kMmaLaneSplitK4Warps`. It is gated to sm_80, `M <= 16`, and `N` divisible by 64.

Changes:
- `gptqmodel_ext/amplin/amplin_kernel.cu`: new kernel, shared-memory constant `kMmaLaneSplitKN64K4SharedBytes = 16 KiB`, `configure_` helper, and `torch::Tensor` wrapper.
- `gptqmodel_ext/amplin/amplin.cpp`: forward declaration, dispatch wrapper, `m.def`, and `m.impl`.
- `gptqmodel/utils/amplin.py`: public Python wrapper, `required_ops` entry, and `_DYNAMIC_CANDIDATES` spec with `k_multiple=512`.
- `gptqmodel/utils/amplin_dynamic_routing_table.json`: regenerated from dynamic micro-benchmarks on the target shapes.

Verification:
- `pytest -q tests/kernels/test_amplin.py`: 88 passed.
- `ruff check gptqmodel/utils/amplin.py`: passed.
- `git diff --check`: passed.

Small-batch (M=1,2,4,6,8,16) result over Laguna S 2.1, GLM 5.2, and Kimi K2.5 shapes, FP16+BF16, raw Marlin baseline, GPU 0 (A100 sm_80, 124 SMs):

| metric | value |
|---|---|
| legal Marlin shapes | 396 |
| Amplin wins | 380 |
| losses | 16 |
| min speedup | 0.874 |
| max speedup | 3.725 |
| mean speedup | 1.306 |
| geomean speedup | 1.265 |

Remaining losses are all within 13% of Marlin; the biggest remaining gap is `kimi-k2.5 dense-up` M=4-16 K=7168 N=18432 (0.91-0.97x) and a few GLM/Kimi BF16 projections. `splitk4` converts most of the earlier `kimi-k2.5 dense-up` M=1-16 losses from ~0.80x into near-ties or wins and raises the geomean from ~1.22 to 1.27.

## 2026-07-25: M=32 micro-tweaks (cp.async A pipeline and launch-bounds) do not close the gap

Re-baselined the three remaining M=32 losses on GPU 0 with the current `devin/1784898994-marlin-style-cpasync` tree.  For each shape the best existing Amplin kernel is shown next to Marlin:

| model | role | M | K | N | best amplin | amplin (us) | marlin (us) | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | tile2 | 535.4 | 445.2 | 1.20 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk16 | 141.4 | 120.1 | 1.18 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | tile2 | 692.4 | 551.1 | 1.26 |

Two quick targeted changes were attempted and then reverted:

1. **2-stage `cp.async` A-tile pipeline in `tiled_fullk`**.  The shared-A buffer was doubled and the next K-group's activation tile was loaded asynchronously while the current group was computed.  The idea was to overlap global A-load latency with MMA compute for the `lm-head` N-dominant shapes.  The change compiled and passed `pytest -q tests/kernels/test_amplin.py -k tile4`, but `tile2` was unchanged or slightly slower on the target shapes and the benchmark overhead was not worth keeping, so the edit was reverted.

2. **`__launch_bounds__(128, 4)` for `mma_lane_m32_n64_tile2_shared_a`**.  This asked the compiler to keep register usage low enough to allow four resident blocks per SM.  Instead, the kernel slowed across the board (glm lm-head 551 us, kimi lm-head 717 us, kimi dense-up 232 us), so it was reverted.

Conclusion: the remaining 18–26 % gaps are not reachable with small scheduling or async-load tweaks on the current N64 register-fragment kernels.  The next realistic routes are (a) a larger N-tile/8-warp `mma_lane` pipeline that still reuses the A tile, or (b) a Marlin-style 4-warp 4-stage `cp.async` micro-kernel with LOP3 register dequantization.

## 2026-07-25: M=32 failed experiments — weight preloading and N128 pipeline

Two more quick experiments on the remaining M=32 large-N losses did not pan out and were reverted.

1. Preload `uint4 packed_words[kSteps]` per K group in `splitkX_pipe2_interleaved_body` and in `tiled_fullk`.  The idea was to hoist global weight loads ahead of the `MmaInstruction` inner loop to hide load latency.  On `kimi-k2.5 dense-up` M=32 K=7168 N=18432 the change made `splitk16` 1.42x slower than Marlin (103.6 us vs Marlin 72.7 us) and regressed `glm-5.2 dense-up` similarly.  It was reverted because the extra register pressure offset any latency savings.

2. Extended the existing 4-warp `gemm_hmma_m32_n128_pipeline4` kernel to an 8-warp N128 block (`gemm_hmma_m32_n128_pipeline4_n128`) by templating `kBlockN` and `kWarps`.  The new kernel compiled and produced correct output for small N, but on the target shapes it was 3-4x slower than Marlin and had numerically wrong output for `N >= 163840` (max abs error ~0.2), so the prototype was reverted.  The existing `gemm_hmma` / `pipeline4` family remains 3-4x slower than the `mma_lane` paths on these shapes, so a Marlin-style `wmma` rewrite is not the immediate win.

Remaining M=32 losses after these attempts (FP16, GPU 0, `--skip-gpu-idle-preflight`):

| model | role | M | K | N | best amplin | amplin (us) | marlin (us) | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | tile2 | ~510 | ~471 | 1.08 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk16 | ~120 | ~117 | 1.03 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | tile2 | ~660 | ~555 | 1.19 |

The gaps are now small (3-19%).  `splitk16` on `dense-up` is memory/occupancy limited (low `Mem Pipes Busy`, 49% L1 hit) and `tile2` on `lm-head` is latency/occupancy limited (45% SM throughput).  Closing these likely requires a larger kernel rewrite (Marlin-style 8-warp N256 4-stage pipeline with the `mma_lane` instruction path, or SASS-level tuning) rather than another micro-tweak.

## 2026-07-25: M=32 shared-A K-split variants `tile2_splitk2`, `tile1_splitk4`, `tile1_splitk8`

Refactored the `tile2_splitk4` wrapper into a generic `amplin_mma_lane_m32_n64_tiled_splitk_cuda<NTiles, KSplit>` helper and added three new M=32 N64 kernels:

- `mma_lane_m32_n64_tile2_splitk2` (2 N64 tiles, K-split=2, 256 threads, 49 KiB shared)
- `mma_lane_m32_n64_tile1_splitk4` (1 N64 tile, K-split=4, 256 threads, 66 KiB shared)
- `mma_lane_m32_n64_tile1_splitk8` (1 N64 tile, K-split=8, 512 threads, 132 KiB shared)

Correctness passes (`pytest -q tests/kernels/test_amplin.py -k tile4`). The static routing table was regenerated for M=32 from a micro-benchmark on GPU 0. Compared with the previous `splitk16`/`tile2` table, the new variants win or are selected on several Laguna and GLM small/medium-N shapes, e.g.:

| model | role | M | K | N | selected | amplin (us) | marlin (us) | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| laguna-s-2.1 | q-proj-9216 | 32 | 3072 | 9216 | tile1_splitk4 | 71.2 | 117.2 | 0.61 |
| laguna-s-2.1 | dense-up | 32 | 3072 | 12288 | tile1_splitk4 | 73.1 | 82.7 | 0.88 |
| glm-5.2 | kv-b-proj | 32 | 512 | 28672 | tile2_splitk2 | 23.5 | 20.0 | 1.17 |
| glm-5.2 | q-b-proj | 32 | 2048 | 4096 | splitk16 | 18.3 | 20.8 | 0.88 |
| glm-5.2 | kv-a-proj | 32 | 6144 | 128 | splitk12x2_coop | 31.7 | 90.0 | 0.35 |

The hardest M=32 losses remain (FP16, GPU 0, `benchmark_amplin_m32_tile4.py`, `--skip-gpu-idle-preflight`):

| model | role | M | K | N | selected | amplin (us) | marlin (us) | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | tile2 | 480.2 | 391.8 | 1.23 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk16 | 100.3 | 69.3 | 1.45 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | tile2 | 627.8 | 478.0 | 1.31 |

NCU on `kimi-k2.5 dense-up M=32 splitk16` shows the kernel is memory-bound (DRAM throughput 31%, L1/TEX 78%, Mem Pipes Busy 10%) and the L1 hit rate is only 49%, suggesting the current direct-global weight loads are not staging/reusing enough. NCU on `kimi-k2.5 lm-head M=32 tile2` shows low SM (45%) and memory (38%) utilization, i.e. latency/occupancy limited. The next promising route is a `cp.async` 2-stage pipeline for weights in the shared-A N64 tile kernel (`tile2`) and/or a smaller-shared K-split variant for `dense-up` that can fit more blocks per SM.

## 2026-07-25: M=32 `splitk20` and `tile2_splitk2` experiments

Commit `4b1a5289` adds `mma_lane_m32_n64_splitk20_pipe2_interleaved` (20 warps, 80 KiB shared partials).  Correctness passes and it is the fastest Amplin path for `glm-5.2 indexer-wq-b` M=32 (61.5 us vs splitk16 62.2 us), but it does not beat `splitk16`/`tile2` on the remaining target losses.

I also prototyped `mma_lane_m32_n64_tile2_splitk2` from the existing `amplin_mma_lane_mN_n64_tiled_splitk_kernel` (2 N64 tiles + K-split=2, 256 threads, 48 KiB shared).  It compiled and passed correctness but was slower than the current best kernels on the target shapes:

- `kimi-k2.5 dense-up`  M=32 K=7168 N=18432: splitk16 139 us, tile2_splitk2 179 us, Marlin 118 us
- `kimi-k2.5 dense-down` M=32 K=18432 N=7168: splitk16 118 us, tile2_splitk2 282 us, Marlin 119 us
- `kimi-k2.5 lm-head`    M=32 K=7168 N=163840: tile2 663 us, tile2_splitk2 848 us, Marlin 524 us
- `glm-5.2 o-proj`       M=32 K=16384 N=6144: splitk16 108 us, tile2_splitk2 252 us, Marlin 105 us
- `glm-5.2 lm-head`      M=32 K=6144 N=154880: tile2 515 us, tile2_splitk2 649 us, Marlin 437 us

The extra K-split increased register pressure and did not close the gaps, so the prototype was reverted.  `splitk16` remains the best Amplin path for dense-up/down/o-proj and `tile2` remains best for lm-head.

## 2026-07-25: Child session M=1-16 wins and M=32 tile2 launch-bounds experiment

A child Devin session on `AMD-A100-2` added `mma_lane_m16_n16/n32/n64 split-k8/12/16 pipe2 interleaved` kernels and exposed them in `gptqmodel/utils/amplin.py` / `_DYNAMIC_CANDIDATES`.  Across the target model shapes, the dynamic router now wins on 395/396 M=1..16 cases vs raw Marlin (FP16/BF16).  The child session also added a generic `MinBlocks` template parameter to the `tiled_fullk` N64 shared-A kernel so occupancy could be tuned per specialization.

I tried raising `tile2` occupancy by setting `MinBlocks=5` for `mma_lane_m32_n64_tile2_shared_a` (128 threads).  NCU showed register count stayed at 86 and median latency on `kimi-k2.5 lm-head` regressed from ~686 us to ~729 us, so the wrapper was reverted to `MinBlocks=1` and only the generic template parameter is kept.  `MinBlocks=8` was also tested earlier and was substantially worse (~904 us), indicating the compiler cannot reduce register pressure on this kernel without spilling.

M=32 remaining losses after the revert (FP16, GPU 7, `--skip-gpu-idle-preflight`):

| model | role | M | K | N | selected | dynamic | marlin | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | tile2 | 509.4us | 470.6us | 1.08 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk16 | 119.3us | 115.6us | 1.03 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | tile2 | 666.1us | 557.3us | 1.20 |

The `tile2` kernel is latency/occupancy limited (45% SM throughput, 28% achieved occupancy, 86 registers/thread, block limit 5).  The `dense-up` shape is very close and may flip with a small K-split/shared-memory tweak.  The `lm-head` cases need a different N-tile/K-split schedule.

## 2026-07-25: Add M=32 N64 `splitk16` K-split kernel and refresh routing table

Added `mma_lane_m32_n64_splitk16_pipe2_interleaved` using the existing `amplin_mma_lane_mN_n64_splitkX_pipe2_interleaved_body` template with `KWarpsTotal = kMmaLaneSplitK16Warps` (16 warps, 512 threads, 64 KiB shared partials).  This sits between `splitk12` (12 warps / 48 KiB) and `splitk24` (24 warps / 96 KiB) and is the fastest Amplin M=32 path for most dense / projection shapes.

Changes:
- `gptqmodel_ext/amplin/amplin_kernel.cu`: new kernel, 64 KiB shared constant, `configure_` helper, and `torch::Tensor` wrapper.
- `gptqmodel_ext/amplin/amplin.cpp`: `TORCH_LIBRARY` `m.def`, dispatch wrapper, and `m.impl`.
- `gptqmodel/utils/amplin.py`: public Python function, `__all__` entry, and `_DYNAMIC_CANDIDATES` spec.
- `scripts/benchmark_amplin_m32_tile4.py`: `splitk16` legality and benchmark entry.
- `tests/kernels/test_amplin.py`: included `splitk16` in the M=32 N64 FP32-dequant reference loop.
- `gptqmodel/utils/amplin_dynamic_routing_table.json`: regenerated to select `splitk16` where it wins.

Verification:
- `pytest -q tests/kernels/test_amplin.py -k dynamic`: 6 passed.
- `pytest -q tests/kernels/test_amplin.py -k m32_n64_splitk24`: 2 passed.
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_m32_tile4.py tests/kernels/test_amplin.py`: passed.
- `git diff --check`: passed.

M=32 dynamic routing result (FP16, GPU 7, 10/30 warmup/iters, 3 rounds, `--skip-gpu-idle-preflight`):

| model | role | M | K | N | selected | dynamic (us) | marlin (us) | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| laguna-s-2.1 | q-proj-9216 | 32 | 3072 | 9216 | splitk16 | 65.8 | 84.1 | 0.78 |
| laguna-s-2.1 | router-gate | 32 | 3072 | 256 | splitk16 | 48.6 | 108.1 | 0.45 |
| glm-5.2 | q-a-proj | 32 | 6144 | 2048 | splitk16 | 57.2 | 83.8 | 0.68 |
| glm-5.2 | q-b-proj | 32 | 2048 | 4096 | splitk16 | 48.0 | 82.5 | 0.58 |
| glm-5.2 | q-b-proj-large | 32 | 2048 | 16384 | splitk16 | 64.8 | 83.8 | 0.77 |
| glm-5.2 | kv-a-proj | 32 | 6144 | 128 | splitk16 | 56.3 | 153.5 | 0.37 |
| glm-5.2 | o-proj | 32 | 16384 | 6144 | splitk16 | 92.9 | 105.2 | 0.88 |
| glm-5.2 | dense-up | 32 | 6144 | 12288 | splitk16 | 83.4 | 94.2 | 0.89 |
| glm-5.2 | dense-down | 32 | 12288 | 6144 | splitk16 | 83.2 | 94.8 | 0.88 |
| glm-5.2 | moe-up | 32 | 6144 | 2048 | splitk16 | 52.9 | 83.4 | 0.63 |
| glm-5.2 | indexer-wq-b | 32 | 2048 | 4096 | splitk16 | 48.2 | 82.6 | 0.58 |
| kimi-k2.5 | q-b-proj | 32 | 1536 | 12288 | splitk16 | 49.3 | 82.6 | 0.60 |
| kimi-k2.5 | o-proj | 32 | 8192 | 7168 | splitk16 | 66.1 | 87.2 | 0.76 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk16 | 119.8 | 116.5 | 1.03 |
| kimi-k2.5 | dense-down | 32 | 18432 | 7168 | splitk16 | 102.4 | 116.2 | 0.88 |

At M=32 Amplin now wins on 30/33 target shapes.  Remaining losses are:
- `glm-5.2 lm-head` M=32 K=6144 N=154880: tile2 503.8 us vs Marlin 471.0 us (1.07x)
- `kimi-k2.5 dense-up` M=32 K=7168 N=18432: splitk16 119.8 us vs Marlin 116.5 us (1.03x)
- `kimi-k2.5 lm-head` M=32 K=7168 N=163840: tile2 653.3 us vs Marlin 554.1 us (1.18x)

## 2026-07-25: Refresh M=32 static routing table with `splitk12` / `tile2` / `coop` wins

Regenerated `gptqmodel/utils/amplin_dynamic_routing_table.json` for the target model shapes at M=32 using the full M=32 candidate sweep (`scripts/benchmark_amplin_m32_tile4.py`).  The table now selects the fastest legal Amplin path per `(M,K,N,dtype)` among `splitk24`, `splitk12`, `coop`, `tile2/tile4/tile8`, `global_a`, and `n32_global_a`.

Result on GPU 7 (PG506-230, sm_80, 124 SMs, FP16, 10/30 warmup/iters, 3 rounds, `--skip-gpu-idle-preflight`):

| model | role | M | K | N | selected | dynamic (us) | marlin (us) | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| laguna-s-2.1 | expert-down | 32 | 1024 | 3072 | splitk12 | 51.5 | 86.4 | 0.60 |
| laguna-s-2.1 | kv/expert-up | 32 | 3072 | 1024 | splitk12 | 51.0 | 88.3 | 0.58 |
| laguna-s-2.1 | q-proj-6144 | 32 | 3072 | 6144 | splitk12 | 50.9 | 86.1 | 0.59 |
| laguna-s-2.1 | q-proj-9216 | 32 | 3072 | 9216 | splitk12 | 68.6 | 84.8 | 0.81 |
| laguna-s-2.1 | dense-up | 32 | 3072 | 12288 | splitk12 | 69.1 | 85.4 | 0.81 |
| laguna-s-2.1 | o-proj-6144 | 32 | 6144 | 3072 | coop | 55.1 | 85.2 | 0.65 |
| laguna-s-2.1 | o-proj-9216 | 32 | 9216 | 3072 | coop | 63.1 | 85.5 | 0.74 |
| laguna-s-2.1 | dense-down | 32 | 12288 | 3072 | coop | 71.2 | 88.8 | 0.80 |
| laguna-s-2.1 | router-gate | 32 | 3072 | 256 | splitk12 | 50.0 | 109.2 | 0.46 |
| glm-5.2 | q-a-proj | 32 | 6144 | 2048 | coop | 55.0 | 85.5 | 0.64 |
| glm-5.2 | q-b-proj | 32 | 2048 | 4096 | splitk12 | 49.2 | 84.5 | 0.58 |
| glm-5.2 | q-b-proj-large | 32 | 2048 | 16384 | splitk12 | 68.0 | 85.8 | 0.79 |
| glm-5.2 | kv-a-proj | 32 | 6144 | 128 | coop | 53.7 | 154.3 | 0.35 |
| glm-5.2 | kv-a-proj-mqa | 32 | 6144 | 576 | coop | 54.2 | 85.6 | 0.63 |
| glm-5.2 | kv-b-proj | 32 | 512 | 28672 | tile4 | 49.7 | 84.1 | 0.59 |
| glm-5.2 | o-proj | 32 | 16384 | 6144 | splitk12 | 116.4 | 108.6 | 1.07 |
| glm-5.2 | dense-up | 32 | 6144 | 12288 | splitk12 | 97.6 | 96.9 | 1.01 |
| glm-5.2 | dense-down | 32 | 12288 | 6144 | splitk24 | 96.4 | 98.3 | 0.98 |
| glm-5.2 | moe-up | 32 | 6144 | 2048 | coop | 54.2 | 85.0 | 0.64 |
| glm-5.2 | moe-down | 32 | 2048 | 6144 | splitk12 | 49.0 | 84.3 | 0.58 |
| glm-5.2 | indexer-wq-b | 32 | 2048 | 4096 | splitk12 | 49.4 | 84.4 | 0.59 |
| glm-5.2 | lm-head | 32 | 6144 | 154880 | tile2 | 513.3 | 472.5 | 1.09 |
| kimi-k2.5 | q-a-proj | 32 | 7168 | 1536 | coop | 59.5 | 85.0 | 0.70 |
| kimi-k2.5 | q-b-proj | 32 | 1536 | 12288 | splitk12 | 50.3 | 85.0 | 0.59 |
| kimi-k2.5 | kv-a-proj-mqa | 32 | 7168 | 576 | coop | 56.3 | 85.7 | 0.66 |
| kimi-k2.5 | kv-b-proj | 32 | 512 | 16384 | global_a | 49.7 | 84.5 | 0.59 |
| kimi-k2.5 | o-proj | 32 | 8192 | 7168 | splitk24 | 73.2 | 88.7 | 0.83 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk12 | 140.0 | 118.0 | 1.19 |
| kimi-k2.5 | dense-down | 32 | 18432 | 7168 | splitk24 | 120.3 | 116.9 | 1.03 |
| kimi-k2.5 | shared-up | 32 | 7168 | 2048 | coop | 57.1 | 85.3 | 0.67 |
| kimi-k2.5 | shared-down | 32 | 2048 | 7168 | splitk12 | 49.9 | 85.1 | 0.59 |
| kimi-k2.5 | router-gate | 32 | 7168 | 384 | coop | 55.8 | 109.8 | 0.51 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | tile2 | 655.0 | 550.5 | 1.19 |

At M=32 Amplin now wins 27/33 shapes, ties/close on `glm-5.2 dense-up` and `dense-down`, and still loses on the large-K large-N `dense-up`/`dense-down`/`lm-head` cases.  The main remaining gaps are:
- `glm-5.2 o-proj` M=32 K=16384 N=6144 (1.07x)
- `glm-5.2 dense-up` M=32 K=6144 N=12288 (1.01x)
- `glm-5.2 lm-head` M=32 K=6144 N=154880 (1.09x)
- `kimi-k2.5 dense-up` M=32 K=7168 N=18432 (1.19x)
- `kimi-k2.5 dense-down` M=32 K=18432 N=7168 (1.03x)
- `kimi-k2.5 lm-head` M=32 K=7168 N=163840 (1.19x)

Verification:
- `pytest -q tests/kernels/test_amplin.py -k dynamic`: 6 passed.
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_m32_tile4.py scripts/benchmark_amplin_dynamic.py tests/kernels/test_amplin.py`: passed.
- `git diff --check`: passed.

## 2026-07-25: PTX register-fragment `mma_lane_m32_n64_tile2_shared_a`

Implemented and registered `mma_lane_m32_n64_tile2_shared_a` by instantiating the existing `amplin_mma_lane_mN_n64_tiled_fullk_cuda_impl` PTX register-fragment kernel with `BlockM = 32` and `NTiles = 2`.  This reduces the per-block N tile width vs `tile4` (2 N64 tiles / 128 threads instead of 4 / 256 threads) while keeping the same shared-A-per-K-group + `ldmatrix` + `mma` + `MmaLaneDequant` design.  The smaller tile should reduce tail-wave waste on large-N shapes and raise block count per SM, at the cost of less per-block compute parallelism.

Changes:
- `gptqmodel_ext/amplin/amplin_kernel.cu`: added `amplin_mma_lane_m32_n64_tile2_shared_a_cuda` wrapper.
- `gptqmodel_ext/amplin/amplin.cpp`: forward declaration, dispatch wrapper, `m.def`, and `m.impl`.
- `gptqmodel/utils/amplin.py`: public Python wrapper + `__all__` export.
- `scripts/benchmark_amplin_m32_tile4.py`: `--m32` tile2 legality and benchmark entry.
- `tests/kernels/test_amplin.py`: added `tile2` to the `m32_n64` FP32-dequant reference test.

Verification:
- `pytest -q tests/kernels/test_amplin.py`: 82 passed.
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_m32_tile4.py tests/kernels/test_amplin.py`: passed.
- `git diff --check`: passed.
- Focused correctness: M=32 K=7168 N=163840 FP16/BF16, M=32 K=7168 N=18432, M=16 K=7168 N=2048; max FP16 error ~1.1e-3, BF16 ~9e-3.

Focused FP16 timing (GPU 0, random BF16-like weights, 20/50 warmup/iters, CUDA events, `--skip-gpu-idle-preflight`):

| shape | K | N | tile2_us | tile4_us | splitk24_us | marlin_us |
|---|---|---:|---:|---:|---:|---:|
| kimi-k2.5 lm-head | 7168 | 163840 | 802.2 | 812.1 | 903.2 | 553.7 |
| glm-5.2 lm-head | 6144 | 154880 | 606.4 | 691.7 | 732.0 | 460.0 |
| kimi-k2.5 dense-up | 7168 | 18432 | 212.9 | 229.9 | 156.8 | 117.4 |
| kimi-k2.5 dense-down | 18432 | 7168 | 408.7 | 505.1 | 133.1 | 116.0 |
| glm-5.2 dense-up | 6144 | 12288 | 165.2 | 198.8 | 108.2 | 93.9 |
| glm-5.2 dense-down | 12288 | 6144 | 285.0 | 350.3 | 105.1 | 94.8 |

`tile2` is the fastest Amplin variant for the `lm-head` shapes (large N / moderate K) and is also faster than `tile4` for most M=32 shapes, but it is still slower than Marlin on the large-N target losses.  For `dense-up`/`dense-down` shapes (large K / moderate N) the existing `splitk24` path remains best.

Nsight Compute on `tile2` for `kimi-k2.5 lm-head` (M=32 K=7168 N=163840, FP16):
- Duration: ~821 us
- Memory Throughput: 77.9%
- L1/TEX Cache Throughput: 86.9%
- Compute (SM) Throughput: 42.7%
- DRAM Throughput: 30.7%
- Registers / thread: 86
- Theoretical occupancy: 31.25% (limited by registers)
- Achieved occupancy: 28.5%
- Waves per SM: 2.06 with a partial tail of 40 blocks (up to 33% tail effect)

The kernel is memory-bound and the partial tail wave plus register-limited occupancy keep it behind Marlin, which achieves ~1 TB/s effective memory throughput on the same shape.  The `tile2` geometry only improved the tail-wave contribution vs `tile4`; it did not remove the fundamental memory/occupancy bottleneck.

Conclusion: `tile2` is a useful smaller-tile variant that becomes the best Amplin path for `lm-head` shapes, but it does not yet beat Marlin on M=32 large-N.  The next routes are:
1. Pipelined `cp.async` for B weights inside `tiled_fullk` to raise L1/TEX throughput and reduce stall cycles.
2. Further reduce register pressure (e.g. per-warp `uint4` B load + `ldmatrix` for A fragments, smaller accumulator tile) to increase occupancy.
3. Persist shared A across multiple `tile2` blocks (one A tile, many N tiles) to amortize the activation load bandwidth.

## 2026-07-25: fix shared-A load stride in `tiled_fullk` (duplicate A loads eliminated)

The `amplin_mma_lane_mN_n64_tiled_fullk_kernel` A-loading loop used `kThreads = NTiles * kMmaLanes`, but the block size for M=32 is `MWarpGroups * NTiles * kMmaLanes` (128 threads for `tile2`).  With the smaller stride, the upper half of the threads repeated the same `cp.async` indices as the lower half, so every activation group was loaded from global memory twice.  Because each block reloads its full MxK activation slice and total A traffic is comparable to B traffic for large-N shapes, this doubled the A-side memory pressure.

Change:
- `gptqmodel_ext/amplin/amplin_kernel.cu`: `kThreads = (BlockM / kMmaM) * NTiles * kMmaLanes` so all block threads load distinct A elements.

Verification:
- `pytest -q tests/kernels/test_amplin.py -k "mma_lane_m32_n64_tile4"`: 2 passed.
- `git diff --check`: passed.
- `ruff check`: passed.
- Correctness: M=32 K=7168 N=163840 FP16/BF16, max error unchanged (~1.1e-3 FP16).

Focused FP16 timing on `kimi-k2.5` M=32 shapes (GPU 0, 3/10 warmup/iters, 3 rounds, `--skip-gpu-idle-preflight`):

| shape | K | N | tile2 before (us) | tile2 after (us) | speedup |
|---|---|---:|---:|---:|---:|
| q-a-proj | 7168 | 1536 | 172.6 | 165.2 | 1.04x |
| q-b-proj | 1536 | 12288 | 75.8 | 74.2 | 1.02x |
| o-proj | 8192 | 7168 | 188.6 | 183.7 | 1.03x |
| dense-up | 7168 | 18432 | 215.4 | 206.8 | 1.04x |
| dense-down | 18432 | 7168 | 413.4 | 396.7 | 1.04x |
| shared-up | 7168 | 2048 | 155.4 | 146.9 | 1.06x |
| shared-down | 2048 | 7168 | 78.9 | 75.5 | 1.04x |
| router-gate | 7168 | 384 | 153.6 | 146.9 | 1.05x |
| lm-head | 7168 | 163840 | 795.8 | 736.8 | 1.08x |

`tile4` and `tile8` benefit similarly because they share the same kernel template.  The gap to Marlin is reduced for the large-N `lm-head` shape but Marlin still wins (`lm-head` 736.8 us vs Marlin 526.2 us).  The remaining bottleneck is memory throughput / occupancy; this change only removes redundant A traffic.

## 2026-07-25: dynamic M-K-N routing kernel (`amplin.dynamic`)

Implemented `amplin.dynamic()` in `gptqmodel/utils/amplin.py`.  Instead of hard-coding a single kernel for a shape, the function looks up `(M, K, N, dtype)` in a static routing table, then a dynamic routing table, and on miss runs a fast micro-benchmark across all legal candidates, picks the fastest, and caches the choice.  A bundled static table (`gptqmodel/utils/amplin_dynamic_routing_table.json`, 217 FP16 entries) is loaded at import so common shapes avoid the first-call search.

Candidates wired into the router (all group-128 symmetric W4A16):
- `gemv` for `M <= 16`
- `gemv_k12288_wide` for `M == 1, K == 12288`
- `gemv_multirow` for `M in {2,4,8,16}, K in {3072,4096,12288}`
- `gemm_hmma` / `gemm_hmma_v0` for `M % 16 == 0`
- `gemm_hmma_m32_n128_pipeline4` for `17 <= M <= 32`
- `mma_lane_m16_n16_*` (padded, splitk4/8/12/16) for `M <= 16`, `N % 16 == 0`
- `mma_lane_m16_n32_*` (splitk8/12/16, pipe2 variants and interleaved) for `M <= 16`, `N % 32 == 0`
- `mma_lane_m16_n64_*` (splitk24, splitk12x2 coop, shared-a, tile4, tile8) for `M <= 16`
- `mma_lane_m32_*` (`global_a`, `n32_global_a`, splitk24, splitk12x2 coop, shared-a, tile2/4/8) for `M <= 32`

Per-shape `dynamic()` vs Marlin (FP16, 5 warmup + 20 iters, 2 rounds, GPU 1 / PG506-232):

| metric | value |
|---|---|
| shapes timed | 231 `(model, role, M, K, N)` rows |
| unique `(M,K,N,dtype)` keys | 217 |
| `dynamic` wins | 14 |
| Marlin wins | 213 |
| ties | 4 |
| mean `dynamic`/Marlin | 0.77 |
| median `dynamic`/Marlin | 0.75 |

Selected `dynamic` wins were mostly large-N `lm-head` shapes (e.g. `kimi-k2.5 lm-head` M=1..6), where `mma_lane_m16_n32_splitk8_pipe2` beat Marlin by a few percent.  The router is now correctly exploiting the N32 split-k paths that the older fixed dispatch missed.

Verification:
- `pytest -q tests/kernels/test_amplin.py`: 88 passed (includes new `test_amplin_dynamic_routes_and_matches_fp32_reference`).
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_dynamic.py tests/kernels/test_amplin.py`: passed.
- `git diff --check`: passed.

### Packed-weight cache follow-up
- Added a per-packer weight cache in `_pack_with_cache` keyed by `(id(qweight), id(scales), packer, dtype)` so `dynamic()` does not re-pack qweight/scales on every call.
- Keeps the router fast enough to be used as the default entry point while the per-shape winner is cached.
- Still passes `pytest -q tests/kernels/test_amplin.py -k test_amplin_dynamic` (6 passed).

## 2026-07-25: cache torch.ops handles and remove `dynamic()` GPU sync

`amplin.dynamic()` was paying extension-loading and `torch.cuda.synchronize` overhead on every call: `_run_candidate` resolved the `torch.ops.gptqmodel_amplin.<kernel>` handle through the public wrapper each time, and `dynamic()` called `amplin_runtime_available()` (which also loads the extension) before dispatch plus a trailing `torch.cuda.synchronize(input.device)`.  This produced ~30–50 us of host-launch overhead per call, which was larger than the kernel time for many small-M shapes and hid the real performance of the selected kernel.

Changes:
- `gptqmodel/utils/amplin.py`:
  - Added `_CANDIDATE_OP_CACHE` and `_get_amplin_op()` so each kernel's `torch.ops` handle is resolved once and reused.
  - Updated `_run_candidate()` to use the cached op handle directly instead of the per-call public wrapper.
  - Added `_ensure_amplin_runtime_available()` so `dynamic()` checks extension availability once and reuses the result.
  - Removed the trailing `torch.cuda.synchronize(input.device)` from `dynamic()`; the op now returns asynchronously like other PyTorch ops.
- `scripts/benchmark_amplin_dynamic.py`: report the selected kernel from the merged static+dynamic routing table (`get_routing_table()`) instead of only the dynamic cache.
- `gptqmodel/utils/amplin_dynamic_routing_table.json`: replaced with a fresh 231-entry table built from a 10-warmup/50-iteration brute-force sweep across the target shape list.

Verification:
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_dynamic.py tests/kernels/test_amplin.py`: passed.
- `pytest -q tests/kernels/test_amplin.py`: 88 passed.
- `git diff --check`: passed.
- Full `scripts/benchmark_amplin_dynamic.py` sweep on GPU 1 (PG506-232, FP16, 20-warmup/30-iters/2-rounds):

| model | wins | ties | losses |
|---|---:|---:|---:|
| laguna-s-2.1 | 63 | 0 | 0 |
| glm-5.2 | 86 | 1 | 2 |
| kimi-k2.5 | 74 | 0 | 3 |
| **total** | **223** | **1** | **5** |

Remaining losses (FP16, median CUDA-event time):

| model | role | M | K | N | dynamic (us) | marlin (us) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | 723.0 | 542.2 | 1.334 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | 143.4 | 117.7 | 1.219 |
| glm-5.2 | lm-head | 32 | 6144 | 154880 | 530.8 | 461.1 | 1.151 |
| glm-5.2 | o-proj | 32 | 16384 | 6144 | 110.4 | 104.5 | 1.056 |
| kimi-k2.5 | dense-down | 32 | 18432 | 7168 | 119.5 | 117.6 | 1.017 |

Conclusion: removing the per-call host overhead turns the existing kernels + static routing table into a net win for 223 of 229 measured shape/batch cells; the only remaining losses are M=32 large-N or large-K shapes.  The next optimization target is a faster M=32 kernel (e.g. a pipelined `cp.async` N64/N128 mega-kernel) to close those last 5 cases, then refresh the table.

## 2026-07-25: M=32 mega-kernel attempts (streaming loads, `__ldcs`, shared-memory profiling)

Tried two quick micro-optimizations on the M=32 large-N/large-K losses and then profiled the worst case with Nsight Compute to find the real bottleneck.

### Streaming (`ld.global.cs`) packed-weight loads
Added a `load_packed_words` helper that used `__ldcs` (cache-streaming / L1-bypass) for every `uint4` packed-weight read in `amplin_mma_lane_mN_n64_*` kernels.  The intent was to reduce the L1/TEX traffic that NCU had flagged on the `tile2_shared_a` path.

Result: no net win and a small regression on `lm-head` shapes.  A full `scripts/benchmark_amplin_dynamic.py --m-values 32 --model all --skip-gpu-idle-preflight` sweep (FP16, 50 warmup / 200 iters, GPU 1) showed the same 5-6 M=32 losses as the baseline and slightly worse `lm-head` numbers:

| model | role | M | K | N | baseline dynamic (us) | `__ldcs` dynamic (us) | marlin (us) | baseline ratio | `__ldcs` ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | 568.7 | 570.5 | 480.2 | 1.184 | 1.188 |
| glm-5.2 | o-proj | 32 | 16384 | 6144 | 122.0 | 115.4 | 107.4 | 1.038 | 1.075 |
| glm-5.2 | dense-up | 32 | 6144 | 12288 | 104.2 | 94.8 | 94.4 | 0.980 | 1.004 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | 159.8 | 142.7 | 116.7 | 1.204 | 1.223 |
| kimi-k2.5 | dense-down | 32 | 18432 | 7168 | 133.8 | 119.3 | 115.7 | 1.023 | 1.031 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | 806.3 | 732.3 | 548.7 | 1.371 | 1.335 |

(The `glm-5.2 dense-up` result moved from a tiny win to a tiny tie/loss, so `__ldcs` is not a reliable improvement.)  The change was reverted.

### Nsight Compute on `kimi-k2.5 lm-head` M=32 with `tile2_shared_a`
Profiled `amplin_mma_lane_mN_n64_tiled_fullk_kernel<__half, 32, 2>` (the `tile2_shared_a` path) on `M=32 K=7168 N=163840`:

- Duration: ~821 us
- Memory Throughput: 78%
- L1/TEX Throughput: 87%
- SM Throughput: 43%
- Key Nsight Compute warning: **uncoalesced shared-memory accesses** with an estimated speedup of **72.68%** (82% of wavefronts excessive).

The root cause is the A-tile shared-memory layout used by `load_mma_fragment_a` + `ldmatrix.sync.aligned.m8n8.x4`.  Consecutive 16-byte `ldmatrix` rows are spaced `kHmmaBlockK` halfs apart (128 halfs = 256 bytes = 64 banks), so many threads in a warp map to the same shared-memory banks and serialize.  The L1/TEX pressure is a symptom, not the primary bottleneck.

### Next step
A swizzled / padded shared-A layout for the `tiled_fullk` and `splitk24` register-fragment kernels is now the most promising mega-kernel route.  Nsight Compute estimates that removing the shared-memory bank conflicts alone could give a ~70% speedup on `lm-head`, which would put `tile2_shared_a` ahead of Marlin on the worst M=32 large-N shape.  After that, a pipelined `cp.async` weight path can be layered on top.

## 2026-07-25: pad shared-A row stride in `tiled_fullk` to reduce `ldmatrix` bank conflicts

Changed `amplin_mma_lane_mN_n64_tiled_fullk_kernel` (the kernel behind `mma_lane_m32_n64_tile2/4/8_shared_a`) to allocate shared A with a padded row stride.  The original stride is `kHmmaBlockK = 128` halfs = 256 bytes, which is an exact multiple of 32 shared-memory banks, so every row of the 16-row `ldmatrix` tile maps to the same bank set and creates a 16-way bank conflict.  The new stride is `kSharedAK = kHmmaBlockK + 8` halfs = 272 bytes, i.e. a row starts 4 banks later.  Rows `r` and `r+8` still collide, so the worst-case conflict drops from 16-way to 2-way while keeping `cp.async` destinations 16-byte aligned.

Changes:
- `gptqmodel_ext/amplin/amplin_kernel.cu`: added `constexpr int kSharedAK = kHmmaBlockK + 8` inside `tiled_fullk`, updated `shared_a` allocation and all A indexing/addressing to use `kSharedAK`.
- Left the unused `tiled_splitk` template unchanged.

Verification:
- `pytest -q tests/kernels/test_amplin.py`: 88 passed.
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_m32_tile4.py tests/kernels/test_amplin.py`: passed.
- `git diff --check`: passed.

Focused FP16 timing on M=32 target shapes (GPU 1 / PG506-232, 20 warmup / 200 iters / 5 rounds, `--skip-gpu-idle-preflight`, `scripts/benchmark_amplin_m32_tile4.py`):

| model | role | M | K | N | tile2 (us) | marlin (us) | tile2/marlin |
|---|---|---:|---:|---:|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | 558.3 | 486.9 | 1.147 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | 724.2 | 581.7 | 1.245 |
| glm-5.2 | dense-up | 32 | 6144 | 12288 | 166.5 | 99.8 | 1.668 |
| glm-5.2 | dense-down | 32 | 12288 | 6144 | 295.1 | 101.0 | 2.922 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | 216.3 | 124.5 | 1.737 |
| kimi-k2.5 | dense-down | 32 | 18432 | 7168 | 417.9 | 124.1 | 3.368 |
| glm-5.2 | o-proj | 32 | 16384 | 6144 | 374.3 | 111.4 | 3.360 |
| kimi-k2.5 | o-proj | 32 | 8192 | 7168 | 175.7 | 92.8 | 1.893 |

`tile2` remains the best Amplin path for the large-N `lm-head` shapes and is now selected by the dynamic router.  For `kimi-k2.5 lm-head`, `tile2` improved from the previous 736.8 us (after the duplicate-A-load fix) to 724.2 us (~1.7% faster), and the `tile2`/Marlin ratio on that shape moved from ~1.33 to ~1.25.  The gap is still large; the kernel remains memory/L1-TEX bound, so the shared-memory bank-conflict reduction alone is not enough.

For the large-K `dense-up`/`dense-down`/`o-proj` shapes, the dynamic router still selects `mma_lane_m32_n64_splitk24_pipe2_interleaved`, which is also slower than Marlin.  Padding `tiled_fullk` does not affect that kernel.

Conclusion: the shared-A padding is a safe micro-optimization that reduces `ldmatrix` bank conflicts and gives a small `lm-head` speedup, but it does not close the M=32 losses.  The next step is a `cp.async`-pipelined weight path and/or a Marlin-style larger-N-tile / 8-warp register-dequant kernel for the large-K losses.

## 2026-07-25: M=32 split-k12 variant to raise occupancy (not a net win)

The `mma_lane_m32_n64_splitk24_pipe2_interleaved` kernel is limited to one block per SM because its 24-warp partials buffer needs 96 KiB of dynamic shared memory.  To test whether lower shared-memory usage increases occupancy, I generalized the `mN` N64 split-K body to accept `KWarpsTotal` as a template parameter and added a 12-warp variant (`mma_lane_m32_n64_splitk12_pipe2_interleaved`).  This halves the partials buffer to 48 KiB and keeps the same `ldmatrix`/`mma`/dequant pipeline.

Changes:
- `gptqmodel_ext/amplin/amplin_kernel.cu`:
  - Templated `amplin_mma_lane_mN_n64_splitkX_pipe2_interleaved_body` on `KWarpsTotal`.
  - Replaced the fixed one-output-per-warp reduction with a loop over `m_warp_groups * kMmaLaneSplitKN64Fragments` outputs, stepped by `kWarpsTotal`, so it works correctly for any warp count.
  - Added `amplin_mma_lane_m32_n64_splitk12_pipe2_interleaved_kernel` and its CUDA wrapper.
- `gptqmodel_ext/amplin/amplin.cpp`: forward declaration, dispatch, `m.def`, `m.impl`.
- `gptqmodel/utils/amplin.py`: public wrapper, op name registration, candidate spec.
- `scripts/benchmark_amplin_m32_tile4.py`: added `splitk12` to the comparison.

Verification:
- `pytest -q tests/kernels/test_amplin.py`: 88 passed.
- `ruff check gptqmodel/utils/amplin.py scripts/benchmark_amplin_m32_tile4.py tests/kernels/test_amplin.py`: passed.
- `git diff --check`: passed.
- Correctness: M=32 K=7168 N=18432 FP16, max error ~1.3e-3.

Nsight Compute on `kimi-k2.5 dense-up` M=32 (K=7168 N=18432, `splitk24`):
- Duration: ~135 us
- Memory Throughput: 50.8%
- Compute (SM) Throughput: 37.0%
- Block Limit Registers: 1
- Block Limit Shared Mem: 1
- Theoretical Occupancy: 37.5%

The occupancy is limited equally by registers and shared memory, so halving shared memory to 48 KiB does not raise the block count if the register footprint stays the same.  Focused timing on M=32 target shapes (GPU 7 / PG506-230, FP16, `scripts/benchmark_amplin_m32_tile4.py` 50 iters / 3 rounds, `--skip-gpu-idle-preflight`):

| model | role | K | N | splitk24 (us) | splitk12 (us) | marlin (us) |
|---|---|---:|---:|---:|---:|---:|
| glm-5.2 | o-proj | 16384 | 6144 | 137.4 | 136.9 | 111.9 |
| glm-5.2 | dense-up | 6144 | 12288 | 118.9 | 117.6 | 100.0 |
| glm-5.2 | dense-down | 12288 | 6144 | 115.7 | 117.0 | 101.3 |
| glm-5.2 | lm-head | 6144 | 154880 | 783.2 | 758.9 | 487.2 |
| kimi-k2.5 | dense-up | 7168 | 18432 | 173.6 | 172.0 | 124.0 |
| kimi-k2.5 | dense-down | 18432 | 7168 | 146.3 | 148.2 | 124.0 |
| kimi-k2.5 | lm-head | 7168 | 163840 | 959.8 | 942.9 | 580.4 |

`splitk12` is within ~1-2% of `splitk24` but still loses to Marlin on the large-N/large-K M=32 cases.  The dynamic router did not select `splitk12` in a full `benchmark_amplin_dynamic.py` M=32 sweep.  Occupancy on `splitk24` is blocked by register usage as much as by shared memory, so a meaningful next step must reduce per-thread register pressure (fewer accumulators / smaller A tile) or increase N-tile width, not just lower shared usage.


## Benchmark: Amplin dynamic vs raw Marlin after M16 N64 splitk8/12/16 kernels

Device: NVIDIA PG506-230
Date: 2026-07-25 05:59:54 UTC

| model | role | M | K | N | dtype | amplin_us | marlin_us | speedup | max_abs_error | selected_kernel |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| laguna-s-2.1 | expert-down | 1 | 1024 | 3072 | fp16 | 48.19 | 56.53 | 1.173 | 2.2995e-04 | mma_lane_m16_n16_splitk4 |
| laguna-s-2.1 | expert-down | 2 | 1024 | 3072 | fp16 | 48.62 | 55.95 | 1.151 | 2.9492e-04 | mma_lane_m16_n16_splitk4 |
| laguna-s-2.1 | expert-down | 4 | 1024 | 3072 | fp16 | 44.85 | 56.03 | 1.249 | 2.4176e-04 | gemv |
| laguna-s-2.1 | expert-down | 6 | 1024 | 3072 | fp16 | 48.91 | 56.4 | 1.153 | 3.0422e-04 | mma_lane_m32_n64_tile2_shared_a |
| laguna-s-2.1 | expert-down | 8 | 1024 | 3072 | fp16 | 50.27 | 56.43 | 1.123 | 3.0422e-04 | mma_lane_m16_n64_tile4_shared_a |
| laguna-s-2.1 | expert-down | 16 | 1024 | 3072 | fp16 | 48.93 | 56.85 | 1.162 | 3.1799e-04 | mma_lane_m16_n64_shared_a |
| laguna-s-2.1 | expert-down | 1 | 1024 | 3072 | bf16 | 49.14 | 57.15 | 1.163 | 1.9872e-03 | mma_lane_m32_n64_tile2_shared_a |
| laguna-s-2.1 | expert-down | 2 | 1024 | 3072 | bf16 | 48.75 | 56.54 | 1.16 | 2.0651e-03 | mma_lane_m16_n64_tile4_shared_a |
| laguna-s-2.1 | expert-down | 4 | 1024 | 3072 | bf16 | 49.7 | 56.61 | 1.139 | 2.2660e-03 | mma_lane_m16_n64_tile4_shared_a |
| laguna-s-2.1 | expert-down | 6 | 1024 | 3072 | bf16 | 49.2 | 56.75 | 1.153 | 2.2660e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | expert-down | 8 | 1024 | 3072 | bf16 | 45.23 | 56.32 | 1.245 | 1.9500e-03 | gemv |
| laguna-s-2.1 | expert-down | 16 | 1024 | 3072 | bf16 | 48.0 | 57.1 | 1.19 | 2.4532e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | kv/expert-up | 1 | 3072 | 1024 | fp16 | 49.78 | 57.86 | 1.162 | 4.7910e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | kv/expert-up | 2 | 3072 | 1024 | fp16 | 45.02 | 57.14 | 1.269 | 4.7910e-04 | gemv_multirow |
| laguna-s-2.1 | kv/expert-up | 4 | 3072 | 1024 | fp16 | 44.61 | 57.78 | 1.295 | 4.7910e-04 | gemv_multirow |
| laguna-s-2.1 | kv/expert-up | 6 | 3072 | 1024 | fp16 | 49.25 | 57.92 | 1.176 | 4.7910e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | kv/expert-up | 8 | 3072 | 1024 | fp16 | 47.98 | 57.22 | 1.192 | 4.7910e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | kv/expert-up | 16 | 3072 | 1024 | fp16 | 47.68 | 56.46 | 1.184 | 4.7910e-04 | mma_lane_m16_n16_splitk4 |
| laguna-s-2.1 | kv/expert-up | 1 | 3072 | 1024 | bf16 | 48.16 | 57.07 | 1.185 | 2.7943e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | kv/expert-up | 2 | 3072 | 1024 | bf16 | 43.71 | 57.76 | 1.321 | 2.8882e-03 | gemv_multirow |
| laguna-s-2.1 | kv/expert-up | 4 | 3072 | 1024 | bf16 | 44.8 | 57.54 | 1.284 | 2.8882e-03 | gemv_multirow |
| laguna-s-2.1 | kv/expert-up | 6 | 3072 | 1024 | bf16 | 48.78 | 58.59 | 1.201 | 3.0891e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | kv/expert-up | 8 | 3072 | 1024 | bf16 | 48.54 | 57.6 | 1.187 | 3.0891e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | kv/expert-up | 16 | 3072 | 1024 | bf16 | 47.84 | 58.1 | 1.214 | 4.0250e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | q-proj-6144 | 1 | 3072 | 6144 | fp16 | 45.87 | 56.82 | 1.239 | 3.7909e-04 | gemv |
| laguna-s-2.1 | q-proj-6144 | 2 | 3072 | 6144 | fp16 | 49.07 | 56.9 | 1.159 | 5.3287e-04 | mma_lane_m16_n16_splitk8 |
| laguna-s-2.1 | q-proj-6144 | 4 | 3072 | 6144 | fp16 | 48.56 | 55.66 | 1.146 | 5.3287e-04 | mma_lane_m16_n16_splitk12 |
| laguna-s-2.1 | q-proj-6144 | 6 | 3072 | 6144 | fp16 | 49.34 | 57.31 | 1.161 | 5.3287e-04 | mma_lane_m16_n16_splitk8 |
| laguna-s-2.1 | q-proj-6144 | 8 | 3072 | 6144 | fp16 | 49.6 | 56.32 | 1.135 | 5.3287e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | q-proj-6144 | 16 | 3072 | 6144 | fp16 | 49.41 | 56.11 | 1.136 | 5.4336e-04 | mma_lane_m16_n32_splitk8 |
| laguna-s-2.1 | q-proj-6144 | 1 | 3072 | 6144 | bf16 | 48.14 | 56.21 | 1.167 | 3.9501e-03 | mma_lane_m16_n16_splitk12 |
| laguna-s-2.1 | q-proj-6144 | 2 | 3072 | 6144 | bf16 | 47.7 | 56.05 | 1.175 | 3.9502e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | q-proj-6144 | 4 | 3072 | 6144 | bf16 | 47.57 | 56.45 | 1.187 | 4.1999e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | q-proj-6144 | 6 | 3072 | 6144 | bf16 | 47.6 | 55.71 | 1.17 | 4.1999e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| laguna-s-2.1 | q-proj-6144 | 8 | 3072 | 6144 | bf16 | 47.28 | 54.99 | 1.163 | 4.1999e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| laguna-s-2.1 | q-proj-6144 | 16 | 3072 | 6144 | bf16 | 47.63 | 56.64 | 1.189 | 4.3247e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| laguna-s-2.1 | q-proj-9216 | 1 | 3072 | 9216 | fp16 | 46.53 | 56.93 | 1.224 | 4.5109e-04 | gemv |
| laguna-s-2.1 | q-proj-9216 | 2 | 3072 | 9216 | fp16 | 49.31 | 56.77 | 1.151 | 5.1141e-04 | mma_lane_m16_n16_splitk12 |
| laguna-s-2.1 | q-proj-9216 | 4 | 3072 | 9216 | fp16 | 49.54 | 56.5 | 1.141 | 5.5718e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | q-proj-9216 | 6 | 3072 | 9216 | fp16 | 49.92 | 57.65 | 1.155 | 5.5718e-04 | mma_lane_m16_n16_splitk8 |
| laguna-s-2.1 | q-proj-9216 | 8 | 3072 | 9216 | fp16 | 50.1 | 57.02 | 1.138 | 5.7030e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | q-proj-9216 | 16 | 3072 | 9216 | fp16 | 51.26 | 56.88 | 1.11 | 6.4111e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | q-proj-9216 | 1 | 3072 | 9216 | bf16 | 48.56 | 57.18 | 1.178 | 3.9542e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | q-proj-9216 | 2 | 3072 | 9216 | bf16 | 49.25 | 56.64 | 1.15 | 4.2946e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| laguna-s-2.1 | q-proj-9216 | 4 | 3072 | 9216 | bf16 | 48.02 | 56.53 | 1.177 | 4.2946e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | q-proj-9216 | 6 | 3072 | 9216 | bf16 | 48.77 | 56.75 | 1.164 | 4.2946e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | q-proj-9216 | 8 | 3072 | 9216 | bf16 | 49.42 | 56.3 | 1.139 | 4.5420e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | q-proj-9216 | 16 | 3072 | 9216 | bf16 | 48.86 | 56.38 | 1.154 | 4.5421e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| laguna-s-2.1 | dense-up | 1 | 3072 | 12288 | fp16 | 48.51 | 56.1 | 1.156 | 5.0628e-04 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| laguna-s-2.1 | dense-up | 2 | 3072 | 12288 | fp16 | 48.61 | 56.66 | 1.166 | 5.0628e-04 | mma_lane_m16_n16_splitk4 |
| laguna-s-2.1 | dense-up | 4 | 3072 | 12288 | fp16 | 49.86 | 56.99 | 1.143 | 5.6887e-04 | mma_lane_m16_n32_splitk8 |
| laguna-s-2.1 | dense-up | 6 | 3072 | 12288 | fp16 | 50.53 | 57.12 | 1.13 | 6.2418e-04 | mma_lane_m16_n32_splitk8 |
| laguna-s-2.1 | dense-up | 8 | 3072 | 12288 | fp16 | 50.05 | 57.39 | 1.147 | 6.2418e-04 | mma_lane_m16_n32_splitk8 |
| laguna-s-2.1 | dense-up | 16 | 3072 | 12288 | fp16 | 52.56 | 57.54 | 1.095 | 6.2418e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| laguna-s-2.1 | dense-up | 1 | 3072 | 12288 | bf16 | 48.94 | 56.86 | 1.162 | 5.2066e-03 | mma_lane_m16_n16_splitk12 |
| laguna-s-2.1 | dense-up | 2 | 3072 | 12288 | bf16 | 47.98 | 56.67 | 1.181 | 5.2066e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| laguna-s-2.1 | dense-up | 4 | 3072 | 12288 | bf16 | 48.14 | 56.14 | 1.166 | 5.2066e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| laguna-s-2.1 | dense-up | 6 | 3072 | 12288 | bf16 | 49.9 | 56.83 | 1.139 | 5.2066e-03 | mma_lane_m16_n64_splitk12_pipe2_interleaved |
| laguna-s-2.1 | dense-up | 8 | 3072 | 12288 | bf16 | 49.33 | 56.67 | 1.149 | 5.2066e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | dense-up | 16 | 3072 | 12288 | bf16 | 49.3 | 58.7 | 1.191 | 5.2066e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| laguna-s-2.1 | o-proj-6144 | 1 | 6144 | 3072 | fp16 | 49.57 | 57.02 | 1.15 | 6.3157e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | o-proj-6144 | 2 | 6144 | 3072 | fp16 | 49.22 | 57.04 | 1.159 | 6.3145e-04 | mma_lane_m16_n32_splitk16_pipe2 |
| laguna-s-2.1 | o-proj-6144 | 4 | 6144 | 3072 | fp16 | 48.69 | 56.32 | 1.157 | 6.3145e-04 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | o-proj-6144 | 6 | 6144 | 3072 | fp16 | 49.34 | 57.2 | 1.159 | 6.3264e-04 | mma_lane_m16_n16_splitk8 |
| laguna-s-2.1 | o-proj-6144 | 8 | 6144 | 3072 | fp16 | 48.74 | 56.51 | 1.16 | 6.5374e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| laguna-s-2.1 | o-proj-6144 | 16 | 6144 | 3072 | fp16 | 48.5 | 56.48 | 1.165 | 7.0822e-04 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | o-proj-6144 | 1 | 6144 | 3072 | bf16 | 49.57 | 56.77 | 1.145 | 4.6527e-03 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | o-proj-6144 | 2 | 6144 | 3072 | bf16 | 48.48 | 57.14 | 1.179 | 4.8348e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | o-proj-6144 | 4 | 6144 | 3072 | bf16 | 48.74 | 56.51 | 1.16 | 4.8348e-03 | mma_lane_m16_n32_splitk12 |
| laguna-s-2.1 | o-proj-6144 | 6 | 6144 | 3072 | bf16 | 48.83 | 56.53 | 1.158 | 5.1608e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | o-proj-6144 | 8 | 6144 | 3072 | bf16 | 48.3 | 55.81 | 1.155 | 5.1608e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | o-proj-6144 | 16 | 6144 | 3072 | bf16 | 48.0 | 56.99 | 1.187 | 5.2752e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | o-proj-9216 | 1 | 9216 | 3072 | fp16 | 48.5 | 57.14 | 1.178 | 5.7602e-04 | gemv |
| laguna-s-2.1 | o-proj-9216 | 2 | 9216 | 3072 | fp16 | 49.5 | 57.58 | 1.163 | 8.7905e-04 | mma_lane_m16_n16_splitk12 |
| laguna-s-2.1 | o-proj-9216 | 4 | 9216 | 3072 | fp16 | 48.69 | 57.55 | 1.182 | 8.7905e-04 | mma_lane_m16_n16_splitk8 |
| laguna-s-2.1 | o-proj-9216 | 6 | 9216 | 3072 | fp16 | 50.48 | 57.5 | 1.139 | 1.1306e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| laguna-s-2.1 | o-proj-9216 | 8 | 9216 | 3072 | fp16 | 49.06 | 56.46 | 1.151 | 1.1306e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | o-proj-9216 | 16 | 9216 | 3072 | fp16 | 50.45 | 57.39 | 1.138 | 1.1306e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| laguna-s-2.1 | o-proj-9216 | 1 | 9216 | 3072 | bf16 | 48.9 | 58.1 | 1.188 | 6.4998e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | o-proj-9216 | 2 | 9216 | 3072 | bf16 | 49.33 | 57.62 | 1.168 | 6.5000e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| laguna-s-2.1 | o-proj-9216 | 4 | 9216 | 3072 | bf16 | 48.19 | 57.66 | 1.197 | 6.5000e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | o-proj-9216 | 6 | 9216 | 3072 | bf16 | 48.91 | 58.26 | 1.191 | 6.5000e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | o-proj-9216 | 8 | 9216 | 3072 | bf16 | 48.43 | 56.38 | 1.164 | 6.5000e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| laguna-s-2.1 | o-proj-9216 | 16 | 9216 | 3072 | bf16 | 50.03 | 57.39 | 1.147 | 7.7891e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| laguna-s-2.1 | dense-down | 1 | 12288 | 3072 | fp16 | 49.31 | 57.57 | 1.167 | 7.1633e-04 | mma_lane_m16_n16_splitk8 |
| laguna-s-2.1 | dense-down | 2 | 12288 | 3072 | fp16 | 49.1 | 58.05 | 1.182 | 1.1849e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | dense-down | 4 | 12288 | 3072 | fp16 | 49.73 | 57.94 | 1.165 | 1.1849e-03 | mma_lane_m16_n16_splitk16 |
| laguna-s-2.1 | dense-down | 6 | 12288 | 3072 | fp16 | 49.86 | 57.76 | 1.159 | 1.1849e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| laguna-s-2.1 | dense-down | 8 | 12288 | 3072 | fp16 | 50.05 | 56.93 | 1.137 | 1.1849e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| laguna-s-2.1 | dense-down | 16 | 12288 | 3072 | fp16 | 53.87 | 56.7 | 1.053 | 1.1849e-03 | mma_lane_m16_n64_splitk12x2_coop_interleaved |
| laguna-s-2.1 | dense-down | 1 | 12288 | 3072 | bf16 | 49.01 | 58.35 | 1.191 | 7.9482e-03 | mma_lane_m16_n16_splitk16 |
| laguna-s-2.1 | dense-down | 2 | 12288 | 3072 | bf16 | 48.99 | 58.05 | 1.185 | 7.9484e-03 | mma_lane_m16_n16_splitk16 |
| laguna-s-2.1 | dense-down | 4 | 12288 | 3072 | bf16 | 49.07 | 57.57 | 1.173 | 7.9484e-03 | mma_lane_m16_n16_splitk16 |
| laguna-s-2.1 | dense-down | 6 | 12288 | 3072 | bf16 | 50.06 | 57.7 | 1.152 | 7.9484e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | dense-down | 8 | 12288 | 3072 | bf16 | 48.59 | 57.07 | 1.175 | 7.9484e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | dense-down | 16 | 12288 | 3072 | bf16 | 53.44 | 57.78 | 1.081 | 1.0014e-02 | mma_lane_m16_n64_splitk12x2_coop_interleaved |
| laguna-s-2.1 | router-gate | 1 | 3072 | 256 | fp16 | 48.37 | 62.45 | 1.291 | 2.9224e-04 | mma_lane_m16_n16_splitk12 |
| laguna-s-2.1 | router-gate | 2 | 3072 | 256 | fp16 | 44.64 | 68.78 | 1.541 | 2.3156e-04 | gemv |
| laguna-s-2.1 | router-gate | 4 | 3072 | 256 | fp16 | 44.27 | 69.54 | 1.571 | 2.4140e-04 | gemv_multirow |
| laguna-s-2.1 | router-gate | 6 | 3072 | 256 | fp16 | 45.1 | 71.3 | 1.581 | 2.4140e-04 | gemv |
| laguna-s-2.1 | router-gate | 8 | 3072 | 256 | fp16 | 44.67 | 63.47 | 1.421 | 4.0185e-04 | gemv_multirow |
| laguna-s-2.1 | router-gate | 16 | 3072 | 256 | fp16 | 44.77 | 67.41 | 1.506 | 4.0185e-04 | gemv_multirow |
| laguna-s-2.1 | router-gate | 1 | 3072 | 256 | bf16 | 47.34 | 62.02 | 1.31 | 3.6875e-03 | mma_lane_m16_n64_splitk12_pipe2_interleaved |
| laguna-s-2.1 | router-gate | 2 | 3072 | 256 | bf16 | 43.28 | 68.7 | 1.587 | 1.9456e-03 | gemv_multirow |
| laguna-s-2.1 | router-gate | 4 | 3072 | 256 | bf16 | 44.05 | 69.7 | 1.582 | 1.9456e-03 | gemv_multirow |
| laguna-s-2.1 | router-gate | 6 | 3072 | 256 | bf16 | 48.96 | 70.74 | 1.445 | 3.6874e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| laguna-s-2.1 | router-gate | 8 | 3072 | 256 | bf16 | 45.02 | 63.81 | 1.417 | 3.2429e-03 | gemv_multirow |
| laguna-s-2.1 | router-gate | 16 | 3072 | 256 | bf16 | 43.92 | 67.62 | 1.54 | 3.2429e-03 | gemv_multirow |
| glm-5.2 | q-a-proj | 1 | 6144 | 2048 | fp16 | 44.86 | 56.9 | 1.268 | 4.6635e-04 | gemv |
| glm-5.2 | q-a-proj | 2 | 6144 | 2048 | fp16 | 46.16 | 56.9 | 1.233 | 4.8113e-04 | gemv |
| glm-5.2 | q-a-proj | 4 | 6144 | 2048 | fp16 | 49.36 | 56.78 | 1.15 | 5.7971e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-a-proj | 6 | 6144 | 2048 | fp16 | 47.81 | 56.75 | 1.187 | 5.9330e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-a-proj | 8 | 6144 | 2048 | fp16 | 48.93 | 57.23 | 1.17 | 5.9330e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-a-proj | 16 | 6144 | 2048 | fp16 | 49.15 | 56.56 | 1.151 | 6.4921e-04 | mma_lane_m16_n32_splitk12_pipe2 |
| glm-5.2 | q-a-proj | 1 | 6144 | 2048 | bf16 | 48.29 | 56.96 | 1.18 | 4.0877e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | q-a-proj | 2 | 6144 | 2048 | bf16 | 48.06 | 56.77 | 1.181 | 4.0876e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-a-proj | 4 | 6144 | 2048 | bf16 | 49.02 | 57.34 | 1.17 | 4.9487e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | q-a-proj | 6 | 6144 | 2048 | bf16 | 48.27 | 56.91 | 1.179 | 4.9888e-03 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | q-a-proj | 8 | 6144 | 2048 | bf16 | 49.22 | 56.58 | 1.15 | 5.3594e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | q-a-proj | 16 | 6144 | 2048 | bf16 | 49.15 | 57.52 | 1.17 | 5.3594e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| glm-5.2 | q-b-proj | 1 | 2048 | 4096 | fp16 | 44.98 | 57.26 | 1.273 | 2.5725e-04 | gemv |
| glm-5.2 | q-b-proj | 2 | 2048 | 4096 | fp16 | 48.96 | 57.28 | 1.17 | 4.0048e-04 | mma_lane_m16_n16_padded |
| glm-5.2 | q-b-proj | 4 | 2048 | 4096 | fp16 | 45.55 | 56.62 | 1.243 | 4.8423e-04 | gemv |
| glm-5.2 | q-b-proj | 6 | 2048 | 4096 | fp16 | 48.67 | 56.05 | 1.152 | 4.9233e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-b-proj | 8 | 2048 | 4096 | fp16 | 48.91 | 56.18 | 1.149 | 4.9233e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | q-b-proj | 16 | 2048 | 4096 | fp16 | 48.05 | 55.84 | 1.162 | 5.0068e-04 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | q-b-proj | 1 | 2048 | 4096 | bf16 | 47.97 | 56.8 | 1.184 | 2.8319e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | q-b-proj | 2 | 2048 | 4096 | bf16 | 47.62 | 56.74 | 1.192 | 3.5501e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-b-proj | 4 | 2048 | 4096 | bf16 | 49.02 | 56.85 | 1.16 | 3.5501e-03 | mma_lane_m16_n64_shared_a |
| glm-5.2 | q-b-proj | 6 | 2048 | 4096 | bf16 | 48.26 | 56.13 | 1.163 | 3.5501e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| glm-5.2 | q-b-proj | 8 | 2048 | 4096 | bf16 | 48.53 | 57.12 | 1.177 | 3.5501e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | q-b-proj | 16 | 2048 | 4096 | bf16 | 48.4 | 56.67 | 1.171 | 3.5499e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| glm-5.2 | q-b-proj-large | 1 | 2048 | 16384 | fp16 | 48.69 | 57.33 | 1.177 | 3.7110e-04 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | q-b-proj-large | 2 | 2048 | 16384 | fp16 | 49.3 | 56.59 | 1.148 | 3.7110e-04 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | q-b-proj-large | 4 | 2048 | 16384 | fp16 | 48.48 | 56.72 | 1.17 | 4.9531e-04 | mma_lane_m16_n16_splitk8 |
| glm-5.2 | q-b-proj-large | 6 | 2048 | 16384 | fp16 | 48.98 | 57.38 | 1.172 | 5.2440e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | q-b-proj-large | 8 | 2048 | 16384 | fp16 | 48.91 | 57.15 | 1.168 | 5.2440e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | q-b-proj-large | 16 | 2048 | 16384 | fp16 | 50.43 | 56.86 | 1.128 | 5.6064e-04 | mma_lane_m16_n32_splitk8 |
| glm-5.2 | q-b-proj-large | 1 | 2048 | 16384 | bf16 | 48.91 | 56.82 | 1.162 | 3.1320e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | q-b-proj-large | 2 | 2048 | 16384 | bf16 | 48.94 | 57.54 | 1.176 | 3.1320e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | q-b-proj-large | 4 | 2048 | 16384 | bf16 | 49.36 | 56.82 | 1.151 | 3.2436e-03 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | q-b-proj-large | 6 | 2048 | 16384 | bf16 | 48.93 | 56.5 | 1.155 | 3.6452e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| glm-5.2 | q-b-proj-large | 8 | 2048 | 16384 | bf16 | 47.82 | 56.66 | 1.185 | 3.6452e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| glm-5.2 | q-b-proj-large | 16 | 2048 | 16384 | bf16 | 48.29 | 56.78 | 1.176 | 3.9300e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| glm-5.2 | kv-a-proj | 1 | 6144 | 128 | fp16 | 44.59 | 87.42 | 1.961 | 4.8089e-04 | gemv |
| glm-5.2 | kv-a-proj | 2 | 6144 | 128 | fp16 | 49.12 | 101.42 | 2.065 | 4.8065e-04 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | kv-a-proj | 4 | 6144 | 128 | fp16 | 50.02 | 102.54 | 2.05 | 5.4908e-04 | mma_lane_m16_n32_splitk12 |
| glm-5.2 | kv-a-proj | 6 | 6144 | 128 | fp16 | 45.63 | 104.08 | 2.281 | 4.8089e-04 | gemv |
| glm-5.2 | kv-a-proj | 8 | 6144 | 128 | fp16 | 45.42 | 90.88 | 2.001 | 4.8089e-04 | gemv |
| glm-5.2 | kv-a-proj | 16 | 6144 | 128 | fp16 | 44.66 | 98.13 | 2.197 | 4.8089e-04 | gemv |
| glm-5.2 | kv-a-proj | 1 | 6144 | 128 | bf16 | 48.54 | 87.14 | 1.795 | 2.9850e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| glm-5.2 | kv-a-proj | 2 | 6144 | 128 | bf16 | 47.86 | 101.54 | 2.122 | 3.5533e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| glm-5.2 | kv-a-proj | 4 | 6144 | 128 | bf16 | 48.75 | 102.13 | 2.095 | 4.7338e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | kv-a-proj | 6 | 6144 | 128 | bf16 | 49.31 | 103.94 | 2.108 | 4.7338e-03 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | kv-a-proj | 8 | 6144 | 128 | bf16 | 48.56 | 90.7 | 1.868 | 4.7338e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | kv-a-proj | 16 | 6144 | 128 | bf16 | 44.43 | 98.11 | 2.208 | 3.5664e-03 | gemv |
| glm-5.2 | kv-a-proj-mqa | 1 | 6144 | 576 | fp16 | 45.46 | 56.94 | 1.253 | 4.2975e-04 | gemv |
| glm-5.2 | kv-a-proj-mqa | 2 | 6144 | 576 | fp16 | 45.12 | 58.29 | 1.292 | 4.4298e-04 | gemv |
| glm-5.2 | kv-a-proj-mqa | 4 | 6144 | 576 | fp16 | 45.73 | 58.13 | 1.271 | 4.7219e-04 | gemv |
| glm-5.2 | kv-a-proj-mqa | 6 | 6144 | 576 | fp16 | 49.18 | 58.08 | 1.181 | 5.6720e-04 | mma_lane_m16_n16_splitk4 |
| glm-5.2 | kv-a-proj-mqa | 8 | 6144 | 576 | fp16 | 49.17 | 57.04 | 1.16 | 5.7065e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | kv-a-proj-mqa | 16 | 6144 | 576 | fp16 | 48.69 | 57.92 | 1.19 | 5.7375e-04 | mma_lane_m16_n16_splitk4 |
| glm-5.2 | kv-a-proj-mqa | 1 | 6144 | 576 | bf16 | 48.74 | 57.02 | 1.17 | 3.6967e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| glm-5.2 | kv-a-proj-mqa | 2 | 6144 | 576 | bf16 | 47.94 | 58.46 | 1.22 | 3.6967e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | kv-a-proj-mqa | 4 | 6144 | 576 | bf16 | 49.06 | 58.13 | 1.185 | 4.5220e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | kv-a-proj-mqa | 6 | 6144 | 576 | bf16 | 48.38 | 58.43 | 1.208 | 4.5434e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| glm-5.2 | kv-a-proj-mqa | 8 | 6144 | 576 | bf16 | 48.88 | 57.12 | 1.169 | 4.8360e-03 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | kv-a-proj-mqa | 16 | 6144 | 576 | bf16 | 48.22 | 58.24 | 1.208 | 5.0366e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | kv-b-proj | 1 | 512 | 28672 | fp16 | 45.25 | 56.86 | 1.257 | 2.4211e-04 | gemv |
| glm-5.2 | kv-b-proj | 2 | 512 | 28672 | fp16 | 46.16 | 57.2 | 1.239 | 2.4199e-04 | gemv |
| glm-5.2 | kv-b-proj | 4 | 512 | 28672 | fp16 | 48.91 | 57.01 | 1.166 | 2.7835e-04 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | kv-b-proj | 6 | 512 | 28672 | fp16 | 49.68 | 56.5 | 1.137 | 3.0118e-04 | mma_lane_m16_n16_splitk4 |
| glm-5.2 | kv-b-proj | 8 | 512 | 28672 | fp16 | 48.88 | 57.26 | 1.172 | 3.0184e-04 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | kv-b-proj | 16 | 512 | 28672 | fp16 | 49.42 | 56.93 | 1.152 | 3.7503e-04 | mma_lane_m16_n16_splitk4 |
| glm-5.2 | kv-b-proj | 1 | 512 | 28672 | bf16 | 49.07 | 57.26 | 1.167 | 1.9576e-03 | mma_lane_m32_n64_tile4_shared_a |
| glm-5.2 | kv-b-proj | 2 | 512 | 28672 | bf16 | 48.75 | 58.16 | 1.193 | 2.0018e-03 | mma_lane_m16_n64_tile8_shared_a |
| glm-5.2 | kv-b-proj | 4 | 512 | 28672 | bf16 | 48.82 | 57.44 | 1.177 | 2.2390e-03 | mma_lane_m32_n64_tile4_shared_a |
| glm-5.2 | kv-b-proj | 6 | 512 | 28672 | bf16 | 48.99 | 56.98 | 1.163 | 2.2390e-03 | mma_lane_m32_n64_tile8_shared_a |
| glm-5.2 | kv-b-proj | 8 | 512 | 28672 | bf16 | 49.23 | 57.25 | 1.163 | 2.2390e-03 | mma_lane_m32_n64_tile4_shared_a |
| glm-5.2 | kv-b-proj | 16 | 512 | 28672 | bf16 | 48.3 | 58.03 | 1.201 | 2.2448e-03 | mma_lane_m32_n64_tile2_shared_a |
| glm-5.2 | o-proj | 1 | 16384 | 6144 | fp16 | 65.17 | 71.74 | 1.101 | 9.1219e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 2 | 16384 | 6144 | fp16 | 65.36 | 72.53 | 1.11 | 1.0240e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 4 | 16384 | 6144 | fp16 | 65.09 | 72.72 | 1.117 | 1.2410e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 6 | 16384 | 6144 | fp16 | 65.94 | 73.62 | 1.116 | 1.2410e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 8 | 16384 | 6144 | fp16 | 65.98 | 72.74 | 1.102 | 1.2410e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 16 | 16384 | 6144 | fp16 | 68.53 | 75.74 | 1.105 | 1.2410e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 1 | 16384 | 6144 | bf16 | 65.39 | 73.09 | 1.118 | 9.5510e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 2 | 16384 | 6144 | bf16 | 66.13 | 74.16 | 1.121 | 9.5501e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 4 | 16384 | 6144 | bf16 | 66.29 | 74.37 | 1.122 | 9.5501e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 6 | 16384 | 6144 | bf16 | 66.78 | 75.02 | 1.123 | 9.5501e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 8 | 16384 | 6144 | bf16 | 66.34 | 73.7 | 1.111 | 1.0403e-02 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | o-proj | 16 | 16384 | 6144 | bf16 | 68.61 | 77.89 | 1.135 | 1.0403e-02 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-up | 1 | 6144 | 12288 | fp16 | 53.94 | 61.23 | 1.135 | 6.7854e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 2 | 6144 | 12288 | fp16 | 54.35 | 61.44 | 1.13 | 7.0965e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 4 | 6144 | 12288 | fp16 | 54.67 | 61.92 | 1.133 | 7.0977e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 6 | 6144 | 12288 | fp16 | 56.99 | 62.5 | 1.097 | 7.0977e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 8 | 6144 | 12288 | fp16 | 57.57 | 62.35 | 1.083 | 7.0977e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-up | 16 | 6144 | 12288 | fp16 | 61.44 | 65.42 | 1.065 | 7.0977e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-up | 1 | 6144 | 12288 | bf16 | 61.28 | 62.46 | 1.019 | 5.3087e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| glm-5.2 | dense-up | 2 | 6144 | 12288 | bf16 | 55.25 | 62.74 | 1.136 | 5.3087e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 4 | 6144 | 12288 | bf16 | 56.35 | 63.66 | 1.13 | 5.3090e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 6 | 6144 | 12288 | bf16 | 57.6 | 63.6 | 1.104 | 5.3090e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | dense-up | 8 | 6144 | 12288 | bf16 | 57.78 | 63.31 | 1.096 | 5.7571e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-up | 16 | 6144 | 12288 | bf16 | 61.15 | 67.17 | 1.098 | 8.2128e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 1 | 12288 | 6144 | fp16 | 54.42 | 61.18 | 1.124 | 8.3685e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 2 | 12288 | 6144 | fp16 | 54.7 | 62.13 | 1.136 | 9.8658e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 4 | 12288 | 6144 | fp16 | 54.74 | 62.05 | 1.134 | 1.0469e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 6 | 12288 | 6144 | fp16 | 55.82 | 63.38 | 1.135 | 1.0469e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 8 | 12288 | 6144 | fp16 | 55.84 | 62.37 | 1.117 | 1.0469e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 16 | 12288 | 6144 | fp16 | 58.35 | 65.84 | 1.128 | 1.1213e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 1 | 12288 | 6144 | bf16 | 55.1 | 62.93 | 1.142 | 6.1283e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 2 | 12288 | 6144 | bf16 | 55.73 | 63.74 | 1.144 | 8.4248e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 4 | 12288 | 6144 | bf16 | 56.29 | 64.46 | 1.145 | 8.4245e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 6 | 12288 | 6144 | bf16 | 56.58 | 65.15 | 1.152 | 8.6837e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 8 | 12288 | 6144 | bf16 | 56.19 | 63.65 | 1.133 | 8.6837e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | dense-down | 16 | 12288 | 6144 | bf16 | 58.37 | 67.9 | 1.163 | 9.1536e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | moe-up | 1 | 6144 | 2048 | fp16 | 45.02 | 56.4 | 1.253 | 4.6635e-04 | gemv |
| glm-5.2 | moe-up | 2 | 6144 | 2048 | fp16 | 45.98 | 57.15 | 1.243 | 4.8113e-04 | gemv |
| glm-5.2 | moe-up | 4 | 6144 | 2048 | fp16 | 49.65 | 57.01 | 1.148 | 5.7971e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | moe-up | 6 | 6144 | 2048 | fp16 | 48.74 | 57.02 | 1.17 | 5.9330e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | moe-up | 8 | 6144 | 2048 | fp16 | 48.26 | 56.91 | 1.179 | 5.9330e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | moe-up | 16 | 6144 | 2048 | fp16 | 48.85 | 56.96 | 1.166 | 6.4921e-04 | mma_lane_m16_n32_splitk12_pipe2 |
| glm-5.2 | moe-up | 1 | 6144 | 2048 | bf16 | 48.54 | 56.21 | 1.158 | 4.0877e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | moe-up | 2 | 6144 | 2048 | bf16 | 47.92 | 57.17 | 1.193 | 4.0876e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | moe-up | 4 | 6144 | 2048 | bf16 | 49.06 | 56.56 | 1.153 | 4.9487e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | moe-up | 6 | 6144 | 2048 | bf16 | 49.04 | 56.98 | 1.162 | 4.9888e-03 | mma_lane_m16_n16_splitk16 |
| glm-5.2 | moe-up | 8 | 6144 | 2048 | bf16 | 48.26 | 56.74 | 1.176 | 5.3594e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | moe-up | 16 | 6144 | 2048 | bf16 | 48.58 | 57.31 | 1.18 | 5.3594e-03 | mma_lane_m16_n32_splitk12_pipe2_interleaved |
| glm-5.2 | moe-down | 1 | 2048 | 6144 | fp16 | 48.4 | 56.74 | 1.172 | 3.8958e-04 | mma_lane_m16_n16_splitk8 |
| glm-5.2 | moe-down | 2 | 2048 | 6144 | fp16 | 45.49 | 56.54 | 1.243 | 2.4360e-04 | gemv |
| glm-5.2 | moe-down | 4 | 2048 | 6144 | fp16 | 48.82 | 57.02 | 1.168 | 4.6110e-04 | mma_lane_m16_n16_splitk8 |
| glm-5.2 | moe-down | 6 | 2048 | 6144 | fp16 | 49.52 | 56.8 | 1.147 | 4.6766e-04 | mma_lane_m16_n16_splitk4 |
| glm-5.2 | moe-down | 8 | 2048 | 6144 | fp16 | 48.72 | 56.62 | 1.162 | 4.6766e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | moe-down | 16 | 2048 | 6144 | fp16 | 48.7 | 56.82 | 1.167 | 5.4288e-04 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | moe-down | 1 | 2048 | 6144 | bf16 | 48.27 | 57.25 | 1.186 | 3.4111e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | moe-down | 2 | 2048 | 6144 | bf16 | 49.06 | 57.09 | 1.164 | 3.4112e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | moe-down | 4 | 2048 | 6144 | bf16 | 48.61 | 56.22 | 1.157 | 3.4112e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | moe-down | 6 | 2048 | 6144 | bf16 | 48.86 | 57.2 | 1.171 | 3.4112e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | moe-down | 8 | 2048 | 6144 | bf16 | 48.67 | 56.78 | 1.167 | 3.4112e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | moe-down | 16 | 2048 | 6144 | bf16 | 47.94 | 57.66 | 1.203 | 4.5259e-03 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | indexer-wq-b | 1 | 2048 | 4096 | fp16 | 45.04 | 57.5 | 1.277 | 2.5725e-04 | gemv |
| glm-5.2 | indexer-wq-b | 2 | 2048 | 4096 | fp16 | 49.3 | 56.93 | 1.155 | 4.0048e-04 | mma_lane_m16_n16_padded |
| glm-5.2 | indexer-wq-b | 4 | 2048 | 4096 | fp16 | 46.13 | 57.31 | 1.242 | 4.8423e-04 | gemv |
| glm-5.2 | indexer-wq-b | 6 | 2048 | 4096 | fp16 | 48.58 | 57.15 | 1.177 | 4.9233e-04 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | indexer-wq-b | 8 | 2048 | 4096 | fp16 | 48.85 | 57.46 | 1.176 | 4.9233e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | indexer-wq-b | 16 | 2048 | 4096 | fp16 | 48.61 | 57.87 | 1.191 | 5.0068e-04 | mma_lane_m16_n32_splitk16_pipe2 |
| glm-5.2 | indexer-wq-b | 1 | 2048 | 4096 | bf16 | 48.35 | 56.9 | 1.177 | 2.8319e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | indexer-wq-b | 2 | 2048 | 4096 | bf16 | 48.85 | 56.77 | 1.162 | 3.5501e-03 | mma_lane_m16_n32_splitk16 |
| glm-5.2 | indexer-wq-b | 4 | 2048 | 4096 | bf16 | 49.04 | 56.98 | 1.162 | 3.5501e-03 | mma_lane_m16_n64_shared_a |
| glm-5.2 | indexer-wq-b | 6 | 2048 | 4096 | bf16 | 48.69 | 56.29 | 1.156 | 3.5501e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| glm-5.2 | indexer-wq-b | 8 | 2048 | 4096 | bf16 | 48.99 | 57.7 | 1.178 | 3.5501e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| glm-5.2 | indexer-wq-b | 16 | 2048 | 4096 | bf16 | 48.8 | 57.47 | 1.178 | 3.5499e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| glm-5.2 | lm-head | 1 | 6144 | 154880 | fp16 | 290.94 | 351.95 | 1.21 | 8.2147e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | lm-head | 2 | 6144 | 154880 | fp16 | 293.09 | 356.45 | 1.216 | 8.2052e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | lm-head | 4 | 6144 | 154880 | fp16 | 290.98 | 353.76 | 1.216 | 8.2052e-04 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 6 | 6144 | 154880 | fp16 | 292.72 | 356.7 | 1.219 | 8.8596e-04 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 8 | 6144 | 154880 | fp16 | 294.16 | 358.18 | 1.218 | 8.8716e-04 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 16 | 6144 | 154880 | fp16 | 302.14 | 385.68 | 1.276 | 1.0233e-03 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 1 | 6144 | 154880 | bf16 | 290.42 | 363.7 | 1.252 | 5.3958e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| glm-5.2 | lm-head | 2 | 6144 | 154880 | bf16 | 292.86 | 361.62 | 1.235 | 5.9876e-03 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 4 | 6144 | 154880 | bf16 | 294.32 | 363.2 | 1.234 | 5.9876e-03 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 6 | 6144 | 154880 | bf16 | 295.68 | 365.81 | 1.237 | 7.6132e-03 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 8 | 6144 | 154880 | bf16 | 300.13 | 369.82 | 1.232 | 7.6149e-03 | mma_lane_m16_n64_tile4_shared_a |
| glm-5.2 | lm-head | 16 | 6144 | 154880 | bf16 | 306.83 | 405.6 | 1.322 | 8.8058e-03 | mma_lane_m16_n64_tile4_shared_a |
| kimi-k2.5 | q-a-proj | 1 | 7168 | 1536 | fp16 | 47.76 | 58.94 | 1.234 | 4.8733e-04 | gemv |
| kimi-k2.5 | q-a-proj | 2 | 7168 | 1536 | fp16 | 47.81 | 59.26 | 1.24 | 4.8757e-04 | gemv |
| kimi-k2.5 | q-a-proj | 4 | 7168 | 1536 | fp16 | 50.98 | 59.14 | 1.16 | 7.4720e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | q-a-proj | 6 | 7168 | 1536 | fp16 | 50.66 | 59.84 | 1.181 | 7.4720e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | q-a-proj | 8 | 7168 | 1536 | fp16 | 50.34 | 58.8 | 1.168 | 7.4720e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | q-a-proj | 16 | 7168 | 1536 | fp16 | 50.88 | 58.48 | 1.149 | 7.4720e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | q-a-proj | 1 | 7168 | 1536 | bf16 | 50.13 | 58.51 | 1.167 | 7.8938e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | q-a-proj | 2 | 7168 | 1536 | bf16 | 50.58 | 58.59 | 1.158 | 7.8940e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | q-a-proj | 4 | 7168 | 1536 | bf16 | 49.97 | 58.66 | 1.174 | 7.8940e-03 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | q-a-proj | 6 | 7168 | 1536 | bf16 | 50.21 | 59.95 | 1.194 | 7.8940e-03 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | q-a-proj | 8 | 7168 | 1536 | bf16 | 50.82 | 57.42 | 1.13 | 7.8940e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | q-a-proj | 16 | 7168 | 1536 | bf16 | 50.56 | 58.5 | 1.157 | 7.8940e-03 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | q-b-proj | 1 | 1536 | 12288 | fp16 | 46.21 | 57.73 | 1.249 | 2.4390e-04 | gemv |
| kimi-k2.5 | q-b-proj | 2 | 1536 | 12288 | fp16 | 50.74 | 57.57 | 1.135 | 3.7152e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | q-b-proj | 4 | 1536 | 12288 | fp16 | 50.02 | 58.14 | 1.163 | 3.7152e-04 | mma_lane_m16_n32_splitk12_pipe2 |
| kimi-k2.5 | q-b-proj | 6 | 1536 | 12288 | fp16 | 50.78 | 58.02 | 1.142 | 3.7152e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | q-b-proj | 8 | 1536 | 12288 | fp16 | 50.26 | 58.75 | 1.169 | 3.7152e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | q-b-proj | 16 | 1536 | 12288 | fp16 | 50.54 | 58.96 | 1.167 | 3.7152e-04 | mma_lane_m16_n32_splitk12 |
| kimi-k2.5 | q-b-proj | 1 | 1536 | 12288 | bf16 | 49.68 | 57.02 | 1.148 | 2.6321e-03 | mma_lane_m16_n16_splitk12 |
| kimi-k2.5 | q-b-proj | 2 | 1536 | 12288 | bf16 | 49.73 | 57.25 | 1.151 | 3.8904e-03 | mma_lane_m16_n32_splitk12 |
| kimi-k2.5 | q-b-proj | 4 | 1536 | 12288 | bf16 | 50.48 | 58.48 | 1.158 | 3.8904e-03 | mma_lane_m16_n64_splitk12_pipe2_interleaved |
| kimi-k2.5 | q-b-proj | 6 | 1536 | 12288 | bf16 | 51.3 | 58.56 | 1.142 | 3.8904e-03 | mma_lane_m16_n64_splitk12_pipe2_interleaved |
| kimi-k2.5 | q-b-proj | 8 | 1536 | 12288 | bf16 | 51.02 | 58.78 | 1.152 | 3.8904e-03 | mma_lane_m16_n32_splitk12_pipe2 |
| kimi-k2.5 | q-b-proj | 16 | 1536 | 12288 | bf16 | 50.74 | 58.3 | 1.149 | 3.8906e-03 | mma_lane_m16_n64_splitk12_pipe2_interleaved |
| kimi-k2.5 | kv-a-proj-mqa | 1 | 7168 | 576 | fp16 | 46.35 | 56.93 | 1.228 | 4.2903e-04 | gemv |
| kimi-k2.5 | kv-a-proj-mqa | 2 | 7168 | 576 | fp16 | 45.63 | 58.37 | 1.279 | 4.2903e-04 | gemv |
| kimi-k2.5 | kv-a-proj-mqa | 4 | 7168 | 576 | fp16 | 46.8 | 58.21 | 1.244 | 4.6289e-04 | gemv |
| kimi-k2.5 | kv-a-proj-mqa | 6 | 7168 | 576 | fp16 | 46.77 | 58.02 | 1.241 | 4.6289e-04 | gemv |
| kimi-k2.5 | kv-a-proj-mqa | 8 | 7168 | 576 | fp16 | 50.85 | 58.93 | 1.159 | 6.1893e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | kv-a-proj-mqa | 16 | 7168 | 576 | fp16 | 50.38 | 58.77 | 1.166 | 6.4003e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | kv-a-proj-mqa | 1 | 7168 | 576 | bf16 | 49.73 | 58.19 | 1.17 | 3.6271e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | kv-a-proj-mqa | 2 | 7168 | 576 | bf16 | 45.81 | 59.12 | 1.291 | 3.7965e-03 | gemv |
| kimi-k2.5 | kv-a-proj-mqa | 4 | 7168 | 576 | bf16 | 50.74 | 59.3 | 1.169 | 5.1389e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | kv-a-proj-mqa | 6 | 7168 | 576 | bf16 | 50.85 | 59.57 | 1.171 | 5.1389e-03 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | kv-a-proj-mqa | 8 | 7168 | 576 | bf16 | 50.88 | 58.8 | 1.156 | 5.1389e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | kv-a-proj-mqa | 16 | 7168 | 576 | bf16 | 50.13 | 58.32 | 1.163 | 5.1389e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | kv-b-proj | 1 | 512 | 16384 | fp16 | 45.79 | 57.5 | 1.256 | 2.3425e-04 | gemv |
| kimi-k2.5 | kv-b-proj | 2 | 512 | 16384 | fp16 | 50.14 | 58.21 | 1.161 | 2.5880e-04 | mma_lane_m32_n64_tile2_shared_a |
| kimi-k2.5 | kv-b-proj | 4 | 512 | 16384 | fp16 | 46.7 | 58.53 | 1.253 | 2.3419e-04 | gemv |
| kimi-k2.5 | kv-b-proj | 6 | 512 | 16384 | fp16 | 51.26 | 59.1 | 1.153 | 2.7806e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | kv-b-proj | 8 | 512 | 16384 | fp16 | 51.14 | 58.59 | 1.146 | 2.9051e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | kv-b-proj | 16 | 512 | 16384 | fp16 | 50.29 | 58.11 | 1.156 | 3.1704e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | kv-b-proj | 1 | 512 | 16384 | bf16 | 46.38 | 58.4 | 1.259 | 1.8864e-03 | gemv |
| kimi-k2.5 | kv-b-proj | 2 | 512 | 16384 | bf16 | 47.04 | 58.18 | 1.237 | 1.9429e-03 | gemv |
| kimi-k2.5 | kv-b-proj | 4 | 512 | 16384 | bf16 | 49.86 | 58.69 | 1.177 | 2.0976e-03 | mma_lane_m16_n64_tile4_shared_a |
| kimi-k2.5 | kv-b-proj | 6 | 512 | 16384 | bf16 | 50.75 | 58.42 | 1.151 | 2.6867e-03 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | kv-b-proj | 8 | 512 | 16384 | bf16 | 50.5 | 58.66 | 1.162 | 2.6866e-03 | mma_lane_m32_n64_tile8_shared_a |
| kimi-k2.5 | kv-b-proj | 16 | 512 | 16384 | bf16 | 50.34 | 59.23 | 1.177 | 2.6866e-03 | mma_lane_m32_n64_tile8_shared_a |
| kimi-k2.5 | o-proj | 1 | 8192 | 7168 | fp16 | 50.99 | 59.22 | 1.161 | 7.3338e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | o-proj | 2 | 8192 | 7168 | fp16 | 51.44 | 59.7 | 1.16 | 7.3349e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | o-proj | 4 | 8192 | 7168 | fp16 | 51.63 | 58.98 | 1.142 | 8.5855e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | o-proj | 6 | 8192 | 7168 | fp16 | 52.5 | 59.62 | 1.136 | 9.3174e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | o-proj | 8 | 8192 | 7168 | fp16 | 52.14 | 59.57 | 1.142 | 9.3174e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | o-proj | 16 | 8192 | 7168 | fp16 | 52.58 | 59.68 | 1.135 | 1.0259e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | o-proj | 1 | 8192 | 7168 | bf16 | 50.82 | 59.38 | 1.168 | 6.9695e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| kimi-k2.5 | o-proj | 2 | 8192 | 7168 | bf16 | 50.69 | 58.61 | 1.156 | 6.9695e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | o-proj | 4 | 8192 | 7168 | bf16 | 51.58 | 60.02 | 1.163 | 7.3092e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| kimi-k2.5 | o-proj | 6 | 8192 | 7168 | bf16 | 52.05 | 59.12 | 1.136 | 7.3092e-03 | mma_lane_m16_n64_splitk16_pipe2_interleaved |
| kimi-k2.5 | o-proj | 8 | 8192 | 7168 | bf16 | 51.39 | 59.34 | 1.155 | 8.9262e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | o-proj | 16 | 8192 | 7168 | bf16 | 51.12 | 61.38 | 1.201 | 8.9264e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-up | 1 | 7168 | 18432 | fp16 | 75.07 | 82.67 | 1.101 | 8.8072e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | dense-up | 2 | 7168 | 18432 | fp16 | 76.48 | 83.14 | 1.087 | 8.8024e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | dense-up | 4 | 7168 | 18432 | fp16 | 80.66 | 84.35 | 1.046 | 8.8024e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | dense-up | 6 | 7168 | 18432 | fp16 | 81.65 | 84.69 | 1.037 | 8.8024e-04 | mma_lane_m16_n32_splitk8 |
| kimi-k2.5 | dense-up | 8 | 7168 | 18432 | fp16 | 81.2 | 83.66 | 1.03 | 9.0432e-04 | mma_lane_m16_n32_splitk8 |
| kimi-k2.5 | dense-up | 16 | 7168 | 18432 | fp16 | 95.44 | 88.05 | 0.923 | 1.0848e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-up | 1 | 7168 | 18432 | bf16 | 75.63 | 84.0 | 1.111 | 6.3446e-03 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | dense-up | 2 | 7168 | 18432 | bf16 | 76.94 | 84.86 | 1.103 | 6.3446e-03 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | dense-up | 4 | 7168 | 18432 | bf16 | 80.32 | 85.58 | 1.066 | 6.3448e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | dense-up | 6 | 7168 | 18432 | bf16 | 84.74 | 85.74 | 1.012 | 6.3448e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | dense-up | 8 | 7168 | 18432 | bf16 | 85.2 | 85.52 | 1.004 | 6.3448e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| kimi-k2.5 | dense-up | 16 | 7168 | 18432 | bf16 | 88.8 | 90.0 | 1.014 | 6.5858e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| kimi-k2.5 | dense-down | 1 | 18432 | 7168 | fp16 | 71.01 | 82.77 | 1.166 | 1.0622e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 2 | 18432 | 7168 | fp16 | 71.12 | 83.54 | 1.175 | 1.1473e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 4 | 18432 | 7168 | fp16 | 71.7 | 83.79 | 1.169 | 1.3602e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 6 | 18432 | 7168 | fp16 | 72.24 | 84.53 | 1.17 | 1.3602e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 8 | 18432 | 7168 | fp16 | 72.45 | 83.47 | 1.152 | 1.3602e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 16 | 18432 | 7168 | fp16 | 74.96 | 87.26 | 1.164 | 1.3607e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 1 | 18432 | 7168 | bf16 | 70.82 | 84.14 | 1.188 | 8.2707e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 2 | 18432 | 7168 | bf16 | 71.49 | 85.39 | 1.194 | 8.2710e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 4 | 18432 | 7168 | bf16 | 72.1 | 85.2 | 1.182 | 1.0396e-02 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 6 | 18432 | 7168 | bf16 | 71.65 | 85.47 | 1.193 | 1.0396e-02 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 8 | 18432 | 7168 | bf16 | 72.32 | 85.01 | 1.175 | 1.0396e-02 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | dense-down | 16 | 18432 | 7168 | bf16 | 74.78 | 89.3 | 1.194 | 1.0991e-02 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | shared-up | 1 | 7168 | 2048 | fp16 | 49.57 | 57.66 | 1.163 | 6.1047e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | shared-up | 2 | 7168 | 2048 | fp16 | 50.26 | 57.97 | 1.153 | 6.1047e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | shared-up | 4 | 7168 | 2048 | fp16 | 50.91 | 58.67 | 1.152 | 6.5851e-04 | mma_lane_m16_n32_splitk8 |
| kimi-k2.5 | shared-up | 6 | 7168 | 2048 | fp16 | 50.43 | 57.76 | 1.145 | 7.4601e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | shared-up | 8 | 7168 | 2048 | fp16 | 50.7 | 58.62 | 1.156 | 7.4601e-04 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | shared-up | 16 | 7168 | 2048 | fp16 | 50.99 | 58.66 | 1.15 | 7.4601e-04 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | shared-up | 1 | 7168 | 2048 | bf16 | 47.01 | 57.95 | 1.233 | 3.6817e-03 | gemv |
| kimi-k2.5 | shared-up | 2 | 7168 | 2048 | bf16 | 49.58 | 57.94 | 1.168 | 5.8935e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | shared-up | 4 | 7168 | 2048 | bf16 | 50.5 | 58.59 | 1.16 | 5.8935e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | shared-up | 6 | 7168 | 2048 | bf16 | 50.06 | 57.79 | 1.154 | 5.8935e-03 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | shared-up | 8 | 7168 | 2048 | bf16 | 50.21 | 58.51 | 1.165 | 5.8935e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | shared-up | 16 | 7168 | 2048 | bf16 | 51.33 | 58.82 | 1.146 | 6.3229e-03 | mma_lane_m16_n64_splitk24_pipe2_interleaved |
| kimi-k2.5 | shared-down | 1 | 2048 | 7168 | fp16 | 46.86 | 58.64 | 1.251 | 4.2987e-04 | gemv |
| kimi-k2.5 | shared-down | 2 | 2048 | 7168 | fp16 | 47.49 | 58.48 | 1.231 | 4.2999e-04 | gemv |
| kimi-k2.5 | shared-down | 4 | 2048 | 7168 | fp16 | 50.59 | 58.72 | 1.161 | 4.3678e-04 | mma_lane_m16_n32_splitk8 |
| kimi-k2.5 | shared-down | 6 | 2048 | 7168 | fp16 | 50.74 | 58.62 | 1.155 | 4.3678e-04 | mma_lane_m16_n32_splitk16 |
| kimi-k2.5 | shared-down | 8 | 2048 | 7168 | fp16 | 51.09 | 58.96 | 1.154 | 4.3678e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | shared-down | 16 | 2048 | 7168 | fp16 | 50.51 | 58.42 | 1.156 | 4.3678e-04 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | shared-down | 1 | 2048 | 7168 | bf16 | 50.5 | 58.88 | 1.166 | 3.1633e-03 | mma_lane_m16_n16_splitk16 |
| kimi-k2.5 | shared-down | 2 | 2048 | 7168 | bf16 | 51.12 | 59.02 | 1.155 | 3.1634e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| kimi-k2.5 | shared-down | 4 | 2048 | 7168 | bf16 | 50.46 | 59.1 | 1.171 | 3.8900e-03 | mma_lane_m16_n32_splitk16 |
| kimi-k2.5 | shared-down | 6 | 2048 | 7168 | bf16 | 51.31 | 58.62 | 1.143 | 3.8900e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | shared-down | 8 | 2048 | 7168 | bf16 | 51.12 | 58.86 | 1.151 | 3.8900e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| kimi-k2.5 | shared-down | 16 | 2048 | 7168 | bf16 | 50.82 | 57.95 | 1.14 | 3.8900e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
| kimi-k2.5 | router-gate | 1 | 7168 | 384 | fp16 | 46.88 | 67.76 | 1.445 | 4.3106e-04 | gemv |
| kimi-k2.5 | router-gate | 2 | 7168 | 384 | fp16 | 47.18 | 76.51 | 1.622 | 4.5669e-04 | gemv |
| kimi-k2.5 | router-gate | 4 | 7168 | 384 | fp16 | 47.46 | 77.04 | 1.623 | 5.2381e-04 | gemv |
| kimi-k2.5 | router-gate | 6 | 7168 | 384 | fp16 | 48.05 | 79.63 | 1.657 | 5.2381e-04 | gemv |
| kimi-k2.5 | router-gate | 8 | 7168 | 384 | fp16 | 47.12 | 70.1 | 1.488 | 5.2381e-04 | gemv |
| kimi-k2.5 | router-gate | 16 | 7168 | 384 | fp16 | 49.81 | 74.3 | 1.492 | 6.3384e-04 | mma_lane_m16_n16_splitk4 |
| kimi-k2.5 | router-gate | 1 | 7168 | 384 | bf16 | 46.3 | 68.1 | 1.471 | 3.7947e-03 | gemv |
| kimi-k2.5 | router-gate | 2 | 7168 | 384 | bf16 | 50.62 | 76.32 | 1.508 | 5.1576e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | router-gate | 4 | 7168 | 384 | bf16 | 46.82 | 77.1 | 1.647 | 3.8912e-03 | gemv |
| kimi-k2.5 | router-gate | 6 | 7168 | 384 | bf16 | 50.75 | 79.74 | 1.571 | 5.4588e-03 | mma_lane_m16_n16_splitk8 |
| kimi-k2.5 | router-gate | 8 | 7168 | 384 | bf16 | 50.02 | 70.0 | 1.4 | 5.4588e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | router-gate | 16 | 7168 | 384 | bf16 | 49.81 | 74.93 | 1.504 | 5.4588e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 1 | 7168 | 163840 | fp16 | 354.69 | 422.14 | 1.19 | 1.0767e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 2 | 7168 | 163840 | fp16 | 359.15 | 426.16 | 1.187 | 1.0788e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 4 | 7168 | 163840 | fp16 | 368.34 | 428.48 | 1.163 | 1.0788e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 6 | 7168 | 163840 | fp16 | 384.35 | 433.58 | 1.128 | 1.0786e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 8 | 7168 | 163840 | fp16 | 408.53 | 435.79 | 1.067 | 1.0788e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 16 | 7168 | 163840 | fp16 | 446.24 | 462.59 | 1.037 | 1.1082e-03 | mma_lane_m16_n64_tile4_shared_a |
| kimi-k2.5 | lm-head | 1 | 7168 | 163840 | bf16 | 356.83 | 440.69 | 1.235 | 6.6519e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 2 | 7168 | 163840 | bf16 | 361.09 | 441.12 | 1.222 | 7.7360e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 4 | 7168 | 163840 | bf16 | 368.96 | 447.54 | 1.213 | 8.0559e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 6 | 7168 | 163840 | bf16 | 379.1 | 450.96 | 1.19 | 8.0576e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 8 | 7168 | 163840 | bf16 | 405.74 | 451.01 | 1.112 | 8.0559e-03 | mma_lane_m16_n32_splitk8_pipe2 |
| kimi-k2.5 | lm-head | 16 | 7168 | 163840 | bf16 | 453.33 | 491.15 | 1.083 | 8.2383e-03 | mma_lane_m16_n64_splitk8_pipe2_interleaved |
