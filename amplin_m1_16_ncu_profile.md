# Amplin M=1-16 small-batch kernels — Nsight Compute profile and strategy

Branch: `devin/1784971296-amplin-m32-ncu` (PR #71)  
GPU: NVIDIA PG506-230 (A100-class), sm_80, 124 SMs, 96 GiB  
Config: GPTQ 4-bit, group_size=128, desc_act=False, sym=True, FP16 and BF16  
Clocks: locked to 1410 MHz SM / 1593 MHz memory for stable timing  

## Current status

Latest static-routing refresh (2 clear wins) measured with `bench_static_m1_16.py`:

| metric | value |
|---|---|
| total shapes | 420 |
| legal Marlin shapes | 396 |
| Amplin wins | 383 |
| losses | 13 |
| geomean speedup | 1.238 |

Routing changes committed to `gptqmodel/utils/amplin_dynamic_routing_table.json`:

| key | old kernel | new kernel | speedup vs Marlin | reason |
|---|---|---|---|---|
| `(4, 9216, 3072, bf16)` | `mma_lane_m16_n64_splitk12x2_coop_interleaved` | `mma_lane_m16_n32_splitk12_pipe2_interleaved` | 0.90x → 1.21x | Laguna `o-proj-9216` M=4 BF16 |
| `(4, 12288, 6144, bf16)` | `mma_lane_m16_n64_splitk12x2_coop_interleaved` | `mma_lane_m16_n64_splitk24_pipe2_interleaved` | 0.96x → 1.08x | GLM `dense-down` M=4 BF16 |

The 13 remaining losses are all within 10% of raw Marlin and are dominated by `kimi-k2.5 dense-up` M=4-16 K=7168 N=18432 plus a few Laguna/GLM BF16 projections. Focused NCU on the GLM win and a representative Kimi loss is below.

## NCU command used

```bash
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --section InstructionStats \
    --launch-skip 10 --launch-count 1 \
    -k 'regex:.*<kernel-name>.*' \
    python profile_m1_16_ncu.py --m <M> --k <K> --n <N> --dtype <fp16|bf16> --kernel <kernel>
```

## 1. Patched win — GLM 5.2 dense-down M=4 K=12288 N=6144 BF16

Kernel: `mma_lane_m16_n64_splitk24_pipe2_interleaved_kernel`  
Selected because it replaces `splitk12x2_coop_interleaved` on this shape.

| metric | value |
|---|---|
| Grid | (96, 1, 1) |
| Block | (768, 1, 1) |
| Duration | 37.25 us |
| SM Busy | 40.17 % |
| Memory Throughput | 45.10 % |
| DRAM Throughput | 45.10 % |
| Achieved Occupancy | 36.89 % |
| Registers / thread | 80 |
| Dynamic shared mem / block | 98.30 KiB |
| Waves / SM | 0.77 |

**Observations:** The kernel is occupancy-bound (37.5% theoretical, limited by 80 registers and 98 KiB shared memory). Only 0.77 waves are launched, leaving SMs idle. Memory throughput is moderate at 45% of peak, so the kernel is latency/occupancy limited rather than bandwidth saturated. Despite this, it is faster than the cooperative `splitk12x2` path for this small M and moderate N.

## 2. Remaining loss — Kimi K2.5 dense-up M=8 K=7168 N=18432 BF16

Kernels compared: `mma_lane_m16_n64_splitk24_pipe2_interleaved_kernel` (current static selection) and `mma_lane_m16_n64_splitk4_pipe2_interleaved_kernel` (best raw candidate from sweep).

| metric | splitk24 pipe2 | splitk4 pipe2 |
|---|---|---|
| Grid | (288, 1, 1) | (288, 1, 1) |
| Block | 768 | 128 |
| Duration | 67.49 us | 63.74 us |
| SM Busy | 38.79 % | 28.28 % |
| Memory Throughput | 43.17 % | 45.59 % |
| DRAM Throughput | 43.17 % | 45.59 % |
| Achieved Occupancy | 36.40 % | 14.49 % |
| Theoretical Occupancy | 37.50 % | 31.25 % |
| Registers / thread | 80 | 96 |
| Dynamic shared mem / block | 98.30 KiB | 16.38 KiB |
| Waves / SM | 2.32 | 0.46 |

**Observations:**

- `splitk24` uses 768 threads and 98 KiB shared memory. It has enough work for 2.32 waves but a 41-block partial-wave tail that NCU flags as a 33% potential slowdown. Occupancy is good (36.4% achieved) but SM busy is only 38.8%, pointing to latency stalls rather than throughput saturation.
- `splitk4` is 5.6% faster in raw duration (63.7 us vs 67.5 us) but has very low achieved occupancy (14.5% vs 31.3% theoretical). The 128-thread block is too small to fill the SM, with only 0.46 waves. The gain comes from lower launch/sync overhead, not from better utilization.
- Neither kernel is close to memory or compute peak, so the remaining `kimi-k2.5 dense-up` losses are latency-limited. A more aggressive N-tile or larger warp count is likely needed to close the gap on this N=18432 shape.

## 3. Strategy for remaining M=1-16 losses

The 13 remaining losses fall into two families:

1. **K-dominant dense projections** (`kimi-k2.5 dense-up` M=4-16 K=7168 N=18432, `glm-5.2 dense-up/down` BF16). These have large K and moderate-to-large N. The current `splitk4`/`splitk24` `mma_lane` kernels are occupancy/launch-bound. A 6- or 8-warp `mma_lane` block with a larger N-tile and fewer split-K waves could reduce tail latency while keeping memory parallelism.

2. **Small/medium N Laguna/GLM projections** (e.g. `laguna-s-2.1 o-proj-9216`, `glm-5.2 kv-a-proj-mqa`). These have small N (3072/576) and are sensitive to block-size and mma-lane fragment count. The `splitk12x2_coop_interleaved` cooperative path sometimes wins but is limited by resident CTAs; tuning the cooperative grid or falling back to a non-cooperative `splitk8`/`splitk12` may help.

Next experiments:

- Add `mma_lane_m16_n64_splitk6_pipe2_interleaved` and `splitk8` variants tuned for the dense-up N=18432 shapes to reduce tail waves and occupancy gaps.
- Profile `glm-5.2 dense-up/down` and `kimi-k2.5 dense-up` with different `kMmaLaneSplitK*` warp counts to find the occupancy/wave-size sweet spot for K=7168/12288.
- Re-run `sweep_candidates_batch.py` for the remaining 13 loss shapes after each candidate addition; apply only changes that beat raw Marlin in the same sweep run.

## 2026-07-25: M16 N64 tile2/tile4 interleaved-dequant attempt

Implemented and registered a generic `mma_lane_mN_n64_tileN_interleaved_dequant` kernel
(templated on `BlockM` and `NTiles`) and exposed M16 `tile2` and `tile4` instantiations
in `gptqmodel_ext/amplin/amplin_kernel.cu`, `amplin.cpp`, and `gptqmodel/utils/amplin.py`.
`pytest -q tests/kernels/test_amplin.py` passed 88/88.

Result from `run_loss_sweep_batch.py` on the remaining M=1-16 loss shapes:
- `mma_lane_m16_n64_tile2_interleaved_dequant` and `tile4` are ~2x slower than the
  existing `splitk4_pipe2_interleaved` path on `kimi-k2.5 dense-up` K=7168 N=18432.
- The wider N-tile does not help the target shapes; both variants are not selected by
  the dynamic router.
- Example (`kimi-k2.5 dense-up` M=16 BF16, raw Marlin ~57.4 µs):
  - `splitk4_pipe2_interleaved` ~61.8 µs (0.93x)
  - `tile2_interleaved_dequant` ~126.9 µs (0.45x)
  - `tile4_interleaved_dequant` ~130.7 µs (0.44x)

The serial N-tile `tile2x2` route was already documented as regressed for M32 in
`amplin_kernel.md`; the M16 tile2/tile4 interleaved path shows the same pattern:
extra register pressure and serialized N work inside the K-group loop outweigh any
A-reuse savings. The prototype code was reverted; only this log entry remains.

Current static-routing table (re-run, A100 sm_80, clocks at max 1410/1593 MHz):

| metric | value |
|---|---|
| total shapes | 420 |
| legal Marlin shapes | 396 |
| Amplin wins | 383 |
| losses | 13 |
| geomean speedup | 1.46x |

The 13 remaining losses are still dominated by `kimi-k2.5 dense-up` M=4/6/8/16
K=7168 N=18432 and `glm-5.2 dense-up` M=8/16 K=6144 N=12288. The next route is a
Marlin-style `cp.async` weight pipeline in the `mma_lane` tile2 path and/or a
more-independent-warps mega-kernel, as recommended in the M32 NCU profile.
