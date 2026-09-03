# Phase 40: H100 W2 lane-pair level table

Phase 40 promotes Phase 39's block-local lane-pair level table for W2. The
matched H100 result improves every focused gate/up and complete-MLP cell. W2
now matches the Machete W4 complete-MLP geometric mean within 0.4% and wins at
M=1, M=2, and M=4.

## Profile gate

The original compact W2 table executed 1,966,080 shared loads and generated
1,872,784 shared-load bank conflicts. It therefore had the same random lookup
serialization as W2.5 and W3.5, despite W2's shorter four-bit state window.

The promoted representation is the exact lane-pair table defined in Phase 39:

\[
T[i,2p]=L[i],
\qquad
T[i,2p+1]=L[i\mathbin{\oplus}(i\gg7)],
\]

where `p` is the consumer lane pair. It changes only shared-memory placement.
The fixed W2 grouped launch descriptor, continuous-window payload, selectors,
PGC result, WGMMA order, and output recovery remain unchanged.

W2's block allocation grows from 27,264 to 42,752 bytes. This is below the
default 48 KiB limit, needs no dynamic shared-memory opt-in, retains the
two-stage TMA pipeline, and adds neither persistent nor transient global VRAM.

## Matched isolated H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate.
Timing uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample for grouped gate/up inner P32 plus paired recovery.

| M/K/N per child | Compact W2 | Lane-pair W2 | vs compact | Better than last benchmark |
|:--|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 30.989 us | 29.154 us | 1.0629x | Yes |
| 2/2048/8192 x2 | 30.794 us | 28.856 us | 1.0672x | Yes |
| 4/2048/8192 x2 | 31.111 us | 29.212 us | 1.0650x | Yes |
| 8/2048/8192 x2 | 31.486 us | 29.535 us | 1.0660x | Yes |
| 16/2048/8192 x2 | 32.221 us | 30.308 us | 1.0631x | Yes |

The isolated geometric mean is **1.0649x**.

## Nsight Compute result

| Metric | Compact W2 | Lane-pair W2 | Change |
|:--|--:|--:|--:|
| Replay duration | 28.864 us | 26.976 us | 1.0700x |
| Shared-load bank conflicts | 1,872,784 | 1,051,197 | -43.87% |
| Shared-load wavefronts | 4,197,173 | 3,343,802 | -20.33% |
| Executed instructions | 10,256,926 | 10,394,898 | +1.35% |
| Eligible warps/cycle | 0.52 | 0.59 | +13.5% |
| Long-scoreboard stalls | 1.40 | 1.36 | -2.9% |
| Registers/thread | 48 | 48 | unchanged |

As in Phase 39, removing serialized shared transactions is more valuable than
the modest added address instruction count.

## Complete Llama 3.2 1B MLP

The complete path includes shared input transform, grouped gate/up P32,
recovery, fused SiLU/product/down preconditioning, split-16 down P32, and down
recovery. Effective throughput counts logical dense-equivalent FLOPs. Marlin
and Machete are figurative W4 baselines, not equal-rate work comparisons.
`Better` compares with Phase 39.

| MKN: gate/up x2; down | QVQ W2 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2; 1/8192/2048 | 49.873 us | 2.018 | 0.588x | 1.014x | Yes |
| 2/2048/8192 x2; 2/8192/2048 | 50.025 us | 4.025 | 0.624x | 1.004x | Yes |
| 4/2048/8192 x2; 4/8192/2048 | 50.638 us | 7.952 | 0.620x | 1.002x | Yes |
| 8/2048/8192 x2; 8/8192/2048 | 51.366 us | 15.678 | 0.570x | 0.991x | Yes |
| 16/2048/8192 x2; 16/8192/2048 | 52.535 us | 30.658 | 0.619x | 0.972x | Yes |

W2 improves **1.0507x** in the complete MLP, with five of five wins. Its
Machete-relative geometric mean is **0.9964x**. The all-rate 20-cell geometric
mean improves **1.0106x**; unchanged W2.5/W3/W3.5 movement is cross-run
telemetry and is not attributed to this W2-only source change.

## Correctness and scope

- All five focused outputs are exact and CUDA Graph stable.
- The complete MLP remains bit-exact to its unfused reference.
- 126 Hopper P32/grouped tests pass across W2-W3.5, ordered reductions,
  child parity, dense-oracle bounds, and graph replay.
- No checkpoint format, bits per weight, kernel arithmetic, split policy,
  persistent VRAM, or external workspace changed.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase40_h100/production_mlp_all_lane_levels_vs_phase39.json`
- `artifacts/a41_phase40_h100/w2_lane_level_profile.json`
- binary reports outside Git under
  `/root/qvq-profiler-artifacts/phase40-w2-levels/`

## Next phase

Every supported W2-W3.5 rate now uses the same measured lane-pair level
layout. Phase 41 should re-profile the resulting common decoder and target the
remaining 1.05 million conflicts without crossing the 48 KiB block limit.
The next candidate must preserve two-stage TMA and should prefer a cheaper
bank permutation or producer-time table construction over a larger table.
