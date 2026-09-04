# QVQ A41/R0 Phase 14: deeper H100 W3 decode/WGMMA overlap

Phase 14 keeps three W3 register-sourced WGMMA input fragments in flight
instead of two. The extra fragment allows one more asynchronous WGMMA group
to overlap the shared-memory P32 decode before a consumer must wait and reuse
operand registers. W2, W2.5, and W3.5 retain the measured depth-two schedule.

This is a scheduling-only change. Decode math, PGC results, level lookups,
WGMMA issue order, FP32 accumulation order, split-K, recovery, payload layout,
CUDA Graph topology, transient workspace, persistent VRAM, and non-Hopper
fallbacks are unchanged.

## Pipeline contract

Let `D_k` decode P32 K16 tile `k` into a register fragment and let `G_k`
issue the corresponding register-sourced WGMMA operation. The mathematical
order remains:

\[
D_0,G_0,D_1,G_1,\ldots,D_{15},G_{15}.
\]

Phase 13 alternated two fragments:

```text
k mod 2:       A0 A1 A0 A1 ...
wait at k>=2:  warpgroup_wait<1>
```

Before reusing `A0` or `A1`, `warpgroup_wait<1>` leaves at most one committed
WGMMA group pending. W3 decode is dominated by shared random level reads and
their dependent integer address/PGC chain, so the two-fragment schedule leaves
less independent work available to the warp-group scheduler.

Phase 14 uses:

```text
k mod 3:       A0 A1 A2 A0 A1 A2 ...
wait at k>=3:  warpgroup_wait<2>
```

The oldest group completes before its fragment is reused, while up to two
newer committed groups remain pending. The accumulator still receives WGMMA
operations in ascending K order, and the existing final `warpgroup_wait<0>`
still completes all work before the pipeline stage is released.

The third fragment is selected only for transition width six (W3). For other
rates the compile-time depth is two and ptxas removes the unreachable third
fragment.

## Matched gate/up A/B

Both sources use the same physical H100, deterministic payload/input seeds,
strict zero-MiB idle admission, 30 warmups, 300 CUDA-event samples, and 50
CUDA Graph replays per sample. The operation includes grouped W3 gate/up inner
P32 and exact paired recovery. Every output is bit-exact, repeatable over ten
ordinary launches, and stable under graph replay.

| M | Depth-2 us | Depth-3 us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 32.318 | 32.059 | 1.0081x | Yes |
| 2 | 32.342 | 32.084 | 1.0081x | Yes |
| 4 | 32.709 | 32.366 | 1.0106x | Yes |
| 8 | 32.978 | 32.739 | 1.0073x | Yes |
| 16 | 33.825 | 33.517 | 1.0092x | Yes |

The geometric-mean gate/up improvement is **1.00865x**.

## Matched Nsight Compute / SASS

Nsight Compute 2026.2.1 captured one retained W3 M1 grouped gate/up WGMMA
launch for the accepted Phase-13 depth-two source and one depth-three launch.
Reports and raw/source-correlated CSV exports are outside Git at:

```text
/root/qvq-profiler-artifacts/phase14-gateup-overlap/
```

The Phase-13 matched source is retained under:

```text
/root/qvq-profiler-artifacts/phase13-gateup-decode/
```

Both captures use the physical H100 UUID, strict zero-MiB idle admission, one
named launch, `--profile-from-start off`, and identical NCU section sets.

| Metric | Depth 2 | Depth 3 | Change |
|:--|--:|--:|--:|
| NCU replay duration | 29.984 us | 29.472 us | **1.0174x** |
| executed warp instructions | 10,687,611 | 10,677,982 | -9,629 (-0.09%) |
| eligible warps/cycle | 0.512 | 0.536 | **+4.86%** |
| active warps | 15.178% | 15.174% | unchanged |
| WGMMA stall / issue-active cycle | 0.421 | 0.377 | **-10.45%** |
| wait stall / issue-active cycle | 1.015 | 0.950 | **-6.39%** |
| long-scoreboard stall | 1.427 | 1.459 | +2.25% |
| registers/thread | 55 | 56 | +1 |
| static shared memory | 30.336 KiB | 30.336 KiB | unchanged |
| local/shared spills | 0 / 0 | 0 / 0 | unchanged |
| `LDS` | 2,228,224 | 2,228,224 | unchanged |
| `HGMMA` | 131,072 | 131,072 | unchanged |
| `WARPGROUP` | 253,952 | 245,760 | -8,192 |

The event-time win is therefore not an instruction-reduction claim. Decode
and tensor instructions are unchanged; the extra register fragment increases
eligible scheduling opportunities and reduces explicit wait/WGMMA stalls.
The one-register increase does not alter active-warp occupancy or spill.

## Complete Llama 3.2 1B MLP

The post-commit artifact executes production SHA `942868f3` with 30 warmups,
200 CUDA-event samples, and 50 CUDA Graph replays per sample. It includes the
complete grouped gate/up, recovery, exact SiLU/down precondition, down P32,
and down recovery path. `vs` is comparator latency divided by QVQ latency;
values below one mean the W4 comparator is faster. `Better` compares with the
committed Phase-13 artifact and records `No` for a regression.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 13 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 67.575 | 1.490 | 0.432x | 0.751x | 1.0015x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 67.518 | 2.982 | 0.464x | 0.749x | 1.0005x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.404 | 5.886 | 0.461x | 0.742x | 1.0012x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.044 | 11.664 | 0.427x | 0.737x | 0.9989x | No |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.343 | 25.032 | 0.507x | 0.791x | 1.0002x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.619 | 1.467 | 0.426x | 0.740x | 0.9993x | No |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.734 | 2.929 | 0.456x | 0.736x | 0.9994x | No |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.269 | 5.813 | 0.455x | 0.733x | 0.9991x | No |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 70.167 | 11.477 | 0.420x | 0.725x | 0.9972x | No |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.194 | 24.705 | 0.501x | 0.781x | 1.0008x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.118 | 1.478 | 0.429x | 0.745x | 1.0086x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.125 | 2.955 | 0.460x | 0.742x | 1.0080x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 68.647 | 5.866 | 0.459x | 0.740x | 1.0071x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.207 | 11.636 | 0.426x | 0.735x | 1.0090x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.609 | 24.929 | 0.505x | 0.788x | 1.0109x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 68.189 | 1.476 | 0.428x | 0.745x | 1.0029x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 68.538 | 2.937 | 0.457x | 0.738x | 1.0000x | No |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 69.088 | 5.828 | 0.456x | 0.735x | 1.0043x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 69.493 | 11.588 | 0.424x | 0.732x | 1.0056x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.995 | 24.781 | 0.502x | 0.783x | 1.0015x | Yes |

The targeted W3 cells improve **5/5** and have **1.0087x** geometric-mean
complete-MLP speedup versus Phase 13. Across all rates, including unchanged
rates and cross-run noise, geometric mean is **1.0028x**. Other all-cell
geometric means are **2.0749x** versus ordinary per-module QVQ, **0.4538x**
versus Marlin W4, and **0.7481x** versus Machete W4. The W4 comparisons are
figurative dense-equivalent efficiency baselines, not equal-work or
equal-quantization-quality claims.

## Correctness, VRAM, and validation

- The 59-test grouped Hopper suite passes across every supported rate and
  M=1/2/4/8/16, including dense-oracle bounds, child equality, ordered split
  reduction, repeatability, CUDA Graph replay, and real Llama projection
  shapes.
- The matched W3 A/B sweep is bit-exact at all five M values and stable over
  ten ordinary launches plus graph replay.
- The complete MLP matrix passes its existing accuracy/reference checks.
- Persistent VRAM, transient workspace, graph nodes, and payload bytes are
  unchanged. The only retained resource increase is one register/thread in
  the W3 WGMMA kernel.
- Builds use at most four Ninja jobs, one NVCC host thread, and one CUDA split
  compile partition.

Artifacts:

- `artifacts/a41_phase14_h100/production_mlp_w3_depth3_vs_phase13.json`
- `artifacts/a41_phase14_h100/w3_depth2_baseline_all_m.json`
- `artifacts/a41_phase14_h100/w3_depth3_candidate_all_m.json`

## Next experiment

Depth three improves overlap without reducing the conflict-heavy `LDS`/PGC
stream. Depth four is the narrow next scheduling probe, but it must beat depth
three without causing register/occupancy damage. If it is neutral or worse,
further pending-depth work should stop and the next phase should directly
reduce shared level-table bank conflicts or delete a larger PGC representation
cost.

Phase 15 rejected depth four and promoted an exact lane-interleaved W3 level
table that reduces measured shared-load conflicts by 43.8%. See
`docs/kernels/qvq_a41_r0_phase15_h100_w3_lane_levels.md`.
