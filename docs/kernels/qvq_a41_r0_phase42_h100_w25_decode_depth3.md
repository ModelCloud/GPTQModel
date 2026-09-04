# Phase 42: H100 W2.5 three-fragment decode pipeline

Phase 42 promotes a third in-flight register-sourced fragment for W2.5. It
overlaps the accepted lane-pair level table's unavoidable second shared-load
wavefront with matrix work and improves all five focused and complete-MLP
cells. W2 and W3.5 retain depth two; W3 already used depth three.

## Schedule

For K-block index `j`, the decoder selects fragment

\[
F_{j\bmod d},
\]

where `d` is the compile-time decode depth. Before reusing a fragment, the
consumer limits outstanding register-sourced WGMMA groups with

\[
\operatorname{warpgroup\_wait}\langle d-1\rangle.
\]

Production previously used `d=2` for W2.5. The promoted specialization uses
`d=3`, so one more independent fragment can decode while earlier fragments
are consumed by WGMMA. State extraction, bank masks, PGC arithmetic, level
values, WGMMA accumulation order, and FP16 output boundaries are unchanged.

The initial broad experiment tested depth three at every rate. W2 regressed
to a 0.9941x geometric mean and W3.5 to 0.9928x. W3 was already depth three.
Only W2.5 passed at 1.0256x, so the production condition is rate-specific.

## Matched isolated H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate.
Timing uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample for grouped gate/up inner P32 plus paired recovery.

| M/K/N per child | Phase 40 depth 2 | W2.5 depth 3 | vs Phase 40 | Better than last benchmark |
|:--|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 30.811 us | 30.067 us | 1.0247x | Yes |
| 2/2048/8192 x2 | 30.678 us | 29.683 us | 1.0335x | Yes |
| 4/2048/8192 x2 | 30.914 us | 30.068 us | 1.0281x | Yes |
| 8/2048/8192 x2 | 31.468 us | 30.432 us | 1.0340x | Yes |
| 16/2048/8192 x2 | 32.105 us | 31.243 us | 1.0276x | Yes |

The isolated geometric mean is **1.0296x**.

## Matched profiler result

| Metric | Depth 2 | Depth 3 | Change |
|:--|--:|--:|--:|
| Replay duration | 28.640 us | 27.776 us | 1.0311x |
| Eligible warps/cycle | 0.58 | 0.61 | +5.2% |
| Executed instructions | 10,800,426 | 10,793,252 | -0.07% |
| Shared-load instructions | 2,228,224 | 2,228,224 | unchanged |
| Shared-load bank conflicts | 1,051,853 | 1,051,592 | unchanged |
| Shared-load wavefronts | 3,337,238 | 3,345,602 | +0.25% |
| Long-scoreboard stalls | 1.36 | 1.36 | unchanged |
| Registers/thread | 53 | 59 | +6 |
| Shared memory/block | 44,800 B | 44,800 B | unchanged |

The win comes from latency overlap, not fewer loads or conflicts. The grid has
only two useful waves on the 132-SM H100, so the six-register increase does not
reduce useful residency for this launch.

## Complete Llama 3.2 1B MLP

Effective throughput counts logical dense-equivalent FLOPs. Marlin and
Machete are figurative W4 baselines, not equal-rate work comparisons.
`Better` compares with Phase 40.

| MKN: gate/up x2; down | QVQ W2.5 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2; 1/8192/2048 | 50.826 us | 1.981 | 0.573x | 0.988x | Yes |
| 2/2048/8192 x2; 2/8192/2048 | 50.964 us | 3.950 | 0.613x | 0.981x | Yes |
| 4/2048/8192 x2; 4/8192/2048 | 51.727 us | 7.784 | 0.608x | 0.980x | Yes |
| 8/2048/8192 x2; 8/8192/2048 | 52.308 us | 15.395 | 0.561x | 0.971x | Yes |
| 16/2048/8192 x2; 16/8192/2048 | 53.312 us | 30.211 | 0.610x | 0.955x | Yes |

W2.5 improves **1.0183x** across the complete MLP. Its Machete-relative
geometric mean is **0.9749x**. The all-rate matrix improves **1.0051x**; only
W2.5 is attributed to this source change.

## Correctness and scope

- All five focused outputs are exact and CUDA Graph stable.
- The complete MLP remains bit-exact to its unfused reference.
- 126 Hopper P32/grouped tests pass across every rate, ordered reductions,
  exact child parity, dense-oracle bounds, and graph replay.
- No checkpoint bytes, shared memory, persistent VRAM, external workspace,
  split policy, or numerical operation changed.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase42_h100/production_mlp_w25_depth3_vs_phase40.json`
- `artifacts/a41_phase42_h100/w25_decode_depth3_profile.json`
- binary reports outside Git under
  `/root/qvq-profiler-artifacts/phase42-w25-depth3/`

## Next phase

The common level layout and rate-specific decode depths are now measured.
Phase 43 should examine whether the W2.5 depth-three win generalizes only to
the Llama gate/up two-wave launch or also to QKV/down split schedules. Any
extension must preserve child-local reduction order and be selected by shape,
not enabled globally from this one result.
