# Phase 44: H100 W3.5 deferred fragment-reuse wait

Phase 44 moves W3.5's depth-two register-fragment reuse wait to the last
correct point. State/window extraction, bank selection, and PGC mapping now
overlap the prior register-sourced WGMMA before the wait protects the first
write into the reused fragment. The full Llama 3.2 1B MLP improves all five
W3.5 cells by 1.0250x--1.0433x.

## Dependency proof and schedule

For K-block `j`, W3.5 selects one of two register fragments:

\[
F_j = F_{j\bmod 2}.
\]

The original loop waited before doing any decode work once a fragment could
be reused:

```text
wait_group<1>
state/window extraction
bank-mask selection
four PGC products
level loads -> overwrite F[j mod 2]
WGMMA(F[j mod 2])
```

Only the level loads write the selected fragment. Window extraction and the
four PGC products use temporary scalar registers and do not read or write
`F[j mod 2]`. The safe dependency boundary is therefore:

```text
state/window extraction       # independent of F[j mod 2]
bank-mask selection           # independent of F[j mod 2]
four PGC products             # independent of F[j mod 2]
wait_group<1>                 # old WGMMA has released F[j mod 2]
level loads -> overwrite F[j mod 2]
WGMMA(F[j mod 2])
```

The WGMMA issue order, accumulator order, decoded values, shared-memory
addresses, and FP16/FP32 boundaries are unchanged. This is a scheduling
transformation, not a mathematical transformation.

The production condition is deliberately W3.5-only (`TransitionBits == 7`).
Other rates keep their previously measured wait placement until independently
benchmarked.

## Complete Llama 3.2 1B MLP

Timing uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample on the physical 132-SM H100. The idle gate required three
samples at 0% utilization and 0 MiB allocated memory. Effective throughput
counts logical dense-equivalent FLOPs. Marlin and Machete are figurative W4
baselines; a ratio above one means QVQ is faster. `Better` compares with the
committed Phase-42 matrix.

| MKN: gate/up x2; down | QVQ W3.5 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2; 1/8192/2048 | 50.135 us | 2.008 | 0.579x | 0.985x | Yes |
| 2/2048/8192 x2; 2/8192/2048 | 50.109 us | 4.018 | 0.623x | 0.984x | Yes |
| 4/2048/8192 x2; 4/8192/2048 | 50.813 us | 7.924 | 0.618x | 0.982x | Yes |
| 8/2048/8192 x2; 8/8192/2048 | 51.364 us | 15.678 | 0.571x | 0.974x | Yes |
| 16/2048/8192 x2; 16/8192/2048 | 52.201 us | 30.854 | 0.622x | 0.960x | Yes |

W3.5 improves **1.0344x** geometrically versus Phase 42, with five of five
wins. Its geometric ratios are **0.6023x versus Marlin W4** and **0.9768x
versus Machete W4**. Across all four rates, including unchanged-rate run
controls, the new matrix improves **1.0095x**.

## Matched Nsight Compute evidence

The baseline is the accepted depth-two W3.5 lane-pair kernel. Both profiles
use the same M1 grouped gate/up geometry, payload layout, launch grid, and
physical H100.

| Metric | Before | Deferred wait | Change |
|:--|--:|--:|--:|
| Kernel duration | 27.424 us | 26.976 us | -1.63% |
| Executed instructions | 10,800,468 | 10,800,390 | unchanged |
| Shared-load instructions | 2,228,224 | 2,228,224 | unchanged |
| Shared-load bank conflicts | 1,050,858 | 1,051,001 | +0.01% |
| Shared-load wavefronts | 3,363,274 | 3,386,057 | +0.68% |
| Eligible warps/cycle | 0.593 | 0.608 | +2.61% |
| Long-scoreboard stalls | 1.374 | 1.324 | -3.63% |
| Registers/thread | 56 | 57 | +1 |
| Shared memory/block | 48,896 B | 48,896 B | unchanged |

The instruction and memory-traffic counts demonstrate that Phase 44 does not
remove decoder work. More eligible warps and fewer long-scoreboard stalls
show that independent decoder arithmetic overlaps the outstanding WGMMA more
effectively. The one-register increase does not change useful residency for
this two-wave launch.

## Correctness and scope

- 126 Hopper P32 and grouped-P32 tests pass across all four rates.
- Grouped child parity, ordered split reductions, dense-oracle error bounds,
  repeatability, and CUDA Graph replay remain covered.
- No checkpoint layout, persistent VRAM, shared-memory allocation, split
  policy, reduction order, decoded value, or external workspace changed.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase44_h100/production_mlp_w35_deferred_wait_vs_phase42.json`
- `artifacts/a41_phase44_h100/w35_deferred_fragment_wait_profile.json`
- binary report outside Git under
  `/root/qvq-profiler-artifacts/phase44-w35-deferred-wait/`

## Next phase

Phase 45 should test the same last-safe-point wait placement for W2, which
also uses a depth-two fragment schedule. It must remain rate-gated unless its
own isolated and complete-MLP matrices improve and it preserves exact output.
