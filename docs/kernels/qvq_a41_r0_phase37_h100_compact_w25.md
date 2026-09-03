# Phase 37: rejected compact H100 W2.5 specialization

Phase 37 independently tested Phase 32's compact fixed Llama gate/up launch for
W2.5. The kernel becomes substantially smaller, but the matched H100 timing is
mixed and regresses by 0.18% geometric mean. The W2.5 gate is fully removed;
production remains fixed only for W2 and W3.

## Exact experiment

The candidate changes only grouped launch geometry/metadata for two K2048 x
N8192, split-one children. W2.5's five-bit state extraction, Phase-28 lane
plan, selector stream, level lookups, PGC math, WGMMA order, paired recovery,
and FP16 boundaries do not change.

## Matched H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate. Timing
uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample for grouped inner P32 plus paired recovery.

| M/K/N per child | Generic W2.5 | Compact fixed W2.5 | vs generic | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 34.735 us | 35.041 us | 0.991x | 0.394x | 0.842x | No |
| 2/2048/8192 x2 | 34.804 us | 34.761 us | 1.001x | 0.403x | 0.841x | Yes |
| 4/2048/8192 x2 | 35.203 us | 35.267 us | 0.998x | 0.398x | 0.825x | No |
| 8/2048/8192 x2 | 35.702 us | 35.690 us | 1.000x | 0.380x | 0.815x | Yes |
| 16/2048/8192 x2 | 36.523 us | 36.523 us | 1.000x | 0.413x | 0.798x | No |

Geometric mean is **0.9982x** and only two of five cells improve. The isolated
gate fails, so no complete-MLP run is warranted.

## Compiled resource result

| Property | Generic W2.5 | Compact fixed W2.5 | Change |
|:--|--:|--:|--:|
| Static SASS instructions | 1,936 | 1,768 | -8.68% |
| Registers/thread | 55 | 48 | -12.7% |
| Stack | 112 B | 8 B | -92.9% |
| Constant parameter space | 1,552 B | 1,448 B | -6.70% |
| Shared memory | 29,312 B | 29,312 B | unchanged |

W2.5 remains dominated by the five-bit decoder's dependency chain. Removing
the prologue footprint does not shorten that critical path and slightly changes
scheduling. Compiled size alone is not a promotion criterion.

- candidate source is fully removed;
- no checkpoint, runtime, CUDA Graph, or VRAM state changed;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifacts:

- `artifacts/a41_phase37_h100/generic_w25_baseline.json`
- `artifacts/a41_phase37_h100/compact_w25_candidate.json`
- `artifacts/a41_phase37_h100/compact_w25_sass_summary.json`

## Next phase

Phase 38 should close the rate matrix by testing W3.5 under the same compact
fixed geometry. It must be isolated and matched because the earlier
full-descriptor experiment regressed. After that result, work should return to
rate-specific dynamic decode dependencies rather than further shared-prologue
specialization.
