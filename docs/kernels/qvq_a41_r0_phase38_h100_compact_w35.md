# Phase 38: rejected compact H100 W3.5 specialization

Phase 38 closes the fixed-prologue rate matrix by testing the compact Llama
gate/up launch for W3.5. The compiled kernel is smaller, but every H100 timing
cell regresses and geometric mean falls 2.48%. The candidate is fully removed.

## Exact experiment

Only the grouped launch plan changes for two K2048 x N8192, split-one children.
W3.5's seven-bit state extraction, selectors, level lookup, PGC math, WGMMA
order, paired recovery, and FP16 rounding are unchanged.

## Matched H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate. Timing
uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample for grouped inner P32 plus paired recovery.

| M/K/N per child | Generic W3.5 | Compact fixed W3.5 | vs generic | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 34.129 us | 35.118 us | 0.972x | 0.391x | 0.815x | No |
| 2/2048/8192 x2 | 34.215 us | 35.107 us | 0.975x | 0.400x | 0.804x | No |
| 4/2048/8192 x2 | 34.790 us | 35.634 us | 0.976x | 0.395x | 0.789x | No |
| 8/2048/8192 x2 | 35.155 us | 36.085 us | 0.974x | 0.377x | 0.782x | No |
| 16/2048/8192 x2 | 36.063 us | 36.843 us | 0.979x | 0.411x | 0.768x | No |

Geometric mean is **0.9752x**. No cell improves, so no complete-MLP run is
warranted.

## Compiled resource result

| Property | Generic W3.5 | Compact fixed W3.5 | Change |
|:--|--:|--:|--:|
| Static SASS instructions | 1,936 | 1,768 | -8.68% |
| Registers/thread | 55 | 48 | -12.7% |
| Stack | 112 B | 8 B | -92.9% |
| Constant parameter space | 1,552 B | 1,448 B | -6.70% |
| Shared memory | 33,408 B | 33,408 B | unchanged |

W3.5's decoder is dependency/scheduling bound rather than occupancy limited.
The prologue reduction perturbs scheduling without shortening the seven-bit
state/PGC critical path. Static size and register reductions alone do not
guarantee lower latency.

- candidate source is fully removed;
- production fixed specializations remain W2 and W3 only;
- no checkpoint, runtime, CUDA Graph, or VRAM state changed;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifacts:

- `artifacts/a41_phase38_h100/generic_w35_baseline.json`
- `artifacts/a41_phase38_h100/compact_w35_candidate.json`
- `artifacts/a41_phase38_h100/compact_w35_sass_summary.json`

## Next phase

Prologue specialization is now exhausted: W2/W3 promote, W2.5/W3.5 reject.
Phase 39 should return to W2.5/W3.5's dynamic decoder. The Phase-28 lane plan
removed some address reconstruction, but `TransitionBits >= 5` still builds
four separate window states and performs repeated PGC diffusion. The next
experiment should profile exact opcode/dependency composition for W2.5 and
W3.5 side by side, then target a shared state-to-level subexpression that
actually reduces executed instructions rather than merely resource pressure.
