# Phase 33: rejected compact runtime scalar ABI

Phase 33 tested whether the Phase-32 fixed W2 gate/up kernel should replace
four scalar runtime arguments with a four-byte K-only parameter. This removes
the unused total-N, split-count, and fallback-bank values without turning K
into a compile-time CUTE shape. It is exact, but it does not remove any SASS
and regresses four of five isolated H100 cells. The candidate is fully removed.

## Candidate ABI

Phase 32 passes:

```text
compact child-bank structure: 8 bytes
runtime K:                   4 bytes
runtime total N:             4 bytes
runtime split count:         4 bytes
runtime fallback bank:       4 bytes
```

The candidate packages only runtime K into a four-byte type for the fixed
instantiation. Generic and ordered-split kernels use their original full
runtime values. No tensor shape, payload, decode math, WGMMA order, recovery,
or rounding boundary changes.

## Matched H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate. Timing
uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample, and covers grouped inner P32 plus exact paired recovery.

| M/K/N per child | Phase 32 | Candidate | vs Phase 32 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 33.021 us | 33.301 us | 0.992x | 0.410x | 0.850x | No |
| 2/2048/8192 x2 | 33.070 us | 33.080 us | 1.000x | 0.420x | 0.840x | No |
| 4/2048/8192 x2 | 33.535 us | 33.615 us | 0.998x | 0.415x | 0.824x | No |
| 8/2048/8192 x2 | 34.031 us | 34.004 us | 1.001x | 0.395x | 0.817x | Yes |
| 16/2048/8192 x2 | 34.780 us | 34.851 us | 0.998x | 0.430x | 0.799x | No |

Geometric mean is **0.9975x**, a 0.25% regression. The isolated gate fails,
so a complete-MLP run is neither necessary nor a valid promotion path.

## SASS and resource evidence

CUDA `cuobjdump` selected the exact W2 fixed kernel from both cached extension
binaries.

| Static property | Phase 32 | Candidate | Change |
|:--|--:|--:|--:|
| SASS instructions | 1,736 | 1,736 | unchanged |
| Registers/thread | 48 | 48 | unchanged |
| Stack | 8 B | 8 B | unchanged |
| Constant parameter space | 1,448 B | 1,436 B | -12 B |
| Shared memory | 27,264 B | 27,264 B | unchanged |

The three removed arguments were already dead after parameter loading. A
smaller constant-parameter reservation alone provides no device instruction
or scheduling benefit.

## Decision and next phase

- candidate source is fully removed;
- production remains Phase 32 at `e43f3405`;
- no checkpoint, runtime, CUDA Graph, or VRAM change remains;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifact:

- `artifacts/a41_phase33_h100/compact_runtime_candidate_isolated.json`
- `artifacts/a41_phase33_h100/compact_runtime_static_sass_summary.json`

Phase 34 should stop shrinking already-dead ABI fields. The next worthwhile
target must delete dynamic decoder work. The W2 fixed kernel still performs a
runtime child-bank selection for every block. A safe experiment is two
separate fixed launches, one per child, each receiving one scalar bank ID and
one N8192 output base. This trades one additional kernel launch for compile-time
segment geometry and may remove segment-dependent address/select work. It must
be timed as the complete paired operation because launch overhead is part of
the tradeoff.
