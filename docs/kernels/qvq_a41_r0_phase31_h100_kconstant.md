# Phase 31: rejected H100 fixed-K CUTE extent

Phase 31 tested whether Phase 30's fixed W2 Llama gate/up specialization should
also replace every device-side runtime K extent with the compile-time value
2048. The change was exact, but the matched H100 speedup was only 0.13%
geometric mean, one of five cells regressed, and the compiled kernel grew by
2.27%. The candidate is fully removed; production remains Phase 30.

## Exact candidate

Phase 30 already guarantees the following before dispatch:

```text
transition bits = 4
children        = 2
child shape     = K2048 x N8192
split count     = 1 per child
```

The candidate retained the shared kernel ABI but selected its K extent by the
template flag:

```cpp
kernel_size_k = FixedGateUp ? 2048 : size_k;
k_tiles = kernel_size_k / 16;
full_input_shape = [16, kernel_size_k];
```

No payload, TMA tile, decode operation, selector, bank choice, level lookup,
WGMMA order, output recovery, or FP16 boundary changed.

## Matched H100 benchmark

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB admission
gate. Both builds use 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA
Graph replays per sample. Timing covers grouped inner P32 plus exact paired
gate/up recovery. Marlin and Machete are W4 baselines.

| M/K/N per child | Phase 30 | Candidate | vs Phase 30 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 33.773 us | 33.663 us | 1.003x | 0.410x | 0.857x | Yes |
| 2/2048/8192 x2 | 33.791 us | 33.736 us | 1.002x | 0.411x | 0.839x | Yes |
| 4/2048/8192 x2 | 34.242 us | 34.188 us | 1.002x | 0.409x | 0.825x | Yes |
| 8/2048/8192 x2 | 34.670 us | 34.649 us | 1.001x | 0.389x | 0.816x | Yes |
| 16/2048/8192 x2 | 35.505 us | 35.516 us | 1.000x | 0.422x | 0.797x | No |

The geometric mean is **1.00134x**. A preliminary complete-MLP run was also
mixed: only M8 and M16 improved against the preceding committed matrix, while
M1/M2/M4 regressed. The candidate therefore fails the all-cell promotion gate.

## Compiled SASS evidence

This experiment did not rely on source inspection alone. CUDA `cuobjdump`
was run on the cached Phase-30 and candidate extension binaries, selecting the
exact `TransitionBits=4, Grouped=true, OrderedSplit=false,
FixedGateUp=true` kernel instantiation.

| Static kernel property | Phase 30 | Candidate | Change |
|:--|--:|--:|--:|
| SASS instructions | 1,760 | 1,800 | **+2.27%** |
| Registers/thread | 48 | 48 | unchanged |
| Stack | 112 B | 112 B | unchanged |
| Shared memory | 27,264 B | 27,264 B | unchanged |
| Local memory | 0 B | 0 B | unchanged |

The TMA atom type already encodes the fixed 16x2048 input descriptor created
by the host. Reintroducing that extent through CUTE's device tensor shape does
not delete the decoder loop; it perturbs template/index lowering and expands
the static program. The tiny timing movement is not a sound production win.

## Decision

- candidate source is fully removed;
- production executable behavior is exactly Phase 30;
- no test, checkpoint, runtime, CUDA Graph, or VRAM state changes remain;
- compilation used Ninja `-j4`, one NVCC host thread, and one split-compile
  partition.

Artifacts:

- `artifacts/a41_phase31_h100/kconstant_candidate_isolated.json`
- `artifacts/a41_phase31_h100/phase30_matched_baseline.json`
- `artifacts/a41_phase31_h100/kconstant_candidate_mlp.json`
- `artifacts/a41_phase31_h100/kconstant_static_sass_summary.json`

## Next phase

Further generic CUTE-extent specialization is unlikely to pay. Phase 32 should
instead test a representation-level change with a larger instruction target:
make the W2 fixed gate/up kernel take the two child alternative-bank identifiers
as compact scalar launch arguments rather than copying the full grouped launch
descriptor by value. This preserves child-local bank semantics while testing
whether the remaining 112-byte stack frame and uniform-register prologue can
be reduced. The experiment must remain W2-only and be rejected unless every M
cell improves with smaller compiled or executed instruction cost.
