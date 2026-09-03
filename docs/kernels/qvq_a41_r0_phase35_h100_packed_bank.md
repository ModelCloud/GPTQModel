# Phase 35: rejected packed W2 bank identifiers

Phase 35 packed the fixed W2 gate/up children's two alternative-bank IDs into
one 32-bit launch scalar. It removes the last eight-byte stack object, but does
not reduce static SASS and regresses every isolated H100 cell. The candidate is
fully removed; production remains Phase 32.

## Exact experiment

Production passes two child-local integer IDs. The candidate stores:

```text
packed = bank_gate | (bank_up << 2)
bank(segment) = (packed >> (2 * segment)) & 3
```

Alternative-bank IDs are guaranteed to be in `[0,3]`, so this mapping is
lossless. No selector, state, level, payload, WGMMA, recovery, or rounding
operation changes.

## H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate. Timing
uses 30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample for grouped P32 plus paired recovery.

| M/K/N per child | Phase 32 | Candidate | vs Phase 32 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 33.021 us | 33.052 us | 0.999x | 0.412x | 0.873x | No |
| 2/2048/8192 x2 | 33.070 us | 33.123 us | 0.998x | 0.422x | 0.853x | No |
| 4/2048/8192 x2 | 33.535 us | 33.703 us | 0.995x | 0.418x | 0.834x | No |
| 8/2048/8192 x2 | 34.031 us | 34.105 us | 0.998x | 0.396x | 0.826x | No |
| 16/2048/8192 x2 | 34.780 us | 34.900 us | 0.997x | 0.433x | 0.809x | No |

Geometric mean is **0.9974x**. No cell improves.

## SASS result and decision

| Static property | Phase 32 | Candidate | Change |
|:--|--:|--:|--:|
| SASS instructions | 1,736 | 1,736 | unchanged |
| Registers/thread | 48 | 48 | unchanged |
| Stack | 8 B | 0 B | -8 B |
| Constant parameter space | 1,448 B | 1,444 B | -4 B |
| Shared memory | 27,264 B | 27,264 B | unchanged |

The indexed parameter read is replaced by a variable shift/mask. Eliminating
the stack reservation does not eliminate total instructions or dependencies,
so the timing result is consistent with the compiled program.

- candidate source is fully removed;
- no complete-MLP run is warranted after a 0/5 isolated result;
- no checkpoint, runtime, CUDA Graph, or VRAM state changed.

Artifacts:

- `artifacts/a41_phase35_h100/packed_bank_candidate.json`
- `artifacts/a41_phase35_h100/packed_bank_sass_summary.json`

## Next phase

Phase 32's compact descriptor was enabled only for W2 because the original
fixed-prologue experiment did not promote W3. The compact-parameter result
changes that cost model. Phase 36 should retest the exact fixed Llama gate/up
geometry for W3 with the compact descriptor: Phase 30 showed an isolated W3
win, and removing the full grouped descriptor may now turn the formerly mixed
complete-MLP result into an all-cell win. W2 remains unchanged during that
experiment.
