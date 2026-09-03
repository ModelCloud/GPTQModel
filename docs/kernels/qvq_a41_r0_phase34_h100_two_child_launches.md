# Phase 34: rejected two-child W2 launches

Phase 34 tested two independent fixed W2 gate/up launches in place of Phase
32's single two-child launch. Each child kernel is slightly smaller, but the
complete paired operation regresses by 26.8%. The candidate is fully removed.

## Design and arithmetic

Production launches one grid:

```text
grid = [128 N64 blocks, 2 children, 1 split]
```

The candidate launches twice:

```text
gate grid = [128, 1, 1], global N64 base = 0
up grid   = [128, 1, 1], global N64 base = 128
```

Each launch receives one child-local alternative-bank ID and an already-offset
output pointer. Trellis/selector storage, state decoding, level lookup, WGMMA
order, recovery, and FP16 rounding remain exact. Timing includes both child
launches and paired recovery under one warmed CUDA Graph replay.

## H100 result

The physical 132-SM H100 passed the strict 0% utilization / 0 MiB gate. Results
use 30 warmups, 200 CUDA-event samples, and 50 graph replays per sample.

| M/K/N per child | Phase 32 | Candidate | vs Phase 32 | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 33.021 us | 45.289 us | 0.729x | 0.303x | 0.645x | No |
| 2/2048/8192 x2 | 33.070 us | 45.249 us | 0.731x | 0.307x | 0.631x | No |
| 4/2048/8192 x2 | 33.535 us | 45.926 us | 0.730x | 0.304x | 0.612x | No |
| 8/2048/8192 x2 | 34.031 us | 46.428 us | 0.733x | 0.290x | 0.606x | No |
| 16/2048/8192 x2 | 34.780 us | 47.124 us | 0.738x | 0.318x | 0.599x | No |

Geometric mean is **0.7322x**. No cell improves.

## SASS explanation

`cuobjdump` reports 1,728 static SASS instructions and zero stack bytes for
one candidate child kernel, compared with 1,736 instructions and eight stack
bytes for the Phase-32 two-child kernel. Registers and shared memory are
unchanged at 48 and 27,264 bytes.

Those eight instructions cannot compensate for paying the complete producer/
consumer pipeline, level initialization, synchronization, and kernel launch
twice. The experiment confirms that the two children must remain one launch.

## Decision and next phase

- candidate source is fully removed;
- production remains Phase 32;
- no checkpoint, runtime, graph, or VRAM change remains;
- no complete-MLP run is warranted after the isolated 26.8% regression.

Artifacts:

- `artifacts/a41_phase34_h100/two_child_launches_candidate.json`
- `artifacts/a41_phase34_h100/two_child_launches_sass_summary.json`

Phase 35 should keep one launch and one shared pipeline. A more promising
remaining control experiment is to pack the two 2-bit alternative-bank IDs
into one scalar and extract the child value with a shift instead of passing an
array. This preserves the Phase-32 launch geometry while testing whether the
last small stack object and indexed local load can disappear. Because the
expected gain is small, static/executed SASS must improve before a full MLP
promotion run.
