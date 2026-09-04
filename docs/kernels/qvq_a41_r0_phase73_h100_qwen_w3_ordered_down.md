# Phase 73: H100 Qwen W3 ordered down recovery

Phase 73 changes only Qwen3.8-27B W3 down execution.  It replaces unordered
split-34 atomics with deterministic split-17 partial planes and reduces those
planes directly inside Phase 72's native composite recovery.  Every target M
improves.  W2 and W2.5 retain Phase 72 because ordered-partial probes regressed.

## Ordered arithmetic

For W3 down, `K=17408` contains 1088 K16 tiles.  Split 17 assigns 64 K16 tiles
to every partition and launches

$$
\frac{5120}{64}\times17=1360
$$

useful decode blocks.  Each block writes a disjoint FP32 partial plane.  The
composite recovery begins with the exact prior reducer order:

$$
v=(((0+p_0)+p_1)+\cdots)+p_{16},
$$

using explicit round-to-nearest FP32 additions, then applies the established
overflow-safe `H40 x H128`, `SV`, optional bias, and FP16 store sequence.
There is no reduced FP32 output tensor and no separate reduction launch.

The low-level API accepts both split 17 and split 34 for experimentation and
tests.  Production promotion is W3, split 17, physical NVIDIA H100, and exact
Qwen down geometry only.  Runtime telemetry reports
`h100_qwen_ordered_composite_down_recovery_launches`.

## Correctness and graph safety

- Split 17 and split 34 fused consumers are bit-exact to their corresponding
  left-to-right ordered reduction followed by Phase 72 recovery.
- Tests cover M1/M16, bias/no-bias, and CUDA Graph replay.
- The complete W3 Qwen MLP replays exactly and stays within `2e-3` of its
  independent child path.
- Dense-P32-oracle mean and maximum errors are unchanged.
- The canonical checkpoint and persistent VRAM are unchanged.  W3 uses a
  transient split-partial workspace rather than atomic accumulation output.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays/sample.
`last/new` compares with Phase 72.  Marlin and Machete are figurative W4
projection-sum baselines; ratios below one mean the W4 baseline is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 201.859 | 0.284x | 0.557x | 0.9997x | no |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 203.629 | 0.287x | 0.544x | 0.9998x | no |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 204.718 | 0.286x | 0.542x | 1.0001x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 209.026 | 0.276x | 0.530x | 0.9979x | no |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 210.978 | 0.319x | 0.526x | 0.9986x | no |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 200.917 | 0.286x | 0.560x | 0.9964x | no |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 201.476 | 0.291x | 0.550x | 1.0015x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 202.802 | 0.289x | 0.547x | 1.0006x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 207.986 | 0.278x | 0.533x | 0.9966x | no |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 211.003 | 0.319x | 0.526x | 0.9963x | no |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 202.320 | 0.284x | 0.556x | 1.0114x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 204.792 | 0.286x | 0.541x | 1.0093x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 205.515 | 0.285x | 0.539x | 1.0101x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 210.000 | 0.275x | 0.528x | 1.0082x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 212.496 | 0.317x | 0.523x | 1.0104x | yes |

The targeted W3 geometric speedup is **1.0099x**, with five of five wins.
Across all rates the comparison is **1.0024x**; W2/W2.5 source is unchanged,
so their small movements are run-to-run variation.  Raw distilled result:
`artifacts/a41_phase73_h100/qwen38_27b_w3_ordered_composite_down.json`.

## Next target

The complete MLP is still roughly twice the W4 Machete projection-sum baseline,
and the grouped gate/up P32 decode is the largest remaining component.  The
next pass should profile Phase 73 at W2/W2.5/W3 and focus on decoder shared-load
conflicts or producer reuse rather than further composite-recovery work.
