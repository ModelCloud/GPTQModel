# Phase 43: H100 W2.5 depth-three shape audit

Phase 43 verifies that Phase 42's W2.5 three-fragment schedule is safe beyond
the grouped gate/up launch. Matched H100 tests show a 1.0121x inner QKV gain
and a 1.0024x split-16 down gain. No shape-specific rollback is needed.

## Matched inner-kernel audit

The depth-two and depth-three executables use identical payloads and inputs.
All outputs are repeatable, CUDA Graph stable, and within `2e-3` of the dense
P32 oracle. QKV retains child-local split counts `(8,8,8)` and ordered
reduction; down retains split 16 and its ordered reduction.

| M | QKV M/2048/(2048,512,512), depth 2 | QKV depth 3 | Better | Down M/8192/2048, depth 2 | Down depth 3 | Better |
|--:|--:|--:|:--:|--:|--:|:--:|
| 1 | 13.231 us | 13.116 us | Yes | 14.955 us | 14.919 us | Yes |
| 2 | 13.261 us | 13.060 us | Yes | 15.040 us | 14.997 us | Yes |
| 4 | 13.179 us | 12.980 us | Yes | 14.991 us | 14.996 us | No |
| 8 | 13.126 us | 12.973 us | Yes | 14.952 us | 14.888 us | Yes |
| 16 | 13.276 us | 13.152 us | Yes | 14.950 us | 14.913 us | Yes |

QKV improves **1.0121x** with five of five wins. Down improves **1.0024x**;
the single 0.03% M4 loss is timing noise and the complete W2.5 MLP still
improves all five cells in Phase 42.

## Complete canonical A41 QKV site

The complete site includes the shared input transform, grouped ordered P32,
Q/K output recovery, folded V output axis, child scales, and biases. Effective
throughput counts logical dense-equivalent FLOPs. Marlin and Machete are W4
figurative baselines. `Better` compares with the last committed canonical
Phase-7 QKV matrix, so it includes all accepted intervening transform,
level-layout, and depth changes rather than isolating Phase 42 alone.

| MKN: Q, K, V | QVQ W2.5 | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | Better than last benchmark |
|:--|--:|--:|--:|--:|:--:|
| 1/2048/(2048,512,512) | 26.536 us | 0.474 | 1.877x | 1.337x | Yes |
| 2/2048/(2048,512,512) | 27.009 us | 0.932 | 2.165x | 1.295x | Yes |
| 4/2048/(2048,512,512) | 27.230 us | 1.848 | 2.164x | 1.294x | Yes |
| 8/2048/(2048,512,512) | 27.497 us | 3.661 | 1.850x | 1.276x | Yes |
| 16/2048/(2048,512,512) | 27.881 us | 7.221 | 1.970x | 1.271x | Yes |

Geometric means are **2.0006x versus Marlin W4** and **1.2946x versus
Machete W4**.

## Decision

The W2.5 depth-three schedule remains rate-specific but does not need a
Llama-shape restriction:

- gate/up complete MLP: 1.0183x, five of five wins;
- grouped split-8 QKV inner: 1.0121x, five of five wins;
- split-16 down inner: 1.0024x geometric mean;
- exactness and child-local reduction ordering unchanged;
- no added shared memory, checkpoint storage, persistent VRAM, or workspace.

Artifacts:

- `artifacts/a41_phase43_h100/production_w25_qkv_depth3_vs_baselines.json`
- `artifacts/a41_phase43_h100/w25_depth3_shape_audit.json`

## Next phase

Phase 44 should return to the complete MLP critical path. W2 and W2.5 are now
at or near Machete W4, while W3/W3.5 remain several percent behind. Profile
the current W3.5 gate/up decoder after the lane-table change and test whether
its depth-two dependency chain benefits from a different decode/WGMMA issue
order rather than another fragment, which Phase 42 already rejected.
