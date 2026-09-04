# Phase 70: H100 Qwen3.8 ordered MLP bridge

Phase 70 fuses the two deterministic gate/up split-K reducers into the
Phase-69 folded Qwen MLP bridge.  It is a production promotion with 15/15
W2--W3 decode wins and exact output equivalence.

## Ordered-partial contract

The grouped Hopper launch can now return its transient child-major partial
planes without allocating or writing the two reduced FP32 child outputs:

```text
[gate split 0..9, each 16x17408]
[up   split 0..9, each 16x17408]
```

The consumer reproduces the prior reducer's left-to-right FP32 order for each
child and output value:

$$
g = (((0 + g_0) + g_1) + \cdots) + g_9,
\qquad
u = (((0 + u_0) + u_1) + \cdots) + u_9.
$$

Explicit round-to-nearest FP32 adds prevent contraction or reassociation.
The reduced values then enter the exact Phase-69 recovery, FP16 rounding,
SiLU, product, down scaling, and M16-padding sequence.  This removes two
reducer launches plus the reduced gate/up FP32 materialization and reload.

The generic grouped-partial API retains each segment's own split count and
physical offset.  Production fusion is deliberately narrower: physical H100,
FP16, Qwen geometry `5120 -> 17408 x2 -> 5120`, and equal split count ten.
All other schedules keep Phase 69 or the original exact path.

## Correctness and graph safety

- Grouped ordered-partial tests cover W2/W2.5/W3/W3.5 and M=1/2/4/8/16 and
  require exact reconstruction of every prior child output.
- The fused split-10 consumer is bit-exact to Phase 69 for M1 and M16, with
  and without independent biases.
- The complete Qwen MLP captures and replays exactly in a CUDA Graph.
- `h100_folded_qwen_fused_ordered_reduction_launches` confirms production use.
- Mean absolute error remains `1.503e-8`--`1.544e-8`; maximum is `4.838e-8`
  against the same-payload dense-P32 Torch oracle.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed by CUDA events, 20 warmups, 60 samples, and 50 replays/sample.
`last/new` compares against Phase 69.  Marlin and Machete are existing W4
projection-sum baselines; ratios below one mean the baseline is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 222.342 | 0.258x | 0.506x | 1.0592x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 225.976 | 0.259x | 0.490x | 1.0565x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 229.384 | 0.256x | 0.483x | 1.0540x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 240.111 | 0.241x | 0.462x | 1.0434x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 252.621 | 0.267x | 0.440x | 1.0258x | yes |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 221.050 | 0.260x | 0.509x | 1.0651x | yes |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 224.951 | 0.260x | 0.493x | 1.0604x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 228.552 | 0.257x | 0.485x | 1.0543x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 238.210 | 0.242x | 0.465x | 1.0465x | yes |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 250.591 | 0.269x | 0.443x | 1.0296x | yes |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 229.145 | 0.250x | 0.491x | 1.0579x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 232.660 | 0.252x | 0.476x | 1.0548x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 236.206 | 0.248x | 0.469x | 1.0510x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 247.084 | 0.234x | 0.449x | 1.0421x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 259.465 | 0.260x | 0.428x | 1.0242x | yes |

Geometric speedup is **1.0483x over Phase 69** and **1.0991x over merged PR
#98**.  Every one of the 15 cells improves.  Raw result:
`artifacts/a41_phase70_h100/qwen38_27b_folded_mlp_ordered.json`.
