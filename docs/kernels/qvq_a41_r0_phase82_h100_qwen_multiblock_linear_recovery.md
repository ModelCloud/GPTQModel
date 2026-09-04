# Phase 82: multiblock H100 Qwen linear recovery

Phase 82 replaces the one-block-per-row Phase-81 recovery for Qwen3.8-27B's
10240-wide and 6144-wide linear outputs with two exact, highly parallel CUDA
grids. The change improves all 15 W2--W3 by M1--M16 cells and lowers geometric
linear-input latency by 25.1 percent.

## Why one block was still slow

The Phase-81 W3/M1 NVIDIA Nsight Systems timeline measured:

| Component | Time |
|:--|--:|
| grouped P32 decode and multiply | 55.231 us |
| 10240-wide recovery | 30.272 us |
| 6144-wide recovery | 13.824 us |
| reducers plus shared input preparation | about 18 us |

Each recovery launched one cooperative thread array per logical row. At M1,
the two transforms therefore occupied only two of the H100's 132 streaming
multiprocessors despite containing independent factor slices and output rows.

## Exact factorized grids

For a composite width $N=B L$, Phase 82 uses:

```text
low grid:  (B base slices, M rows)
high grid: (B output base rows, M rows)
```

The low grid owns one independent length-$L$ slice. It applies the exact
normalization and ascending butterflies, including the established
overflow-preserving FP16 rounding function $R$, then stores one FP32
workspace value per element.

The high grid owns one output base row. For each local column it evaluates

$$
y_{b,l}=R\left(\sum_{s=0}^{B-1} H_B[b,s]x_{s,l}\right)
$$

with the same source-ascending FP32 fused multiply-add order as Phase 81. It
then performs the same FP16-rounded output-scale multiply, optional bias add,
and final FP16 store.

The concrete grids are:

| Output | Factorization | Low blocks/row | High blocks/row |
|--:|:--|--:|--:|
| 10240 | H40 x H256 | 40 | 40 |
| 6144 | H12 x H512 | 12 | 12 |

At M1 this exposes 104 useful blocks instead of two. The temporary workspace
is `M*N*sizeof(float)` for each child and is allocated from the CUDA Graph
private pool during capture; checkpoint and persistent virtual-memory usage
do not change.

## Correctness and graph safety

- The two-stage result is bit-exact to the Torch composite oracle at M1 and
  M16, with and without bias, for both new widths.
- The grouped linear-input site is bit-exact across five CUDA Graph replays
  for W2, W2.5, and W3.
- Maximum error against the dense-P32 FP32 Torch oracle is `1.774e-5`;
  maximum mean absolute error is `2.479e-6`.
- Runtime selection is visible through
  `h100_qwen_linear_multiblock_recovery_launches`.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays per
sample. `last/new` compares with committed Phase 81. Marlin and Machete are
figurative W4 projection-sum baselines; ratios below one mean the W4 baseline
is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x10240 + 1x5120x6144 | 90.741 | 0.2568x | 0.5489x | 1.3503x | yes |
| 2 | 2 | 2x5120x10240 + 2x5120x6144 | 92.457 | 0.2609x | 0.5323x | 1.3427x | yes |
| 2 | 4 | 4x5120x10240 + 4x5120x6144 | 92.927 | 0.2616x | 0.5286x | 1.3383x | yes |
| 2 | 8 | 8x5120x10240 + 8x5120x6144 | 92.943 | 0.2496x | 0.5280x | 1.3392x | yes |
| 2 | 16 | 16x5120x10240 + 16x5120x6144 | 96.895 | 0.2784x | 0.5098x | 1.2845x | yes |
| 2.5 | 1 | 1x5120x10240 + 1x5120x6144 | 90.066 | 0.2588x | 0.5531x | 1.3639x | yes |
| 2.5 | 2 | 2x5120x10240 + 2x5120x6144 | 92.589 | 0.2606x | 0.5316x | 1.3474x | yes |
| 2.5 | 4 | 4x5120x10240 + 4x5120x6144 | 93.189 | 0.2608x | 0.5271x | 1.3420x | yes |
| 2.5 | 8 | 8x5120x10240 + 8x5120x6144 | 93.536 | 0.2480x | 0.5246x | 1.3357x | yes |
| 2.5 | 16 | 16x5120x10240 + 16x5120x6144 | 97.232 | 0.2774x | 0.5081x | 1.2836x | yes |
| 3 | 1 | 1x5120x10240 + 1x5120x6144 | 86.768 | 0.2686x | 0.5741x | 1.3633x | yes |
| 3 | 2 | 2x5120x10240 + 2x5120x6144 | 89.056 | 0.2709x | 0.5527x | 1.3548x | yes |
| 3 | 4 | 4x5120x10240 + 4x5120x6144 | 89.540 | 0.2715x | 0.5486x | 1.3520x | yes |
| 3 | 8 | 8x5120x10240 + 8x5120x6144 | 89.913 | 0.2580x | 0.5458x | 1.3444x | yes |
| 3 | 16 | 16x5120x10240 + 16x5120x6144 | 93.307 | 0.2891x | 0.5294x | 1.2967x | yes |

Geometric speedup is **1.3357x over Phase 81**, with 15 of 15 wins. Raw
distilled results are in
`artifacts/a41_phase82_h100/qwen38_27b_multiblock_linear_recovery.json`.

## Next target

The recovery bottleneck has been reduced enough that the 55-microsecond W3
grouped inner kernel is again the largest component. Profile the Phase-82
timeline before choosing between more decoded-level overlap and replacing the
remaining M1--M4 staged shared-input transform with a parallel composite form.
