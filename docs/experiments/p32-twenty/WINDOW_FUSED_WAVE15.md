# Integrated window wave 15

Wave 15 tested split-K=4 for the direct decode/MMA Triton path on the immutable
F6 seed-7 checkpoint. Two tile configurations ran concurrently on the eight
UUID-pinned GPUs. Each configuration covered 12 real P32 projections and:

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 288 cases passed both local correctness checks: the FP32 teacher gate and
the window-reference gate. The worst observed teacher mean absolute error was
`0.0027283180`, with worst absolute error `0.0124638379`. The worst drift from
the production window reference was mean `2.5793297e-05` and maximum
`0.0034790039`.

## Full-layer speed ratios

Values are full-layer speedups, `production-window latency / fused latency`;
values above 1.0 are faster. Each median is across the 12 projections.

| M | BM64/BN64/split4 | BM32/BN32/split4 |
|---:|---:|---:|
| 1 | 0.951x | 0.940x |
| 2 | 0.945x | 0.942x |
| 4 | 0.935x | 0.952x |
| 8 | 0.943x | 0.938x |
| 16 | 0.951x | 0.960x |
| 32 | 0.938x | 0.936x |
| 64 | 0.950x | 0.967x |
| 128 | 0.958x | 0.964x |
| 256 | 0.953x | 0.962x |
| 512 | 1.043x | 0.999x |
| 1024 | 1.093x | 0.988x |
| 2048 | 1.225x | 1.013x |

Split-K=4 does not improve the best split-1 configuration consistently. The
BM64/BN64 path reaches 1.225x at M=2048, while BM32/BN32 is near parity at
M=512–2048; both remain below the requested 2x target. The extra split-K
reduction therefore does not advance the window kernel.

The result is a useful negative control: direct decode/MMA fusion preserves
the reconstructed outputs under split-K=4, but the reduction and scheduling
cost outweigh any available parallelism. Producer/consumer overlap,
fragment-layout fusion, and transform fusion remain open implementation work.

## Reproducibility

The eight raw worker reports are archived here:

- [BM64/BN64/split4 GPU0](results/window-fused-wave15/bm64bn64s4-gpu0)
- [BM64/BN64/split4 GPU1](results/window-fused-wave15/bm64bn64s4-gpu1)
- [BM64/BN64/split4 GPU2](results/window-fused-wave15/bm64bn64s4-gpu2)
- [BM64/BN64/split4 GPU3](results/window-fused-wave15/bm64bn64s4-gpu3)
- [BM32/BN32/split4 GPU4](results/window-fused-wave15/bm32bn32s4-gpu4)
- [BM32/BN32/split4 GPU5](results/window-fused-wave15/bm32bn32s4-gpu5)
- [BM32/BN32/split4 GPU6](results/window-fused-wave15/bm32bn32s4-gpu6)
- [BM32/BN32/split4 GPU7](results/window-fused-wave15/bm32bn32s4-gpu7)
