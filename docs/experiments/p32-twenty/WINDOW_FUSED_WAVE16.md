# Integrated window wave 16

Wave 16 completed the remaining direct decode/MMA tile-layout sweep on the
immutable F6 seed-7 checkpoint. BM16/BN64 was tested with split-K=1 on GPUs
0–3 and split-K=2 on GPUs 4–7. Each arm covered 12 real P32 projections at:

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 288 cases passed the FP32-teacher and production-window local gates. The
worst teacher mean absolute error was `0.0027259162`; worst absolute error was
`0.0125326961`. The worst mean drift from window was `0.0001104819`, and the
worst absolute drift was `0.0045776367`.

## Full-layer speed ratios

Ratios are `fused full-layer latency / production window full-layer latency`;
values below 1.0 are faster. Each median is across the 12 projections.

| M | BM16/BN64/split1 | BM16/BN64/split2 |
|---:|---:|---:|
| 1 | 0.947x | 0.921x |
| 2 | 0.947x | 0.954x |
| 4 | 0.970x | 0.948x |
| 8 | 0.955x | 0.944x |
| 16 | 0.933x | 0.968x |
| 32 | 0.957x | 0.936x |
| 64 | 0.941x | 0.940x |
| 128 | 0.949x | 0.956x |
| 256 | 0.968x | 0.937x |
| 512 | 0.971x | 0.976x |
| 1024 | 0.990x | 0.969x |
| 2048 | 0.992x | 0.973x |

The BM16/BN64 split-K=2 arm is the fastest of this wave at M=1, but neither
arm reaches a 2x speedup and neither consistently beats the stronger
split-1 configurations from earlier waves. The tile-layout sweep therefore
does not advance the window kernel. Fragment-ready output, producer/consumer
overlap, and fused SU/Hadamard transforms remain unimplemented follow-ups.

## Reproducibility

Raw worker reports:

- [BM16/BN64/split1 GPU0](results/window-fused-wave16/bm16bn64s1-gpu0)
- [BM16/BN64/split1 GPU1](results/window-fused-wave16/bm16bn64s1-gpu1)
- [BM16/BN64/split1 GPU2](results/window-fused-wave16/bm16bn64s1-gpu2)
- [BM16/BN64/split1 GPU3](results/window-fused-wave16/bm16bn64s1-gpu3)
- [BM16/BN64/split2 GPU4](results/window-fused-wave16/bm16bn64s2-gpu4)
- [BM16/BN64/split2 GPU5](results/window-fused-wave16/bm16bn64s2-gpu5)
- [BM16/BN64/split2 GPU6](results/window-fused-wave16/bm16bn64s2-gpu6)
- [BM16/BN64/split2 GPU7](results/window-fused-wave16/bm16bn64s2-gpu7)
