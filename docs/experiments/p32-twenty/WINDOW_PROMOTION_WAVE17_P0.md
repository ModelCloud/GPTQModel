# Integrated window promotion wave 17: exact control

The `promotion_k=0` arm of wave 17 reran the BM64/BN64/split-1 fused window
kernel after adding the blockwise promotion path. It covers 12 real P32
projections and 12 row counts on the immutable F6 seed-7 checkpoint.

All 144 cases passed both local gates. The maximum teacher mean error was
`0.0027259162`, maximum teacher absolute error `0.0125326961`, maximum mean
drift from production window `0.0001104819`, and maximum window drift
`0.0045776367`.

Values are full-layer speedups, `production-window latency / fused latency`;
values above 1.0 are faster.

| M | Exact-control speedup median |
|---:|---:|
| 1 | 0.951x |
| 2 | 0.972x |
| 4 | 0.941x |
| 8 | 0.925x |
| 16 | 0.948x |
| 32 | 0.960x |
| 64 | 0.967x |
| 128 | 0.951x |
| 256 | 0.949x |
| 512 | 1.137x |
| 1024 | 1.080x |
| 2048 | 1.225x |

This control confirms that the new promotion parameter leaves the fused
operator valid when disabled. It is not a promotion candidate; the remaining
wave arms test K16, K32, K64, K128, and K256 partial accumulation.

Raw reports:

- [GPU0](results/window-promotion-wave17/p0-gpu0)
- [GPU1](results/window-promotion-wave17/p0-gpu1)
- [GPU2](results/window-promotion-wave17/p0-gpu2)
- [GPU3](results/window-promotion-wave17/p0-gpu3)
