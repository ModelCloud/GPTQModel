# Integrated window wave 7

The direct decode/MMA Triton path was tested with `BM=64`, `BN=32`, and
`split=1` on all eight host GPUs, using the immutable F6 seed-7 checkpoint.
The four workers covered 12 real projections and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 144 cases passed the local correctness gate. Median fused/window
full-layer ratios across the 12 projections were:

| M | Median fused/window |
|---:|---:|
| 1 | 0.962x |
| 16 | 0.944x |
| 512 | 1.097x |
| 2048 | 1.279x |

The complete per-M curve is in the four raw reports. BM64/BN32 improves the
large-M median over the smaller tile configurations, but remains slower at
small M and below the 2x target. It is a screening result rather than a
promotion candidate.

Raw reports: [GPU 4](results/window-fused-wave7-bm64/gpu4.json), [GPU
5](results/window-fused-wave7-bm64/gpu5.json), [GPU
6](results/window-fused-wave7-bm64/gpu6.json), and [GPU
7](results/window-fused-wave7-bm64/gpu7.json).
