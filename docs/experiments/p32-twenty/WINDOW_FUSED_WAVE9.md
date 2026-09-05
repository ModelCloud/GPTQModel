# Integrated window wave 9

The direct decode/MMA Triton path was tested with `BM=64`, `BN=64`, and
`split=1` on the immutable F6 seed-7 checkpoint. The eight-point shape coverage
was expanded to 12 row counts for 12 real projections:

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 144 cases passed the local correctness gate. Median fused/window full-layer
ratios were:

| M | Median fused/window |
|---:|---:|
| 1 | 0.956x |
| 2 | 0.966x |
| 4 | 0.947x |
| 8 | 0.949x |
| 16 | 0.939x |
| 32 | 0.982x |
| 64 | 0.949x |
| 128 | 0.963x |
| 256 | 0.955x |
| 512 | 1.081x |
| 1024 | 1.091x |
| 2048 | 1.216x |

BM64/BN64 is the strongest tested direct decode/MMA configuration at large M,
but it remains slower at small M and below the 2x target. A few small-M
per-projection timing outliers remain in this harness and require repeat
sampling before treating the large-M gain as production-stable. All 144 fused
outputs passed; maximum teacher error was 0.012533 and maximum window drift was
0.004578.

Raw reports are [GPU 0](results/window-fused-wave9-bm64bn64/gpu0.json), [GPU
1](results/window-fused-wave9-bm64bn64/gpu1.json), [GPU
2](results/window-fused-wave9-bm64bn64/gpu2.json), and [GPU
3](results/window-fused-wave9-bm64bn64/gpu3.json).
