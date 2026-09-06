# Integrated window wave 8

The direct decode/MMA Triton path was tested with `BM=32`, `BN=64`, and
`split=1` on the immutable F6 seed-7 checkpoint. The sweep covered 12 real
projections and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 144 cases passed the local correctness gate. The median fused/window
full-layer ratios were:

| M | Median fused/window |
|---:|---:|
| 1 | 0.943x |
| 2 | 0.956x |
| 4 | 0.966x |
| 8 | 0.951x |
| 16 | 0.969x |
| 32 | 0.975x |
| 64 | 0.947x |
| 128 | 0.950x |
| 256 | 0.949x |
| 512 | 1.012x |
| 1024 | 1.027x |
| 2048 | 1.023x |

Several small-M per-projection timings show large reciprocal outliers in this
configuration, so the medians should be treated as a screening result pending
a dedicated timing-repeat investigation. The configuration does not establish
a stable speed promotion. All 144 fused outputs passed the local gate; the
maximum recorded teacher error was 0.012533 and the maximum window drift was
0.004608.

Raw reports are [GPU 0](results/window-fused-wave8-bm32bn64/gpu0.json), [GPU
1](results/window-fused-wave8-bm32bn64/gpu1.json), [GPU
2](results/window-fused-wave8-bm32bn64/gpu2.json), and [GPU
3](results/window-fused-wave8-bm32bn64/gpu3.json).
