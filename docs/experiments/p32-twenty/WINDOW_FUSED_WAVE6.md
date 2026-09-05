# Integrated window wave 6

The direct decode/MMA Triton path was retested with `BM=32`, `BN=32`, and
`split=1` on the immutable F6 seed-7 checkpoint. The sweep covered 12 real
projections and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 144 cases passed the local correctness gate. Fused/window full-layer
speedup medians were:

| M | Median fused/window |
|---:|---:|
| 1 | 0.959x |
| 2 | 0.962x |
| 4 | 0.970x |
| 8 | 0.954x |
| 16 | 0.972x |
| 32 | 0.954x |
| 64 | 0.962x |
| 128 | 0.959x |
| 256 | 0.961x |
| 512 | 0.999x |
| 1024 | 1.004x |
| 2048 | 1.033x |

BM32 is effectively tied with production window at the largest M and slower
at smaller M; it does not meet the promotion target. Raw reports are [worker
0](results/window-fused-wave6-bm32/worker0.json), [worker
1](results/window-fused-wave6-bm32/worker1.json), [worker
2](results/window-fused-wave6-bm32/worker2.json), and [worker
3](results/window-fused-wave6-bm32/worker3.json).
