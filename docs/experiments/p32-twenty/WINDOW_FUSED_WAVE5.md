# Integrated window wave 5

The existing Triton window GEMM was tested with direct decode into its MMA
operand tile using `BM=16`, `BN=32`, and `split=1`. The immutable F6 seed-7
weights and captured real activations were used for 12 projections and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 144 fused cases passed the local teacher gate and the fused output stayed
bitwise deterministic within the run. The median fused/window full-layer ratio
(below 1 means slower) was:

| M | Median fused/window |
|---:|---:|
| 1 | 0.959x |
| 2 | 0.961x |
| 4 | 0.988x |
| 8 | 0.955x |
| 16 | 0.966x |
| 32 | 0.980x |
| 64 | 0.943x |
| 128 | 0.953x |
| 256 | 0.957x |
| 512 | 0.946x |
| 1024 | 0.912x |
| 2048 | 0.889x |

This configuration therefore does not advance as a speed candidate: it adds
the direct decode/MMA fusion but remains slower than the production window
path, especially at large M. The raw per-projection reports are [worker
0](results/window-fused-wave5-bm16/worker0.json), [worker
1](results/window-fused-wave5-bm16/worker1.json), [worker
2](results/window-fused-wave5-bm16/worker2.json), and [worker
3](results/window-fused-wave5-bm16/worker3.json).

The result still provides an exact-format fragment-consumption baseline for
future layout, transform-fusion, and pipeline variants.
