# Integrated window wave 14

The direct decode/MMA Triton path was tested with split-K=2 on the immutable
F6 seed-7 checkpoint. Two tile configurations ran concurrently across all eight
UUID-mapped GPUs. Each configuration covered 12 real projections and

`M = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`.

All 288 cases in each configuration passed the local correctness gate.
Median full-layer speedups were `production-window latency / fused latency`;
values above 1.0 are faster:

| M | BM64/BN64/split2 | BM32/BN32/split2 |
|---:|---:|---:|
| 1 | 0.978x | 0.952x |
| 2 | 0.959x | 0.950x |
| 4 | 0.969x | 0.929x |
| 8 | 0.977x | 0.965x |
| 16 | 0.947x | 0.940x |
| 32 | 0.944x | 0.950x |
| 64 | 0.956x | 0.944x |
| 128 | 0.957x | 0.959x |
| 256 | 0.937x | 0.942x |
| 512 | 1.051x | 1.007x |
| 1024 | 1.094x | 0.992x |
| 2048 | 1.214x | 1.022x |

Split-K=2 does not beat the corresponding split-1 BM64/BN64 result consistently
and adds reduction work. It does not advance. Maximum teacher error was 0.012510 and
maximum window drift was 0.002350 for both configurations.

Raw reports are in [the wave-14 result directory](results/window-fused-wave14/).
