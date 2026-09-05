# Integrated window promotion wave 17: K32 and K64

The BM64/BN64/split-1 fused window kernel was tested with FP16 partial
accumulation and FP32 promotion every K32 and K64 elements on the immutable
F6 seed-7 checkpoint. Each arm covers 12 real P32 projections and 12 row
counts.

Neither interval passes the global local gate. K32 passes 143/144 cases; its
single failure is layer-1 `mlp.down_proj` at M=1. K64 passes 142/144; its two
failures are layer-1 `mlp.down_proj` at M=1 and M=2.

| Promotion | Cases | Max teacher mean | Max teacher error | Max window mean drift | Max window drift |
|---:|---:|---:|---:|---:|---:|
| K32 | 143/144 | 0.00589171 | 0.0272694 | 0.00514145 | 0.0240825 |
| K64 | 142/144 | 0.00739415 | 0.0339589 | 0.00680621 | 0.0321151 |

The median fused/window ratios are:

| M | K32 | K64 |
|---:|---:|---:|
| 1 | 0.957x | 0.953x |
| 2 | 0.972x | 0.956x |
| 4 | 0.943x | 0.970x |
| 8 | 0.941x | 0.980x |
| 16 | 0.940x | 0.984x |
| 32 | 0.953x | 0.973x |
| 64 | 0.956x | 0.964x |
| 128 | 0.946x | 0.985x |
| 256 | 0.964x | 0.951x |
| 512 | 1.057x | 1.055x |
| 1024 | 1.047x | 1.058x |
| 2048 | 1.178x | 1.187x |

Shorter promotion intervals are faster than the exact control at some small
M values, but both fail on the sensitive layer-1 down projection. K32 and K64
are therefore not global policies. Per-module dispatch remains possible only
after K128 and K256 are checked and the failing module is kept on FP32.

Raw reports:

- [K32 GPU0](results/window-promotion-wave17/p32-gpu0)
- [K32 GPU1](results/window-promotion-wave17/p32-gpu1)
- [K32 GPU2](results/window-promotion-wave17/p32-gpu2)
- [K32 GPU3](results/window-promotion-wave17/p32-gpu3)
- [K64 GPU4](results/window-promotion-wave17/p64-gpu4)
- [K64 GPU5](results/window-promotion-wave17/p64-gpu5)
- [K64 GPU6](results/window-promotion-wave17/p64-gpu6)
- [K64 GPU7](results/window-promotion-wave17/p64-gpu7)
