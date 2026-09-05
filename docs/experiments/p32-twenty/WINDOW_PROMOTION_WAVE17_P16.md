# Integrated window promotion wave 17: K16

The BM64/BN64/split-1 fused window kernel was run with FP16 partial
accumulation and FP32 promotion every K16 elements on the immutable F6 seed-7
checkpoint. The arm covers 12 real P32 projections and 12 row counts.

There were 143/144 local passes. The single failure is:

| Projection | M | Teacher mean error | Teacher max error | Window mean drift | Window max drift |
|---|---:|---:|---:|---:|---:|
| `model.layers.1.mlp.down_proj` | 1 | 0.00479307 | 0.0204201 | 0.00401083 | 0.0159455 |

This is a real integrated-kernel failure against both gates, so K16 is not a
global promotion policy. Its median fused/window ratios were:

| M | K16 fused/window median |
|---:|---:|
| 1 | 0.965x |
| 2 | 0.940x |
| 4 | 0.974x |
| 8 | 0.944x |
| 16 | 0.946x |
| 32 | 0.962x |
| 64 | 0.967x |
| 128 | 0.953x |
| 256 | 0.968x |
| 512 | 1.051x |
| 1024 | 1.072x |
| 2048 | 1.163x |

The arm is faster than the exact control at several row counts, but the
accuracy failure disqualifies it from blanket deployment. A per-projection
policy remains possible only after the remaining intervals are measured.

Raw reports:

- [GPU4](results/window-promotion-wave17/p16-gpu4)
- [GPU5](results/window-promotion-wave17/p16-gpu5)
- [GPU6](results/window-promotion-wave17/p16-gpu6)
- [GPU7](results/window-promotion-wave17/p16-gpu7)
