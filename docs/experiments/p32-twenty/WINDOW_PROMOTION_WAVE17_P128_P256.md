# Integrated window promotion wave 17: K128 and K256

The BM64/BN64/split-1 fused window kernel was tested with FP16 partial
accumulation and FP32 promotion every K128 and K256 elements on the immutable
F6 seed-7 checkpoint. Each arm covers 12 real P32 projections and 12 row
counts.

| Promotion | Passing cases | Max teacher mean | Max teacher error | Max window mean drift | Max window drift |
|---:|---:|---:|---:|---:|---:|
| K128 | 142/144 | 0.00972403 | 0.0431132 | 0.00928312 | 0.0399685 |
| K256 | 132/144 | 0.0133808 | 0.0542338 | 0.0131684 | 0.0549069 |

K128 fails layer-1 `mlp.down_proj` at M=1 and M=2. K256 fails that same
projection at every M. The intervals are therefore unsuitable as global
policies and the maximum-error gate is decisive even where average error is
small.

| M | K128 fused/window median | K256 fused/window median |
|---:|---:|---:|
| 1 | 0.973x | 0.960x |
| 2 | 0.953x | 0.946x |
| 4 | 0.975x | 0.959x |
| 8 | 0.961x | 0.926x |
| 16 | 0.953x | 0.944x |
| 32 | 0.946x | 0.979x |
| 64 | 0.969x | 0.959x |
| 128 | 0.966x | 0.966x |
| 256 | 0.936x | 0.956x |
| 512 | 1.074x | 1.072x |
| 1024 | 1.055x | 1.071x |
| 2048 | 1.188x | 1.201x |

The full promotion sweep provides a per-module lead: most tested projections
remain numerically safe with shorter intervals, while layer-1 down must retain
the exact FP32 accumulation path. Model-level quality and profile validation
are still required before dispatching such a policy.

Raw reports:

- [K128 GPU0](results/window-promotion-wave17/p128-gpu0)
- [K128 GPU1](results/window-promotion-wave17/p128-gpu1)
- [K128 GPU2](results/window-promotion-wave17/p128-gpu2)
- [K128 GPU3](results/window-promotion-wave17/p128-gpu3)
- [K256 GPU4](results/window-promotion-wave17/p256-gpu4)
- [K256 GPU5](results/window-promotion-wave17/p256-gpu5)
- [K256 GPU6](results/window-promotion-wave17/p256-gpu6)
- [K256 GPU7](results/window-promotion-wave17/p256-gpu7)
