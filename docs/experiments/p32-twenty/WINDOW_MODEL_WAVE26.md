# Window model policy tile wave 26

Wave26 compares four production-window controls with four per-module promotion
policy runs using BM64/BN32, all on the exact F6 seed-7 snapshot. The controls
ran on host GPUs 0–3 and policy on host GPUs 4–7. Each report has 16 bounded
quality rows and the nine requested prefill sizes plus growing-KV decode.

## Pooled same-wave result

Values are full-model forward medians in milliseconds. Speedup is control time
divided by policy time.

| M / regime | Control | Policy BM64/BN32 | Speedup |
|---|---:|---:|---:|
| 1 | 40.388 | 47.525 | 0.850x |
| 2 | 40.163 | 46.861 | 0.857x |
| 4 | 39.795 | 46.946 | 0.848x |
| 8 | 39.819 | 47.800 | 0.833x |
| 16 | 40.472 | 47.765 | 0.847x |
| 32 | 40.123 | 47.692 | 0.841x |
| 128 | 41.142 | 48.168 | 0.854x |
| 512 | 76.114 | 56.200 | 1.354x |
| 2048 | 289.930 | 210.435 | 1.378x |
| decode | 41.463 | 46.510 | 0.891x |

The BM64/BN32 policy is slower through M=128 and in decode, then reaches
1.354x at M=512 and 1.378x at M=2048. This improves on the BM64/BN64 policy
repeat at large M, but it does not meet the 2x full-model target. The PPL was
`26.9915252378` for all controls and `26.9933600871` for all policy reports
on the bounded 16-row input set. Wave23 remains the downstream quality
comparison for window versus policy.

## Per-GPU medians

| Arm | GPU | M=1 | M=16 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 38.750 | 39.180 | 40.838 | 76.119 | 290.633 | 39.428 |
| control | 1 | 42.206 | 40.570 | 41.220 | 88.248 | 333.188 | 41.778 |
| control | 2 | 39.725 | 40.375 | 42.078 | 75.886 | 289.212 | 41.149 |
| control | 3 | 41.052 | 40.590 | 41.063 | 76.109 | 289.227 | 42.250 |
| policy BM64/BN32 | 4 | 47.034 | 45.950 | 49.316 | 56.263 | 210.620 | 47.018 |
| policy BM64/BN32 | 5 | 45.798 | 47.457 | 45.896 | 56.902 | 210.126 | 45.422 |
| policy BM64/BN32 | 6 | 48.015 | 48.073 | 47.020 | 56.137 | 210.541 | 46.002 |
| policy BM64/BN32 | 7 | 52.173 | 55.520 | 51.027 | 56.082 | 210.329 | 48.766 |

Raw reports:

- [control GPU 0](results/window-model-wave26/control-gpu0.json)
- [control GPU 1](results/window-model-wave26/control-gpu1.json)
- [control GPU 2](results/window-model-wave26/control-gpu2.json)
- [control GPU 3](results/window-model-wave26/control-gpu3.json)
- [policy GPU 4](results/window-model-wave26/policy-bm64bn32-gpu4.json)
- [policy GPU 5](results/window-model-wave26/policy-bm64bn32-gpu5.json)
- [policy GPU 6](results/window-model-wave26/policy-bm64bn32-gpu6.json)
- [policy GPU 7](results/window-model-wave26/policy-bm64bn32-gpu7.json)
