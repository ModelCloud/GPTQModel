# Window model policy tile wave 26 partial results (superseded)

The complete comparison is now documented in [WINDOW_MODEL_WAVE26.md](WINDOW_MODEL_WAVE26.md).

Wave26 compares four production-window controls with four per-module promotion
policy arms using BM64/BN32. The controls completed on host GPUs 0–3; policy
jobs on GPUs 4–7 remain active.

| Control GPU | PPL | M=1 | M=16 | M=128 | M=512 | M=2048 | decode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26.9915252 | 38.750 | 39.180 | 40.838 | 76.119 | 290.633 | 39.428 |
| 1 | 26.9915252 | 42.206 | 40.570 | 41.220 | 88.248 | 333.188 | 41.778 |
| 2 | 26.9915252 | 39.725 | 40.375 | 42.078 | 75.886 | 289.212 | 41.149 |
| 3 | 26.9915252 | 41.052 | 40.590 | 41.063 | 76.109 | 289.227 | 42.250 |

The four controls show ordinary cross-GPU timing variation, especially at
M=512 and M=2048. Their pooled control median is 40.389 ms at M=1, 40.478 ms
at M=16, 41.142 ms at M=128, 76.114 ms at M=512, 289.920 ms at M=2048, and
41.464 ms in decode. Policy results will be compared against these same-wave
controls after all four reports complete.

Raw reports:

- [control GPU 0](results/window-model-wave26/control-gpu0.json)
- [control GPU 1](results/window-model-wave26/control-gpu1.json)
- [control GPU 2](results/window-model-wave26/control-gpu2.json)
- [control GPU 3](results/window-model-wave26/control-gpu3.json)
