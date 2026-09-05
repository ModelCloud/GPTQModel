# Window model tile wave 24

This completed model-level tile sweep used the exact F6 seed-7 snapshot, 16
fixed model input rows, and the same full-model prefill/decode harness for each
arm. The control is the production window path. Fused arms use exact K0 FP32
accumulation; only the tile geometry and split-K setting vary.

## Full timing table

Values are full-model forward medians in milliseconds. Decode is the growing
KV-cache prompt-128 / 32-new-token measurement. The speedup column is control
latency divided by arm latency; values above 1.0 are faster.

| Arm | GPU | PPL | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production window control | 0 | 26.9915252 | 40.432 | 41.209 | 39.977 | 39.749 | 41.820 | 40.744 | 40.663 | 75.952 | 289.741 | 41.463 |
| BM16/BN32 split1 | 1 | 26.9936504 | 46.681 | 44.982 | 50.779 | 46.574 | 44.863 | 45.395 | 45.455 | 96.301 | 386.221 | 48.117 |
| BM16/BN64 split1 | 2 | 26.9936504 | 47.839 | 49.163 | 46.636 | 48.917 | 46.469 | 47.945 | 53.349 | 74.602 | 288.192 | 45.472 |
| BM32/BN32 split1 | 3 | 26.9936504 | 46.983 | 47.216 | 47.511 | 45.932 | 46.866 | 45.883 | 48.795 | 71.963 | 273.143 | 45.332 |
| BM32/BN64 split1 | 4 | 26.9936504 | 47.262 | 50.055 | 47.112 | 49.604 | 47.190 | 46.748 | 46.606 | 70.729 | 263.853 | 48.166 |
| BM64/BN32 split1 | 5 | 26.9936504 | 46.699 | 47.121 | 46.472 | 45.998 | 56.667 | 49.210 | 49.360 | 53.912 | 203.340 | 52.851 |
| BM64/BN64 split1 | 6 | 26.9936504 | 45.301 | 44.660 | 44.256 | 45.386 | 44.434 | 44.798 | 46.135 | 57.111 | 206.746 | 44.512 |
| BM64/BN64 split2 | 7 | 26.9880283 | 46.637 | 48.118 | 47.686 | 48.513 | 47.432 | 47.264 | 47.285 | 54.464 | 204.606 | 47.943 |

| Arm | M=1 | M=16 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|
| BM16/BN32 split1 speedup | 0.866x | 0.932x | 0.895x | 0.789x | 0.750x | 0.862x |
| BM16/BN64 split1 speedup | 0.845x | 0.900x | 0.762x | 1.018x | 1.005x | 0.912x |
| BM32/BN32 split1 speedup | 0.861x | 0.892x | 0.833x | 1.055x | 1.061x | 0.915x |
| BM32/BN64 split1 speedup | 0.855x | 0.886x | 0.872x | 1.074x | 1.098x | 0.861x |
| BM64/BN32 split1 speedup | 0.866x | 0.738x | 0.824x | 1.409x | 1.425x | 0.785x |
| BM64/BN64 split1 speedup | 0.893x | 0.941x | 0.881x | 1.330x | 1.401x | 0.932x |
| BM64/BN64 split2 speedup | 0.867x | 0.882x | 0.860x | 1.395x | 1.416x | 0.865x |

The model result confirms the local shape trend: BM64 is the strongest family
at M=512/2048, with BM64/BN32 split1 reaching 1.425x at M=2048 in this
single run. Every fused geometry is slower than the production window at M=1
and in decode. No arm approaches the 2x full-model promotion threshold, and
the M=512/2048 gains require matched repeats before being treated as stable.

PPL differences across this 16-row input set are not a quality conclusion.
Wave23 supplies the downstream ARC/GSM8K comparison for the production window
and per-module policy. This wave does not add a new downstream task run.

Raw reports:

- [control GPU 0](results/window-model-wave24/control-gpu0.json)
- [BM16/BN32 GPU 1](results/window-model-wave24/bm16bn32-gpu1.json)
- [BM16/BN64 GPU 2](results/window-model-wave24/bm16bn64-gpu2.json)
- [BM32/BN32 GPU 3](results/window-model-wave24/bm32bn32-gpu3.json)
- [BM32/BN64 GPU 4](results/window-model-wave24/bm32bn64-gpu4.json)
- [BM64/BN32 GPU 5](results/window-model-wave24/bm64bn32-gpu5.json)
- [BM64/BN64 split1 GPU 6](results/window-model-wave24/bm64bn64s1-gpu6.json)
- [BM64/BN64 split2 GPU 7](results/window-model-wave24/bm64bn64s2-gpu7.json)
