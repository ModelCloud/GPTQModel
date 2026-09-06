# Window model tile wave 24 partial results (superseded)

The complete comparison is now documented in [WINDOW_MODEL_WAVE24.md](WINDOW_MODEL_WAVE24.md).

This wave tests model-level window tile shapes on the exact F6 seed-7 snapshot.
The production window control and the BM64/BN64 split1 fused arm have
completed. Six other tile arms are still running.

| Arm | Host GPU | PPL | M=1 | M=16 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production window control | 0 | 26.9915252 | 40.432 | 41.820 | 40.663 | 75.952 | 289.741 | 41.463 |
| BM64/BN64 split1, exact K0 | 6 | 26.9936504 | 45.301 | 44.434 | 46.135 | 57.111 | 206.746 | 44.512 |

The completed fused arm is slower at small M and decode, and is faster at
M=512 and M=2048 in this single model run. The local repeat wave showed that
these regimes need matched repeats; this partial result is therefore a tile
screen, not a promotion decision. The six remaining arms cover BM16/32/64,
BN32/64, and split-K=2.

Raw reports:

- [control GPU 0](results/window-model-wave24/control-gpu0.json)
- [BM64/BN64 split1 GPU 6](results/window-model-wave24/bm64bn64s1-gpu6.json)
