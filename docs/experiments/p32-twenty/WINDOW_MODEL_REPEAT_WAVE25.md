# Window model leading tile repeat wave 25

This wave repeats the two leading wave24 geometries on four GPUs each using
the exact F6 seed-7 snapshot. BM64/BN32/split1 ran on host GPUs 0–3 and
BM64/BN64/split1 ran on host GPUs 4–7. The production-window medians below
come from the matched four-GPU window repeat in
[wave21](WINDOW_MODEL_REPEAT_WAVE21.md).

## Pooled results

Values are full-model milliseconds. Speedup is the production-window median
divided by the repeated fused median.

| M / regime | Window | BM64/BN32 | Speedup | BM64/BN64 | Speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 40.282 | 47.504 | 0.848x | 46.453 | 0.867x |
| 2 | 41.150 | 46.763 | 0.880x | 46.252 | 0.890x |
| 4 | 41.002 | 46.519 | 0.881x | 45.851 | 0.894x |
| 8 | 41.006 | 46.917 | 0.874x | 45.825 | 0.895x |
| 16 | 40.525 | 47.249 | 0.858x | 46.155 | 0.878x |
| 32 | 40.536 | 46.097 | 0.879x | 46.710 | 0.868x |
| 128 | 41.269 | 46.789 | 0.882x | 46.406 | 0.889x |
| 512 | 76.079 | 53.882 | 1.412x | 56.827 | 1.339x |
| 2048 | 289.911 | 203.683 | 1.423x | 206.763 | 1.402x |
| decode | 40.543 | 47.826 | 0.848x | 46.462 | 0.873x |

The large-prefill gains reproduce across four GPUs: BM64/BN32 reaches 1.423x
at M=2048 and 1.412x at M=512; BM64/BN64 reaches 1.402x and 1.339x. Both
remain slower at M=1–128 and in decode. This is a stable linear-model regime
finding, not a full-model 2x result.

Both arms produced PPL `26.9936504279` on the 16-row bounded input set. This
matches within-arm across all four repeats and is close to the window baseline;
the completed ARC/GSM8K comparison remains the quality evidence.

## Per-GPU medians

| Arm | GPU | M=1 | M=16 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM64/BN32 | 0 | 47.175 | 47.250 | 46.068 | 53.723 | 203.520 | 48.679 |
| BM64/BN32 | 1 | 48.095 | 47.248 | 46.849 | 61.749 | 232.002 | 52.930 |
| BM64/BN32 | 2 | 46.764 | 53.255 | 46.749 | 53.962 | 203.846 | 46.973 |
| BM64/BN32 | 3 | 47.834 | 46.496 | 46.829 | 53.802 | 202.763 | 46.073 |
| BM64/BN64 | 4 | 47.827 | 49.473 | 47.229 | 58.925 | 206.939 | 47.459 |
| BM64/BN64 | 5 | 46.039 | 46.153 | 47.058 | 56.793 | 206.659 | 45.527 |
| BM64/BN64 | 6 | 46.867 | 46.157 | 45.754 | 56.774 | 206.847 | 45.383 |
| BM64/BN64 | 7 | 45.592 | 45.278 | 45.192 | 56.861 | 206.679 | 47.398 |

Raw reports:

- [BM64/BN32 GPU 0](results/window-model-repeat-wave25/bm64bn32-gpu0.json)
- [BM64/BN32 GPU 1](results/window-model-repeat-wave25/bm64bn32-gpu1.json)
- [BM64/BN32 GPU 2](results/window-model-repeat-wave25/bm64bn32-gpu2.json)
- [BM64/BN32 GPU 3](results/window-model-repeat-wave25/bm64bn32-gpu3.json)
- [BM64/BN64 GPU 4](results/window-model-repeat-wave25/bm64bn64s1-gpu4.json)
- [BM64/BN64 GPU 5](results/window-model-repeat-wave25/bm64bn64s1-gpu5.json)
- [BM64/BN64 GPU 6](results/window-model-repeat-wave25/bm64bn64s1-gpu6.json)
- [BM64/BN64 GPU 7](results/window-model-repeat-wave25/bm64bn64s1-gpu7.json)
