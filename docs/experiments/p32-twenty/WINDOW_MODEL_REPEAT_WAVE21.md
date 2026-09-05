# Window model repeat wave 21

This matched repeat checks the wave20 full-model timing on the exact F6 seed-7
snapshot. Four production-window controls ran on host GPUs 0–3 and four
per-module promotion-policy arms ran on host GPUs 4–7. Each completed with 16
quality rows, nine prefill sizes, and a 32-sample growing-KV decode measurement.

The repeat resolves the wave20 M=1 anomaly: the production control median is
now 40.282 ms pooled across four GPUs, rather than the earlier 331.399 ms
single-GPU median. The earlier M=1 comparison is therefore discarded.

## Pooled medians

Values are milliseconds; the speedup is production-window time divided by
policy time, so values above 1.0 are faster for policy.

| M / regime | Window median | Policy median | Policy speedup |
|---|---:|---:|---:|
| 1 | 40.282 | 48.805 | 0.825x |
| 2 | 41.150 | 48.476 | 0.849x |
| 4 | 41.002 | 47.890 | 0.856x |
| 8 | 41.006 | 48.143 | 0.852x |
| 16 | 40.525 | 47.644 | 0.851x |
| 32 | 40.536 | 48.894 | 0.829x |
| 128 | 41.269 | 48.042 | 0.859x |
| 512 | 76.079 | 58.789 | 1.294x |
| 2048 | 289.911 | 212.245 | 1.366x |
| decode, prompt 128 / 32 new | 40.543 | 47.285 | 0.857x |

The policy is slower through M=128 and in decode. At M=512 it reaches 1.294x,
and at M=2048 it reaches 1.366x in the pooled repeat. This remains a model
baseline result, not the requested full production scorecard: the policy still
retains transform/output boundaries, and no ARC/GSM8K or Nsight counter wave
was run here.

## Per-GPU report medians

| Arm | Host GPU | PPL | M=1 | M=16 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| window | 0 | 26.9915252 | 44.771 | 41.749 | 42.086 | 76.158 | 290.632 | 45.308 |
| window | 1 | 26.9915252 | 39.857 | 41.035 | 41.088 | 87.850 | 332.500 | 40.800 |
| window | 2 | 26.9915252 | 40.708 | 39.468 | 41.449 | 76.000 | 289.191 | 40.287 |
| window | 3 | 26.9915252 | 39.716 | 40.014 | 40.088 | 75.870 | 289.012 | 40.024 |
| policy | 4 | 26.9933601 | 46.811 | 48.596 | 55.854 | 58.828 | 212.072 | 46.184 |
| policy | 5 | 26.9933601 | 50.663 | 47.912 | 48.004 | 58.751 | 212.877 | 56.702 |
| policy | 6 | 26.9933601 | 52.508 | 46.354 | 47.767 | 58.903 | 212.419 | 45.954 |
| policy | 7 | 26.9933601 | 46.948 | 47.377 | 48.081 | 58.675 | 211.897 | 48.387 |

PPL is deterministic within each arm for this fixed input set. The small
window/policy difference is insufficient to establish a quality change; broad
held-out and downstream evaluation remain required.

The queue also contains four failed wave21 policy records caused by an invalid
argv insertion during the first enqueue. They exited before model execution
with `--inputs` missing its value. Corrected jobs have distinct IDs and are the
eight reports archived below; the failed records remain visible in the queue
state for auditability.

Raw reports:

- [window GPU 0](results/window-model-repeat-wave21/window-gpu0.json)
- [window GPU 1](results/window-model-repeat-wave21/window-gpu1.json)
- [window GPU 2](results/window-model-repeat-wave21/window-gpu2.json)
- [window GPU 3](results/window-model-repeat-wave21/window-gpu3.json)
- [policy GPU 4](results/window-model-repeat-wave21/policy-fixed-gpu4.json)
- [policy GPU 5](results/window-model-repeat-wave21/policy-fixed-gpu5.json)
- [policy GPU 6](results/window-model-repeat-wave21/policy-fixed-gpu6.json)
- [policy GPU 7](results/window-model-repeat-wave21/policy-fixed-gpu7.json)
