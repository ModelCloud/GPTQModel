# Window quality wave 23

This completed downstream comparison used the exact F6 seed-7 snapshot and the
same benchmark inputs for production window and per-module fused promotion
policy. Each arm was repeated twice on separate GPUs. ARC-Challenge used all
1,172 test examples; GSM8K used the 128-example slice.

| Arm | Repeats | ARC raw | ARC length-normalized | GSM8K |
|---|---:|---:|---:|---:|
| production window | 2 | 390/1172 | 428/1172 | 59/128 |
| fused policy | 2 | 390/1172 | 429/1172 | 58/128 |

The repeated task outputs agree within each arm. The policy changes one
length-normalized ARC answer and loses one GSM8K answer on this run. These
differences do not establish a post-quant quality advantage. The policy timing
result remains useful only for the regimes recorded in
[window model repeat wave21](WINDOW_MODEL_REPEAT_WAVE21.md): it is slower at
small M and decode, and reaches 1.366x at M=2048 in the pooled repeat.

Raw reports and evaluator outputs:

- [window ARC GPU 0 report](results/window-quality-wave23/window-arc-gpu0.json)
- [window ARC GPU 0 evaluator output](results/window-quality-wave23/window-arc-gpu0-task-results.json)
- [window ARC GPU 1 report](results/window-quality-wave23/window-arc-gpu1.json)
- [window ARC GPU 1 evaluator output](results/window-quality-wave23/window-arc-gpu1-task-results.json)
- [policy ARC GPU 2 report](results/window-quality-wave23/policy-arc-gpu2.json)
- [policy ARC GPU 2 evaluator output](results/window-quality-wave23/policy-arc-gpu2-task-results.json)
- [policy ARC GPU 3 report](results/window-quality-wave23/policy-arc-gpu3.json)
- [policy ARC GPU 3 evaluator output](results/window-quality-wave23/policy-arc-gpu3-task-results.json)
- [window GSM8K GPU 4 report](results/window-quality-wave23/window-gsm-gpu4.json)
- [window GSM8K GPU 4 evaluator output](results/window-quality-wave23/window-gsm-gpu4-task-results.json)
- [window GSM8K GPU 5 report](results/window-quality-wave23/window-gsm-gpu5.json)
- [window GSM8K GPU 5 evaluator output](results/window-quality-wave23/window-gsm-gpu5-task-results.json)
- [policy GSM8K GPU 6 report](results/window-quality-wave23/policy-gsm-gpu6.json)
- [policy GSM8K GPU 6 evaluator output](results/window-quality-wave23/policy-gsm-gpu6-task-results.json)
- [policy GSM8K GPU 7 report](results/window-quality-wave23/policy-gsm-gpu7.json)
- [policy GSM8K GPU 7 evaluator output](results/window-quality-wave23/policy-gsm-gpu7-task-results.json)
