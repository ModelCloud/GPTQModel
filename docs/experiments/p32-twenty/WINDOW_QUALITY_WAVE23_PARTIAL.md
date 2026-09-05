# Window quality wave 23 partial results

The first two production-window ARC-Challenge repeats completed from the
wave23 downstream scorecard. Both used the exact F6 seed-7 snapshot and the
same 1,172-example test split.

| Arm | Host GPU | ARC raw | ARC length-normalized |
|---|---:|---:|---:|
| production window | 0 | 0.3327645 (390/1172) | 0.3651877 (428/1172) |
| production window | 1 | 0.3327645 (390/1172) | 0.3651877 (428/1172) |

The first two production-window GSM8K repeats also agree:

| Arm | Host GPU | GSM8K |
|---|---:|---:|
| production window | 4 | 0.4609375 (59/128) |
| production window | 5 | 0.4609375 (59/128) |

The completed repeats are identical within each task. The two policy ARC
repeats and two policy GSM8K repeats are still running and will be added after
their reports complete.

Raw reports:

- [window ARC GPU 0 report](results/window-quality-wave23/window-arc-gpu0.json)
- [window ARC GPU 0 task results](results/window-quality-wave23/window-arc-gpu0-task-results.json)
- [window ARC GPU 1 report](results/window-quality-wave23/window-arc-gpu1.json)
- [window ARC GPU 1 task results](results/window-quality-wave23/window-arc-gpu1-task-results.json)
- [window GSM8K GPU 4 report](results/window-quality-wave23/window-gsm-gpu4.json)
- [window GSM8K GPU 4 task results](results/window-quality-wave23/window-gsm-gpu4-task-results.json)
- [window GSM8K GPU 5 report](results/window-quality-wave23/window-gsm-gpu5.json)
- [window GSM8K GPU 5 task results](results/window-quality-wave23/window-gsm-gpu5-task-results.json)
