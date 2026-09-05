# Window quality wave 23 partial results

The first two production-window ARC-Challenge repeats completed from the
wave23 downstream scorecard. Both used the exact F6 seed-7 snapshot and the
same 1,172-example test split.

| Arm | Host GPU | ARC raw | ARC length-normalized |
|---|---:|---:|---:|
| production window | 0 | 0.3327645 (390/1172) | 0.3651877 (428/1172) |
| production window | 1 | 0.3327645 (390/1172) | 0.3651877 (428/1172) |

The two repeats are identical. The two policy ARC repeats and four GSM8K
repeats are still running and will be added after their reports complete.

Raw reports:

- [window ARC GPU 0 report](results/window-quality-wave23/window-arc-gpu0.json)
- [window ARC GPU 0 task results](results/window-quality-wave23/window-arc-gpu0-task-results.json)
- [window ARC GPU 1 report](results/window-quality-wave23/window-arc-gpu1.json)
- [window ARC GPU 1 task results](results/window-quality-wave23/window-arc-gpu1-task-results.json)
