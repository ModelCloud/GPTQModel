# QVQ held-out evaluation monitor

`scripts/qvq_eval_monitor.py` maintains a durable queue for missing GSM8K
Platinum and D300 evaluations. It discovers checkpoints with a completed
`qvq_quantize_run.json`, excludes the audited-contaminated `div300-sources-500k`
artifacts, detects evaluators already running, and assigns new jobs only to
GPUs whose utilization is below 5% and memory use below 2 GB. Results are
written atomically by the evaluator and then appended to the experiment ledger
with the report path and metric. State is kept in
`docs/experiments/qvq_eval_monitor_state.json`; a lock prevents duplicate
monitors. Failed jobs are retried up to three times.

Run continuously across all GPUs:

```bash
python scripts/qvq_eval_monitor.py --interval 30 --gpus 0,1,2,3,4,5,6,7
```

Use `--once --dry-run` to inspect scheduling without starting a process.
GSM8K calibration/evaluation data remain separate from D300 data; the
contaminated D300-source-shaped calibration artifact is never scheduled.
