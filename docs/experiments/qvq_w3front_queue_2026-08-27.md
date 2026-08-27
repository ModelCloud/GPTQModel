# W3-frontier allocation queue — 2026-08-27

All eight arms use Llama 3.2 1B Instruct, `qvq_v2b2_p32`, YAQA/reselect,
the same 128 NM + 182 YAQA calibration rows, and the passing strict
`disjointness-llama32-benchmark-replay-v2.json` contract.  They are plain
precision-allocation controls (no module replay); the queue wrapper omits
replay streams unless the config explicitly enables them.

| Arm ID | Physical GPU | Configuration | Est. effective BPW | Queue state |
| --- | ---: | --- | ---: | --- |
| `3126a8` | 0 | QK2.5 + VO3.5 + Gate2.5 + Up3.5 + Down3, reg .15 | 3.0232 | quantizing |
| `d20085` | 1 | Same allocation, reg .10 | 3.0232 | quantizing |
| `f9a5ad` | 2 | Same allocation, reg .20 | 3.0232 | quantizing |
| `365c2d` | 3 | QK1.5 + VO3.5 + Gate3 + Up3.5 + Down3, reg .15 | 3.0749 | quantizing |
| `a96ce0` | 4 | QK1.0 + VO3.5 + Gate3 + Up3.5 + Down3, reg .15 | 3.0318 | quantizing |
| `c2fbf3` | 5 | QK2.5 + VO3.5 + Gate2 + Up3.5 + Down3.5, reg .15 | 3.0232 | quantizing |
| `032830` | 6 | QK2.5 + VO3.5 + MLP3; Up3.5 in layers 8–15, reg .15 | 3.0922 | quantizing |
| `bc35b5` | 7 | QK2.5 + VO3.5 + MLP3; Up3.5 in layers 0–7, reg .15 | 3.0922 | quantizing |

After each checkpoint is saved, the durable evaluator monitor schedules Mini-GSM
fast metrics, GSM8K Platinum, and canonical D300, in that order.  Results are
recorded in the append-only experiment log and immutable per-checkpoint JSON
reports.
