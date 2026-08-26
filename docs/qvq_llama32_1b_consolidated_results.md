# Llama 3.2 1B Instruct QvQ experiment results

Consolidated report of completed arms and current queued work. All canonical
D300 values below use the pinned divergence-300 development manifest
(`701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b`) and the
independent greedy 32-token protocol. `D300 Top-1` means aligned-token
agreement through horizon 32; exact counts are exact 32-token trajectories.

## Completed arms (ordered by D300 Top-1, descending)

| Family / arm | Rate | Calibration | D300 Top-1 | Exact / 300 | Mean first divergence | GSM8K Platinum | Status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Q07 uniform | W3 | YAQA 302k, reg .10 | **33.6667%** | 16 | 9.9133 | 41.27%* | higher-rate reference |
| Q07 uniform | W2.5 | YAQA 302k, reg .10 | **25.1771%** | 5 | 7.5000 | 33.58%* | higher-rate reference |
| YAQA regularization | W2 | YAQA 302k, reg .020 | **18.7813%** | **6** | **5.3633** | 17.87% | **current W2 leader** |
| YAQA regularization | W2 | YAQA 302k, reg .015 | 17.9792% | 3 | 5.3200 | queued/recorded | D300 complete |
| Full-reference | W2 | full-reference mix, reg .30 | 17.7292% | 3 | 4.5700 | pending/ledger stale | D300 complete |
| Seed control | W2 | YAQA 302k, seed 1 | 17.7083% | 2 | 4.9667 | not run | complete |
| YAQA + 2-epoch alignment | W2 | YAQA 302k, reg .05/.10 layers | 17.0938% | 3 | 4.5733 | 20.68% | teacher-forced control |
| D300-source-shaped | W2 | source-shaped 500k, reg .20 | 17.0729% | 5 | 4.9933 | 11.91% | complete |
| YAQA regularization | W2 | YAQA 302k, reg .0225 | 16.9271% | 6 | 4.6667 | queued/recorded | D300 complete |
| YAQA + `mlp.down_proj` W3 | W2 + W3 down | AIME mix | 16.7396% | 2 | 4.8267 | 27.79% | mixed-rate diagnostic |
| Fixed-block LDLQ | W2 | YAQA 302k, reg .005 | 16.3021% | 5 | 4.9133 | not run | complete |
| YAQA regularization | W2 | YAQA 302k, reg .030 | 16.2188% | 5 | 4.4333 | queued/recorded | D300 complete |
| YAQA regularization | W2 | YAQA 302k, reg .025 | 15.3854% | 2 | 4.2667 | queued/recorded | D300 complete |
| YAQA + output alignment | W2 | YAQA 302k, reg .05/.10 layers | 15.1667% | 2 | 4.5733 | not run | superseded |
| YAQA AIME mix | W2 | AIME 2526, 216 rows | 14.9792% | 0 | 3.9967 | 17.37% | complete |
| Full-reference | W2 | full-reference mix, reg .10 | 14.5417% | 4 | 4.3300 | pending/ledger stale | D300 complete |
| YAQA regularization | W2 | YAQA 302k, reg .005 | 14.1042% | 2 | 4.5167 | not run | complete |
| Full-reference | W2 | full-reference mix, reg .20 | 14.0313% | 4 | 4.1467 | 23.16% | complete |
| YAQA regularization | W2 | YAQA 302k, reg .0175 | 13.9271% | 1 | 4.2433 | queued/recorded | D300 complete |
| YAQA regularization | W2 | YAQA 302k, reg .0125 | 13.4375% | 2 | 3.6233 | not run | complete |

`*` GSM8K values for Q07 W2.5/W3 are reported in the existing Q07 ledger;
they are not flat-W2 results. The W2.5/W3 rows are included for rate/fidelity
context only and do not count toward the flat-W2 target.

## Current leader and protocol

The flat-W2 leader is checkpoint
`/root/qvq-results/llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3`, using
`scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020.json`, ordinary NM rows 0--127,
and YAQA rows 0--181 (302,193 valid tokens). Its canonical D300 score is
18.7813% (1,803/9,600), with 6/300 exact trajectories and mean first
divergence token 5.3633. Its full GSM8K Platinum score is 17.8660%.

## In progress / queued

| Arm | GPU | State | Evaluation |
| --- | ---: | --- | --- |
| YAQA reg .010 + output alignment | 1 | D300 evaluation in progress | canonical D300 |
| layer damping + output alignment | 2 | quantization complete; D300 pending | canonical D300 |
| layer damping + 2-epoch alignment | 3 | D300 evaluation in progress | canonical D300 |
| YAQA reg .025 + output alignment | 4 | quantization complete; D300 pending | canonical D300 |
| fixed-block LDLQ reg .020 | 2 | quantization complete; D300 pending | canonical D300 |
| `mlp.down_proj` W2.5 reg .020 | 4 | quantization in progress | canonical D300 |
| fixed-block LDLQ reg .015 | 1 | queued behind D300 | canonical D300 |
| fixed-block LDLQ reg .0225 | 3 | queued behind D300 | canonical D300 |

The authoritative per-arm machine-readable ledgers are in
`docs/experiments/`; the append-only chronology is
`docs/qvq_llama32_1b_experiment_log_2026-08-25.md`.
