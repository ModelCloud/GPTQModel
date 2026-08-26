# Llama 3.2 1B Instruct QvQ experiment results

Consolidated report of completed arms and current queued work. All canonical
D300 values below use the pinned divergence-300 development manifest
(`701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b`) and the
independent greedy 32-token protocol. `D300 Top-1` means aligned-token
agreement through horizon 32; exact counts are exact 32-token trajectories.

## Completed arms (ordered by D300 Top-1, descending)

| Arm ID | Family / arm | Rate | Eff. BPW* | Calibration | D300 Top-1 | Exact / 300 | Mean first divergence | GSM8K Platinum | Status |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| a07075 | Q07 uniform | W3 | 3.0232 | YAQA 302k, reg .10 | **33.6667%** | 16 | 9.9133 | 41.27%* | higher-rate reference |
| e616e5 | Q07 uniform | W2.5 | 2.5232 | YAQA 302k, reg .10 | **25.1771%** | 5 | 7.5000 | 33.58%* | higher-rate reference |
| 94710c | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .020 | **18.7813%** | **6** | **5.3633** | 17.87% | **current W2 leader** |
| 2d08b1 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .010 | 18.4063% | 4 | 5.1667 | pending | D300 complete; alignment arm |
| b5283c | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .015 | 17.9792% | 3 | 5.3200 | queued/recorded | D300 complete |
| 6a0ee7 | Full-reference | W2 | 2.0232 | full-reference mix, reg .30 | 17.7292% | 3 | 4.5700 | pending/ledger stale | D300 complete |
| 64876c | Seed control | W2 | 2.0232 | YAQA 302k, seed 1 | 17.7083% | 2 | 4.9667 | not run | complete |
| d64c9f | YAQA + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 17.0938% | 3 | 4.5733 | 20.68% | teacher-forced control |
| 4aa38f | D300-source-shaped | W2 | 2.0232 | source-shaped 500k, reg .20 | 17.0729% | 5 | 4.9933 | 11.91% | complete |
| 7661a7 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0225 | 16.9271% | 6 | 4.6667 | queued/recorded | D300 complete |
| 702443 | YAQA + `mlp.down_proj` W3 | W2 + W3 down | ~2.3565 | AIME mix | 16.7396% | 2 | 4.8267 | 27.79% | mixed-rate diagnostic |
| 048f31 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .005 | 16.3021% | 5 | 4.9133 | not run | complete |
| 7616fa | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .030 | 16.2188% | 5 | 4.4333 | queued/recorded | D300 complete |
| 1a6f78 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .025 | 15.3854% | 2 | 4.2667 | queued/recorded | D300 complete |
| 62ba43 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 15.1667% | 2 | 4.5733 | not run | superseded |
| 48252b | layer damping + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, layer-damped | 15.0729% | 2 | 4.6000 | pending | D300 complete |
| d7ee8e | YAQA AIME mix | W2 | 2.0232 | AIME 2526, 216 rows | 14.9792% | 0 | 3.9967 | 17.37% | complete |
| 6c2a99 | YAQA + `mlp.down_proj` W2.5 | W2 + W2.5 down | ~2.1898 | YAQA 302k, reg .020 | 14.8229% | 1 | 4.4267 | pending | mixed-rate diagnostic |
| cd6559 | Full-reference | W2 | 2.0232 | full-reference mix, reg .10 | 14.5417% | 4 | 4.3300 | pending/ledger stale | D300 complete |
| c763d1 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .020 | 14.4583% | 5 | 4.7033 | pending | D300 complete |
| f111c3 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .005 | 14.1042% | 2 | 4.5167 | not run | complete |
| 1a635f | Full-reference | W2 | 2.0232 | full-reference mix, reg .20 | 14.0313% | 4 | 4.1467 | 23.16% | complete |
| 0b4049 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0175 | 13.9271% | 1 | 4.2433 | queued/recorded | D300 complete |
| b12aa6 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0125 | 13.4375% | 2 | 3.6233 | not run | complete |

`*` Eff. BPW is logical payload rate plus the common 0.023168-bpw auxiliary
overhead. The mixed W2 + W3-down estimate assumes one-third of projection
weights use W3. GSM8K values for Q07 W2.5/W3 are reported in the existing Q07 ledger;
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

| Arm ID | Arm | Eff. BPW | GPU | State | Evaluation |
| --- | ---: | ---: | --- | --- | --- |
| 66c46d | layer damping + output alignment | 2.0232 | 2 | quantization complete; D300 pending | canonical D300 |
| 89c6b9 | YAQA reg .025 + output alignment | 2.0232 | 4 | quantization complete; D300 pending | canonical D300 |
| 4db5cf | fixed-block LDLQ reg .015 | 2.0232 | 1 | queued behind D300 | canonical D300 |
| 5ac42a | fixed-block LDLQ reg .0225 | 2.0232 | 3 | queued behind D300 | canonical D300 |
| 46d985 | clean fixed-block LDLQ reg .020 replica | 2.0232 | 6 | quantization in progress | canonical D300 |
| 37238d | clean fixed-block LDLQ reg .0225 replica | 2.0232 | 7 | quantization in progress | canonical D300 |

## Arm lookup

The Arm ID is the stable key shared by this report, the JSON ledgers, and
queued-run notes. Metrics are in the row above; configuration and checkpoint
locations are indexed here.

| Arm IDs | Configuration / checkpoint source |
| --- | --- |
| a07075, e616e5 | Q07 rate-reference ledger and checkpoints under `/root/qvq-results/`; see the Q07 experiment ledger in `docs/experiments/`. |
| 94710c, b5283c, 64876c, 7661a7, 7616fa, 1a6f78, 62ba43, f111c3, 0b4049, b12aa6 | YAQA regularization configs/checkpoints under `scripts/configs/` and `/root/qvq-results/`. |
| 6a0ee7, cd6559, 1a635f | Full-reference sweep ledger in `docs/experiments/`; checkpoints under `/root/qvq-results/`. |
| d64c9f | Alignment sweep ledger `docs/experiments/2026-08-26-llama32-w2-alignment-sweep.json`. |
| 4aa38f | Source-shaped 500k ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. |
| 702443, d7ee8e | AIME/precision ledger in `docs/experiments/`; checkpoints under `/root/qvq-results/`. |
| 048f31 | Fixed-block LDLQ ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. |
| 2d08b1 | Checkpoint `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1`; metrics `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1-div300-dev-v1.json`. |
| 66c46d, 48252b, 89c6b9 | Alignment sweep ledger `docs/experiments/2026-08-26-llama32-w2-alignment-sweep.json`. |
| c763d1, 6c2a99, 4db5cf, 5ac42a | Follow-up ledger `docs/experiments/2026-08-26-llama32-w2-fixed-down-followup.json`; configs under `scripts/configs/`. |
| 46d985 | Clean GPU6 fixed-block LDLQ reg .020 replica; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_fixed_reg020.json`; checkpoint `/root/qvq-results/llama32-1b-w2-fixed-reg020-gpu6`. |
| 37238d | Clean GPU7 fixed-block LDLQ reg .0225 replica; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_fixed_reg0225.json`; checkpoint `/root/qvq-results/llama32-1b-w2-fixed-reg0225-gpu7`. |

The authoritative per-arm machine-readable ledgers are in
`docs/experiments/`; the append-only chronology is
`docs/qvq_llama32_1b_experiment_log_2026-08-25.md`.

## Data-contamination audit

Calibration/evaluation separation is now enforced by
`scripts/check_calibration_disjointness.py`. The YAQA-182 and full-reference
292-row mixes both pass normalized user-question checks against the D300
development split and all 1,209 GSM8K Platinum test rows; D300 and GSM8K also
have no normalized collisions. Manifests: `docs/experiments/disjointness-yaqa182.json`
and `docs/experiments/disjointness-full-reference.json`. Quantization can be
made fail-closed with `--disjointness-manifest`; historical full-reference
GSM8K scores remain marked as contamination-risk until semantic/source audits
of derived datasets (for example OpenMathInstruct) are complete.
