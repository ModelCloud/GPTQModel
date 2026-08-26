# Llama 3.2 1B Instruct QvQ experiment results

Consolidated report of completed arms and current queued work. All canonical
D300 values below use the pinned divergence-300 development manifest
(`701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b`) and the
independent greedy 32-token protocol. `D300 Top-1` means aligned-token
agreement through horizon 32; exact counts are exact 32-token trajectories.

## Completed arms (ordered by D300 Top-1, descending)

| Arm ID | Family / arm | Rate | Eff. BPW* | Calibration | D300 Top-1 | Exact / 300 | Mean first divergence | GSM8K Platinum | Status |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| C01 | Q07 uniform | W3 | 3.0232 | YAQA 302k, reg .10 | **33.6667%** | 16 | 9.9133 | 41.27%* | higher-rate reference |
| C02 | Q07 uniform | W2.5 | 2.5232 | YAQA 302k, reg .10 | **25.1771%** | 5 | 7.5000 | 33.58%* | higher-rate reference |
| C03 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .020 | **18.7813%** | **6** | **5.3633** | 17.87% | **current W2 leader** |
| P01 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .010 | 18.4063% | 4 | 5.1667 | pending | D300 complete; alignment arm |
| C04 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .015 | 17.9792% | 3 | 5.3200 | queued/recorded | D300 complete |
| C05 | Full-reference | W2 | 2.0232 | full-reference mix, reg .30 | 17.7292% | 3 | 4.5700 | pending/ledger stale | D300 complete |
| C06 | Seed control | W2 | 2.0232 | YAQA 302k, seed 1 | 17.7083% | 2 | 4.9667 | not run | complete |
| C07 | YAQA + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 17.0938% | 3 | 4.5733 | 20.68% | teacher-forced control |
| C08 | D300-source-shaped | W2 | 2.0232 | source-shaped 500k, reg .20 | 17.0729% | 5 | 4.9933 | 11.91% | complete |
| C09 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0225 | 16.9271% | 6 | 4.6667 | queued/recorded | D300 complete |
| C10 | YAQA + `mlp.down_proj` W3 | W2 + W3 down | ~2.3565 | AIME mix | 16.7396% | 2 | 4.8267 | 27.79% | mixed-rate diagnostic |
| C11 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .005 | 16.3021% | 5 | 4.9133 | not run | complete |
| C12 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .030 | 16.2188% | 5 | 4.4333 | queued/recorded | D300 complete |
| C13 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .025 | 15.3854% | 2 | 4.2667 | queued/recorded | D300 complete |
| C14 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 15.1667% | 2 | 4.5733 | not run | superseded |
| C15 | YAQA AIME mix | W2 | 2.0232 | AIME 2526, 216 rows | 14.9792% | 0 | 3.9967 | 17.37% | complete |
| C16 | Full-reference | W2 | 2.0232 | full-reference mix, reg .10 | 14.5417% | 4 | 4.3300 | pending/ledger stale | D300 complete |
| C17 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .005 | 14.1042% | 2 | 4.5167 | not run | complete |
| C18 | Full-reference | W2 | 2.0232 | full-reference mix, reg .20 | 14.0313% | 4 | 4.1467 | 23.16% | complete |
| C19 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0175 | 13.9271% | 1 | 4.2433 | queued/recorded | D300 complete |
| C20 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0125 | 13.4375% | 2 | 3.6233 | not run | complete |

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
| P02 | layer damping + output alignment | 2.0232 | 2 | quantization complete; D300 pending | canonical D300 |
| P03 | layer damping + 2-epoch alignment | 2.0232 | 3 | D300 evaluation in progress | canonical D300 |
| P04 | YAQA reg .025 + output alignment | 2.0232 | 4 | quantization complete; D300 pending | canonical D300 |
| P05 | fixed-block LDLQ reg .020 | 2.0232 | 2 | quantization complete; D300 pending | canonical D300 |
| P06 | `mlp.down_proj` W2.5 reg .020 | ~2.5232 | 4 | quantization in progress | canonical D300 |
| P07 | fixed-block LDLQ reg .015 | 2.0232 | 1 | queued behind D300 | canonical D300 |
| P08 | fixed-block LDLQ reg .0225 | 2.0232 | 3 | queued behind D300 | canonical D300 |

## Arm lookup

The Arm ID is the stable key shared by this report, the JSON ledgers, and
queued-run notes. Metrics are in the row above; configuration and checkpoint
locations are indexed here.

| Arm IDs | Configuration / checkpoint source |
| --- | --- |
| C01--C02 | Q07 rate-reference ledger and checkpoints under `/root/qvq-results/`; see the Q07 experiment ledger in `docs/experiments/`. |
| C03--C04, C06, C09, C12--C14, C17, C19--C20 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020.json` (or the corresponding reg value); YAQA checkpoints under `/root/qvq-results/`. |
| C05, C16, C18 | Full-reference sweep ledger in `docs/experiments/`; checkpoints under `/root/qvq-results/`. |
| C07 | Alignment sweep ledger `docs/experiments/2026-08-26-llama32-w2-alignment-sweep.json`. |
| C08 | Source-shaped 500k ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. |
| C10, C15 | AIME/precision ledger in `docs/experiments/`; checkpoints under `/root/qvq-results/`. |
| C11 | Fixed-block LDLQ ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. |
| P01 | Checkpoint `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1`; metrics `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1-div300-dev-v1.json`. |
| P02--P04 | Alignment sweep ledger `docs/experiments/2026-08-26-llama32-w2-alignment-sweep.json`. |
| P05--P08 | Follow-up ledger `docs/experiments/2026-08-26-llama32-w2-fixed-down-followup.json`; configs `scripts/configs/llama32_1b_v2b2_p32_yaqa_fixed_reg015.json`, `...reg020.json`, `...reg0225.json`, and `...yaqa_mlp_down_w25_reg020.json`. |

The authoritative per-arm machine-readable ledgers are in
`docs/experiments/`; the append-only chronology is
`docs/qvq_llama32_1b_experiment_log_2026-08-25.md`.
