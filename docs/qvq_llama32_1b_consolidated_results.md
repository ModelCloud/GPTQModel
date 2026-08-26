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
| 94710c | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .020 | **18.7813%** | **6** | **5.3633** | 22.08% | **current W2 leader** |
| 2d08b1 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .010 | 18.4063% | 4 | 5.1667 | 21.67% | D300 complete; alignment arm |
| b5283c | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .015 | 17.9792% | 3 | 5.3200 | 23.82% | D300 complete |
| 6a0ee7 | Full-reference | W2 | 2.0232 | full-reference mix, reg .30 | 17.7292% | 3 | 4.5700 | 19.60% | D300 complete |
| 64876c | Seed control | W2 | 2.0232 | YAQA 302k, seed 1 | 17.7083% | 2 | 4.9667 | not run | complete |
| d64c9f | YAQA + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 17.0938% | 3 | 4.5733 | 22.75% | teacher-forced control |
| 7661a7 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0225 | 16.9271% | 6 | 4.6667 | 21.26% | D300 complete |
| 702443 | YAQA + `mlp.down_proj` W3 | W2 + W3 down | ~2.3565 | AIME mix | 16.7396% | 2 | 4.8267 | 27.79% | mixed-rate diagnostic |
| 048f31 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .005 | 16.3021% | 5 | 4.9133 | 20.43% | complete |
| 7616fa | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .030 | 16.2188% | 5 | 4.4333 | 19.11% | D300 complete |
| 1a6f78 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .025 | 15.3854% | 2 | 4.2667 | 20.60% | D300 complete |
| 62ba43 | YAQA + output alignment | W2 | 2.0232 | YAQA 302k, reg .05/.10 layers | 15.1667% | 2 | 4.5733 | not run | superseded |
| 48252b | layer damping + 2-epoch alignment | W2 | 2.0232 | YAQA 302k, layer-damped | 15.0729% | 2 | 4.6000 | 22.75% | D300 complete |
| d7ee8e | YAQA AIME mix | W2 | 2.0232 | AIME 2526, 216 rows | 14.9792% | 0 | 3.9967 | 17.37% | complete |
| 6c2a99 | YAQA + `mlp.down_proj` W2.5 | W2 + W2.5 down | ~2.1898 | YAQA 302k, reg .020 | 14.8229% | 1 | 4.4267 | 26.88% | mixed-rate diagnostic |
| cd6559 | Full-reference | W2 | 2.0232 | full-reference mix, reg .10 | 14.5417% | 4 | 4.3300 | 21.42% | D300 complete |
| c763d1 | Fixed-block LDLQ | W2 | 2.0232 | YAQA 302k, reg .020 | 14.4583% | 5 | 4.7033 | 23.41% | D300 complete |
| f111c3 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .005 | 14.1042% | 2 | 4.5167 | 20.43% | complete |
| 1a635f | Full-reference | W2 | 2.0232 | full-reference mix, reg .20 | 14.0313% | 4 | 4.1467 | 23.33% | reverified complete |
| 0b4049 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0175 | 13.9271% | 1 | 4.2433 | 19.19% | D300 complete |
| b12aa6 | YAQA regularization | W2 | 2.0232 | YAQA 302k, reg .0125 | 13.4375% | 2 | 3.6233 | 22.50% | complete |

`*` Eff. BPW is logical payload rate plus the common 0.023168-bpw auxiliary
overhead. The mixed W2 + W3-down estimate assumes one-third of projection
weights use W3. GSM8K values for Q07 W2.5/W3 are reported in the existing Q07 ledger;
they are not flat-W2 results. The W2.5/W3 rows are included for rate/fidelity
context only and do not count toward the flat-W2 target.

### Newly completed review follow-ups

| Artifact | Configuration | D300 token top-1 | Exact / 300 | GSM8K Platinum | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | YAQA reselect reg 0.15 + output alignment | 14.9271% | 5 | 22.9942% (278/1209) | complete |
| `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | YAQA reselect reg 0.15 + `mlp.down_proj` W2.5 | 17.8021% | 3 | 23.4078% (283/1209) | complete |

## Invalidated arms

These arms produced numerical results but should not be used for leaderboard
or comparison purposes because the calibration artifact failed the strict
row-level contamination preflight.

| Arm ID | Family / arm | Rate | Eff. BPW* | Calibration | D300 Top-1 | Exact / 300 | Mean first divergence | GSM8K Platinum | Status |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 4aa38f | D300-source-shaped | W2 | 2.0232 | source-shaped 500k, reg .20 | 17.0729% | 5 | 4.9933 | 11.91% | **invalid / contaminated** — calibration artifact `calibration_div300_sources.parquet` failed the disjointness audit (1 normalized D300 overlap, 33 internal duplicate groups) |

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

The live monitor also has three replay quantizations active at the time of
this snapshot (2026-08-26 UTC), with evaluator wrappers already reserved for
each output:

| Artifact ID | Active checkpoint | Quantization | Evaluation queue |
| --- | --- | --- | --- |
| `f6c9e1` | `llama32-1b-w2-reg020-all-subset-replay-disjoint-tip-gpu5` | active (GPU 0) | D300 pending behind quantization |
| `d3eac8` | `llama32-1b-w2-reg020-replay-aggressive-tip-gpu3` | active (GPU 0) | D300 pending behind quantization |
| `a9a4df` | `llama32-1b-w2-reg020-replay-256x256-tip-gpu2` | active (GPU 2) | D300 pending behind quantization |

These rows are intentionally separate from the completed-arm leaderboard:
they have no `qvq_quantize_run.json` completion marker yet. The monitor state
file (`docs/experiments/qvq_eval_monitor_state.json`) is the source of truth
for transitions from active/queued to complete/failed, while the artifact
inventory above records the corresponding filesystem state.

## Complete artifact inventory (live reconciliation)

This inventory is generated from `/root/qvq-results/llama32-1b*` checkpoint
directories and is intended to include every quantization artifact, including
experiments that have not yet produced canonical evaluations. `active` means a
quantizer or evaluator process currently references the artifact; `pending`
means no result file has been observed yet. The six-character path keys are
stable local cross-reference IDs for artifacts that do not yet have a ledger Arm
ID.

| Artifact ID | Checkpoint / experiment | Quant | D300 | GSM8K Platinum |
| --- | --- | --- | --- | --- |
| `ff5436` | `llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1` | completed | completed | completed |
| `107daa` | `llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1` | completed | completed | completed |
| `20f598` | `llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365` | completed | pending | completed |
| `0a0dc5` | `llama32-1b-v2b2p32-yaqa322k-fixed-reg005-pr34-bb1c250e` | completed | pending | completed |
| `f303b2` | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn005-mlp010` | completed | pending | completed |
| `cdeee7` | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn010-mlp005` | completed | pending | completed |
| `4fefa2` | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5` | completed | pending | completed |
| `043c3f` | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da` | completed | pending | completed |
| `bd20aa` | `llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32` | completed | pending | completed |
| `3591fa` | `llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e` | completed | pending | completed |
| `3c871e` | `llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365` | completed | pending | completed |
| `1dd616` | `llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365` | completed | pending | completed |
| `9536f5` | `llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365` | completed | pending | completed |
| `de750c` | `llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e` | completed | pending | completed |
| `c80256` | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e` | completed | pending | completed |
| `5072f6` | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e` | completed | pending | completed |
| `1c69af` | `llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a` | completed | pending | completed |
| `83093c` | `llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e` | completed | pending | completed |
| `0f21eb` | `llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e` | completed | pending | completed |
| `659ec3` | `llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e` | completed | pending | completed |
| `b8fde6` | `llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25` | completed | pending | completed |
| `7e67aa` | `llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e` | completed | pending | completed |
| `3c7eab` | `llama32-1b-v2b2p32-yaqa322k-rolehybrid-qk010-rest005-dev32` | completed | pending | completed |
| `4c7cf9` | `llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5` | completed | completed | completed |
| `14ef78` | `llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5` | completed | completed | completed |
| `acb522` | `llama32-1b-w2-aime2526-mlpdown-w3-main-fdfd760e` | completed | completed | completed |
| `d65704` | `llama32-1b-w2-aime2526-yaqa-mix-main-1f6c2132` | completed | completed | completed |
| `fa6724` | `llama32-1b-w2-div300-sources-500k-mlpdown-w3-main-4f0d767e` | completed | completed | completed |
| `838950` | `llama32-1b-w2-div300-sources-500k-reg010-main-41106a47` | completed | completed | completed |
| `d9937c` | `llama32-1b-w2-div300-sources-500k-reg020-main-5544143c` | completed | completed | completed |
| `17a515` | `llama32-1b-w2-div300-sources-500k-reg030-main-41106a47` | completed | completed | completed |
| `a9ce03` | `llama32-1b-w2-fixed-reg015-gpu1` | completed | completed | completed |
| `375733` | `llama32-1b-w2-fixed-reg020-gpu2` | completed | completed | completed |
| `1f01e7` | `llama32-1b-w2-fixed-reg020-gpu6` | completed | completed | completed |
| `51bc8a` | `llama32-1b-w2-fixed-reg020-tip-gpu0` | completed | completed | completed |
| `bbcf3e` | `llama32-1b-w2-fixed-reg0225-gpu3` | completed | completed | completed |
| `bb21bc` | `llama32-1b-w2-fixed-reg0225-gpu7` | completed | completed | completed |
| `6ae421` | `llama32-1b-w2-fixedreg005-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `aa2c5f` | `llama32-1b-w2-full-reference-reg010-main-5544143c` | completed | completed | completed |
| `83261e` | `llama32-1b-w2-full-reference-reg020-main-5544143c` | completed | completed | completed |
| `2d8dc3` | `llama32-1b-w2-full-reference-reg030-main-5544143c` | completed | completed | completed |
| `e0344c` | `llama32-1b-w2-mlp-down-w25-reg020-gpu4` | completed | completed | completed |
| `02c72f` | `llama32-1b-w2-reg0025-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `3ad097` | `llama32-1b-w2-reg005-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `39bff7` | `llama32-1b-w2-reg0125-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `c15676` | `llama32-1b-w2-reg020-replay-256x256-tip-gpu6` | active | pending | pending |
| `c7ef00` | `llama32-1b-w2-reg020-replay-256x256-tip-gpu7` | active | pending | pending |
| `2c5e38` | `llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `c86bfc` | `llama32-1b-w2-seed1-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `20bc50` | `llama32-1b-w2-shiftcal128-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `899803` | `llama32-1b-w2-spectral-yaqa302k-main-a12ce4e3` | completed | completed | completed |
| `169f36` | `llama32-1b-w2-yaqa-cal322k-main-de25f203` | completed | completed | completed |
| `635a92` | `llama32-1b-w2-yaqa-main-de25f203` | completed | completed | completed |
| `a0ac7a` | `llama32-1b-w2-yaqa-optmix322k-main-de25f203` | completed | completed | completed |
| `97c8a8` | `llama32-1b-w2-yaqa-optmix551k-main-54ccd365` | completed | completed | completed |
| `05ede7` | `llama32-1b-w2-yaqa302k-layerdamp-align-gpu2` | completed | pending | completed |
| `1f126d` | `llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3` | completed | completed | completed |
| `61047f` | `llama32-1b-w2-yaqa302k-reg010-align-gpu1` | completed | completed | completed |
| `6491a4` | `llama32-1b-w2-yaqa302k-reg015-main-5cfb8314` | completed | completed | completed |
| `872b3e` | `llama32-1b-w2-yaqa302k-reg0175-main-5cfb8314` | completed | completed | completed |
| `0cb22d` | `llama32-1b-w2-yaqa302k-reg020-align-main-ff906097` | completed | completed | completed |
| `acc5d3` | `llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc` | completed | completed | completed |
| `45b81b` | `llama32-1b-w2-yaqa302k-reg0225-main-5cfb8314` | completed | completed | completed |
| `315984` | `llama32-1b-w2-yaqa302k-reg025-align-gpu4` | completed | pending | completed |
| `43db13` | `llama32-1b-w2-yaqa302k-reg025-main-5cfb8314` | completed | completed | completed |
| `603c9a` | `llama32-1b-w2-yaqa302k-reg030-main-5cfb8314` | completed | completed | completed |

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
| 4aa38f | Source-shaped 500k ledger in `docs/experiments/`; checkpoint under `/root/qvq-results/`. **Invalidated**: the 993-row `calibration_div300_sources.parquet` artifact used by this arm failed the row-level disjointness audit and is not eligible for quantization/evaluation. |
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
GSM8K scores pass the available row-level audit and are valid on that evidence.
Semantic similarity from derived datasets (for example OpenMathInstruct) is a
separate, optional source-level audit and is not evidence of contamination.

The first D300-source-shaped 500k artifact failed this audit: calibration row
165 matched D300 row 212 after normalization, and 33 additional duplicate
groups were found within the candidate mix. That artifact is not eligible for
quantization/evaluation. A filtered artifact retaining 959 unique rows passes
the strict audit at
`docs/experiments/disjointness-div300-sources-disjoint.json`.
Arm `4aa38f`, which used the ineligible 993-row artifact, is therefore listed
in the "Invalidated arms" section above and should not be compared as a
completed flat-W2 result.
