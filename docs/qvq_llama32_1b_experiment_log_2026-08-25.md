# Llama 3.2 1B QVQ experiment ledger

This is the append-only ledger for the flat-W2 Llama 3.2 1B Instruct accuracy-recovery campaign. The analysis and
metric definitions live in [the fidelity audit](qvq_llama32_1b_divergence32_2026-08-25.md); this file records what
was actually run, including negative and invalid results.

## Logging contract

Every attempted quantization, alignment, evaluation, and relevant test run must be added here before its result is
used to choose another arm. Each entry records:

- status (`running`, `completed`, `rejected`, `failed`, `invalid`, or `superseded`) and the selection decision;
- source commit, dense model, exact input paths/slices, tracked config, and full command-line overrides;
- checkpoint and report paths, even when a candidate is rejected;
- all primary results, not only the metric that improved;
- failures, stale-artifact discoveries, and protocol corrections without deleting the original record.

An evaluation is not attributable to a quantization merely because its filename looks right. Before recording it,
the evaluator checkpoint must equal the intended checkpoint, the checkpoint must contain `qvq_quantize_run.json`,
and the serialized run config/data provenance must match this ledger. A repeated score is treated as suspicious until
those identities are checked. Local result paths are retained for auditability; they are not claimed to be stored in
Git.

## Frozen data and evaluation protocol

These values apply to every quantization arm below unless an entry explicitly replaces them.

| Field | Resolved value |
| --- | --- |
| Dense model | `/monster/data/model/Llama-3.2-1B-Instruct` |
| Ordinary lifecycle calibration | `/monster/data/model/dataset/nm-calibration/llm.parquet`, rows `0..127` (128 rows) |
| YAQA Sketch-B data | `/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet`, rows `0..181` (182 independent sequences) |
| YAQA valid output samples | 302,193 in the quantization manifests |
| Quantization batch/concat/sort | batch `1`, concat `0`, sort `desc` |
| Quantization device | `cuda:0` |
| Development rollout manifest | `/root/qvq-data/divergence300-v1/divergence300-development.jsonl` |
| Manifest identity | SHA-256 `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2`; 300 prompts; 889,765 prompt tokens; 49 prompts capped |
| Manifest sources | 40 Terminal-Bench 2.1, 100 SWE-bench Verified, 60 MathArena 2025/26, 50 non-English Multi-IF, 50 LongBench v2 |
| Prompt rendering | Llama native chat template, `add_generation_prompt=True`, same encoded prompt for both models, left truncation only above 16,384 tokens |
| Rollout | independent greedy argmax, 32 new tokens, FP16, SDPA, no sampling |
| Primary metric | aligned token-ID matches across all 9,600 independently generated positions |
| Secondary trajectory metric | exact 32-token trajectories out of 300 and the complete survival curve |
| Locked ordinary evaluation | `nm-calibration/llm.parquet`, rows `512..811`; never used for candidate selection |
| Locked Divergence manifest | `/root/qvq-data/divergence300-v1/divergence300-locked.jsonl`; untouched during this campaign |

The common quantization invocation is fully resolved by substituting the entry's `CONFIG`, `OUTPUT`, and `COMMIT`:

```bash
python scripts/qvq_quantize.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output OUTPUT \
  --quant-config CONFIG \
  --calibration-dataset /monster/data/model/dataset/nm-calibration/llm.parquet \
  --calibration-row-start 0 --calibration-rows 128 \
  --yaqa-dataset /root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet \
  --yaqa-row-start 0 --yaqa-rows 182 \
  --batch-size 1 --concat-size 0 --calibration-sort desc --device cuda:0 \
  --no-qvq-telemetry
```

The common corrected rollout invocation is fully resolved by substituting `CHECKPOINT` and `REPORT`:

```bash
python scripts/qvq_evaluate.py divergence300 \
  --dense-model /monster/data/model/Llama-3.2-1B-Instruct \
  --checkpoint CHECKPOINT \
  --dataset /root/qvq-data/divergence300-v1/divergence300-development.jsonl \
  --device cuda:0 --output REPORT --max-prompt-tokens 16384 \
  --dtype float16 --attn-implementation sdpa
```

## Full quantization configuration matrix

All listed JSON files are complete tracked configurations, not fragments. Common fields in every flat-W2 file are
`bits=2`, `format=qvq_v2b2_p32`, `bank_count=2`, `rounding=yaqa`, `device=cuda:0`,
`offload_to_disk=false`, YAQA `minimum_sequences=182`, `batch_size=1`, `sequence_sort=desc`,
`activation_checkpointing=true`, `v2b2_family_mode=reselect`, and `sample_strategy=full`. The table enumerates every
remaining field, so the table plus those common fields is also a complete resolved config.

| ID | Complete tracked config | Seed | Base regularization | W2 rate regularization | Dynamic/module override | Chat weighting | Output alignment / other |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Q01 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_layer_damping.json` | 0 | .05 | .05 | layers 0/6/10/12: `.10` | disabled | none |
| Q02 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_layer_damping_align.json` | 0 | .05 | .05 | layers 0/6/10/12: `.10` | disabled | Adam, LR 1e-5, 1 epoch, 32 train / 16 val batches, val .2, pristine Hessian |
| Q03 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_layer_damping_align2e64.json` | 0 | .05 | .05 | layers 0/6/10/12: `.10` | disabled | Adam, LR 1e-5, 2 epochs, 64 train / 24 val batches, val .2, pristine Hessian |
| Q04 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg0025.json` | 0 | .025 | .025 | none | disabled | none |
| Q05 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg005.json` | 0 | .05 | .05 | none | disabled | none |
| Q06 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg005_align.json` | 0 | .05 | .05 | none | disabled | same 1-epoch alignment as Q02 |
| Q07 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010.json` | 0 | .05 | .10 | none | disabled | none |
| Q08 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_align.json` | 0 | .05 | .10 | none | disabled | same 1-epoch alignment as Q02 |
| Q09 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020.json` | 0 | .20 | .20 | none | disabled | none |
| Q10 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_chat_weighted.json` | 0 | .05 | .10 | none | enabled, content `.97` | none |
| Q11 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg0125.json` | 0 | .05 | .125 | none | disabled | none |
| D01 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_attention_only.json` | 0 | .05 | .10 | exclude every MLP module | disabled | diagnostic, not an all-linear W2 payload |
| D02 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_mlp_only.json` | 0 | .05 | .10 | exclude every attention module | disabled | diagnostic, not an all-linear W2 payload |
| D03 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_seed1.json` | 1 | .05 | .10 | none | disabled | seed control |
| D04 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_spectral.json` | 0 | .05 | .10 | none | disabled | spectral refinement, rank 16, lambda .25 |
| D05 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_attention_replay.json` | 0 | .05 | .10 | none | disabled | attention-QKVO greedy final-logit replay; banks 1/2/3; 2 folds; KL gate .001; Top-N limit .0025; disjoint confirmation required |
| D06 | `scripts/configs/llama32_1b_v2b2_p32_yaqa_mlp_down_w25.json` | 0 | .05 | .05 at W2; .02 at W2.5 | `mlp.down_proj` at W2.5 | disabled | bitrate diagnostic, not flat W2 |

## Experiment outcomes

`D300 aligned` is the primary aligned-token score over 9,600 positions. Proxy-only results are deliberately kept in
a different column because they are teacher-forced/shared-prefix diagnostics and cannot be promoted as D300.

| ID | Commit | Status / decision | Checkpoint | Corrected D300 aligned | Exact @32 | Other result |
| --- | --- | --- | --- | ---: | ---: | --- |
| Q01 | `2f34e1da` lineage | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32` | 1,343/9,600 = 13.9896% | 1/300 | mean first divergence 4.1667 |
| Q02 | `f6375fa5` | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5` | 1,456/9,600 = 15.1667% | 2/300 | mean first divergence 4.5733 |
| Q03 | `2f34e1da` | superseded by Q07 | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da` | 1,641/9,600 = 17.0938% | 3/300 | locked KL .239397, Top-1 82.6090%, SP-Top1@32-W50 83.0222% |
| Q04 | `54ccd365` lineage | proxy-only; not promoted | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365` | not run | not run | rows 428..511: final Top-1 80.2898%; obsolete first-32 proxy 70.1265% |
| Q05 | `54ccd365` lineage | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365` | 1,210/9,600 = 12.6042% | 2/300 | mean first divergence 4.0100 |
| Q06 | `54ccd365` lineage | proxy-only; not promoted | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365` | not run | not run | rows 428..511: final Top-1 80.8682%; obsolete first-32 proxy 71.1310% |
| Q07 | `54ccd365` lineage | **current completed leader** | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365` | **1,717/9,600 = 17.8854%** | 3/300 | mean first divergence 4.8533; locked KL .272819, Top-1 81.2380%, Top-5 72.6887%, Top-10 71.6885%, SP-Top1@32-W50 82.4350% |
| Q08 | `ecf7081e` lineage | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e` | 1,653/9,600 = 17.2188% | 2/300 | loses 64 matches to Q07 despite stronger early survival |
| Q09 | `ecf7081e` lineage | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e` | 1,568/9,600 = 16.3333% | 3/300 | loses 149 matches to Q07 |
| Q10 | `d7eae64a` | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a` | 1,465/9,600 = 15.2604% | 4/300 | token-1 71%; mean first divergence 5.0333; loses 252 aggregate matches |
| Q11 | `8ae2db25` | rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25` | 1,290/9,600 = 13.4375% | 2/300 | mean first divergence 3.6233; loses 427 matches to Q07 |
| D01 | `ecf7081e` lineage | diagnostic only | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e` | not run | not run | rows 428..511: final Top-1 90.5371%; shared-prefix first-32 90.0298% |
| D02 | `ecf7081e` lineage | diagnostic only | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e` | not run | not run | rows 428..511: final Top-1 83.0951%; shared-prefix first-32 72.6935% |
| D03 | `ecf7081e` lineage | proxy-only; rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e` | not run | not run | rows 428..511: final Top-1 80.5823%; shared-prefix first-32 70.3125% |
| D04 | `ecf7081e` lineage | proxy-only; rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e` | not run | not run | rows 428..511: final Top-1 80.7286%; shared-prefix first-32 69.7917% |
| D05 | `ecf7081e` lineage | proxy-only; rejected | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e` | not run | not run | rows 428..511: final Top-1 80.4593%; shared-prefix first-32 69.7173% |
| D06 | `ecf7081e` lineage | diagnostic only; bitrate exception | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e` | not run | not run | locked KL .229711; Top-1 82.3401%; SP-Top1@32-W50 83.5570% |

Corrected D300 report paths, in Q01--Q11 order when available:

```text
/root/qvq-results/llama32-1b-v2b2p32-layerhybrid-div300-dev-v1.json
/root/qvq-results/llama32-1b-v2b2p32-layerdamp-align-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-layerdamp-align2e64-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-align-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg020-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-div300-dev-v2.json
/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0125-div300-dev-v2.json
```

Q07's locked ordinary evaluation used this exact invocation and wrote
`/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-main-locked-r512-n300-v3.json`:

```bash
python scripts/qvq_evaluate.py diagnostics \
  --dense-model /monster/data/model/Llama-3.2-1B-Instruct \
  --checkpoint /root/qvq-results/llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365 \
  --dataset /monster/data/model/dataset/nm-calibration/llm.parquet --dataset-split train \
  --row-start 512 --rows 300 --device cuda:0 \
  --output /root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-main-locked-r512-n300-v3.json \
  --divergence-rows 300 --divergence-tokens 32 --skip-independent-rollout --include-topn
```

## Post-quant alignment experiments

These modify only the fixed-trellis SU/SV tensors of Q03; the W2 payload remains fixed. Each JSON report serializes
the full resolved config and split token hashes.

| ID | Full resolved training config | Status | Teacher-forced result | Corrected D300 result |
| --- | --- | --- | --- | --- |
| A01 | NM rows 128..427 train, 428..467 validation, 468..511 evaluation; max length 1,024; batch 1; eval batch 4; Adam; LR 1e-5; 1 epoch; grad accumulation 2 with `sum`; SU/SV scope; QTIP-FP32 forward. Report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1.json` | accepted by local validation but rejected for D300 | held-out rows 512..811: KL .198997, Top-1 84.1893% | 1,449/9,600 = 15.0938%; exact 1/300; mean first divergence 4.1333 |
| A02 | Optimized YAQA rows 0..181 train; NM rows 428..467 validation and 468..511 evaluation; all other settings identical to A01. Report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1.json` | accepted by local validation but rejected for D300 | held-out rows 512..811: KL .213309, Top-1 83.7821% | 1,578/9,600 = 16.4375%; exact 2/300; mean first divergence 4.5833 |
| A03 | Same as A02 plus `teacher_rollout_tokens=32`; dense teacher greedily generates the continuation; prompt/template tokens remain context but loss applies only to continuation tokens. Report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-rollout32-v1.json` | rejected by strict local gate; no checkpoint written | eval KL .273135 -> .265773 and Top-1 80.6132% -> 81.1481%, but JSD .059245 -> .059727 regressed | not run because the candidate failed before publication |

A01 and A02 commands use `scripts/qvq_e2e_align.py`; their complete arguments are reproduced by their report
`config` objects. A03 was introduced at commit `21d92956` and intentionally fails closed when any acceptance metric
regresses.

## Proxy searches and protocol-invalid evidence

These runs are retained because they explain past choices, but none can substitute for corrected D300.

| ID | Full configuration / artifact | Result | Disposition |
| --- | --- | --- | --- |
| P01 | Greedy layer splice of uniform `.10` payloads into uniform `.05`, old 32-position proxy on NM rows 128..159; `/root/qvq-results/layer-hybrid-reg005-reg010-search-r128-n32.json` | base 75.1953%; accepted layer 0 -> 75.3906%, layer 6 -> 76.0742%, layer 10 -> 76.3672%, layer 12 -> 77.1484% | superseded: corrected D300 later showed uniform `.10` is stronger than this hybrid |
| P02 | Greedy role splice under the same old proxy; `/root/qvq-results/role-hybrid-reg005-reg010-search-r128-n32.json` | base 75.1953%; accepted `q_proj`, then `k_proj`; final 76.0742% | rejected; no corrected-D300 evidence |
| P03 | Earlier baseline versus “optimized 302k mix” table | both rows incorrectly reported KL .324339, Top-1 80.0348%, Top-5 71.4363%, Top-10 70.5156%, old first-32 71.4331% | **invalid stale-checkpoint comparison**. A real full-mix YAQA quant changed the checkpoint and produced KL .264975 and old first-32 76.0206%; neither old first-32 value is D300 |

P03 is the concrete reason checkpoint identity and serialized data provenance are now mandatory in this ledger.

## Test and validation log

| Commit / scope | Exact command | Result |
| --- | --- | --- |
| `31d800b2`, Viterbi pruning configuration/default | `pytest -q tests/test_qvq_viterbi_pruning_config.py` | 41 passed; default verified as `auto`, `norm_band`, exact, baseline fallback |
| `31d800b2`, repaired CUDA 13.3 pruning dispatch | `pytest -q tests/test_qvq_cuda.py::test_qvq_pruning_policy_dispatches_eligible_cells tests/test_qvq_cuda.py::test_qvq_pruning_policy_default_argument_matches_auto` | 9 passed across eligible W2.5/W3 two-bank/four-bank cells; native dispatch counter proved the exact norm-band path ran and matched the baseline bit-for-bit |
| `21d92956`, teacher-rollout alignment | `pytest -q tests/test_qvq_e2e_alignment.py` | 7 passed |
| `21d92956`, changed Python files | `ruff check scripts/qvq_e2e_align.py tests/test_qvq_e2e_alignment.py` | passed |
| `d7eae64a`, chat token weighting and harness provenance | `pytest -q tests/test_prepare_dataset.py tests/test_qvq_unified_harness.py` | 46 passed |
| `d7eae64a`, changed Python files | `ruff check ...` and `git diff --check` | passed |
| `8ae2db25`, Q11 config validation | quantization manifest inspection | 112 quant rows; every module damping `.125`; YAQA 182/182 sequences; 302,193 valid samples; no fallback |
| `8ae2db25`, Q11 corrected D300 | common corrected rollout command above | report checkpoint and manifest SHA matched; 300/300 prompts completed; 1,290/9,600 aligned; 2/300 exact |
| `723a8f95`, Q07 locked ordinary metrics | exact diagnostics command above | 300/300 rows; 105,618 tokens; KL .272819; Top-1 81.2380%; Top-5 72.6887%; Top-10 71.6885%; SP-Top1@32-W50 82.4350% |
| Q07 full downstream pre-run | `pytest -q tests/test_qvq_unified_harness.py tests/test_validate_qvq_lifecycle.py` | 55 passed; full MMLU humanities mapping covered |
| Q07 full downstream pre-run | broad `ruff check` on the four touched Python files | failed on 14 existing style/executable-bit findings; failure retained rather than silently omitted |
| Q07 full downstream pre-run | `ruff check --select F401,F811,F821,F822,F823` on the four touched Python files | passed |
| Q07 full downstream tasks | command recorded in `w2_leader.md`; GSM8K Platinum 1,209 rows, MMLU STEM full, MMLU humanities full | running; report `/root/qvq-results/llama32-1b-v2b2p32-reg010-leader-full-tasks-v1.json` |
| Q03 full downstream tasks | same task/batch contract as Q07 | queued; report `/root/qvq-results/llama32-1b-v2b2p32-layerdamp-align2e64-full-tasks-v1.json` |
| downstream progress fix | `pytest -q tests/test_qvq_unified_harness.py -k 'humanities or incremental'` plus focused Ruff correctness gate | 2 passed; Ruff passed |
| Q07 full downstream attempt 1 | full task command in `w2_leader.md`; continuous refill active, paged attention false | interrupted manually during GSM8K at user request; no report or partial score published |
| downstream paged/continuous gate | `pytest -q tests/test_qvq_unified_harness.py -k 'humanities or incremental or paged_continuous'` plus focused Ruff correctness gate | 3 passed; Ruff passed |
| Q07 full downstream attempt 2 | exact full command in `w2_leader.md`; Evalution 0.0.12; paged attention true; continuous refill active | GSM8K Platinum completed all 1,209 rows at `acc,num=0.2415`, zero invalid, generation 599.716s; MMLU STEM reached 630/12,612 choice requests before intentional dependency/progress-display restart; no combined JSON report published |
| Evalution update and MMLU row progress | upgraded PyPI Evalution 0.0.12 -> 0.0.14, LogBar 0.4.12 -> 0.4.13, PyPcre 0.6.0 -> 0.6.2; `pytest -q tests/test_qvq_unified_harness.py -k 'humanities or incremental or paged_continuous or completed_rows'`; focused Ruff correctness gate | 4 passed, 22 deselected, 14 dependency warnings; Ruff passed; latest Evalution still counts four choice requests per MMLU row, so QvQ adapter now reports completed/max question rows without changing continuous-refill work |
| Q07 full downstream attempt 3 | Evalution 0.0.14; full three-task model-serial command; paged attention and continuous refill verified | intentionally interrupted at GSM8K 643/1,209, running `numeric=0.2348`, zero invalid, before report publication; superseded by benchmark-paired order requested by user |
| paired incremental task runner | same checkpoints, batch 16, CUDA:0, `paged|flash_attention_2`, deterministic generation; one task per invocation in Q07/Q03 pairs; later tasks use `--resume` | task JSON is atomically published after every completed suite; resume rejects checkpoint/runtime mismatches and skips already-completed tasks; focused tests 4 passed and focused Ruff passed |
| paired GSM8K Platinum, Q07 then Q03 | full 1,209 rows each; Evalution 0.0.14; batch 16; CUDA:0; `paged|flash_attention_2`; continuous refill; instruct chat template; deterministic generation | Q07 `acc,num=0.2415219189`, 592.787s; Q03 `acc,num=0.2067824648`, 598.368s; Q07 wins by 0.0347394541 absolute; both atomically published |
| paired full MMLU cancellation | Q07 STEM first, full 3,153 rows planned; four likelihood choices/row; batch 16; paged attention; continuous refill; Q03 STEM and both full humanities queued behind it | Q07 STEM manually stopped at 217/3,153 rows on user direction because it was too slow; no partial metric published; remaining three MMLU invocations never started; no evaluator or GPU compute process remains |
| Q07 higher-rate sweep, W2.5 | Q07-identical model/data/YAQA controls; `qvq_v2b2_p32`; 2 banks; bits 2.5; exact-rate uniform YAQA regularization 0.10; full 182-row optimized YAQA mix; seed 0; batch 1; reselect; no chat weighting, alignment, spectral refinement, or scale optimization | queued for quantization only; no evaluation authorized |
| Q07 higher-rate sweep, W3.0 | Q07-identical model/data/YAQA controls; `qvq_v2b2_p32`; 2 banks; bits 3.0; exact-rate uniform YAQA regularization 0.10; full 182-row optimized YAQA mix; seed 0; batch 1; reselect; no chat weighting, alignment, spectral refinement, or scale optimization | queued behind W2.5 for quantization only; no evaluation authorized |
| Q07 higher-rate sweep attempt 1 | commit `6223d8ff`; W2.5 first, W3 serial; pre-pruning `origin/main`; one GPU; otherwise exact tracked W2.5/W3 configs above | W2.5 manually stopped during YAQA factor finalization on user direction; no checkpoint/output directory published; W3 never started |
| Q07 higher-rate sweep attempt 2 | merged `origin/main` PR #50; explicit `viterbi_pruning={mode:auto,strategy:norm_band,exact:true,fallback:baseline}`; W2.5 and W3 concurrent; each process isolated to one NVIDIA PG506-230 UUID; all Q07 data/YAQA controls unchanged | both runs completed all 182 YAQA rows / 302,193 tokens, then failed closed at layer 0 before quantizing a module because the partial CUDA 13.3 toolkit lacked `cusparse.h`; no checkpoint/output directory published |
| CUDA 13.3 JIT environment repair | installed matching `libcusparse-dev-13-3`, `libcublas-dev-13-3`, and `libcusolver-dev-13-3`; rebuilt the QVQ torch.ops extension once from commit `31d800b2` | extension compiled and loaded successfully in 162s; no source/config semantics changed |
| Q07 higher-rate sweep attempt 3 | same explicit exact-auto pruning and exact Q07 controls as attempt 2; W2.5 output `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-31d800b2-r2` on physical GPU UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`; W3 output `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-31d800b2-r2` on `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28`; each process sees only its assigned GPU as `cuda:0` | running concurrently; 182 YAQA rows / 302,193 tokens and 128 ordinary lifecycle rows / 49,725 tokens; quantization only, no evaluation authorized |

## Current decision

The completed flat-W2 leader remains Q07 at 17.8854%, 683 aligned positions short of the 25% development target.
Q11's midpoint damping was not intermediate in behavioral fidelity: it lost 427 matches to Q07 and 278 matches to
uniform `.20`. Concurrent W2.5 and W3 quantization-only arms are running with exact automatic Viterbi pruning. Nothing
in this ledger claims the target has been reached, and no proxy-only
arm is eligible for promotion.
