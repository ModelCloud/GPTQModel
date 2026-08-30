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
| Q07 higher-rate sweep attempt 3 | same explicit exact-auto pruning and exact Q07 controls as attempt 2; W2.5 output `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-31d800b2-r2` on physical GPU UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`; W3 output `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-31d800b2-r2` on `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28`; each process sees only its assigned GPU as `cuda:0` | manually stopped in layer 3 after telemetry showed only generic Viterbi work; no checkpoint published. The run was intentionally rejected because it could not report how many candidates norm-band pruning actually skipped |
| Exact pruning work telemetry | native CTA-reduced counters measure baseline candidate opportunities versus candidates actually evaluated; cumulative device snapshots also report eligible dispatches, automatic baseline fallbacks, skipped candidates, reduction ratio, and norm-rank cache entries; per-module telemetry records exact-auto policy and packed round-trip verification | focused config, manifest, bit-exact dispatch, default-dispatch, and measured-reduction suite: 52 passed. Deterministic random-input W3 B2-P32 smoke: 149,815,296 possible; 147,733,888 evaluated; 2,081,408 skipped (1.389316%); states, selectors, and squared errors bit-exact against forced baseline |
| Q07 higher-rate sweep attempt 4 | telemetry-capable quantizer code `329cc0a5`; exact-auto pruning; all Q07 controls unchanged; W2.5 output `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5` on physical GPU UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`; W3 output `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5` on `GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28` | **completed**, quantization only. Both arms: 112 quant rows; 112/112 packed round-trip verifications; 182 YAQA rows / 302,193 valid tokens; 128 ordinary lifecycle rows / 49,725 tokens; zero YAQA bank-0 fallbacks and zero input/output damping retries. W2.5 prepared/quantized in 2,225.101s and saved in 1.833s; W3 in 2,196.369s and 1.122s. Each reports exact `norm_band`, 181,792 baseline fallbacks, zero eligible dispatches, and therefore zero candidates evaluated/skipped. Current-main exact pruning excludes Q07 `reselect` family-batched grid solves, so pruning is configured but is a correctness-preserving no-op for this comparable workflow. W2.5 shard-set digest `8fa2aa51b51d5d148fef9f7abf83a9cb8755f4e3b9490700275f79e1344c78a9` (815 MiB); W3 `af360fbdc02092edd67288e20864c09058b38eb871244a58bab3fadf529f59b2` (873 MiB), proving distinct payloads. Because the branch received docs-only commit `72123a55` during execution, the reports' late-read `commit` field says `72123a55`; the embedded quantizer version and loaded native code correctly identify `329cc0a5`. Future runs freeze HEAD at process start. Basic locked diagnostics were subsequently authorized and run separately; no downstream GSM8K/MMLU evaluation is authorized. |
| Q07 W2.5 locked fidelity | checkpoint above; NM rows 512--811; 300 rows / 105,618 tokens; Top-N enabled; teacher-forced shared prefix uses 50% context warmup | KL `.125240`; Top-1 `87.1177%`; Top-5 `79.9364%`; Top-10 `79.4622%`; SP-Top1@32-W50 `87.8146%`; exact shared-prefix `7.7181%`. Full report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5-locked-r512-n300-full-v1.json`, SHA-256 `b40dd718cf6dd322a1e02f3620703cc950bfade871809e11a5427eda5450d1bf` |
| Q07 W3 locked fidelity | same locked dataset and protocol as W2.5 | KL `.060929`; Top-1 `91.1634%`; Top-5 `85.3099%`; Top-10 `85.0076%`; SP-Top1@32-W50 `91.4115%`; exact shared-prefix `13.4228%`. Full report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5-locked-r512-n300-full-v1.json`, SHA-256 `535b41a38f6b2707690bf08e988e7ac05799e9bf382cb706e5323b55c5e9dae6` |
| Q07 W2.5 canonical D300 | pinned mixed-source development manifest, SHA-256 `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2`; 300 prompts / 889,765 prompt tokens; 49 at 16,384-token cap; FP16 SDPA; native chat template; independent greedy 32-token trajectories | aligned Top-1 `25.1771%` (`2,417/9,600`); exact `5/300` (`1.6667%`); mean first divergence `7.5000`. Report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5-div300-dev-v1.json`, SHA-256 `7b93c80590e12fcfcf249855a6d971ac390c08f34a1fc3ad1127caa02080b6c4` |
| Q07 W3 canonical D300 | identical pinned manifest and decoding contract | aligned Top-1 `33.6667%` (`3,232/9,600`); exact `16/300` (`5.3333%`); mean first divergence `9.9133`. Report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5-div300-dev-v1.json`, SHA-256 `08a4bc6ccd8b4a3b4ead7b21fb8de5a47c1e1d611cd7c3ee2bd588ccb3fa1a09` |
| Higher-rate NM independent-rollout diagnostic | NM rows 512--811, not the canonical mixed-source D300 dataset | W2.5 `25.6146%`, exact `18/300`; W3 `33.8125%`, exact `27/300`. Retained as noncanonical diagnostics and never compared against Q07's canonical `17.8854%` |

## Current decision

The completed flat-W2 leader remains Q07 at 17.8854%, 683 aligned positions short of the 25% development target.
Q11's midpoint damping was not intermediate in behavioral fidelity: it lost 427 matches to Q07 and 278 matches to
uniform `.20`. Telemetry-capable W2.5 and W3 quantization arms completed successfully and were evaluated on the
locked ordinary and canonical mixed-source D300 protocols. W3 wins every recorded fidelity metric; W2.5 crosses the
25% canonical-D300 target by 17 aligned positions and W3 crosses it by 832. Neither received GSM8K/MMLU evaluation.
Nothing in this ledger
claims the target has been reached, and no proxy-only
arm is eligible for promotion.

## AIME 2025/2026 YAQA augmentation (queued)

To test whether math-focused activation coverage improves post-quant GSM8K
without contaminating evaluation, `build_aime2526_mix.py` downloads the pinned
local HF cache entries for `MathArena/aime_2025` and `MathArena/aime_2026`,
formats each problem with the same competition prompt wrapper used by D300,
and filters exact canonical-message SHA-256 intersections with the D300
development manifest.  Of 60 source problems, 26 overlap D300 and are excluded;
34 new rows are appended to the existing 182-row YAQA mix, producing 216 rows.
The resulting parquet SHA-256 is recorded in
`dataset/calibration_mix_500k_llama3.2_1b/calibration_aime2526.json`; no GSM8K
Platinum examples are used for calibration.

The W2 arm uses `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg010_aime2526.json`
(flat QVQ V2B2-P32, YAQA, regularization .05, full 216-row mix, reselect family
mode, seed 0) and ordinary lifecycle rows 0--127.  It is running on the freed
GPU4 as `/root/qvq-results/llama32-1b-w2-aime2526-yaqa-mix-main-1f6c2132`;
canonical D300 and GSM8K Platinum are evaluated only after quantization.

The already-completed higher-rate Q07 checkpoints were then evaluated first on
the full 1,209-row GSM8K Platinum task with Evalution 0.0.14, batch 16,
continuous refill, and `paged|flash_attention_2`: W2.5 reached `acc,num=
0.3358` and W3 reached `0.4127`, both with zero invalid generations. Reports:
`/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5-gsm8k-platinum-v1.json`
and the corresponding W3 path. Four completed W2 sweep arms are now undergoing
the same canonical D300 development check; their reports will be appended when
the 300-prompt runs finish.

### Reproducibility ledger

The machine-readable record for this AIME arm is
`docs/experiments/2026-08-25-llama32-w2-aime2526.json`.  The concurrent W2
regularization/seed sweep arms, each using the same 302,193-token YAQA mix and
ordinary rows 0--127, are recorded here with their live D300 report paths:

| Arm | Checkpoint | D300 report | Status |
| --- | --- | --- | --- |
| reg .0025 | `/root/qvq-results/llama32-1b-w2-reg0025-yaqa302k-main-a12ce4e3` | `/root/qvq-results/llama32-1b-w2-reg0025-yaqa302k-main-a12ce4e3-div300-dev-v1.json` | rollout running |
| reg .005 | `/root/qvq-results/llama32-1b-w2-reg005-yaqa302k-main-a12ce4e3` | `/root/qvq-results/llama32-1b-w2-reg005-yaqa302k-main-a12ce4e3-div300-dev-v1.json` | rollout running |
| reg .0125 | `/root/qvq-results/llama32-1b-w2-reg0125-yaqa302k-main-a12ce4e3` | `/root/qvq-results/llama32-1b-w2-reg0125-yaqa302k-main-a12ce4e3-div300-dev-v1.json` | rollout running |
| reg .020 | `/root/qvq-results/llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3` | `/root/qvq-results/llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3-div300-dev-v1.json` | rollout running |
| seed 1 | `/root/qvq-results/llama32-1b-w2-seed1-yaqa302k-main-a12ce4e3` | `/root/qvq-results/llama32-1b-w2-seed1-yaqa302k-main-a12ce4e3-div300-dev-v1.json` | rollout running |
| fixed-block LDLQ reg .005 | `/root/qvq-results/llama32-1b-w2-fixedreg005-yaqa302k-main-a12ce4e3` | — | quantized; D300 pending |

All rows above use the pinned D300 manifest and must be compared using its
aligned token top-1, exact trajectory survival, and mean first divergence; no
NM-only proxy result is substituted.

### Mixed-rate W2 / W3 MLP-down diagnostic (queued)

`llama32_1b_v2b2_p32_yaqa_aime_mlp_down_w30.json` keeps every module at flat
W2 and overrides only `.*\\.mlp\\.down_proj` to W3. It uses the same 216-row
leakage-safe AIME YAQA mix and ordinary rows 0--127. The quantizer is queued on
GPU4 as `/root/qvq-results/llama32-1b-w2-aime2526-mlpdown-w3-main-fdfd760e`;
canonical D300 and full GSM8K Platinum are queued after the checkpoint. The
machine-readable record is
`docs/experiments/2026-08-25-llama32-w2-mlpdown-w3.json`.

The AIME-only flat-W2 checkpoint finished quantization at
`/root/qvq-results/llama32-1b-w2-aime2526-yaqa-mix-main-1f6c2132`. Its canonical
D300 rollout is queued on GPU4 behind the mixed-rate diagnostic; the queue and
completion marker are recorded in
`docs/experiments/2026-08-25-llama32-w2-aime2526.json`.

The mixed-rate checkpoint has now completed quantization at
`/root/qvq-results/llama32-1b-w2-aime2526-mlpdown-w3-main-fdfd760e`. Its
canonical D300 rollout is running on GPU4, followed automatically by GSM8K
Platinum; no metric is promoted until both reports are published.

The flat-W2 AIME checkpoint's D300 evaluation was moved to free GPU7 and is
now running immediately; this avoids serializing the decisive flat-W2 result
behind the mixed-rate GSM8K task.

The mixed-rate diagnostic has completed canonical D300: W2 everywhere with
only `mlp.down_proj` at W3 reached `16.7396%` Divergence-32 top-1, `2/300`
exact trajectories, and mean first divergence `4.8267`. This is below the
best flat-W2 reg .020 arm (18.7813%), so the extra down-projection bit did not
improve this AIME mix; the full GSM8K Platinum task is now running for the
real-world comparison.

The mixed-rate checkpoint's GSM8K Platinum task completed all 1,209 examples
with zero invalid generations at `acc,num=0.2779`. This is logged alongside
its canonical D300 result in
`docs/experiments/2026-08-25-llama32-w2-mlpdown-w3.json`.

The flat-W2 AIME arm's canonical D300 is complete: `14.9792%` Divergence-32
top-1, `0/300` exact trajectories, mean first divergence `3.9967`. It is below
the 18.7813% reg .020 baseline, so AIME-only augmentation is rejected for
promotion; its full report and hash are in
`docs/experiments/2026-08-25-llama32-w2-aime2526.json`.

The same flat-W2 AIME checkpoint scored `acc,num=0.1737` on all 1,209 GSM8K
Platinum examples with zero invalid generations. This confirms the AIME-only
mix is not a useful promotion candidate for either canonical D300 or GSM8K.

### Full reference YAQA W2 arm (queued)

The original scan deliberately held out a 110-row reference set. It is proven
disjoint from benchmark/YAQA rows and is now appended to the 182-row mix,
creating a 292-row calibration set (`713e034c...f916`, provenance in
`calibration_full_reference.json`). This is the largest locally available
leakage-safe mix and is the next flat-W2 test using the best completed
regularization (`reg .020`). Quantization is running on GPU0, followed by
canonical D300 and GSM8K Platinum. Full queue/config/results are recorded in
`docs/experiments/2026-08-25-llama32-w2-full-reference-reg020.json`.

To bracket regularization on the larger mix, parallel full-reference W2 arms
at `.10` and `.30` are queued on GPUs2/3. Their complete configs, checkpoints,
canonical D300 reports, and GSM8K reports are tracked in
`docs/experiments/2026-08-25-llama32-w2-full-reference-reg-sweep.json`.

The full-reference reg .020 arm has completed D300 at `14.0313%` top-1,
`4/300` exact trajectories, mean first divergence `4.1467`; it is below the
302k-row reg .020 result (18.7813%) and is not a promotion candidate. Its
GSM8K Platinum run is still completing.

That reg .020 full-reference arm's GSM8K Platinum result is now complete at
`acc,num=0.2316` (1,209 examples, zero invalid), also below the original
302k-row W2 leader's downstream behavior.

The bracket's reg `.10` and `.30` arms completed D300 at `14.5417%` and
`17.7292%`, respectively; neither exceeds the 18.7813% completed W2 leader.
Their GSM8K Platinum tasks are running, and all report hashes/configuration
details are in the sweep JSON ledger.

The completed GSM8K Platinum scores for the full-reference bracket are
reg .10 `0.2142`, reg .20 `0.2316`, and reg .30 `0.1960` (all 1,209 rows,
zero invalid). None offsets the D300 degradation from expanding the generic
mix.

### 500k-token D300-source-shaped W2 arm (queued)

`build_div300_source_mix.py` produced 993 rows / 501,692 tokenizer tokens from
unused Terminal-Bench, SWE-Bench, MathArena, Multi-IF, and LongBench source
rows. It excludes exact D300 prompt hashes and existing calibration hashes,
and caps each source prompt before token counting. The W2 reg .20 arm is now
queued on GPU4; its complete provenance, checkpoint, D300 report, and GSM8K
chain are in `docs/experiments/2026-08-25-llama32-w2-div300-sources-500k.json`.

The 500k-token source-shaped checkpoint completed quantization with
`valid_output_samples=502,685`. Its canonical D300 rollout completed on GPU4:
independent token top-1 at 32 is `17.0729%`, exact 32-token trajectories are
`5/300 (1.6667%)`, and mean first divergence is `4.9933`. This is below the
current completed flat-W2 leader (`18.7813%`), so the 500k expansion does not
meet the 25% target. The chained GSM8K Platinum run is now active; its result
The chained GSM8K Platinum evaluation is complete: `acc,num=0.1191` on
`1,209` rows, with one invalid output, using Evalution 0.0.14, batch 16,
continuous batching, and paged attention. Its report SHA-256 is
`540bbb783ab7f94f5efdd442a8ae0714dca233221f5439fca92a2dcd05e4c2e9`.
D300 report SHA-256 is
`9d61f42c894efc894f2ff2530c70a3a59b2c85635a17b6432b1da3eabab9caec`.

### Completed W2 regularization/seed D300 results

### Queued 500k source-shaped W2 + MLP down W3 arm

To test whether extra precision only in the SwiGLU `down_proj` recovers the
lost behavior, a mixed-rate arm is queued on GPU5. All other modules remain
flat W2; `+:.*\\.mlp\\.down_proj` is W3. It reuses the disjoint 501,692-token
D300-source-shaped calibration set, with YAQA regularization `.05` (W2) and
`.10` (W3). Quantization session `66993` chains to canonical D300 and GSM8K
Platinum evaluation in session `15403`; the complete ledger is
`docs/experiments/2026-08-25-llama32-w2-div300-sources-500k-mlpdown-w3.json`.

### Queued 500k source-shaped flat-W2 regularization bracket

To continue the flat-W2 target after the `.02` source-shaped arm reached
`17.0729%`, two disjoint, canonical evaluations are queued using the same
501,692-token calibration set: YAQA regularization `.01` on GPU0 (session
`20309`) and `.03` on GPU1 (session `75708`). Their machine-readable ledger is
`docs/experiments/2026-08-25-llama32-w2-div300-sources-500k-reg-bracket.json`.

### Queued focused bracket around the best W2 regularization

The completed leader uses YAQA regularization `0.2` (the historical `reg020`
label). To test whether its gain is a narrow optimum, five flat-W2 arms using
the same disjoint 302k YAQA mix are queued at `0.15`, `0.175`, `0.225`, `0.25`,
and `0.3` on GPUs 2, 3, 4, 6, and 7. Each has a canonical D300/GSM8K watcher;
the full session/checkpoint ledger is
`docs/experiments/2026-08-25-llama32-w2-yaqa302k-reg-focused-bracket.json`.

The completed-arm machine-readable summary is
`docs/experiments/2026-08-25-llama32-w2-sweep-a12ce4e3.json`. All use the same
canonical D300 manifest and protocol; `Divergence-32 top-1` below is the report's
`independent_token_top1_agreement_at_32` field.

| Arm | D300 top-1 | Exact 32-token trajectories | Mean first divergence | Result |
| --- | ---: | ---: | ---: | --- |
| reg .0125 | 13.4375% | 2/300 (0.6667%) | 3.6233 | below target |
| reg .020 | **18.7813%** | 6/300 (2.0000%) | 5.3633 | current completed W2 sweep best |
| seed 1 | 17.7083% | 2/300 (0.6667%) | 4.9667 | below target |
| reg .0025 | 16.8229% | 1/300 (0.3333%) | 4.2433 | below target |
| reg .005 | 14.1042% | 2/300 (0.6667%) | 4.5167 | below target |
| fixed-block LDLQ reg .005 | 16.3021% | 5/300 (1.6667%) | 4.9133 | below target |

The 25% W2 target remains unverified; the best completed arm is 6.2187 points
short. reg .0025 and reg .005 D300 reports are still running.

### PR progress note — completed arm results

All values below use the canonical 300-prompt D300 manifest. New completed-arm
results should be appended to this table before being reported in PR notes.

| Arm | Quantization rate | Calibration arm | D300 top-1 @32 | Exact / 300 | Mean first divergence | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| YAQA reg .020 | Flat W2 | YAQA 302k | **18.7813%** | 6 | 5.3633 | Best completed W2 |
| YAQA seed 1 | Flat W2 | YAQA 302k | 17.7083% | 2 | 4.9667 | Below leader |
| Full reference reg .030 | Flat W2 | Full reference | 17.7292% | 3 | 4.5700 | Below leader |
| D300-source reg .020 | Flat W2 | 501,692-token source mix | 17.0729% | 5 | 4.9933 | Below leader |
| Fixed-block LDLQ reg .005 | Flat W2 | YAQA 302k | 16.3021% | 5 | 4.9133 | Below leader |
| AIME 2025/26 | Flat W2 | AIME mix | 14.9792% | 0 | 3.9967 | Below leader |
| AIME + MLP down W3 | W2 + W3 down | AIME mix | 16.7396% | 2 | 4.8267 | Mixed-rate diagnostic |

The focused `.15/.175/.225/.25/.30` bracket has now completed canonical D300;
the chained GSM8K checks are still running:

| Arm | D300 top-1 @32 | Exact / 300 | Mean first divergence | Status |
| --- | ---: | ---: | ---: | --- |
| YAQA reg .015 | 17.9792% | 3 | 5.3200 | GSM8K running |
| YAQA reg .0175 | 13.9271% | 1 | 4.2433 | GSM8K running |
| YAQA reg .0225 | 16.9271% | 6 | 4.6667 | GSM8K running |
| YAQA reg .025 | 15.3854% | 2 | 4.2667 | GSM8K running |
| YAQA reg .030 | 16.2188% | 5 | 4.4333 | GSM8K running |

The best remains the prior reg .020 arm at 18.7813%; no flat-W2 bracket arm
has reached the 25% D300 target.

### Queued aligned flat-W2 YAQA arm

To test whether post-quant output alignment can recover the remaining gap
without increasing bits, a new flat-W2 arm combines the best YAQA 302k mix and
reg .020 with one epoch of pristine-Hessian output alignment (32 train and 16
validation batches). Its config is
`scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020_align.json`; quantization is
running on GPU6 at checkpoint
`/root/qvq-results/llama32-1b-w2-yaqa302k-reg020-align-main-ff906097`, followed
by canonical D300 using the unchanged manifest and protocol.

An additional flat-W2 sensitivity arm is queued on GPU7 at
`/root/qvq-results/llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc`.
It keeps reg .020 globally but sets YAQA regularization to `.4` for layers
0, 6, 10, and 12, the sensitive-layer set used by earlier dynamic tests. Its
config is `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020_sensitive_hi.json`
and it is chained to canonical D300.

GSM8K Platinum for the best YAQA reg .020 checkpoint is now running on the
newly freed GPU0 (session `98290`); the earlier GPU2 waiter (`24963`) was
superseded. It uses batch 16 with continuous batching and paged attention.

That second-level verification is complete: the full report records
`acc,num=0.17866004962779156` with Evalution `0.0.14`, batch 16,
`paged|flash_attention_2`, and continuous batching/paged-attention required.
The checkpoint's canonical D300 score remains `18.78125%` (6/300 exact
prompts), so this confirms the current leader but does not meet the flat-W2
`25%` D300 target.

### Queued full-subset final-logit replay control

The prior `D05` attention replay was only a proxy evaluation and was rejected;
it did not establish a canonical D300 gain. The implementation does support
cross-fitted replay over coupled semantic subsets, so a complete flat-W2
control is queued after the leader GSM8K job: `attention_qkvo`, `mlp_gate_up`,
and `mlp_down`, with greedy alternative-bank selection, final-logit horizon,
two search folds, and disjoint confirmation. Ordinary calibration, YAQA, replay
search, and replay confirmation are separate slices. The config and queue
ledger are `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg020_all_subset_replay.json`
and `docs/experiments/2026-08-25-llama32-w2-reg020-all-subset-replay.json`.

### Replay/subset control audit

The queued control is whole-model propagated replay, not independent local
reconstruction: each candidate bank is evaluated through the live model to
final logits, selected only after two search folds, a relative-KL improvement
of at least `0.001`, a top-N regression limit of `0.0025`, and disjoint
confirmation. Supported coupled subsets include attention Q/K, V/O, Q/K/V/O,
MLP gate/up, MLP down, and all three MLP projections. The queued control covers
all attention projections plus gate/up and down.

The old QKV-only and MLP-only figures were proxy diagnostics, not canonical
D300, so they do not establish real inference improvement. No completed
full-D300 replay result has yet beaten the YAQA reg `.020` leader. This control
is intended to measure cross-layer/subset error propagation against final
logits, followed by GSM8K Platinum verification.

### Second-level verification policy

From this point forward, every new D300 leader or arm within one percentage
point of the current flat-W2 leader (`18.7813%`) receives a queued GSM8K
Platinum evaluation after D300. The focused regularization bracket already has
that chain; the all-subset replay control now has a dedicated GSM8K watcher
(session `15693`) as well. This keeps D300 selection and real-task verification
separate while ensuring close candidates are not promoted on D300 alone.

### Parallel flat-W2 alignment sweep (2026-08-26)

Four additional YAQA 302k flat-W2 arms were launched on previously idle GPUs,
using the same 128 ordinary calibration rows plus 182 YAQA rows and the
canonical disjoint D300 protocol. These are pending quantization and will be
evaluated/queued for GSM8K Platinum if they become leaders or land within one
percentage point of the current 18.7813% leader:

| GPU | Arm | Config | Checkpoint | Status |
| --- | --- | --- | --- | --- |
| 1 | reg .010 + output alignment | `llama32_1b_v2b2_p32_yaqa_reg010_align.json` | `llama32-1b-w2-yaqa302k-reg010-align-gpu1` | running |
| 2 | layer damping + output alignment | `llama32_1b_v2b2_p32_yaqa_layer_damping_align.json` | `llama32-1b-w2-yaqa302k-layerdamp-align-gpu2` | running |
| 3 | layer damping + 2-epoch/64-batch alignment | `llama32_1b_v2b2_p32_yaqa_layer_damping_align2e64.json` | `llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3` | running |
| 4 | reg .025 + output alignment | `llama32_1b_v2b2_p32_yaqa_reg025_align.json` | `llama32-1b-w2-yaqa302k-reg025-align-gpu4` | running |

No result is treated as a gain until canonical D300 completes on the exact
same manifest/protocol.

### Queued fixed-block LDLQ and down-projection capacity follow-up

Based on the observed fixed-block LDLQ gain at reg .005 and the strong
down-projection W2.5 signal, the next same-protocol factorial follow-up is
queued behind the current alignment jobs. It uses the same ordinary 128 rows,
YAQA 182 rows, and disjoint canonical D300 manifest:

| GPU | Arm | Config | Status |
| --- | --- | --- | --- |
| 1 | fixed-block LDLQ, reg .15 | `llama32_1b_v2b2_p32_yaqa_fixed_reg015.json` | queued |
| 2 | fixed-block LDLQ, reg .20 | `llama32_1b_v2b2_p32_yaqa_fixed_reg020.json` | queued |
| 3 | fixed-block LDLQ, reg .225 | `llama32_1b_v2b2_p32_yaqa_fixed_reg0225.json` | queued |
| 4 | `mlp.down_proj` W2.5, reg .20 | `llama32_1b_v2b2_p32_yaqa_mlp_down_w25_reg020.json` | queued |

Each watcher starts only after its current GPU's alignment quantizer exits,
then runs canonical D300 automatically. These arms isolate the two highest
priority controls from the review before any combination is attempted.

Queue correction: the initial watcher used a `pgrep -f` pattern that matched
its own shell command, so the follow-up did not start when GPUs 2 and 4 freed.
That watcher was replaced with a non-self-matching `[q]vq_quantize.py` pattern;
fixed-reg .020 and down-projection W2.5 are now running on GPUs 2 and 4, while
fixed-reg .015 and .0225 wait behind the still-active alignment jobs on GPUs 1
and 3.

| monitor | `llama32-1b-w2-full-reference-reg020-main-5544143c` | `gsm8k_platinum_cot` | complete; metric=0.23325062034739455; report `/root/qvq-results/llama32-1b-w2-full-reference-reg020-main-5544143c-gsm8k-platinum-reverify-v2.json` |

| monitor | `llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1` | `gsm8k_platinum_cot` | complete; metric=0.23325062034739455; report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365` | `gsm8k_platinum_cot` | complete; metric=0.23490488006617039; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-fixed-reg005-pr34-bb1c250e` | `gsm8k_platinum_cot` | complete; metric=0.20678246484698098; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-fixed-reg005-pr34-bb1c250e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn005-mlp010` | `gsm8k_platinum_cot` | complete; metric=0.2109181141439206; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-hybrid-attn005-mlp010-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn010-mlp005` | `gsm8k_platinum_cot` | complete; metric=0.20926385442514475; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-hybrid-attn010-mlp005-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32` | `gsm8k_platinum_cot` | complete; metric=0.19768403639371382; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg015-gpu1` | `divergence300` | complete; metric=0.16739583333333333; report `/root/qvq-results/llama32-1b-w2-fixed-reg015-gpu1-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg020-gpu2` | `divergence300` | complete; metric=0.14458333333333334; report `/root/qvq-results/llama32-1b-w2-fixed-reg020-gpu2-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg020-gpu6` | `divergence300` | complete; metric=0.14458333333333334; report `/root/qvq-results/llama32-1b-w2-fixed-reg020-gpu6-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg0225-gpu3` | `divergence300` | complete; metric=0.1709375; report `/root/qvq-results/llama32-1b-w2-fixed-reg0225-gpu3-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg0225-gpu7` | `divergence300` | complete; metric=0.1709375; report `/root/qvq-results/llama32-1b-w2-fixed-reg0225-gpu7-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-full-reference-reg010-main-5544143c` | `gsm8k_platinum_cot` | complete; metric=0.2142266335814723; report `/root/qvq-results/llama32-1b-w2-full-reference-reg010-main-5544143c-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-full-reference-reg030-main-5544143c` | `gsm8k_platinum_cot` | complete; metric=0.19602977667493796; report `/root/qvq-results/llama32-1b-w2-full-reference-reg030-main-5544143c-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-mlp-down-w25-reg020-gpu4` | `divergence300` | complete; metric=0.14822916666666666; report `/root/qvq-results/llama32-1b-w2-mlp-down-w25-reg020-gpu4-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.17866004962779156; report `/root/qvq-results/llama32-1b-w2-reg020-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3` | `divergence300` | complete; metric=0.15072916666666666; report `/root/qvq-results/llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg010-align-gpu1` | `divergence300` | complete; metric=0.1840625; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-main-5cfb8314` | `gsm8k_platinum_cot` | complete; metric=0.23821339950372208; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-main-5cfb8314-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-main-5cfb8314` | `divergence300` | complete; metric=0.17979166666666666; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-main-5cfb8314-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg0175-main-5cfb8314` | `gsm8k_platinum_cot` | complete; metric=0.19189412737799835; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg0175-main-5cfb8314-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg0175-main-5cfb8314` | `divergence300` | complete; metric=0.13927083333333334; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg0175-main-5cfb8314-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg0225-main-5cfb8314` | `gsm8k_platinum_cot` | complete; metric=0.21257237386269645; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg0225-main-5cfb8314-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg0225-main-5cfb8314` | `divergence300` | complete; metric=0.16927083333333334; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg0225-main-5cfb8314-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg025-main-5cfb8314` | `gsm8k_platinum_cot` | complete; metric=0.20595533498759305; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg025-main-5cfb8314-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg025-main-5cfb8314` | `divergence300` | complete; metric=0.15385416666666665; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg025-main-5cfb8314-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg030-main-5cfb8314` | `gsm8k_platinum_cot` | complete; metric=0.19106699751861042; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg030-main-5cfb8314-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg030-main-5cfb8314` | `divergence300` | complete; metric=0.1621875; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg030-main-5cfb8314-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5` | `gsm8k_platinum_cot` | complete; metric=0.21257237386269645; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg0225-gpu3` | `gsm8k_platinum_cot` | complete; metric=0.22580645161290322; report `/root/qvq-results/llama32-1b-w2-fixed-reg0225-gpu3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg0225-gpu7` | `gsm8k_platinum_cot` | complete; metric=0.22580645161290322; report `/root/qvq-results/llama32-1b-w2-fixed-reg0225-gpu7-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixedreg005-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.20264681555004135; report `/root/qvq-results/llama32-1b-w2-fixedreg005-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da` | `gsm8k_platinum_cot` | complete; metric=0.20926385442514475; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365` | `gsm8k_platinum_cot` | complete; metric=0.21257237386269645; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg020-tip-gpu0` | `gsm8k_platinum_cot` | complete; metric=0.23407775020678245; report `/root/qvq-results/llama32-1b-w2-fixed-reg020-tip-gpu0-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.23986765922249792; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg020-gpu2` | `gsm8k_platinum_cot` | complete; metric=0.23407775020678245; report `/root/qvq-results/llama32-1b-w2-fixed-reg020-gpu2-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg020-gpu6` | `gsm8k_platinum_cot` | complete; metric=0.23407775020678245; report `/root/qvq-results/llama32-1b-w2-fixed-reg020-gpu6-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-mlp-down-w25-reg020-gpu4` | `gsm8k_platinum_cot` | complete; metric=0.26881720430107525; report `/root/qvq-results/llama32-1b-w2-mlp-down-w25-reg020-gpu4-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.20595533498759305; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg015-gpu1` | `gsm8k_platinum_cot` | complete; metric=0.22332506203473945; report `/root/qvq-results/llama32-1b-w2-fixed-reg015-gpu1-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365` | `gsm8k_platinum_cot` | complete; metric=0.21009098428453268; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365` | `gsm8k_platinum_cot` | complete; metric=0.19272125723738626; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg005-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.20430107526881722; report `/root/qvq-results/llama32-1b-w2-reg005-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg0125-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.22497932175351532; report `/root/qvq-results/llama32-1b-w2-reg0125-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg0025-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.2076095947063689; report `/root/qvq-results/llama32-1b-w2-reg0025-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.3746898263027295; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a` | `gsm8k_platinum_cot` | complete; metric=0.21588089330024815; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-seed1-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.19933829611248965; report `/root/qvq-results/llama32-1b-w2-seed1-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-shiftcal128-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.21670802315963605; report `/root/qvq-results/llama32-1b-w2-shiftcal128-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-spectral-yaqa302k-main-a12ce4e3` | `gsm8k_platinum_cot` | complete; metric=0.21670802315963605; report `/root/qvq-results/llama32-1b-w2-spectral-yaqa302k-main-a12ce4e3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.19933829611248965; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa-optmix322k-main-de25f203` | `gsm8k_platinum_cot` | complete; metric=0.19272125723738626; report `/root/qvq-results/llama32-1b-w2-yaqa-optmix322k-main-de25f203-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa-main-de25f203` | `gsm8k_platinum_cot` | complete; metric=0.17038875103391232; report `/root/qvq-results/llama32-1b-w2-yaqa-main-de25f203-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-layerdamp-align-gpu2` | `gsm8k_platinum_cot` | complete; metric=0.22249793217535152; report `/root/qvq-results/llama32-1b-w2-yaqa302k-layerdamp-align-gpu2-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa-optmix551k-main-54ccd365` | `gsm8k_platinum_cot` | complete; metric=0.2787427626137304; report `/root/qvq-results/llama32-1b-w2-yaqa-optmix551k-main-54ccd365-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.2961124896608768; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.20678246484698098; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa-cal322k-main-de25f203` | `gsm8k_platinum_cot` | complete; metric=0.17038875103391232; report `/root/qvq-results/llama32-1b-w2-yaqa-cal322k-main-de25f203-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3` | `gsm8k_platinum_cot` | complete; metric=0.22746071133167908; report `/root/qvq-results/llama32-1b-w2-yaqa302k-layerdamp-align2e64-gpu3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg020-align-main-ff906097` | `gsm8k_platinum_cot` | complete; metric=0.21836228287841192; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg020-align-main-ff906097-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg025-align-gpu4` | `gsm8k_platinum_cot` | complete; metric=0.24813895781637718; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg025-align-gpu4-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25` | `gsm8k_platinum_cot` | complete; metric=0.22497932175351532; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg010-align-gpu1` | `gsm8k_platinum_cot` | complete; metric=0.21670802315963605; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg010-align-gpu1-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc` | `gsm8k_platinum_cot` | complete; metric=0.22084367245657568; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.21670802315963605; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-fixed-reg020-tip-gpu0` | `divergence300` | complete; metric=0.14458333333333334; report `/root/qvq-results/llama32-1b-w2-fixed-reg020-tip-gpu0-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e` | `gsm8k_platinum_cot` | complete; metric=0.19520264681555005; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-shiftcal128-yaqa302k-main-a12ce4e3` | `divergence300` | complete; metric=0.18427083333333333; report `/root/qvq-results/llama32-1b-w2-shiftcal128-yaqa302k-main-a12ce4e3-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-spectral-yaqa302k-main-a12ce4e3` | `divergence300` | complete; metric=0.181875; report `/root/qvq-results/llama32-1b-w2-spectral-yaqa302k-main-a12ce4e3-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa-cal322k-main-de25f203` | `divergence300` | complete; metric=0.13135416666666666; report `/root/qvq-results/llama32-1b-w2-yaqa-cal322k-main-de25f203-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1` | `gsm8k_platinum_cot` | complete; metric=0.22167080231596362; report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-rolehybrid-qk010-rest005-dev32` | `gsm8k_platinum_cot` | complete; metric=0.17535153019023986; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-rolehybrid-qk010-rest005-dev32-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa-main-de25f203` | `divergence300` | complete; metric=0.13135416666666666; report `/root/qvq-results/llama32-1b-w2-yaqa-main-de25f203-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg020-align-main-ff906097` | `divergence300` | complete; metric=0.15072916666666666; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg020-align-main-ff906097-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc` | `divergence300` | complete; metric=0.1528125; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg020-sensitive-hi-main-39e26ebc-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1` | `divergence300` | complete; metric=0.15520833333333334; report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-r128n300-v1-div300-dev-v1.json` |

| queued | `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | `quantization + gsm8k_platinum_cot + divergence300` | reg=0.15 reselect with output alignment; YAQA rows 0:182; NM rows 0:128; disjointness manifest `disjointness-yaqa182.json`; active session PID 3414890 |

| queued | `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | `quantization + gsm8k_platinum_cot + divergence300` | reg=0.15 reselect with `mlp.down_proj` W2.5; YAQA rows 0:182; NM rows 0:128; disjointness manifest `disjointness-yaqa182.json`; active session PID 3415015 |

| queued-next | `llama32-1b-w2-yaqa302k-reg015-gateup-w25` | `quantization + gsm8k_platinum_cot + divergence300` | reg=0.15 reselect; `mlp.gate_proj` + `mlp.up_proj` W2.5; same clean YAQA/NM slices; starts when a GPU frees |

| queued-next | `llama32-1b-w2-yaqa302k-reg015-qk-w25` | `quantization + gsm8k_platinum_cot + divergence300` | reg=0.15 reselect; attention `q_proj` + `k_proj` W2.5; same clean YAQA/NM slices; starts when a GPU frees |

| monitor | `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | `divergence300` | complete; D300 token top-1=0.1492708333; exact32=0.0166666667; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | `gsm8k_platinum_cot` | complete; acc=0.2299421009 (278/1209); report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | `divergence300` | complete; D300 token top-1=0.1780208333; exact32=0.01; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | `gsm8k_platinum_cot` | complete; acc=0.2340777502 (283/1209); report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7-gsm8k-platinum-v1.json` |

| active | `llama32-1b-w2-reg015-atomic-swiglu` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15; atomic SwiGLU gate/up/down replay; search rows 0:32, confirmation rows 32:64; physical GPU 4; session 69859 |

| active | `llama32-1b-w2-reg015-smooth-swiglu` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15; Smooth-SwiGLU only; physical GPU 5; session 33771 |

| active | `llama32-1b-w2-reg015-smooth-atomic-swiglu` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15; Smooth-SwiGLU plus atomic gate/up/down replay; search rows 0:32, confirmation rows 32:64; physical GPU 6; session 36247 |

| queued | `llama32-1b-w2-reg015-mlpall-w25` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15; all MLP gate/up/down W2.5; YAQA rows 0:182, NM rows 0:128; physical GPU 4 |

| queued | `llama32-1b-w2-reg015-attnall-w25` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15; attention Q/K/V/O W2.5; YAQA rows 0:182, NM rows 0:128; physical GPU 6 |

| monitor | `llama32-1b-w2-reg020-all-subset-replay-disjoint-tip-gpu5` | `gsm8k_platinum_cot` | complete; metric=0.18031430934656742; report `/root/qvq-results/llama32-1b-w2-reg020-all-subset-replay-disjoint-tip-gpu5-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | `gsm8k_platinum_cot` | complete; metric=0.23407775020678245; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1` | `divergence300` | complete; metric=0.17114583333333333; report `/root/qvq-results/llama32-1b-v2b2p32-align2e64-e2e-susv-yaqa182-v1-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365` | `divergence300` | complete; metric=0.1753125; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | `gsm8k_platinum_cot` | complete; metric=0.22994210090984285; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1` | `divergence300` | complete; metric=0.14927083333333332; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-align-tip-gpu1-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7` | `divergence300` | complete; metric=0.17802083333333332; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-mlpdown-w25-tip-gpu7-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-fixed-reg005-pr34-bb1c250e` | `divergence300` | complete; metric=0.15166666666666667; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-fixed-reg005-pr34-bb1c250e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn005-mlp010` | `divergence300` | complete; metric=0.14583333333333334; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-hybrid-attn005-mlp010-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg020-all-subset-replay-disjoint-tip-gpu5` | `divergence300` | complete; metric=0.1771875; report `/root/qvq-results/llama32-1b-w2-reg020-all-subset-replay-disjoint-tip-gpu5-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-hybrid-attn010-mlp005` | `divergence300` | complete; metric=0.15072916666666666; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-hybrid-attn010-mlp005-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5` | `divergence300` | complete; metric=0.14479166666666668; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align-main-f6375fa5-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32` | `divergence300` | complete; metric=0.13958333333333334; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerhybrid-r005-r010-dev32-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da` | `divergence300` | complete; metric=0.17489583333333333; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e` | `divergence300` | complete; metric=0.13572916666666668; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-mlpdown-w25-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365` | `divergence300` | complete; metric=0.16364583333333332; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0025-main-54ccd365-div300-dev-v1.json` |

| validation | `3f17c40b` → `d68c9d00` | `SwiGLU/lifecycle test suite` | complete; `70 passed, 2 skipped`; command `pytest -q tests/test_qvq_swiglu.py tests/test_qvq_lifecycle.py tests/test_qvq_module_granular_replay_config.py`; CUDA guard changes included; log `/tmp/swiglu_latest_tests.log` |

| active | `llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00` | `quantization + divergence300 + gsm8k_platinum_cot` | latest native Smooth-SwiGLU path; reg=0.15; clean YAQA/NM slices; CUDA physical GPU 6; session 18704 |

| monitor | `llama32-1b-w2-yaqa302k-reg015-gateup-w25` | `divergence300` | complete; D300 token top-1=0.1728125; exact32=0.0066666667; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-gateup-w25-div300-dev-v1.json`; GSM8K running on GPU 1 |

| monitor | `llama32-1b-w2-yaqa302k-reg015-qk-w25` | `divergence300` | complete; D300 token top-1=0.1610416667; exact32=0.0066666667; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-qk-w25-div300-dev-v1.json`; GSM8K running on GPU 2 |

| monitor | `llama32-1b-w2-reg015-mlpall-w25` | `divergence300` | complete; D300 token top-1=0.1908333333; exact32=0.01; report `/root/qvq-results/llama32-1b-w2-reg015-mlpall-w25-div300-dev-v1.json`; GSM8K running on GPU 4 |

| monitor | `llama32-1b-w2-reg015-attnall-w25` | `divergence300` | complete; D300 token top-1=0.2041666667; exact32=0.01; report `/root/qvq-results/llama32-1b-w2-reg015-attnall-w25-div300-dev-v1.json`; GSM8K running on GPU 5 |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu` | `divergence300` | complete; D300 token top-1=0.1730208333; exact32=0.0066666667; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-div300-dev-v1.json`; GSM8K running on GPU 7 |

| monitor | `llama32-1b-w2-yaqa302k-reg015-gateup-w25` | `gsm8k_platinum_cot` | complete; acc=0.2506203474 (303/1209); report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-gateup-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-qk-w25` | `gsm8k_platinum_cot` | complete; acc=0.2133995037 (258/1209); report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-qk-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-mlpall-w25` | `gsm8k_platinum_cot` | complete; acc=0.3159636063 (382/1209); report `/root/qvq-results/llama32-1b-w2-reg015-mlpall-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-attnall-w25` | `gsm8k_platinum_cot` | complete; acc=0.2605459057 (315/1209); report `/root/qvq-results/llama32-1b-w2-reg015-attnall-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu` | `gsm8k_platinum_cot` | complete; acc=0.2051282051 (248/1209); report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00` | `gsm8k_platinum_cot` | complete; acc=0.2051282051 (248/1209); latest-code rerun; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-qk-w25` | `divergence300` | complete; metric=0.16104166666666667; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-qk-w25-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-mlpall-w25` | `divergence300` | complete; metric=0.19083333333333333; report `/root/qvq-results/llama32-1b-w2-reg015-mlpall-w25-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu` | `divergence300` | complete; metric=0.17302083333333335; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-attnall-w25` | `divergence300` | complete; metric=0.20416666666666666; report `/root/qvq-results/llama32-1b-w2-reg015-attnall-w25-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-gateup-w25` | `divergence300` | complete; metric=0.1728125; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-gateup-w25-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu` | `gsm8k_platinum_cot` | complete; metric=0.20512820512820512; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-attnall-w25` | `gsm8k_platinum_cot` | complete; metric=0.26054590570719605; report `/root/qvq-results/llama32-1b-w2-reg015-attnall-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-qk-w25` | `gsm8k_platinum_cot` | complete; metric=0.21339950372208435; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-qk-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-mlpall-w25` | `gsm8k_platinum_cot` | complete; metric=0.3159636062861869; report `/root/qvq-results/llama32-1b-w2-reg015-mlpall-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg015-gateup-w25` | `gsm8k_platinum_cot` | complete; metric=0.2506203473945409; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg015-gateup-w25-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00` | `gsm8k_platinum_cot` | complete; metric=0.20512820512820512; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00-gsm8k-platinum-v1.json` |

| queued | `llama32-1b-w2-reg015-atomic-swiglu-tip-3f17c40b` | `quantization + divergence300 + gsm8k_platinum_cot` | latest atomic SwiGLU implementation; reg=0.15; replay search rows 0:32, confirmation rows 32:64; clean disjoint manifest; waiting for GPU 4 |

| queued | `llama32-1b-w2-reg015-smooth-atomic-swiglu-tip-3f17c40b` | `quantization + divergence300 + gsm8k_platinum_cot` | latest Smooth + atomic SwiGLU implementation; reg=0.15; replay search rows 0:32, confirmation rows 32:64; clean disjoint manifest; queued after atomic arm |

| restarted-queued | `19d89a` / `llama32-1b-w2-atomic-swiglu-tip-3f17c40b` | `quantization + divergence300 + gsm8k_platinum_cot` | prior worker exited before checkpoint publication; durable wrapper requeued on physical GPU 4 using the latest atomic config and clean disjoint replay rows 0:32 / 32:64 |

| restarted-queued | `ebec00` / `llama32-1b-w2-smooth-atomic-swiglu-tip-3f17c40b` | `quantization + divergence300 + gsm8k_platinum_cot` | durable wrapper waits for Atomic arm `19d89a` and then runs the latest Smooth + atomic config on physical GPU 6; clean disjoint replay rows 0:32 / 32:64 |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e` | `divergence300` | complete; metric=0.265; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365` | `divergence300` | complete; metric=0.120625; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365` | `divergence300` | complete; metric=0.18677083333333333; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00` | `divergence300` | complete; metric=0.17302083333333335; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e` | `divergence300` | complete; metric=0.17864583333333334; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e` | `divergence300` | complete; metric=0.16447916666666668; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-replay-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a` | `divergence300` | complete; metric=0.15260416666666668; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-chatw97-main-d7eae64a-div300-dev-v1.json` |

| queued | `d71136` / `llama32-1b-w2-reg015-vo-w25-d71136` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15 reselect; self-attention `v_proj` + `o_proj` W2.5; same clean YAQA/NM slices and disjoint manifest; waits for a free GPU |

| queued | `bb0aa2` / `llama32-1b-w2-reg015-gate-down-w25-bb0aa2` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15 reselect; MLP `gate_proj` + `down_proj` W2.5; same clean YAQA/NM slices and disjoint manifest; waits for a free GPU |

| queued | `45a387` / `llama32-1b-w2-reg015-up-down-w25-45a387` | `quantization + divergence300 + gsm8k_platinum_cot` | reg=0.15 reselect; MLP `up_proj` + `down_proj` W2.5; same clean YAQA/NM slices and disjoint manifest; waits for a free GPU |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e` | `divergence300` | complete; metric=0.19916666666666666; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-mlp-only-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e` | `divergence300` | complete; metric=0.16864583333333333; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-seed1-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e` | `divergence300` | complete; metric=0.190625; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-spectral-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25` | `divergence300` | complete; metric=0.13677083333333334; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg0125-main-8ae2db25-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e` | `divergence300` | complete; metric=0.15447916666666667; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg020-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-rolehybrid-qk010-rest005-dev32` | `divergence300` | complete; metric=0.12125; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-rolehybrid-qk010-rest005-dev32-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo-w25-d71136` | `gsm8k_platinum_cot` | complete; metric=0.2547559966914806; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w25-d71136-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg020-replay-aggressive-tip-gpu3` | `gsm8k_platinum_cot` | complete; metric=0.18610421836228289; report `/root/qvq-results/llama32-1b-w2-reg020-replay-aggressive-tip-gpu3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg020-replay-aggressive-tip-gpu3` | `divergence300` | complete; metric=0.17270833333333332; report `/root/qvq-results/llama32-1b-w2-reg020-replay-aggressive-tip-gpu3-div300-dev-v1.json` |

| correction | `d71136`, `bb0aa2`, `45a387` | `effective BPW` | corrected payload-rate estimates: V+O W2.5 = 2.0663 BPW; Gate+Down and Up+Down W2.5 = 2.2990 BPW (including common 0.023168 auxiliary overhead) |

| status | `19d89a` / `llama32-1b-w2-atomic-swiglu-tip-3f17c40b` | `quantization` | started successfully on physical GPU 4 after queue gate changed to require low utilization plus >70 GiB free; D300/GSM8K remain pending |

| monitor | `llama32-1b-w2-reg015-vo-w25-d71136` | `divergence300` | complete; metric=0.19177083333333333; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w25-d71136-div300-dev-v1.json` |

| queued | `738a13` / `llama32-1b-w2-reg015-vo-w30-738a13` | `quantization + divergence300 + gsm8k_platinum_cot` | V+O W3 bitrate-matched against all-attention W2.5; reg=0.15; same clean YAQA/NM calibration and pinned disjointness manifest; queued on physical GPU 7 |

| queued | `09674b` / `llama32-1b-w2-reg015-o-w25-09674b` | `quantization + divergence300 + gsm8k_platinum_cot` | O-only W2.5 attention control; reg=0.15; same clean YAQA/NM calibration and pinned disjointness manifest; queued on physical GPU 0 |

| queued | `0cd45d` / `llama32-1b-w2-reg015-v-w25-0cd45d` | `quantization + divergence300 + gsm8k_platinum_cot` | V-only W2.5 attention control; reg=0.15; same clean YAQA/NM calibration and pinned disjointness manifest; queued on physical GPU 1 |

| queue-adjustment | `09674b` / `llama32-1b-w2-reg015-o-w25-09674b` | `scheduler` | moved from physical GPU 0 (53 GiB free, below the safety threshold) to idle physical GPU 2 (95 GiB free); same output path and configuration, no data reset |

| started | `738a13` / `llama32-1b-w2-reg015-vo-w30-738a13` | `quantization` | started on physical GPU 7 with valid W3 V+O config; YAQA Sketch-B capture in progress |

| started | `09674b` / `llama32-1b-w2-reg015-o-w25-09674b` | `quantization` | started on physical GPU 2 after relocation; YAQA Sketch-B capture in progress |

| started | `0cd45d` / `llama32-1b-w2-reg015-v-w25-0cd45d` | `quantization` | started on physical GPU 1; YAQA Sketch-B capture in progress |

| restarted | `bb0aa2` / `llama32-1b-w2-reg015-gate-down-w25-bb0aa2` | `quantization` | old wrapper was still using the pre-fix memory gate and remained waiting despite an idle GPU; restarted on physical GPU 3 with repaired valid JSON config |

| restarted | `45a387` / `llama32-1b-w2-reg015-up-down-w25-45a387` | `quantization` | old wrapper was still using the pre-fix memory gate and remained waiting despite an idle GPU; restarted on physical GPU 5 with repaired valid JSON config |

| failure+fix | `19d89a` / `llama32-1b-w2-atomic-swiglu-tip-3f17c40b` | `atomic subset staging` | the first post-`dd089f8f` retry reached cleanup but failed because full replay paths were still used to index the layer-relative `StageSubset` dictionary (`KeyError: model.layers.0.mlp.gate_proj`). Fixed in `85d43bba` by resolving wrappers through `NamedModule.full_name`; regression coverage now exercises relative subset keys against a nested model tree. The Atomic arm was restarted on physical GPU 4 with the same disjoint replay rows. |

| protocol-fix | `review-c1048540` | `Divergence-300 decoding` | replaced `generate(min_new_tokens=32)` with an explicit 32-step argmax loop; EOS is now an ordinary token and no stopping processor changes the fixed horizon. Existing D300 scores produced by the old EOS-suppressing protocol require rerun. |

| contamination-fix | `review-c1048540` | `disjointness` | manifests now bind selected calibration slices and both D300 development/locked JSONL files by SHA-256; benchmark quantization fails closed with `--require-disjointness`. The former 959-row source-shaped artifact is marked invalid after 56 locked-split normalized collisions. |

| data-fix | `review-c1048540` | `D300-source calibration builder` | source-mix construction excludes the union of development and locked D300 prompt hashes (exact and normalized). Regenerated output is 981 rows and passes `docs/experiments/disjointness-div300-sources-v2.json`. |

| reporting-fix | `review-c1048540` | `evaluation snapshots` | checkpoint snapshots are now digest-qualified and append-only; legacy fixed filenames are compatibility aliases written only when absent, so prior evidence cannot be overwritten. |

| invalidated+restarted | `19d89a`, `ebec00` | `Atomic/Smooth+Atomic replay` | prior workers used the 959-row replay artifact that collides with locked D300 prompts; those processes were stopped before completion. Atomic was relaunched with regenerated 981-row source mix, strict bound manifest `disjointness-llama32-benchmark-replay-v2.json`, and corrected literal-greedy code; Smooth+Atomic is waiting on that clean checkpoint. |

| invalidated+stopped | `llama32-1b-w2-reg020-replay-256x256-tip-gpu2` | `replay quantization` | this in-progress arm also consumed the historical 959-row replay artifact; its quantizer and waiting D300 workers were stopped before any completion marker or score was published. |

| wording-correction | `review-c1048540` | `25% comparison target` | historical ledger references to a “25% target” are retained for traceability but mean an internal aligned-token development heuristic only; Unsloth's scalar aggregation is unpublished, so no numerical equivalence is claimed. |

| monitor | `llama32-1b-w2-reg015-o-w25-09674b` | `gsm8k_platinum_cot` | complete; metric=0.22994210090984285; report `/root/qvq-results/llama32-1b-w2-reg015-o-w25-09674b-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo-w30-738a13` | `gsm8k_platinum_cot` | complete; metric=0.24731182795698925; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w30-738a13-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-v-w25-0cd45d` | `gsm8k_platinum_cot` | complete; metric=0.2142266335814723; report `/root/qvq-results/llama32-1b-w2-reg015-v-w25-0cd45d-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-gate-down-w25-bb0aa2` | `gsm8k_platinum_cot` | complete; metric=0.27956989247311825; report `/root/qvq-results/llama32-1b-w2-reg015-gate-down-w25-bb0aa2-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-up-down-w25-45a387` | `gsm8k_platinum_cot` | complete; metric=0.2812241521918941; report `/root/qvq-results/llama32-1b-w2-reg015-up-down-w25-45a387-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo-w30-738a13` | `divergence300` | complete; metric=0.17927083333333332; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w30-738a13-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-v-w25-0cd45d` | `divergence300` | complete; metric=0.17291666666666666; report `/root/qvq-results/llama32-1b-w2-reg015-v-w25-0cd45d-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-gate-down-w25-bb0aa2` | `divergence300` | complete; metric=0.19479166666666667; report `/root/qvq-results/llama32-1b-w2-reg015-gate-down-w25-bb0aa2-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-o-w25-09674b` | `divergence300` | complete; metric=0.17302083333333335; report `/root/qvq-results/llama32-1b-w2-reg015-o-w25-09674b-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-up-down-w25-45a387` | `divergence300` | complete; metric=0.19958333333333333; report `/root/qvq-results/llama32-1b-w2-reg015-up-down-w25-45a387-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-atomic-swiglu-tip-3f17c40b` | `gsm8k_platinum_cot` | complete; metric=0.19933829611248965; report `/root/qvq-results/llama32-1b-w2-atomic-swiglu-tip-3f17c40b-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-atomic-swiglu-tip-3f17c40b` | `divergence300` | complete; metric=0.1659375; report `/root/qvq-results/llama32-1b-w2-atomic-swiglu-tip-3f17c40b-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-smooth-atomic-swiglu-tip-3f17c40b` | `gsm8k_platinum_cot` | complete; metric=0.22249793217535152; report `/root/qvq-results/llama32-1b-w2-smooth-atomic-swiglu-tip-3f17c40b-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-smooth-atomic-swiglu-tip-3f17c40b` | `divergence300` | complete; metric=0.17614583333333333; report `/root/qvq-results/llama32-1b-w2-smooth-atomic-swiglu-tip-3f17c40b-div300-dev-v1.json` |

| queued | `0f642c` | `up-down-w30-0f642c` | Up+Down W3; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_up_down_w30.json`; physical GPU 0; clean YAQA/NM calibration with `disjointness-llama32-benchmark-v2.json`; D300/GSM8K queued by monitor |

| queued | `569a95` | `up-down-w35-569a95` | Up+Down W3.5; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_up_down_w35.json`; physical GPU 1; clean YAQA/NM calibration with `disjointness-llama32-benchmark-v2.json`; D300/GSM8K queued by monitor |

| failed | `cdfa75` | `up-down-w40-cdfa75` | Up+Down W4; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_up_down_w40.json` (`format=qvq_v4`); physical GPU 2; failed in `gptqmodel/quantization/qvq.py:1327` because trusted CUDA Viterbi supports only V2 sequences; no checkpoint or benchmark result. |

| queued | `ef21af` | `vo-w30-ef21af` | V+O W3; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo_w30.json`; physical GPU 3; clean YAQA/NM calibration with `disjointness-llama32-benchmark-v2.json`; D300/GSM8K queued by monitor |

| queued | `60a68a` | `vo-w35-60a68a` | V+O W3.5; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo_w35.json`; physical GPU 4; clean YAQA/NM calibration with `disjointness-llama32-benchmark-v2.json`; D300/GSM8K queued by monitor |

| failed | `98daf5` | `vo-w40-98daf5` | V+O W4; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo_w40.json` (`format=qvq_v4`); physical GPU 5; failed in `gptqmodel/quantization/qvq.py:1327` because trusted CUDA Viterbi supports only V2 sequences; no checkpoint or benchmark result. |

| failed+requeued | `151a0a`, `c17e71` | `W4 initial launch` | Initial W4 configs used `qvq_v2b2_p32`, which correctly rejected rates above W3.5. Replaced with model-wide `qvq_v4` configs and requeued as `cdfa75` and `98daf5`; no checkpoint or benchmark result was produced by the failed attempts. |

| queued | `d13602` | `V+O W2 + Smooth + Atomic SwiGLU replay` | config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo_w20_smooth_atomic.json`; output `/root/qvq-results/llama32-1b-w2-vo-w20-smooth-atomic-d13602`; physical GPU 6; base W2 with explicit V/O W2 override, Smooth-SwiGLU enabled, atomic gate/up/down replay; clean NM rows 0:128, YAQA rows 0:182, replay search rows 0:32 and confirmation rows 32:64; strict manifest `disjointness-llama32-benchmark-replay-v2.json`; D300 and GSM8K Platinum queued after quantization. |

| queued | `b7d172` | `V+O W2.5 + Smooth + Atomic SwiGLU replay` | config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo_w25_smooth_atomic.json`; output `/root/qvq-results/llama32-1b-w2-vo-w25-smooth-atomic-b7d172`; physical GPU 7; V/O W2.5 dynamic override, Smooth-SwiGLU enabled, atomic gate/up/down replay; clean NM rows 0:128, YAQA rows 0:182, replay search rows 0:32 and confirmation rows 32:64; strict manifest `disjointness-llama32-benchmark-replay-v2.json`; D300 and GSM8K Platinum queued after quantization. |

| started | `d13602`, `b7d172` | `V+O Smooth + Atomic quantization` | both queue wrappers passed the atomic-checkpoint gate and started successfully on physical GPUs 6 and 7 at commit `9f2246d5`; output directories are `/root/qvq-results/llama32-1b-w2-vo-w20-smooth-atomic-d13602` and `/root/qvq-results/llama32-1b-w2-vo-w25-smooth-atomic-b7d172`. D300 and GSM8K Platinum remain pending until each `qvq_quantize_run.json` is published; the 120-second evaluation watcher will schedule them automatically. |

| monitor | `llama32-1b-w2-reg015-vo-w30-ef21af` | `gsm8k_platinum_cot` | complete; metric=0.24731182795698925; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w30-ef21af-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo-w35-60a68a` | `gsm8k_platinum_cot` | complete; metric=0.2663358147229115; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w35-60a68a-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-up-down-w30-0f642c` | `gsm8k_platinum_cot` | complete; metric=0.3068651778329198; report `/root/qvq-results/llama32-1b-w2-reg015-up-down-w30-0f642c-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-up-down-w35-569a95` | `gsm8k_platinum_cot` | complete; metric=0.31679073614557485; report `/root/qvq-results/llama32-1b-w2-reg015-up-down-w35-569a95-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo-w30-ef21af` | `divergence300` | complete; metric=0.17927083333333332; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w30-ef21af-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo-w35-60a68a` | `divergence300` | complete; metric=0.19083333333333333; report `/root/qvq-results/llama32-1b-w2-reg015-vo-w35-60a68a-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-up-down-w30-0f642c` | `divergence300` | complete; metric=0.21739583333333334; report `/root/qvq-results/llama32-1b-w2-reg015-up-down-w30-0f642c-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-up-down-w35-569a95` | `divergence300` | complete; metric=0.21302083333333333; report `/root/qvq-results/llama32-1b-w2-reg015-up-down-w35-569a95-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-vo-w20-smooth-atomic-d13602` | `gsm8k_platinum_cot` | complete; metric=0.22249793217535152; report `/root/qvq-results/llama32-1b-w2-vo-w20-smooth-atomic-d13602-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-vo-w25-smooth-atomic-b7d172` | `gsm8k_platinum_cot` | complete; metric=0.2531017369727047; report `/root/qvq-results/llama32-1b-w2-vo-w25-smooth-atomic-b7d172-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-vo-w20-smooth-atomic-d13602` | `divergence300` | complete; metric=0.17614583333333333; report `/root/qvq-results/llama32-1b-w2-vo-w20-smooth-atomic-d13602-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-vo-w25-smooth-atomic-b7d172` | `divergence300` | complete; metric=0.19083333333333333; report `/root/qvq-results/llama32-1b-w2-vo-w25-smooth-atomic-b7d172-div300-dev-v1.json` |

| queued | `595f38` | `V+O W3.5 + Up+Down W3` | GPU 0; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown30.json`; output `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown30-595f38`; flat `qvq_v2b2_p32`, two banks, reg .15; clean NM rows 0:128 + YAQA rows 0:182; strict manifest `disjointness-llama32-benchmark-v2.json`; D300/GSM8K queued after quantization |

| queued | `bf96be` | `V+O W3.5 + Up+Down W3 + Atomic` | GPU 1; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown30_atomic.json`; output `/root/qvq-results/llama32-1b-w2-vo35-updown30-atomic-bf96be`; clean NM/YAQA slices; replay search rows 0:32 and confirmation rows 32:64; strict replay manifest `disjointness-llama32-benchmark-replay-v2.json`; D300/GSM8K queued after quantization |

| queued | `bf272e` | `V+O W3.5 + Up+Down W3 + Smooth+Atomic` | GPU 2; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown30_smooth_atomic.json`; output `/root/qvq-results/llama32-1b-w2-vo35-updown30-smooth-atomic-bf272e`; Smooth group size 16 / max 512 tokens plus atomic replay; clean disjoint slices and replay rows; D300/GSM8K queued after quantization |

| queued | `56c940` | `V+O W2.5 + Up+Down W3` | GPU 3; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo25_updown30.json`; output `/root/qvq-results/llama32-1b-w2-reg015-vo25-updown30-56c940`; flat V2B2/P32 controls and clean disjoint calibration; D300/GSM8K queued after quantization |

| queued | `e5ca9f` | `V+O W3.5 + Up+Down W3.5` | GPU 4; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_updown35.json`; output `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown35-e5ca9f`; flat V2B2/P32 controls and clean disjoint calibration; D300/GSM8K queued after quantization |

| queued | `baeb18` | `V+O W3.5 + all MLP W2.5` | GPU 5; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_mlpall25.json`; output `/root/qvq-results/llama32-1b-w2-reg015-vo35-mlpall25-baeb18`; flat V2B2/P32 controls and clean disjoint calibration; D300/GSM8K queued after quantization |

| queued | `95ef88` | `Up+Down W3 + Atomic` | GPU 6; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_updown30_atomic.json`; output `/root/qvq-results/llama32-1b-w2-updown30-atomic-95ef88`; clean NM/YAQA slices; replay search rows 0:32 and confirmation rows 32:64; strict replay manifest; D300/GSM8K queued after quantization |

| queued | `4e4d0d` | `Up+Down W3 + Smooth+Atomic` | GPU 7; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_updown30_smooth_atomic.json`; output `/root/qvq-results/llama32-1b-w2-updown30-smooth-atomic-4e4d0d`; Smooth group size 16 / max 512 tokens plus atomic replay; clean disjoint slices and replay rows; D300/GSM8K queued after quantization |

| started | `595f38`, `bf96be`, `bf272e`, `56c940`, `e5ca9f`, `baeb18`, `95ef88`, `4e4d0d` | `8-GPU mixed-rate matrix` | queue wrappers launched on physical GPUs 0--7; all eight quantizers passed the low-utilization/free-memory gate and are loading/capturing the 302,193-token YAQA mix (atomic arms also load 13,147-token search and 22,609-token confirmation slices); D300 and GSM8K Platinum remain pending until each checkpoint marker is published; monitor will schedule both automatically |

| queued | `1fee11` | `V+O W2.5 + all MLP W2.5` | GPU 0; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo25_mlpall25.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo25-mlpall25-1fee11`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `b2dee2` | `V+O W3 + all MLP W2.5` | GPU 1; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo30_mlpall25.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo30-mlpall25-b2dee2`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `dc38d2` | `Q+K W2.5 + V+O W3.5 + all MLP W2.5` | GPU 2; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_qk25_vo35_mlpall25.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-mlpall25-dc38d2`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `8dd86a` | `V+O W3.5 + Gate W2.5 + Up/Down W3` | GPU 3; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_gate25_updown30.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown30-8dd86a`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `5fddf5` | `V+O W3.5 + all MLP W3` | GPU 4; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_mlpall30.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-mlpall30-5fddf5`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `83b60c` | `Q+K W2.5 + V+O W3.5 + Up/Down W3.5` | GPU 5; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_qk25_vo35_updown35.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-updown35-83b60c`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `4f018a` | `V+O W3.5 + Gate W2.5 + Up/Down W3.5` | GPU 6; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_gate25_updown35.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown35-4f018a`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |
| queued | `8e67f6` | `V+O W3.5 + Gate W3.5 + Up/Down W3` | GPU 7; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_vo35_gate35_updown30.json`; output `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate35-updown30-8e67f6`; plain V2B2/P32 reselect, reg .15, clean NM/YAQA calibration, strict benchmark manifest; D300/GSM8K queued after quantization |

| started | `1fee11`, `b2dee2`, `dc38d2`, `8dd86a`, `5fddf5`, `83b60c`, `4f018a`, `8e67f6` | `Pareto allocation sweep` | all eight queue wrappers launched with `setsid` on physical GPUs 0--7 at 03:20 UTC; quantizers are active and loading the 302,193-token YAQA mix; monitor will schedule D300 and GSM8K Platinum after each checkpoint marker |

| queued | `88be04` | `flat W2 (Q/K/V/O + gate/up/down)` | GPU 0; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w20.json`; output `/root/qvq-results/llama32-1b-flat-flat-w20-88be04`; V2B2/P32, reg .15, clean NM/YAQA calibration, strict benchmark manifest; queued behind active Pareto arm |
| queued | `509b7f` | `flat W2.5 (Q/K/V/O + gate/up/down)` | GPU 1; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w25.json`; output `/root/qvq-results/llama32-1b-flat-flat-w25-509b7f`; V2B2/P32, reg .15, clean NM/YAQA calibration, strict benchmark manifest; queued behind active Pareto arm |
| queued | `1040a5` | `flat W3 (Q/K/V/O + gate/up/down)` | GPU 2; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w30.json`; output `/root/qvq-results/llama32-1b-flat-flat-w30-1040a5`; V2B2/P32, reg .15, clean NM/YAQA calibration, strict benchmark manifest; queued behind active Pareto arm |
| queued | `b72667` | `flat W3.5 (Q/K/V/O + gate/up/down)` | GPU 3; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w35.json`; output `/root/qvq-results/llama32-1b-flat-flat-w35-b72667`; V2B2/P32, reg .15, clean NM/YAQA calibration, strict benchmark manifest; queued behind active Pareto arm |
| queued | `862367` | `flat W1.5 (Q/K/V/O + gate/up/down)` | GPU 4; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_flat_w15.json`; output `/root/qvq-results/llama32-1b-flat-flat-w15-862367`; V2B2/P32, reg .15, clean NM/YAQA calibration, strict benchmark manifest; queued behind active Pareto arm |

| started | `88be04`, `509b7f`, `1040a5`, `b72667`, `862367` | `flat-rate baseline sweep` | five detached queue wrappers launched with `setsid` at 03:35 UTC on physical GPUs 0--4; all are waiting for their assigned GPU to become idle, then will quantize and trigger D300/GSM8K automatically |

| started | `862367` | `flat W1.5` | GPU 4 became available and quantization started; W2/W2.5/W3/W3.5 wrappers remain queued behind active work |

| queued | `add422` | `W3 anchor: Q/K W2.5 + V/O W3.5 + MLP W3` | assigned idle GPU 0; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_mlp3.json`; output `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422`; clean YAQA/NM calibration and strict benchmark manifest |
| queued | `3151c6` | `W3 anchor: Q/K W3 + V/O W3.5 + MLP W3` | assigned idle GPU 1; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk3_vo35_mlp3.json`; output `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6`; clean YAQA/NM calibration and strict benchmark manifest |
| queued | `2ae00f` | `W3 anchor: Q/K W2.5 + V/O W3.5 + Gate W3 + Up W3.5 + Down W3` | assigned idle GPU 4; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_gate3_up35_down3.json`; output `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f`; clean YAQA/NM calibration and strict benchmark manifest |
| queued | `457643` | `W3 anchor: Q/K W2.5 + V/O W3.5 + Gate/Up W3 + Down W3.5` | assigned idle GPU 5; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_gate3_up3_down35.json`; output `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643`; clean YAQA/NM calibration and strict benchmark manifest |
| queued | `b3bdcd` | `W3 anchor: Q/K W2.5 + V/O W3.5 + MLP W3.5` | assigned idle GPU 6; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk25_vo35_mlp35.json`; output `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd`; clean YAQA/NM calibration and strict benchmark manifest |
| queued | `370e9f` | `W3 anchor: Q/K W3 + V/O W3.5 + MLP W3.5` | assigned GPU 7; config `scripts/configs/llama32_1b_v2b2_p32_yaqa_reg015_w3anchor_qk3_vo35_mlp35.json`; output `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f`; queued until GPU 7 is idle |

| started | `add422`, `3151c6`, `2ae00f`, `457643`, `b3bdcd`, `370e9f` | `W3-anchor reallocation sweep` | six detached queue wrappers launched at 06:19 UTC; five are quantizing on currently idle GPUs and one is waiting on GPU 7; monitor will schedule D300/GSM8K after each checkpoint |

| monitor | `llama32-1b-w2-reg015-vo25-updown30-56c940` | `gsm8k_platinum_cot` | complete; metric=0.3465674110835401; report `/root/qvq-results/llama32-1b-w2-reg015-vo25-updown30-56c940-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo35-updown30-595f38` | `gsm8k_platinum_cot` | complete; metric=0.3738626964433416; report `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown30-595f38-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo35-updown35-e5ca9f` | `gsm8k_platinum_cot` | complete; metric=0.40281224152191897; report `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown35-e5ca9f-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo35-mlpall25-baeb18` | `gsm8k_platinum_cot` | complete; metric=0.3523573200992556; report `/root/qvq-results/llama32-1b-w2-reg015-vo35-mlpall25-baeb18-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo25-updown30-56c940` | `divergence300` | complete; metric=0.21145833333333333; report `/root/qvq-results/llama32-1b-w2-reg015-vo25-updown30-56c940-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo35-mlpall25-baeb18` | `divergence300` | complete; metric=0.21875; report `/root/qvq-results/llama32-1b-w2-reg015-vo35-mlpall25-baeb18-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo35-updown30-595f38` | `divergence300` | complete; metric=0.23760416666666667; report `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown30-595f38-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-vo35-updown35-e5ca9f` | `divergence300` | complete; metric=0.2555208333333333; report `/root/qvq-results/llama32-1b-w2-reg015-vo35-updown35-e5ca9f-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-updown30-atomic-95ef88` | `gsm8k_platinum_cot` | complete; metric=0.31679073614557485; report `/root/qvq-results/llama32-1b-w2-updown30-atomic-95ef88-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-updown30-smooth-atomic-4e4d0d` | `gsm8k_platinum_cot` | complete; metric=0.3159636062861869; report `/root/qvq-results/llama32-1b-w2-updown30-smooth-atomic-4e4d0d-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-vo35-updown30-smooth-atomic-bf272e` | `gsm8k_platinum_cot` | complete; metric=0.3655913978494624; report `/root/qvq-results/llama32-1b-w2-vo35-updown30-smooth-atomic-bf272e-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-updown30-atomic-95ef88` | `divergence300` | complete; metric=0.17375; report `/root/qvq-results/llama32-1b-w2-updown30-atomic-95ef88-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-updown30-smooth-atomic-4e4d0d` | `divergence300` | complete; metric=0.19635416666666666; report `/root/qvq-results/llama32-1b-w2-updown30-smooth-atomic-4e4d0d-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-vo35-updown30-smooth-atomic-bf272e` | `divergence300` | complete; metric=0.24375; report `/root/qvq-results/llama32-1b-w2-vo35-updown30-smooth-atomic-bf272e-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-vo35-updown30-atomic-bf96be` | `gsm8k_platinum_cot` | complete; metric=0.3664185277088503; report `/root/qvq-results/llama32-1b-w2-vo35-updown30-atomic-bf96be-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-vo35-updown30-atomic-bf96be` | `divergence300` | complete; metric=0.243125; report `/root/qvq-results/llama32-1b-w2-vo35-updown30-atomic-bf96be-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo25-mlpall25-1fee11` | `gsm8k_platinum_cot` | complete; metric=0.3564929693961952; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo25-mlpall25-1fee11-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-gate25-updown30-8dd86a` | `gsm8k_platinum_cot` | complete; metric=0.4052936311000827; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown30-8dd86a-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-gate35-updown30-8e67f6` | `gsm8k_platinum_cot` | complete; metric=0.3970223325062035; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate35-updown30-8e67f6-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-qk25-vo35-mlpall25-dc38d2` | `gsm8k_platinum_cot` | complete; metric=0.3705541770057899; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-mlpall25-dc38d2-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-gate25-updown35-4f018a` | `gsm8k_platinum_cot` | complete; metric=0.43010752688172044; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown35-4f018a-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-qk25-vo35-updown35-83b60c` | `gsm8k_platinum_cot` | complete; metric=0.39950372208436724; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-updown35-83b60c-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo30-mlpall25-b2dee2` | `gsm8k_platinum_cot` | complete; metric=0.3631100082712986; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo30-mlpall25-b2dee2-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-qk25-vo35-mlpall25-dc38d2` | `divergence300` | complete; metric=0.2490625; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-mlpall25-dc38d2-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-qk25-vo35-updown35-83b60c` | `divergence300` | complete; metric=0.2891666666666667; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-qk25-vo35-updown35-83b60c-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo25-mlpall25-1fee11` | `divergence300` | complete; metric=0.22791666666666666; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo25-mlpall25-1fee11-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo30-mlpall25-b2dee2` | `divergence300` | complete; metric=0.2328125; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo30-mlpall25-b2dee2-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-gate25-updown30-8dd86a` | `divergence300` | complete; metric=0.27625; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown30-8dd86a-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-mlpall30-5fddf5` | `gsm8k_platinum_cot` | complete; metric=0.40860215053763443; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-mlpall30-5fddf5-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-flat-flat-w15-862367` | `gsm8k_platinum_cot` | complete; metric=0.034739454094292806; report `/root/qvq-results/llama32-1b-flat-flat-w15-862367-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-flat-flat-w30-1040a5` | `gsm8k_platinum_cot` | complete; metric=0.4292803970223325; report `/root/qvq-results/llama32-1b-flat-flat-w30-1040a5-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-gate25-updown35-4f018a` | `divergence300` | complete; metric=0.3198958333333333; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate25-updown35-4f018a-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-gate35-updown30-8e67f6` | `divergence300` | complete; metric=0.31302083333333336; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-gate35-updown30-8e67f6-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-pareto-vo35-mlpall30-5fddf5` | `divergence300` | complete; metric=0.32385416666666667; report `/root/qvq-results/llama32-1b-w2-reg015-pareto-vo35-mlpall30-5fddf5-div300-dev-v1.json` |

| monitor | `llama32-1b-flat-flat-w15-862367` | `divergence300` | complete; metric=0.051354166666666666; report `/root/qvq-results/llama32-1b-flat-flat-w15-862367-div300-dev-v1.json` |

| monitor | `llama32-1b-flat-flat-w35-b72667` | `gsm8k_platinum_cot` | complete; metric=0.45244003308519437; report `/root/qvq-results/llama32-1b-flat-flat-w35-b72667-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-flat-flat-w35-b72667` | `divergence300` | complete; metric=0.3955208333333333; report `/root/qvq-results/llama32-1b-flat-flat-w35-b72667-div300-dev-v1.json` |

| monitor | `llama32-1b-flat-flat-w30-1040a5` | `divergence300` | complete; metric=0.28729166666666667; report `/root/qvq-results/llama32-1b-flat-flat-w30-1040a5-div300-dev-v1.json` |

| monitor | `llama32-1b-flat-flat-w20-88be04` | `gsm8k_platinum_cot` | complete; metric=0.19933829611248965; report `/root/qvq-results/llama32-1b-flat-flat-w20-88be04-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-flat-flat-w25-509b7f` | `gsm8k_platinum_cot` | complete; metric=0.34987593052109184; report `/root/qvq-results/llama32-1b-flat-flat-w25-509b7f-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-flat-flat-w20-88be04` | `divergence300` | complete; metric=0.1290625; report `/root/qvq-results/llama32-1b-flat-flat-w20-88be04-div300-dev-v1.json` |

| monitor | `llama32-1b-flat-flat-w25-509b7f` | `divergence300` | complete; metric=0.2579166666666667; report `/root/qvq-results/llama32-1b-flat-flat-w25-509b7f-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422` | `gsm8k_platinum_cot` | complete; metric=0.42597187758478083; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643` | `gsm8k_platinum_cot` | complete; metric=0.43837882547559964; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f` | `gsm8k_platinum_cot` | complete; metric=0.44086021505376344; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd` | `gsm8k_platinum_cot` | complete; metric=0.43755169561621177; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-reg025-align-gpu4` | `divergence300` | complete; metric=0.1821875; report `/root/qvq-results/llama32-1b-w2-yaqa302k-reg025-align-gpu4-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6` | `gsm8k_platinum_cot` | complete; metric=0.42431761786600497; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643` | `micro_math` | complete; metric='mini_exact=0.03125; delta_ce=-0.021742815628470705; delta_kl=0.04003602080551533; answer_logprob_delta=-0.0032508871448573783; answer_margin_delta=0.1560306833751166; critical_top1=0.965484180249281'; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643-micro-math-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd` | `micro_math` | complete; metric='mini_exact=0.015625; delta_ce=-0.0019490177405047313; delta_kl=0.03215443051687906; answer_logprob_delta=0.13036983671473035; answer_margin_delta=0.5176714142756675; critical_top1=0.9712368168744008'; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd-micro-math-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.018630105641725028; delta_kl=0.04573368571885855; answer_logprob_delta=-0.5056118564819222; answer_margin_delta=-0.061873649483296406; critical_top1=0.968040907638223'; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6-micro-math-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f` | `micro_math` | complete; metric='mini_exact=0.0625; delta_ce=-0.03405955619913879; delta_kl=0.041724604708176904; answer_logprob_delta=0.12168091535568237; answer_margin_delta=1.0290966887972248; critical_top1=0.9677213167146053'; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f-micro-math-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f` | `gsm8k_platinum_cot` | complete; metric=0.4317617866004963; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.03384144711139347; delta_kl=0.04752059730350729; answer_logprob_delta=-0.21475686422034876; answer_margin_delta=0.3830740345058157; critical_top1=0.9661233620965165'; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422-micro-math-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f` | `micro_math` | complete; metric='mini_exact=0.0625; delta_ce=0.012403984347447177; delta_kl=0.03044081982059848; answer_logprob_delta=-0.03612032755097346; answer_margin_delta=0.12432724682252798; critical_top1=0.9725151805688719'; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f-micro-math-v1.json` |

| monitor | `llama32-1b-w2-yaqa302k-layerdamp-align-gpu2` | `divergence300` | complete; metric=0.15739583333333335; report `/root/qvq-results/llama32-1b-w2-yaqa302k-layerdamp-align-gpu2-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643` | `divergence300` | complete; metric=0.37666666666666665; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up3-down35-457643-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f` | `divergence300` | complete; metric=0.35885416666666664; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-gate3-up35-down3-2ae00f-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422` | `divergence300` | complete; metric=0.3509375; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp3-add422-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd` | `divergence300` | complete; metric=0.381875; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk25-vo35-mlp35-b3bdcd-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6` | `divergence300` | complete; metric=0.35125; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp3-3151c6-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f` | `divergence300` | complete; metric=0.3958333333333333; report `/root/qvq-results/llama32-1b-w2-reg015-w3anchor-qk3-vo35-mlp35-370e9f-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-gate2-up35-down35-c2fbf3` | `micro_math` | complete; metric='mini_exact=0.03125; delta_ce=0.0032523682427008167; delta_kl=0.07477968862095022; answer_logprob_delta=-0.2800725244764072; answer_margin_delta=0.7598060494038597; critical_top1=0.9562160434643656'; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-gate2-up35-down35-c2fbf3-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-gate25-up35-down3-3126a8` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.02336849823058306; delta_kl=0.0560215421343802; answer_logprob_delta=0.038635078205991144; answer_margin_delta=1.143054221993062; critical_top1=0.9613294982422499'; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-gate25-up35-down3-3126a8-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-early-bc35b5` | `micro_math` | complete; metric='mini_exact=0.015625; delta_ce=-0.036543823703529425; delta_kl=0.043611477800716815; answer_logprob_delta=-0.07279278805006796; answer_margin_delta=0.9263907475257988; critical_top1=0.9651645893256632'; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-early-bc35b5-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk15-vo35-gate3-up35-down3-365c2d` | `micro_math` | complete; metric='mini_exact=0.015625; delta_ce=-0.015307418543610589; delta_kl=0.0692426967201299; answer_logprob_delta=-0.09730607716005239; answer_margin_delta=0.8060869530065736; critical_top1=0.95845317992969'; report `/root/qvq-results/llama32-1b-w2-w3front-qk15-vo35-gate3-up35-down3-365c2d-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-reg020-qk25-vo35-gate25-up35-down3-f9a5ad` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.02336849823058306; delta_kl=0.0560215421343802; answer_logprob_delta=0.038635078205991144; answer_margin_delta=1.143054221993062; critical_top1=0.9613294982422499'; report `/root/qvq-results/llama32-1b-w2-w3front-reg020-qk25-vo35-gate25-up35-down3-f9a5ad-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk10-vo35-gate3-up35-down3-a96ce0` | `micro_math` | complete; metric='mini_exact=0.03125; delta_ce=0.08307256123096723; delta_kl=0.17377319535635927; answer_logprob_delta=-0.13578394099847593; answer_margin_delta=1.3621755286828796; critical_top1=0.9258549057206775'; report `/root/qvq-results/llama32-1b-w2-w3front-qk10-vo35-gate3-up35-down3-a96ce0-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-late-032830` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.0333439450398843; delta_kl=0.046058909400929916; answer_logprob_delta=0.12845530705665476; answer_margin_delta=0.7590521342718779; critical_top1=0.966762543943752'; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-late-032830-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-reg010-qk25-vo35-gate25-up35-down3-d20085` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.02336849823058306; delta_kl=0.0560215421343802; answer_logprob_delta=0.038635078205991144; answer_margin_delta=1.143054221993062; critical_top1=0.9613294982422499'; report `/root/qvq-results/llama32-1b-w2-w3front-reg010-qk25-vo35-gate25-up35-down3-d20085-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk10-vo35-gate3-up35-down3-a96ce0` | `gsm8k_platinum_cot` | complete; metric=0.22084367245657568; report `/root/qvq-results/llama32-1b-w2-w3front-qk10-vo35-gate3-up35-down3-a96ce0-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-gate2-up35-down35-c2fbf3` | `gsm8k_platinum_cot` | complete; metric=0.39950372208436724; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-gate2-up35-down35-c2fbf3-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-gate25-up35-down3-3126a8` | `gsm8k_platinum_cot` | complete; metric=0.42349048800661704; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-gate25-up35-down3-3126a8-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk15-vo35-gate3-up35-down3-365c2d` | `gsm8k_platinum_cot` | complete; metric=0.3945409429280397; report `/root/qvq-results/llama32-1b-w2-w3front-qk15-vo35-gate3-up35-down3-365c2d-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-early-bc35b5` | `gsm8k_platinum_cot` | complete; metric=0.4119106699751861; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-early-bc35b5-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-reg010-qk25-vo35-gate25-up35-down3-d20085` | `gsm8k_platinum_cot` | complete; metric=0.42349048800661704; report `/root/qvq-results/llama32-1b-w2-w3front-reg010-qk25-vo35-gate25-up35-down3-d20085-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-late-032830` | `gsm8k_platinum_cot` | complete; metric=0.4325889164598842; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-late-032830-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-reg020-qk25-vo35-gate25-up35-down3-f9a5ad` | `gsm8k_platinum_cot` | complete; metric=0.42349048800661704; report `/root/qvq-results/llama32-1b-w2-w3front-reg020-qk25-vo35-gate25-up35-down3-f9a5ad-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk10-vo35-gate3-up35-down3-a96ce0` | `divergence300` | complete; metric=0.19302083333333334; report `/root/qvq-results/llama32-1b-w2-w3front-qk10-vo35-gate3-up35-down3-a96ce0-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-gate2-up35-down35-c2fbf3` | `divergence300` | complete; metric=0.2891666666666667; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-gate2-up35-down35-c2fbf3-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk15-vo35-gate3-up35-down3-365c2d` | `divergence300` | complete; metric=0.2755208333333333; report `/root/qvq-results/llama32-1b-w2-w3front-qk15-vo35-gate3-up35-down3-365c2d-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-gate25-up35-down3-3126a8` | `divergence300` | complete; metric=0.30604166666666666; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-gate25-up35-down3-3126a8-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-early-bc35b5` | `divergence300` | complete; metric=0.35739583333333336; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-early-bc35b5-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-late-032830` | `divergence300` | complete; metric=0.3302083333333333; report `/root/qvq-results/llama32-1b-w2-w3front-qk25-vo35-mlp3-up35-late-032830-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-reg010-qk25-vo35-gate25-up35-down3-d20085` | `divergence300` | complete; metric=0.30604166666666666; report `/root/qvq-results/llama32-1b-w2-w3front-reg010-qk25-vo35-gate25-up35-down3-d20085-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w3front-reg020-qk25-vo35-gate25-up35-down3-f9a5ad` | `divergence300` | complete; metric=0.30604166666666666; report `/root/qvq-results/llama32-1b-w2-w3front-reg020-qk25-vo35-gate25-up35-down3-f9a5ad-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w32-late-gate-down35-5d19be` | `micro_math` | complete; metric='mini_exact=0.0625; delta_ce=-0.02976131507359323; delta_kl=0.041087277049774044; answer_logprob_delta=0.19963203081444128; answer_margin_delta=1.1970082041042953; critical_top1=0.968040907638223'; report `/root/qvq-results/llama32-1b-w2-w32-late-gate-down35-5d19be-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w32-late-down35-7a3d91` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.03254310226949764; delta_kl=0.04110309593289826; answer_logprob_delta=0.15748406079278063; answer_margin_delta=0.9660776622259795; critical_top1=0.9674017257909875'; report `/root/qvq-results/llama32-1b-w2-w32-late-down35-7a3d91-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w37-qk30-fp16-down15-41ce2f` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=0.01148309132129927; delta_kl=0.03015698138352613; answer_logprob_delta=-0.032177195620180954; answer_margin_delta=0.1338091323624796; critical_top1=0.9728347714924896'; report `/root/qvq-results/llama32-1b-w2-w37-qk30-fp16-down15-41ce2f-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w37-qk30-fp16-up15-d53e87` | `micro_math` | complete; metric='mini_exact=0.0625; delta_ce=0.012660188873638798; delta_kl=0.030178287042420045; answer_logprob_delta=-0.03787105474899064; answer_margin_delta=0.1218185709483588; critical_top1=0.9734739533397252'; report `/root/qvq-results/llama32-1b-w2-w37-qk30-fp16-up15-d53e87-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w35-qk25-9ab413` | `micro_math` | complete; metric='mini_exact=0.015625; delta_ce=-0.0019490177405047313; delta_kl=0.03215443051687906; answer_logprob_delta=0.13036983671473035; answer_margin_delta=0.5176714142756675; critical_top1=0.9712368168744008'; report `/root/qvq-results/llama32-1b-w2-w35-qk25-9ab413-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w35-qk30-f0276c` | `micro_math` | complete; metric='mini_exact=0.0625; delta_ce=0.012403984347447177; delta_kl=0.03044081982059848; answer_logprob_delta=-0.03612032755097346; answer_margin_delta=0.12432724682252798; critical_top1=0.9725151805688719'; report `/root/qvq-results/llama32-1b-w2-w35-qk30-f0276c-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w32-mid-down35-c84f7a` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.0446749097048761; delta_kl=0.04016066288985497; answer_logprob_delta=0.2123295553584597; answer_margin_delta=1.018183010727612; critical_top1=0.968040907638223'; report `/root/qvq-results/llama32-1b-w2-w32-mid-down35-c84f7a-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w32-late-gate35-e6b204` | `micro_math` | complete; metric='mini_exact=0.046875; delta_ce=-0.03310974608763225; delta_kl=0.04098139690999987; answer_logprob_delta=0.12644648907789544; answer_margin_delta=1.1331510116804893; critical_top1=0.9705976350271652'; report `/root/qvq-results/llama32-1b-w2-w32-late-gate35-e6b204-micro-math-v1.json` |

| monitor | `llama32-1b-w2-w32-late-down35-7a3d91` | `gsm8k_platinum_cot` | complete; metric=0.4392059553349876; report `/root/qvq-results/llama32-1b-w2-w32-late-down35-7a3d91-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w32-late-gate-down35-5d19be` | `gsm8k_platinum_cot` | complete; metric=0.4334160463192721; report `/root/qvq-results/llama32-1b-w2-w32-late-gate-down35-5d19be-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w32-mid-down35-c84f7a` | `gsm8k_platinum_cot` | complete; metric=0.43424317617866004; report `/root/qvq-results/llama32-1b-w2-w32-mid-down35-c84f7a-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w32-late-gate35-e6b204` | `gsm8k_platinum_cot` | complete; metric=0.43010752688172044; report `/root/qvq-results/llama32-1b-w2-w32-late-gate35-e6b204-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w35-qk25-9ab413` | `gsm8k_platinum_cot` | complete; metric=0.43755169561621177; report `/root/qvq-results/llama32-1b-w2-w35-qk25-9ab413-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w35-qk30-f0276c` | `gsm8k_platinum_cot` | complete; metric=0.4317617866004963; report `/root/qvq-results/llama32-1b-w2-w35-qk30-f0276c-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w37-qk30-fp16-down15-41ce2f` | `gsm8k_platinum_cot` | complete; metric=0.4292803970223325; report `/root/qvq-results/llama32-1b-w2-w37-qk30-fp16-down15-41ce2f-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w37-qk30-fp16-up15-d53e87` | `gsm8k_platinum_cot` | complete; metric=0.43507030603804797; report `/root/qvq-results/llama32-1b-w2-w37-qk30-fp16-up15-d53e87-gsm8k-platinum-v1.json` |

| monitor | `llama32-1b-w2-w32-late-gate-down35-5d19be` | `divergence300` | complete; metric=0.36552083333333335; report `/root/qvq-results/llama32-1b-w2-w32-late-gate-down35-5d19be-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w32-late-down35-7a3d91` | `divergence300` | complete; metric=0.37364583333333334; report `/root/qvq-results/llama32-1b-w2-w32-late-down35-7a3d91-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w32-late-gate35-e6b204` | `divergence300` | complete; metric=0.38375; report `/root/qvq-results/llama32-1b-w2-w32-late-gate35-e6b204-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w32-mid-down35-c84f7a` | `divergence300` | complete; metric=0.38229166666666664; report `/root/qvq-results/llama32-1b-w2-w32-mid-down35-c84f7a-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w35-qk25-9ab413` | `divergence300` | complete; metric=0.381875; report `/root/qvq-results/llama32-1b-w2-w35-qk25-9ab413-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w37-qk30-fp16-down15-41ce2f` | `divergence300` | complete; metric=0.3965625; report `/root/qvq-results/llama32-1b-w2-w37-qk30-fp16-down15-41ce2f-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w35-qk30-f0276c` | `divergence300` | complete; metric=0.3958333333333333; report `/root/qvq-results/llama32-1b-w2-w35-qk30-f0276c-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w37-qk30-fp16-up15-d53e87` | `divergence300` | complete; metric=0.39447916666666666; report `/root/qvq-results/llama32-1b-w2-w37-qk30-fp16-up15-d53e87-div300-dev-v1.json` |

| monitor | `highrate-llama32_1b_highrate_up4_l8_15` | `divergence300` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_up4_l8_15-div300-dev-v1.json` |

| monitor | `highrate-llama32_1b_highrate_flat35_up45_l14_15` | `divergence300` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_flat35_up45_l14_15-div300-dev-v1.json` |

| monitor | `highrate-llama32_1b_highrate_up4_l8_15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_up4_l8_15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_up4_l12_13_up6_l14_15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_up4_l12_13_up6_l14_15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_flat35_up55_l15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_flat35_up55_l15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_up7_l14_15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_up7_l14_15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_flat35_up4_l12_15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_flat35_up4_l12_15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_flat35_up45_l14_15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_flat35_up45_l14_15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_up5_l12_15` | `gsm8k_platinum_cot` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_up5_l12_15-gsm8k-platinum-v1.json` |

| monitor | `highrate-llama32_1b_highrate_flat35_up4_l12_15` | `divergence300` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_flat35_up4_l12_15-div300-dev-v1.json` |

| monitor | `highrate-llama32_1b_highrate_flat35_up55_l15` | `divergence300` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_flat35_up55_l15-div300-dev-v1.json` |

| monitor | `highrate-llama32_1b_highrate_up4_l12_13_up6_l14_15` | `divergence300` | complete; metric=0.0; report `/root/qvq-results/highrate-llama32_1b_highrate_up4_l12_13_up6_l14_15-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-w1fixed_w1_o4_all` | `micro_math` | complete; mini_exact=0.0625; delta_ce=0.02224291131355466; delta_kl=0.038414432980345484; answer_logprob_delta=-0.00970140900184859; answer_margin_delta=0.5846316209479944; critical_top1=0.9728347714924896; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_o4_all-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_down35_l14_15` | `micro_math` | complete; mini_exact=0.078125; delta_ce=0.018829467952278335; delta_kl=0.04057879572445887; answer_logprob_delta=-0.010773078274371019; answer_margin_delta=0.5003627378549149; critical_top1=0.970917225950783; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_down35_l14_15-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_gate35_l14_15` | `micro_math` | complete; mini_exact=0.078125; delta_ce=0.0193485676820823; delta_kl=0.04109152620292587; answer_logprob_delta=-0.1487484285190924; answer_margin_delta=0.2071284393766033; critical_top1=0.9715564077980186; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_gate35_l14_15-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_interleaved` | `micro_math` | complete; mini_exact=0.046875; delta_ce=0.020745615986427615; delta_kl=0.04065807150825085; answer_logprob_delta=-0.17996072902608273; answer_margin_delta=0.23934908055547457; critical_top1=0.9721955896452541; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_interleaved-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l10_13` | `micro_math` | complete; mini_exact=0.078125; delta_ce=0.01569464097475672; delta_kl=0.04056159508173154; answer_logprob_delta=-0.04215912543126007; answer_margin_delta=0.6849863422450735; critical_top1=0.9734739533397252; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l10_13-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l12_15` | `micro_math` | complete; mini_exact=0.03125; delta_ce=0.017248062983803127; delta_kl=0.0401411907854153; answer_logprob_delta=0.004756561410960866; answer_margin_delta=0.5962328270300111; critical_top1=0.9718759987216363; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l12_15-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l8_11` | `micro_math` | complete; mini_exact=0.03125; delta_ce=0.019443390762893815; delta_kl=0.040513154806641236; answer_logprob_delta=-0.03587996603837654; answer_margin_delta=0.49700250198592; critical_top1=0.970917225950783; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l8_11-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_v4_up4_l13_15` | `micro_math` | complete; mini_exact=0.0625; delta_ce=0.012771078163615847; delta_kl=0.039942951749366296; answer_logprob_delta=0.04268171271281456; answer_margin_delta=0.5250160445028277; critical_top1=0.9699584531799297; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_v4_up4_l13_15-micro-math-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_o4_all` | `gsm8k_platinum_cot` | complete; metric=0.43755169561621177 (529/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_o4_all-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_down35_l14_15` | `gsm8k_platinum_cot` | complete; metric=0.4267990074441687 (516/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_down35_l14_15-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_gate35_l14_15` | `gsm8k_platinum_cot` | complete; metric=0.42018196856906537 (508/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_gate35_l14_15-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_interleaved` | `gsm8k_platinum_cot` | complete; metric=0.42018196856906537 (508/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_interleaved-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l10_13` | `gsm8k_platinum_cot` | complete; metric=0.42349048800661704 (512/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l10_13-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l12_15` | `gsm8k_platinum_cot` | complete; metric=0.4185277088502895 (506/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l12_15-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l8_11` | `gsm8k_platinum_cot` | complete; metric=0.43093465674110837 (521/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l8_11-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_v4_up4_l13_15` | `gsm8k_platinum_cot` | complete; metric=0.42762613730355664 (517/1209; invalid=0); report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_v4_up4_l13_15-gsm8k-platinum-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_o4_all` | `divergence300` | complete; aligned_token_top1=0.3486458333333333; exact=16/300; mean_first_divergence=10.496666666666666; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_o4_all-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_down35_l14_15` | `divergence300` | complete; aligned_token_top1=0.35739583333333336; exact=20/300; mean_first_divergence=10.923333333333334; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_down35_l14_15-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_gate35_l14_15` | `divergence300` | complete; aligned_token_top1=0.3734375; exact=20/300; mean_first_divergence=11.213333333333333; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_gate35_l14_15-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_interleaved` | `divergence300` | complete; aligned_token_top1=0.3539583333333333; exact=20/300; mean_first_divergence=11.086666666666666; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_interleaved-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l10_13` | `divergence300` | complete; aligned_token_top1=0.35333333333333333; exact=18/300; mean_first_divergence=10.843333333333334; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l10_13-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l12_15` | `divergence300` | complete; aligned_token_top1=0.35270833333333335; exact=19/300; mean_first_divergence=10.96; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l12_15-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_up4_l8_11` | `divergence300` | complete; aligned_token_top1=0.35260416666666666; exact=18/300; mean_first_divergence=10.673333333333334; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_up4_l8_11-div300-dev-v1.json` |
| monitor | `llama32-1b-w2-w1fixed_w1_v4_up4_l13_15` | `divergence300` | complete; aligned_token_top1=0.37114583333333334; exact=20/300; mean_first_divergence=11.323333333333334; report `/root/qvq-results/llama32-1b-w2-w1fixed_w1_v4_up4_l13_15-div300-dev-v1.json` |

## Wave-3 corrected frontier evaluations (2026-08-28 UTC)

| monitor | arm_id | task | result | report |
| --- | --- | --- | --- | --- |
| monitor | `b6429f` | `micro_math` | complete; mini_exact=0.046875 (3/64); answer_logprob_delta=-0.025164622424253776; answer_margin_delta=0.4311086455387856; critical_top1=0.970917225950783 | `/root/qvq-results/llama32-1b-w2-w3queued_b6429f_llama32_1b_frontier_w3_control_qk25_vo35_g3_u35_d3-micro-math-v3.json` |
| monitor | `42a659` | `micro_math` | complete; mini_exact=0.0625 (4/64); answer_logprob_delta=-0.14303467567287273; answer_margin_delta=0.3544583391787401; critical_top1=0.9725151805688719 | `/root/qvq-results/llama32-1b-w2-w3queued_42a659_llama32_1b_frontier_w3_o4_early_up3-micro-math-v3.json` |
| monitor | `5dc744` | `micro_math` | complete; mini_exact=0.046875 (3/64); answer_logprob_delta=-0.26148495256011167; answer_margin_delta=0.5950916062540083; critical_top1=0.9731543624161074 | `/root/qvq-results/llama32-1b-w2-w3queued_5dc744_llama32_1b_frontier_w3_o4_early_gate25-micro-math-v3.json` |
| monitor | `286d6d` | `micro_math` | complete; mini_exact=0.046875 (3/64); answer_logprob_delta=0.06851246864048403; answer_margin_delta=0.4319104151939278; critical_top1=0.975071907957814 | `/root/qvq-results/llama32-1b-w2-w3queued_286d6d_llama32_1b_frontier_w3_o4_q2-micro-math-v3.json` |
| monitor | `4e7424` | `micro_math` | complete; mini_exact=0.03125 (2/64); answer_logprob_delta=0.6117219907134327; answer_margin_delta=1.9054487997026586; critical_top1=0.975071907957814 | `/root/qvq-results/llama32-1b-w2-w3queued_4e7424_llama32_1b_frontier_w3_flat35_up4_l12_15_seed1-micro-math-v3.json` |
| monitor | `d0be49` | `micro_math` | complete; mini_exact=0.03125 (2/64); answer_logprob_delta=-0.6450147655472827; answer_margin_delta=-0.7907120434205923; critical_top1=0.9776286353467561 | `/root/qvq-results/llama32-1b-w2-w3queued_d0be49_llama32_1b_frontier_w3_flat35_up4_l7_15-micro-math-v3.json` |
| monitor | `b6429f` | `gsm8k_platinum_cot` | complete; accuracy=0.4284532671629446 (518/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w3queued_b6429f_llama32_1b_frontier_w3_control_qk25_vo35_g3_u35_d3-gsm8k-platinum-v3.json` |
| monitor | `42a659` | `gsm8k_platinum_cot` | complete; accuracy=0.4218362282878412 (510/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w3queued_42a659_llama32_1b_frontier_w3_o4_early_up3-gsm8k-platinum-v3.json` |
| monitor | `5dc744` | `gsm8k_platinum_cot` | complete; accuracy=0.42018196856906537 (508/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w3queued_5dc744_llama32_1b_frontier_w3_o4_early_gate25-gsm8k-platinum-v3.json` |
| monitor | `286d6d` | `gsm8k_platinum_cot` | complete; accuracy=0.4358974358974359 (527/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w3queued_286d6d_llama32_1b_frontier_w3_o4_q2-gsm8k-platinum-v3.json` |
| monitor | `4e7424` | `gsm8k_platinum_cot` | complete; accuracy=0.46153846153846156 (558/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w3queued_4e7424_llama32_1b_frontier_w3_flat35_up4_l12_15_seed1-gsm8k-platinum-v3.json` |
| monitor | `d0be49` | `gsm8k_platinum_cot` | complete; accuracy=0.44086021505376344 (533/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w3queued_d0be49_llama32_1b_frontier_w3_flat35_up4_l7_15-gsm8k-platinum-v3.json` |
| monitor | `a7e34b` | `micro_math` | complete; mini_exact=0.03125 (2/64); answer_logprob_delta=-0.9995669491255461; answer_margin_delta=-1.4543243522074685; critical_top1=0.9741131351869607 | `/root/qvq-results/llama32-1b-w2-w2fixed_a7e34b_llama32_1b_frontier_w2_flat35_up4_down4_l12_15-micro-math-v2.json` |
| monitor | `a7e34b` | `gsm8k_platinum_cot` | complete; accuracy=0.4400330851943755 (532/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w2fixed_a7e34b_llama32_1b_frontier_w2_flat35_up4_down4_l12_15-gsm8k-platinum-v2.json` |
| monitor | `9769b1` | `micro_math` | complete; mini_exact=0.03125 (2/64) | `/root/qvq-results/llama32-1b-w2-w2fixed_9769b1_llama32_1b_frontier_w2_flat35_seed1-micro-math-v2.json` |

## Wave-4 W3.2 sensitivity queue (2026-08-28 UTC)

| queue | arm_id | allocation | corrected_estimated_bpw | GPU | state |
| --- | --- | --- | ---: | ---: | --- |
| queue | `c84a1e` | W3.2 anchor, seed 1 | 3.161099 | 0 | quantizing |
| queue | `f1d903` | Anchor + V4 all | 3.204202 | 1 | quantizing |
| queue | `7b6e2a` | Anchor + Up4 layer 6 | 3.169720 | 2 | quantizing |
| queue | `a4c918` | Anchor + Up4 layers 6–7 | 3.178340 | 3 | quantizing |
| queue | `d2f507` | Anchor + V4 all + Up4 layer 6 | 3.212823 | 4 | quantizing |
| queue | `8e3b61` | Anchor + Up4 layers 6–8 | 3.186961 | 5 | quantizing |
| queue | `5a0dce` | Anchor + Up4 layers 6–9 | 3.195582 | 6 | quantizing |
| queue | `b7f294` | Anchor + V4 all + Up4 layers 6–8 | 3.230065 | 7 | quantizing |

## Wave-5 sparse Up-layer isolation queue (2026-08-28 UTC)

| arm_id | allocation | estimated effective BPW | GSM8K Platinum | Mini exact | answer-logprob Δ | state | config | checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `a91f6c` | W3.2 anchor + Up4 layer 9 | 3.169720 | **44.0033% (532/1209)** | 4.6875% | +0.124061 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l9.json` | `/root/qvq-results/llama32-1b-w2-w5queued_a91f6c_llama32_1b_frontier_w5_anchor_up4_l9` |
| `3d7e42` | W3.2 anchor + Up4 layers 6 and 9 | 3.178340 | **43.5070% (526/1209)** | 3.1250% | +0.106767 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l9.json` | `/root/qvq-results/llama32-1b-w2-w5queued_3d7e42_llama32_1b_frontier_w5_anchor_up4_l6_l9` |
| `e8b5a0` | W3.2 anchor + Up4 layers 7 and 8 | 3.178340 | **45.1613% (546/1209)** | 6.2500% | −0.036569 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l7_l8.json` | `/root/qvq-results/llama32-1b-w2-w5queued_e8b5a0_llama32_1b_frontier_w5_anchor_up4_l7_l8` |
| `8980aa` | W3.2 anchor + Up4 layers 6 and 8 | 3.178340 | **45.5749% (551/1209)** | 4.6875% | −0.019751 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l8.json` | `/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_w5_anchor_up4_l6_l8` |
| `1450c0` | W3.2 anchor + Up4 layers 6 and 10 | 3.178340 | **43.7552% (529/1209)** | 4.6875% | −0.025394 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l10.json` | `/root/qvq-results/llama32-1b-w2-w5queued_1450c0_llama32_1b_frontier_w5_anchor_up4_l6_l10` |
| `a63cf6` | W3.2 anchor + Up4 layers 5 and 9 | 3.178340 | **44.5823% (539/1209)** | 4.6875% | +0.086033 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l5_l9.json` | `/root/qvq-results/llama32-1b-w2-w5queued_a63cf6_llama32_1b_frontier_w5_anchor_up4_l5_l9` |
| `aeb16f` | W3.2 anchor + Up4 layers 6, 9, and 12 | 3.186961 | **43.8379% (530/1209)** | 3.1250% | +0.137476 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l9_l12.json` | `/root/qvq-results/llama32-1b-w2-w5queued_aeb16f_llama32_1b_frontier_anchor_up4_l6_l9_l12` |
| `a8920f` | W3.2 anchor + Up4 layers 6, 9, 12, and 15 | 3.195582 | **43.1762% (522/1209)** | 3.1250% | +0.129048 | complete | `scripts/configs/llama32_1b_frontier_w5_anchor_up4_l6_l9_l12_l15.json` | `/root/qvq-results/llama32-1b-w2-w5queued_a8920f_llama32_1b_frontier_anchor_up4_l6_l9_l12_l15` |
| monitor | `9769b1` | `gsm8k_platinum_cot` | complete; accuracy=0.4665012406947891 (564/1209); invalid=0 | `/root/qvq-results/llama32-1b-w2-w2fixed_9769b1_llama32_1b_frontier_w2_flat35_seed1-gsm8k-platinum-v2.json` |

## Wave-5 GSM8K/Mini-GSM completions (2026-08-28 UTC)

All eight sparse Up-layer arms completed both required evaluations on 1,209
GSM8K Platinum rows and 64 Mini-GSM rows. Reports are keyed by the same arm IDs
used by the queue and checkpoint paths.

| arm_id | GSM8K Platinum | Mini exact | answer-logprob Δ | reports |
| --- | ---: | ---: | ---: | --- |
| `a91f6c` | **44.0033% (532/1209)** | 4.6875% | +0.124061 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_a91f6c_llama32_1b_frontier_anchor_up4_l9-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_a91f6c_llama32_1b_frontier_anchor_up4_l9-micro-math-v1.json) |
| `3d7e42` | **43.5070% (526/1209)** | 3.1250% | +0.106767 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_3d7e42_llama32_1b_frontier_anchor_up4_l6_l9-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_3d7e42_llama32_1b_frontier_anchor_up4_l6_l9-micro-math-v1.json) |
| `e8b5a0` | **45.1613% (546/1209)** | 6.2500% | −0.036569 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_e8b5a0_llama32_1b_frontier_anchor_up4_l7_l8-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_e8b5a0_llama32_1b_frontier_anchor_up4_l7_l8-micro-math-v1.json) |
| `8980aa` | **45.5749% (551/1209)** | 4.6875% | −0.019751 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_anchor_up4_l6_l8-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_8980aa_llama32_1b_frontier_anchor_up4_l6_l8-micro-math-v1.json) |
| `1450c0` | **43.7552% (529/1209)** | 4.6875% | −0.025394 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_1450c0_llama32_1b_frontier_anchor_up4_l6_l10-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_1450c0_llama32_1b_frontier_anchor_up4_l6_l10-micro-math-v1.json) |
| `a63cf6` | **44.5823% (539/1209)** | 4.6875% | +0.086033 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_a63cf6_llama32_1b_frontier_anchor_up4_l5_l9-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_a63cf6_llama32_1b_frontier_anchor_up4_l5_l9-micro-math-v1.json) |
| `aeb16f` | **43.8379% (530/1209)** | 3.1250% | +0.137476 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_aeb16f_llama32_1b_frontier_anchor_up4_l6_l9_l12-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_aeb16f_llama32_1b_frontier_anchor_up4_l6_l9_l12-micro-math-v1.json) |
| `a8920f` | **43.1762% (522/1209)** | 3.1250% | +0.129048 | [`gsm`](/root/qvq-results/llama32-1b-w2-w5queued_a8920f_llama32_1b_frontier_anchor_up4_l6_l9_l12_l15-gsm8k-platinum-v1.json), [`mini`](/root/qvq-results/llama32-1b-w2-w5queued_a8920f_llama32_1b_frontier_anchor_up4_l6_l9_l12_l15-micro-math-v1.json) |

## Wave-4 GSM8K completions (partial, 2026-08-28 UTC)

Seven Wave-4 held-out GSM8K Platinum evaluations have completed with nonzero scores and zero invalid-generation indications in the evaluator logs. Mini-GSM is running afterward; arm `f1d903` remains in GSM8K evaluation.

| arm_id | GSM8K Platinum | report | next state |
| --- | ---: | --- | --- |
| `c84a1e` | **43.5070% (526/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_c84a1e_llama32_1b_frontier_w4_anchor_seed1-gsm8k-platinum-v1.json` | Mini-GSM running |
| `7b6e2a` | **44.2514% (535/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_7b6e2a_llama32_1b_frontier_w4_anchor_up4_l6-gsm8k-platinum-v1.json` | Mini-GSM running |
| `a4c918` | **44.0033% (532/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_a4c918_llama32_1b_frontier_w4_anchor_up4_l6_7-gsm8k-platinum-v1.json` | Mini-GSM running |
| `d2f507` | **43.5897% (527/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_d2f507_llama32_1b_frontier_w4_anchor_v4_up4_l6-gsm8k-platinum-v1.json` | Mini-GSM running |
| `8e3b61` | **44.0860% (533/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_8e3b61_llama32_1b_frontier_w4_anchor_up4_l6_8-gsm8k-platinum-v1.json` | Mini-GSM running |
| `5a0dce` | **44.9959% (544/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_5a0dce_llama32_1b_frontier_w4_anchor_up4_l6_9-gsm8k-platinum-v1.json` | Mini-GSM running |
| `b7f294` | **43.1762% (522/1209)** | `/root/qvq-results/llama32-1b-w2-w4queued_b7f294_llama32_1b_frontier_w4_anchor_v4_up4_l6_8-gsm8k-platinum-v1.json` | Mini-GSM running |

## Wave-4 evaluation completions (2026-08-28 UTC)

All eight Wave-4 arms completed GSM8K Platinum and Mini-GSM with reports keyed by arm ID. GSM8K was evaluated on 1,209 rows; Mini-GSM used 64 rows. The compact fields below are the screening metrics; full JSON reports contain the complete metric payload.

| arm_id | GSM8K Platinum | Mini exact | answer-logprob Δ | answer-margin Δ | reports |
| --- | ---: | ---: | ---: | ---: | --- |
| `c84a1e` | **43.5070% (526/1209)** | 1.5625% | +0.524729 | +2.329941 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_c84a1e_llama32_1b_frontier_w4_anchor_seed1-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_c84a1e_llama32_1b_frontier_w4_anchor_seed1-micro-math-v1.json) |
| `f1d903` | **44.0860% (533/1209)** | 4.6875% | +0.042380 | +0.356396 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_f1d903_llama32_1b_frontier_w4_anchor_v4_all-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_f1d903_llama32_1b_frontier_w4_anchor_v4_all-micro-math-v1.json) |
| `7b6e2a` | **44.2514% (535/1209)** | 4.6875% | +0.052418 | +0.329762 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_7b6e2a_llama32_1b_frontier_w4_anchor_up4_l6-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_7b6e2a_llama32_1b_frontier_w4_anchor_up4_l6-micro-math-v1.json) |
| `a4c918` | **44.0033% (532/1209)** | 4.6875% | +0.052846 | +0.297890 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_a4c918_llama32_1b_frontier_w4_anchor_up4_l6_7-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_a4c918_llama32_1b_frontier_w4_anchor_up4_l6_7-micro-math-v1.json) |
| `d2f507` | **43.5897% (527/1209)** | 4.6875% | +0.031549 | +0.257762 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_d2f507_llama32_1b_frontier_w4_anchor_v4_up4_l6-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_d2f507_llama32_1b_frontier_w4_anchor_v4_up4_l6-micro-math-v1.json) |
| `8e3b61` | **44.0860% (533/1209)** | 3.1250% | −0.037423 | +0.150835 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_8e3b61_llama32_1b_frontier_w4_anchor_up4_l6_8-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_8e3b61_llama32_1b_frontier_w4_anchor_up4_l6_8-micro-math-v1.json) |
| `5a0dce` | **45.0000% (544/1209)** | 3.1250% | +0.029364 | +0.225668 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_5a0dce_llama32_1b_frontier_w4_anchor_up4_l6_9-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_5a0dce_llama32_1b_frontier_w4_anchor_up4_l6_9-micro-math-v1.json) |
| `b7f294` | **43.1762% (522/1209)** | 3.1250% | −0.032823 | +0.112466 | [`gsm8k`](/root/qvq-results/llama32-1b-w2-w4queued_b7f294_llama32_1b_frontier_w4_anchor_v4_up4_l6_8-gsm8k-platinum-v1.json), [`micro`](/root/qvq-results/llama32-1b-w2-w4queued_b7f294_llama32_1b_frontier_w4_anchor_v4_up4_l6_8-micro-math-v1.json) |

## Wave-4 quantization completions (2026-08-28 UTC)

All eight Wave-4 W3.2 sensitivity arms completed quantization successfully; held-out Mini-GSM and GSM8K Platinum evaluations are queued on the now-free GPUs. No evaluation score is inferred from quantization completion.

| arm_id | GPU | allocation | corrected estimated BPW | quantization finished | checkpoint | evaluation state |
| --- | ---: | --- | ---: | --- | --- | --- |
| `c84a1e` | 0 | W3.2 anchor, seed 1 | 3.161099 | 14:18:40Z | `/root/qvq-results/llama32-1b-w2-w4queued_c84a1e_llama32_1b_frontier_w4_anchor_seed1` | queued: micro_math, GSM8K Platinum |
| `f1d903` | 1 | Anchor + V4 all | 3.204202 | 14:21:13Z | `/root/qvq-results/llama32-1b-w2-w4queued_f1d903_llama32_1b_frontier_w4_anchor_v4_all` | queued: micro_math, GSM8K Platinum |
| `7b6e2a` | 2 | Anchor + Up4 layer 6 | 3.169720 | 14:18:37Z | `/root/qvq-results/llama32-1b-w2-w4queued_7b6e2a_llama32_1b_frontier_w4_anchor_up4_l6` | queued: micro_math, GSM8K Platinum |
| `a4c918` | 3 | Anchor + Up4 layers 6–7 | 3.178340 | 14:17:53Z | `/root/qvq-results/llama32-1b-w2-w4queued_a4c918_llama32_1b_frontier_w4_anchor_up4_l6_7` | queued: micro_math, GSM8K Platinum |
| `d2f507` | 4 | Anchor + V4 all + Up4 layer 6 | 3.212823 | 14:17:49Z | `/root/qvq-results/llama32-1b-w2-w4queued_d2f507_llama32_1b_frontier_w4_anchor_v4_up4_l6` | queued: micro_math, GSM8K Platinum |
| `8e3b61` | 5 | Anchor + Up4 layers 6–8 | 3.186961 | 14:17:58Z | `/root/qvq-results/llama32-1b-w2-w4queued_8e3b61_llama32_1b_frontier_w4_anchor_up4_l6_8` | queued: micro_math, GSM8K Platinum |
| `5a0dce` | 6 | Anchor + Up4 layers 6–9 | 3.195582 | 14:17:18Z | `/root/qvq-results/llama32-1b-w2-w4queued_5a0dce_llama32_1b_frontier_w4_anchor_up4_l6_9` | queued: micro_math, GSM8K Platinum |
| `b7f294` | 7 | Anchor + V4 all + Up4 layers 6–8 | 3.230065 | 14:16:19Z | `/root/qvq-results/llama32-1b-w2-w4queued_b7f294_llama32_1b_frontier_w4_anchor_v4_up4_l6_8` | queued: micro_math, GSM8K Platinum |

## Wave-6 L8 interaction mapping queue (2026-08-28 UTC)

Eight matched-budget controls were quantized from the corrected W3.2 anchor;
full GSM8K Platinum and Mini-GSM evaluations have now completed for every
checkpoint.

| arm_id | allocation | estimated effective BPW | GSM8K Platinum | Mini exact | Answer-logprob Δ | state | config | checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `35d823` | W3.2 anchor + Up4 layer 8 | 3.169720 | **44.8304% (542/1209)** | 6.25% | −0.007874 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l8.json` | `/root/qvq-results/llama32-1b-w2-w6queued_35d823_llama32_1b_frontier_anchor_up4_l8` |
| `4b5abc` | W3.2 anchor + Up4 layers 5 and 8 | 3.178340 | **44.5823% (539/1209)** | 6.25% | −0.048598 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l5_l8.json` | `/root/qvq-results/llama32-1b-w2-w6queued_4b5abc_llama32_1b_frontier_anchor_up4_l5_l8` |
| `a35e15` | W3.2 anchor + Up4 layers 8 and 9 | 3.178340 | **44.9132% (543/1209)** | 4.69% | +0.067993 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l8_l9.json` | `/root/qvq-results/llama32-1b-w2-w6queued_a35e15_llama32_1b_frontier_anchor_up4_l8_l9` |
| `fd5aff` | W3.2 anchor + Up4 layers 8 and 10 | 3.178340 | **44.4169% (537/1209)** | 4.69% | −0.079265 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l8_l10.json` | `/root/qvq-results/llama32-1b-w2-w6queued_fd5aff_llama32_1b_frontier_anchor_up4_l8_l10` |
| `8b2e24` | W3.2 anchor + Up4 layers 8 and 12 | 3.178340 | **43.8379% (530/1209)** | 6.25% | +0.022460 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l8_l12.json` | `/root/qvq-results/llama32-1b-w2-w6queued_8b2e24_llama32_1b_frontier_anchor_up4_l8_l12` |
| `7aef62` | W3.2 anchor + Up4 layers 8 and 15 | 3.178340 | **43.9206% (531/1209)** | 6.25% | −0.021847 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l8_l15.json` | `/root/qvq-results/llama32-1b-w2-w6queued_7aef62_llama32_1b_frontier_anchor_up4_l8_l15` |
| `18536a` | W3.2 anchor + Up4 layers 6, 8, and 9 | 3.186961 | **44.3342% (536/1209)** | 3.13% | +0.047220 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l6_l8_l9.json` | `/root/qvq-results/llama32-1b-w2-w6queued_18536a_llama32_1b_frontier_anchor_up4_l6_l8_l9` |
| `4f89fc` | W3.2 anchor + Up4 layers 6, 8, and 12 | 3.186961 | **44.7477% (541/1209)** | 4.69% | +0.012833 | evaluation_complete | `scripts/configs/llama32_1b_frontier_w6_anchor_up4_l6_l8_l12.json` | `/root/qvq-results/llama32-1b-w2-w6queued_4f89fc_llama32_1b_frontier_anchor_up4_l6_l8_l12` |

## Current frontier anchor snapshot (2026-08-28 UTC)

Future allocation waves must retain these completed full-GSM8K comparison
points and may replace one only after checkpoint identity, effective BPW, and
held-out reports are verified.

| Budget band | Anchor arm | Allocation | Effective BPW | GSM8K Platinum |
| --- | --- | --- | ---: | ---: |
| ~2.02 | `b5283c` | Flat W2, reg 0.15 | 2.0232 | 23.8213% (288/1209) |
| ~3.02 | `1040a5` | Flat W3 | 3.0232 | 42.9280% (519/1209) |
| ~3.18 | `8980aa` | W3.2 anchor + Up4 L6,L8 | 3.178340 | **45.5749% (551/1209)** |
| ~3.52 | `9769b1` | Flat W3.5, seed 1 | 3.5232 | **46.65% (564/1209)** |
| ~3.56 | `4e7424` | Flat W3.5 + Up4 L12–15, seed 1 | 3.5577 | 46.15% (558/1209) |
| ~3.63 | `049d0c` | Flat W3.5 + V4 all + Up4 L9–15 | 3.6266 | 45.9884% (556/1209) |

The ~3.2 Wave-5 `L6+L8` arm remains the best anchor in that band; Wave-6's
best `L8+L9` arm reached 543/1209 and did not replace it. Mini-GSM is retained
as a screening diagnostic only; full GSM8K is the promotion metric.

## Wave-7 full single-layer Up sensitivity sweep launched (2026-08-28 UTC)

Sixteen matched-budget arms were launched from the corrected W3.2 anchor. Each
arm promotes one `mlp.up_proj` from W3.5 to W4, giving an estimated 3.169720
BPW. Two quantizers run concurrently per GPU; evaluation is exclusive per GPU
and follows each checkpoint immediately.

| layer | arm_id | gpu | allocation | estimated BPW | state |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | `e62172` | 0 | anchor + Up4 L0 | 3.169720 | evaluation complete |
| 1 | `4bab0e` | 1 | anchor + Up4 L1 | 3.169720 | evaluation complete |
| 2 | `6fba6c` | 2 | anchor + Up4 L2 | 3.169720 | evaluation complete |
| 3 | `3ff286` | 3 | anchor + Up4 L3 | 3.169720 | evaluation complete |
| 4 | `e1b702` | 4 | anchor + Up4 L4 | 3.169720 | evaluation complete |
| 5 | `788234` | 5 | anchor + Up4 L5 | 3.169720 | evaluation complete |
| 6 | `5f91ab` | 6 | anchor + Up4 L6 (repeat) | 3.169720 | evaluation complete |
| 7 | `5ea33c` | 7 | anchor + Up4 L7 | 3.169720 | evaluation complete |
| 8 | `b8fb5c` | 0 | anchor + Up4 L8 (repeat) | 3.169720 | evaluation complete |
| 9 | `d693ff` | 1 | anchor + Up4 L9 (repeat) | 3.169720 | GSM8K evaluation running |
| 10 | `d6fab2` | 2 | anchor + Up4 L10 | 3.169720 | evaluation complete |
| 11 | `bd7c59` | 3 | anchor + Up4 L11 | 3.169720 | evaluation complete |
| 12 | `bb10df` | 4 | anchor + Up4 L12 | 3.169720 | GSM8K evaluation running |
| 13 | `adff86` | 5 | anchor + Up4 L13 | 3.169720 | GSM8K evaluation running |
| 14 | `254d84` | 6 | anchor + Up4 L14 | 3.169720 | evaluation complete |
| 15 | `515065` | 7 | anchor + Up4 L15 | 3.169720 | evaluation complete |

The repeated layers are intentional cross-wave controls. No score is inferred
until the checkpoint and both held-out reports are complete. Queue manifest:
`docs/experiments/frontier_wave7_queue_20260828.json`.

## Wave-8 pair-interaction mapping completed (2026-08-29 UTC)

Sixteen matched-budget pair arms were queued from the corrected W3.2 anchor.
Each promotes two `mlp.up_proj` modules from W3.5 to W4, giving 3.178340
effective payload BPW. The positive `L6+L8` and negative `L8+L12` controls are
intentional repeats. Two quantization sessions may share a 96-GB GPU; the
per-GPU lock serializes quantization/evaluation on that device.

| # | arm_id | Up4 layers | GPU | effective BPW | state |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `fb8247` | L5 + L12 | 0 | 3.178340 | evaluation complete |
| 2 | `735365` | L5 + L15 | 1 | 3.178340 | evaluation complete |
| 3 | `e5488a` | L12 + L15 | 2 | 3.178340 | evaluation complete |
| 4 | `6f37dc` | L9 + L12 | 3 | 3.178340 | evaluation complete |
| 5 | `9cd131` | L9 + L15 | 4 | 3.178340 | evaluation complete |
| 6 | `fdbd68` | L6 + L8 | 5 | 3.178340 | evaluation complete |
| 7 | `91a008` | L8 + L12 | 6 | 3.178340 | evaluation complete |
| 8 | `4e58f5` | L5 + L6 | 7 | 3.178340 | evaluation complete |
| 9 | `c4cc84` | L6 + L12 | 0 | 3.178340 | evaluation complete |
| 10 | `cef938` | L6 + L15 | 1 | 3.178340 | evaluation complete |
| 11 | `fe2ba7` | L5 + L7 | 2 | 3.178340 | evaluation complete |
| 12 | `446290` | L7 + L12 | 3 | 3.178340 | evaluation complete |
| 13 | `719a90` | L7 + L15 | 4 | 3.178340 | evaluation complete |
| 14 | `cecb41` | L4 + L5 | 5 | 3.178340 | evaluation complete |
| 15 | `1e3a24` | L4 + L12 | 6 | 3.178340 | evaluation complete |
| 16 | `864bb1` | L4 + L8 | 7 | 3.178340 | evaluation complete |

Queue manifest: `docs/experiments/frontier_wave8_queue_20260829.json`.

### Wave-8 completed evaluation snapshot (2026-08-29 UTC)

All sixteen pair arms completed both full GSM8K Platinum (1,209 rows) and
Mini-GSM (64 rows).

| arm_id | Up4 layers | GSM8K Platinum | Mini exact | answer-logprob Δ | answer-margin Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| `fb8247` | L5 + L12 | 512/1209 (42.3490%) | 3.1250% | +0.409967 | +0.070590 |
| `6f37dc` | L9 + L12 | 524/1209 (43.3416%) | 1.5625% | +0.475437 | −0.038672 |
| `9cd131` | L9 + L15 | 522/1209 (43.1762%) | 3.1250% | +0.462224 | −0.032238 |
| `4e58f5` | L5 + L6 | 511/1209 (42.2663%) | 4.6875% | +0.338698 | −0.105156 |
| `735365` | L5 + L15 | 518/1209 (42.8453%) | 6.25% | +0.391713 | +0.076822 |
| `e5488a` | L12 + L15 | 525/1209 (43.4243%) | 4.6875% | +0.456282 | +0.012602 |
| `fdbd68` | L6 + L8 | 519/1209 (42.9280%) | 4.6875% | +0.362137 | −0.217387 |
| `91a008` | L8 + L12 | 542/1209 (44.8304%) | 1.5625% | +0.430909 | −0.022157 |
| `c4cc84` | L6 + L12 | 537/1209 (44.4169%) | 6.25% | +0.407893 | −0.164052 |
| `fe2ba7` | L5 + L7 | 518/1209 (42.8453%) | 4.6875% | +0.257663 | −0.178204 |
| `446290` | L7 + L12 | 536/1209 (44.3342%) | 4.6875% | +0.334293 | −0.228091 |
| `719a90` | L7 + L15 | 520/1209 (43.0108%) | 6.25% | +0.318594 | −0.226357 |
| `cecb41` | L4 + L5 | 494/1209 (40.8602%) | 3.125% | +0.478479 | +0.217460 |
| `1e3a24` | L4 + L12 | 520/1209 (43.0108%) | 3.125% | +0.545433 | +0.179761 |
| `864bb1` | L4 + L8 | 520/1209 (43.0108%) | 3.125% | +0.501772 | +0.141465 |
| `cef938` | L6 + L15 | 518/1209 (42.8453%) | 6.25% | +0.393213 | −0.167946 |

## Wave-9 local marginal sweep queued (2026-08-29 UTC)

Eight local marginal arms were launched from the corrected W3.2 anchor. Arms
1–6 use one additional precision increment and arms 7–8 use two; all remain
under the 3.2 effective-BPW ceiling. Queue manifest:
`docs/experiments/frontier_wave9_queue_20260829.json`.

| # | arm_id | allocation | GPU | effective BPW | state |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `5b847c` | Up4 L7 + L8 + L12 | 0 | 3.186961 | evaluation_complete |
| 2 | `611404` | Up4 L6 + L7 + L8 | 1 | 3.186961 | evaluation_complete |
| 3 | `6fe5ca` | Up4 L8 + L12; Down3.5 L12 | 2 | 3.186961 | evaluation_complete |
| 4 | `228564` | Up4 L6 + L8; Down3.5 L8 | 3 | 3.186961 | evaluation_complete |
| 5 | `370c3a` | Up4 L8 + L11 + L12 | 4 | 3.186961 | evaluation_complete |
| 6 | `0ea650` | Up4 L6 + L8 + L11 | 5 | 3.186961 | evaluation_complete |
| 7 | `186871` | Up4 L8 + L12; Down3.5 L8 + L12 | 6 | 3.195582 | evaluation_complete |
| 8 | `37eb97` | Up4 L6 + L8; Down3.5 L6 + L8 | 7 | 3.195582 | evaluation_complete |

### Wave-9 partial results (2026-08-29 UTC)

All eight Wave-9 arms completed GSM8K Platinum and Mini-Math. Full metric
payloads are stored in
`docs/experiments/frontier_wave9_queue_20260829.json`.

| arm_id | allocation | effective BPW | GSM8K | Mini exact | answer-logprob Δ | answer-margin Δ | state |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `5b847c` | Up4 L7 + L8 + L12 | 3.186961 | 529/1209 (43.7552%) | 6.25% | +0.290639 | −0.265506 | evaluation_complete |
| `0ea650` | Up4 L6 + L8 + L11 | 3.186961 | 525/1209 (43.4243%) | 6.25% | +0.387690 | −0.225125 | evaluation_complete |
| `611404` | Up4 L6 + L7 + L8 | 3.186961 | 528/1209 (43.6725%) | 7.8125% | +0.237316 | −0.422956 | evaluation_complete |
| `6fe5ca` | Up4 L8 + L12; Down3.5 L12 | 3.186961 | 529/1209 (43.7552%) | 4.6875% | +0.436002 | −0.064639 | evaluation_complete |
| `228564` | Up4 L6 + L8; Down3.5 L8 | 3.186961 | 524/1209 (43.3416%) | 9.375% | +0.204094 | −0.556634 | evaluation_complete |
| `370c3a` | Up4 L8 + L11 + L12 | 3.186961 | 535/1209 (44.2514%) | 6.25% | +0.458041 | −0.030357 | evaluation_complete |
| `186871` | Up4 L8 + L12; Down3.5 L8 + L12 | 3.195582 | 530/1209 (43.8379%) | 6.25% | +0.283112 | −0.399024 | evaluation_complete |
| `37eb97` | Up4 L6 + L8; Down3.5 L6 + L8 | 3.195582 | 528/1209 (43.6725%) | 6.25% | +0.273898 | −0.297230 | evaluation_complete |

## Wave-10 determinism canary replays (2026-08-30 UTC)

Before launching new quantization, four repeated GSM8K evaluations per canary
checkpoint are running with identical task settings. This isolates evaluator
variance from the historical 551/1209 versus 519/1209 checkpoint discrepancy.
Normalized metrics digests are written beside each immutable report. See
`docs/experiments/frontier_wave10_determinism_queue_20260830.json` and
`scripts/run_llama32_wave10_canary_replays.sh`.

| Arm IDs | Source checkpoint | GPUs | state |
| --- | --- | --- | --- |
| `w10-8980aa-r1` … `r4` | `8980aa` | 0–3 | evaluation_complete |
| `w10-fdbd68-r1` … `r4` | `fdbd68` | 4–7 | evaluation_complete |

### Wave-10 canary replay results (2026-08-30 UTC)

All four repeats for each canary checkpoint completed with identical
normalized GSM8K metric digests. The current evaluator therefore has no
within-checkpoint variance; its scores are 541/1209 for `8980aa` and 537/1209
for `fdbd68`, versus historical records of 551 and 519 respectively. This
remaining discrepancy is checkpoint/code provenance, not evaluator randomness.

| source arm | repeats | GSM8K each | digest SHA-256 | state |
| --- | ---: | ---: | --- | --- |
| `8980aa` | 4 | 541/1209 (44.7477%) | `5b154128…e8529c` | evaluation_complete |
| `fdbd68` | 4 | 537/1209 (44.4169%) | `ef896c0a…51d202` | evaluation_complete |

## Wave-7 evaluation results snapshot (2026-08-29 UTC)

All sixteen full single-layer Up4 sweep arms completed both held-out GSM8K
Platinum (1,209 rows) and Mini-GSM (64 rows). All arms use 3.169720 effective
payload BPW.

| layer | arm_id | GSM8K Platinum | Mini exact | answer-logprob Δ | answer-margin Δ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `e62172` | 532/1209 (44.0033%) | 3.1250% | +0.152838 | +0.745247 |
| 1 | `4bab0e` | 530/1209 (43.8379%) | 3.1250% | +0.149789 | +0.558371 |
| 2 | `6fba6c` | 531/1209 (43.9206%) | 4.6875% | +0.092876 | +0.639518 |
| 3 | `3ff286` | 535/1209 (44.2514%) | 4.6875% | +0.178998 | +0.534700 |
| 4 | `e1b702` | 536/1209 (44.3342%) | 4.6875% | +0.133087 | +0.830440 |
| 5 | `788234` | 541/1209 (44.7477%) | 4.6875% | +0.102059 | +0.461900 |
| 6 | `5f91ab` | 532/1209 (44.0033%) | 1.5625% | +0.133829 | +0.516292 |
| 7 | `5ea33c` | 525/1209 (43.4243%) | 3.1250% | +0.175927 | +0.609579 |
| 8 | `b8fb5c` | 539/1209 (44.5823%) | 4.6875% | +0.072855 | +0.506915 |
| 9 | `d693ff` | 537/1209 (44.4169%) | 3.1250% | +0.216500 | +0.677904 |
| 10 | `d6fab2` | 527/1209 (43.5897%) | 4.6875% | +0.077571 | +0.534136 |
| 11 | `bd7c59` | 535/1209 (44.2514%) | 4.6875% | +0.203073 | +0.858973 |
| 12 | `bb10df` | 540/1209 (44.6650%) | 3.1250% | +0.168476 | +0.730859 |
| 13 | `adff86` | 531/1209 (43.9206%) | 4.6875% | +0.256177 | +0.834301 |
| 14 | `254d84` | 535/1209 (44.2514%) | 3.1250% | +0.170530 | +0.627562 |
| 15 | `515065` | 539/1209 (44.5823%) | 4.6875% | +0.145288 | +0.634211 |

The machine-readable queue manifest contains the corresponding report paths and
live state for all sixteen arms:
`docs/experiments/frontier_wave7_queue_20260828.json`.
