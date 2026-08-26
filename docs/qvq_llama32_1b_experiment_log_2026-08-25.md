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

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e` | `divergence300` | complete; metric=0.265; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-attention-only-main-ecf7081e-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365` | `divergence300` | complete; metric=0.120625; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-main-54ccd365-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365` | `divergence300` | complete; metric=0.18677083333333333; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg005-align-main-54ccd365-div300-dev-v1.json` |

| monitor | `llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00` | `divergence300` | complete; metric=0.17302083333333335; report `/root/qvq-results/llama32-1b-w2-reg015-smooth-swiglu-tip-d68c9d00-div300-dev-v1.json` |

| monitor | `llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e` | `divergence300` | complete; metric=0.17864583333333334; report `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-reg010-align-main-ecf7081e-div300-dev-v1.json` |
