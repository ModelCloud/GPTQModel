# Llama 3.2 1B flat-W2 leader

This is the durable best-of-sweep record for Llama 3.2 1B Instruct at flat W2. “Leader” means the highest corrected
Divergence-300 aligned-token agreement, not the lowest teacher-forced KL. The append-only campaign ledger is
[`docs/qvq_llama32_1b_experiment_log_2026-08-25.md`](docs/qvq_llama32_1b_experiment_log_2026-08-25.md).

## Selected checkpoint

- checkpoint: `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365`;
- quantization commit: `54ccd365fdd9e0c1546521afc8c68f28269c7f86`;
- dense model: `/monster/data/model/Llama-3.2-1B-Instruct`;
- payload: all 112 linear modules at W2, `qvq_v2b2_p32`, two banks, estimated 2.03125 BPW;
- quantization wall time: 1,347.388 seconds plus 1.426 seconds load and 0.990 seconds save.

Complete quantization configuration:

```json
{
  "bits": 2,
  "format": "qvq_v2b2_p32",
  "bank_count": 2,
  "rounding": "yaqa",
  "device": "cuda:0",
  "offload_to_disk": false,
  "yaqa": {
    "seed": 0,
    "regularization": 0.05,
    "regularization_by_rate": [[2.0, 0.1]],
    "minimum_sequences": 182,
    "batch_size": 1,
    "sequence_sort": "desc",
    "activation_checkpointing": true,
    "v2b2_family_mode": "reselect",
    "sample_strategy": "full"
  }
}
```

Resolved defaults in the serialized checkpoint are symmetric quantization, `pgc16-v1`, RHT incoherence, vector
size 2, 16x16 tiles, trellis window 16, full-row grouping, family reselection, no spectral refinement, no chat-token
reweighting, no output alignment, no module replay, and no precision exception.

## Calibration and data contract

| Role | Exact data |
| --- | --- |
| Lifecycle forward calibration | `/monster/data/model/dataset/nm-calibration/llm.parquet`, rows 0--127 |
| YAQA Sketch-B | `/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet`, all 182 rows |
| YAQA activity | 182 independent sequences, 302,193 valid output samples, native Llama chat-template rendering |
| Locked ordinary evaluation | NM rows 512--811, 300 rows / 105,618 valid tokens |
| D300 development evaluation | `/root/qvq-data/divergence300-v1/divergence300-development.jsonl`, SHA-256 `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2` |

Calibration, YAQA, candidate-selection, locked ordinary, and D300 content are disjoint under the recorded identity
checks. The D300 locked manifest remains untouched.

## Best-of-sweep metrics

| Metric | Result |
| --- | ---: |
| Corrected D300 aligned-token Top-1 through 32 | **17.8854% (1,717 / 9,600)** |
| Exact 32-token D300 trajectories | 1.0000% (3 / 300) |
| Exact-trajectory 95% Wilson interval | 0.3407%--2.8983% |
| Mean first divergence token | 4.8533 |
| Locked final KL | 0.272819 |
| Locked token Top-1 | 81.2380% |
| Locked Top-5 overlap | 72.6887% |
| Locked Top-10 overlap | 71.6885% |
| Shared-prefix Top-1 agreement@32, with 50% context warmup | 82.4350% |
| Shared-prefix exact agreement | 3.3557% |
| Mean first shared-prefix mismatch | 7.2383 |
| Legacy shared-prefix first-32 Top-1 | 70.5999% |

| Horizon | D300 cumulative aligned Top-1 | Exact-prefix survival |
| ---: | ---: | ---: |
| 1 | 67.3333% | 67.3333% |
| 2 | 61.3333% | 55.0000% |
| 4 | 54.8333% | 42.6667% |
| 8 | 44.2083% | 14.6667% |
| 16 | 30.6875% | 4.6667% |
| 32 | **17.8854%** | **1.0000%** |

| D300 source | Aligned Top-1 | Matches | Exact @32 |
| --- | ---: | ---: | ---: |
| SWE-bench Verified | 22.0000% | 704 / 3,200 | 0 / 100 |
| LongBench v2 | 19.3750% | 310 / 1,600 | 1 / 50 |
| Terminal-Bench 2.1 | 17.5781% | 225 / 1,280 | 1 / 40 |
| MathArena 2025/26 | 15.1563% | 291 / 1,920 | 0 / 60 |
| Non-English Multi-IF | 11.6875% | 187 / 1,600 | 1 / 50 |

## Sweep ranking under corrected D300

| Arm | Aligned Top-1 | Matches | Exact @32 | Decision |
| --- | ---: | ---: | ---: | --- |
| Uniform `.05` | 12.6042% | 1,210 | 2 | rejected |
| Hybrid `.05/.10` | 13.9896% | 1,343 | 1 | rejected |
| Hybrid + alignment | 15.1667% | 1,456 | 2 | rejected |
| Hybrid + two-epoch alignment | 17.0938% | 1,641 | 3 | teacher-forced leader, not D300 leader |
| **Uniform `.10`** | **17.8854%** | **1,717** | 3 | **selected** |
| Uniform `.10` + alignment | 17.2188% | 1,653 | 2 | rejected |
| Uniform `.125` | 13.4375% | 1,290 | 2 | rejected |
| Uniform `.20` | 16.3333% | 1,568 | 3 | rejected |
| Uniform `.10` + 97/3 chat weighting | 15.2604% | 1,465 | 4 | rejected |
| Post-quant NM SU/SV alignment | 15.0938% | 1,449 | 1 | rejected |
| Post-quant YAQA SU/SV alignment | 16.4375% | 1,578 | 2 | rejected |

The 25% development target is 2,400 matching positions. This checkpoint is short by 683 positions. The hybrid
two-epoch checkpoint remains the flat-W2 teacher-forced leader at KL 0.239397, Top-1 82.6090%, Top-5 74.3085%,
Top-10 73.6837%, and warm shared-prefix Top-1 83.0222%.

## Higher-rate Q07 sweep

The Q07 calibration and YAQA contract was rerun without changing its data, seed, bank count, family reselection, or
regularization. Only the trellis rate changed from W2 to W2.5 or W3. Both higher-rate checkpoints use V2B2-P32,
two banks, seed 0, batch 1, all 182 optimized YAQA rows / 302,193 valid tokens, ordinary NM rows 0--127, exact-rate
YAQA regularization `.10`, `v2b2_family_mode="reselect"`, and no alignment, chat-token weighting, spectral
refinement, or scale optimization. Their complete checked-in configs are
`scripts/configs/llama32_1b_v2b2_p32_yaqa_w25_reg010.json` and
`scripts/configs/llama32_1b_v2b2_p32_yaqa_w30_reg010.json`.

| Metric | Flat W2 Q07 | W2.5 | W3.0 |
| --- | ---: | ---: | ---: |
| Final KL | 0.272819 | 0.125240 | **0.060929** |
| Token Top-1 | 81.2380% | 87.1177% | **91.1634%** |
| Top-5 overlap | 72.6887% | 79.9364% | **85.3099%** |
| Top-10 overlap | 71.6885% | 79.4622% | **85.0076%** |
| Shared-prefix Top-1 agreement@32, with 50% context warmup | 82.4350% | 87.8146% | **91.4115%** |
| Canonical D300 aligned-token Top-1 through 32 | 17.8854% | 25.1771% | **33.6667%** |
| Canonical D300 aligned matches | 1,717 / 9,600 | 2,417 / 9,600 | **3,232 / 9,600** |
| Canonical exact trajectories | 3 / 300 | 5 / 300 | **16 / 300** |
| Canonical mean first divergence token | 4.8533 | 7.5000 | **9.9133** |

W3.0 wins every measured fidelity metric. W2.5 is consistently intermediate and crosses the 25% D300 development
target by 17 positions. W3 exceeds it by 832 positions. No GSM8K or MMLU evaluation was run for either higher-rate
checkpoint.

Artifacts and immutable evidence:

| Arm | Checkpoint | Locked ordinary report | Canonical D300 report |
| --- | --- | --- | --- |
| W2.5 | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5` | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5-locked-r512-n300-full-v1.json`, SHA-256 `b40dd718cf6dd322a1e02f3620703cc950bfade871809e11a5427eda5450d1bf` | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w25-reg010-pruneauto-main-329cc0a5-div300-dev-v1.json`, SHA-256 `7b93c80590e12fcfcf249855a6d971ac390c08f34a1fc3ad1127caa02080b6c4` |
| W3.0 | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5` | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5-locked-r512-n300-full-v1.json`, SHA-256 `535b41a38f6b2707690bf08e988e7ac05799e9bf382cb706e5323b55c5e9dae6` | `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-w30-reg010-pruneauto-main-329cc0a5-div300-dev-v1.json`, SHA-256 `08a4bc6ccd8b4a3b4ead7b21fb8de5a47c1e1d611cd7c3ee2bd588ccb3fa1a09` |

The separate independent rollouts over NM rows 512--811 scored 25.6146% (W2.5) and 33.8125% (W3). They are valid
diagnostics but are **not** canonical D300 and are not used in the ranking above. Canonical D300 uses the pinned
mixed-source manifest with SHA-256 `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2`.

## Teacher-forced hybrid-aligned leader

Checkpoint: `/root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da`.

This is the comparison arm for determining whether the teacher-forced metric winner or corrected-D300 winner better
predicts downstream accuracy. It retains flat W2 V2B2-P32 and the same optimized 182-row YAQA data. Regularization is
`.10` on every projection in layers 0, 6, 10, and 12 and `.05` elsewhere, followed by fixed-trellis output alignment.

```json
{
  "bits": 2,
  "format": "qvq_v2b2_p32",
  "bank_count": 2,
  "rounding": "yaqa",
  "device": "cuda:0",
  "offload_to_disk": false,
  "dynamic": {
    "+:model\\.layers\\.(0|6|10|12)\\..*": {"yaqa_regularization": 0.1}
  },
  "yaqa": {
    "seed": 0,
    "regularization": 0.05,
    "regularization_by_rate": [[2.0, 0.05]],
    "minimum_sequences": 182,
    "batch_size": 1,
    "sequence_sort": "desc",
    "activation_checkpointing": true,
    "v2b2_family_mode": "reselect",
    "sample_strategy": "full"
  },
  "output_alignment": {
    "learning_rate": 0.00001,
    "epochs": 2,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "maximum_train_batches": 64,
    "maximum_validation_batches": 24,
    "validation_fraction": 0.2,
    "minimum_relative_improvement": 0.0,
    "pristine_hessian": true
  }
}
```

| Metric | Hybrid-aligned leader | Uniform `.10` D300 leader |
| --- | ---: | ---: |
| Corrected D300 aligned Top-1 | 17.0938% | **17.8854%** |
| D300 aligned matches | 1,641 / 9,600 | **1,717 / 9,600** |
| Exact D300 trajectories | 3 / 300 | 3 / 300 |
| Locked final KL | **0.239397** | 0.272819 |
| Locked token Top-1 | **82.6090%** | 81.2380% |
| Locked Top-5 overlap | **74.3085%** | 72.6887% |
| Locked Top-10 overlap | **73.6837%** | 71.6885% |
| Warm shared-prefix Top-1 | **83.0222%** | 82.4350% |

## Full downstream task evaluation

Status: **GSM8K pair complete; full MMLU cancelled as too slow**.

| Task | Uniform `.10` D300 leader | Hybrid-aligned leader | Winner |
| --- | ---: | ---: | --- |
| GSM8K Platinum CoT, full 1,209 rows | **24.1522%** | 20.6782% | **Uniform `.10` by 3.4740 points** |

Atomic reports:

- `/root/qvq-results/llama32-1b-v2b2p32-reg010-leader-full-tasks-v1.json`
- `/root/qvq-results/llama32-1b-v2b2p32-layerdamp-align2e64-full-tasks-v1.json`

The requested tasks are full-dataset evaluations with no row cap:

- `gsm8k_platinum_cot`, native instruct chat template enabled;
- `mmlu_stem`, chat template disabled;
- `mmlu_humanities`, all 13 Evalution MMLU humanities subjects, chat template disabled.

```bash
python scripts/qvq_evaluate.py tasks \
  --checkpoint /root/qvq-results/llama32-1b-v2b2p32-yaqa322k-effective-reg010-main-54ccd365 \
  --output /root/qvq-results/llama32-1b-v2b2p32-reg010-leader-full-tasks-v1.json \
  --batch-size 16 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
  --task gsm8k_platinum_cot --task mmlu_stem --task mmlu_humanities
```

Each completed suite is now atomically appended to this report. Later paired suites use `--resume`, which validates
the checkpoint and full runtime contract before adding a missing task.

The same full evaluation is queued for the hybrid-aligned leader:

```bash
python scripts/qvq_evaluate.py tasks \
  --checkpoint /root/qvq-results/llama32-1b-v2b2p32-yaqa322k-layerdamp-align2e64-main-2f34e1da \
  --output /root/qvq-results/llama32-1b-v2b2p32-layerdamp-align2e64-full-tasks-v1.json \
  --batch-size 16 --device cuda:0 --attn-implementation 'paged|flash_attention_2' \
  --task gsm8k_platinum_cot --task mmlu_stem --task mmlu_humanities
```

Execution was interleaved by benchmark to reduce time/runtime drift. The GSM8K pair completed. Q07 STEM was then
stopped at 217/3,153 question rows on user direction because the full suite was too slow; Q03 STEM and both humanities
runs were cancelled before starting. Atomic per-task publication means no partial MMLU metric appears in either
report and the completed GSM8K evidence remains valid.

Pre-run validation:

| Command | Result |
| --- | --- |
| `pytest -q tests/test_qvq_unified_harness.py tests/test_validate_qvq_lifecycle.py` | 55 passed, 14 pre-existing dependency warnings |
| broad `ruff check` on the four touched Python files | failed on 14 existing style/executable-bit findings; no undefined-name or task-mapping failure |
| `ruff check --select F401,F811,F821,F822,F823` on the four touched Python files | passed |
| `pytest -q tests/test_qvq_unified_harness.py -k 'humanities or incremental'` | 2 passed; full humanities and forced incremental-progress contracts covered |
| `pytest -q tests/test_qvq_unified_harness.py -k 'humanities or incremental or paged_continuous'` | 3 passed; paged/continuous defaults also covered |
| `pytest -q tests/test_qvq_unified_harness.py -k 'humanities or incremental or paged_continuous or completed_rows'` | 4 passed on Evalution 0.0.14; MMLU progress maps four completed choice likelihoods to one completed question row |
| `ruff check --select E9,F63,F7,F82 scripts/qvq_evaluate.py tests/test_qvq_unified_harness.py` | passed after the Evalution 0.0.14 and MMLU row-progress update |

The first Q07 attempt used continuous refill but reported `paged_attention=False`; it was manually interrupted during
GSM8K at the user's request so live scoring and paged attention could be enabled. No partial score was published or
used. The restarted commands require `paged|flash_attention_2`; paged mode is also the Evalution/GPTQModel switch for
native continuous batching. A run is valid only if startup reports `backend=continuous_batching`,
`paged_attention=True`, and `generation submission mode=continuous_refill`.

The second Q07 attempt completed GSM8K Platinum at `acc,num=0.2415` over all 1,209 rows with zero invalid
generations, then reached 630/12,612 MMLU STEM choice likelihoods. It was intentionally interrupted before publishing
a combined report to upgrade the runtime from Evalution 0.0.12 to the current PyPI release, Evalution 0.0.14, and to
make MMLU display completed question rows (`completed / 3,153` for STEM) instead of internal choice requests
(`completed / 12,612`). The underlying four-choice likelihood computation and continuous-refill scheduling are
unchanged.

The third Q07 attempt was superseded at 643/1,209 GSM8K rows (`numeric=0.2348`, zero invalid) before publication when
the evaluation order changed from model-serial to benchmark-paired. The report-producing paired queue uses one task
per invocation and atomic `--resume`; a failure can no longer discard already completed suites.

The paired report-producing GSM8K runs completed under Evalution 0.0.14, paged attention, and continuous refill:
Q07 scored `0.2415219189` in 592.787 seconds and Q03 scored `0.2067824648` in 598.368 seconds. This first real-world
task favors the D300 metric winner, not the teacher-forced KL/Top-N winner.
