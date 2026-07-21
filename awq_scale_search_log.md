# AWQ ScaleSearch Experiment Log

This file records the ScaleSearch accuracy work completed on 2026-07-20 so future changes can be compared against the same model, calibration set, hardware, and Evalution tasks.

## Decision

- `AWQConfig.scale_search_refine_steps=0` is the stable default.
- A Qwen3-8B Instruct sweep over steps 0, 2, 4, 6, and 8 independently confirmed step 0 on ARC Challenge and GSM8K Platinum CoT. Step 6 was close, but did not improve any scored metric.
- Ratio refinement remains opt-in because lower calibration reconstruction loss did not consistently improve downstream accuracy.
- If refinement is required experimentally, QKV-only refinement was safer than non-QKV-only or uniform refinement, but it still underperformed the coarse default on GSM8K Platinum CoT.
- `scale_search_refine_steps=1` is invalid. Valid values are `0` or integers greater than one.
- Per-group refinement is configured through `AWQConfig.dynamic` module patterns. The AWQ processor contains no model-specific projection-name classification.

## Controlled setup

```text
+----------------------+------------------------------------------------------------------+
| Item                 | Value                                                            |
+----------------------+------------------------------------------------------------------+
| Model                | /monster/data/model/Llama-3.2-1B-Instruct                       |
| Quantization         | AWQ, GEMM, W4, group size 128, symmetric, desc_act=False        |
| Source dtype         | float16                                                          |
| Calibration dataset  | /monster/data/model/dataset/nm-calibration, config=LLM          |
| Calibration rows     | 512                                                              |
| Packed length        | 2048, descending-length sort                                    |
| Calibration SHA-256  | c767262e3b85f894849259188838d110b972c54e2aa2620dce973219a37f14ef |
| Evaluator            | Evalution only                                                  |
| Primary task         | gsm8k_platinum_cot, full 1,209 examples                         |
| Evaluation backend   | MARLIN, eager attention, batch size 64                          |
| GPU                  | NVIDIA PG506-230/232, 96 GB, compute capability 8.0             |
| Python               | 3.14.5                                                           |
| PyTorch              | 2.12.0+cu130                                                     |
| Transformers         | 5.14.1                                                           |
| GPT-QModel base      | 7.0.0+ultra, revision 673289e2fa5879a984e313c8423006d9b0f9f362 |
+----------------------+------------------------------------------------------------------+
```

The benchmark runners pin `sys.path` to this repository and fail if `gptqmodel` resolves from another editable checkout. Saved summaries also record the imported module path and visible GPU.

## Uniform refinement-step sweep

All checkpoints used the identical 512-record calibration fingerprint. Step 0 was repeated on a second PG506-230 GPU to check score stability. Steps 2-9 were quantized concurrently, so their quantization wall times are not suitable for performance comparisons.

```text
+----------+----------+-------------+-----------+------------------+---------+
| Steps    | Correct  | acc,num     | Delta pp  | Mean recon MSE   | Eval s  |
+----------+----------+-------------+-----------+------------------+---------+
| 0        | 459/1209 | 0.379652605 |   +0.000  | 1.555181729e-04  | 176.85  |
| 0 repeat | 461/1209 | 0.381306865 |   +0.165  | 1.555181729e-04  | 174.69  |
| 2        | 420/1209 | 0.347394541 |   -3.226  | 1.542515438e-04  | 172.33  |
| 3        | 401/1209 | 0.331679074 |   -4.797  | 1.542744719e-04  | 159.76  |
| 4        | 399/1209 | 0.330024814 |   -4.963  | 1.531309354e-04  | 156.88  |
| 5        | 424/1209 | 0.350703060 |   -2.895  | 1.540371688e-04  | 160.80  |
| 6        | 423/1209 | 0.349875931 |   -2.978  | 1.532317542e-04  | 176.54  |
| 7        | 419/1209 | 0.346567411 |   -3.309  | 1.539206156e-04  | 159.43  |
| 8        | 399/1209 | 0.330024814 |   -4.963  | 1.540838521e-04  | 164.83  |
| 9        | 426/1209 | 0.352357320 |   -2.730  | 1.541234927e-04  | 169.05  |
| 10       | 412/1209 | 0.340777502 |   -3.888  | 1.543022073e-04  | 175.07  |
+----------+----------+-------------+-----------+------------------+---------+
```

Observations:

- The coarse search was the clear GSM8K winner and reproduced within two correct answers.
- Step 9 was the best refined result at 426/1209, still 33 answers below step 0.
- Step 4 achieved the lowest mean reconstruction MSE but tied for the worst downstream score.
- Across these ten settings, mean reconstruction MSE and GSM8K accuracy had Pearson correlation `+0.652570`: lower local MSE tended to accompany lower task accuracy in this sweep.
- Higher `refine_steps` is therefore not a monotonic quality control.

Exact data: `tests/benchmark/awq_scalesearch_gsm8k_sweep_a100.json`.

## Broad-task A/B: coarse versus uniform step 5

This paired run used matching PG506-230 GPUs and the same calibration fingerprint. It shows why refinement remains available but should not be the default.

```text
+------------------------+-----------------------------+-------------+-------------+-------------+
| Task                   | Metric                      | Step 0      | Step 5      | Delta       |
+------------------------+-----------------------------+-------------+-------------+-------------+
| ARC Challenge          | accuracy,loglikelihood      | 0.301194539 | 0.315699659 | +0.014505120 |
| ARC Challenge          | accuracy,loglikelihood_norm | 0.345563140 | 0.341296928 | -0.004266212 |
| MMLU-STEM              | acc,ll                      | 0.362829052 | 0.374246749 | +0.011417697 |
| GSM8K Platinum CoT     | acc,num                     | 0.379652605 | 0.350703060 | -0.028949545 |
| Reconstruction         | mean MSE                    | 1.55518e-04 | 1.54037e-04 | -0.9523%    |
| Quantization           | seconds                     | 237.37      | 252.82      | +6.51%      |
+------------------------+-----------------------------+-------------+-------------+-------------+
```

Uniform step 5 improved ARC raw accuracy and MMLU-STEM, but reduced normalized ARC accuracy and lost 35 GSM8K answers. It was not a Pareto improvement.

## Projection-selective step-4 experiment

Two paired checkpoints isolated refinement to QKV projections or to every non-QKV projection. The uniform step-4 checkpoint from the sweep is included for context; its quantization time is omitted because it ran under concurrent load.

The benchmark runner maps its QKV/non-QKV experiment flags to configurable Llama module-path patterns in `AWQConfig.dynamic`. The mapping lives outside the core processor and can be replaced for another model architecture.

```text
+------------------+-----------+-------------+----------+-------------+-----------+------------------+---------+
| Policy           | QKV steps | Other steps | Correct  | acc,num     | Delta pp  | Mean recon MSE   | Quant s |
+------------------+-----------+-------------+----------+-------------+-----------+------------------+---------+
| Coarse default   |     0     |      0      | 459/1209 | 0.379652605 |   +0.000  | 1.555181729e-04  | 237.37  |
| QKV-only refine  |     4     |      0      | 425/1209 | 0.351530190 |   -2.812  | 1.539339813e-04  | 267.76  |
| Non-QKV refine   |     0     |      4      | 416/1209 | 0.344086022 |   -3.557  | 1.545834563e-04  | 250.59  |
| Uniform refine   |     4     |      4      | 399/1209 | 0.330024814 |   -4.963  | 1.531309354e-04  | n/a     |
+------------------+-----------+-------------+----------+-------------+-----------+------------------+---------+
```

QKV-only refinement recovered 26 answers relative to uniform step 4 and beat non-QKV-only refinement by 9 answers. It nevertheless lost 34 answers versus coarse and added 12.8% quantization time. Non-QKV-only refinement lost 43 answers versus coarse and added 5.6% quantization time.

Per-group reconstruction loss explains which groups were optimized, but not the final task ordering:

```text
+----------------+------------------+------------------+------------------+
| Policy         | QKV mean MSE     | Gate/up mean MSE | Down mean MSE    |
+----------------+------------------+------------------+------------------+
| Coarse         | 1.177451500e-04  | 2.117521875e-04  | 1.563692125e-04  |
| Uniform step 4 | 1.168623625e-04  | 2.063588500e-04  | 1.554808250e-04  |
| QKV-only       | 1.155670000e-04  | 2.108814938e-04  | 1.551399000e-04  |
| Non-QKV-only   | 1.172239250e-04  | 2.100461938e-04  | 1.557365750e-04  |
+----------------+------------------+------------------+------------------+
```

Exact data: `tests/benchmark/awq_scalesearch_projection_policy_gsm8k_a100.json`.

## Qwen3-8B Instruct cross-model sweep

This sweep tested whether the finer local search begins to help at 8B scale or on a non-Llama architecture. The requested `/model/data/model` directory was absent, so the run used the matching local Qwen3-8B Instruct checkpoint at `/monster/data/model/Qwen3-8B`. Its config identifies `Qwen3ForCausalLM`, includes a chat template, and declares native `bfloat16` weights.

An initial float16 diagnostic made every ScaleSearch candidate non-finite at decoder layer 3 for all five settings. A short native-BF16 smoke crossed that layer and saved successfully, after which every full BF16 run completed. The float16 attempts are excluded from the score comparison.

```text
+----------------------+------------------------------------------------------------------+
| Item                 | Value                                                            |
+----------------------+------------------------------------------------------------------+
| Model                | /monster/data/model/Qwen3-8B                                    |
| Architecture         | Qwen3ForCausalLM, 36 layers, 32 attention heads, 8 KV heads     |
| Quantization         | AWQ, GEMM, W4, group size 128, symmetric, desc_act=False        |
| Source dtype         | bfloat16                                                         |
| Calibration dataset  | /monster/data/model/dataset/nm-calibration, config=LLM          |
| Calibration rows     | 512                                                              |
| Packed calibration   | 89 batches; 182,272 total and 181,796 non-padding tokens        |
| Packed length        | 2048, descending-length sort                                    |
| Calibration SHA-256  | c767262e3b85f894849259188838d110b972c54e2aa2620dce973219a37f14ef |
| Evaluator            | Evalution only                                                  |
| Tasks                | ARC Challenge (1,172); GSM8K Platinum CoT (1,209)               |
| Evaluation backend   | MARLIN, eager attention, chat template, batch size 64            |
| Generation           | greedy, max_new_tokens=256, seed=42                             |
| GPU                  | NVIDIA PG506-230/232, 96 GB, compute capability 8.0             |
| Repository revision  | cdea42af2f8b3f815c3ac29d0c510051d9619194                         |
+----------------------+------------------------------------------------------------------+
```

ARC and GSM were run as separate Evalution processes so all idle GPUs could score concurrently. All five quantization jobs also ran concurrently; their wall times include shared-host contention and should only be treated as approximate overhead indicators.

```text
+-------+---------------+-------------+----------------+--------------+-----------+-------------+------------------+----------+-------+--------+
| Steps | ARC raw count | ARC raw acc | ARC norm count | ARC norm acc | GSM count | GSM acc,num | Mean recon MSE   | Quant s  | ARC s | GSM s  |
+-------+---------------+-------------+----------------+--------------+-----------+-------------+------------------+----------+-------+--------+
| 0     | 447/1172      | 0.381399317 | 479/1172       | 0.408703072  | 239/1209  | 0.197684036 | 3.079221050e-02  | 1131.84  | 41.98 | 736.18 |
| 2     | 434/1172      | 0.370307167 | 464/1172       | 0.395904437  | 213/1209  | 0.176178660 | 3.116722633e-02  | 1170.77  | 36.57 | 726.83 |
| 4     | 439/1172      | 0.374573379 | 464/1172       | 0.395904437  | 221/1209  | 0.182795699 | 3.038909344e-02  | 1162.69  | 36.50 | 724.24 |
| 6     | 444/1172      | 0.378839590 | 473/1172       | 0.403583618  | 237/1209  | 0.196029777 | 3.063534393e-02  | 1233.55  | 37.47 | 723.48 |
| 8     | 441/1172      | 0.376279863 | 467/1172       | 0.398464164  | 218/1209  | 0.180314309 | 3.009028004e-02  | 1234.77  | 36.44 | 725.21 |
+-------+---------------+-------------+----------------+--------------+-----------+-------------+------------------+----------+-------+--------+
```

Step 0 was the only Pareto winner across the three downstream accuracy metrics. Step 6 came closest, but still lost 3 raw ARC answers, 6 normalized ARC answers, and 2 GSM answers; it also took about 9.0% longer to quantize under the concurrent setup. Step 8 reduced mean reconstruction MSE by 2.28% yet lost 6 raw ARC, 12 normalized ARC, and 21 GSM answers, again demonstrating that finer local reconstruction is not a reliable downstream objective.

The deterministic checks reproduced exactly:

```text
+----------------------+------------------+------------------+
| Repeat               | Primary          | Repeat           |
+----------------------+------------------+------------------+
| Step 0 ARC raw       | 0.381399317406   | 0.381399317406   |
| Step 0 ARC norm      | 0.408703071672   | 0.408703071672   |
| Step 0 GSM           | 0.197684036394   | 0.197684036394   |
| Step 6 ARC raw       | 0.378839590444   | 0.378839590444   |
| Step 6 ARC norm      | 0.403583617747   | 0.403583617747   |
+----------------------+------------------+------------------+
```

Exact data: `tests/benchmark/awq_scalesearch_qwen3_8b_arc_gsm_sweep_a100.json`.

## Interpretation

AWQ ScaleSearch minimizes local reconstructed module-output MSE on the calibration data. That proxy does not directly optimize final logits, language-model likelihood, or reasoning accuracy. Group-wise INT4 rounding makes the objective discontinuous, and small per-layer improvements can compound into worse downstream decisions. A finer local optimum can also fit the calibration activations more closely without generalizing to evaluation prompts.

The canonical 20-point coarse grid behaves like a useful regularizer while remaining faster. Refinement may still help selected task families, but it needs broader validation or a held-out final-logit/NLL gate before it can safely become automatic.

## Implementation and reproduction

Relevant files:

- `gptqmodel/looper/awq_processor.py`: transactional candidate evaluation, two-stage refinement, replay-mask alignment, and dynamic per-group policy resolution.
- `gptqmodel/quantization/config.py`: stable global default and validation of serializable dynamic per-module overrides.
- `scripts/benchmark_awq_scalesearch_variant.py`: guarded 512-sample quantization runner.
- `scripts/evaluate_awq_scalesearch_variant.py`: guarded repository Evalution runner.
- `tests/models/awq/test_llama3_2.py`: full A100 coarse-default regression scores.

Example policy commands:

```bash
# QKV-only step 4
python scripts/benchmark_awq_scalesearch_variant.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output /tmp/awq-qkv4 --variant qkv4 --refine-steps 0 \
  --qkv-refine-steps 4 --non-qkv-refine-steps 0

# Non-QKV-only step 4
python scripts/benchmark_awq_scalesearch_variant.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output /tmp/awq-non-qkv4 --variant non_qkv4 --refine-steps 0 \
  --qkv-refine-steps 0 --non-qkv-refine-steps 4
```

Score either checkpoint with full Evalution:

```bash
python scripts/evaluate_awq_scalesearch_variant.py \
  --model /tmp/awq-qkv4 --variant qkv4 \
  --output /tmp/awq-qkv4/gsm8k_evalution_summary.json \
  --batch-size 64 --tasks gsm8k_platinum_cot
```

Large generated checkpoints remain under `/tmp` and are intentionally not committed.

## Validation completed

- Focused ScaleSearch/configuration suite: `6 passed`, `8 subtests passed`.
- Qwen3-8B native-BF16 sweep: all five 512-row quantization processes exited 0 and saved loadable two-shard checkpoints.
- Qwen3-8B Evalution sweep: all five ARC and five GSM8K Platinum processes exited 0; step-0 ARC/GSM and step-6 ARC repeats matched exactly.
- Qwen3-8B benchmark JSON was checked field-for-field against every raw quantization and Evalution summary.
- Post-sweep dynamic-mapping/configuration rerun: `4 passed`, `8 subtests passed` through the Transformers 5.14 hub-compatibility bootstrap.
- Replay-mask tests: `2 passed`.
- Ruff on every touched Python file: passed.
- `python -m py_compile` on implementation and runners: passed.
- `git diff --check`: passed.
- Both committed benchmark JSON files parse with `jq empty`.
- Quantization and scoring were run from this repository on CUDA hardware.
- Evaluation used Evalution only; lm-eval was not used for scoring.
