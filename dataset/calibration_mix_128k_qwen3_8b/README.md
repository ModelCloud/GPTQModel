# Calibration mix `calibration_mix_128k_qwen3_8b`

- **ID:** `calibration_mix_128k_qwen3_8b`
- **Name:** 128K-token calibration mix for `Qwen/Qwen3-8B` (best-score floor)
- **Model used for ranking:** `Qwen/Qwen3-8B`
- **Selection mode:** `gain_per_token`
- **Target token floor:** 131,072
- **Final token count:** 127,402
- **Final coverage score:** 103,576.82
- **Cumulative gain:** 1,076,071.19

## What this dataset is

This is a pre-quantization calibration mix selected by the coverage scanner in
`optimize/calibration_coverage.py`. It is the greedy-selected union of
~32K-token shards from two public calibration corpora:

- `lemon07r/bartowski-imatrix-v5-semantic` (Unsloth/Bartowski iMatrix v5)
- `neuralmagic/calibration` with config `LLM`

The file `calibration.parquet` contains the selected examples as a
`messages` column (chat-template compatible). `dataset_info.json` holds the
metadata, `report.json`/`report.md` hold the full scanner report, and
`generate.py` reproduces the whole pipeline.

## How the mix was built

1. **Shard preparation**
   - `lemon07r/bartowski-imatrix-v5-semantic` rows 0-249 were tokenized with the
     Qwen3-8B tokenizer and packed into 2 ~32K-token shards (`imatrix_00`,
     `imatrix_01`).
   - `neuralmagic/calibration` `LLM` rows 0-499 were tokenized with the same
     tokenizer and packed into 4 ~32K-token shards (`nm_llm_00` ... `nm_llm_03`).
   - A held-out reference was built from the rows immediately following each
     candidate block, totaling 15,504 tokens.

2. **Coverage scan**
   - `optimize/calibration_coverage.py` ran on CPU with `--torch-dtype float16`
     and `--target-tokens 131072 --target-tokens-mode gain_per_token`.
   - The scanner computes per-module activation profiles (`diag`, `chan_max`,
     percentile sketch) and scores each shard by importance-weighted tail
     under-coverage versus the reference.

3. **Greedy selection**
   - The greedy loop added the shard with the largest conditional gain per token
     until the 128K-token floor was reached, then kept adding while the next
     marginal gain remained positive (target-as-floor behavior).
   - On this model, the remaining two shards (`nm_llm_00` and `nm_llm_01`)
     became redundant after the fourth shard, so the positive-gain mix stops
     slightly below the 128K floor at 127,402 tokens.

## Composition of the selected mix

| step | shard | tokens | cumulative | conditional gain | score after |
|------|-------|--------|------------|------------------|-------------|
| 1 | `imatrix_01` | 30,825 | 30,825 | 1,053,951.26 | 125,696.75 |
| 2 | `nm_llm_03` | 32,082 | 62,907 | 14,339.04 | 111,357.71 |
| 3 | `imatrix_00` | 32,474 | 95,381 | 4,040.06 | 107,317.65 |
| 4 | `nm_llm_02` | 32,021 | 127,402 | 3,740.83 | 103,576.82 |

- The selected mix is 153 examples and 127,402 tokens.
- `nm_llm_00` and `nm_llm_01` were redundant versus the selected mix
  (negative conditional gain), so the scanner stopped before the 131,072 token
  floor.

## Comparison with the Qwen3-0.6B baseline

The Qwen3-0.6B mix (id `calibration_mix_128k_qwen3_0.6b`) used the same two
source corpora and a 131,072 token floor. It ended at a larger 192,181-token
positive-gain mix with score 19,430.45.

Similarities:
- Both models pick `imatrix_01` first (the most complementary single shard).
- Both models pick `nm_llm_03` second.

Differences:
- For Qwen3-8B, the next most complementary shard is `imatrix_00`; for
  Qwen3-0.6B it is `nm_llm_00`.
- The 8B model reaches effective coverage saturation at 127K tokens, with the
  remaining neuralmagic LLM shards becoming redundant. The 0.6B model continues
  to benefit from additional shards through ~192K tokens.
- This likely reflects the larger capacity / different layer distribution of the
  8B model: the 108 target groups (216 modules) are covered by a smaller, more
  complementary subset of the source corpora.

## Reproducing it

Run `generate.py` from this folder with the free-threaded GPT-QModel venv:

```bash
PYTHON_GIL=0 /home/ubuntu/.venv-gptq-gil0/bin/python generate.py
```

It will re-download the source datasets, build the shards, run the coverage
scanner, and regenerate `calibration.parquet`, `dataset_info.json`, and the
reports.

## Usage as a calibration dataset

The parquet can be passed directly to the scanner or to a quantizer that accepts
`messages` columns. For the scanner:

```bash
PYTHON_GIL=0 /home/ubuntu/.venv-gptq-gil0/bin/python optimize/calibration_coverage.py \
  --model Qwen/Qwen3-8B \
  --dataset dataset/calibration_mix_128k_qwen3_8b/calibration.parquet:calibration_mix_128k \
  --reference ... \
  --output-dir /tmp/out
```

Replace `...` with your held-out reference dataset.
