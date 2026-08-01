# Calibration mix `calibration_mix_128k_qwen3_0.6b`

- **ID:** `calibration_mix_128k_qwen3_0.6b`
- **Name:** 128K-token calibration mix for `Qwen/Qwen3-0.6B` (best-score floor)
- **Model used for ranking:** `Qwen/Qwen3-0.6B`
- **Selection mode:** `gain_per_token`
- **Target token floor:** 131,072
- **Final token count:** 192,181
- **Final coverage score:** 19,430.45
- **Cumulative gain:** 209,784.84

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
     Qwen3-0.6B tokenizer and packed into 2 ~32K-token shards (`imatrix_00`,
     `imatrix_01`).
   - `neuralmagic/calibration` `LLM` rows 0-499 were tokenized with the same
     tokenizer and packed into 4 ~32K-token shards (`nm_llm_00` ... `nm_llm_03`).
   - A held-out reference was built from the rows immediately following each
     candidate block, totaling ~15,665 tokens.

2. **Coverage scan**
   - `optimize/calibration_coverage.py` ran with
     `--target-tokens 131072 --target-tokens-mode gain_per_token`.
   - The scanner computes per-module activation profiles (`diag`, `chan_max`,
     percentile sketch) and scores each shard by importance-weighted tail
     under-coverage versus the reference.

3. **Greedy selection**
   - The greedy loop added the shard with the largest conditional gain per token
     until the 128K-token floor was reached, then kept adding while the next
     marginal gain remained positive (target-as-floor behavior).

## Composition of the selected mix

| step | shard | tokens | cumulative | conditional gain | score after |
|------|-------|--------|------------|------------------|-------------|
| 1 | `imatrix_01` | 30,825 | 30,825 | 202,184.33 | 27,030.96 |
| 2 | `nm_llm_03` | 32,442 | 63,267 | 5,968.95 | 21,062.01 |
| 3 | `nm_llm_00` | 32,633 | 95,900 | 858.79 | 20,203.22 |
| 4 | `nm_llm_01` | 31,926 | 127,826 | 288.63 | 19,914.58 |
| 5 | `imatrix_00` | 32,474 | 160,300 | 158.38 | 19,756.21 |
| 6 | `nm_llm_02` | 31,881 | 192,181 | 325.76 | 19,430.45 |

- The 128K floor is first reached at step 4 (127,826 tokens, score 19,914.58).
- Step 5 is the first prefix that is *at or above* the floor (160,300 tokens,
  score 19,756.21).
- The full positive-gain mix is step 6 (192,181 tokens, score 19,430.45) and is
  the lowest-score dataset in this run.

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
  --model Qwen/Qwen3-0.6B \
  --dataset dataset/calibration_mix_128k_qwen3_0.6b/calibration.parquet:calibration_mix_128k \
  --reference ... \
  --output-dir /tmp/out
```

Replace `...` with your held-out reference dataset.
