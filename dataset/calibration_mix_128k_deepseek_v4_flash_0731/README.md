# DeepSeek-V4-Flash-0731 128K-token calibration mix

Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
Defused BF16 checkpoint: `/monster/data/model/DeepSeek-V4-Flash-0731-BF16-Defused`
Scan target: 131,072 tokens (`gain_per_token` mode)

## Selected mix

`calibration.parquet` contains the greedy-selected calibration rows (185 examples, `messages` column with one user message per example).

Selected source shards (in order):

1. `wiki_zh_01.txt`
2. `tulu_01.txt`
3. `nm_llm_03.txt`
4. `wiki_ar_00.txt`
5. `wiki_ja_00.txt`
6. `nm_llm_01.txt`
7. `wiki_ko_00.txt`
8. `tulu_00.txt`

- **Total tokens:** 181,705
- **Final score:** 105,001.03
- **Cumulative gain:** 3,331,542.96
- **Examples:** 185

## Source pools

The candidate shards came from:

- `lemon07r/bartowski-imatrix-v5-semantic` (`imatrix_*`)
- `neuralmagic/calibration` / `LLM` (`nm_llm_*`)
- `ise-uiuc/Magicoder-OSS-Instruct-75K` (`code_*`)
- `nvidia/OpenMathInstruct-2` (`math_*`)
- `HuggingFaceFW/fineweb-edu` / `sample-10BT` (`fineweb_edu_*`)
- `emozilla/pg19` (`pg19_*`)
- `allenai/tulu-3-sft-mixture` (`tulu_*`)
- `wikimedia/wikipedia` (`wiki_*`)

## Coverage scan details

- GPUs: 6, 7
- Layers per GPU: 6
- `torch_dtype`: `bfloat16`
- `concat_size`: 512
- `sketch_samples`: 64
- MoE expert coverage was enabled, but the 16-token expert floor was not reached, so complementarity was computed without per-expert routing coverage.

See `report.md` / `report.json` for per-dataset standalone scores and greedy ranking, and `dataset_info.json` for structured metadata.
