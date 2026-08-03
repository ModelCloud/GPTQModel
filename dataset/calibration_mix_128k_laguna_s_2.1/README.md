# Calibration mix `calibration_mix_128k_laguna_s_2.1`

- **ID:** `calibration_mix_128k_laguna_s_2.1`
- **Name:** 128K-token-floor calibration mix for `poolside/Laguna-S-2.1` (best-score mix, MoE-aware)
- **Model used for ranking:** local `/monster/data/model/Laguna-S-2.1-PER-LAYER` (per-layer reshard of `poolside/Laguna-S-2.1`, 219GB bf16, 48-layer 256-expert top-10 MoE)
- **Selection mode:** `gain_per_token` with MoE expert coverage + routing bypass
- **Target token floor:** 131,072 (soft target; redundant data is never added to fill it)
- **Final token count:** 59,075 (41 examples; floor not reached — every other shard had negative conditional gain)
- **Final coverage score:** 37,088.17 (from a starting score of 1,595,391.99)
- **Cumulative gain:** 1,558,303.82
- **MoE:** routing bypass ON — all 47 MoE layers x 256 experts (12,032 experts) received every token; min routed tokens per expert = 59,075; uncovered reference routed mass = 0.0

## What this dataset is

This is a pre-quantization calibration mix selected by the coverage scanner in
`optimize/calibration_coverage.py` running in MoE-aware mode: per-expert
routed-token counts and per-expert input Hessian diagonals are profiled via
hooks on the fused expert modules, expert coverage is folded into the score,
and `--moe-routing-bypass` raises every router's `top_k` from 10 to 256 during
scanning so all experts see every token's activations (measured cost only
~1.2-1.4x, not 25.6x, because the fused expert GEMMs already touch most
experts per 1K-token chunk).

The candidate pool was 23 ~32K-token shards from ten public corpora spanning
chat, instruction, web/edu, code, math, five languages, and long-form text:

- `lemon07r/bartowski-imatrix-v5-semantic` — 2 shards
- `neuralmagic/calibration` config `LLM` — 4 shards
- `ise-uiuc/Magicoder-OSS-Instruct-75K` (code instruct) — 2 shards
- `nvidia/OpenMathInstruct-2` (math) — 2 shards
- `wikimedia/wikipedia` `20231101.{zh,ru,ja,ar,de,ko}` — zh x2, others x1
- `emozilla/pg19` (long-form books) — 2 shards
- `allenai/tulu-3-sft-mixture` (broad SFT) — 2 shards
- `HuggingFaceFW/fineweb-edu` `sample-10BT` (edu web) — 2 shards

`calibration.parquet` contains the selected examples as a `messages` column
(chat-template compatible; raw text rows are wrapped as user messages and
rendered through the Laguna-S-2.1 chat template via Tokenicer, which applies
the `fix_mistral_regex` tokenizer correction). `dataset_info.json` holds
metadata, `report.json`/`report.md` the full scanner report, and `generate.py`
reproduces the pipeline. `report_v3_bypass_123k.md`/`.json` preserve the
previous 14-shard bypass run for comparison.

## How the mix was built

1. **Shard preparation** — sources are streamed until each hits its token
   budget and tokenized with 32 GIL-free threads (`PYTHON_GIL=0`,
   free-threaded CPython; fast-tokenizer encodes are thread-safe and results
   stay index-ordered, so no locks are needed). Raw text is capped at 6,000
   chars/row and rows are grouped to ~1,024-token chunks. A held-out
   reference (47,729 tokens) is built from the rows immediately following
   each candidate block in every source.
2. **Coverage scan** — bf16 sharded across 4 physical GPUs
   (`--physical-gpu 0,1,2,3`, `device_map=auto`), 32 greedy threads,
   `--moe-expert-coverage --moe-routing-bypass --target-tokens 131072
   --target-tokens-mode gain_per_token`. 144 layer-level target groups plus
   47 fused expert modules are profiled; ~105-127 tok/s with bypass.
3. **Greedy selection** — adds the shard with the largest conditional gain
   per token. Floors are soft targets: once every remaining shard has
   negative conditional gain (redundant), selection stops and the shortfall
   is reported as a warning rather than padding the mix with redundant data.

## Composition of the selected mix

| step | shard | tokens | cumulative | conditional gain | score after |
|------|-------|--------|------------|------------------|-------------|
| 1 | `wiki_zh_01` (Chinese Wikipedia) | 27,484 | 27,484 | 1,543,530.20 | 51,861.78 |
| 2 | `tulu_00` (tulu-3 SFT mixture) | 31,591 | 59,075 | 14,773.62 | 37,088.17 |

Warning emitted: `Target token floor 131072 not reached (reached 59075);
remaining datasets are redundant (negative conditional gain) and are never
added just to fill the floor.`

## Complementarity vs the selected mix (all remaining shards redundant)

`tulu_01` (-722), `nm_llm_01` (-1,257), `nm_llm_03` (-1,295), `nm_llm_02`
(-1,605), `wiki_zh_00` (-1,995), `nm_llm_00` (-2,052), `math_00` (-2,090),
`math_01` (-2,157), `wiki_ko_00` (-2,272), `wiki_ja_00` (-2,386), `code_01`
(-2,469), `code_00` (-2,502), `imatrix_01` (-2,780), `fineweb_edu_01`
(-2,923), `fineweb_edu_00` (-3,395), `imatrix_00` (-3,573), `wiki_de_00`
(-3,917), `wiki_ru_00` (-4,315), `pg19_00` (-4,879), `pg19_01` (-5,176),
`wiki_ar_00` (-5,439).

## Standalone scores (lower is better)

| shard | tokens | standalone score |
|-------|--------|------------------|
| `tulu_00` | 31,591 | 43,018.1 |
| `tulu_01` | 31,381 | 46,151.1 |
| `wiki_zh_00` | 29,673 | 49,433.1 |
| `wiki_ko_00` | 31,339 | 50,623.0 |
| `wiki_zh_01` | 27,484 | 51,861.8 |
| `wiki_ja_00` | 27,622 | 52,347.9 |
| `wiki_ru_00` | 32,033 | 57,174.9 |
| `imatrix_00` | 31,645 | 57,254.1 |
| `fineweb_edu_01` | 31,524 | 58,045.4 |
| `fineweb_edu_00` | 32,306 | 58,904.4 |
| `nm_llm_01` | 30,907 | 60,062.3 |
| `imatrix_01` | 31,743 | 60,239.1 |
| `nm_llm_03` | 30,742 | 62,589.6 |
| `wiki_ar_00` | 31,422 | 65,120.6 |
| `nm_llm_02` | 30,314 | 65,715.3 |
| `pg19_01` | 31,359 | 66,674.5 |
| `pg19_00` | 32,187 | 67,083.1 |
| `nm_llm_00` | 31,041 | 67,243.4 |
| `code_00` | 31,927 | 69,835.5 |
| `wiki_de_00` | 31,528 | 70,195.7 |
| `code_01` | 31,812 | 70,344.5 |
| `math_00` | 31,032 | 77,665.6 |
| `math_01` | 30,971 | 80,357.5 |

## Observations

- Even with 23 shards across ten corpora and full expert activation via
  routing bypass, the objective saturates at ~59K tokens: Laguna-S-2.1's
  activation-tail coverage (dense layers and all 12,032 experts) is covered by
  a 2-shard complementary pair, after which everything else is redundant.
- The winning pair is one non-English shard (zh Wikipedia) plus one broad
  English SFT shard (tulu-3) — the same "one multilingual + one broad chat"
  pattern seen in every earlier run, with the specific shards rotating as the
  pool changes.
- Under bypass, every expert sees every token, so per-expert calibration
  coverage equals the dense token count (59,075 tokens/expert) with zero
  uncovered routed mass; reaching a hard 128K would require accepting
  redundant data (not done by design) or substantially different sources.

## Reproducing it

Run `generate.py` from this folder with a free-threaded GPT-QModel venv:

```bash
LAGUNA_MOE_ROUTING_BYPASS=1 PYTHON_GIL=0 python generate.py
```

Environment overrides: `LAGUNA_MODEL_PATH` (default
`/monster/data/model/Laguna-S-2.1-PER-LAYER`), `LAGUNA_PHYSICAL_GPUS` (default
`0,1,2,3`), `LAGUNA_MOE_COVERAGE` (default `1`), `LAGUNA_MOE_ROUTING_BYPASS`
(default `0`; when enabled the per-expert floor equals `--target-tokens`, so
`LAGUNA_TARGET_MOE_EXPERT_TOKENS` must not be set), and
`LAGUNA_DEFUSE_EXPERTS` (default `0`; Defuser-based per-expert `nn.Linear`
profiling, unvalidated on the full Laguna checkpoint).

## Usage as a calibration dataset

The parquet can be passed directly to the scanner or to a quantizer that
accepts `messages` columns. For the scanner:

```bash
PYTHON_GIL=0 python optimize/calibration_coverage.py \
  --model /monster/data/model/Laguna-S-2.1-PER-LAYER \
  --trust-remote-code \
  --dataset dataset/calibration_mix_128k_laguna_s_2.1/calibration.parquet:calibration_mix \
  --reference ... \
  --physical-gpu 0,1,2,3 \
  --moe-expert-coverage --moe-routing-bypass \
  --output-dir /tmp/out
```

Replace `...` with your held-out reference dataset.
