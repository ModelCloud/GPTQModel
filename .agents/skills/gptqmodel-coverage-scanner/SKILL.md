---
name: gptqmodel-coverage-scanner
description: |
  End-to-end testing of `optimize/calibration_coverage.py`, a pre-quantization
  calibration dataset coverage scanner. Covers dependency setup, model selection
  (Llama/Qwen-style `*_proj` and GPT-2-style `Conv1D` naming), CPU-only execution,
  output validation, and expected complementarity behavior.
---

# GPT-QModel coverage scanner (`optimize/calibration_coverage.py`)

## One-liner

Install the repo in an editable venv, pick a small dense model, provide two
`--dataset` text files and one `--reference` text file separated by the script’s
`--text-separator`, and run on CPU (no `--physical-gpu`).

## Devin Secrets Needed

None for public models. Optional `HF_TOKEN` if you hit unauthenticated HF Hub rate
limits (set as `HF_TOKEN` env).

## Environment setup

```bash
python -m venv /tmp/gptq-coverage-venv
/tmp/gptq-coverage-venv/bin/pip install --upgrade pip setuptools wheel
/tmp/gptq-coverage-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
cd /path/to/GPT-QModel-Ultra
/tmp/gptq-coverage-venv/bin/pip install -e .[test,quality]
```

- `torch` must be present before the editable install because `gptqmodel/__init__.py`
  imports `torch` at package load.
- `pip install -e .` is pure setuptools and does not compile `gptqmodel_ext`.
  The script only needs `Fallback` and `resolve_threshold` from `gptqmodel`.
- If `torch` is older than the internal minimum (e.g. 2.8.0 vs a requested >=2.11.0),
  the package prints a warning and skips cpp extensions; the scanner still runs.

## Model selection

The script matches modules by suffix: `q_proj`, `k_proj`, `v_proj`, `gate_proj`,
`up_proj`, `down_proj`, plus GPT-2/OpenAI-GPT `Conv1D` modules `c_attn`, `c_fc`, and
`c_proj`. It then deduplicates shared inputs (`q/k/v` and `gate/up`, and `c_attn`).

- **Llama/Qwen-style dense models** (`HuggingFaceTB/SmolLM2-135M-Instruct`,
  `Qwen/Qwen2.5-0.5B-Instruct`, `meta-llama/Llama-3.2-1B` if access is granted)
  are expected to have `*_proj` layers.
- **GPT-2 / OpenAI-GPT style models** (`openai-community/gpt2`) use `Conv1D`
  layers named `c_attn`, `c_fc`, `c_proj`. The script groups `c_attn` as a shared
  input attention group, `c_fc` as a shared input MLP group, and treats both
  attention `c_proj` and MLP `c_proj` as separate down-projection modules.
- If no target modules are found, the script prints a warning and all scores are
  zero; verify the model has one of the supported naming schemes.

## Minimal invocation

```bash
/tmp/gptq-coverage-venv/bin/python optimize/calibration_coverage.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --dataset /tmp/coverage_test/data/cal1.txt \
  --dataset /tmp/coverage_test/data/cal2.txt \
  --reference /tmp/coverage_test/data/ref.txt \
  --output-dir /tmp/coverage_test/out \
  --max-samples 2 \
  --concat-size 64 \
  --min-length 5 \
  --torch-dtype float32
```

Create `.txt` files with `--text-separator` default `===========`:

```text
This is the first calibration sample.===========
This is the second calibration sample.===========
This is the third calibration sample.
```

## Expected runtime output checks

- `[model] Found N target groups (M modules)` with `N > 0`.
- `[scan] <name>: done, T total tokens` for each dataset and the reference.
- `[score] Computing ...` followed by `Wrote coverage_report.json/md to <dir>`.
- If `Found 0 target groups` appears, the model module names are not supported.

## Output file checks

- `coverage_report.json` contains top-level keys:
  `reference`, `per_dataset`, `greedy_ranking`, `selected_mix`, `complementarity`, `fallback`.
- `coverage_report.md` mirrors the JSON with section headers:
  `## Per-dataset standalone scores`, `## Greedy ranking`, `## Selected mix`,
  `## Complementarity vs selected mix`.
- All `standalone_score` values should be non-negative finite numbers.
- `greedy_ranking[].score_after` should be monotonically non-increasing.
- `selected_mix.score` should equal the last `greedy_ranking[].score_after`.

## Complementarity table semantics

- Datasets that are part of the greedy-selected mix show `verdict: "selected"` and
  `conditional_gain` equal to the gain they contributed when they were added.
- Datasets **not** selected show `verdict: "complementary"` or `"redundant"`
  depending on whether adding them to the final selected mix reduces the score.
- The complementarity table header is `| dataset | conditional gain | verdict |`.

## CPU vs GPU preflight

- Omit `--physical-gpu` to run on CPU; the script falls back to `float32` on CPU
  even if `--torch-dtype bfloat16` or `float16` is requested.
- Passing `--physical-gpu N` makes the script call `nvidia-smi` (stdlib only)
  *before* importing `torch`. On a machine without `nvidia-smi` it fails with
  `FileNotFoundError: nvidia-smi`, proving the preflight is stdlib-only and
  torch is not loaded first.

## Common pitfalls

- Too few tokens per module may trigger `fallback` entries when module token counts
  fall below `--fallback-threshold` (default `0.5%` of the dataset token count).
- `gpt2` and other `Conv1D` models now work, but only if their layers follow the
  `c_attn` / `c_fc` / `c_proj` naming convention.
