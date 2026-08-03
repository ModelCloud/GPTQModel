---
name: test-template-reuse
description: Reuse existing tests under `tests/` as templates before writing new ones. Use when asked to add, run, or adapt a test so we avoid re-inventing harness code and stay consistent with current conventions.
---

# Reuse existing tests as templates

When a user asks for a test, verification script, or evaluation, do not write a new file from scratch.
Instead, locate the closest existing test and adapt it with the smallest delta.

## Search

1. Start by searching `tests/` for files that already exercise the same model, backend, format, bit width, or kernel.
   - `find_file_by_name` for naming patterns: `*qwen*`, `*llama3_2*`, `*gptq_p*`, `*pangolin*`, `*3bit*`.
   - `grep` for relevant symbols: model IDs (`/monster/data/model/...`, `/mnt/...`), `BACKEND`, `FORMAT`, `QuantizeConfig`, test classes inheriting `ModelTest` or `GptqPAccuracyBase`.
2. Read every promising candidate with `read`, especially shared base classes (`model_test.py`, `gptq_p_accuracy_base.py`) and focused kernel files (`tests/test_planar_triton_kernels.py`, `tests/kernels/test_*.py`).

## Choose the template

- Model-level e2e tests: subclass `ModelTest` or `GptqPAccuracyBase`; change only `NATIVE_MODEL_ID`, `BITS`, `FORMAT`, `LOAD_BACKEND`, `SAVE_PATH`, and expected metrics.
- Kernel-level correctness: copy the closest parametrize block and replace synthetic weights with real model weights when asked, or keep synthetic if the goal is kernel isolation.
- Full-model vs unit: prefer a unit/layer test when the goal is to verify one kernel/format on one model; only scale to a full-model test when the user asks for post-quant accuracy or metrics.

## Adapt, do not invent

- Keep the original imports, helpers, fixtures, and assertions.
- Change only the variables that differ (model path, bit width, backend, batch size, layer path).
- If expected metrics are unknown, set a permissive tolerance or skip metric assertions; do not invent numbers.
- Preserve existing `CUDA_VISIBLE_DEVICES`, `nvidia-smi` preflight, and `ruff`/`pytest` conventions.

## Verify

- Run `ruff check` on the adapted/new file.
- Run `pytest -q` on only that file before any broader suite.
- When possible, compare the new result against a dense or higher-precision reference and print max abs diff / mean squared diff.

## When to add a brand-new file

Only create a new test file if no existing test covers the same abstraction (e.g., a new model family, a new kernel category, or a new evaluation task). Otherwise, add a new test class or parametrized case to the most relevant existing file.
