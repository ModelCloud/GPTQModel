# Per-Layer Micro-Quantization Ledger

## Goal

Add a public, per-layer, incremental quantization API to GPT-QModel Ultra that allows:

1. Quantizing one transformer layer at a time while leaving all other layers untouched.
2. Saving the partially quantized model.
3. Loading the partial checkpoint and quantizing another layer with a different calibration dataset and/or `QuantizeConfig`.
4. Repeating until every target layer is quantized.

Mixed quantization **method** (e.g. GPTQ + AWQ) is explicitly out of scope for this pass; mixed quantize-config for the same method is supported.

## Design

### Public API

`BaseQModel.quantize(..., layer_scope=None, freeze_others=True)` accepts an optional `layer_scope` argument:

- `None` -> existing behavior: quantize every non-excluded module.
- `int` -> quantize exactly that layer index (e.g. `0`).
- `list[int]` / `list[str]` -> quantize the listed layers.
- `str` (PCRE regex) -> quantize layers whose full layer name matches.
- `slice` -> quantize a range of layers.

`freeze_others=True` adds dynamic overrides that exclude all non-scope transformer layers so the loop stops after the last scope layer and no dense modules outside the scope are re-quantized.

`BaseQModel.requant(quantize_config=None, calibration, layer_scope=None, freeze_others=True, ...)` is a new method for continuing from a partial (or fully) quantized model:

- Optionally replaces `self.quantize_config` with a brand-new config.
- Applies `layer_scope` exclusions.
- Calls `self.quantize(...)`.
- Leaves non-matched modules untouched (already quantized modules are skipped by `ModuleLooper.create_named_modules`).
- If `layer_scope` is omitted, it defaults to all layers that are not already quantized.

### Dynamic override plumbing

Layer scope is implemented as temporary `quantize_config.dynamic` entries.  During a scoped call we emit **negative** overrides for non-scope layers:

```text
-:.*\.layers\.<i>\..*  -> False   for every layer index i that is NOT in scope
```

These negative patterns are **not** persisted.  After the call finishes we rebuild `quantize_config.dynamic` from the user's original overrides, strip the temporary negatives, and add **positive** overrides for any layers whose effective `bits`/`group_size`/`desc_act`/`sym` differ from the base config.  This lets later `save`/`from_quantized`/`requant` calls recreate the correct per-layer contract.

Dynamic pattern matching is performed with full module names (`model.layers.0.self_attn.q_proj` etc.). `find_last_quantized_layer_index` already evaluates `dynamic_get` against full layer paths, so the loop naturally stops after the highest scope layer.

### Skipping already-quantized modules during re-quantization

When a model is loaded from a partial checkpoint it contains a mixture of `BaseQuantLinear` modules (already quantized) and dense `nn.Linear` modules. `ModuleLooper.create_named_modules` must not try to re-preprocess `BaseQuantLinear` modules.

Implementation: in `create_named_modules`, after wrapping a module as `NamedModule`, check `isinstance(module.module, BaseQuantLinear)` and add the name to `skipped_modules` before `processor.preprocess` is called. This lets the forward still propagate through quantized layers and only dense modules in the scope are processed.

### Partial checkpoint load support

A saved partial checkpoint contains:

- `qweight` / `qzeros` / `scales` / `g_idx` for quantized modules.
- `weight` / `bias` for dense modules.

`from_quantized` currently builds a `modules` dict from the model and then replaces **all** of them with `QuantLinear` classes, which breaks dense modules because they have no `qweight` key in the checkpoint.

Fix: before `make_quant`, scan the checkpoint for keys ending in `.qweight`. Only modules whose `qweight` key is present remain in `modules`; dense modules are removed. `load_checkpoint_in_model_then_tie_weights` then loads `weight` into the dense `nn.Linear` modules and `qweight` into the new `QuantLinear` modules.

### Save

`BaseQModel.save` already branches on `self.quantized`. After `quantize`/`requant` of a partial model, `self.quantized` is `True` (set by the processor `finalize`). `save_quantized` writes all parameters to safetensors. Quantized layers write `qweight`/etc; dense layers write `weight`/etc.

Because the saved `quantize_config.json` now contains positive dynamic overrides for any layer whose effective config differs from the base, a full checkpoint can be reloaded correctly and a later `requant` can continue with a new base config while preserving already-quantized layers.

## Review fixes (post-creation of PR #150)

- [x] Restrict partial-checkpoint `qweight` probe to GPTQ/AWQ; exclude QQQ (uses `B`/`s_channel`/`s_group`) and only filter when the checkpoint is actually partial.
- [x] Honor `freeze_others=False` in `_apply_layer_scope`.
- [x] Preserve `offload_to_disk`, `offload_to_disk_path`, and `_offload_temp_dir` when `requant` replaces `QuantizeConfig`.
- [x] Rebuild `dynamic` in precedence order: negatives → captured positives → user positives.
- [x] Populate `module_tree_flags` before skipping already-quantized modules in `ModuleLooper.create_named_modules`.
- [x] Add `BaseQuantizeConfig._invalidate_dynamic_cache()` and call it before reassigning `dynamic`.
- [x] Emit per-module positive dynamic overrides from `_capture_quantized_layer_dynamic` instead of a single layer-wide pattern, so module-level mixed config is preserved correctly.
- [x] Merge `requant` `QuantizeConfig` field-by-field instead of replacing the object, preserving `meta`, `adapter`, existing `dynamic` overrides, and offload state.
- [x] Guard `BitBLASQuantLinear` and `BaseQuantizeConfig.extract_adapter_rank_patterns` against non-dict dynamic entries (layer-scope exclusions).
- [x] Back-fill `in_features`/`out_features`/`module_dtype` for skipped `BaseQuantLinear` modules in `ModuleLooper.create_named_modules`.

## Progress

- [x] Explored `base.py`, `module_looper.py`, `stage_layer.py`, `config.py`, `loader.py`, `model.py`.
- [x] Environment installed (uv venv + `pip install -e ".[test]"`).
- [x] Added `quantize(..., layer_scope, freeze_others)`.
- [x] Added `requant(...)`.
- [x] Skip already-quantized modules in `ModuleLooper.create_named_modules`.
- [x] Add checkpoint `qweight` scan to `from_quantized` loader.
- [x] Added focused unit + end-to-end tests in `tests/test_per_layer_quantization.py`.
- [x] Validated tiny CPU Llama end-to-end (layer 0 -> save/load -> layer 1 with different bits -> full save/load -> forward).
- [x] Existing `tests/test_llama_3_2.py` still passes.
- [x] `ruff check` (with `format/ruff.toml`) passes on changed files.
- [x] `git diff --check` passes.
- [x] Open PR: https://github.com/ModelCloud/GPT-QModel-Ultra/pull/150.
- [x] End-to-end CPU validation on `Qwen/Qwen2.5-0.5B` passed (via testing agent).
