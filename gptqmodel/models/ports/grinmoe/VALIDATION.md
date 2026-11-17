# GRIN-MoE validation

Environment: Python 3.14.6 free-threaded (`sys._is_gil_enabled() == False`),
Transformers 5.23.0, Torch 2.15.0+cpu. Commands run from the repository root;
pytest selects `tests/pytest.ini`. No trained weight shards were downloaded.

## Reproducible offline suite

```bash
PYTHON_GIL=0 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest -q \
  tests/test_grinmoe_local_port.py \
  tests/module_tree/test_grinmoe_local_port.py \
  tests/test_awq_scale_search_safety.py \
  tests/test_awq_moe_router_scaling.py \
  tests/test_awq_empty_groups.py \
  tests/test_awq_position_kwargs.py \
  tests/test_awq_scale_search_memory.py \
  tests/test_bloom_awq_replay.py \
  tests/module_tree/test_awq_relationships.py \
  tests/module_tree/test_bloom_awq_relationships.py \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_definition_subset_ordering.py \
  tests/test_local_model_loading.py \
  tests/test_hf_config_compat.py \
  tests/test_trust_remote_code_isolation.py -rs --tb=short
```

Result: **705 passed, 4 skipped, 14 warnings in 10.31 s**. Skips are the
unavailable DeciLM, EXAONE and Nanbeige checkpoint fixtures in
`test_awq_position_kwargs.py`, and optional `peft.tuners.lora.awq` coverage in
`test_hf_config_compat.py`. No GRIN-MoE case skips. Warnings concern existing
Transformers `rope_scaling` deprecation.

```bash
ruff check gptqmodel/models/definitions/grinmoe.py \
  gptqmodel/models/ports/grinmoe \
  tests/test_grinmoe_local_port.py tests/module_tree/test_grinmoe_local_port.py
git diff --check
```

Both pass. `setuptools.find_packages()` discovers
`gptqmodel.models.ports.grinmoe`. There are no active repository commit hooks.

## Model-specific invariants

| Test node (under `tests/`) | Coverage |
| --- | --- |
| `test_grinmoe_local_port.py::test_definition_owned_selection_ignores_checkpoint_code` | Poisoned `auto_map`; both API-trust and existing safety-bypass values; exact package classes; no checkpoint-code marker. |
| `test_grinmoe_local_port.py::test_public_dense_load_and_generation` | Public `GPTQModel.load`, all trust/bypass combinations, real forward/generation and plan consumer. |
| `test_grinmoe_local_port.py::test_full_model_matches_source_equations` | Independent equations for attention, LayerNorm, experts and two-pass routing; eager/SDPA, biases on/off, 1/2/4 KV heads, hidden states/router logits/loss. |
| `test_grinmoe_local_port.py::test_sparsemixer_is_not_normalized_top2_softmax` | Distinct distributions, negative scores, ties and asymmetric experts. |
| `test_grinmoe_local_port.py::test_rotary_retains_upstream_fp32_phases_after_low_precision_cast` | Exact source-table phases at long/nonuniform positions after FP32/FP16/BF16 model casts. |
| `test_grinmoe_local_port.py::test_prefill_cache_continuation_generation_and_causality` | Real dynamic-cache prefill and two-token continuation; distinct left padding, causal perturbation, cached/uncached generation, FP32/BF16, eager/SDPA, window crossing. |
| `test_grinmoe_local_port.py::test_sdpa_attention_output_fallback_preserves_upstream_window_boundary` | SDPA attention-output fallback uses eager semantics, including the window's exact lower boundary. |
| `test_grinmoe_local_port.py::test_generation_config_keeps_full_cache_despite_sliding_window_metadata` | Inherited metadata cannot replace full-cache semantics; generation crosses the window with cached/uncached equality. |
| `test_grinmoe_local_port.py::test_explicit_additive_attention_mask_and_inputs_embeds_match_padding_mask` | Explicit additive 4D mask and embedding-input path equal ordinary 2D padding-mask output. |
| `test_grinmoe_local_port.py::test_failed_mask_validation_preserves_cache_and_retry_matches_fresh` | Invalid continuation mask fails before cache mutation; same cache remains usable and retry matches full fresh output. |
| `test_grinmoe_local_port.py::test_safetensors_reload_resave_and_initialized_state_dict_load` | Tied/untied dense weights, repeated safe save/reload, already initialized targets, `assign=False/True`, exact logits. |
| `test_grinmoe_local_port.py::test_training_sparsemixer_gradients_and_tuple_outputs` | Finite nonzero router gradients; loss backward and tuple/attention outputs. |
| `test_grinmoe_local_port.py::test_plain_awq_quantizes_and_packs_every_projection_without_norm_folds` | Actual CPU plain-AWQ pack/save/reload with both fallback policies; all attention/expert projections packed, no norm/router folds, exact packed tensors and logits after public reload. |
| `module_tree/test_grinmoe_local_port.py::test_actual_checkpoint_owners_plan_roles_lifecycle_and_looper` | Actual CPU/meta model; all state-dict owners, explicit nonquantized flags, exact quantized set, expert placeholders, roles, lifecycle/looper consumers, negative undeclared paths, bias/GQA/expert-count variants. |
| `module_tree/test_grinmoe_local_port.py::test_exact_published_checkpoint_key_inventory_on_real_meta_model` | Full published dimensions; exact upstream key digest/count and representative shapes without allocating checkpoint tensors. |

## Red/green evidence

1. **Remote-selection regression.** In a disposable Python process only, replace
   `GrinMoeQModel.loader` with `AutoModelForCausalLM` and restore
   `require_trust_remote_code=True`. Run
   `test_definition_owned_selection_ignores_checkpoint_code[False-False]`.
   It fails at the behavioral assertion requiring a package-owned supported
   config: Transformers demands checkpoint custom code. With the shipped loader
   the same node passes, as do all four permission combinations. This is not a
   collection/import failure and no production source is mutated.
2. **Eager window boundary.** The pinned source's mask helper includes
   `key == query - sliding_window`. A preliminary strict `>` implementation
   fails the three eager `test_full_model_matches_source_equations` variants;
   the three SDPA controls pass. Changing only the boundary to `>=` makes all
   six pass. The attention-output regression additionally pins exactly three
   attended keys when `sliding_window=2`.
3. **Low-precision rotary phases.** Upcasting a cast `inv_freq` buffer cannot
   recover FP32 phases. The rotary test fails for FP16 and BF16 while its FP32
   control passes; maximum cosine errors are 0.50927734375 and 1.015625.
   Rebuilding the short FP32 frequency vector on the input device makes all
   three exact comparisons pass. No mutable cosine cache is introduced.
4. **Generation cache ownership.** Restoring `layer_types` from arbitrary
   metadata after establishing the model's full-cache declaration fails
   `test_generation_config_keeps_full_cache_despite_sliding_window_metadata`.
   Setting the architecture-owned declaration after base-config initialization
   passes the invariant and cached/uncached generation through the window.

## Direct source cross-check

In addition to the committed independent functional oracle, an offline experiment
used the hash-verified pinned source listed in `PROVENANCE.md`. Only inspected
decoder/attention/rotary/expert/SparseMixer AST definitions were evaluated in an
explicit namespace, not the checkpoint module or its imports. Identical loaded
layer weights, embedding/norm/head tensors, and the original Transformers mask
helper were used; an adapter mapped old `get_usable_length` to current dynamic
cache length. Two distinct padded samples, a window of three, eager and SDPA,
uncached full forward, prefill and two-token continuation gave **maximum
absolute FP32 logit error 0.0**. This source experiment is supplementary; normal
tests are fully offline and do not require upstream files or AST execution.

## Integration and remaining qualification

No central registry JSON entry changes: `models.grinmoe` remains
`GrinMoeQModel`, and `cases.GrinMoeQModel/gptq` retains the same seven ordered
blocks. There is no pre-existing GRIN-specific `tests/models` test or obsolete
Transformers-version requirement to remove.

Parent-owned follow-ups: add GRIN-MoE to the central local-port coverage table;
link the model-specific diagram from the shared reference/PR description;
replace the stale custom/remote description in `docs/fusion-model-coverage.md`
without claiming fusion qualification. If extending central AWQ family tests,
add `grinmoe` to `FAMILY_MODEL_TYPES` and `PLAIN_QUANTIZATION_FAMILIES`, return
the port's `tiny_config()` in `_tiny_config`, and instantiate via
`qmodel_cls.loader.from_config(...)` in `_tiny_model`, not HF Auto classes.

**Not qualified:** trained-weight quality/perplexity, uncached full checkpoint
tensor loading, GPU/ROCm kernels or dtypes, FlashAttention, multi-GPU/free-threaded
concurrent inference. Static/legacy-tuple caches are deliberately unsupported;
this port's cache API is `DynamicCache`. CPU execution with the GIL disabled is
not a claim of concurrent accelerator safety.
