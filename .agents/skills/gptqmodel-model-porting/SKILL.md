---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-model-porting
description: Port a model's config and modeling code into gptqmodel/models/ports on the Transformers 5.23 ABI, with no transformers.models.* imports, no checkpoint Python and no backward-compatibility paths. Use when adding a model, alias or definition, or when editing any port.
---

# Model porting on the Transformers 5.23 ABI

## The contract

GPT-QModel quantizes, runs inference and trains **only** on the Transformers
5.23 ABI. Every supported model is selected by its definition's
`LocalModelLoader` and runs GPT-QModel's own config and modeling code from
`gptqmodel/models/ports/<port>/`. There are no exceptions:

1. **Every `MODEL_MAP` definition has a `LocalModelLoader`.** No definition
   loads through `AutoModel*`, `AutoConfig`, checkpoint `auto_map` or
   `trust_remote_code`. A model that Transformers implements natively is still
   ported, so a Transformers upgrade cannot change GPT-QModel's math, module
   names or state_dict keys.
2. **Ports import Transformers framework infrastructure only.** Allowed:
   `PreTrainedModel`, `PreTrainedConfig`, `cache_utils` (`Cache`,
   `DynamicCache`, cache layers), `masking_utils`, `ALL_ATTENTION_FUNCTIONS` /
   `modeling_flash_attention_utils`, `modeling_rope_utils`,
   `modeling_layers` (`GradientCheckpointingLayer`), `generation`
   (`GenerationMixin`, generation config), `activations`, `modeling_outputs`,
   `initialization`, `integrations`, `processing_utils`,
   `tokenization_utils_*`/`TokenizersBackend`, `image_processing_utils*`,
   `feature_extraction_utils`, `utils`. **Nothing from `transformers.models.*`**:
   not a config, block, norm, MLP, attention, rotary, vision tower, processor
   or helper function. Copy what you need into the port.
3. **5.23 conventions only, zero back-compat.** Write the port the way 5.23
   native models are written (read `models/qwen3/modeling_qwen3.py` and
   `models/qwen3_5/modeling_qwen3_5.py` in the 5.23 source):
   - positions from `position_ids` / `past_key_values.get_seq_length()`; no
     `cache_position` parameters or bookkeeping;
   - caches are `Cache` subclasses or 5.23 cache layers; kwarg
     `past_key_values`; no tuple caches, `from_legacy_cache`,
     `to_legacy_cache`, `get_usable_length`, `seen_tokens`, `_reorder_cache`;
   - masks from `masking_utils` (`create_causal_mask`,
     `create_sliding_window_causal_mask`, `create_bidirectional_mask`, ...);
     no `_prepare_4d_*`, `AttentionMaskConverter`, `_update_causal_mask`;
   - attention through `ALL_ATTENTION_FUNCTIONS` / `eager_attention_forward`;
     layers subclass `GradientCheckpointingLayer`;
   - outputs via `@merge_with_config_defaults`, `@capture_outputs`,
     `_can_record_outputs` and `@can_return_tuple`; no manual
     `all_hidden_states` / `all_self_attns`, `use_return_dict`, or
     `if not return_dict: return tuple(...)`;
   - config fields `rope_parameters`, `dtype`, `layer_types`; heads take
     `logits_to_keep`; `_tied_weights_keys` is a dict;
   - weight init through `transformers.initialization`;
   - no `_supports_cache_class` / `_supports_static_cache` /
     `_supports_quantized_cache` / `_supports_flash_attn_2`, no version
     checks, no `try/except ImportError` around Transformers symbols.
   A released `config.json` may still contain an old field such as
   `rope_scaling`; convert it once at config-parse time. Nothing downstream
   reads the old name.
4. **No global changes to Transformers.** Ports never register with
   `AutoConfig`/`AutoModel`, and GPT-QModel never patches, monkeypatches or
   registers into Transformers or the HF ecosystem (optimum, OpenVINO,
   safetensors, huggingface_hub), not even temporarily or scoped. GPT-QModel
   checkpoints load through `GPTQModel.load`.
   `tests/test_transformers_unpatched.py` enforces this.
5. **Checkpoint fidelity.** Keep the released checkpoint's state_dict keys,
   numerics and module names. A port reproducing a non-native original keeps
   that original's math; a port of a Transformers-native family must match
   Transformers 5.23 exactly.

## Procedure

1. **Source.** Start from the 5.23 implementation (native family) or the
   pinned original release (non-native). Record the source path/revision in
   the port's `NOTICE`. Keep upstream copyright headers.
2. **Vendor.** Create `gptqmodel/models/ports/<port>/` with `__init__.py`,
   `configuration_<port>.py`, `modeling_<port>.py` (plus processor/tokenizer
   modules if the definition loads them). Rewrite relative imports to the
   allowed framework modules above; replace every `transformers.models.*`
   import with local code (cross-family pieces such as a SigLIP tower are
   copied into the port that uses them, or into a shared port that both
   import).
3. **Bind.** Give the definition a `LocalModelLoader(config_class=...,
   model_class=..., processor_class=..., tokenizer_class=...)`; set
   `config_architectures` only for released configs that omit `model_type`.
4. **Prove it.**
   - **Native-family oracle:** on a tiny random config, the port's logits,
     generated tokens (cached generation vs no-cache full forward) and
     state_dict keys/shapes equal Transformers 5.23's native class exactly.
     The oracle may import `transformers.models.*` **in tests only**.
   - **Release inventory:** the full-size meta model's key/shape inventory
     matches the released checkpoint index (store a sha256, not the JSON).
   - **Isolation:** loads without `trust_remote_code` and without the bypass,
     and ignores a hostile `auto_map`.
   - **Plans:** module-plan tests and an explicit AWQ policy
     ([model-plan tests](../gptqmodel-model-plan-tests/SKILL.md),
     [AWQ onboarding](../gptqmodel-awq-model-onboarding/SKILL.md)).
   - **Guard:** `tests/test_model_port_policy.py` passes (it fails on any
     `transformers.models.*` import in a port, any stale 4.x pattern, or any
     definition without a `LocalModelLoader`).
5. **Keep it lean.** Ship only runtime code/data in the package. Test fixtures
   live under `tests/fixtures/ports/<port>/`; prefer digests to large JSON.
