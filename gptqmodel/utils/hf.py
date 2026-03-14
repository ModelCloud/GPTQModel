# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from transformers import GenerationConfig, PreTrainedModel


# Compatibility wrapper for no_init_weights across different transformers versions
# transformers >= 5.0.0: from transformers.initialization import no_init_weights
# transformers < 5.0.0: from transformers.modeling_utils import no_init_weights
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from ..utils.logger import setup_logger


__all__ = ["no_init_weights"]

log = setup_logger()

def _resolve_input_embedding_weight_name(model: PreTrainedModel) -> Optional[str]:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return None

    try:
        input_embeddings = get_input_embeddings()
    except Exception:
        return None

    if input_embeddings is None:
        return None

    weight = getattr(input_embeddings, "weight", None)
    if weight is None:
        return None

    for name, param in model.named_parameters(remove_duplicate=False):
        if param is weight:
            return name

    for name, module in model.named_modules(remove_duplicate=False):
        if module is input_embeddings:
            return f"{name}.weight" if name else "weight"

    return None


# Older remote model files sometimes store `_tied_weights_keys` as a plain list
# like `["lm_head.weight"]`. transformers 5.x now expects `{target: source}`,
# and otherwise later save/load helpers fail with `'list' object has no attribute 'keys'`.
def _resolve_legacy_tied_weights_mapping(model: PreTrainedModel, tied_mapping) -> dict[str, str]:
    if not isinstance(tied_mapping, (list, tuple, set)):
        return {}

    if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        return {}

    source_name = _resolve_input_embedding_weight_name(model)
    if source_name is None:
        return {}

    return {
        # Legacy list entries only name the tied target, so resolve them back
        # to the input embedding weight name expected by transformers 5.x.
        target_name: source_name
        for target_name in tied_mapping
        if isinstance(target_name, str) and target_name != source_name
    }


# Rewrite legacy list-based `_tied_weights_keys` in-place so transformers 5.x
# save/load code stops crashing on older trust_remote_code models that still use
# the pre-5.x list format.
def _normalize_legacy_tied_weights_keys(model: PreTrainedModel) -> None:
    for _name, submodule in model.named_modules(remove_duplicate=False):
        tied_mapping = getattr(submodule, "_tied_weights_keys", None)
        if not isinstance(tied_mapping, (list, tuple, set)):
            continue

        if isinstance(submodule, PreTrainedModel):
            submodule._tied_weights_keys = _resolve_legacy_tied_weights_mapping(submodule, tied_mapping)
        else:
            submodule._tied_weights_keys = {}


# Bridge a few transformers 5.x API changes so older trust_remote_code model
# files still import and initialize without editing the cached remote source.
def _patch_transformers_remote_code_compat() -> None:
    try:
        from transformers.utils import import_utils
    except Exception:
        return

    if not hasattr(import_utils, "is_torch_fx_available"):
        # transformers 5.x removed `import_utils.is_torch_fx_available`, but
        # older remote model files still import it during module import.
        def is_torch_fx_available() -> bool:
            return hasattr(torch, "fx")

        import_utils.is_torch_fx_available = is_torch_fx_available

    if not getattr(PreTrainedModel, "_gptqmodel_legacy_tied_weights_patch", False):
        original_get_expanded_tied_weights_keys = PreTrainedModel.get_expanded_tied_weights_keys

        def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
            # transformers 5.x expects `_tied_weights_keys` to be a dict, while
            # older trust_remote_code models still declare `["lm_head.weight"]`.
            # Handle the legacy form here so HF tied-weight expansion still works.
            tied_mapping = getattr(self, "_tied_weights_keys", None)
            if not isinstance(tied_mapping, (list, tuple, set)):
                return original_get_expanded_tied_weights_keys(self, all_submodels=all_submodels)

            if all_submodels:
                expanded_tied_weights = {}
                for prefix, submodule in self.named_modules(remove_duplicate=False):
                    if isinstance(submodule, PreTrainedModel):
                        submodel_tied_weights = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                        if prefix != "":
                            submodel_tied_weights = {
                                f"{prefix}.{k}": f"{prefix}.{v}" for k, v in submodel_tied_weights.items()
                            }
                        expanded_tied_weights.update(submodel_tied_weights)
                return expanded_tied_weights

            if not getattr(getattr(self, "config", None), "tie_word_embeddings", False):
                return {}

            resolved_mapping = _resolve_legacy_tied_weights_mapping(self, tied_mapping)
            self._tied_weights_keys = resolved_mapping
            return resolved_mapping

        PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys
        PreTrainedModel._gptqmodel_legacy_tied_weights_patch = True


# Restore the pre-transformers-5 RoPE config shape expected by older MiniCPM
# remote code before HF instantiates the architecture from config.
def _normalize_remote_code_config_compat(config: Any) -> None:
    # transformers 5.x normalizes RoPE config to `rope_type`, but older
    # MiniCPM remote code still reads `rope_scaling["type"]` or expects `None`.
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict) or "type" in rope_scaling:
        return

    rope_type = rope_scaling.get("rope_type")
    factor = rope_scaling.get("factor")

    if rope_type in (None, "default") and factor is None:
        config.rope_scaling = None
        return

    config.rope_scaling = dict(rope_scaling)
    config.rope_scaling["type"] = rope_type


_patch_transformers_remote_code_compat()

def _sanitize_generation_config(cfg: GenerationConfig, *, drop_sampling_fields: bool = False) -> bool:
    changed = False
    if cfg is None:
        return changed

    if getattr(cfg, "do_sample", None) is not True:
        cfg.do_sample = True
        changed = True

    return changed


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    try:
        config_dict, kwargs = GenerationConfig.get_config_dict(path)
    except Exception:
        return None

    cleaned = dict(config_dict)
    if cleaned.get("do_sample") is not True:
        cleaned["do_sample"] = True

    cfg = GenerationConfig.from_dict(cleaned, **kwargs)
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    return cfg


# TODO FIXME! Pre-quantized use AutoModelForCausalLM.from_pretrained() but post-quantized use AutoModelForCausalLM.from_config()
def autofix_hf_model_config(model: PreTrainedModel, path: str = None):
    if model.can_generate():
        # sync config first
        if path:
            log.info(f"Model: Loaded `generation_config`: {model.generation_config}")
            try:
                cfg = _load_sanitized_generation_config(path)
                if cfg is None:
                    cfg = GenerationConfig.from_pretrained(pretrained_model_name=path, do_sample=True)
                    _sanitize_generation_config(cfg, drop_sampling_fields=False)
                if cfg != model.generation_config:
                    # migrated pad_token_id to config
                    if hasattr(model.generation_config, "pad_token_id"):
                        cfg.pad_token_id = model.generation_config.pad_token_id

                    model.generation_config = cfg
                    log.info(
                        "Model: Auto-fixed `generation_config` mismatch between model and `generation_config.json`.")
                    log.info(f"Model: Updated `generation_config`: {model.generation_config}")
                else:
                    pass
                    # logger.info(f"Model: loaded `generation_config` matching `generation_config.json`.")
            except Exception:
                log.info("Model: `generation_config.json` not found. Skipped checking.")

        # print(f"Before autofix_hf_model_config: {model.generation_config}")
        autofix_hf_generation_config(model.generation_config)
        # print(f"After autofix_hf_model_config: {model.generation_config}")


def autofix_hf_generation_config(cfg: GenerationConfig):
    _sanitize_generation_config(cfg, drop_sampling_fields=False)
    # HF has recently started to perform very strict validation model save which results in warnings on load()
    # to become exceptions on save().
    if cfg.do_sample is False:
        errors = 0
        if hasattr(cfg, "temperature") and cfg.temperature is not None and cfg.temperature != 1.0:
            errors += 1
        if hasattr(cfg, "top_p") and cfg.top_p is not None and cfg.top_p != 1.0:
            errors += 1
        if hasattr(cfg, "min_p") and cfg.min_p is not None:
            errors += 1
        if hasattr(cfg, "typical_p") and cfg.typical_p is not None and cfg.typical_p != 1.0:
            errors += 1
        # contrastive search uses top_k
        if (hasattr(cfg, "top_k") and cfg.top_k is not None and cfg.top_k != 50) and (hasattr(cfg, "penalty_alpha") and cfg.penalty_alpha is None):
            errors += 1
        if hasattr(cfg, "epsilon_cutoff") and cfg.epsilon_cutoff is not None and cfg.epsilon_cutoff != 0.0:
            errors += 1
        if hasattr(cfg, "eta_cutoff") and cfg.eta_cutoff is not None and cfg.eta_cutoff != 0.0:
            errors += 1

        # fix wrong do_sample
        if errors > 0:
            cfg.do_sample = True
            log.info("Model: Auto-Fixed `generation_config` by setting `do_sample=True`.")


def sanitize_generation_config_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return False

    changed = False

    if data.get("do_sample") is not True:
        data["do_sample"] = True
        changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

    return changed

# load hf model with empty tensors on meta device (zero tensor memory usage)
def build_shell_model(
    loader,
    config: Any,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **model_init_kwargs,
):
    """
    Instantiate the HF architecture with all parameters and buffers on 'meta' (no CPU RAM).
    Preserves the full module topology (Linear/MLP/Attention/etc.).

    Args:
        model_id_or_path: Hugging Face model ID or local path.
        dtype: Target dtype for model parameters (replaces `torch_dtype`).
        trust_remote_code: Allow loading custom model classes.
    """
    init_kwargs = model_init_kwargs.copy()

    del init_kwargs["device_map"]
    del init_kwargs["_fast_init"]
    # All nn.Parameters and buffers are created

    if trust_remote_code:
        _patch_transformers_remote_code_compat()
        _normalize_remote_code_config_compat(config)

    # All nn.Parameters and buffers are created on 'meta' and initializers are skipped.
    pb = log.spinner(title="Model loading...", interval=0.1)
    try:
        with init_empty_weights(include_buffers=True):
            shell = loader.from_config(
                config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                **init_kwargs
            )
    finally:
        pb.close()

    if trust_remote_code and isinstance(shell, PreTrainedModel):
        _normalize_legacy_tied_weights_keys(shell)

    return shell
