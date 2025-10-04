# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
import re
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, GenerationConfig, PreTrainedModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from ..utils.logger import setup_logger


log = setup_logger()

GENERATION_SAMPLING_FIELDS = ("temperature", "top_p")

_DUPLICATE_CONFIG_PATTERN = re.compile(r"'([^']+)' is already used by a Transformers config")


def _ensure_pretrained_model_defaults():
    fallback_attrs = {
        "is_parallelizable": False,
        "_no_split_modules": (),
        "_keep_in_fp32_modules": (),
        "_skip_keys_device_placement": (),
    }
    for name, value in fallback_attrs.items():
        if not hasattr(PreTrainedModel, name):
            setattr(PreTrainedModel, name, value)


_ensure_pretrained_model_defaults()


def _clear_generation_field(cfg: GenerationConfig, field: str) -> bool:
    """Remove a generation field from the config and return True if mutated."""
    changed = False
    # GenerationConfig keeps its data in both __dict__ and _internal_dict
    for attr in ("__dict__", "_internal_dict"):
        container = getattr(cfg, attr, None)
        if isinstance(container, dict) and field in container:
            container.pop(field, None)
            changed = True

    if hasattr(cfg, field):
        try:
            delattr(cfg, field)
            changed = True
        except AttributeError:
            pass

    if hasattr(cfg, field):
        value = getattr(cfg, field)
        if value is not None:
            setattr(cfg, field, None)
            changed = True

    return changed


def _deregister_auto_config(model_type: str) -> bool:
    removed = False
    for attr in ("_mapping", "_extra_content", "_modules"):
        container = getattr(CONFIG_MAPPING, attr, None)
        if isinstance(container, dict) and model_type in container:
            container.pop(model_type, None)
            removed = True
    return removed


def safe_auto_config_from_pretrained(*args, **kwargs):
    trust_flag = kwargs.get("trust_remote_code")
    prev_env = None
    if trust_flag:
        prev_env = os.environ.get("TRANSFORMERS_TRUST_REMOTE_CODE")
        os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
    try:
        return AutoConfig.from_pretrained(*args, **kwargs)
    except ValueError as err:
        message = str(err)
        if "already used by a Transformers config" not in message:
            raise
        match = _DUPLICATE_CONFIG_PATTERN.search(message)
        if not match:
            raise
        model_type = match.group(1)
        if not _deregister_auto_config(model_type):
            raise
        log.info("Transformers: cleared cached config registration for `%s` and retrying load.", model_type)
        return AutoConfig.from_pretrained(*args, **kwargs)
    finally:
        if trust_flag:
            if prev_env is None:
                os.environ.pop("TRANSFORMERS_TRUST_REMOTE_CODE", None)
            else:
                os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = prev_env


def _sanitize_generation_config(cfg: GenerationConfig, *, drop_sampling_fields: bool = False) -> bool:
    changed = False
    if cfg is None:
        return changed

    do_sample = getattr(cfg, "do_sample", None)
    if do_sample is not True:
        for field in GENERATION_SAMPLING_FIELDS:
            if _clear_generation_field(cfg, field):
                changed = True

    if drop_sampling_fields:
        for field in GENERATION_SAMPLING_FIELDS:
            if _clear_generation_field(cfg, field):
                changed = True
    return changed


def _load_sanitized_generation_config(path: str) -> Optional[GenerationConfig]:
    try:
        config_dict, kwargs = GenerationConfig.get_config_dict(path)
    except Exception:
        return None

    cleaned = dict(config_dict)
    removed = False
    for field in GENERATION_SAMPLING_FIELDS:
        if field in cleaned:
            cleaned.pop(field, None)
            removed = True
    if cleaned.get("do_sample") is not True:
        cleaned["do_sample"] = True

    cfg = GenerationConfig.from_dict(cleaned, **kwargs)
    if removed:
        log.info("Model: Removed unsupported sampling fields from `generation_config.json` during load.")
    _sanitize_generation_config(cfg, drop_sampling_fields=True)
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
                    _sanitize_generation_config(cfg, drop_sampling_fields=True)
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
    _sanitize_generation_config(cfg, drop_sampling_fields=True)
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
    for field in GENERATION_SAMPLING_FIELDS:
        if field in data:
            data.pop(field, None)
            changed = True

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

    configured_dtype = init_kwargs.pop("dtype", None)
    if dtype is None and configured_dtype is not None:
        dtype = configured_dtype
    elif dtype is not None and configured_dtype is not None and configured_dtype != dtype:
        log.info("Shell model: overriding duplicate dtype argument from kwargs with explicit `dtype` parameter.")
    init_kwargs.pop("device_map", None)
    init_kwargs.pop("_fast_init", None)
    # All nn.Parameters and buffers are created

    # All nn.Parameters and buffers are created on 'meta' and initializers are skipped.
    if dtype is not None:
        setattr(config, "dtype", dtype)
        store = getattr(config, "__dict__", None)
        if isinstance(store, dict) and store.get("torch_dtype") != dtype:
            store.pop("torch_dtype", None)

    with init_empty_weights(include_buffers=True):
        shell = loader.from_config(
            config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **init_kwargs
        )

    return shell
