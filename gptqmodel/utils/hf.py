# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect
import json
import sys
import warnings
from functools import lru_cache
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel


# Compatibility wrapper for no_init_weights across different transformers versions
# transformers >= 5.0.0: from transformers.initialization import no_init_weights
# transformers < 5.0.0: from transformers.modeling_utils import no_init_weights
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

from ..utils.logger import setup_logger


__all__ = [
    "no_init_weights",
    "normalize_hf_config_compat",
    "prepare_remote_code_compat",
    "prepare_remote_model_init_compat",
    "has_native_transformers_causallm_support",
    "resolve_trust_remote_code",
    "load_tokenizer",
]

log = setup_logger()
_TRUST_REMOTE_CODE_OVERRIDE_WARNED: set[tuple[str, str, str]] = set()


@lru_cache(maxsize=None)
def _detect_native_transformers_causallm_support(model_id_or_path: str) -> tuple[bool, Optional[str], Optional[str]]:
    try:
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=False)
    except Exception as exc:
        log.debug("HF: native transformers support check failed for `%s`: %s", model_id_or_path, exc)
        return False, None, None

    model_type = getattr(config, "model_type", None)
    try:
        model_cls = AutoModelForCausalLM._model_mapping[type(config)]
    except Exception as exc:
        log.debug(
            "HF: config `%s` for `%s` has no native AutoModelForCausalLM mapping: %s",
            type(config).__name__,
            model_id_or_path,
            exc,
        )
        return False, model_type, None

    return True, model_type, getattr(model_cls, "__name__", str(model_cls))


def resolve_trust_remote_code(model_id_or_path: Optional[str], *, trust_remote_code: bool) -> bool:
    if not trust_remote_code or not model_id_or_path:
        return trust_remote_code

    native_supported, model_type, model_cls_name = _detect_native_transformers_causallm_support(str(model_id_or_path))
    if not native_supported:
        return True

    warning_key = (str(model_id_or_path), model_type or "unknown", model_cls_name or "unknown")
    if warning_key not in _TRUST_REMOTE_CODE_OVERRIDE_WARNED:
        log.warning(
            "HF: overriding trust_remote_code=True to False for `%s` because model_type `%s` is integrated in installed transformers as `%s`.",
            model_id_or_path,
            model_type or "unknown",
            model_cls_name or "unknown",
        )
        _TRUST_REMOTE_CODE_OVERRIDE_WARNED.add(warning_key)

    return False


def has_native_transformers_causallm_support(model_id_or_path: Optional[str]) -> bool:
    if not model_id_or_path:
        return False

    native_supported, _, _ = _detect_native_transformers_causallm_support(str(model_id_or_path))
    return native_supported


# Older remote model files sometimes store `_tied_weights_keys` as a plain list
# like `["lm_head.weight"]`. transformers 5.x now expects `{target: source}`,
# and otherwise later save/load helpers fail with `'list' object has no attribute 'keys'`.
def _resolve_legacy_tied_weights_mapping(model: PreTrainedModel, tied_mapping) -> dict[str, str]:
    if not isinstance(tied_mapping, (list, tuple, set)):
        return {}

    if not getattr(model.config, "tie_word_embeddings", False):
        return {}

    input_embeddings = model.get_input_embeddings()
    input_weight = getattr(input_embeddings, "weight", None)
    if input_weight is None:
        return {}

    source_name = None
    for name, param in model.named_parameters(remove_duplicate=False):
        if param is input_weight:
            source_name = name
            break

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

    try:
        from transformers import cache_utils
    except Exception:
        cache_utils = None

    if not hasattr(import_utils, "is_torch_fx_available"):
        # transformers 5.x removed `import_utils.is_torch_fx_available`, but
        # older remote model files still import it during module import.
        def is_torch_fx_available() -> bool:
            return hasattr(torch, "fx")

        import_utils.is_torch_fx_available = is_torch_fx_available

    if cache_utils is not None and not hasattr(cache_utils, "SlidingWindowCache") and hasattr(cache_utils, "StaticCache"):
        # transformers 5.x folds sliding-window behavior into StaticCache
        # layers, but older remote code still imports the legacy symbol.
        cache_utils.SlidingWindowCache = cache_utils.StaticCache

    cache_base_cls = getattr(cache_utils, "Cache", None) if cache_utils is not None else None
    if cache_base_cls is not None and not hasattr(cache_base_cls, "get_max_length") and hasattr(cache_base_cls, "get_max_cache_shape"):
        # Older remote decoders expect `get_max_length()`, while newer
        # transformers renamed that API to `get_max_cache_shape()`.
        def get_max_length(self, layer_idx: int = 0) -> Optional[int]:
            max_length = self.get_max_cache_shape(layer_idx)
            return None if max_length is None or max_length < 0 else max_length

        cache_base_cls.get_max_length = get_max_length

    if cache_base_cls is not None and not hasattr(cache_base_cls, "get_usable_length") and hasattr(cache_base_cls, "get_seq_length"):
        # Recreate the pre-5.x cache eviction helper so remote attention code
        # can compute usable KV length against the newer cache interface.
        def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
            max_length = self.get_max_length(layer_idx=layer_idx)
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length

        cache_base_cls.get_usable_length = get_usable_length

    dynamic_cache_cls = getattr(cache_utils, "DynamicCache", None) if cache_utils is not None else None
    if dynamic_cache_cls is not None and not hasattr(dynamic_cache_cls, "to_legacy_cache"):
        # Older remote generation code still serializes cache state as
        # `Tuple[(key, value), ...]`; rebuild that view from layer storage.
        def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
            legacy_cache = ()
            for layer in self.layers:
                if not getattr(layer, "is_initialized", False):
                    continue
                legacy_cache += ((layer.keys, layer.values),)
            return legacy_cache

        dynamic_cache_cls.to_legacy_cache = to_legacy_cache

    if dynamic_cache_cls is not None and not hasattr(dynamic_cache_cls, "from_legacy_cache"):
        # Accept legacy tuple caches by replaying them into the current
        # layer-based DynamicCache implementation.
        @classmethod
        def from_legacy_cache(cls, past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                    cache.update(key_states, value_states, layer_idx)
            return cache

        dynamic_cache_cls.from_legacy_cache = from_legacy_cache

    if not getattr(PreTrainedModel, "_gptqmodel_legacy_tied_weights_patch", False) and hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
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

            if not getattr(self.config, "tie_word_embeddings", False):
                return {}

            return _resolve_legacy_tied_weights_mapping(self, tied_mapping)

        PreTrainedModel.get_expanded_tied_weights_keys = get_expanded_tied_weights_keys
        PreTrainedModel._gptqmodel_legacy_tied_weights_patch = True

    if not getattr(PreTrainedModel, "_gptqmodel_missing_all_tied_weights_patch", False):
        original_getattr = PreTrainedModel.__getattr__

        def __getattr__(self, name: str):
            if name == "all_tied_weights_keys":
                # Older remote-code models may skip `post_init()`, so lazily
                # synthesize the tied-weight map the first time HF asks for it.
                tied_keys = self.get_expanded_tied_weights_keys(all_submodels=True)
                object.__setattr__(self, name, tied_keys)
                return tied_keys

            return original_getattr(self, name)

        PreTrainedModel.__getattr__ = __getattr__
        PreTrainedModel._gptqmodel_missing_all_tied_weights_patch = True


def _normalize_chatglm_remote_code_config_compat(config: Any) -> None:
    if getattr(config, "model_type", None) != "chatglm":
        return

    if not hasattr(config, "seq_length") or hasattr(config, "max_length"):
        return

    # Older ChatGLM remote model code still reads `config.max_length`, while
    # newer transformers only preserves the serialized `seq_length` field.
    config.attribute_map = dict(getattr(config, "attribute_map", {}) or {})
    config.attribute_map["max_length"] = "seq_length"


def _normalize_rope_parameters_config_compat(config: Any) -> None:
    rope_parameters = getattr(config, "rope_parameters", None)
    if (
        isinstance(rope_parameters, dict)
        and rope_parameters.get("rope_type") is not None
        and rope_parameters.get("rope_theta") is not None
    ):
        return

    convert_rope_params = getattr(config, "convert_rope_params_to_dict", None)
    if callable(convert_rope_params):
        try:
            convert_rope_params()
        except Exception as exc:
            log.debug("Config: RoPE conversion fallback for %s failed: %s", type(config).__name__, exc)
        else:
            rope_parameters = getattr(config, "rope_parameters", None)
            if (
                isinstance(rope_parameters, dict)
                and rope_parameters.get("rope_type") is not None
                and rope_parameters.get("rope_theta") is not None
            ):
                return

    legacy_rope_scaling = getattr(config, "rope_scaling", None)
    rope_parameters = dict(legacy_rope_scaling) if isinstance(legacy_rope_scaling, dict) else dict(rope_parameters or {})

    if not rope_parameters and getattr(config, "rope_theta", None) is None and getattr(config, "default_theta", None) is None:
        return

    rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
    if rope_parameters.get("rope_theta") is None:
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "default_theta", 10_000.0)
        rope_parameters["rope_theta"] = rope_theta

    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor is not None:
        rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)

    if rope_parameters["rope_type"] in {"llama3", "yarn", "longrope"}:
        original_max_position_embeddings = getattr(config, "original_max_position_embeddings", None)
        if original_max_position_embeddings is None:
            original_max_position_embeddings = getattr(config, "max_position_embeddings", None)
        if original_max_position_embeddings is not None:
            rope_parameters.setdefault("original_max_position_embeddings", original_max_position_embeddings)

    config.rope_parameters = rope_parameters


# Restore config fields renamed by transformers 5.x before older trust_remote_code
# model files instantiate their architectures from the config object.
def _normalize_remote_code_config_compat(config: Any) -> None:
    _normalize_chatglm_remote_code_config_compat(config)

    # transformers 5.x normalizes RoPE config to `rope_type`, but older
    # MiniCPM remote code still reads `rope_scaling["type"]` or expects `None`.
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return

    if rope_scaling.get("rope_type") == "default" and set(rope_scaling).issubset({"rope_type", "rope_theta"}):
        # transformers 5.x materializes default RoPE metadata into
        # `rope_scaling`, but older remote MiniCPM code treats any dict here
        # as a scaled-RoPE config and expects an explicit `factor`.
        config.rope_scaling = None
        return

    if "type" in rope_scaling:
        return

    rope_type = rope_scaling.get("rope_type")
    if rope_type is None:
        return

    rope_scaling = dict(rope_scaling)
    rope_scaling["type"] = rope_type
    config.rope_scaling = rope_scaling


def normalize_hf_config_compat(config: Any, *, trust_remote_code: bool = False) -> None:
    # Some transformers 5.x model classes now read `config.rope_parameters`
    # directly during `from_config()`, but older local configs may only carry
    # legacy RoPE fields or nothing but a default `rope_theta`.
    _normalize_rope_parameters_config_compat(config)

    if not trust_remote_code:
        return

    _patch_transformers_remote_code_compat()
    _normalize_remote_code_config_compat(config)
    # Some config classes synchronize `rope_scaling` and `rope_parameters`, so
    # remote-code normalization that clears legacy default `rope_scaling` can
    # also reset `rope_parameters` back to None. Re-apply the RoPE backfill
    # after remote-code field cleanup so from_config() sees stable metadata.
    _normalize_rope_parameters_config_compat(config)


def prepare_remote_code_compat(config: Any) -> None:
    # Remote-code loads need both the transformers API shims and any config
    # field migrations applied before instantiation happens.
    normalize_hf_config_compat(config, trust_remote_code=True)


def prepare_remote_model_init_compat(model_id_or_path: Optional[str], config: Any) -> None:
    if not model_id_or_path or getattr(config, "model_type", None) != "phi4mm":
        return

    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForCausalLM")
    if not isinstance(class_ref, str):
        return

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_cls = get_class_from_dynamic_module(class_ref, str(model_id_or_path))
    except Exception as exc:
        log.debug("HF: remote model init compat pre-import failed for `%s`: %s", model_id_or_path, exc)
        return

    module_root = model_cls.__module__.rsplit(".", maxsplit=1)[0]
    speech_module = sys.modules.get(f"{module_root}.speech_conformer_encoder")
    remote_module = sys.modules.get(model_cls.__module__)
    outer_model_cls = model_cls if isinstance(model_cls, type) else None
    input_mode_enum = getattr(remote_module, "InputMode", None) if remote_module is not None else None

    if speech_module is not None and not getattr(speech_module, "_gptqmodel_scalar_tensor_meta_patch", False):
        speech_torch = getattr(speech_module, "torch", None)
        original_tensor = getattr(speech_torch, "tensor", None)
        if speech_torch is not None and original_tensor is not None:
            def _is_phi4mm_subsampling_scalar_init() -> bool:
                for frame_info in inspect.stack(context=0):
                    if frame_info.filename.endswith("speech_conformer_encoder.py") and frame_info.lineno == 1426:
                        return True
                return False

            # Phi-4 MM remote audio init creates scalar tensors only to derive Python
            # output sizes in NemoConvSubsampling; forcing just that scalar tensor onto
            # CPU keeps meta init safe without perturbing other meta-only buffers.
            def tensor_compat(data, *args, **kwargs):
                current_device = getattr(torch.utils._device, "CURRENT_DEVICE", None)
                if (
                    kwargs.get("device") is None
                    and current_device == torch.device("meta")
                    and isinstance(data, (int, float, bool))
                    and _is_phi4mm_subsampling_scalar_init()
                ):
                    kwargs = dict(kwargs)
                    kwargs["device"] = "cpu"
                return original_tensor(data, *args, **kwargs)

            speech_torch.tensor = tensor_compat

        positional_encoding_cls = getattr(speech_module, "AbsolutePositionalEncoding", None)
        if positional_encoding_cls is not None and not getattr(positional_encoding_cls, "_gptqmodel_meta_extend_patch", False):
            original_extend_pe = positional_encoding_cls.extend_pe

            def _is_phi4mm_positional_seed_call() -> bool:
                for frame_info in inspect.stack(context=0):
                    if frame_info.filename.endswith("speech_conformer_encoder.py") and frame_info.lineno == 895:
                        return True
                return False

            # The remote implementation seeds extend_pe() with a CPU scalar tensor.
            # Under meta init, promote that seed tensor back to meta before the
            # original method allocates its positional buffer.
            def extend_pe_compat(self, x):
                if isinstance(x, torch.Tensor) and x.device.type != "meta" and _is_phi4mm_positional_seed_call():
                    x = x.to(device="meta")
                return original_extend_pe(self, x)

            positional_encoding_cls.extend_pe = extend_pe_compat
            positional_encoding_cls._gptqmodel_meta_extend_patch = True

        speech_module._gptqmodel_scalar_tensor_meta_patch = True

    if (
        outer_model_cls is not None
        and hasattr(outer_model_cls, "forward")
        and not getattr(outer_model_cls, "_gptqmodel_input_mode_patch", False)
    ):
        original_forward = outer_model_cls.forward

        # Text-only callers like lm_eval do not pass `input_mode`; infer the
        # correct Phi-4 MM mode from the provided modality tensors instead.
        def forward_compat(self, *args, **kwargs):
            if kwargs.get("input_mode") is None:
                kwargs = dict(kwargs)
                has_vision = any(
                    kwargs.get(name) is not None
                    for name in ("input_image_embeds", "image_sizes", "image_attention_mask")
                )
                has_audio = any(
                    kwargs.get(name) is not None
                    for name in ("input_audio_embeds", "audio_embed_sizes", "audio_attention_mask")
                )

                if has_vision and has_audio:
                    kwargs["input_mode"] = input_mode_enum.VISION_SPEECH if input_mode_enum is not None else 3
                elif has_vision:
                    kwargs["input_mode"] = input_mode_enum.VISION if input_mode_enum is not None else 1
                elif has_audio:
                    kwargs["input_mode"] = input_mode_enum.SPEECH if input_mode_enum is not None else 2
                else:
                    kwargs["input_mode"] = input_mode_enum.LANGUAGE if input_mode_enum is not None else 0
            return original_forward(self, *args, **kwargs)

        outer_model_cls.forward = forward_compat
        outer_model_cls._gptqmodel_input_mode_patch = True

    inner_model_cls = getattr(remote_module, "Phi4MMModel", None) if remote_module is not None else None
    if inner_model_cls is not None and not hasattr(inner_model_cls, "prepare_inputs_for_generation"):
        # PEFT expects the inner model it wraps to expose this hook, even
        # though Phi-4 MM only defines the full implementation on the outer
        # CausalLM class.
        def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None, inputs_embeds=None, **kwargs):
            model_inputs = dict(kwargs)
            if inputs_embeds is not None and past_key_values is None:
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs["input_ids"] = input_ids
            model_inputs["past_key_values"] = past_key_values
            return model_inputs

        inner_model_cls.prepare_inputs_for_generation = prepare_inputs_for_generation

    try:
        import importlib.util
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.awq as peft_awq
    except Exception:
        peft_awq = None
    else:
        if not getattr(peft_awq, "_gptqmodel_awq_probe_patch", False):
            # PEFT later imports `awq.modules.linear`, so the availability
            # probe must require that concrete submodule instead of top-level
            # namespace packages that are missing the actual runtime.
            @lru_cache(maxsize=None)
            def is_auto_awq_available() -> bool:
                try:
                    return importlib.util.find_spec("awq.modules.linear") is not None
                except ModuleNotFoundError:
                    return False

            peft_import_utils.is_auto_awq_available = is_auto_awq_available
            peft_awq.is_auto_awq_available = is_auto_awq_available
            peft_awq._gptqmodel_awq_probe_patch = True


def load_tokenizer(tokenizer_or_path, *, model_config: Any = None, **kwargs):
    from tokenicer import Tokenicer

    warnings.warn(
        "gptqmodel.utils.hf.load_tokenizer() is deprecated; use Tokenicer.load(..., model_config=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Tokenicer.load(tokenizer_or_path, model_config=model_config, **kwargs)


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

    normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)

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
