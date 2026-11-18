# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
import csv
import json
import os
import re
import shutil
from contextlib import nullcontext
from os.path import isfile, join
from typing import Any, Dict, List, Optional, Union

import pcre
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from .._quant_methods import (
    REFUSED_HF_QUANT_METHODS,
    refused_hf_load_remedy,
)
from ..adapter.adapter import HF_ADAPTER_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora
from ..adapter.peft import LoraConfig
from ..nn_modules.qlinear import BaseQuantLinear, GPTQQuantLinear
from ..nn_modules.qlinear._fused_train_state import QZEROS_FORMAT_BUFFER
from ..quantization.config import (
    FORMAT,
    GGUF_EXPORT_QUANT_METHOD,
    META_FIELD_ACT_GROUP_AWARE,
    META_FIELD_DAMP_AUTO_INCREMENT,
    META_FIELD_DAMP_PERCENT,
    META_FIELD_FOEM_ENABLED,
    META_FIELD_GPTAQ_ENABLED,
    META_FIELD_MSE,
    META_FIELD_QUANTIZER,
    META_FIELD_STATIC_GROUPS,
    META_FIELD_TRUE_SEQUENTIAL,
    META_FIELD_URI,
    META_QUANTIZER_GPTQMODEL,
    META_VALUE_URI,
    METHOD,
    MIN_VERSION_WITH_V2,
    resolve_quant_format,
)
from ..utils.backend import BACKEND
from ..utils.checkpoint_paths import (
    CheckpointPathError,
    assert_auxiliary_copy_allowed,
    resolve_checkpoint_path,
    validate_relative_checkpoint_path,
)
from ..utils.checkpoint_state import checkpoint_serialization
from ..utils.exl3_error_ledger import write_exl3_error_ledger
from ..utils.exllamav3 import build_exllamav3_tensor_storage
from ..utils.hf import (
    get_hf_config_dtype,
    load_model_config,
    resolve_trust_remote_code,
    sanitize_generation_config_file,
    suspend_hf_weight_init,
)
from ..utils.logger import setup_logger
from ..utils.model import (
    OffloadTensorRef,
    TensorSource,
    _atomic_shard_path,
    _checkpoint_tensor_keys,
    _converted_safetensors_names,
    _validate_offload_sources_outside_save_dir,
    convert_loaded_qzeros_for_format_change,
    copy_py_files,
    find_modules,
    get_model_files_size,
    get_module_by_name,
    get_module_name,
    get_state_dict_for_save,
    is_embeddings_module_quantized,
    load_checkpoint_in_model_then_tie_weights,
    make_quant,
    resolve_resave_format_conversion,
    streaming_state_dict_to_shards,
)
from ..utils.structure import alias_all_from_turtle_if_meta, alias_from_turtle_for_submodule
from ..utils.tied_weights import (
    lm_head_is_tied,
    retie_tied_weights,
    word_embeddings_tied,
)
from ..utils.torch import torch_empty_cache
from ..version import __version__
from ._const import DEFAULT_MAX_SHARD_SIZE, DEVICE, EXPERT_INDEX_PLACEHOLDER
from .ports.native.auto import get_tokenizer_config


log = setup_logger()

PROCESS_LOG_NAME = "process"
PROCESS_LOG_LAYER = "layer"
PROCESS_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_LOSS_KIND = "loss_kind"
QUANT_LOG_NSAMPLES = "samples"
QUANT_LOG_DAMP = "damp"
PROCESS_LOG_TIME = "time"
PROCESS_LOG_FWD_TIME = "fwd_time"
PROCESS_USED_MEMORY = "(v)ram"

EORA_DEFAULT_FILE = "eora.safetensors"


_QUANTIZED_TENSOR_SUFFIXES = (
    "qweight",
    "qzeros",
    "scales",
    "g_idx",
    "qscale",
    "qzero",
    "weight_scale_inv",
    "weight_scale",
    "weight_bias",
    "weight_scb",
    "suh",
    "svh",
    "s_group",
    "s_channel",
    "theta",
    "channel_scales",
    "BH",
    "pre_scale",
)


def _export_shell_key_converter(qmodel, model):
    """Return the checkpoint-to-runtime key converter for the export shell.

    The shell is built from config, so it carries the runtime (Transformers)
    module names while the checkpoint may use a legacy layout (for example the
    Transformers 4.x ``language_model.model.layers.*`` keys of a Gemma-3 VL
    checkpoint). Accelerate matches tensor names verbatim, so the fill has to
    convert them the same way the native loader does (issue #4802).
    """

    from .loader import _checkpoint_key_converter

    return _checkpoint_key_converter(type(qmodel), model, to_runtime=True)


def _validate_export_shell_quantized_slots(model, checkpoint, key_converter=None):
    """Fail closed when the export shell has no slot for checkpoint quant tensors.

    The shell is rebuilt from config and filled from the checkpoint. A
    quantized checkpoint tensor without a destination module means the
    shell's module tree does not match the checkpoint (the fused-expert MoE
    case before #4589). Accelerate would ignore it and ``save()`` would
    publish a checkpoint missing that payload, so raise instead of writing.

    ``key_converter`` maps checkpoint names to the shell's runtime names. It is
    required for legacy layouts: a Gemma-3 VL checkpoint stores
    ``language_model.model.layers.*`` while the shell exposes
    ``model.language_model.layers.*`` (issue #4802).
    """

    checkpoint_keys = _checkpoint_tensor_keys(checkpoint, verify_shards=True)
    if checkpoint_keys is None:
        return
    if callable(key_converter):
        checkpoint_keys = {name for key in checkpoint_keys for name in _converted_safetensors_names(key, key_converter)}

    shell_keys = set(model.state_dict().keys())
    dropped = sorted(
        key
        for key in checkpoint_keys
        if key.rsplit(".", 1)[-1] in _QUANTIZED_TENSOR_SUFFIXES
        and key not in shell_keys
    )
    if not dropped:
        return

    examples = ", ".join(dropped[:8])
    raise ValueError(
        "Refusing to re-save a quantized checkpoint whose export shell has no "
        f"module for {len(dropped)} quantized tensor(s): {examples}. "
        "The rebuilt module tree does not match the checkpoint (for fused-expert "
        "MoE this means the expert shell was not defused). Saving would silently "
        "drop those tensors and write an unusable checkpoint."
    )


def _convert_checkpoint_state_dict_keys(state_dict: Dict[str, Any], key_converter) -> Dict[str, Any]:
    converted_state_dict = {}
    for name, tensor in state_dict.items():
        converted_name = key_converter(name)
        if converted_name in converted_state_dict:
            raise ValueError(f"Checkpoint key conversion produced duplicate tensor name: {converted_name}")
        if isinstance(tensor, TensorSource):
            tensor.name = converted_name
        converted_state_dict[converted_name] = tensor
    return converted_state_dict

# disable gptqmodel split_by layer feature (until sglang pr is merged since our dir struct is not compatible)
# SUPPORTED_SPLIT_BY = {None, "layer"}
SUPPORTED_SPLIT_BY = {None}
_MAX_SHARD_SIZE_RE = pcre.compile(
    r"\s*(\d+)\s*([KMGTP]?i?B?)\s*",
    flags=pcre.Flag.CASELESS,
)


def _parse_split_by(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("split_by must be a string or None.")

    normalized = value.strip().lower()
    if normalized in ("", "none"):
        return None
    if normalized not in SUPPORTED_SPLIT_BY:
        raise ValueError(f"Unsupported split_by value: {value}. Supported values: None, 'layer'.")
    return normalized


def _parse_max_shard_size(value: Optional[Union[int, str]]) -> Optional[int]:
    """Resolve a ``max_shard_size`` argument to bytes.

    Module level so every export path -- including formats that publish their
    own packed state -- honours the same shard limit. Units follow Hugging
    Face's convention: ``KB``/``MB``/``GB`` are decimal and
    ``KiB``/``MiB``/``GiB`` are binary. A bare ``K``/``M``/``G`` keeps the
    historical binary meaning so existing callers do not change size.
    """

    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = _MAX_SHARD_SIZE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid max_shard_size value: {value}")
    base = int(match.group(1))
    suffix = match.group(2).upper()
    if suffix.startswith("I"):
        raise ValueError(f"Invalid max_shard_size value: {value}")
    exponent = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}.get(suffix[:1], 0)
    if exponent == 0:
        return base
    binary = suffix.endswith("IB") or not suffix.endswith("B")
    multiplier = (1024 if binary else 1000) ** exponent
    return base * multiplier


def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Stringify caller-supplied safetensors header metadata."""

    if meta is None:
        return {}
    if not isinstance(meta, dict):
        raise TypeError("safetensors_metadata must be a dictionary.")
    normalized: Dict[str, str] = {}
    for key, value in meta.items():
        try:
            new_key = str(key)
            new_value = str(value)
        except Exception as exc:
            raise TypeError(
                f"safetensors_metadata: both keys and values must be strings and conversion failed for ({key}, {value}): {exc}"
            )
        if new_key in normalized:
            log.warn(
                f"Duplicate metadata key '{new_key}' after conversion to string; overwriting previous value."
            )
        normalized[new_key] = new_value
    return normalized


# Source-only config fields are preserved, but a live value always wins. These
# keys are excluded because they are either stripped on purpose or are fully
# regenerated for the export.
_CONFIG_MERGE_SKIP_KEYS = frozenset(
    {
        "attn_implementation",
        "_attn_implementation",
        "quantization_config",
        "quantize_config",
    }
)


def _merge_missing_config_fields(saved: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Fill keys the saved config dropped, recursively, without overriding live values."""
    for key, value in source.items():
        if key in _CONFIG_MERGE_SKIP_KEYS:
            continue
        if key not in saved:
            saved[key] = value
        elif isinstance(saved[key], dict) and isinstance(value, dict):
            _merge_missing_config_fields(saved[key], value)


_PER_EXPERT_MODULE_PATTERN = re.compile(r"(?:^|\.)experts\.\d+\.")


def _has_per_expert_quantized_modules(model: Optional[torch.nn.Module]) -> bool:
    """Return whether ``model`` stores its experts as per-expert quantized linears."""

    if model is None:
        return False
    for name, module in model.named_modules():
        if getattr(module, "QUANT_TYPE", None) is None:
            continue
        if _PER_EXPERT_MODULE_PATTERN.search(name):
            return True
    return False


def hf_compat_warnings(
    quantize_config: Any,
    runtime_format: FORMAT,
    model: Optional[torch.nn.Module] = None,
) -> List[str]:
    """Return the ways this checkpoint is unsafe for a plain HF loader.

    GPT-QModel can serialize more than `transformers.AutoModelForCausalLM`
    consumes, and both gaps below are silent: HF random-initializes the missing
    `lm_head.weight` and builds uninitialized AWQ layers for excluded modules.
    Reporting them at save time is the only place the information still exists.
    """

    del runtime_format  # the caveats are format-independent
    warnings: List[str] = []
    if bool(getattr(quantize_config, "lm_head", False)):
        warnings.append(
            "`lm_head=True` stores the output head as quantized `lm_head.qweight` "
            "and drops `lm_head.weight`, so `AutoModelForCausalLM.from_pretrained` "
            "random-initializes the head and returns garbage logits. Load this "
            "checkpoint with `GPTQModel.load` or vLLM."
        )
    if (
        getattr(quantize_config, "method", None) == METHOD.AWQ
        and bool(getattr(quantize_config, "dynamic", None))
    ):
        warnings.append(
            "This AWQ checkpoint carries `dynamic` per-module overrides "
            "(`bits`/`group_size`). A loader that does not install GPT-QModel's "
            "native AWQ bridge builds every module from the global `bits` and "
            "`group_size`, so the overrides are ignored: on CPU the checkpoint "
            "loads with silently wrong logits and on CUDA the Marlin pack raises "
            "a qzeros shape error. Transformers' own AWQ quantizer only reaches "
            "the bridge after it has already replaced the layers, so an older "
            "Transformers or a third-party loader still needs `import gptqmodel` "
            "before loading. Load this checkpoint with `GPTQModel.load` (or a "
            "Transformers version with the native AWQ bridge) to honor the "
            "overrides."
        )
    if (
        getattr(quantize_config, "method", None) == METHOD.AWQ
        and getattr(quantize_config, "bits", None) == 8
    ):
        warnings.append(
            "8-bit AWQ is packed in GPT-QModel's natural lane order (lane i is "
            "output column i). vLLM `awq_marlin` and AutoAWQ undo a `[0, 2, 1, 3]` "
            "interleave, so they decode these weights and zero points in the "
            "wrong order and lose accuracy silently. Load 8-bit AWQ with "
            "`GPTQModel.load`, or export 4-bit AWQ for vLLM."
        )
    if getattr(quantize_config, "method", None) == METHOD.GGUF:
        warnings.append(
            "Packed GGUF checkpoints are not readable by `transformers`' "
            "`GgufHfQuantizer`, which owns `quant_method: \"gguf\"` and expects a "
            "`.gguf` file. This checkpoint publishes "
            f"`quant_method: \"{GGUF_EXPORT_QUANT_METHOD}\"` so an HF loader "
            "refuses it instead of random-initializing every linear layer. Load "
            "it with `GPTQModel.load`."
        )
    method = getattr(quantize_config, "method", None)
    method_name = getattr(method, "value", method)
    if method_name in REFUSED_HF_QUANT_METHODS:
        warnings.append(
            f"`quant_method: \"{method_name}\"` is a GPT-QModel-only layout that "
            "`transformers` cannot read: `AutoModelForCausalLM.from_pretrained` "
            "only warns `Unknown quantization type` and then random-initializes "
            "every linear layer. Importing `gptqmodel` installs a refusing "
            "quantizer, so plain HF raises instead. Load this checkpoint with "
            f"{refused_hf_load_remedy(method_name)}."
        )
    if _has_per_expert_quantized_modules(model):
        warnings.append(
            "This checkpoint stores per-expert quantized weights "
            "(`experts.<n>.*.qweight`). Transformers 5 builds fused 3-D expert "
            "tensors and its built-in GPTQ/AWQ loaders only replace `nn.Linear`, "
            "so a loader without the native bridge reports every expert tensor "
            "as MISSING/UNEXPECTED and silently random-initializes the experts "
            "(garbage output, no error). Import `gptqmodel` before Transformers "
            "to install the native loading bridge for both methods, or load this "
            "checkpoint with `GPTQModel.load`."
        )
    return warnings


def _quantization_config_for_model_config(
    quantize_config: Any,
    runtime_format: FORMAT,
) -> Dict[str, Any]:
    """Return the compact quantization descriptor embedded in config.json."""
    payload = quantize_config.to_dict()
    if runtime_format == FORMAT.EXL3:
        payload.pop("tensor_storage", None)
    return payload


def _gptq_explicit_quantized_modules(
    quantize_config: Any,
    model: torch.nn.Module,
) -> Optional[List[List[str]]]:
    """Return full module names Optimum must quantize for a GPTQ `-:` checkpoint.

    Optimum builds an uninitialized quantized layer for every module it decides
    to quantize and matches ``modules_in_block_to_quantize`` entries with
    ``name.endswith``. A GPTQ checkpoint that relies on dynamic ``-:``
    exclusions carries no exclusion list, so plain HF loads silently build
    Marlin layers for the excluded modules. Declaring the exact quantized set
    is the only lossless bridge. Optimum expects a list of groups and flattens
    it with ``sum(..., [])``, so the names are wrapped in one group. ``None``
    keeps the historical payload for checkpoints without exclusions.
    """

    if getattr(quantize_config, "method", None) != METHOD.GPTQ:
        return None
    if not quantize_config.has_dynamic_exclusions():
        return None
    return [sorted(find_modules(model, layers=[BaseQuantLinear]))]


def _save_model_configs_without_weights(
        model: torch.nn.Module,
        save_dir: str,
        source_model_dir: Optional[str] = None,
        trust_remote_code: bool = False,
) -> None:
    """Serialize HF model metadata without invoking weight-file cleanup.

    ``PreTrainedModel.save_pretrained(state_dict={})`` is not a config-only
    operation: Transformers treats the empty dictionary as the complete weight
    set and removes pre-existing checkpoint shards. Quantized exports write
    their weights separately below, so use the same metadata preparation as
    Transformers and call the config serializers directly.
    """

    os.makedirs(save_dir, exist_ok=True)
    config = model.config

    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        config.dtype = str(dtype).split(".")[-1]
    config.architectures = [model.__class__.__name__.removeprefix("FSDP")]

    is_remote_code = getattr(model, "is_remote_code", None)
    if callable(is_remote_code) and is_remote_code():
        custom_object_save(model, save_dir, config=config)

    # Read the source config before serializing: an in-place export overwrites
    # ``config.json`` and would otherwise lose the fields we are trying to keep.
    source_config: Optional[Dict[str, Any]] = None
    if source_model_dir is not None:
        source_config_path = os.path.join(source_model_dir, "config.json")
        if os.path.isfile(source_config_path):
            with open(source_config_path, "r", encoding="utf-8") as handle:
                source_config = json.load(handle)
            if not isinstance(source_config, dict):
                raise ValueError("Model config.json must contain a JSON object.")

    config.save_pretrained(save_dir)
    saved_config_path = os.path.join(save_dir, "config.json")

    # Some remote model configurations contain runtime-significant extension
    # fields that their registered Transformers config class does not declare.
    # Loading and reserializing such a config silently drops those fields (for
    # example DeepSeek V4's ``num_hash_layers``). Preserve source-only fields,
    # including ones nested in sub-configs such as ``text_config``, while
    # allowing the live serialized config—including the new
    # quantization_config—to override every field it knows about.
    saved_config: Optional[Dict[str, Any]] = None
    if source_config is not None:
        with open(saved_config_path, "r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
        if not isinstance(saved_config, dict):
            raise ValueError("Model config.json must contain a JSON object.")
        _merge_missing_config_fields(saved_config, source_config)

    # A natively supported model loaded with the `trust_remote_code` override is
    # saved without its `.py` files. Keeping the source `auto_map` would then
    # point at files that are not in the export and make
    # `AutoConfig.from_pretrained(out, trust_remote_code=True)` fail with a
    # missing `configuration_*.py`. Drop it whenever remote code is not copied.
    if not trust_remote_code:
        if saved_config is None:
            with open(saved_config_path, "r", encoding="utf-8") as handle:
                saved_config = json.load(handle)
            if not isinstance(saved_config, dict):
                raise ValueError("Model config.json must contain a JSON object.")
        saved_config.pop("auto_map", None)

    if saved_config is not None:
        with open(saved_config_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(saved_config, indent=2, sort_keys=True) + "\n")

    can_generate = getattr(model, "can_generate", None)
    generation_config = getattr(model, "generation_config", None)
    if callable(can_generate) and can_generate() and generation_config is not None:
        _save_generation_config(generation_config, save_dir)


def _save_generation_config(generation_config, save_dir: str) -> None:
    """Save a generation config without rewriting its declared ``do_sample``.

    Transformers >= 5 refuses to save a config whose sampling-only fields
    contradict a non-sampling ``do_sample``. GPT-QModel preserves the
    checkpoint's declared value, so fall back to the same serializer without
    that strict check instead of flipping ``do_sample`` on.
    """
    try:
        generation_config.save_pretrained(save_dir)
        return
    except ValueError as exc:
        if "Fix these issues to save the configuration." not in str(exc):
            raise

    to_json_file = getattr(generation_config, "to_json_file", None)
    if not callable(to_json_file):
        raise
    to_json_file(
        os.path.join(save_dir, "generation_config.json"),
        use_diff=True,
        keys_to_pop=["compile_config"],
    )


def _quantized_module_types(model: torch.nn.Module) -> tuple[type, ...]:
    """Return the module classes in `model` that own a packed quantization payload.

    LazyTurtle's save-time materialization resolves tensors by checkpoint name.
    A quantized module stores its packed codes in a buffer whose leaf name can
    match the source checkpoint tensor (`weight`), so aliasing it from the
    source would replace the packed payload with the original dense weight.
    The on-disk offload bundle written from `module.state_dict()` is
    authoritative for those modules, and `get_state_dict_for_save` restores
    them from there.
    """

    return tuple(
        sorted(
            {
                type(module)
                for module in model.modules()
                if getattr(module, "QUANT_TYPE", None) is not None
            },
            key=lambda cls: (cls.__module__, cls.__qualname__),
        )
    )


def _materialize_remaining_meta_params_from_turtle(
    model: torch.nn.Module,
    turtle_model,
    *,
    skip_module_types: tuple[type, ...] = (),
) -> int:
    """Best-effort fallback for meta params that survive normal turtle sync."""

    if (
            turtle_model is None
            or not hasattr(turtle_model, "_resolve_checkpoint_tensor_source")
            or not hasattr(turtle_model, "_weight_map")
            or not hasattr(turtle_model, "model_local_path")
    ):
        return 0

    restored = 0
    pending_by_shard: Dict[str, List[tuple[str, str, str, torch.nn.Parameter, Optional[int], Optional[int], Optional[int]]]] = {}

    for full_name, param in list(model.named_parameters()):
        if not (getattr(param, "is_meta", False) or param.device.type == "meta"):
            continue

        module_path, leaf = full_name.rsplit(".", 1)
        if skip_module_types and isinstance(model.get_submodule(module_path), skip_module_types):
            continue
        resolved_name, expert_index, split_index, split_dim = turtle_model._resolve_checkpoint_tensor_source(module_path, leaf)
        if resolved_name is None:
            continue
        shard = turtle_model._weight_map.get(resolved_name)
        if shard is None:
            continue
        pending_by_shard.setdefault(shard, []).append(
            (resolved_name, module_path, leaf, param, expert_index, split_index, split_dim)
        )

    for shard, entries in pending_by_shard.items():
        shard_path = os.path.join(turtle_model.model_local_path, shard)
        unique_names = {name for name, _module_path, _leaf, _param, _expert_index, _split_index, _split_dim in entries}

        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handler:
                tensors = {name: handler.get_tensor(name) for name in unique_names}
        except RuntimeError as exc:
            log.warn("Model save: skipping shard `%s` during meta materialization due to runtime error: %s", shard, exc)
            continue

        for tensor_name, module_path, leaf, param, expert_index, split_index, split_dim in entries:
            source = tensors.get(tensor_name)
            if source is None:
                continue
            target = source
            if expert_index is not None:
                if expert_index >= target.shape[0]:
                    continue
                target = target.narrow(0, expert_index, 1).squeeze(0)
            if split_index is not None and split_dim is not None:
                if target.shape[split_dim] % 2 != 0:
                    continue
                chunk = target.shape[split_dim] // 2
                target = target.narrow(split_dim, split_index * chunk, chunk)
            if target.dtype != param.dtype:
                target = target.to(dtype=param.dtype)
            if tuple(target.shape) != tuple(param.shape):
                continue
            module = model.get_submodule(module_path)
            replacement = torch.nn.Parameter(target.detach().clone(), requires_grad=param.requires_grad)
            setattr(module, leaf, replacement)
            restored += 1

    return restored


def _materialize_meta_layers_from_turtle(
    model: torch.nn.Module,
    turtle_model,
    *,
    skip_module_types: tuple[type, ...] = (),
) -> int:
    if turtle_model is None or not hasattr(turtle_model, "materialize_submodule"):
        return 0

    layer_paths = set()
    for full_name, param in model.named_parameters():
        if not (getattr(param, "is_meta", False) or param.device.type == "meta"):
            continue
        parts = full_name.split(".")
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 < len(parts):
                layer_paths.add(".".join(parts[: i + 2]))

    materialized = 0
    for path in sorted(layer_paths):
        try:
            submodule = model.get_submodule(path)
            alias_from_turtle_for_submodule(
                target_model=model,
                turtle_model=turtle_model,
                target_submodule=submodule,
                device=torch.device("cpu"),
                non_blocking=False,
                skip_module_types=skip_module_types,
            )
            materialized += 1
        except Exception as exc:
            log.warn("Model save: failed to materialize meta layer `%s` from turtle: %s", path, exc)

    return materialized


def _checkpoint_prefixes_for_replacement(prefix: str, turtle_model) -> set[str]:
    prefixes = {prefix}
    resolver = getattr(turtle_model, "_resolve_checkpoint_module_path", None)
    if callable(resolver):
        resolved = resolver(prefix)
        if resolved:
            prefixes.add(resolved)
    tensor_resolver = getattr(turtle_model, "_resolve_checkpoint_tensor_source", None)
    if callable(tensor_resolver):
        for leaf in ("weight", "bias"):
            try:
                checkpoint_name, _expert_index, _split_index, _split_dim = tensor_resolver(prefix, leaf)
            except Exception:
                checkpoint_name = None
            if checkpoint_name:
                prefixes.add(checkpoint_name[: -(len(leaf) + 1)] if checkpoint_name.endswith(f".{leaf}") else checkpoint_name)
    return prefixes


def _tensor_matches_prefixes(tensor_name: str, prefixes: set[str]) -> bool:
    return tensor_name in prefixes or any(tensor_name.startswith(f"{prefix}.") for prefix in prefixes)


# Buffers owned by AUTO_NF12 source storage. They are calibration scratch, not a
# checkpoint layout: a saved model must carry the dense `weight` instead.
_NF12_PAYLOAD_BUFFERS = (
    "nf12_packed",
    "nf12_escapes",
    "nf12_exp_base",
    "nf12_row_packed",
    "nf12_row_escapes",
)


def _module_declares_nf12_checkpoint(quantize_config, module_name: str) -> bool:
    """Report whether `module_name` is an NF12-quantized module for this config."""

    if quantize_config is None:
        return False
    resolver = getattr(quantize_config, "resolve_module_config", None)
    resolved = resolver(module_name) if callable(resolver) else None
    if resolved is not None and resolved.excluded:
        return False
    module_format = getattr(quantize_config, "format", None)
    if resolved is not None:
        module_format = resolved.overrides.get("format", module_format)
    return (
        resolve_quant_format(module_format, getattr(quantize_config, "method", None))
        == FORMAT.NF12
    )


def _validate_nf12_payload_ownership(owner, *, prefixes: Optional[List[str]] = None) -> None:
    """Refuse to serialize AUTO_NF12 source buffers under unquantized modules.

    `auto_dtype_cast` packs a dense source into `nf12_*` buffers for the length
    of one calibration forward and restores the exact source before the
    quantizer consumes it. A packed module that survives into `save()` is
    silent corruption: the writer emits `nf12_*` tensors instead of `weight`
    while `quantize_config.json` still describes the module as plain
    unquantized, and the loader leaves the dense weight uninitialised (NaN
    logits). Fail closed instead of publishing that checkpoint.
    """

    model = getattr(owner, "model", None)
    if model is None:
        return
    quantize_config = getattr(owner, "quantize_config", None)
    scope = set(prefixes) if prefixes else None
    offenders = []
    for module_name, module in model.named_modules():
        buffers = getattr(module, "_buffers", None)
        if not buffers or not any(name in buffers for name in _NF12_PAYLOAD_BUFFERS):
            continue
        if scope is not None and not _tensor_matches_prefixes(module_name, scope):
            continue
        if _module_declares_nf12_checkpoint(quantize_config, module_name):
            continue
        offenders.append(module_name or "<root>")
    if not offenders:
        return
    sample = ", ".join(sorted(offenders)[:8])
    if len(offenders) > 8:
        sample += f", ... ({len(offenders)} total)"
    raise ValueError(
        "Save aborted: AUTO_NF12 source storage is still installed on modules the "
        f"quantize config does not declare as NF12 (`{sample}`). Saving would write "
        "`nf12_*` buffers in place of `weight` for those modules, and reloading the "
        "checkpoint would leave their dense weights uninitialised (NaN logits). "
        "Restore the dense source module before saving."
    )


def _module_name(model, target_module) -> Optional[str]:
    if target_module is None:
        return None
    return next((name for name, module in model.named_modules() if module is target_module), None)


def _save_module_replacement_safetensors(
    model,
    turtle_model,
    replacement_prefixes: List[str],
    *,
    save_dir: str,
    metadata: Dict[str, str],
) -> tuple[List[str], Dict[str, str], int, List[str]]:
    """Rewrite only checkpoint shards containing replaced modules."""
    if turtle_model is None or not hasattr(turtle_model, "_weight_map") or not hasattr(turtle_model, "model_local_path"):
        raise ValueError("Module replacement save requires a LazyTurtle checkpoint source.")
    prefixes = sorted({prefix for prefix in replacement_prefixes if isinstance(prefix, str) and prefix})
    if not prefixes:
        raise ValueError("Module replacement save requires at least one module prefix.")

    config = getattr(model, "config", None)
    # The loaded model's snapshot already resolved tying from the decoder config
    # and the checkpoint (#4799); the config.json flag is only a fallback.
    resolved_tie = lm_head_is_tied(model)
    tie_word_embeddings = word_embeddings_tied(config)
    if resolved_tie is not None:
        tie_word_embeddings = resolved_tie
    elif not tie_word_embeddings:
        source_config_path = os.path.join(turtle_model.model_local_path, "config.json")
        if os.path.exists(source_config_path):
            try:
                with open(source_config_path, "r", encoding="utf-8") as handle:
                    tie_word_embeddings = bool(json.load(handle).get("tie_word_embeddings", False))
            except (OSError, TypeError, ValueError):
                pass
    input_prefix = _module_name(model, model.get_input_embeddings()) if hasattr(model, "get_input_embeddings") else None
    output_prefix = _module_name(model, model.get_output_embeddings()) if hasattr(model, "get_output_embeddings") else None

    # Untying happens before endpoint quantization. If only the input endpoint
    # was requested, the cloned output head must still be serialized because
    # the saved config is no longer tied.
    if (
        tie_word_embeddings
        and input_prefix
        and output_prefix
        and input_prefix in prefixes
        and output_prefix not in prefixes
    ):
        prefixes = sorted({*prefixes, output_prefix})

    index_name = getattr(turtle_model, "_index_filename", "model.safetensors.index.json")
    index_path = os.path.join(save_dir, index_name) if index_name else None
    existing_weight_map = {}
    if index_path and os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            existing_weight_map = json.load(handle).get("weight_map", {})

    drop_by_shard: Dict[str, set[str]] = {}
    replacements_by_shard: Dict[str, Dict[str, torch.Tensor]] = {}
    removed_tensor_names: List[str] = []
    matched_shards_by_prefix: Dict[str, List[str]] = {}
    for prefix in prefixes:
        checkpoint_prefixes = _checkpoint_prefixes_for_replacement(prefix, turtle_model)
        runtime_prefix = {prefix}
        matched_shards = []
        for tensor_name, shard_name in turtle_model._weight_map.items():
            if _tensor_matches_prefixes(tensor_name, checkpoint_prefixes):
                drop_by_shard.setdefault(shard_name, set()).add(tensor_name)
                if tensor_name not in removed_tensor_names:
                    removed_tensor_names.append(tensor_name)
                if shard_name not in matched_shards:
                    matched_shards.append(shard_name)
        for tensor_name, shard_name in existing_weight_map.items():
            if not _tensor_matches_prefixes(tensor_name, runtime_prefix):
                continue
            drop_by_shard.setdefault(shard_name, set()).add(tensor_name)
            if tensor_name not in removed_tensor_names:
                removed_tensor_names.append(tensor_name)
            if shard_name not in matched_shards:
                matched_shards.append(shard_name)
        matched_shards_by_prefix[prefix] = matched_shards

    tied_input_shards = list(matched_shards_by_prefix.get(input_prefix, [])) if input_prefix else []
    if tie_word_embeddings and input_prefix and not tied_input_shards:
        input_checkpoint_prefixes = _checkpoint_prefixes_for_replacement(input_prefix, turtle_model)
        for tensor_name, shard_name in turtle_model._weight_map.items():
            if _tensor_matches_prefixes(tensor_name, input_checkpoint_prefixes) and shard_name not in tied_input_shards:
                tied_input_shards.append(shard_name)

    for prefix in prefixes:
        matched_shards = matched_shards_by_prefix[prefix]
        if not matched_shards and tie_word_embeddings and prefix == output_prefix:
            matched_shards = tied_input_shards
        if not matched_shards:
            raise ValueError(f"Could not find checkpoint tensor for module `{prefix}`.")
        try:
            module = model.get_submodule(prefix)
        except AttributeError as exc:
            raise ValueError(f"Module replacement module `{prefix}` is not present in the model tree.") from exc
        replacements = replacements_by_shard.setdefault(matched_shards[0], {})
        drop_by_shard.setdefault(matched_shards[0], set())
        # This is a checkpoint write: a kernel that replaced its packed tensors
        # with a private runtime layout must fail closed here instead of
        # publishing words no loader can read. Read-only `state_dict()` callers
        # (PEFT) stay outside this context and still get the live tensors.
        with checkpoint_serialization():
            module_state = module.state_dict()
        for relative_name, tensor in module_state.items():
            replacements[f"{prefix}.{relative_name}" if relative_name else prefix] = tensor.detach().cpu()

    rewritten_files = []
    tensor_to_filename = {}
    total_size_bytes = 0
    runtime_prefixes = set(prefixes)
    for shard_name, dropped_names in drop_by_shard.items():
        if not shard_name.endswith(".safetensors"):
            raise NotImplementedError("Module replacement save currently supports safetensors checkpoints only.")
        # `shard_name` comes from the checkpoint's `weight_map`. Reject an
        # absolute or `..` entry before it is joined to the source or output
        # directory: the write must never leave `save_dir`.
        source_path = resolve_checkpoint_path(
            turtle_model.model_local_path,
            shard_name,
            kind="checkpoint weight_map shard",
            follow_symlinks=False,
        )
        target_path = resolve_checkpoint_path(
            save_dir, shard_name, kind="checkpoint weight_map shard"
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        read_path = target_path if os.path.exists(target_path) else source_path
        tensors = {}
        with safe_open(read_path, framework="pt", device="cpu") as handler:
            for tensor_name in handler.keys():
                if tensor_name in dropped_names or _tensor_matches_prefixes(tensor_name, runtime_prefixes):
                    if tensor_name not in removed_tensor_names:
                        removed_tensor_names.append(tensor_name)
                    continue
                tensors[tensor_name] = handler.get_tensor(tensor_name)
                tensor_to_filename[tensor_name] = shard_name
        for tensor_name, tensor in replacements_by_shard.get(shard_name, {}).items():
            tensors[tensor_name] = tensor
            tensor_to_filename[tensor_name] = shard_name
        with _atomic_shard_path(target_path) as temporary:
            save_file(tensors, temporary, metadata=metadata)
        rewritten_files.append(shard_name)
        total_size_bytes += os.path.getsize(target_path)
    return rewritten_files, tensor_to_filename, total_size_bytes, removed_tensor_names


def _copy_missing_checkpoint_files(source_dir: str, save_dir: str) -> List[str]:
    source_root = os.path.realpath(source_dir)
    save_root = os.path.realpath(save_dir)
    if source_root == save_root:
        return []
    if os.path.commonpath([source_root, save_root]) == source_root:
        raise ValueError("Module replacement save destination must not be nested inside its checkpoint source.")
    copied = []
    for root, dirnames, filenames in os.walk(source_root):
        dirnames.sort()
        relative_root = os.path.relpath(root, source_root)
        target_root = save_root if relative_root == "." else os.path.join(save_root, relative_root)
        for filename in sorted(filenames):
            target_path = os.path.join(target_root, filename)
            if os.path.exists(target_path):
                continue
            source_path = os.path.join(root, filename)
            assert_auxiliary_copy_allowed(source_root, source_path, kind="module replacement auxiliary file")
            os.makedirs(target_root, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied.append(os.path.relpath(target_path, save_root))
    return copied


def _cleanup_saved_weight_files(
    save_dir: str,
    expected_files: List[str],
    model_base_name: str,
    model_save_name: str,
) -> None:
    expected = set(expected_files)
    shard_pattern = pcre.compile(
        rf"{pcre.escape(model_base_name)}-\d{{5}}-of-\d{{5}}\.safetensors"
    )

    for filename in os.listdir(save_dir):
        full_filename = join(save_dir, filename)
        if not isfile(full_filename):
            continue
        if filename == model_save_name and filename not in expected:
            os.remove(full_filename)
            continue
        if filename == model_save_name + ".index.json" and filename not in expected:
            os.remove(full_filename)
            continue
        if shard_pattern.fullmatch(filename) and filename not in expected:
            os.remove(full_filename)



def _read_safetensors_weight_map(model_local_path: str) -> Optional[Dict[str, str]]:
    index_path = join(model_local_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    except Exception as exc:
        log.warn(f"Model: Failed to inspect original safetensors index at '{index_path}': {exc}")
        return None
    if not isinstance(weight_map, dict):
        return None
    return {
        name: filename
        for name, filename in weight_map.items()
        if isinstance(filename, str) and filename.endswith(".safetensors")
    }


def _resolve_out_of_model_source_files(
    model_local_path: str,
    source_files: Optional[List[str]] = None,
) -> List[str]:
    if source_files:
        return sorted(dict.fromkeys(source_files))

    weight_map = _read_safetensors_weight_map(model_local_path)
    if weight_map is not None:
        return sorted(set(weight_map.values()))

    return sorted(
        filename
        for filename in os.listdir(model_local_path)
        if filename.endswith(".safetensors") and isfile(join(model_local_path, filename))
    )


def _load_native_w4a_for_save(model: torch.nn.Module, checkpoint: str, *, activation_mode: str) -> None:
    """Load native checkpoint tensors exactly into a CPU serialization shell.

    Generic CPU GPTQ linears do not declare NVFP4 calibration buffers. Register
    those buffers for NVFP4 before loading. Both W4A modes assign tensors rather
    than casting checkpoint values to the shell's FP16 dtype, which can change
    or overflow BF16 tensors. Runtime weight caches and producer buffers remain
    absent from the native checkpoint.
    """
    if activation_mode == "w4a_nvfp4":
        for module in model.modules():
            if isinstance(module, BaseQuantLinear):
                module.register_buffer("activation_global_scale_bits", torch.ones((), dtype=torch.float32).view(torch.int32))
    if checkpoint.endswith(".index.json"):
        with open(checkpoint, encoding="utf-8") as handle:
            index = json.load(handle)
        files = [join(os.path.dirname(checkpoint), name) for name in sorted(set(index["weight_map"].values()))]
    elif checkpoint.endswith(".safetensors"):
        files = [checkpoint]
    else:
        raise ValueError("W4A export requires a safetensors file or shard index")
    # The qzeros layout marker is a persistent buffer so in-process state-dict
    # transports keep it, but a checkpoint write deliberately omits it. It is
    # therefore never present in a native W4A checkpoint and must not be part of
    # the completeness check (issue #4171 item 2).
    expected = {
        name for name in model.state_dict()
        if name != QZEROS_FORMAT_BUFFER
        and not name.endswith("." + QZEROS_FORMAT_BUFFER)
    }
    seen = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            values = {name: handle.get_tensor(name) for name in handle.keys()}
        if seen.intersection(values):
            raise ValueError("Duplicate native W4A tensor across checkpoint shards")
        result = model.load_state_dict(values, strict=False, assign=True)
        if result.unexpected_keys:
            raise ValueError(f"Unrecognized native W4A tensors: {result.unexpected_keys}")
        seen.update(values)
    retie_tied_weights(model)
    # A tied tensor can legitimately be omitted from safetensors. Every other
    # parameter/buffer must have been loaded, including the calibration bits.
    state = model.state_dict()
    def storage_key(value):
        return value.data_ptr(), value.dtype, tuple(value.shape)
    restored_storage = {storage_key(state[name]) for name in seen}
    missing = sorted(name for name in expected - seen if storage_key(state[name]) not in restored_storage)
    if missing:
        raise ValueError(f"Missing native W4A checkpoint tensors: {missing}")


def _load_tensors_by_selectors(
    model_local_path: str,
    prefixes: List[str],
    suffixes: Optional[List[str]] = None,
    tensor_names: Optional[List[str]] = None,
    source_files: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Gather source tensors selected by prefix, suffix, or exact name."""

    tensors: Dict[str, torch.Tensor] = {}
    suffixes = suffixes or []
    tensor_name_set = set(tensor_names or [])
    source_file_names = _resolve_out_of_model_source_files(model_local_path, source_files)

    def selected(name: str) -> bool:
        return (
            name in tensor_name_set
            or any(name.startswith(prefix) for prefix in prefixes)
            or any(name.endswith(suffix) for suffix in suffixes)
        )

    for source_file_name in source_file_names:
        source_tensor_path = os.path.join(model_local_path, source_file_name)
        if not os.path.exists(source_tensor_path):
            continue
        try:
            with safe_open(source_tensor_path, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    if selected(tensor_name):
                        if tensor_name not in tensors:
                            tensors[tensor_name] = f.get_tensor(tensor_name)
        except Exception as exc:
            log.warn(
                f"Model: Failed to read tensors from {source_file_name} while scanning "
                f"for out-of-model selectors: {exc}"
            )
    # Prefixes absent from the index stay optional, but tensors the index declares
    # must not be silently dropped when their shard is missing or unreadable.
    weight_map = None if source_files else _read_safetensors_weight_map(model_local_path)
    missing = sorted(
        name
        for name in weight_map or {}
        if selected(name) and name not in tensors
    )
    if missing:
        shards = sorted({weight_map[name] for name in missing})
        raise ValueError(
            f"Model: {len(missing)} tensors declared in the safetensors index of "
            f"`{model_local_path}` are missing from or unreadable in {shards}, e.g. {missing[:3]}."
        )
    return tensors


def _tensor_source_from_tensor(name: str, tensor: torch.Tensor) -> TensorSource:
    # Create a TensorSource wrapper so the merged tensor behaves like original state_dict entries.
    # Wrap a raw tensor into a TensorSource so it can be merged into state_dict.
    return TensorSource(
        name=name,
        torch_dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        source=tensor,
    )


def _merge_selected_tensors_into_state_dict(
    prefixes: List[str],
    suffixes: List[str],
    tensor_names: List[str],
    model_local_path: str,
    state_dict: Dict[str, TensorSource],
) -> None:
    # Inject matched tensors into the ongoing state_dict before sharding.
    merged = 0
    normalized_prefixes = [prefix if prefix.endswith(".") else f"{prefix}." for prefix in prefixes]
    tensors = _load_tensors_by_selectors(
        model_local_path,
        normalized_prefixes,
        suffixes,
        tensor_names,
    )
    for name, tensor in tensors.items():
        state_dict[name] = _tensor_source_from_tensor(name, tensor)
        merged += 1
    if merged:
        log.info(
            "Model: Merged %s tensors selected by prefixes=%s suffixes=%s names=%s into the state dict",
            merged,
            normalized_prefixes,
            suffixes,
            tensor_names,
        )
    else:
        log.warn(
            "Model: No tensors matched out-of-model selectors prefixes=%s suffixes=%s names=%s",
            normalized_prefixes,
            suffixes,
            tensor_names,
        )


def _validated_save_state_overlay(owner):
    """Return one validated out-of-tree model overlay contract, if attached."""

    provider = getattr(owner, "save_state_overlay", None)
    if not callable(provider):
        return None
    contract = provider()
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise TypeError("save-state overlay contract must be a dictionary")
    overlay_model = contract.get("model")
    prefixes = contract.get("replace_prefixes")
    if not isinstance(overlay_model, torch.nn.Module):
        raise TypeError("save-state overlay has no torch module")
    if (
        not isinstance(prefixes, (list, tuple))
        or not prefixes
        or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
        or len(set(prefixes)) != len(prefixes)
    ):
        raise ValueError("save-state overlay prefixes must be unique non-empty strings")
    normalized = tuple(
        prefix if prefix.endswith(".") else f"{prefix}." for prefix in prefixes
    )
    return contract, overlay_model, normalized


def _validated_save_state_overlay_suffixes(contract):
    expected_suffixes = contract.get("expected_suffixes")
    if expected_suffixes is None:
        return None
    if (
        not isinstance(expected_suffixes, (list, tuple))
        or not expected_suffixes
        or any(
            not isinstance(suffix, str) or not suffix
            for suffix in expected_suffixes
        )
        or len(set(expected_suffixes)) != len(expected_suffixes)
    ):
        raise ValueError(
            "save-state overlay expected suffixes must be unique non-empty strings"
        )
    return set(expected_suffixes)


def _validated_save_state_overlay_removals(contract):
    """Return source-checkpoint prefixes superseded under different names."""

    prefixes = contract.get("remove_prefixes")
    if prefixes is None:
        return (), None
    if (
        not isinstance(prefixes, (list, tuple))
        or not prefixes
        or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
        or len(set(prefixes)) != len(prefixes)
    ):
        raise ValueError(
            "save-state overlay removal prefixes must be unique non-empty strings"
        )
    normalized = tuple(
        prefix if prefix.endswith(".") else f"{prefix}." for prefix in prefixes
    )
    expected_suffixes = contract.get("remove_expected_suffixes")
    if (
        not isinstance(expected_suffixes, (list, tuple))
        or not expected_suffixes
        or any(
            not isinstance(suffix, str) or not suffix
            for suffix in expected_suffixes
        )
        or len(set(expected_suffixes)) != len(expected_suffixes)
    ):
        raise ValueError(
            "save-state overlay removal suffixes must be unique non-empty strings"
        )
    return normalized, set(expected_suffixes)


def _validate_exllamav3_publication_modules(
    owner,
    validated_overlay=None,
) -> int:
    """Fail before META sync if a completed EXL3 result is still native.

    Recovery controllers may defer packed-module installation while they
    advance through activation boundaries.  Every module represented in the
    authoritative EXL3 error ledger must be present as an EXL3 module in the
    publication tree (or its declared save-state overlay) before the generic
    saver is allowed to materialize remaining META tensors.  Otherwise a
    native source weight can be reconstructed accidentally, defeating bounded
    recovery and potentially exhausting host memory on large MoE models.
    """

    records = [
        entry.get("exl3_error_ledger_record")
        for entry in (getattr(owner, "quant_log", None) or ())
        if isinstance(entry, dict)
        and isinstance(entry.get("exl3_error_ledger_record"), dict)
    ]
    if not records:
        return 0
    modules = [record.get("module") for record in records]
    if any(not isinstance(module, str) or not module for module in modules):
        raise RuntimeError(
            "EXL3 publication ledger contains a missing module identity"
        )
    expected = set(modules)
    actual = {
        name
        for name, module in owner.model.named_modules()
        if getattr(module, "QUANT_TYPE", None) == "exl3"
    }
    validated = validated_overlay or _validated_save_state_overlay(owner)
    if validated is not None:
        _contract, overlay_model, _normalized = validated
        actual.update(
            name
            for name, module in overlay_model.named_modules()
            if getattr(module, "QUANT_TYPE", None) == "exl3"
        )
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(
            "EXL3 publication tree is missing packed modules: "
            f"actual={len(expected) - len(missing)} expected={len(expected)} "
            "missing="
            + ", ".join(missing[:8])
        )
    return len(expected)


def _exllamav3_checkpoint_passthrough_plan(owner, tensor_storage):
    """Describe an exact native checkpoint passthrough when names are stable.

    A lazy shell uses runtime module names, which can differ from the original
    safetensors keys after Transformers applies a conversion map.  Serializing
    untouched tensors from that shell silently renames native checkpoint state
    and can also materialize it unnecessarily.  When every EXL3 replacement
    has the same authoritative ``<module>.weight`` name in the source index,
    retain all other source entries byte-for-byte under their original names.
    """

    checkpoint_source = getattr(owner, "turtle_model", None)
    weight_map = getattr(checkpoint_source, "_weight_map", None)
    source_root = getattr(checkpoint_source, "model_local_path", None)
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    if not isinstance(source_root, str) or not source_root:
        return None
    if not isinstance(tensor_storage, dict) or not tensor_storage:
        return None

    quant_names = set()
    replaced_source_modules = set()
    source_candidates = getattr(
        owner,
        "exllamav3_checkpoint_module_candidates",
        None,
    )
    fused_replacements = getattr(owner, "exllamav3_checkpoint_replaced_tensors", None)
    replaced_fused_names = None
    if callable(fused_replacements):
        replaced_fused_names = fused_replacements(tensor_storage)
        if (
            not isinstance(replaced_fused_names, (set, frozenset))
            or not replaced_fused_names
            or not all(isinstance(name, str) for name in replaced_fused_names)
            or not replaced_fused_names.issubset(weight_map)
        ):
            raise RuntimeError("EXL3 fused checkpoint replacement set is invalid")
    for module_name, entry in tensor_storage.items():
        if not isinstance(module_name, str) or not module_name:
            return None
        stored = entry.get("stored_tensors") if isinstance(entry, dict) else None
        if not isinstance(stored, dict) or not stored:
            return None
        names = set(stored)
        if any(
            not isinstance(name, str) or not name.startswith(f"{module_name}.")
            for name in names
        ):
            return None
        quant_names.update(names)
        if replaced_fused_names is not None:
            continue
        if callable(source_candidates):
            candidates = source_candidates(module_name)
            if (
                not isinstance(candidates, (list, tuple))
                or not candidates
                or any(
                    not isinstance(candidate, str) or not candidate
                    for candidate in candidates
                )
                or len(set(candidates)) != len(candidates)
            ):
                raise RuntimeError(
                    "EXL3 checkpoint module candidates must be unique "
                    f"non-empty strings for {module_name}"
                )
        else:
            candidates = (module_name,)
        matches = [
            candidate
            for candidate in candidates
            if f"{candidate}.weight" in weight_map
        ]
        if len(matches) != 1:
            if callable(source_candidates):
                raise RuntimeError(
                    "EXL3 checkpoint module mapping is not unique for "
                    f"{module_name}: {matches}"
                )
            return None
        replaced_source_modules.add(matches[0])

    if replaced_fused_names is not None:
        return {
            "checkpoint_source": checkpoint_source,
            "quant_names": frozenset(quant_names),
            "replaced_source_names": frozenset(replaced_fused_names),
        }

    # Conversion-based architectures are eligible only when the publication
    # module identities themselves are also source-checkpoint identities. This
    # makes passthrough exact and keeps the fallback behavior for architectures
    # whose quantized prefixes require a more involved conversion contract.
    # An EXL3 module replaces the complete native module state, not only its
    # weight tensor.  Native FP8 checkpoints commonly keep auxiliary tensors
    # such as ``weight_scale_inv`` under the same prefix; preserving those
    # after replacing the weight leaks stale source state into the published
    # tensor namespace.  Match the complete authoritative source prefix, just
    # like the ordinary save-state overlay path does.
    replaced_source_names = set()
    for name in weight_map:
        owner_name = name
        while "." in owner_name:
            owner_name = owner_name.rsplit(".", 1)[0]
            if owner_name in replaced_source_modules:
                replaced_source_names.add(name)
                break
    return {
        "checkpoint_source": checkpoint_source,
        "quant_names": frozenset(quant_names),
        "replaced_source_names": frozenset(replaced_source_names),
    }


def _torch_dtype_from_safetensors_name(name: str) -> torch.dtype:
    mapping = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    optional = {
        "F8_E4M3": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
        "F8_E8M0": "float8_e8m0fnu",
    }
    attribute = optional.get(name)
    if attribute is not None and hasattr(torch, attribute):
        return getattr(torch, attribute)
    dtype = mapping.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported safetensors dtype in checkpoint passthrough: {name}")
    return dtype


def _is_safe_relative_checkpoint_shard(shard_name: str) -> bool:
    """Return whether an index shard entry is a safe relative path.

    Safetensors indexes may reference shards in nested subdirectories, as the
    public ``reshard()`` per-layer layout does (``model.layers.0/layer.safetensors``).
    Accept those, but reject absolute or drive-qualified paths, backslashes,
    NUL bytes, and empty, ``.`` or ``..`` components so a shard cannot escape
    the checkpoint root.
    """

    try:
        validate_relative_checkpoint_path(shard_name, kind="checkpoint shard name")
    except CheckpointPathError:
        return False
    return True


def _exllamav3_owned_native_source_names(
    model,
    checkpoint_source,
    names,
) -> Optional[set]:
    """Return the subset of native source names the loaded model owns (#5029).

    The source index describes the whole on-disk checkpoint, which can contain
    entries the model never instantiated: layers above a reduced
    ``num_hidden_layers`` and MTP / next-token heads are the common cases. An
    exact passthrough that republishes them writes dense state the model does
    not have, inflating the saved checkpoint and the reload peak.

    An entry is owned when its owning module exists in the loaded model (the
    model's own state covers that module, possibly under an aliased spelling),
    or when a runtime parameter/buffer resolves to it through the checkpoint
    source's runtime -> checkpoint alias chain. ``None`` means ownership cannot
    be determined, which callers must treat as "keep every entry".
    """

    candidate_names = {str(name) for name in names}
    if not candidate_names:
        return set()

    named_modules = getattr(model, "named_modules", None)
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_modules) or not callable(named_parameters):
        return None

    try:
        module_paths = set()
        runtime_names = set()
        for module_name, module in named_modules():
            module_paths.add(module_name)
            non_persistent = getattr(module, "_non_persistent_buffers_set", ())
            for buffer_name, _buffer in module.named_buffers(recurse=False):
                if buffer_name in non_persistent:
                    continue
                runtime_names.add(
                    f"{module_name}.{buffer_name}" if module_name else buffer_name
                )
        runtime_names.update(name for name, _parameter in named_parameters())
    except Exception:
        log.debug(
            "Model save: could not enumerate the loaded model's owned tensors",
            exc_info=True,
        )
        return None

    owned = {
        name
        for name in candidate_names
        if name.rpartition(".")[0] in module_paths
    }
    remaining = candidate_names - owned
    if remaining:
        # A converted or renamed checkpoint spells the same tensor differently.
        # Resolve every runtime name through the source's alias chain so an
        # owned tensor is never dropped just because its runtime spelling
        # differs from the checkpoint key.
        alias_resolver = getattr(checkpoint_source, "checkpoint_source_aliases", None)
        if callable(alias_resolver):
            for runtime_name in runtime_names:
                for candidate in alias_resolver(runtime_name):
                    if candidate in remaining:
                        owned.add(candidate)
                        remaining.discard(candidate)
                if not remaining:
                    break
    return owned


def _build_exllamav3_checkpoint_passthrough_state_dict(
    owner,
    plan,
    *,
    offload_root=None,
    validated_overlay=None,
) -> Dict[str, TensorSource]:
    """Build packed EXL3 state plus exact, zero-copy native source entries."""

    quant_names = set(plan["quant_names"])
    state_dict = get_state_dict_for_save(
        owner.model,
        offload_root=offload_root,
        include_names=quant_names,
    )
    _apply_save_state_overlay(
        owner,
        state_dict,
        validated_overlay=validated_overlay,
        require_source_removals=False,
    )
    actual_quant_names = set(state_dict)
    if actual_quant_names != quant_names:
        missing = sorted(quant_names - actual_quant_names)
        unexpected = sorted(actual_quant_names - quant_names)
        raise RuntimeError(
            "EXL3 checkpoint passthrough packed tensor census differs: "
            f"actual={len(actual_quant_names)} expected={len(quant_names)} "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    checkpoint_source = plan["checkpoint_source"]
    source_root = os.path.realpath(checkpoint_source.model_local_path)
    weight_map = checkpoint_source._weight_map
    replaced = set(plan["replaced_source_names"])
    native_names = set(weight_map) - replaced
    owned_native_names = _exllamav3_owned_native_source_names(
        owner.model,
        checkpoint_source,
        native_names,
    )
    if owned_native_names is None:
        log.info(
            "Model save: checkpoint ownership is unknown; preserving all %s "
            "native source tensors.",
            len(native_names),
        )
    else:
        skipped_native_names = sorted(native_names - owned_native_names)
        native_names = owned_native_names
        if skipped_native_names:
            log.info(
                "Model save: skipped %s native source tensor(s) the loaded model "
                "does not own: %s",
                len(skipped_native_names),
                skipped_native_names[:8],
            )
    by_shard: Dict[str, List[str]] = {}
    for name in native_names:
        shard_name = weight_map.get(name)
        if not isinstance(shard_name, str) or not shard_name:
            raise RuntimeError(f"Invalid checkpoint shard for native tensor {name}")
        by_shard.setdefault(shard_name, []).append(name)

    native_sources: Dict[str, TensorSource] = {}
    for shard_name in sorted(by_shard):
        # Require a safe relative path: nested per-layer shards are valid, but
        # absolute paths, traversal, and backslashes are not. Hugging Face
        # snapshots intentionally store basename-only symlinks to the
        # repository's shared ``blobs`` directory, which is outside the
        # snapshot root after realpath resolution, so resolved paths are not
        # required to remain under the source root.
        if not _is_safe_relative_checkpoint_shard(shard_name):
            raise RuntimeError(f"Checkpoint shard name is unsafe: {shard_name}")
        shard_path = os.path.realpath(os.path.join(source_root, shard_name))
        if not os.path.isfile(shard_path):
            raise RuntimeError(f"Checkpoint shard is not a regular file: {shard_name}")
        with safe_open(shard_path, framework="pt", device="cpu") as handler:
            available = set(handler.keys())
            expected = set(by_shard[shard_name])
            missing = sorted(expected - available)
            if missing:
                raise RuntimeError(
                    f"Checkpoint shard is missing native tensors: {shard_name} {missing[:8]}"
                )
            for name in sorted(expected):
                tensor_slice = handler.get_slice(name)
                dtype = _torch_dtype_from_safetensors_name(tensor_slice.get_dtype())
                shape = tuple(tensor_slice.get_shape())
                native_sources[name] = TensorSource(
                    name=name,
                    torch_dtype=dtype,
                    shape=shape,
                    source=OffloadTensorRef(
                        path=shard_path,
                        torch_dtype=dtype,
                        shape=shape,
                        format="safetensors",
                        weight_name=name,
                    ),
                )

    collisions = set(native_sources).intersection(state_dict)
    if collisions:
        raise RuntimeError(
            "EXL3 checkpoint passthrough native/packed collision: "
            + ", ".join(sorted(collisions)[:8])
        )
    combined = {
        name: native_sources.get(name, state_dict.get(name))
        for name in sorted(set(native_sources).union(state_dict))
    }
    if any(source is None for source in combined.values()):
        raise RuntimeError("EXL3 checkpoint passthrough produced an empty tensor source")
    log.info(
        "Model save: preserving %s native source tensors under exact checkpoint names; "
        "publishing %s packed EXL3 tensors without native shell materialization.",
        len(native_sources),
        len(state_dict),
    )
    return combined


def _build_exllamav3_tensor_storage_for_save(
    owner,
    validated_overlay=None,
) -> Dict[str, Dict[str, Any]]:
    """Include EXL3 modules supplied by a disjoint save-state overlay.

    DeepSeek V4 quantizes its MTP body through an attached model rather than
    placing those modules in the target model tree. The tensor payload overlay
    already replaces the native MTP prefixes during save; its EXL3 metadata
    must be assembled from the same exact prefix contract before config files
    are written.
    """

    storage = build_exllamav3_tensor_storage(owner.model)
    validated = validated_overlay or _validated_save_state_overlay(owner)
    if validated is None:
        return storage
    contract, overlay_model, normalized = validated
    overlay_storage = build_exllamav3_tensor_storage(overlay_model)
    expected_modules = {prefix[:-1] for prefix in normalized}
    selected = {
        name: entry
        for name, entry in overlay_storage.items()
        if name in expected_modules
    }
    if set(selected) != expected_modules:
        missing = sorted(expected_modules - set(selected))
        raise ValueError(
            "save-state overlay has no EXL3 tensor-storage metadata for "
            + ", ".join(missing)
        )
    expected_suffixes = _validated_save_state_overlay_suffixes(contract)
    if expected_suffixes is not None:
        for module, entry in selected.items():
            stored_tensors = entry.get("stored_tensors") if isinstance(entry, dict) else None
            prefix = f"{module}."
            actual_names = set(stored_tensors) if isinstance(stored_tensors, dict) else set()
            expected_names = {f"{prefix}{suffix}" for suffix in expected_suffixes}
            actual_suffixes = {name[len(prefix):] for name in actual_names if name.startswith(prefix)}
            if actual_names != expected_names or actual_suffixes != expected_suffixes:
                raise ValueError(
                    f"save-state overlay tensor-storage contract differs for {module}: "
                    f"actual={sorted(actual_suffixes)} "
                    f"expected={sorted(expected_suffixes)}"
                )
    for module in expected_modules:
        storage.pop(module, None)
    collisions = set(selected).intersection(storage)
    if collisions:
        raise ValueError(
            "save-state overlay tensor-storage collides outside replacement prefixes: "
            + ", ".join(sorted(collisions))
        )
    storage.update(selected)
    return storage


def _apply_save_state_overlay(
    owner,
    state_dict: Dict[str, TensorSource],
    validated_overlay=None,
    *,
    require_source_removals: bool = True,
) -> None:
    """Replace complete tensor prefixes from an attached quantization model."""

    validated = validated_overlay or _validated_save_state_overlay(owner)
    if validated is None:
        return
    contract, overlay_model, normalized = validated
    overlay = get_state_dict_for_save(
        overlay_model,
        offload_root=contract.get("offload_root"),
        include_prefixes=normalized,
    )
    selected = {
        name: source
        for name, source in overlay.items()
        if any(name.startswith(prefix) for prefix in normalized)
    }
    missing = [
        prefix
        for prefix in normalized
        if not any(name.startswith(prefix) for name in selected)
    ]
    if missing:
        raise ValueError(
            "save-state overlay has no replacement tensors for " + ", ".join(missing)
        )
    expected_suffixes = _validated_save_state_overlay_suffixes(contract)
    if expected_suffixes is not None:
        for prefix in normalized:
            actual_suffixes = {
                name[len(prefix) :]
                for name in selected
                if name.startswith(prefix)
            }
            if actual_suffixes != expected_suffixes:
                raise ValueError(
                    f"save-state overlay tensor contract differs for {prefix}: "
                    f"actual={sorted(actual_suffixes)} "
                    f"expected={sorted(expected_suffixes)}"
                )
    removal_prefixes, removal_suffixes = _validated_save_state_overlay_removals(
        contract
    )
    removed_native = [
        name
        for name in state_dict
        if any(name.startswith(prefix) for prefix in removal_prefixes)
    ]
    if require_source_removals:
        for prefix in removal_prefixes:
            actual_suffixes = {
                name[len(prefix) :]
                for name in removed_native
                if name.startswith(prefix)
            }
            if actual_suffixes != removal_suffixes:
                raise ValueError(
                    f"save-state overlay source-removal contract differs for {prefix}: "
                    f"actual={sorted(actual_suffixes)} "
                    f"expected={sorted(removal_suffixes)}"
                )
    stale = set(removed_native)
    stale.update(
        name
        for name in state_dict
        if any(name.startswith(prefix) for prefix in normalized)
    )
    for name in stale:
        del state_dict[name]
    collisions = set(selected).intersection(state_dict)
    if collisions:
        raise ValueError(
            "save-state overlay collides outside replacement prefixes: "
            + ", ".join(sorted(collisions))
        )
    state_dict.update(selected)
    log.info(
        "Model: Replaced %s tensor prefixes (%s source tensors, including %s "
        "checkpoint aliases -> %s overlay tensors)",
        len(normalized),
        len(stale),
        len(removed_native),
        len(selected),
    )


def _normalize_out_of_model_tensors_entries(
    entries: Optional[List[Union[str, Dict[str, Any]]]]
) -> tuple[List[str], List[str], List[str], List[str]]:
    # Normalize configured files/prefixes into explicit lists.
    copy_files: List[str] = []
    prefixes: List[str] = []
    suffixes: List[str] = []
    tensor_names: List[str] = []
    if not entries:
        return copy_files, prefixes, suffixes, tensor_names

    raw_entries = list(entries) if isinstance(entries, (list, tuple)) else [entries]
    for entry in raw_entries:
        if isinstance(entry, str):
            copy_files.append(entry)
            continue
        if not isinstance(entry, dict):
            raise TypeError("out_of_model_tensors entries must be dict.")

        files_value = entry.get("files")
        if files_value is not None:
            files = [files_value] if isinstance(files_value, str) else list(files_value)
            for file in files:
                if not isinstance(file, str) or not file:
                    raise ValueError("`files` entries must be non-empty strings.")
                copy_files.append(file)

        prefixes_value = entry.get("prefixes")
        if prefixes_value is not None:
            prefix_list = [prefixes_value] if isinstance(prefixes_value, str) else list(prefixes_value)
            for prefix in prefix_list:
                if not isinstance(prefix, str) or not prefix:
                    raise ValueError("`prefixes` entries must be non-empty strings.")
                prefixes.append(prefix)

        suffixes_value = entry.get("suffixes")
        if suffixes_value is not None:
            suffix_list = [suffixes_value] if isinstance(suffixes_value, str) else list(suffixes_value)
            for suffix in suffix_list:
                if not isinstance(suffix, str) or not suffix:
                    raise ValueError("`suffixes` entries must be non-empty strings.")
                suffixes.append(suffix)

        tensors_value = entry.get("tensors")
        if tensors_value is not None:
            tensors = [tensors_value] if isinstance(tensors_value, str) else list(tensors_value)
            for tensor_name in tensors:
                if not isinstance(tensor_name, str) or not tensor_name:
                    raise ValueError("`tensors` entries must be non-empty strings.")
                tensor_names.append(tensor_name)

    return copy_files, prefixes, suffixes, tensor_names


def _resolve_layer_split_group(tensor_name: str, layer_prefixes: List[str]) -> tuple[str, bool]:
    for prefix in sorted((prefix for prefix in layer_prefixes if prefix), key=len, reverse=True):
        expected_prefix = f"{prefix}."
        if not tensor_name.startswith(expected_prefix):
            continue
        remainder = tensor_name[len(expected_prefix):]
        layer_idx, dot, _ = remainder.partition(".")
        if layer_idx.isdigit() and dot:
            return f"{prefix}.{layer_idx}", True

    if "." in tensor_name:
        return tensor_name.rsplit(".", 1)[0], False
    return "", False


def _module_is_leaf(model, module_name: str) -> bool:
    if not module_name:
        return False
    try:
        module = get_module_by_name(model, module_name)
    except Exception:
        return False
    return not any(True for _ in module.named_children())


def _cleanup_legacy_leaf_group_dir(save_dir: str, group_name: str) -> None:
    legacy_dir = join(save_dir, group_name)
    if not os.path.isdir(legacy_dir):
        return

    for cleanup_base_name, cleanup_save_name in {
        ("layer", "layer.safetensors"),
        ("model", "model.safetensors"),
    }:
        _cleanup_saved_weight_files(
            save_dir=legacy_dir,
            expected_files=[],
            model_base_name=cleanup_base_name,
            model_save_name=cleanup_save_name,
        )

    try:
        if not os.listdir(legacy_dir):
            os.rmdir(legacy_dir)
    except OSError:
        pass


def _stream_state_dict_to_layer_dirs(
    state_dict: Dict[str, Any],
    save_dir: str,
    model_base_name: str,
    model_save_name: str,
    metadata: Dict[str, str],
    max_shard_size: Optional[int],
    layer_prefixes: List[str],
    model,
) -> tuple[List[str], Dict[str, str], int]:
    grouped_state_dict: Dict[str, Dict[str, Any]] = {}
    layer_groups: Dict[str, bool] = {}
    for tensor_name, tensor_source in state_dict.items():
        group_name, is_layer_group = _resolve_layer_split_group(tensor_name, layer_prefixes)
        group = grouped_state_dict.setdefault(group_name, {})
        group[tensor_name] = tensor_source
        layer_groups[group_name] = is_layer_group

    expected_files: List[str] = []
    tensor_to_filename: Dict[str, str] = {}
    total_size = 0
    root_expected_files: List[str] = []
    cleanup_specs = {(model_base_name, model_save_name)}
    if model_base_name != "model" or model_save_name != "model.safetensors":
        cleanup_specs.add(("model", "model.safetensors"))

    for group_dir_name, group_state_dict in grouped_state_dict.items():
        is_layer_group = layer_groups.get(group_dir_name, False)
        is_leaf_group = (not is_layer_group) and _module_is_leaf(model, group_dir_name)

        if is_layer_group:
            group_dir = join(save_dir, group_dir_name)
            group_model_base_name = model_base_name
            group_model_save_name = model_save_name
            relative_prefix = f"{group_dir_name}/"
            group_cleanup_specs = cleanup_specs
        elif is_leaf_group and group_dir_name:
            group_dir = save_dir
            group_model_base_name = group_dir_name
            group_model_save_name = f"{group_dir_name}.safetensors"
            relative_prefix = ""
            group_cleanup_specs = {(group_model_base_name, group_model_save_name)}
        else:
            group_dir = save_dir if not group_dir_name else join(save_dir, group_dir_name)
            group_model_base_name = model_base_name
            group_model_save_name = model_save_name
            relative_prefix = "" if not group_dir_name else f"{group_dir_name}/"
            group_cleanup_specs = cleanup_specs

        os.makedirs(group_dir, exist_ok=True)

        group_expected_files, group_tensor_to_filename, group_total_size = streaming_state_dict_to_shards(
            group_state_dict,
            save_dir=group_dir,
            model_base_name=group_model_base_name,
            single_file_name=group_model_save_name,
            metadata=metadata,
            max_shard_size=max_shard_size,
        )
        total_size += group_total_size

        for cleanup_base_name, cleanup_save_name in group_cleanup_specs:
            _cleanup_saved_weight_files(
                save_dir=group_dir,
                expected_files=group_expected_files,
                model_base_name=cleanup_base_name,
                model_save_name=cleanup_save_name,
            )

        if is_leaf_group and group_dir_name:
            _cleanup_legacy_leaf_group_dir(save_dir=save_dir, group_name=group_dir_name)
        elif group_dir_name:
            _cleanup_saved_weight_files(
                save_dir=save_dir,
                expected_files=[],
                model_base_name=group_dir_name,
                model_save_name=f"{group_dir_name}.safetensors",
            )

        if not group_dir_name and not is_leaf_group:
            root_expected_files.extend(group_expected_files)

        for filename in group_expected_files:
            relative_filename = f"{relative_prefix}{filename}" if relative_prefix else filename
            expected_files.append(relative_filename)

        for tensor_name, filename in group_tensor_to_filename.items():
            relative_filename = f"{relative_prefix}{filename}" if relative_prefix else filename
            tensor_to_filename[tensor_name] = relative_filename

    for cleanup_base_name, cleanup_save_name in cleanup_specs:
        _cleanup_saved_weight_files(
            save_dir=save_dir,
            expected_files=root_expected_files,
            model_base_name=cleanup_base_name,
            model_save_name=cleanup_save_name,
        )

    return expected_files, tensor_to_filename, total_size

def _export_shell_device(qcfg) -> Optional[DEVICE]:
    """Device constraint the re-save export shell asks the kernel selector for.

    The shell only materializes the checkpoint's packed buffer layout and then
    loads the checkpoint into it; it never runs a forward pass. Some formats
    (AWQ GEMV / GEMV_FAST) only have accelerator kernels, so pinning AUTO to
    CPU rejects every candidate and fails the whole re-save with "No valid
    quant linear" (issue #4305). For those formats the shell drops the device
    constraint so the selector can still return the layout class.

    Formats that do have a CPU kernel keep asking for CPU: on an accelerator
    host, an unconstrained AUTO could otherwise pick a faster accelerator
    kernel whose in-memory buffer layout differs from the checkpoint's.
    """

    from ..utils.importer import get_auto_backend_kernel_mapping

    fmt = resolve_quant_format(qcfg.format, qcfg.method)
    method = qcfg.export_quant_method()
    candidates = get_auto_backend_kernel_mapping().get(method, {}).get(fmt, {})
    if any(
        DEVICE.ALL in cls.SUPPORTS_DEVICES or DEVICE.CPU in cls.SUPPORTS_DEVICES
        for cls in candidates.values()
    ):
        return DEVICE.CPU
    return None


def ModelWriter(cls):
    def save_pretrained(
            self,
            save_dir: str,
            **kwargs,
    ):
        log.warn("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    cls.save_pretrained = save_pretrained

    def _eora_save(self, save_dir: str, model_save_dir: str = None):
        assert isinstance(self.quantize_config.adapter, Lora)

        assert hasattr(self, 'lora_results')

        # save lora tensors
        if self.lora_results:  # TODO REFRACTOR
            weights = {}
            target_modules = set()
            # convert the dict into safetensors compatible dict
            for key, adapter in self.lora_results.items():
                assert isinstance(adapter, Lora)
                # PEFT matches module names case-sensitively; MPT (`Wqkv`) and
                # Baichuan (`W_pack`) must keep their canonical casing in both
                # the tensor keys and `target_modules`.
                simple_module_name = key.split(".")[-1] # mlp.gate_proj => gate_proj
                target_modules.add(simple_module_name)

                # while key.startswith('model.'):
                #     key = key.removeprefix('model.') # some HF models use model. or model.model.

                # must normalize key since HF can load weights as `model.` or not based on what AutoModel is used
                weight_key = f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}{key}"

                weights[f"{weight_key}.lora_A.weight"] = adapter.lora_A
                weights[f"{weight_key}.lora_B.weight"] = adapter.lora_B
                log.info(f"Adapter: EoRA weights found -> `{weight_key}.lora_A/Lora_B.weight`, rank = `{adapter.rank}`")

            weight_file_path = f"{save_dir.removesuffix('/')}/{HF_ADAPTER_FILE_NAME}"

            # Dynamic rank: export the ranks the adapters were actually built
            # with, one exact key per module. `dynamic` is first-match while
            # PEFT applies every `rank_pattern` key on its own, so copying the
            # rules verbatim would export ranks that were never trained. See
            # docs/adapter-rank-pattern-matching.md.
            rank_pattern = {}
            if getattr(self.quantize_config, "dynamic", None):
                rank_pattern = self.quantize_config.extract_adapter_rank_patterns(
                    module_ranks={
                        key: adapter.rank for key, adapter in self.lora_results.items()
                    },
                )

            # EoRA's residual targets a unit-scale correction. PEFT resolves
            # alpha per module, so mirroring dynamic ranks keeps every module's
            # `alpha / rank` ratio at one.
            alpha_pattern = dict(rank_pattern)
            lora_cfg = LoraConfig(base_model_name_or_path=model_save_dir,
                                  r=self.quantize_config.adapter.rank,
                                  lora_alpha=self.quantize_config.adapter.rank,
                                  target_modules=list(target_modules),
                                  rank_pattern=rank_pattern,
                                  alpha_pattern=alpha_pattern)
            lora_cfg.save_pretrained(save_dir=save_dir)

            log.info(f"Adapter: Saving EoRA weights to -> `{save_dir}`")

            save_file(tensors=weights, filename=weight_file_path, metadata={"format": "pt"})

            del self.lora_results  # TODO REFRACTOR

    cls.eora_save = _eora_save

    def save_quantized_embeddings(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
    ):
        """Save a module-replacement quantized model as a complete checkpoint."""
        del max_shard_size
        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")
        prefixes = sorted({
            prefix for prefix in getattr(self, "_module_replacement_prefixes", set())
            if isinstance(prefix, str) and prefix
        })
        if not prefixes:
            raise ValueError("Module replacement save requires quantized module replacement prefixes.")
        _validate_nf12_payload_ownership(self, prefixes=prefixes)
        checkpoint_source = self.turtle_model or getattr(self, "_module_replacement_source", None)
        if checkpoint_source is None or not hasattr(checkpoint_source, "_weight_map") or not hasattr(checkpoint_source, "model_local_path"):
            raise ValueError("Module replacement save requires a LazyTurtle checkpoint source.")

        os.makedirs(save_dir, exist_ok=True)
        _copy_missing_checkpoint_files(checkpoint_source.model_local_path, save_dir)

        quantizers = [f"{META_QUANTIZER_GPTQMODEL}:{__version__}"]
        if meta_quantizer:
            if len(meta_quantizer.split(":")) == 2:
                quantizers.append(meta_quantizer.replace(" ", ""))
            else:
                log.warn(f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'")
        self.quantize_config.meta_set_versionable(key=META_FIELD_QUANTIZER, value=quantizers)
        self.quantize_config.meta_set(key=META_FIELD_URI, value=META_VALUE_URI)

        metadata = {} if safetensors_metadata is None else {str(k): str(v) for k, v in safetensors_metadata.items()}
        metadata["format"] = "pt"
        rewritten_files, tensor_to_filename, _total_size, removed_names = _save_module_replacement_safetensors(
            self.model,
            checkpoint_source,
            prefixes,
            save_dir=save_dir,
            metadata=metadata,
        )
        index_name = getattr(checkpoint_source, "_index_filename", "model.safetensors.index.json")
        index_path = join(save_dir, index_name) if index_name else None
        if index_path and os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.setdefault("weight_map", {})
            for tensor_name in removed_names:
                weight_map.pop(tensor_name, None)
            weight_map.update(tensor_to_filename)
            index.setdefault("metadata", {})["total_size"] = sum(
                os.path.getsize(os.path.join(save_dir, filename))
                for filename in set(weight_map.values())
                if os.path.exists(os.path.join(save_dir, filename))
            )
            with open(index_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(index, indent=2, sort_keys=True) + "\n")
        elif len(rewritten_files) > 1:
            log.warn("Module replacement save: no safetensors index found in `%s`; updated shards only.", save_dir)

        serialized_config = copy.deepcopy(self.quantize_config)
        serialized_config.save_pretrained(save_dir)
        quant_config_path = os.path.join(save_dir, "quantize_config.json")
        with open(quant_config_path, "r", encoding="utf-8") as handle:
            quantization_config = json.load(handle)
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        model_config["quantization_config"] = quantization_config
        runtime_config = getattr(self.model, "config", None)
        if runtime_config is not None and hasattr(runtime_config, "tie_word_embeddings"):
            model_config["tie_word_embeddings"] = bool(runtime_config.tie_word_embeddings)
        if not getattr(self, "trust_remote_code", False):
            # The `.py` files are not copied below, so an inherited `auto_map`
            # would point at missing files and break AutoConfig loading.
            model_config.pop("auto_map", None)
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(model_config, indent=2) + "\n")

        if self.trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=self.model_local_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)
        if hasattr(self, "processor") and isinstance(self.processor, ProcessorMixin):
            self.processor.save_pretrained(save_dir)

    cls.save_quantized_embeddings = save_quantized_embeddings

    def save_quantized(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            split_by: Optional[str] = None,
    ):
        """save quantized model and configs to local disk"""
        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        if getattr(self.quantize_config, "method", None) == METHOD.TT_BLOCKFLOAT:
            # Block float is a pre-quantized storage format: the modules already
            # hold packed codes on the hardware grid, so the generic writer has
            # nothing to pack and would rebuild a dense model it cannot
            # represent. Export the packed state verbatim instead.
            from ..utils.tt_blockfloat_quant import (
                DEFAULT_MAX_SHARD_BYTES,
                save_blockfloat_model,
            )

            tokenizer = getattr(self, "tokenizer", None)
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            if not hasattr(tokenizer, "save_pretrained"):
                tokenizer = None
            # Honour the same shard limit and header metadata the generic path
            # uses. Ignoring them silently changed the published layout and
            # dropped caller metadata.
            shard_bytes = _parse_max_shard_size(max_shard_size)
            header_metadata = _normalize_metadata(safetensors_metadata)
            info = save_blockfloat_model(
                self.model,
                save_dir,
                bits=self.quantize_config.bits,
                embed_bits=int(getattr(self.quantize_config, "embed_bits", 8) or 8),
                tokenizer=tokenizer,
                safetensors_metadata=header_metadata,
                max_shard_bytes=(
                    DEFAULT_MAX_SHARD_BYTES if shard_bytes is None else shard_bytes
                ),
            )
            # Generation settings are part of the save contract. This early
            # return would otherwise drop ``generation_config.json`` and change
            # stopping behaviour on reload.
            generation_config = getattr(self.model, "generation_config", None)
            can_generate = getattr(self.model, "can_generate", None)
            if generation_config is not None and (
                not callable(can_generate) or can_generate()
            ):
                _save_generation_config(generation_config, save_dir)
            log.info(
                f"Saved tt-blockfloat checkpoint to {save_dir}: "
                f"{info['layers']} packed modules, {info['payload_bytes'] / (1 << 20):.2f} MiB "
                f"in {info['shards']} shard(s)"
            )
            return info

        (
            copy_tensor_files,
            prefix_entries,
            suffix_entries,
            tensor_name_entries,
        ) = _normalize_out_of_model_tensors_entries(
            getattr(self, "out_of_model_tensors", None)
        )
        out_of_model_state_dict: Dict[str, TensorSource] = {}
        if prefix_entries or suffix_entries or tensor_name_entries:
            _merge_selected_tensors_into_state_dict(
                prefix_entries, suffix_entries, tensor_name_entries,
                self.model_local_path, out_of_model_state_dict,
            )

        os.makedirs(save_dir, exist_ok=True)

        if self.quant_log:
            write_exl3_error_ledger(
                save_dir,
                (
                    entry["exl3_error_ledger_record"]
                    for entry in self.quant_log
                    if isinstance(entry.get("exl3_error_ledger_record"), dict)
                ),
            )
            with open(os.path.join(save_dir, "quant_log.csv"), mode='w', newline='') as file:
                w = csv.writer(file)
                w.writerow([PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_LOSS,
                            QUANT_LOG_LOSS_KIND, QUANT_LOG_NSAMPLES, QUANT_LOG_DAMP, PROCESS_LOG_TIME])
                w.writerows([[entry.get(PROCESS_LOG_LAYER), entry.get(PROCESS_LOG_MODULE), entry.get(QUANT_LOG_LOSS),
                              entry.get(QUANT_LOG_LOSS_KIND), entry.get(QUANT_LOG_NSAMPLES), entry.get(QUANT_LOG_DAMP),
                              entry.get(PROCESS_LOG_TIME)] for entry in self.quant_log])

        pre_quantized_size_mb = get_model_files_size(self.model_local_path)
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        quantizers = [f"{META_QUANTIZER_GPTQMODEL}:{__version__}"]
        if meta_quantizer:
            if len(meta_quantizer.split(":")) == 2:
                quantizers.append(meta_quantizer.replace(" ",""))
            else:
                log.warn(f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'")

        # write gptqmodel tooling fingerprint to config
        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=quantizers
        )


        self.quantize_config.meta_set(
            key=META_FIELD_URI,
            value=META_VALUE_URI,
        )

        # meta: write config fields to meta if they doe not participate in inference
        gptaq_cfg = getattr(self.quantize_config, "gptaq", None)

        foem_cfg = getattr(self.quantize_config, "foem", None)

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_PERCENT,
            value=getattr(self.quantize_config, "damp_percent", None)
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_AUTO_INCREMENT,
            value=getattr(self.quantize_config, "damp_auto_increment", None)
        )

        self.quantize_config.meta_set(
            key=META_FIELD_STATIC_GROUPS,
            value=getattr(self.quantize_config, "static_groups", None)
        )

        self.quantize_config.meta_set(
            key=META_FIELD_TRUE_SEQUENTIAL,
            value=self.quantize_config.true_sequential
        )

        self.quantize_config.meta_set(
            key=META_FIELD_MSE,
            value=getattr(getattr(self.quantize_config, "mse", None), "norm", None)
        )

        self.quantize_config.meta_set(
            key=META_FIELD_GPTAQ_ENABLED,
            value=None if gptaq_cfg is None else {
                "alpha": gptaq_cfg.alpha,
                "device": (
                    gptaq_cfg.device
                    if isinstance(gptaq_cfg.device, str)
                    else str(gptaq_cfg.device)
                ),
            }
        )

        self.quantize_config.meta_set(
            key=META_FIELD_FOEM_ENABLED,
            value=None if foem_cfg is None else {
                "alpha": foem_cfg.alpha,
                "beta": foem_cfg.beta,
                "device": (
                    foem_cfg.device
                    if isinstance(foem_cfg.device, str)
                    else str(foem_cfg.device)
                ),
            }
        )

        self.quantize_config.meta_set(
            key=META_FIELD_ACT_GROUP_AWARE,
            value=getattr(self.quantize_config, "act_group_aware", None)
        )

        # The config, quantize_config and model may be edited in place in save_quantized.
        config = copy.deepcopy(self.model.config)

        quantize_config = copy.deepcopy(self.quantize_config)

        save_state_overlay = _validated_save_state_overlay(self)
        quant_method = getattr(quantize_config, "method", getattr(quantize_config, "quant_method", None))
        runtime_format = resolve_quant_format(quantize_config.format, quant_method)

        # Fail before writing any config when a loaded checkpoint's format was
        # changed to a layout that cannot be converted exactly.
        if self.load_quantized_model:
            resolve_resave_format_conversion(quantize_config)

        checkpoint_passthrough_plan = None

        if runtime_format == FORMAT.GPTQ_V2:
            log.warn(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPT-QModel version >= {MIN_VERSION_WITH_V2}."
            )

        for compat_warning in hf_compat_warnings(
            quantize_config,
            runtime_format,
            model=getattr(self, "model", None),
        ):
            log.warn(f"Model: {compat_warning}")

        if runtime_format == FORMAT.EXL3:
            _validate_exllamav3_publication_modules(
                self,
                validated_overlay=save_state_overlay,
            )
            tensor_storage = _build_exllamav3_tensor_storage_for_save(
                self,
                validated_overlay=save_state_overlay,
            )
            quantize_config.tensor_storage = tensor_storage
            self.quantize_config.tensor_storage = copy.deepcopy(tensor_storage)
            checkpoint_passthrough_plan = _exllamav3_checkpoint_passthrough_plan(
                self,
                tensor_storage,
            )

        # Build the save-time shell in a local. Replacing ``self.model`` would
        # leave the caller's live model pointing at a CPU kernel that returns
        # wrong logits after ``save()``.
        model_for_save = self.model
        if self.load_quantized_model and runtime_format != FORMAT.EXL3:
            model_for_save = self.get_model_with_quantize(
                qcfg=quantize_config,
                model_id_or_path=self.model_local_path,
            )
            live_generation_config = getattr(self.model, "generation_config", None)
            if live_generation_config is not None:
                # The save-time shell derives its generation config from the
                # model config. Preserve the live checkpoint settings instead.
                model_for_save.generation_config = copy.deepcopy(live_generation_config)

        # --- start config save block ---
        # Keep the Transformers config small. EXL3's per-tensor storage map can
        # be tens of megabytes and already has an authoritative home in
        # quantize_config.json. Embedding a second copy in config.json exceeds
        # Hub's config renderer limit and makes ordinary AutoConfig loading
        # needlessly expensive. vLLM/GPTQModel load the complete EXL3 manifest
        # from the external quantization config filename.
        config.quantization_config = _quantization_config_for_model_config(
            quantize_config,
            runtime_format,
        )
        explicit_quantized_modules = _gptq_explicit_quantized_modules(
            quantize_config, model_for_save
        )
        if explicit_quantized_modules is not None:
            # Optimum quantizes exactly this set; without it, dynamic `-:`
            # exclusions load as uninitialized Marlin layers in plain HF.
            config.quantization_config["modules_in_block_to_quantize"] = (
                explicit_quantized_modules
            )
        model_for_save.config = config

        def strip_attention_impl_fields(target: Any) -> Dict[str, Any]:
            removed: Dict[str, Any] = {}
            for attr in ("attn_implementation", "_attn_implementation"):
                if hasattr(target, attr):
                    removed[attr] = getattr(target, attr)
                    # Avoid AttributeError: property '_attn_implementation' of 'Qwen2Config' object has no deleter
                    try:
                        delattr(target, attr)
                    except Exception:
                        pass
            return removed

        generation_config = getattr(model_for_save, "generation_config", None)
        removed_config_attention_attrs: Dict[str, Any] = {}
        removed_generation_attention_attrs: Dict[str, Any] = {}

        try:
            removed_config_attention_attrs = strip_attention_impl_fields(model_for_save.config)
            if generation_config is not None:
                removed_generation_attention_attrs = strip_attention_impl_fields(generation_config)

            # Save model metadata directly. Calling model.save_pretrained with
            # an empty state_dict deletes resumable shards in recent
            # Transformers releases because they are absent from the supplied
            # (empty) checkpoint.
            _save_model_configs_without_weights(
                model_for_save,
                save_dir,
                self.model_local_path,
                trust_remote_code=bool(getattr(self, "trust_remote_code", False)),
            )
        finally:
            for attr, value in removed_config_attention_attrs.items():
                setattr(model_for_save.config, attr, value)
            if generation_config is not None:
                for attr, value in removed_generation_attention_attrs.items():
                    setattr(generation_config, attr, value)

        gen_config_path = os.path.join(save_dir, "generation_config.json")
        if sanitize_generation_config_file(gen_config_path):
            log.info("Model: Sanitized `generation_config.json` before packaging.")

        # Save `quantize_config.json`
        prepare_quantize_config_for_save = getattr(self, "prepare_quantize_config_for_save", None)
        if callable(prepare_quantize_config_for_save):
            prepare_quantize_config_for_save(quantize_config)
        quantize_config.save_pretrained(save_dir)

        if quantize_config.dynamic:
            # Transformers sorts nested keys, but dynamic rules use first-match order.
            quant_config_path = os.path.join(save_dir, "quantize_config.json")
            with open(quant_config_path, "r", encoding="utf-8") as handle:
                saved_quant_config = json.load(handle)
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path, "r", encoding="utf-8") as handle:
                saved_config = json.load(handle)
            saved_config["quantization_config"]["dynamic"] = (
                saved_quant_config["dynamic"]
            )
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(saved_config, indent=2) + "\n")

        log.info("Model: Saved quantized config metadata to `%s`.", save_dir)

        # --- end config save block ---

        offload_root = self.quantize_config.offload_to_disk_path if getattr(self.quantize_config, "offload_to_disk", False) else None

        # Ordinary shell saves need native runtime tensors materialized. Exact
        # checkpoint passthrough instead reads untouched tensors directly under
        # their authoritative source names and therefore deliberately skips
        # this memory-heavy conversion step.
        if not self.load_quantized_model and checkpoint_passthrough_plan is None:
            # A quantized module owns its packed payload, which can share the
            # `weight` leaf with the dense source checkpoint tensor. Aliasing
            # those tensors by name would overwrite the packed codes, so leave
            # every module with a `QUANT_TYPE` to `get_state_dict_for_save`,
            # which restores it from the offload bundle the quantizer wrote.
            quantized_module_types = _quantized_module_types(model_for_save)
            suspend_staging = getattr(
                self.turtle_model, "suspend_active_source_staging", None
            )
            direct_source_scope = (
                suspend_staging() if callable(suspend_staging) else nullcontext()
            )
            with direct_source_scope:
                alias_all_from_turtle_if_meta(
                    shell_model=model_for_save,
                    turtle_model=self.turtle_model,
                    skip_module_types=quantized_module_types,
                )
                materialized_layers = _materialize_meta_layers_from_turtle(
                    model_for_save,
                    self.turtle_model,
                    skip_module_types=quantized_module_types,
                )
                if materialized_layers:
                    log.info(
                        "Model save: materialized %s meta layer modules from turtle source.",
                        materialized_layers,
                    )
                restored_meta = _materialize_remaining_meta_params_from_turtle(
                    model_for_save,
                    self.turtle_model,
                    skip_module_types=quantized_module_types,
                )
                if restored_meta:
                    log.info(
                        "Model save: materialized %s remaining meta params from turtle source.",
                        restored_meta,
                    )
        elif checkpoint_passthrough_plan is not None:
            log.info(
                "Model save: exact native checkpoint passthrough enabled; "
                "skipping native shell materialization."
            )

        if checkpoint_passthrough_plan is not None:
            state_dict = _build_exllamav3_checkpoint_passthrough_state_dict(
                self,
                checkpoint_passthrough_plan,
                offload_root=offload_root,
                validated_overlay=save_state_overlay,
            )
        else:
            state_dict = get_state_dict_for_save(model_for_save, offload_root=offload_root)
        if checkpoint_passthrough_plan is None:
            state_dict.update(out_of_model_state_dict)
            out_of_model_state_dict.clear()
            _apply_save_state_overlay(
                self,
                state_dict,
                validated_overlay=save_state_overlay,
            )
        checkpoint_key_for_save = getattr(self, "checkpoint_key_for_save", None)
        if callable(checkpoint_key_for_save) and checkpoint_passthrough_plan is None:
            state_dict = _convert_checkpoint_state_dict_keys(state_dict, checkpoint_key_for_save)

        # The layer split path writes into several subdirectories, so guard
        # references against the export root before any group can be written.
        _validate_nf12_payload_ownership(self)
        _validate_offload_sources_outside_save_dir(state_dict, save_dir)

        model_base_name = "model"
        model_save_name = model_base_name + ".safetensors"

        if not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            log.warn("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        max_shard_size_bytes = _parse_max_shard_size(max_shard_size)
        metadata_dict = _normalize_metadata(safetensors_metadata)
        metadata_dict["format"] = "pt"
        split_by_mode = _parse_split_by(split_by)

        if split_by_mode == "layer":
            expected_files, tensor_to_filename, total_size_bytes = _stream_state_dict_to_layer_dirs(
                state_dict,
                save_dir=save_dir,
                model_base_name="layer",
                model_save_name="layer.safetensors",
                metadata=metadata_dict,
                max_shard_size=max_shard_size_bytes,
                layer_prefixes=self.extract_layers_node(),
                model=model_for_save,
            )
        else:
            expected_files, tensor_to_filename, total_size_bytes = streaming_state_dict_to_shards(
                state_dict,
                save_dir=save_dir,
                model_base_name=model_base_name,
                single_file_name=model_save_name,
                metadata=metadata_dict,
                max_shard_size=max_shard_size_bytes,
            )
            _cleanup_saved_weight_files(
                save_dir=save_dir,
                expected_files=expected_files,
                model_base_name=model_base_name,
                model_save_name=model_save_name,
            )

        total_size_mb = total_size_bytes / (1024 * 1024)

        if split_by_mode == "layer" or len(expected_files) > 1:
            index = {
                "metadata": {"total_size": total_size_bytes},
                "weight_map": tensor_to_filename,
            }
            index_save_name = model_save_name + ".index.json"
            index_save_path = join(save_dir, index_save_name)
            with open(index_save_path, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
        else:
            index_save_path = join(save_dir, model_save_name + ".index.json")
            if os.path.exists(index_save_path):
                os.remove(index_save_path)

        state_dict.clear()

        # save lora
        if self.quantize_config.adapter:
            _eora_save(self, save_dir=eora_path if eora_path else self.quantize_config.adapter.path, model_save_dir=save_dir)

        # Copy any requested safetensors files without modifying the index
        for tensor_file_name in copy_tensor_files:
            original_tensor_path = os.path.join(self.model_local_path, tensor_file_name)
            if not os.path.exists(original_tensor_path):
                log.warn(
                    f"Model: out_of_model_tensors configured with '{tensor_file_name}', "
                    f"but the file was not found at '{original_tensor_path}'"
                )
                continue

            target_tensor_path = os.path.join(save_dir, tensor_file_name)
            shutil.copy2(original_tensor_path, target_tensor_path)
            log.info(
                f"Model: Copied {tensor_file_name} from original model directory to quantized model directory"
            )

        # If the saved model is a loaded quantized model, do not calculate the size diff.
        if not self.load_quantized_model:
            total_size_gb = total_size_mb / 1024
            size_diff_mb = pre_quantized_size_mb - total_size_mb
            size_diff_gb = size_diff_mb / 1024
            percent_diff = (size_diff_mb / pre_quantized_size_mb) * 100
            log.info(f"Pre-Quantized model size: {pre_quantized_size_mb:.2f}MB, {pre_quantized_size_gb:.2f}GB")
            log.info(f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB")
            log.info(f"Size difference: {size_diff_mb:.2f}MB, {size_diff_gb:.2f}GB - {percent_diff:.2f}%")

        # need to copy .py files for model/tokenizers not yet merged to HF transformers
        if self.trust_remote_code:
            copy_py_files(save_dir, model_id_or_path=self.model_local_path)

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        # Save the processor *after* the tokenizer. `processor.save_pretrained()` re-saves its
        # tokenizer and then writes the processor-level `chat_template.jinja`; saving in the
        # reverse order lets a text-only tokenizer template clobber the audio/vision-aware
        # processor template (e.g. Qwen2-Audio), breaking multimodal chat rendering on reload.
        if hasattr(self, "processor") and isinstance(self.processor, ProcessorMixin):
            self.processor.save_pretrained(save_dir)

        copy_checkpoint_auxiliary_files_for_save = getattr(self, "copy_checkpoint_auxiliary_files_for_save", None)
        if callable(copy_checkpoint_auxiliary_files_for_save):
            copy_checkpoint_auxiliary_files_for_save(save_dir)

        if self.tokenizer:
            # Use source model's tokenizer_class for cross-version compatibility
            # (transformers 5.0 save_pretrained writes "TokenizersBackend" which older versions can't load)
            source_tokenizer_config = get_tokenizer_config(self.model_local_path)
            source_tokenizer_class = source_tokenizer_config.get("tokenizer_class")

            if source_tokenizer_class:
                # fix https://github.com/huggingface/transformers/issues/35832
                # if source tokenizer_class lacks "Fast" suffix but tokenizer is actually fast, add it
                if (not source_tokenizer_class.endswith("Fast")) and isinstance(self.tokenizer.tokenizer, PreTrainedTokenizerFast):
                    source_tokenizer_class = source_tokenizer_class + "Fast"

                saved_tokenizer_config = get_tokenizer_config(save_dir)
                if saved_tokenizer_config.get("tokenizer_class") != source_tokenizer_class:
                    saved_tokenizer_config["tokenizer_class"] = source_tokenizer_class
                    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                        json.dump(saved_tokenizer_config, f, indent=2, ensure_ascii=False)


    cls.save_quantized = save_quantized

    def get_model_with_quantize(self, qcfg, model_id_or_path):

        # Re-saving a loaded quantized model rebuilds the shell here. transformers'
        # ``from_config`` wraps construction in ``local_torch_dtype``, a
        # process-global ``torch.set_default_dtype``; without the shared lock this
        # save-side build races concurrent loads (issue #4204, regression of #4129).
        from .loader import model_load_lock

        config_load_kwargs = dict(getattr(self, "_config_load_kwargs", {}) or {})
        trust_remote_code = resolve_trust_remote_code(
            model_id_or_path, trust_remote_code=self.trust_remote_code, **config_load_kwargs
        )
        config = load_model_config(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            **config_load_kwargs,
        )
        model_dtype = get_hf_config_dtype(config)

        with model_load_lock(), suspend_hf_weight_init():
            model = self.loader.from_config(
                config, dtype=model_dtype, trust_remote_code=trust_remote_code
            )

            # Re-saving a loaded quantized MoE must rebuild the same per-expert
            # module tree the loader builds. ``from_config`` produces the fused
            # 3-D expert parameters (OLMoE, PhiMoE, ...), and ``make_quant`` can
            # only replace ``nn.Linear``, so without defusing here the
            # checkpoint's per-expert quantized tensors have no destination and
            # ``save()`` writes the fused zero shell (issue #4589). Mirror the
            # load path: park the fused tensors on meta, let Defuser build the
            # per-expert linears, then give those shells empty host storage.
            from .loader import (
                _convert_model_with_defuser,
                _materialize_meta_expert_linears,
                _meta_fused_expert_params,
            )

            _meta_fused_expert_params(model)
            _convert_model_with_defuser(type(self), model, cleanup_original=True)
            _materialize_meta_expert_linears(model)

            modules = find_modules(model)
            ignore_modules = [self.lm_head] + self.get_base_modules(model)
            external_inclusions = qcfg.external_module_inclusions()

            # A checkpoint may quantize the input embedding and/or the output
            # head through `dynamic` (requantize / quantize embed endpoints).
            # The loader keeps those modules and `make_quant` swaps in a
            # `TorchQuantEmbeddings`, but this shell only kept `lm_head` when
            # `qcfg.lm_head` was set, so the re-saved checkpoint had no
            # `embed_tokens.weight` / `lm_head.weight` for the dense endpoint
            # and the reload raised (#5207). Detect them exactly like the
            # loader does so both paths cannot drift.
            try:
                input_embeddings = model.get_input_embeddings()
            except NotImplementedError:
                input_embeddings = None
            try:
                output_embeddings = model.get_output_embeddings()
            except NotImplementedError:
                output_embeddings = None
            input_embed_name = get_module_name(model, input_embeddings) if input_embeddings is not None else None
            output_embed_name = get_module_name(model, output_embeddings) if output_embeddings is not None else None
            input_embed_quantized, output_embed_quantized = is_embeddings_module_quantized(
                model_dir=model_id_or_path,
                input_embed_name=input_embed_name,
                output_embed_name=output_embed_name,
            )

            for name in list(modules.keys()):
                if input_embed_quantized and name == input_embed_name:
                    continue
                if output_embed_quantized and name == output_embed_name:
                    continue

                # allow loading of quantized lm_head
                if qcfg.lm_head and name == self.lm_head:
                    continue

                declared = {
                    item.split(":", 1)[0]
                    for sublist in self.simple_layer_modules(config, qcfg)
                    for item in sublist
                }
                layer_prefix = next(
                    (f"{prefix}." for prefix in self.extract_layers_node() if name.startswith(f"{prefix}.")),
                    None,
                )
                relative = name[len(layer_prefix):] if layer_prefix is not None else None
                if relative is not None:
                    layer_index, separator, remainder = relative.partition(".")
                    relative = remainder if separator and layer_index.isdigit() else relative
                relative_match = relative is not None and any(
                    relative == leaf or (relative.startswith(f"{leaf}.") and EXPERT_INDEX_PLACEHOLDER in leaf)
                    for leaf in declared
                )
                inclusion_match = (
                    external_inclusions is None
                    or relative is None
                    or any(
                        relative.endswith(entry) or name.endswith(entry)
                        for entry in external_inclusions
                    )
                )
                if any(
                        name == ignore_module or name.startswith(f"{ignore_module}.")
                        for ignore_module in ignore_modules
                ) or not relative_match or not inclusion_match:
                    # log non-lm-head quantizerd modules only
                    if name is not self.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]
                    continue

                if not self.should_quantize_module(model, name, modules[name], qcfg):
                    log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            make_quant(
                model,
                qcfg=qcfg,
                quant_result=modules,
                backend=BACKEND.AUTO,
                lm_head_name=self.lm_head,
                pack=True,
                # Layout-only selection: see ``_export_shell_device``.
                device=_export_shell_device(qcfg),
                # Register quantized-module biases at the checkpoint/model width
                # so a bf16/fp32 bias cannot overflow an fp16 placeholder
                # before ``load_checkpoint_in_model`` copies it in.
                bias_dtype=model_dtype,
            )

        if qcfg.activation_mode in {"w4afp8", "w4a_nvfp4"}:
            # The native W4A load assigns the raw checkpoint words directly.
            # Declare the source layout first, exactly as the non-W4A branch
            # does: otherwise the export shell labels the source words with its
            # own (target) format and a re-save shifts every zero point by one
            # (issue #4323).
            source_format = getattr(qcfg, "_checkpoint_format_on_disk", None)
            if source_format is None:
                source_format = resolve_quant_format(qcfg.format, qcfg.method)
            if source_format in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
                for module in model.modules():
                    if isinstance(module, GPTQQuantLinear):
                        module._checkpoint_format_on_disk = source_format
                        module.qzero_format(
                            format=1 if source_format == FORMAT.GPTQ else 2
                        )
            _load_native_w4a_for_save(model, self.checkpoint_file_name, activation_mode=qcfg.activation_mode)
            # Validate a requested layout change, but do not apply the v1/v2
            # shift here: ``_load_from_state_dict`` already normalized the live
            # words to this kernel's runtime layout, and ``_save_to_state_dict``
            # converts that layout to the target checkpoint format. Applying
            # ``convert_loaded_qzeros_for_format_change`` would shift a second
            # time.
            resolve_resave_format_conversion(qcfg)
            target_format = resolve_quant_format(qcfg.format, qcfg.method)
            for module in model.modules():
                if isinstance(module, GPTQQuantLinear):
                    module._checkpoint_format_on_disk = target_format
        else:
            # The export shell starts with zero-filled buffers, so its live
            # qzeros layout must be declared before the checkpoint words land.
            # accelerate refills every key a shard omits from these buffers;
            # without the declaration a stale v1 placeholder would relabel the
            # shard's v2 words and convert them again (issue #4166).
            source_format = getattr(qcfg, "_checkpoint_format_on_disk", None)
            if source_format is None:
                source_format = resolve_quant_format(qcfg.format, qcfg.method)
            if source_format in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
                for module in model.modules():
                    if isinstance(module, GPTQQuantLinear):
                        module._checkpoint_format_on_disk = source_format
                        module.qzero_format(
                            format=1 if source_format == FORMAT.GPTQ else 2
                        )
            shell_key_converter = _export_shell_key_converter(self, model)
            load_checkpoint_in_model_then_tie_weights(
                model,
                # The checkpoint may use a legacy layout; fill the shell under
                # its runtime names (issue #4802).
                key_converter=shell_key_converter,
                # Do not cast the export shell. Each quantized module
                # allocates its buffers in the checkpoint's storage dtype
                # (packed FP8 weight, FP16 GPTQ scales, FP32 QQQ
                # ``s_channel``), and accelerate's ``dtype=None`` path copies a
                # checkpoint tensor into that existing dtype. Passing the model
                # dtype narrowed every packed/scale buffer to bf16/fp16, so an
                # FP8 re-save was no longer loadable and GPTQ/QQQ re-saves were
                # lossy (issue #4303).
                dtype=None,
                checkpoint=self.checkpoint_file_name,
                # The shell allocates non-quantized buffers (for example a
                # quantized module's bias) in the model dtype. When a legacy
                # checkpoint stores one at another dtype, restore the
                # checkpoint value instead of republishing the narrowed shell
                # slot (issue #4379).
                restore_checkpoint_buffer_dtypes=True,
                # Load each tensor exactly once. With ``device_map=None``
                # accelerate 1.15 calls ``set_model_state_dict`` once per
                # checkpoint file, so the whole module ``load_state_dict``
                # runs for every shard and a partial shard is completed from
                # the module's own buffers.
                device_map={"": "cpu"},
                # offload_state_dict=True,
                # offload_buffers=True,
            )
            _validate_export_shell_quantized_slots(
                model, self.checkpoint_file_name, key_converter=shell_key_converter,
            )
            resave_conversion = resolve_resave_format_conversion(qcfg)
            if resave_conversion is not None:
                source_format, target_format = resave_conversion
                converted = convert_loaded_qzeros_for_format_change(
                    model, source_format, target_format,
                )
                log.info(
                    "Model: converted qzeros for format change `%s` -> `%s` on %s modules.",
                    source_format,
                    target_format,
                    converted,
                )
            # The rebuilt shell now carries the target layout; do not convert
            # again if its state dict is serialized directly.
            target_format = resolve_quant_format(qcfg.format, qcfg.method)
            for module in model.modules():
                if isinstance(module, GPTQQuantLinear):
                    module._checkpoint_format_on_disk = target_format
        torch_empty_cache()
        return model

    cls.get_model_with_quantize = get_model_with_quantize

    return cls
