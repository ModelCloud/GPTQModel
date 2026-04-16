# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
from os.path import isfile, join
from typing import Any, Dict, List, Optional, Union

import pcre
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, PreTrainedTokenizerFast, ProcessorMixin
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from ..adapter.adapter import HF_ADAPTER_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora
from ..adapter.peft import LoraConfig
from ..quantization.config import (
    FORMAT,
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
    MIN_VERSION_WITH_V2,
    resolve_quant_format,
)
from ..utils.backend import BACKEND
from ..utils.exllamav3 import build_exllamav3_tensor_storage
from ..utils.hf import (
    _normalize_legacy_tied_weights_keys,
    prepare_remote_code_compat,
    sanitize_generation_config_file,
    sanitize_model_config,
    suspend_hf_weight_init,
)
from ..utils.logger import setup_logger
from ..utils.model import (
    TensorSource,
    copy_py_files,
    find_modules,
    get_model_files_size,
    get_module_by_name,
    get_state_dict_for_save,
    load_checkpoint_in_model_then_tie_weights,
    make_quant,
    streaming_state_dict_to_shards,
)
from ..utils.structure import alias_all_from_turtle_if_meta
from ..utils.torch import torch_empty_cache
from ..version import __version__
from ._const import DEFAULT_MAX_SHARD_SIZE, DEVICE


log = setup_logger()

PROCESS_LOG_NAME = "process"
PROCESS_LOG_LAYER = "layer"
PROCESS_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_NSAMPLES = "samples"
QUANT_LOG_DAMP = "damp"
PROCESS_LOG_TIME = "time"
PROCESS_LOG_FWD_TIME = "fwd_time"
PROCESS_USED_MEMORY = "(v)ram"

EORA_DEFAULT_FILE = "eora.safetensors"

# disable gptqmodel split_by layer feature (until sglang pr is merged since our dir struct is not compatible)
# SUPPORTED_SPLIT_BY = {None, "layer"}
SUPPORTED_SPLIT_BY = {None}
_MAX_SHARD_SIZE_RE = pcre.compile(
    r"\s*(\d+)([KMGTP]?B?)\s*",
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



def _resolve_out_of_model_source_files(
    model_local_path: str,
    source_files: Optional[List[str]] = None,
) -> List[str]:
    if source_files:
        return sorted(dict.fromkeys(source_files))

    index_path = join(model_local_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            weight_map = index_data.get("weight_map", {})
            if isinstance(weight_map, dict):
                return sorted(
                    {
                        filename
                        for filename in weight_map.values()
                        if isinstance(filename, str) and filename.endswith(".safetensors")
                    }
                )
        except Exception as exc:
            log.warn(f"Model: Failed to inspect original safetensors index at '{index_path}': {exc}")

    return sorted(
        filename
        for filename in os.listdir(model_local_path)
        if filename.endswith(".safetensors") and isfile(join(model_local_path, filename))
    )


def _load_tensors_by_prefixes(
    model_local_path: str,
    prefixes: List[str],
    source_files: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    # Gather tensors whose names match any of the requested prefixes.
    # Gather tensors whose names match any of the requested prefixes from all available shards.
    tensors: Dict[str, torch.Tensor] = {}
    source_file_names = _resolve_out_of_model_source_files(model_local_path, source_files)
    for source_file_name in source_file_names:
        source_tensor_path = os.path.join(model_local_path, source_file_name)
        if not os.path.exists(source_tensor_path):
            continue
        try:
            with safe_open(source_tensor_path, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    if any(tensor_name.startswith(prefix) for prefix in prefixes):
                        if tensor_name not in tensors:
                            tensors[tensor_name] = f.get_tensor(tensor_name)
        except Exception as exc:
            log.warn(
                f"Model: Failed to read tensors from {source_file_name} while scanning for prefixes "
                f"{prefixes}: {exc}"
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


def _merge_prefix_tensors_into_state_dict(
    prefixes: List[str], model_local_path: str, state_dict: Dict[str, TensorSource]
) -> None:
    # Inject matched tensors into the ongoing state_dict before sharding.
    merged = 0
    normalized_prefixes = [prefix if prefix.endswith(".") else f"{prefix}." for prefix in prefixes]
    tensors = _load_tensors_by_prefixes(model_local_path, normalized_prefixes)
    for name, tensor in tensors.items():
        state_dict[name] = _tensor_source_from_tensor(name, tensor)
        merged += 1
    if merged:
        log.info(f"Model: Merged {merged} tensors with prefixes {normalized_prefixes} into the state dict")
    else:
        log.warn(f"Model: No tensors matched prefixes {normalized_prefixes} while merging into the state dict")


def _normalize_out_of_model_tensors_entries(
    entries: Optional[List[Union[str, Dict[str, Any]]]]
) -> tuple[List[str], List[str]]:
    # Normalize configured files/prefixes into explicit lists.
    copy_files: List[str] = []
    prefixes: List[str] = []
    if not entries:
        return copy_files, prefixes

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

    return copy_files, prefixes


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
                key = key.lower()
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

            # dynamic rank
            rank_pattern = {}
            if self.quantize_config.dynamic:
                rank_pattern = self.quantize_config.extract_adapter_rank_patterns()

            lora_cfg = LoraConfig(base_model_name_or_path=model_save_dir,
                                  r=self.quantize_config.adapter.rank,
                                  lora_alpha=self.quantize_config.adapter.rank,
                                  target_modules=list(target_modules),
                                  rank_pattern=rank_pattern)
            lora_cfg.save_pretrained(save_dir=save_dir)

            log.info(f"Adapter: Saving EoRA weights to -> `{save_dir}`")

            save_file(tensors=weights, filename=weight_file_path, metadata={"format": "pt"})

            del self.lora_results  # TODO REFRACTOR

    cls.eora_save = _eora_save

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
        os.makedirs(save_dir, exist_ok=True)

        if self.quant_log:
            with open(os.path.join(save_dir, "quant_log.csv"), mode='w', newline='') as file:
                w = csv.writer(file)
                w.writerow([PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, QUANT_LOG_DAMP, PROCESS_LOG_TIME])
                w.writerows([[entry.get(PROCESS_LOG_LAYER), entry.get(PROCESS_LOG_MODULE), entry.get(QUANT_LOG_LOSS),
                              entry.get(QUANT_LOG_DAMP), entry.get(PROCESS_LOG_TIME)] for entry in self.quant_log])

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
            value=getattr(self.quantize_config, "mse", None)
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
        sanitize_model_config(self.model.config)
        config = copy.deepcopy(self.model.config)

        quantize_config = copy.deepcopy(self.quantize_config)

        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        runtime_format = resolve_quant_format(quantize_config.format, quantize_config.method)

        if runtime_format == FORMAT.GPTQ_V2:
            log.warn(
                f"Using 'format = {FORMAT.GPTQ_V2}': the serialized model is only supported by GPT-QModel version >= {MIN_VERSION_WITH_V2}."
            )

        if runtime_format == FORMAT.EXL3:
            tensor_storage = build_exllamav3_tensor_storage(self.model)
            quantize_config.tensor_storage = tensor_storage
            self.quantize_config.tensor_storage = copy.deepcopy(tensor_storage)

        if self.load_quantized_model and runtime_format != FORMAT.EXL3:
            self.model = self.get_model_with_quantize(
                qcfg=quantize_config,
                model_id_or_path=self.model_local_path,
            )

        # --- start config save block ---
        # Save quantized config
        config.quantization_config = quantize_config.to_dict()
        self.model.config = config

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

        generation_config = getattr(self.model, "generation_config", None)
        removed_config_attention_attrs: Dict[str, Any] = {}
        removed_generation_attention_attrs: Dict[str, Any] = {}

        try:
            removed_config_attention_attrs = strip_attention_impl_fields(self.model.config)
            if generation_config is not None:
                removed_generation_attention_attrs = strip_attention_impl_fields(generation_config)
            _normalize_legacy_tied_weights_keys(self.model)

            # Save model config, including generation_config
            # Use empty state_dict hack to bypass saving weights
            self.model.save_pretrained(save_dir, state_dict={}, is_main_process=True)
        finally:
            for attr, value in removed_config_attention_attrs.items():
                setattr(self.model.config, attr, value)
            if generation_config is not None:
                for attr, value in removed_generation_attention_attrs.items():
                    setattr(generation_config, attr, value)

        gen_config_path = os.path.join(save_dir, "generation_config.json")
        if sanitize_generation_config_file(gen_config_path):
            log.info("Model: Sanitized `generation_config.json` before packaging.")

        # Save `quantize_config.json`
        quantize_config.save_pretrained(save_dir)

        def debug_saved_config(path):
            # List all files in the directory
            files = os.listdir(path)
            print("Files in directory:")
            for file in files:
                print(file)

            config_file_paths = ["generation_config.json", "config.json"]
            for file_name in config_file_paths:
                full_path = os.path.join(path, file_name)
                if os.path.isfile(full_path):
                    print(f"Content of saved `{file_name}`:")
                    with open(full_path, 'r') as config_file:
                        config_data = json.load(config_file)
                        print(json.dumps(config_data, indent=4))
                else:
                    print(f"`{file_name}` does not exist in the directory.")

        debug_saved_config(save_dir)

        # Save processor related config files. For example: preprocessor_config.json, chat_template.json
        if hasattr(self,"processor") and isinstance(self.processor, ProcessorMixin):
            self.processor.save_pretrained(save_dir)
        # --- end config save block ---

        # Due to shell/turtle state, we need to sync the modules from turtle to shell
        if not self.load_quantized_model:
            alias_all_from_turtle_if_meta(shell_model=self.model, turtle_model=self.turtle_model)

        offload_root = self.quantize_config.offload_to_disk_path if getattr(self.quantize_config, "offload_to_disk", False) else None
        state_dict = get_state_dict_for_save(self.model, offload_root=offload_root)
        copy_tensor_files, prefix_entries = _normalize_out_of_model_tensors_entries(
            getattr(self, "out_of_model_tensors", None)
        )
        if prefix_entries:
            _merge_prefix_tensors_into_state_dict(prefix_entries, self.model_local_path, state_dict)

        model_base_name = "model"
        model_save_name = model_base_name + ".safetensors"

        if not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            log.warn("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        def _parse_max_shard_size(value: Optional[Union[int, str]]) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            match = _MAX_SHARD_SIZE_RE.fullmatch(value)
            if not match:
                raise ValueError(f"Invalid max_shard_size value: {value}")
            base = int(match.group(1))
            suffix = match.group(2).upper()
            multiplier = 1
            if suffix.startswith("K"):
                multiplier = 1024
            elif suffix.startswith("M"):
                multiplier = 1024 ** 2
            elif suffix.startswith("G"):
                multiplier = 1024 ** 3
            elif suffix.startswith("T"):
                multiplier = 1024 ** 4
            elif suffix.startswith("P"):
                multiplier = 1024 ** 5
            return base * multiplier

        def _normalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
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
                model=self.model,
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

        config = AutoConfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
        )
        prepare_remote_code_compat(config)

        with suspend_hf_weight_init():
            model = cls.loader.from_config(
                config, dtype=torch.float16
            )

            modules = find_modules(model)
            ignore_modules = [self.lm_head] + self.get_base_modules(model)

            for name in list(modules.keys()):
                # allow loading of quantized lm_head
                if qcfg.lm_head and name == self.lm_head:
                    continue

                if any(name.startswith(ignore_module) for ignore_module in ignore_modules) or all(
                        not name.endswith(ignore_module) for sublist in self.simple_layer_modules(config, qcfg) for ignore_module in sublist
                ):
                    # log non-lm-head quantizerd modules only
                    if name is not self.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]


            make_quant(
                model,
                qcfg=qcfg,
                quant_result=modules,
                backend=BACKEND.AUTO,
                lm_head_name=cls.lm_head,
                pack=True,
                device=DEVICE.CPU,
            )

        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=torch.float16,
            # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=self.checkpoint_file_name,
            # device_map=device_map,
            # offload_state_dict=True,
            # offload_buffers=True,
        )
        torch_empty_cache()
        return model

    cls.get_model_with_quantize = get_model_with_quantize

    return cls
