# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

import torch
import torch._dynamo
import torch.nn as nn
from tokenicer import Tokenicer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    modeling_utils,
)


try:  # Optional dependency for huggingface datasets support
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - datasets may not be installed
    HFDataset = None
    HFIterableDataset = None

from .. import DEVICE_THREAD_POOL
from ..adapter.adapter import Adapter
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.lookahead import configure_default_lookahead
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, METHOD, QUANTIZE_BLACK_LIST, dynamic_get
from ..quantization.rotation.rotation import fuse_layer_norms, rotate_model
from ..utils.backend import BACKEND
from ..utils.data import collate_data
from ..utils.device import get_device
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import QuantizationRegionTimer, setup_logger
from ..utils.model import MODALITY, find_modules, get_module_by_name_prefix, move_to
from ..utils.offload import offload_to_disk
from ..utils.structure import alias_from_turtle_for_submodule
from ..utils.torch import TORCH_HAS_COMPILE, torch_compile
from ._const import (
    CALIBRATION_DATASET_CONCAT_CHAR,
    CPU,
    DEFAULT_MAX_SHARD_SIZE,
    DEVICE,
    EXPERT_INDEX_PLACEHOLDER,
    META,
)
from .loader import ModelLoader
from .writer import ModelWriter


if TYPE_CHECKING:
    try:
        from datasets import Dataset as HFDatasetType
        from datasets import IterableDataset as HFIterableDatasetType
    except Exception:  # pragma: no cover - optional dependency
        HFDatasetType = HFIterableDatasetType = object


class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


def generate_node_for_awq_scaling(inp, prev_op, module_kwargs, nodes_size, subset, module2inspect):
    n = {
        "prev_op": prev_op,
        "layers": subset,
        "inp": inp,
    }
    if nodes_size == 0:
        # Only the first node needs kwargs
        n["kwargs"] = module_kwargs

    if module2inspect is not None:
        n["module2inspect"] = module2inspect

    return n, None

def check_support_param_buffer_assignment(*args, **kwargs):
    return False


def apply_module_tree_override(module_tree, override):
    """
    Recursively find the corresponding key of override in module_tree and override it.
    """
    if isinstance(module_tree, dict) and isinstance(override, dict):
        for k, v in override.items():
            if k in module_tree and isinstance(module_tree[k], (dict, list)) and isinstance(v, (dict, list)):
                module_tree[k] = apply_module_tree_override(module_tree[k], v)
            else:
                module_tree[k] = v
    elif isinstance(module_tree, list) and isinstance(override, list):
        for o in override:
            if isinstance(o, dict):
                for b in module_tree:
                    if isinstance(b, dict):
                        apply_module_tree_override(b, o)
    return module_tree


NOT_QUANTIZE_FLAG = ":!"


# Fix cpu memory leak.
# See https://github.com/huggingface/transformers/issues/34366
modeling_utils.check_support_param_buffer_assignment = check_support_param_buffer_assignment

log = setup_logger()

class BaseQModel(nn.Module):
    # name of lm_head
    lm_head: str = "lm_head"

    # a tree node of all the roots that contain quantizable modules
    module_tree: List[str] = None
    # Override module_tree according to different QUANT_METHOD
    module_tree_overrides: dict[METHOD, List[str]] = None

    # Strict=True -> all layer_modules must exists in model
    # Some models (deepseek2-lite) dynamically create lora modules based on config.rank
    layer_modules_strict = True

    pre_lm_head_norm_module: str = None

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # list modules where they must match the shape of previous module in execution to consider for scaling optimization
    awq_scale_optimize_shape_dependent_modules: List[str] = None

    # some models require trust_remove_code = True (dbrx_converted)
    require_trust_remote_code = None
    # some models require transformer version(internalm require '<=4.42.2')
    require_pkgs_version: Optional[List[str]] = None
    # some models require a specific dtype, such as float16
    require_dtype: Optional[str|torch.dtype] = None
    require_fast_init: bool = True

    # some models require Processor? For example, Qwen2VLImageProcessor.
    require_load_processor = False

    # TODO: use a better name and what if the value is not at the config root?
    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts
    dynamic_expert_index: Optional[str] = None

    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    loader = AutoModelForCausalLM

    # monkey patch api for trust_remote_code=True models that have broken transformer compat
    require_monkeypatch = False

    # some models have broken attention mask codes so we need to only use batch 1 with no masks
    support_batch_quantize = True

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    supports_desc_act = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    server = None

    support_batch_quantize = True

    ATTENTION_MASKS_DTYPE = torch.bool # default to bool

    ATTENTION_MASKS_REQUIRED_FOR_INPUT: bool = False

    INPUT_EMBEDDING_EXTRA_ARGS = None

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: QuantizeConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: str = None,
        # turtle model is a sympathetic model used to reduce cpu ram usage
        # during quantization stage.
        turtle_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__()

        quant_method = quantize_config.quant_method
        # override module_tree if need
        if self.module_tree_overrides is not None and self.module_tree_overrides.get(quant_method) is not None:
            log.info(f'Module Tree: overridden by METHOD.{quant_method.upper()}')
            # setting cls.module_tree
            type(self).module_tree = apply_module_tree_override(self.module_tree, self.module_tree_overrides[quant_method])

        # record configuration early so model lifecycle hooks can rely on them
        self.compiled = False  # set to True while compile() is triggered successfully
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        self.quantize_config = quantize_config
        self.quant_region_timer = QuantizationRegionTimer(logger=log)
        self._turtle_reload_threshold_bytes = self._resolve_turtle_reload_threshold()
        self._turtle_reload_accum_bytes = 0
        self._turtle_materialized_ids: Set[int] = set()

        self.processor: ProcessorMixin = None

        self.model = self.after_model_load(model, load_quantized_model=load_quantized_model)
        self.turtle_model = turtle_model

        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")
            self.model.tokenizer = self.tokenizer.tokenizer # helpful for CI tests
        else:
            self.tokenizer = tokenizer # TODO none?
            self.model.tokenizer = tokenizer # helpful for CI tests # TODO none?

        # auto-fix model config erors
        if isinstance(self.model, PreTrainedModel):
            autofix_hf_model_config(self.model, path=model_local_path)

        self._turtle_lock = threading.RLock()

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        # stores all per-layer quant stats such as avg loss and processing time
        self.quant_log = []

        if self.require_load_processor:
            self.processor = AutoProcessor.from_pretrained(model_local_path)

        # apply patching of broken trust_remote_code models here
        if self.require_monkeypatch:
            self.monkey_patch()

        # hack: circular import
        from ..adapter.adapter import Lora

        # check adapter load and print info so users knows lora(s) are applied
        if isinstance(self.quantize_config.adapter, Lora):
            loaded_loras = 0
            qmodules = find_modules(self.model, layers=[BaseQuantLinear])
            for name, m in qmodules.items():
                if all(hasattr(m.adapter, name) for name in Lora.parameter_keys()):
                    loaded_loras += 1

            log.info(f"Adapter: `{loaded_loras}` EoRA/Lora adapters loaded for `{len(qmodules)}` modules.")

        # print kernel info:
        log.info(f"Kernel: loaded -> `[{', '.join(cls.__name__ for cls in self.kernels())}]`")

        self._auto_configure_lookahead()

    @classmethod
    def extract_layers_node(cls):
        """
        Given a module_tree structure, return the layers_node string.
        It concatenates everything up to (but not including) the first "#" with '.'.
        Example:
            ["model", "layers", "#", {...}] -> ["model.layers"]
        """

        prefix_parts = []
        for node in cls.module_tree:
            if node == "#":
                break
            if isinstance(node, str):
                prefix_parts.append(node)
            else:
                break  # stop if unexpected nested structure

        return [".".join(prefix_parts)] if prefix_parts else []

    @classmethod
    def build_moe_modules_if_need(cls, model_config, layer_modules, is_awq_quantize: bool = False):
        # MoE models
        if model_config is not None and cls.dynamic_expert_index is not None:
            num_experts = cls.get_num_experts(model_config)

            moe_simple = []
            for names in layer_modules:
                moe_simple.append([])

                has_expert = all(EXPERT_INDEX_PLACEHOLDER in n for n in names)

                if not has_expert:
                    moe_simple[-1].extend(names)
                    continue

                if is_awq_quantize:
                    # AWQ Required
                    # result like: ['mlp.experts.0.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.1.up_proj', ...]
                    for index in range(num_experts):
                        for n in names:
                            moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
                else:
                    # result like: ['mlp.experts.0.gate_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.up_proj', ...]
                    for n in names:
                        for index in range(num_experts):
                            moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))

            return moe_simple

        return layer_modules

    @classmethod
    def get_num_experts(cls, model_config):
        if hasattr(model_config, "text_config"):
            num_experts = getattr(model_config.text_config, cls.dynamic_expert_index)
        elif hasattr(model_config, "thinker_config"):
            num_experts = getattr(model_config.thinker_config.text_config, cls.dynamic_expert_index)
        else:
            num_experts = getattr(model_config, cls.dynamic_expert_index)
        return num_experts

    @classmethod
    def filter_not_quantize_module(cls, layer_modules, quantize_config):
        layer_modules = [
            [name for name in block if NOT_QUANTIZE_FLAG not in name]
            for block in layer_modules
        ]
        layer_modules = [block for block in layer_modules if block]  # 去掉空 block

        if getattr(quantize_config, "dynamic", None):
            new_layer_modules = []
            for modules in layer_modules:
                filtered = [
                    m for m in modules
                    if dynamic_get(quantize_config.dynamic, module_name=m) is not False
                ]
                if filtered:
                    new_layer_modules.append(filtered)
            layer_modules = new_layer_modules

        return layer_modules

    # Inside each `LlamaDecoderLayer` layer are many internal modules
    # List them in the order executed in model forward() code
    # Many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    @classmethod
    def simple_layer_modules(cls, model_config, quantize_config, is_awq_quantize: bool = False):
        layer_modules = cls.build_layer_modules(cls.module_tree)

        layer_modules = cls.build_moe_modules_if_need(model_config, layer_modules, is_awq_quantize)

        layer_modules = cls.filter_not_quantize_module(layer_modules, quantize_config)

        # print(f"simple_layer_modules layer_modules: {layer_modules}")
        return layer_modules

    @classmethod
    def full_layer_modules(cls, model_config=None, is_awq_quantize: bool = False):
        full = cls.build_layer_modules(cls.module_tree)
        full = cls.build_moe_modules_if_need(model_config, full, is_awq_quantize)
        # print(f"full layer_modules: {full}")
        return full

    def prepare_dataset(
        self,
        calibration_dataset: Union[
            List[Dict[str, Union[List[int], torch.LongTensor]]],
            List[str],
            List[List[int]],
            "HFDatasetType",
            "HFIterableDatasetType",
        ],
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        calibration_data_min_length: int = 10,
    ):
        hf_dataset_types: tuple = ()
        if HFDataset is not None:
            hf_dataset_types += (HFDataset,)
        if HFIterableDataset is not None:
            hf_dataset_types += (HFIterableDataset,)

        if isinstance(calibration_dataset, str):
            raise ValueError("Quantize: calibration dataset must be iterable, not a single string.")

        if hf_dataset_types and isinstance(calibration_dataset, hf_dataset_types):
            raw_examples = list(calibration_dataset)
        elif isinstance(calibration_dataset, list):
            raw_examples = calibration_dataset
        elif isinstance(calibration_dataset, Sequence) and not isinstance(calibration_dataset, (bytes, bytearray)):
            raw_examples = list(calibration_dataset)
        else:
            raw_examples = list(calibration_dataset)

        if len(raw_examples) == 0:
            raise ValueError("Quantize: calibration dataset is empty.")

        def _require_tokenizer(reason: str) -> None:
            if self.tokenizer is None:
                raise ValueError(f"tokenizer must be provided when {reason}.")

        def _to_2d_long_tensor(value: Any, name: str, idx: int) -> torch.Tensor:
            try:
                tensor = torch.as_tensor(value, dtype=torch.long)
            except Exception as exc:
                raise ValueError(f"Quantize: failed to convert `{name}` to tensor for calibration item {idx}.") from exc

            if tensor.ndim == 0:
                raise ValueError(f"Quantize: `{name}` for calibration item {idx} must be 1D or 2D, got scalar.")
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(
                    f"Quantize: `{name}` for calibration item {idx} must be rank 1 or 2, got rank {tensor.ndim}."
                )
            return tensor

        def _pack_ids(ids_value: Any, mask_value: Any, idx: int) -> Dict[str, torch.Tensor]:
            ids_tensor = _to_2d_long_tensor(ids_value, "input_ids", idx)

            if mask_value is None:
                mask_tensor = torch.ones_like(ids_tensor, dtype=torch.long)
            else:
                mask_tensor = _to_2d_long_tensor(mask_value, "attention_mask", idx)
                if mask_tensor.shape != ids_tensor.shape:
                    if mask_tensor.numel() == ids_tensor.numel():
                        mask_tensor = mask_tensor.reshape(ids_tensor.shape)
                    else:
                        raise ValueError(
                            f"Quantize: attention_mask shape {tuple(mask_tensor.shape)} does not match input_ids shape "
                            f"{tuple(ids_tensor.shape)} for calibration item {idx}."
                        )

            return {
                "input_ids": ids_tensor.detach(),
                "attention_mask": mask_tensor.detach(),
            }

        def _tokenize_text_value(text_value: Any, idx: int) -> Dict[str, torch.Tensor]:
            _require_tokenizer("calibration data contains raw text")
            tokenized = self.tokenizer(
                text_value,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask")
            return _pack_ids(input_ids, attention_mask, idx)

        def _tokenize_messages_value(messages_value: Any, idx: int) -> Dict[str, torch.Tensor]:
            _require_tokenizer("calibration data uses the `messages` feature")
            apply_fn = getattr(self.tokenizer, "apply_template", None)
            if apply_fn is None:
                raise ValueError("tokenizer must expose `apply_template` to handle `messages` calibration data.")
            try:
                templated = apply_fn(messages_value, tokenize=False)
            except TypeError:
                templated = apply_fn(messages_value)

            if templated is None:
                raise ValueError(f"tokenizer.apply_template returned None for calibration item {idx}.")

            if hasattr(templated, "get"):
                ids_value = templated.get("input_ids")
                mask_value = templated.get("attention_mask")
                text_value = templated.get("text")
                if ids_value is not None:
                    return _pack_ids(ids_value, mask_value, idx)
                if text_value is not None:
                    return _tokenize_text_value(text_value, idx)

            if isinstance(templated, (list, tuple)):
                if len(templated) > 0 and isinstance(templated[0], int):
                    return _pack_ids(list(templated), None, idx)
                raise ValueError(
                    f"tokenizer.apply_template returned an unsupported sequence type for calibration item {idx}."
                )

            if torch.is_tensor(templated):
                return _pack_ids(templated, None, idx)

            if isinstance(templated, str):
                return _tokenize_text_value(templated, idx)

            raise ValueError(
                f"tokenizer.apply_template returned unsupported type {type(templated)} for calibration item {idx}."
            )

        processed_examples: List[Dict[str, torch.Tensor]] = []
        for idx, example in enumerate(raw_examples):
            if isinstance(example, dict):
                if "messages" in example:
                    apply_fn = getattr(self.tokenizer, "apply_template", None) if self.tokenizer else None
                    if apply_fn is None:
                        if "text" in example:
                            processed_examples.append(_tokenize_text_value(example["text"], idx))
                            continue
                        raise ValueError(
                            "tokenizer must expose `apply_template` or calibration data must provide `text` when using `messages`."
                        )
                    processed_examples.append(_tokenize_messages_value(example["messages"], idx))
                    continue
                if "text" in example:
                    processed_examples.append(_tokenize_text_value(example["text"], idx))
                    continue
                if "input_ids" in example:
                    processed_examples.append(_pack_ids(example["input_ids"], example.get("attention_mask"), idx))
                    continue
                raise ValueError(
                    f"Quantize: unsupported calibration example structure at index {idx}: keys={list(example.keys())}"
                )

            if isinstance(example, str):
                processed_examples.append(_tokenize_text_value(example, idx))
                continue

            if isinstance(example, (list, tuple)):
                if all(isinstance(x, int) for x in example):
                    processed_examples.append(_pack_ids(list(example), None, idx))
                    continue
                raise ValueError(
                    f"Quantize: list-based calibration example at index {idx} must contain only integers."
                )

            if torch.is_tensor(example):
                processed_examples.append(_pack_ids(example, None, idx))
                continue

            try:
                processed_examples.append(_pack_ids(example, None, idx))
            except Exception as exc:
                raise ValueError(
                    f"Quantize: unsupported calibration example type {type(example)} at index {idx}."
                ) from exc

        calibration_dataset = processed_examples

        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_calibration_dataset = []
        too_short_calibration_data_count = 0

        max_positions = None
        max_positions_source = None
        trimmed_row_count = 0
        longest_trimmed_row = 0

        def _maybe_resolve_length(value, source_name):
            nonlocal max_positions, max_positions_source
            try:
                if value is None:
                    return False
                limit = int(value)
            except Exception:
                return False
            if limit <= 0:
                return False
            if max_positions is None or limit < max_positions:
                max_positions = limit
                max_positions_source = source_name
            return True

        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            primary_names = ("max_position_embeddings",)
            fallback_names = (
                "max_sequence_length",
                "max_seq_len",
                "n_positions",
                "seq_length",
            )

            for attr_name in primary_names:
                if _maybe_resolve_length(getattr(model_config, attr_name, None), attr_name):
                    break
            if max_positions is None:
                for attr_name in fallback_names:
                    if _maybe_resolve_length(getattr(model_config, attr_name, None), attr_name):
                        break

        for example in calibration_dataset:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])

            if max_positions is not None:
                trimmed = False
                trimmed_input_ids = []
                trimmed_attention_mask = []

                for row_ids, row_mask in zip(input_ids, attention_mask):
                    row_len = len(row_ids)
                    if row_len > max_positions:
                        trimmed = True
                        trimmed_row_count += 1
                        longest_trimmed_row = max(longest_trimmed_row, row_len)
                        trimmed_input_ids.append(row_ids[:max_positions])
                        trimmed_attention_mask.append(row_mask[:max_positions])
                    else:
                        trimmed_input_ids.append(row_ids)
                        trimmed_attention_mask.append(row_mask)

                if trimmed:
                    input_ids = trimmed_input_ids
                    attention_mask = trimmed_attention_mask

            # filter if input_ids is too short
            if len(input_ids[0]) <= calibration_data_min_length:
                too_short_calibration_data_count += 1
                continue

            new_calibration_dataset.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

        if too_short_calibration_data_count > 0:
            log.warn(f"Quantize: {too_short_calibration_data_count} input_ids with length <= {calibration_data_min_length} were removed. "
                     f"Use quantize(calibration_data_min_length={calibration_data_min_length}) to set a custom minimum length.")

        if trimmed_row_count > 0:
            log.info(
                "Quantize: trimmed %s calibration rows above %s=%s (longest original length=%s)",
                trimmed_row_count,
                max_positions_source,
                max_positions,
                longest_trimmed_row,
            )

        if calibration_dataset_concat_size:
            _require_tokenizer("`calibration_dataset_concat_size` is specified")
            concatenated_data = []
            input_ids_buff = []
            attention_mask_buff = []
            current_length = 0

            new_line = self.tokenizer(CALIBRATION_DATASET_CONCAT_CHAR, return_tensors="pt")
            new_line_input_ids = _convert_tensor_to_list(new_line["input_ids"])[0]
            new_line_attention_mask = _convert_tensor_to_list(new_line["attention_mask"])[0]
            new_line_input_ids_len = len(new_line_input_ids)

            for example in new_calibration_dataset:
                input_ids = example["input_ids"][0]
                attention_mask = example["attention_mask"][0]

                if current_length + len(input_ids) + new_line_input_ids_len >= calibration_dataset_concat_size:
                    if len(input_ids_buff) > 0:
                        remaining_space = calibration_dataset_concat_size - current_length
                        # if there is remaining space, add the remaining input to the current block
                        if remaining_space > 0:
                            input_ids_buff.extend(new_line_input_ids)
                            input_ids_buff.extend(input_ids[:remaining_space - new_line_input_ids_len])
                            attention_mask_buff.extend(new_line_attention_mask)
                            attention_mask_buff.extend(attention_mask[:remaining_space - new_line_input_ids_len])

                            concatenated_data.append({
                                "input_ids": [input_ids_buff],
                                "attention_mask": [attention_mask_buff]
                            })
                        else:
                            # if there is no remaining space, add the current block to the concatenated data
                            concatenated_data.append({
                                "input_ids": [input_ids_buff],
                                "attention_mask": [attention_mask_buff]
                            })

                        input_ids_buff = input_ids[:calibration_dataset_concat_size]
                        attention_mask_buff = attention_mask[:calibration_dataset_concat_size]
                        current_length = len(input_ids_buff)
                    else:
                        input_ids_buff = input_ids[:calibration_dataset_concat_size]
                        attention_mask_buff = attention_mask[:calibration_dataset_concat_size]
                        current_length = len(input_ids_buff)
                else:
                    if len(input_ids_buff) > 0:
                        input_ids_buff.extend(new_line_input_ids)
                        attention_mask_buff.extend(new_line_attention_mask)
                        current_length += new_line_input_ids_len

                    input_ids_buff.extend(input_ids)
                    attention_mask_buff.extend(attention_mask)
                    current_length += len(input_ids)


            if input_ids_buff:
                padding_length = calibration_dataset_concat_size - len(input_ids_buff)
                if padding_length > 0:
                    input_ids_buff.extend([self.tokenizer.pad_token_id] * padding_length)
                    attention_mask_buff.extend([0] * padding_length)
                concatenated_data.append({
                    "input_ids": [input_ids_buff],
                    "attention_mask": [attention_mask_buff]
                })

            new_calibration_dataset = concatenated_data

        # Sort or shuffle calibration dataset
        if calibration_dataset_sort == "asc":
            log.info("Calibration: Sort in ascending order by length")
            sorted_dataset = sorted(
                new_calibration_dataset,
                key=lambda item: len(item["input_ids"][0])
            )
        elif calibration_dataset_sort == "desc":
            log.info("Calibration: Sort in descending order by length")
            sorted_dataset = sorted(
                new_calibration_dataset,
                key=lambda item: len(item["input_ids"][0]),
                reverse=True
            )
        elif calibration_dataset_sort == "shuffle":
            log.info("Calibration: Sort by random shuffle")
            sorted_dataset = new_calibration_dataset[:]  # shallow copy
            random.shuffle(sorted_dataset)
        else:
            log.info("Calibration: Native order")
            sorted_dataset = new_calibration_dataset  # fallback: no sort

        if self.support_batch_quantize:
            new_calibration_dataset_batched = [
                collate_data(sorted_dataset[start: start + batch_size], self.tokenizer.pad_token_id)
                for start in range(0, len(sorted_dataset), batch_size)
            ]

            # total tokens counters
            total_padded = 0
            total_non_padded = 0

            for batch in new_calibration_dataset_batched:
                # attention_mask is shape [batch_size, seq_len]
                mask = batch["attention_mask"]

                # count where mask == 0 (padded tokens)
                total_padded += (mask == 0).sum().item()

                # count where mask == 1 (non-padded tokens)
                total_non_padded += (mask == 1).sum().item()

            log.info(f"Calibration: Total padded tokens: {total_padded}")
            log.info(f"Calibration: Total non-padded tokens: {total_non_padded}")
            log.info(f"Calibration: Total tokens: {total_non_padded + total_padded}")
        else:
            new_calibration_dataset_batched = [
                {
                    "input_ids": torch.tensor(block["input_ids"], dtype=torch.long),
                }
                for block in sorted_dataset
            ]

        return new_calibration_dataset_batched

    def quantize(
        self,
        calibration: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = "desc",  # valid values are asc, desc, shuffle
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter = None,
        adapter_calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]] = None,
        # minimum length of calibration data, default is 10
        calibration_data_min_length: int = 10,
    ) -> Dict[str, List[Dict[str, str]]]:
        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.reset()

        self._turtle_reload_accum_bytes = 0
        self._turtle_materialized_ids = set()

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}"
            )

        if not self.support_batch_quantize:
            log.warn("Quantize: batch_size overridden by model class definition to `disabled`")
            batch_size = 1 # but actually disabled

        if self.quantize_config.format == FORMAT.MARLIN:
            raise ValueError(
                "FORMAT.MARLIN is deprecated for quantization. Please switch to FORMAT.GPTQ. GPTQMOdel will auto-use Marlin kernel for accelerated inference for FORMAT.GPTQ."
            )

        if self.quantize_config.quant_method == METHOD.AWQ:
            if self.quantize_config.format == FORMAT.GEMV_FAST:
                # AWQ GEMV_FAST only supports pack_dtype is torch.int16
                log.info("Quantize Model: Auto fix `pack_dtype` to `torch.int16`")
                self.quantize_config.pack_dtype = torch.int16
            elif self.quantize_config.format == FORMAT.MARLIN:
                # AWQ MARLIN only supports zero_point is false
                log.info("Quantize Model: Auto fix `zero_point` to `False`")
                self.quantize_config.zero_point = False

        if self.support_batch_quantize is False:
            batch_size = 1
            log.warn("Batch quantization is not supported for this model. Setting batch_size to 1.")

        requested_backend = backend
        if isinstance(requested_backend, str):
            requested_backend = BACKEND(requested_backend.lower())

        preferred_backend = requested_backend
        if preferred_backend in (None, BACKEND.AUTO):
            preferred_backend = BACKEND.TORCH

        # Validate quant linear before quantization starts
        _ = select_quant_linear(
            bits=self.quantize_config.bits,
            dynamic=self.quantize_config.dynamic,
            group_size=self.quantize_config.group_size,
            desc_act=self.quantize_config.desc_act,
            sym=self.quantize_config.sym,
            backend=preferred_backend,
            format=self.quantize_config.format,
            quant_method=self.quantize_config.quant_method,
            device=DEVICE(self.quantize_config.device),
            pack=True,
            pack_dtype=self.quantize_config.pack_dtype,
        )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        if self.quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        # overwrite quantize_config.adapter
        if adapter is not None:
            self.quantize_config.adapter = adapter

        from ..adapter.adapter import Lora
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper

        # has lora process
        needs_lora = isinstance(self.quantize_config.adapter, Lora)

        args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration": calibration,
            "prepare_dataset_func": self.prepare_dataset,
            "calibration_concat_size": calibration_concat_size,
            "calibration_sort": calibration_sort,
            "batch_size": batch_size,
            "calculate_w_wq_diff": needs_lora,  # lora needs original w - wq delta
        }

        self.qlinear_kernel = select_quant_linear(
                bits=self.quantize_config.bits,
                group_size=self.quantize_config.group_size,
                desc_act=self.quantize_config.desc_act,
                sym=self.quantize_config.sym,
                pack=True,
                dynamic=self.quantize_config.dynamic,
                device=self.quantize_config.device,
                pack_dtype=self.quantize_config.pack_dtype,
                multi_select=False,
                backend=preferred_backend,
                format=self.quantize_config.format,
                quant_method=self.quantize_config.quant_method,
            )

        # rotate model
        if self.quantize_config.rotation:
            from gptqmodel.models.definitions.llama import LlamaQModel
            from gptqmodel.models.definitions.qwen2 import Qwen2QModel
            if not isinstance(self, (LlamaQModel, Qwen2QModel)):
                raise ValueError(f"rotation only supports: llama/qwen2 model, "
                                    f"current model is {self.__class__.__name__}")

            if self.model.config.tie_word_embeddings:
                log.info("Rotation requires word embeddings to be untied. Untying.")
                self.model.config.tie_word_embeddings = False
                lm_head, _ = get_module_by_name_prefix(self.model, self.lm_head)
                lm_head.weight = nn.Parameter(lm_head.weight.data.clone())

            module_name_args = {
                "layers_node": self.extract_layers_node(),
                "lm_head_name": self.lm_head
            }
            self.model = fuse_layer_norms(model=self.model,
                                            pre_lm_head_norm_module_name=self.pre_lm_head_norm_module,
                                            **module_name_args)

            # MPS does not support float64.
            rotation_device = self.quantize_config.device if self.quantize_config.device != DEVICE.MPS else DEVICE.CPU
            self.model, _ = rotate_model(model=self.model, rotate_mode=self.quantize_config.rotation,
                                            device=rotation_device, **module_name_args)

        # init processor with default GPTQ processor
        from ..looper.tensorparallel_weight_processor import TensorParallelWeightProcessor

        if self.quantize_config.quant_method == METHOD.QQQ:
            from ..looper.qqq_processor import QQQProcessor

            quantize_processor = [
                TensorParallelWeightProcessor(**args),
                QQQProcessor(**args),
            ]
        elif self.quantize_config.quant_method == METHOD.AWQ:
            from ..looper.awq_processor import AWQProcessor

            os.environ["AWQ_BATCH_SIZE"] = str(batch_size)

            # if self.model.config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
            #     raise TypeError(f"{self.model.config.model_type} isn't supported yet.")

            awq_args = dict(args)
            awq_args["gptq_model"] = self
            awq_args["model"] = self.model
            awq_args["batch_size"] = batch_size

            quantize_processor = [
                TensorParallelWeightProcessor(**args),
                AWQProcessor(**awq_args),
            ]
        else:
            from ..looper.gptq_processor import GPTQProcessor

            quantize_processor = [
                TensorParallelWeightProcessor(**args),
                GPTQProcessor(**args),
            ]

        if self.quantize_config.v2 is True:
            from ..looper.native_processor import NativeProcessor

            # During the deepcopy process, self.prepare_dataset will be deeply copied along with self. However,
            # self has a threading.RLock() , which is not serializable.
            args_to_copy = {k: v for k, v in args.items() if k != "prepare_dataset_func"}
            args_clone = copy.deepcopy(args_to_copy)
            args_clone["prepare_dataset_func"] = args["prepare_dataset_func"]

            args_clone.pop("calculate_w_wq_diff", None)
            quantize_processor.insert(0, NativeProcessor(**args_clone))

        processors = quantize_processor
        # Append EoRA processor for lora adapter
        if needs_lora:
            processors.append(
                EoraProcessor(
                    tokenizer=self.tokenizer,
                    qcfg=self.quantize_config,
                    calibration=adapter_calibration_dataset if adapter_calibration_dataset is not None else calibration,
                    prepare_dataset_func=self.prepare_dataset,
                    calibration_concat_size=calibration_concat_size,
                    calibration_sort=calibration_sort,
                    batch_size=batch_size,
                )
            )

        # prepare processor worker (looper)
        module_looper = ModuleLooper(self, processors=processors)

        result = module_looper.loop(
            backend=backend,
            fail_safe=self.quantize_config.fail_safe,
        )

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.flush()

        return result

    def _eora_generate(
        self,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter,
        quantized_modules: Dict[str, TorchQuantLinear],
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        if self.quantized:
            raise EnvironmentError("eora_generate() is called a model that is already quantized")

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        from ..adapter.adapter import Lora
        from ..looper.dequantize_processor import DequantizeProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper
        from ..looper.tensorparallel_weight_processor import TensorParallelWeightProcessor

        self.quantize_config.adapter = adapter

        assert isinstance(self.quantize_config.adapter, Lora)

        # init processor with EoRA processor
        processors = [
            TensorParallelWeightProcessor(
                tokenizer=self.tokenizer,
                qcfg=self.quantize_config,
                calibration=calibration_dataset,
                prepare_dataset_func=self.prepare_dataset,
                calibration_concat_size=calibration_dataset_concat_size,
                calibration_sort=calibration_dataset_sort,
                batch_size=batch_size,
            ),
            DequantizeProcessor(
                quantized_modules=quantized_modules,
            ),
            EoraProcessor(
                tokenizer=self.tokenizer,
                qcfg=self.quantize_config,
                calibration=calibration_dataset,
                prepare_dataset_func=self.prepare_dataset,
                calibration_concat_size=calibration_dataset_concat_size,
                calibration_sort=calibration_dataset_sort,
                batch_size=batch_size,
            ),
        ]

        # prepare processor worker (looper)
        module_looper = ModuleLooper(model=self, processors=processors)

        module_looper.loop()

        self.eora_save(save_dir=adapter.path, model_save_dir=self.model_local_path)
        return

    def to(self, device: Union[str, torch.device]):
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            return self
        else:
            raise f"{self.model.__class__.__name__} does not support the to() method"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, inputs=None, **kwargs):
        with torch.inference_mode():
            # fix hf generate not applying correct pad token
            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                if self.tokenizer is None:
                    raise ValueError("You passed in an `input` to `generate()` of type `str` but model is missing `model.tokenizer`. Please set `model.tokenizer = my_tokenizer`.")
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # placeholder, noop, and alert users to correct static api
    def push_to_hub(self,
                    repo_id: str,
                    quantized_path: str,  # saved local directory path
                    private: bool = False,
                    exists_ok: bool = False,  # set to true if repo already exists
                    token: Optional[str] = None):

        log.error("`push_to_hub()` api cannot be used on the model instance. Please use `GPTQModel.push_to_hub()` static api instead.")

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            **kwargs,
    ):
        timer = getattr(self, "quant_region_timer", None)
        start_time = time.perf_counter() if timer else None

        try:
            if self.quantized:
                # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
                #untie_weights(self.model)

                self.save_quantized(
                    save_dir=save_dir,
                    safetensors_metadata=safetensors_metadata,
                    max_shard_size=max_shard_size,
                    meta_quantizer=meta_quantizer,
                    eora_path=eora_path)

                # overwrite quant_override_files
                for name, value in self.quant_override_files.items():
                    json_path = os.path.join(save_dir, name)
                    with open(json_path, "w", encoding="utf-8") as f:
                        if isinstance(value, str):
                            f.write(value)
                        else:
                            f.write(json.dumps(value))
            else:
                self.save_pretrained(save_dir=save_dir, **kwargs)
        finally:
            if timer is not None and start_time is not None:
                try:
                    target = os.path.abspath(save_dir)
                except (TypeError, ValueError, OSError):
                    target = str(save_dir)
                timer.record(
                    "model_save",
                    time.perf_counter() - start_time,
                    source=target,
                )
                timer.flush()


    # returns all the loaded qlinear types, returns empty [] if non-found
    def kernels(self) -> List[Type[BaseQuantLinear]]:
        if not isinstance(self.model, nn.Module):
            return []
        loaded_kernels = set()
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for k, v in modules.items():
            loaded_kernels.add(v.__class__)

        return list(loaded_kernels)

    def _auto_configure_lookahead(self) -> None:
        if not isinstance(self.model, nn.Module):
            return

        quant_modules = [module for module in self.model.modules() if isinstance(module, TorchQuantLinear)]
        if not quant_modules:
            return

        if not any(getattr(module, "_lookahead_enabled", False) for module in quant_modules):
            return

        configure_default_lookahead(self.model)

    def compile(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        log.warn("Deprecation: `model.compile()` is deprecated. Please use `model.optimize()` instead.")
        return self.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if not self.quantized:
            log.warn("model is not quantized, skip compiling...")
            return self

        if TORCH_HAS_COMPILE:
            self.compiled = False
            log.warn("To use compile(), you need to have torch version >= 2.6.0, please "
                           "upgrade it by `pip install -U torch torchaudio torchvision`")
            return self

        # needed by eora
        # torch._dynamo.config.capture_scalar_outputs = True

        log.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for name in modules.keys():
            modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # supress errors until PyTorch fixed: https://github.com/pytorch/pytorch/issues/132635
        # torch._dynamo.config.suppress_errors = True
        log.info(f"Compiling model with backend: `{backend}`, mode: `{mode}`")

        self.model = torch_compile(self.model, fullgraph=fullgraph, backend=backend, mode=mode)

        #trigger kernel compilation hooks
        # if self.compiled:
        #     modules = find_modules(self.model, layers=[BaseQuantLinear])
        #     for name in modules.keys():
        #         modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # logger.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        # modules = find_modules(self.model, layers=[BaseQuantLinear])
        # for name in modules.keys():
        #     modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        return self

    def serve(self,
               host: str = "0.0.0.0",
               port: int = 80,
               async_mode: bool = False):
        from ..utils.openai_server import OpenAiServer
        self.server = OpenAiServer(model=self)
        self.server.start(host=host, port=port, async_mode=async_mode)

    def serve_shutdown(self):
        if self.server is not None:
            self.server.shutdown()

    def serve_wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        if self.server is not None:
            self.server.wait_until_ready(timeout=timeout, check_interval=check_interval)

    def before_model_load(self, load_quantized_model):
        pass

    def after_model_load(self, model, load_quantized_model):
        return model

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            offload_to_disk(model=self.model, module=self.get_base_modules(model=self.model), disk_path=self.quantize_config.offload_to_disk_path)

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.pre_lm_head_norm_module:
            norm, _ = get_module_by_name_prefix(self.model, [self.pre_lm_head_norm_module])
            norm = self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def pre_quantize(self, module: nn.Module) -> nn.Module:
        if get_device(module) == META:
            return self.shell_module_materialize(
                target_submodule=module,
                device=self.quantize_config.device,
            )
        elif get_device(module) == CPU and self.quantize_config.device != CPU:
            return move_to(module, device=self.quantize_config.device)
        else:
            return module

    def post_quantize(self, module: nn.Module) -> nn.Module:
        #return self.offload_to_disk(module=module)
        return move_to(module, device=CPU)

    def move_embed(self, device: str):
        for embed_module_name in self.get_base_modules(self.model):
            embed_module, _ = get_module_by_name_prefix(self.model, embed_module_name)
            if embed_module is not None:
                self.shell_module_materialize(
                    target_submodule=embed_module,
                    device=device,
                )

    def awq_skip_modules_for_scaling(self) -> bool:
        pass

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = []
        last_module = None  # most recent norm obj (from a '!...' block)
        last_module_name = None
        last_module_root = None  # self_attn.* has root == self_attn, mlp.* has root == mlp

        num_experts = None
        if self.model.config is not None and self.dynamic_expert_index is not None:
            num_experts = self.get_num_experts(self.model.config)

        def strip_not_quantize_flag(module_name):
            if NOT_QUANTIZE_FLAG in module_name:
                return module_name[:module_name.find(NOT_QUANTIZE_FLAG)]
            else:
                return module_name

        for i, block in enumerate(self.full_layer_modules(self.model.config, is_awq_quantize=True)):
            not_quantized = all(NOT_QUANTIZE_FLAG in name for name in block)
            if not_quantized:
                # Remember the latest norm (use the last entry if multiple are present)
                last_module_name = strip_not_quantize_flag(block[-1])
                last_module, _ = get_module_by_name_prefix(module, last_module_name)
                continue

            if num_experts is not None and len(block) == num_experts and last_module is not None and last_module_name is not None:
                # mlp.experts.0.down_proj
                target_suffix = last_module_name.split(".")[-1]
                for name in block:
                    prev_op_name = ".".join(name.split(".")[:-1] + [target_suffix])
                    prev_op, _ = get_module_by_name_prefix(module, prev_op_name)
                    assert prev_op is not None

                    m, _ = get_module_by_name_prefix(module, name)
                    subset = [m]
                    n, root = generate_node_for_awq_scaling(inp=input_feat[name], prev_op=prev_op,
                                                            module_kwargs=module_kwargs, nodes_size=len(nodes),
                                                            subset=subset, module2inspect=None)
                    if root is not None and last_module_root != root:
                        last_module_root = root

                    nodes.append(n)
            else:
                # Normal execution subset
                subset = []
                skip = False
                for name in block:
                    if NOT_QUANTIZE_FLAG not in name:
                        if name == "mlp.gate":
                            log.debug(f'"{name}" skipped.')
                            skip = True

                        m, _ = get_module_by_name_prefix(module, name)
                        # If the Model uses GQA (Grouped Query Attention), attention out will be skipped.
                        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                        if (self.awq_scale_optimize_shape_dependent_modules is not None
                                and name in self.awq_scale_optimize_shape_dependent_modules
                                and isinstance(last_module, nn.Linear)
                                and last_module.weight.shape != m.weight.shape):
                            # log.debug(f'"{name}" attention out skipped.')
                            skip = True

                        subset.append(m)

                if skip:
                    continue

                assert len(subset) > 0
                prev_op = last_module
                assert prev_op is not None

                root_split = block[0].split(".")
                module2inspect = None
                if len(root_split) >= 2:
                    root = root_split[0]
                    if root != last_module_root:
                        last_module_root = root
                        module2inspect, _ = get_module_by_name_prefix(module, root)

                if num_experts is not None and len(block) == 2 * num_experts and module2inspect is not None:
                    inp = input_feat[last_module_root]
                else:
                    inp = input_feat[block[0]]

                n, root = generate_node_for_awq_scaling(inp=inp, prev_op=prev_op,
                                                        module_kwargs=module_kwargs, nodes_size=len(nodes),
                                                        subset=subset, module2inspect=module2inspect)

                nodes.append(n)

            # Update tracker to the LAST item of this block
            last_module_name = strip_not_quantize_flag(block[-1])
            last_module, _ = get_module_by_name_prefix(module, last_module_name)

        import torch
        def format_nodes(nodes):
            out = []
            for n in nodes:
                entry = {}
                for k, v in n.items():
                    if isinstance(v, torch.Tensor):
                        entry[k] = f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
                    elif isinstance(v, dict):
                        entry[k] = [
                            f"Key: {kk}, Value: Tensor(shape={tuple(x.shape)}, dtype={x.dtype}); " if isinstance(x,
                                                                                                                 torch.Tensor) else type(
                                x).__name__
                            for kk, x in v.items()
                        ]
                    else:
                        entry[k] = v
                out.append(entry)
            return out

        # print("DEBUG AWQ NODES:", format_nodes(nodes))
        return nodes

    def _clone_model_init_kwargs(self, source: PreTrainedModel) -> Dict[str, Any]:
        kwargs = getattr(source, "_model_init_kwargs", {}) or {}
        if isinstance(kwargs, dict):
            return dict(kwargs)
        return copy.deepcopy(kwargs)

    def _resolve_turtle_reload_threshold(self) -> int:
        if not getattr(self.quantize_config, "offload_to_disk", False):
            return 0

        default_bytes = 512 * 1024 ** 3 #512MB
        raw = os.getenv("GPTQMODEL_RELOAD_THRESHOLD")
        if raw is None or raw.strip() == "":
            return default_bytes

        value = raw.strip().lower()
        if value in {"0", "off", "disable", "disabled", "none"}:
            return 0

        units = {
            "b": 1,
            "kb": 1024,
            "mb": 1024 ** 2,
            "gb": 1024 ** 3,
            "tb": 1024 ** 4,
        }

        match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", value)
        if match is None:
            log.warn(
                "GPTQMODEL_RELOAD_THRESHOLD value `%s` is invalid; defaulting to 512MB.",
                raw,
            )
            return default_bytes

        amount = float(match.group(1))
        unit = match.group(2) or "b"
        multiplier = units.get(unit, None)
        if multiplier is None:
            log.warn(
                "GPTQMODEL_RELOAD_THRESHOLD unit `%s` is unsupported; defaulting to bytes.",
                unit,
            )
            multiplier = 1

        threshold = int(amount * multiplier)
        if threshold < 0:
            threshold = 0
        return threshold

    def _estimate_module_bytes(self, module: nn.Module) -> int:
        if module is None:
            return 0

        total = 0
        seen: Set[int] = set()
        tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type == "meta":
                continue
            try:
                ptr = tensor.data_ptr()
            except (RuntimeError, AssertionError):
                ptr = None
            if ptr is not None:
                if ptr in seen:
                    continue
                seen.add(ptr)
            total += tensor.numel() * tensor.element_size()
        return total

    def _maybe_auto_reload_after_alias(
        self,
        module: nn.Module,
        target_submodule: nn.Module,
    ) -> None:
        if self.turtle_model is None:
            return

        threshold = self._turtle_reload_threshold_bytes
        if threshold <= 0:
            return

        module_id = id(module)
        if module_id in self._turtle_materialized_ids:
            return

        bytes_added = self._estimate_module_bytes(module)
        self._turtle_materialized_ids.add(module_id)

        if bytes_added <= 0:
            return

        self._turtle_reload_accum_bytes += bytes_added

        if self._turtle_reload_accum_bytes >= threshold:
            label = (
                getattr(target_submodule, "full_name", None)
                or getattr(target_submodule, "name", None)
                or getattr(module, "full_name", None)
                or module.__class__.__name__
            )
            self.reload_turtle_model(source=f"auto:{label}")
            self._turtle_reload_accum_bytes = 0

    def reload_turtle_model(self, *, source: Optional[str] = None) -> None:
        if self.quantize_config.offload_to_disk is False:
            return

        timer = getattr(self, "quant_region_timer", None)
        timing_ctx = timer.measure("model_reload", source=source) if timer else nullcontext()

        with timing_ctx:
            def _do_reload():
                with self._turtle_lock:
                    turtle_model = self.turtle_model
                    model_local_path = self.model_local_path
                    loader = self.loader

                    assert turtle_model is not None and model_local_path is not None

                    reload_kwargs = self._clone_model_init_kwargs(turtle_model)
                    config = turtle_model.config
                    del turtle_model

                    new_model = loader.from_pretrained(
                        model_local_path,
                        config=config,
                        low_cpu_mem_usage=True,
                        **reload_kwargs,
                    )
                    new_model._model_init_kwargs = reload_kwargs
                    new_model.eval()
                    self.turtle_model = new_model
                    self._turtle_reload_accum_bytes = 0
            reload_spinner = log.spinner(title="Turtle model reloading...", interval=0.1)
            try:
                DEVICE_THREAD_POOL.submit("model_loader:cpu", _do_reload).result()
            finally:
                reload_spinner.close()

    # transfer actually materizlied module from turtle (real) to shell
    def shell_module_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: torch.device,
            non_blocking: bool = False,
    ) -> torch.nn.Module:
        with self._turtle_lock:
            turtle_model = self.turtle_model

            if turtle_model is None:
                if get_device(target_submodule) != device:
                    target_submodule.to(device)

                return target_submodule

            module = alias_from_turtle_for_submodule(
                target_model=self.model,
                turtle_model=turtle_model,
                target_submodule=target_submodule,
                device=device,
            )
        self._maybe_auto_reload_after_alias(module, target_submodule)
        return module

    ## overrides nn.module.train()
    # def train(self, mode=True):
    #     old_mode = self.training
    #     # Call the parent class's train() method to set the training mode
    #     super().train(mode)
    #
    #     if old_mode == mode:
    #         return
    #
    #     # Custom behavior when switching to training mode
    #     if mode:
    #         if not self.SUPPORTS_TRAINING:
    #             err = f"{self.__class__.__name__}: MODEL switching to training mode."
    #             log.error(err)
    #             raise NotImplementedError(err)
    #         else:
    #             log.info(f"{self.__class__.__name__}: MODEL switching to training mode.")
    #     else:
    #         log.info(f"{self.__class__.__name__}: `MODEL switching to eval mode.")
    @classmethod
    def build_layer_modules(cls, tree):
        """
        tree format:
          [<model_name>, <submodule>, "#", { parent_module: ( "child[:!][:grp]", ... ), ... }]
        Rules:
          - ':!' means participates in inference but is NOT quantized; keep this marker in output.
          - ':<digit>' means grouping; children with the same group id are emitted in the same block.
          - Both can appear together, e.g. 'module_name:!:2'.
          - Supports nested dict structures for MoE models with experts.
          - Special key "#" in nested dicts means direct children under parent (no additional nesting).
          - EXPERT_INDEX_PLACEHOLDER in keys will be handled by simple_layer_modules for MoE expansion.
        Output:
          _layer_modules = [ [items...], [items...], ... ]
        """
        mapping = None
        for item in tree:
            if isinstance(item, dict):
                mapping = item
                break
        if mapping is None:
            raise ValueError("Mapping configuration not found in the tree.")

        out_blocks = []

        def process_entries(parent, entries, parent_group_offset=0):
            """Process entries recursively to handle nested dict structures for MoE"""
            groups = defaultdict(list)

            # Handle tuple/list of strings (traditional format)
            if isinstance(entries, (tuple, list)):
                for ent in entries:
                    parts = ent.split(':')
                    child = parts[0]

                    flags = parts[1:]
                    has_bang = ('!' in flags)
                    # first numeric tag is the group id; default 0
                    grp = next((int(p) for p in flags if p.isdigit()), 0)
                    # Apply parent group offset to avoid conflicts between different nesting levels
                    grp += parent_group_offset

                    # Store the full path including parent for later use
                    # Special case: if parent ends with the same name as child, don't duplicate
                    if parent.endswith(f".{child}"):
                        full_path = parent
                    else:
                        full_path = f"{parent}.{child}" if parent != child else child
                    groups[grp].append((full_path, has_bang))

            # Handle nested dict structure (MoE format)
            elif isinstance(entries, dict):
                # Calculate max group number used at current level to avoid conflicts
                max_current_group = 0
                for sub_parent, sub_entries in entries.items():
                    if isinstance(sub_entries, (tuple, list)):
                        for ent in sub_entries:
                            parts = ent.split(':')
                            flags = parts[1:]
                            grp = next((int(p) for p in flags if p.isdigit()), 0)
                            max_current_group = max(max_current_group, grp)

                # Process nested entries with appropriate group offset
                current_offset = parent_group_offset
                for sub_parent, sub_entries in entries.items():
                    if sub_parent == "#":
                        # Special case: "#" means expert index placeholder
                        # Create a template path that will be expanded later by simple_layer_modules
                        template_parent = f"{parent}.{EXPERT_INDEX_PLACEHOLDER}"
                        # Use a higher offset for expert modules to avoid conflicts with parent level
                        expert_offset = current_offset + max_current_group + 100  # Large offset to avoid conflicts

                        # Handle special case where sub_entries is ("#",) or "#" - this means use the parent path directly
                        if (isinstance(sub_entries, (tuple, list)) and len(sub_entries) == 1 and sub_entries[0] == "#") or sub_entries == "#":
                            # For ("#",) or "#" format, use the template_parent directly with default group 0
                            groups[expert_offset].append((template_parent, False))
                        else:
                            sub_groups = process_entries(template_parent, sub_entries, expert_offset)
                            for grp, items in sub_groups.items():
                                groups[grp].extend(items)
                    else:
                        # Nested structure: process recursively with full path
                        # Special case: empty string key means use parent path directly
                        if sub_parent == "":
                            full_sub_parent = parent
                        else:
                            full_sub_parent = f"{parent}.{sub_parent}"
                        sub_groups = process_entries(full_sub_parent, sub_entries, current_offset)
                        for grp, items in sub_groups.items():
                            groups[grp].extend(items)
                        # Update offset for next sibling to avoid conflicts
                        if sub_groups:
                            current_offset = max(sub_groups.keys()) + 1

            return groups

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)

            # Emit per-group, skipping pure-:! blocks (norm-only), but
            # preserving :! markers on mixed blocks if they ever occur.
            for g in sorted(groups):
                items = groups[g]
                # if every entry is :!, skip this block (matches your expected output)
                # if all(has_bang for _, has_bang in items):
                #     continue

                block = []
                for full_path, has_bang in items:
                    # The full path is already constructed in process_entries
                    if has_bang:
                        full_path += NOT_QUANTIZE_FLAG
                    block.append(full_path)

                out_blocks.append(block)

        return out_blocks

    @classmethod
    def get_base_modules(cls, model):
        """
        Return list of base modules directly under 'model' but not 'model.layers'.
        """
        # Find the index of "#"
        tree = cls.module_tree
        try:
            sharp_idx = tree.index("#")
        except ValueError:
            raise ValueError("module_tree must contain '#' to separate hierarchy")

        assert sharp_idx > 0, "failed to get_base_modules"
        # root_path = ["model"] or ["model", "language_model"]
        root_path = tree[:sharp_idx-1]

        out = []
        # Traverse each layer in root_path
        for i in range(len(root_path)):
            path = root_path[:i + 1]
            base = model
            exclude = tree[len(path)]

            for node in path:
                base = getattr(base, node)

            for name, _ in base.named_children():
                if name != exclude:
                    out.append(".".join(path + [name]))

        # print(f"Base Modules: {out}")
        return out

    def generate_layers_modules_tree_simple(self, node):
        """
        Recursively walk a nested list/dict structure and:
          1. Drop dict entries where *all* values are ':!' flagged.
          2. Remove ':!' and ':<digit>' markers from strings.
        """

        # If it's a list, recurse into each element
        if isinstance(node, list):
            return [self.generate_layers_modules_tree_simple(x) for x in node]

        # If it's a dict, process each key -> value
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                # Expand tuple-of-strings blocks (special handling)
                if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                    # Rule 1: check if ALL entries are :!
                    if all(any(p == "!" for p in x.split(":")[1:]) for x in v):
                        continue  # skip this parent entirely

                    # Rule 2: strip :! and :digit markers
                    cleaned = tuple(x.split(":")[0] for x in v)
                    new_dict[k] = cleaned
                else:
                    # Recurse deeper
                    new_dict[k] = self.generate_layers_modules_tree_simple(v)
            return new_dict

        # If it's a plain string (unlikely here), strip markers
        if isinstance(node, str):
            return node.split(":")[0]

        # For other types, return as-is
        return node

    def tied_word_embedding(self) -> bool:
        return getattr(self.model.config, "tie_word_embeddings", False)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception as exc:  # torch Modules raise AttributeError here
            model = self.__dict__.get("model")
            if model is None:
                model = self._modules.get("model") if hasattr(self, "_modules") else None
            if model is not None and item != "model":
                return getattr(model, item)
            raise exc

__all__ = ["BaseQModel"]

BaseQModel = ModelLoader(ModelWriter(BaseQModel))
