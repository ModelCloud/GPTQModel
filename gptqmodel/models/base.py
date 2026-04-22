# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import count
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
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.fp4 import TorchFP4Linear
from ..nn_modules.qlinear.fp8 import TorchFP8Linear
from ..nn_modules.qlinear.lookahead import configure_default_lookahead
from ..nn_modules.qlinear.torch import TorchLinear
from ..quantization.config import (
    FORMAT,
    METHOD,
    QUANTIZE_BLACK_LIST,
    AutoModuleDecoderConfig,
    BaseQuantizeConfig,
    GcMode,
    VramStrategy,
    dynamic_get,
    resolve_quant_format,
)
from ..quantization.dtype import (
    available_float8_dtypes,
    dequantize_f4_e2m1,
    dequantize_fp8,
    device_supports_dtype,
    device_supports_native_fp4,
    is_fp4_packed_dtype,
)
from ..quantization.rotation.rotation import fuse_layer_norms, rotate_model
from ..utils.attn_mask import normalize_seq_mask
from ..utils.backend import BACKEND, normalize_backend
from ..utils.calibration import prepare_calibration_dataset
from ..utils.device import get_device
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import QuantizationRegionTimer, setup_logger
from ..utils.model import MODALITY, _module_has_meta_tensors, find_modules, get_module_by_name_prefix, move_to
from ..utils.model_dequant import infer_block_shape
from ..utils.structure import (
    LazyTurtle,
    _get_parent_and_leaf_by_path,
    _get_qualified_name,
    alias_from_turtle_for_submodule,
)
from ..utils.torch import TORCH_HAS_COMPILE, torch_compile
from ._const import (
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

    from ..looper.named_module import NamedModule


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
    if module_kwargs is not None:
        # Preserve per-node kwargs for every scaling group. In multi-batch AWQ
        # replays these can differ by feature bucket, so falling back to a
        # layer-global "latest batch" mask on later nodes can reintroduce
        # sequence-length mismatches during scale search.
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
CAPTURE_ONLY_FLAG = ":?"
MOE_FLAG = ":moe"
NON_QUANTIZE_FLAGS = (NOT_QUANTIZE_FLAG, CAPTURE_ONLY_FLAG)


# Fix cpu memory leak.
# See https://github.com/huggingface/transformers/issues/34366
modeling_utils.check_support_param_buffer_assignment = check_support_param_buffer_assignment

log = setup_logger()

class BaseQModel(nn.Module):
    # name of lm_head
    lm_head: str = "lm_head"

    # Special rotary_emb path
    rotary_embedding: str | None = None

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

    # some models require extra python packages and/or specific version of pkgs such as transformer version(internalm require '<=4.42.2')
    require_pkgs: Optional[List[str]] = None

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

    # Dense-pool strategy support list
    supported_dense_vram_strategies: List[VramStrategy] = [
        VramStrategy.EXCLUSIVE,
        VramStrategy.BALANCED,
    ]

    # MoE expert-pool strategy support list
    supported_moe_vram_strategies: List[VramStrategy] = [
        VramStrategy.EXCLUSIVE,
        VramStrategy.BALANCED,
    ]

    # some models have broken attention mask codes so we need to only use batch 1 with no masks
    support_batch_quantize = True

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    # Some models have optional layers that are not loaded or supported by HF so even when they exist in the original
    # model, they are not properly saved on save(). GLM 4.5/4.6 (air) with MTP layers is such example.
    # Provide either a safetensors filename (the file is copied through if present) or a prefix (all `prefix.` tensors
    # are merged into the main state dict so they end up in model.safetensors).
    out_of_model_tensors: Optional[Dict[str, Union[str | List[str]]]] = None

    supports_desc_act = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    server = None

    support_offload_to_disk = True
    # Optional runtime->checkpoint overrides for LazyTurtle. Prefer reversed
    # `WeightRenaming` entries; legacy runtime->checkpoint dicts are still accepted.
    HF_CONVERSION_MAP_REVERSED: Optional[Any] = None

    moe_expert_module_name_prefixes = [".expert"]

    ATTENTION_MASKS_DTYPE = torch.bool # default to bool

    ATTENTION_MASKS_REQUIRED_FOR_INPUT: bool = False

    INPUT_EMBEDDING_EXTRA_ARGS = None

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: Optional[BaseQuantizeConfig],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: str = None,
        # Lazy turtle is the checkpoint-backed source used to materialize shell modules on demand.
        turtle_model: Optional[LazyTurtle] = None,
    ):
        super().__init__()

        if quantize_config:
            quant_method = quantize_config.method
            # override module_tree if need
            if self.module_tree_overrides is not None and self.module_tree_overrides.get(quant_method) is not None:
                log.info(f'Module Tree: overridden by METHOD.{quant_method.upper()}')
                # setting cls.module_tree
                type(self).module_tree = apply_module_tree_override(self.module_tree, self.module_tree_overrides[quant_method])

            if type(self).module_tree is None:
                type(self).module_tree = self._auto_detect_module_tree(model, quant_method)

        # If module_tree is still None after auto-detection, raise an error indicating unsupported model type
        if type(self).module_tree is None:
            raise ValueError(f"Unsupport model_type {model.config.model_type}, and failed to auto-detect module tree for model {model}")


        # record configuration early so model lifecycle hooks can rely on them
        self.compiled = False  # set to True while compile() is triggered successfully
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        self.quantize_config = quantize_config
        self.quant_region_timer = QuantizationRegionTimer(logger=log)
        self._runtime_generate = None

        self.processor: ProcessorMixin = None

        self.model = self.after_model_load(model, load_quantized_model=load_quantized_model)
        self.turtle_model = turtle_model
        # Captures forward-role auto-decoder choices for regression tests and debug logs.
        self.auto_module_decoder_events: List[Dict[str, Any]] = []

        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(
                    tokenizer,
                    trust_remote_code=trust_remote_code,
                    model_config=getattr(self.model, "config", None),
                )
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
        # Reject activation-quantized checkpoints at load time so the rest of
        # the floatx decoder stack can continue assuming dense activations.
        self._configure_modelopt_runtime()

        self._turtle_lock = threading.RLock()

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        # stores all per-layer quant stats such as avg loss and processing time
        self.quant_log = []

        if self.require_load_processor:
            self.processor = AutoProcessor.from_pretrained(model_local_path, trust_remote_code=self.require_trust_remote_code)

        # apply patching of broken trust_remote_code models here
        if self.require_monkeypatch:
            self.monkey_patch()

        # hack: circular import
        from ..adapter.adapter import Lora

        # check adapter load and print info so users knows lora(s) are applied
        if quantize_config and isinstance(self.quantize_config.adapter, Lora):
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
                module_name, _ = cls._parse_module_flags(node)
                prefix_parts.append(module_name)
            else:
                break  # stop if unexpected nested structure

        return [".".join(prefix_parts)] if prefix_parts else []

    @classmethod
    def _parse_module_aliases(cls, module_spec: str) -> List[str]:
        """
        Parse a module specification into its ordered runtime/checkpoint aliases.

        The first alias is the runtime shell name. Any later aliases are
        alternate checkpoint names declared directly in the model definition.
        """
        parts = module_spec.split(":") if isinstance(module_spec, str) else []
        name = parts[0] if parts else module_spec
        if not isinstance(name, str):
            return [name]
        aliases = [alias for alias in name.split("|") if alias]
        return aliases or [name]

    @classmethod
    def _parse_module_flags(cls, module_spec: str) -> tuple[str, List[str]]:
        """
        Parse a module specification into module name and flags.
        Example: "gate:moe:!" -> ("gate", ["moe", "!"])
        """
        parts = module_spec.split(":") if isinstance(module_spec, str) else []
        aliases = cls._parse_module_aliases(module_spec) if isinstance(module_spec, str) else [module_spec]
        name = aliases[0] if aliases else module_spec
        flags = [p for p in parts[1:] if p]
        return name, flags

    @classmethod
    def has_moe_flag(cls, module_spec: str) -> bool:
        """
        Check if a module specification has the :moe flag.
        """
        if not isinstance(module_spec, str):
            return False
        _, flags = cls._parse_module_flags(module_spec)
        return MOE_FLAG.lstrip(":") in flags

    @classmethod
    def resolve_hf_conversion_map_reversed(cls, target_model: Optional[nn.Module] = None) -> Optional[Any]:
        configured_map = getattr(cls, "HF_CONVERSION_MAP_REVERSED", None)
        if configured_map is not None:
            return copy.deepcopy(configured_map)

        inferred_map = LazyTurtle.infer_hf_conversion_map_reversed(target_model=target_model)
        return copy.deepcopy(inferred_map) if inferred_map is not None else None

    @classmethod
    def _collect_moe_modules_from_tree(cls, tree_node, parent_path="", parent_is_moe=False) -> Set[str]:
        """
        Recursively collect all module paths that have the :moe flag.
        Returns a set of full module paths (e.g., "mlp", "mlp.experts", "mlp.shared_experts").
        """
        moe_modules = set()

        if isinstance(tree_node, dict):
            for key, value in tree_node.items():
                # Skip the layer index placeholder
                if key == "#":
                    # Recursively process the value if it's a dict
                    if isinstance(value, dict):
                        moe_modules.update(cls._collect_moe_modules_from_tree(value, parent_path, parent_is_moe))
                    continue

                # Build full path
                module_name, _ = cls._parse_module_flags(key) if isinstance(key, str) else (key, [])
                if parent_path:
                    full_path = f"{parent_path}.{module_name}"
                else:
                    full_path = module_name

                # Check if this key has :moe flag
                is_moe = cls.has_moe_flag(key) if isinstance(key, str) else False
                if is_moe or parent_is_moe:
                    moe_modules.add(full_path)

                # Recursively process nested structures
                if isinstance(value, (dict, tuple, list)):
                    moe_modules.update(
                        cls._collect_moe_modules_from_tree(value, full_path, parent_is_moe or is_moe)
                    )

        elif isinstance(tree_node, (tuple, list)):
            for item in tree_node:
                if isinstance(item, str) and cls.has_moe_flag(item):
                    module_name, _ = cls._parse_module_flags(item)
                    if parent_path:
                        moe_modules.add(f"{parent_path}.{module_name}")
                    else:
                        moe_modules.add(module_name)
                elif isinstance(item, dict):
                    moe_modules.update(cls._collect_moe_modules_from_tree(item, parent_path, parent_is_moe))

        return moe_modules

    @classmethod
    def get_moe_modules(cls) -> Set[str]:
        """
        Get all MoE module paths from the model's module_tree.
        Returns a set of module paths that have the :moe flag.

        Example: {"mlp", "mlp.experts", "mlp.shared_experts", "mlp.gate"}
        """
        if cls.module_tree is None:
            return set()

        return cls._collect_moe_modules_from_tree(cls.module_tree)

    @classmethod
    def is_moe_module(cls, module_path: str) -> bool:
        """
        Check if a given module path is an MoE module based on :moe flags.

        Args:
            module_path: Full module path like "model.layers.0.mlp.experts.5.gate_proj"

        Returns:
            True if any parent in the path is marked with :moe flag
        """
        moe_modules = cls.get_moe_modules()

        # Check if any MoE module is a prefix of this path
        for moe_module in moe_modules:
            # Handle layer index in path (e.g., "model.layers.0.mlp" should match "mlp")
            if f".{moe_module}" in module_path or module_path.endswith(moe_module):
                return True
            # Also check for patterns like "mlp.experts.5" matching "mlp.experts"
            path_parts = module_path.split(".")
            for i in range(len(path_parts)):
                partial_path = ".".join(path_parts[i:])
                if partial_path.startswith(moe_module + ".") or partial_path == moe_module:
                    return True

        return False

    @classmethod
    def get_moe_module_name(cls) -> Optional[str]:
        """
        Get the name of the MoE module from module_tree.

        Each layer can have only ONE MoE module marked with :moe flag.
        For example:
        - GLM-4: "mlp:moe" -> returns "mlp"
        - MiniMax-M2: "block_sparse_moe:moe" -> returns "block_sparse_moe"

        Returns:
            The name of the MoE module (without flags), or None if no MoE module is defined
        """
        if cls.module_tree is None:
            return None

        # Find the dict that represents layer structure (after "#")
        layer_structure = None
        found_hash = False
        for item in cls.module_tree:
            if item == "#":
                found_hash = True
                continue
            if found_hash and isinstance(item, dict):
                layer_structure = item
                break

        if layer_structure is None:
            return None

        # Look for a key with :moe flag at the top level of layer structure
        for key in layer_structure.keys():
            if cls.has_moe_flag(key):
                # Extract module name without flags
                module_name, _ = cls._parse_module_flags(key)
                return module_name

        return None

    @classmethod
    def build_moe_modules_if_need(cls, model_config, layer_modules, is_awq_quantize: bool = False):
        # MoE models
        if model_config is not None and cls.dynamic_expert_index is not None:
            num_experts = cls.get_num_experts(model_config)

            moe_simple = []
            capture_only_modules = None
            for names in layer_modules:
                moe_simple.append([])

                has_expert = any(EXPERT_INDEX_PLACEHOLDER in n for n in names)
                has_capture_only = all(CAPTURE_ONLY_FLAG in n for n in names)
                if has_capture_only:
                    capture_only_modules = list(names)
                    continue

                if not has_expert:
                    moe_simple[-1].extend(names)
                    continue

                if is_awq_quantize:
                    # AWQ expands expert placeholders into concrete expert paths while
                    # preserving the non-expert segments exactly where the model
                    # definition placed them. This keeps the expanded block aligned
                    # with forward execution order instead of forcing shared-expert
                    # modules to the tail of every mixed MoE block.
                    segments = []
                    current_segment = []
                    current_is_expert_segment = None
                    for n in names:
                        is_expert_entry = EXPERT_INDEX_PLACEHOLDER in n
                        if current_is_expert_segment is None:
                            current_is_expert_segment = is_expert_entry
                        if is_expert_entry != current_is_expert_segment:
                            segments.append((current_is_expert_segment, current_segment))
                            current_segment = []
                            current_is_expert_segment = is_expert_entry
                        current_segment.append(n)

                    if current_segment:
                        segments.append((current_is_expert_segment, current_segment))

                    # Example:
                    # ['shared_expert.gate_proj', 'shared_expert.up_proj', 'experts.#.gate_proj', 'experts.#.up_proj']
                    # becomes
                    # ['shared_expert.gate_proj', 'shared_expert.up_proj', 'experts.0.gate_proj', 'experts.0.up_proj', ...]
                    for is_expert_segment, segment_names in segments:
                        if not is_expert_segment:
                            moe_simple[-1].extend(segment_names)
                            continue
                        for index in range(num_experts):
                            for n in segment_names:
                                moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
                    # Currently, only need to add `capture_only_modules` to `['mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']`
                    # or ['mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    # or ['mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    add_capture_only_module = len(names) == (4 if any("shared_expert" in n for n in names) else 2)
                    if add_capture_only_module and capture_only_modules:
                        # Extend all elements in capture_only_modules
                        moe_simple[-1].extend(capture_only_modules)
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
        def should_quantize(name: str) -> bool:
            # Check if the module name contains any NON_QUANTIZE_FLAGS that indicates it should NOT be quantized
            return not any(flag in name for flag in NON_QUANTIZE_FLAGS)

        filtered_layer_modules = []
        for block in layer_modules:
            filtered_block = [name for name in block if should_quantize(name)]
            filtered_layer_modules.append(filtered_block)
        layer_modules = filtered_layer_modules

        layer_modules = [block for block in layer_modules if block]  # Remove empty blocks

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
    def simple_layer_modules(cls, model_config, quantize_config, is_awq_quantize: bool = False, include_capture_only: bool = False):
        layer_modules = cls.build_layer_modules(cls.module_tree, include_capture_only=include_capture_only)

        layer_modules = cls.build_moe_modules_if_need(model_config, layer_modules, is_awq_quantize)

        layer_modules = cls.filter_not_quantize_module(layer_modules, quantize_config)

        # print(f"simple_layer_modules layer_modules: {layer_modules}")
        return layer_modules

    @classmethod
    def full_layer_modules(cls, model_config=None, is_awq_quantize: bool = False, include_capture_only: bool = False):
        full = cls.build_layer_modules(cls.module_tree, include_capture_only=include_capture_only)
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
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
    ):
        return prepare_calibration_dataset(
            self,
            calibration_dataset=calibration_dataset,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            calibration_dataset_sort=calibration_dataset_sort,
            batch_size=batch_size,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            logger=log,
        )

    def quantize(
        self,
        calibration: Optional[Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]] = None,
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
        calibration_concat_separator: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        if self.quantize_config is None or not isinstance(self.quantize_config, BaseQuantizeConfig):
            raise AttributeError("`quantize_config` must be not None")

        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.reset()

        if self.quantize_config.method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.method}"
            )

        if not self.support_batch_quantize:
            log.warn("Quantize: batch_size overridden by model class definition to `disabled`")
            batch_size = 1 # but actually disabled

        format_code = resolve_quant_format(self.quantize_config.format, self.quantize_config.method)

        if format_code == FORMAT.MARLIN:
            raise ValueError(
                "FORMAT.MARLIN is deprecated for quantization. Please switch to FORMAT.GPTQ. GPTQMOdel will auto-use Marlin kernel for accelerated inference for FORMAT.GPTQ."
            )

        export_quant_method = self.quantize_config.export_quant_method()

        if export_quant_method == METHOD.AWQ:
            if format_code in [FORMAT.GEMV_FAST, FORMAT.LLM_AWQ]:
                # AWQ GEMV_FAST / LLM_AWQ only supports pack_dtype is torch.int16
                log.info("Quantize Model: Auto fix `pack_dtype` to `torch.int16`")
                self.quantize_config.pack_dtype = torch.int16

        if self.support_batch_quantize is False:
            batch_size = 1
            log.warn("Batch quantization is not supported for this model. Setting batch_size to 1.")

        requested_backend = backend
        requested_backend = normalize_backend(requested_backend, quant_method=export_quant_method)

        preferred_backend = requested_backend
        if preferred_backend in (None, BACKEND.AUTO):
            if export_quant_method == METHOD.AWQ:
                if format_code == FORMAT.GEMM:
                    # Weight-only RTN->AWQ export should stay on the portable torch kernel.
                    preferred_backend = (
                        BACKEND.AWQ_TORCH
                        if self.quantize_config.uses_weight_only_lifecycle()
                        else BACKEND.AWQ_GEMM
                    )
                elif format_code == FORMAT.BITBLAS:
                    preferred_backend = BACKEND.AWQ_BITBLAS
                elif format_code == FORMAT.GEMV:
                    preferred_backend = BACKEND.AWQ_GEMV
                elif format_code in [FORMAT.GEMV_FAST, FORMAT.LLM_AWQ]:
                    preferred_backend = BACKEND.AWQ_GEMV_FAST
                else:
                    raise ValueError(f"Unsupported FORMAT: `{self.quantize_config.format}` with `METHOD.AWQ`")
            elif self.quantize_config.method == METHOD.QQQ:
                preferred_backend = BACKEND.QQQ
            elif self.quantize_config.method == METHOD.PARO:
                preferred_backend = BACKEND.PAROQUANT_CUDA
            elif self.quantize_config.method == METHOD.EXL3:
                preferred_backend = BACKEND.EXL3_EXLLAMA_V3
            elif self.quantize_config.method == METHOD.GGUF:
                preferred_backend = BACKEND.AUTO
            elif self.quantize_config.method == METHOD.FP8:
                preferred_backend = BACKEND.FP8_TORCH
            elif self.quantize_config.method == METHOD.BITSANDBYTES:
                preferred_backend = BACKEND.BITSANDBYTES
            else:
                preferred_backend = BACKEND.GPTQ_TORCH

        if self.quantize_config.method == METHOD.EXL3:
            if preferred_backend not in (BACKEND.AUTO, BACKEND.EXL3_EXLLAMA_V3):
                raise ValueError("EXL3 quantization only supports BACKEND.AUTO or BACKEND.EXL3_EXLLAMA_V3.")

            if not torch.cuda.is_available():
                raise ValueError("EXL3 quantization requires CUDA/HIP.")

            quant_device = self.quantize_config.device
            if isinstance(quant_device, DEVICE):
                quant_device_type = quant_device.type
            elif isinstance(quant_device, torch.device):
                quant_device_type = quant_device.type
            else:
                quant_device_type = str(quant_device).split(":")[0].lower()

            if quant_device_type != "cuda":
                raise ValueError("EXL3 quantization requires a CUDA/HIP quantization device.")
        else:
            # Validate quant linear before quantization starts
            _ = select_quant_linear(
                bits=self.quantize_config.runtime_bits,
                dynamic=self.quantize_config.dynamic,
                group_size=self.quantize_config.group_size,
                desc_act=self.quantize_config.desc_act,
                sym=self.quantize_config.sym,
                backend=preferred_backend,
                format=format_code,
                quant_method=export_quant_method,
                device=DEVICE(self.quantize_config.device),
                pack=True,
                pack_dtype=self.quantize_config.pack_dtype,
            )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(
                    tokenizer,
                    trust_remote_code=self.trust_remote_code,
                    model_config=getattr(self.model, "config", None),
                )
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        if format_code == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        # overwrite quantize_config.adapter
        if adapter is not None:
            self.quantize_config.adapter = adapter

        if self.quantize_config.method == METHOD.EXL3:
            self.qlinear_kernel = ExllamaV3Linear
        else:
            self.qlinear_kernel = select_quant_linear(
                    bits=self.quantize_config.runtime_bits,
                    group_size=self.quantize_config.group_size,
                    desc_act=self.quantize_config.desc_act,
                    sym=self.quantize_config.sym,
                    pack=True,
                    dynamic=self.quantize_config.dynamic,
                    device=DEVICE(self.quantize_config.device),
                    pack_dtype=self.quantize_config.pack_dtype,
                    multi_select=False,
                    backend=preferred_backend,
                    format=format_code,
                    quant_method=export_quant_method,
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

        if self.quantize_config.uses_weight_only_lifecycle():
            result = self._quantize_weight_only(
                calibration=calibration,
                calibration_concat_size=calibration_concat_size,
                calibration_sort=calibration_sort,
                batch_size=batch_size,
                backend=backend,
                calibration_concat_separator=calibration_concat_separator,
            )
        else:
            if calibration is None:
                raise ValueError(
                    "Calibration dataset is required unless a weight-only quantize config is configured."
                )
            result = self._quantize_with_calibration(
                calibration=calibration,
                calibration_concat_size=calibration_concat_size,
                calibration_sort=calibration_sort,
                batch_size=batch_size,
                backend=backend,
                adapter_calibration_dataset=adapter_calibration_dataset,
                calibration_concat_separator=calibration_concat_separator,
            )

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.flush()

        return result

    def _quantize_with_calibration(
        self,
        *,
        calibration,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        backend: Optional[BACKEND],
        adapter_calibration_dataset,
        calibration_concat_separator: Optional[str],
    ):
        from ..adapter.adapter import Lora
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper
        from ..looper.module_preprocessor import ModulePreProcessor

        needs_lora = isinstance(self.quantize_config.adapter, Lora)

        args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration": calibration,
            "prepare_dataset_func": self.prepare_dataset,
            "calibration_concat_size": calibration_concat_size,
            "calibration_sort": calibration_sort,
            "calibration_concat_separator": calibration_concat_separator,
            "batch_size": batch_size,
            "calculate_w_wq_diff": needs_lora,
        }

        preprocessors = []
        if getattr(self.quantize_config, "preprocessors", None):
            preprocessors.append(ModulePreProcessor(**args))

        if self.quantize_config.method == METHOD.EXL3:
            from ..looper.exllamav3_processor import EXL3Processor

            if needs_lora:
                raise NotImplementedError("EXL3 quantization does not support adapter/EoRA generation.")

            if getattr(self.quantize_config, "gptaq", None) is not None:
                raise NotImplementedError("EXL3 quantization does not support GPTAQ/native activation capture.")

            if getattr(self.quantize_config, "foem", None) is not None:
                raise NotImplementedError("EXL3 quantization does not support FOEM/native activation capture.")

            exl3_args = {
                "tokenizer": self.tokenizer,
                "qcfg": self.quantize_config,
                "calibration": calibration,
                "prepare_dataset_func": self.prepare_dataset,
                "calibration_concat_size": calibration_concat_size,
                "calibration_sort": calibration_sort,
                "calibration_concat_separator": calibration_concat_separator,
                "batch_size": batch_size,
                "lm_head_name": self.lm_head,
            }
            quantize_processor = preprocessors + [
                EXL3Processor(**exl3_args),
            ]
        elif self.quantize_config.method == METHOD.QQQ:
            from ..looper.qqq_processor import QQQProcessor

            quantize_processor = preprocessors + [
                QQQProcessor(**args),
            ]
        elif self.quantize_config.method == METHOD.AWQ:
            from ..looper.awq_processor import AWQProcessor

            os.environ["AWQ_BATCH_SIZE"] = str(batch_size)

            awq_args = dict(args)
            awq_args["gptq_model"] = self
            awq_args["model"] = self.model
            awq_args["batch_size"] = batch_size

            quantize_processor = preprocessors + [
                AWQProcessor(**awq_args),
            ]
        elif self.quantize_config.method == METHOD.PARO:
            from ..looper.paroquant_processor import ParoQuantProcessor

            os.environ["AWQ_BATCH_SIZE"] = str(batch_size)

            paro_args = dict(args)
            paro_args["gptq_model"] = self
            paro_args["model"] = self.model
            paro_args["batch_size"] = batch_size

            quantize_processor = preprocessors + [
                ParoQuantProcessor(**paro_args),
            ]
        else:
            from ..looper.gptq_processor import GPTQProcessor

            quantize_processor = preprocessors + [
                GPTQProcessor(**args),
            ]

        if getattr(self.quantize_config, "gptaq", None) is not None:
            from ..looper.native_processor import NativeProcessor

            args_to_copy = {k: v for k, v in args.items() if k != "prepare_dataset_func"}
            args_clone = copy.deepcopy(args_to_copy)
            args_clone["prepare_dataset_func"] = args["prepare_dataset_func"]

            args_clone.pop("calculate_w_wq_diff", None)
            quantize_processor.insert(0, NativeProcessor(**args_clone))

        if getattr(self.quantize_config, "foem", None) is not None:
            if self.quantize_config.foem.alpha > 0:
                from ..looper.native_processor import NativeProcessor

                args_to_copy = {k: v for k, v in args.items() if k != "prepare_dataset_func"}
                args_clone = copy.deepcopy(args_to_copy)
                args_clone["prepare_dataset_func"] = args["prepare_dataset_func"]

                args_clone.pop("calculate_w_wq_diff", None)
                quantize_processor.insert(0, NativeProcessor(**args_clone))

        processors = quantize_processor
        if needs_lora:
            processors.append(
                EoraProcessor(
                    tokenizer=self.tokenizer,
                    qcfg=self.quantize_config,
                    calibration=adapter_calibration_dataset if adapter_calibration_dataset is not None else calibration,
                    prepare_dataset_func=self.prepare_dataset,
                    calibration_concat_size=calibration_concat_size,
                    calibration_sort=calibration_sort,
                    calibration_concat_separator=calibration_concat_separator,
                    batch_size=batch_size,
                )
            )

        module_looper = ModuleLooper(self, processors=processors)

        gc_context = (
            DEVICE_THREAD_POOL.no_auto_gc()
            if self.quantize_config.gc_mode == GcMode.ON_STAGE_END
            else nullcontext()
        )

        with gc_context:
            return module_looper.loop(
                backend=backend,
                fallback=self.quantize_config.fallback,
            )

    def _quantize_weight_only(
        self,
        *,
        calibration,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        backend: Optional[BACKEND],
        calibration_concat_separator: Optional[str],
    ):
        del calibration_concat_size, calibration_sort, batch_size, calibration_concat_separator

        from ..adapter.adapter import Lora
        from ..looper.weight_only_looper import WeightOnlyLooper
        from ..looper.weight_only_processor import WeightOnlyProcessor

        if calibration is not None:
            log.info("Weight-only quantization selected; ignoring provided calibration dataset.")

        if isinstance(self.quantize_config.adapter, Lora):
            raise NotImplementedError(
                "Weight-only quantization does not support adapter/EoRA generation."
            )

        if getattr(self.quantize_config, "gptaq", None) is not None:
            raise NotImplementedError(
                "Weight-only quantization does not support GPTAQ/native activation capture."
            )

        if getattr(self.quantize_config, "foem", None) is not None:
            raise NotImplementedError(
                "Weight-only quantization does not support FOEM/native activation capture."
            )

        processor = WeightOnlyProcessor(
            tokenizer=self.tokenizer,
            qcfg=self.quantize_config,
        )
        module_looper = WeightOnlyLooper(model=self, processor=processor)

        gc_context = (
            DEVICE_THREAD_POOL.no_auto_gc()
            if self.quantize_config.gc_mode == GcMode.ON_STAGE_END
            else nullcontext()
        )

        with gc_context:
            return module_looper.loop(backend=backend)

    def _eora_generate(
        self,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter,
        quantized_modules: Dict[str, TorchLinear],
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        calibration_concat_separator: Optional[str] = None,
    ):
        if self.quantized:
            raise EnvironmentError("eora_generate() is called a model that is already quantized")

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(
                    tokenizer,
                    trust_remote_code=self.trust_remote_code,
                    model_config=getattr(self.model, "config", None),
                )
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        from ..adapter.adapter import Lora
        from ..looper.dequantize_processor import DequantizeProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper
        from ..looper.module_preprocessor import ModulePreProcessor

        self.quantize_config.adapter = adapter

        assert isinstance(self.quantize_config.adapter, Lora)

        # init processor with EoRA processor
        processors = []
        if getattr(self.quantize_config, "preprocessors", None):
            processors.append(
                ModulePreProcessor(
                    tokenizer=self.tokenizer,
                    qcfg=self.quantize_config,
                    calibration=calibration_dataset,
                    prepare_dataset_func=self.prepare_dataset,
                    calibration_concat_size=calibration_dataset_concat_size,
                    calibration_sort=calibration_dataset_sort,
                    calibration_concat_separator=calibration_concat_separator,
                    batch_size=batch_size,
                ),
            )
        processors.extend(
            [
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
                    calibration_concat_separator=calibration_concat_separator,
                    batch_size=batch_size,
                ),
            ]
        )

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

    def move_input_capture_example(
        self,
        example: Dict[str, Any],
        data_device: torch.device,
    ) -> Dict[str, Any]:
        for key, value in example.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if not torch.is_tensor(item):
                        continue

                    if item.ndim == 1:
                        item = item.unsqueeze(0)

                    value[index] = move_to(item, device=data_device)
            elif torch.is_tensor(value):
                if value.ndim == 1:
                    value = value.unsqueeze(0)

                example[key] = move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)

    def finalize_input_capture_example(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.ATTENTION_MASKS_DTYPE is torch.long and "attention_mask" in example:
            example["attention_mask"] = example["attention_mask"].long()

        return example

    def run_input_capture(
        self,
        example: Dict[str, Any],
        use_cache: bool,
        data_device: torch.device,
    ):
        if self.INPUT_EMBEDDING_EXTRA_ARGS:
            return self.model.generate(
                **example,
                **self.INPUT_EMBEDDING_EXTRA_ARGS,
            )

        return self.model(**example, use_cache=use_cache)

    def _generate_with_runtime(self, runtime_generate, inputs=None, **kwargs):
        def _normalize_generate_attention_mask(input_ids, attention_mask):
            if not torch.is_tensor(attention_mask) or attention_mask.ndim <= 2:
                return attention_mask

            seq_len = None
            if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
                seq_len = input_ids.shape[-1]

            return normalize_seq_mask(attention_mask, seq_len=seq_len)

        if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
            kwargs.setdefault("prompts", inputs)
        elif hasattr(inputs, "get") and not torch.is_tensor(inputs):
            merged_kwargs = dict(inputs)
            merged_kwargs.update(kwargs)
            kwargs = merged_kwargs
        elif inputs is not None:
            kwargs.setdefault("input_ids", inputs)

        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = _normalize_generate_attention_mask(
                kwargs.get("input_ids"),
                kwargs["attention_mask"],
            )

        return runtime_generate(self.model, **kwargs)

    def generate(self, inputs=None, **kwargs):
        with torch.inference_mode():
            # fix hf generate not applying correct pad token
            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            runtime_generate = getattr(self, "_runtime_generate", None)
            if runtime_generate is not None:
                return self._generate_with_runtime(runtime_generate, inputs=inputs, **kwargs)

            def _normalize_generate_attention_mask(input_ids, attention_mask):
                if not torch.is_tensor(attention_mask) or attention_mask.ndim <= 2:
                    return attention_mask

                seq_len = None
                if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
                    seq_len = input_ids.shape[-1]

                return normalize_seq_mask(attention_mask, seq_len=seq_len)

            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                if self.tokenizer is None:
                    raise ValueError("You passed in an `input` to `generate()` of type `str` but model is missing `model.tokenizer`. Please set `model.tokenizer = my_tokenizer`.")
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left")
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = _normalize_generate_attention_mask(
                        inputs.get("input_ids"),
                        inputs["attention_mask"],
                    )
                inputs = inputs.to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            if hasattr(inputs, "get") and not torch.is_tensor(inputs):
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = _normalize_generate_attention_mask(
                        inputs.get("input_ids"),
                        inputs["attention_mask"],
                    )
                return self.model.generate(**inputs, **kwargs)

            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = _normalize_generate_attention_mask(
                    kwargs.get("input_ids", inputs),
                    kwargs["attention_mask"],
                )

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            split_by: Optional[str] = None,
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
                    eora_path=eora_path,
                    split_by=split_by)

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

    def _active_auto_module_decoder_config(self) -> Optional[AutoModuleDecoderConfig]:
        """Return the active auto-decoder preprocessor config, if any."""

        preprocessors = getattr(self.quantize_config, "preprocessors", None) or []
        for preprocessor in reversed(preprocessors):
            if isinstance(preprocessor, AutoModuleDecoderConfig):
                return preprocessor
        return None

    def materialize_passthrough_modules_for_save(self) -> int:
        """Decode passthrough floatx modules in-place before saving when configured."""

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None or decoder_cfg.passthrough_save_policy != "decode":
            return 0

        decoded_count = 0
        for _, module in list(self.model.named_modules()):
            if isinstance(module, BaseQuantLinear) or not hasattr(module, "weight"):
                continue

            checkpoint_tensors = None
            if isinstance(self.turtle_model, LazyTurtle):
                checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=module,
                    recurse=False,
                )
            if not checkpoint_tensors:
                checkpoint_tensors = dict(module.state_dict(keep_vars=True))
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue

            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
            )
            if decoder_kind is None:
                continue

            decoded_module = self._build_decoder_quant_source_module(
                module,
                checkpoint_tensors=checkpoint_tensors,
                target_dtype=decoder_cfg.target_dtype,
            )
            self._replace_live_submodule(module, decoded_module)
            decoded_count += 1

        return decoded_count

    def materialize_passthrough_modules_for_eval(
        self,
        device: torch.device,
        *,
        respect_forward_policy: bool = False,
    ) -> int:
        """Materialize passthrough floatx modules into live evaluation modules."""

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None:
            return 0

        target_device = torch.device(device)
        decoded_count = 0
        for _, module in list(self.model.named_modules()):
            if isinstance(module, BaseQuantLinear) or not hasattr(module, "weight"):
                continue

            checkpoint_tensors = None
            if isinstance(self.turtle_model, LazyTurtle):
                checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=module,
                    recurse=False,
                )
            if not checkpoint_tensors:
                checkpoint_tensors = dict(module.state_dict(keep_vars=True))
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue

            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
            )
            if decoder_kind is None:
                continue

            forward_module = None
            if respect_forward_policy and decoder_cfg.passthrough_forward_policy != "decode":
                if decoder_kind == "fp8" and device_supports_dtype(target_device, weight.dtype, require_validation=False):
                    forward_module = self._build_fp8_forward_module(
                        target_submodule=module,
                        checkpoint_tensors=checkpoint_tensors,
                        device=target_device,
                        target_dtype=decoder_cfg.target_dtype,
                    )
                elif decoder_kind == "fp4" and device_supports_native_fp4(target_device, require_validation=False):
                    forward_module = self._build_fp4_forward_module(
                        target_submodule=module,
                        checkpoint_tensors=checkpoint_tensors,
                        device=target_device,
                        target_dtype=decoder_cfg.target_dtype,
                    )

            if forward_module is None:
                decoded_module = self._build_decoder_quant_source_module(
                    module,
                    checkpoint_tensors=checkpoint_tensors,
                    target_dtype=decoder_cfg.target_dtype,
                )
                forward_module = self._build_decoder_forward_module(
                    quant_source=decoded_module,
                    device=target_device,
                )
            self._replace_live_submodule(module, forward_module)
            decoded_count += 1

        return decoded_count

    def decoded_passthrough_state_dict_entries_for_save(self) -> tuple[Dict[str, torch.Tensor], List[str]]:
        """Return dense state-dict entries that should replace native passthrough tensors on save."""

        decoder_cfg = self._active_auto_module_decoder_config()
        if decoder_cfg is None or decoder_cfg.passthrough_save_policy != "decode":
            return {}, []

        decoded_entries: Dict[str, torch.Tensor] = {}
        decoded_prefixes: List[str] = []
        for module_name, module in list(self.model.named_modules()):
            if not module_name or isinstance(module, BaseQuantLinear) or not hasattr(module, "weight"):
                continue

            checkpoint_tensors = None
            if isinstance(self.turtle_model, LazyTurtle):
                checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=module,
                    recurse=False,
                )
            if not checkpoint_tensors:
                checkpoint_tensors = dict(module.state_dict(keep_vars=True))
            weight = checkpoint_tensors.get("weight")
            if not isinstance(weight, torch.Tensor):
                continue

            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
            )
            if decoder_kind is None:
                continue

            decoded_module = self._build_decoder_quant_source_module(
                module,
                checkpoint_tensors=checkpoint_tensors,
                target_dtype=decoder_cfg.target_dtype,
            )
            decoded_prefixes.append(module_name)
            for key, tensor in decoded_module.state_dict().items():
                decoded_entries[f"{module_name}.{key}"] = tensor.detach().cpu()

        return decoded_entries, decoded_prefixes


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

        quant_modules = [module for module in self.model.modules() if isinstance(module, TorchLinear)]
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

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        pass

    def after_model_load(self, model, load_quantized_model):
        return model

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            # This hook is now disabled as it's handled by the ModuleLooper after input capture.
            # offload_to_disk(model=self.model, module=self.get_base_modules(model=self.model), disk_path=self.quantize_config.offload_to_disk_path)
            pass

    def capture_first_layer_positional_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
    ) -> List[torch.Tensor]:
        """Normalize first-layer positional inputs so cached forwards can replay decoder layers directly."""

        if kwargs.get("hidden_states") is not None:
            return [move_to(kwargs["hidden_states"], device=batch_device)]
        if args:
            return [move_to(args[0], device=batch_device)]
        return []

    def capture_first_layer_input_kwargs(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        batch_device: torch.device,
        layer_input_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Allow model definitions to persist extra first-layer replay metadata during calibration capture."""

        return layer_input_kwargs

    def prepare_layer_replay_kwargs(
        self,
        layer: nn.Module,
        layer_input: List[torch.Tensor],
        additional_inputs: Dict[str, Any],
        target_device: torch.device,
    ) -> Dict[str, Any]:
        """Allow model definitions to refresh layer-specific kwargs before cached layer replay."""

        return additional_inputs

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
        if get_device(module) == META or _module_has_meta_tensors(module):
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

    def _replace_live_submodule(
        self,
        current_submodule: nn.Module,
        replacement: nn.Module,
    ) -> nn.Module:
        """Replace one live model submodule in place and return the replacement."""

        module_path = _get_qualified_name(self.model, current_submodule)
        parent, leaf = _get_parent_and_leaf_by_path(self.model, module_path)
        setattr(parent, leaf, replacement)
        return replacement

    def _build_decoder_quant_source_module(
        self,
        target_submodule: nn.Module,
        *,
        checkpoint_tensors: Optional[Dict[str, torch.Tensor]] = None,
        target_dtype: torch.dtype,
    ) -> nn.Module:
        """Build a dense CPU source module from checkpoint tensors for quantization."""

        quant_source = copy.deepcopy(target_submodule)
        if _module_has_meta_tensors(quant_source):
            quant_source = quant_source.to_empty(device=CPU)
        else:
            quant_source = quant_source.to(device=CPU)
        weight = None if checkpoint_tensors is None else checkpoint_tensors.get("weight")
        if isinstance(weight, torch.Tensor) and hasattr(quant_source, "weight"):
            decoder_kind = self._decoder_weight_format(
                weight=weight,
                checkpoint_tensors=checkpoint_tensors,
            )
            result_shape = tuple(getattr(quant_source.weight, "shape", weight.shape))
            scale = (
                self._decoder_fp4_effective_scale(
                    checkpoint_tensors=checkpoint_tensors,
                    result_shape=result_shape,
                )
                if decoder_kind == "fp4"
                else self._decoder_scale_tensor(
                    scale_tensor=checkpoint_tensors.get("weight_scale"),
                    result_shape=result_shape,
                )
            )
            scale_inv = None
            if not isinstance(scale, torch.Tensor):
                scale_inv = self._decoder_scale_tensor(
                    scale_tensor=checkpoint_tensors.get("weight_scale_inv"),
                    result_shape=result_shape,
                )
            if decoder_kind == "fp8":
                decoded_weight = dequantize_fp8(
                    weight,
                    scale=scale if isinstance(scale, torch.Tensor) else None,
                    scale_inv=scale_inv if isinstance(scale_inv, torch.Tensor) else None,
                    axis=None,
                    target_dtype=target_dtype,
                )
            elif decoder_kind == "fp4":
                decoded_weight = dequantize_f4_e2m1(
                    weight,
                    scale=scale if isinstance(scale, torch.Tensor) else None,
                    scale_inv=scale_inv if isinstance(scale_inv, torch.Tensor) else None,
                    axis=None,
                    target_dtype=target_dtype,
                )
            else:
                decoded_weight = weight.to(dtype=target_dtype)

            existing_weight = getattr(quant_source, "weight")
            quant_source.weight = nn.Parameter(
                decoded_weight.to(device=CPU, dtype=target_dtype),
                requires_grad=getattr(existing_weight, "requires_grad", False),
            )

        bias = None if checkpoint_tensors is None else checkpoint_tensors.get("bias")
        if isinstance(bias, torch.Tensor) and getattr(quant_source, "bias", None) is not None:
            existing_bias = quant_source.bias
            quant_source.bias = nn.Parameter(
                bias.to(device=CPU, dtype=target_dtype),
                requires_grad=getattr(existing_bias, "requires_grad", False),
            )

        quant_source = quant_source.to(dtype=target_dtype)
        quant_source.eval()
        setattr(quant_source, "target_device", torch.device(CPU))
        return quant_source

    def _decoder_block_size(self) -> Optional[tuple[int, int]]:
        """Read the checkpoint's floatx block size metadata when present."""

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if isinstance(quant_config, dict):
            block_size = quant_config.get("weight_block_size")
        else:
            block_size = getattr(quant_config, "weight_block_size", None)
        if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
            return int(block_size[0]), int(block_size[1])
        return None

    def _decoder_quant_method_name(self) -> str:
        """Return the checkpoint quantizer family declared in model config."""

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if isinstance(quant_config, dict):
            value = quant_config.get("quant_method")
        else:
            value = getattr(quant_config, "quant_method", None)
        return str(value or "").strip().lower()

    def _uses_modelopt_runtime(self) -> bool:
        """Return ``True`` when the checkpoint declares ModelOpt runtime semantics."""

        return self._decoder_quant_method_name() == "modelopt"

    def _modelopt_activation_quantization_mode(self) -> Optional[str]:
        """Describe unsupported ModelOpt activation quantization metadata when present."""

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if not isinstance(quant_config, dict):
            return None

        config_groups = quant_config.get("config_groups")
        if isinstance(config_groups, dict):
            for group_cfg in config_groups.values():
                if not isinstance(group_cfg, dict):
                    continue
                input_activations = group_cfg.get("input_activations")
                if isinstance(input_activations, dict):
                    num_bits = input_activations.get("num_bits")
                    if isinstance(num_bits, (int, float)) and int(num_bits) < 16:
                        return "input_activations"

        kv_cache_scheme = quant_config.get("kv_cache_scheme")
        if isinstance(kv_cache_scheme, dict):
            num_bits = kv_cache_scheme.get("num_bits")
            if isinstance(num_bits, (int, float)) and int(num_bits) < 16:
                return "kv_cache_scheme"

        if isinstance(self.turtle_model, LazyTurtle):
            keys = self.turtle_model._weight_map.keys()
        else:
            keys = self.model.state_dict().keys()
        if any(str(name).endswith((".input_scale", ".k_scale", ".v_scale")) for name in keys):
            return "checkpoint_scales"
        return None

    def _configure_modelopt_runtime(self) -> None:
        """Reject unsupported ModelOpt activation quantization at load time."""

        if not self._uses_modelopt_runtime():
            return

        unsupported_mode = self._modelopt_activation_quantization_mode()
        if unsupported_mode is not None:
            log.error("GPT-QModel currently does not support loading of activation quantized models")
            raise ValueError(
                "GPT-QModel currently does not support loading of activation quantized models. "
                "GPTQModel does not support loading ModelOpt checkpoints with activation quantization. "
                "Only dense-activation weight-only variants such as W8A16/FP8 and W4A16/FP4 are supported. "
                f"Detected unsupported metadata: {unsupported_mode}."
            )

    def _decoder_scale_tensor(
        self,
        *,
        scale_tensor: Optional[torch.Tensor],
        result_shape: tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        """Expand padded floatx block grids to the dense weight shape when needed."""

        if not isinstance(scale_tensor, torch.Tensor):
            return None
        if scale_tensor.ndim != 2 or len(result_shape) != 2:
            return scale_tensor

        rows, cols = result_shape
        blocks_r, blocks_c = scale_tensor.shape
        if rows % blocks_r == 0 and cols % blocks_c == 0:
            return scale_tensor

        block_size = self._decoder_block_size()
        if block_size is None:
            return scale_tensor

        block_rows, block_cols = block_size
        if blocks_r * block_rows < rows or blocks_c * block_cols < cols:
            return scale_tensor

        expanded = scale_tensor.repeat_interleave(block_rows, dim=0)
        expanded = expanded.repeat_interleave(block_cols, dim=1)
        return expanded[:rows, :cols].contiguous()

    def _decoder_fp4_effective_scale(
        self,
        *,
        checkpoint_tensors: Dict[str, torch.Tensor],
        result_shape: tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        """Resolve NVFP4 weight scales, including ModelOpt's secondary global scale."""

        scale = self._decoder_scale_tensor(
            scale_tensor=checkpoint_tensors.get("weight_scale"),
            result_shape=result_shape,
        )
        if not isinstance(scale, torch.Tensor):
            return None
        scale_2 = checkpoint_tensors.get("weight_scale_2")
        if isinstance(scale_2, torch.Tensor):
            scale = scale.to(torch.float32) * scale_2.to(torch.float32)
        return scale

    def _decoder_weight_format(
        self,
        *,
        weight: torch.Tensor,
        checkpoint_tensors: Dict[str, torch.Tensor],
    ) -> Optional[str]:
        """Infer which floatx decoder matches one checkpoint weight tensor."""

        if weight.dtype in available_float8_dtypes():
            return "fp8"
        if is_fp4_packed_dtype(weight.dtype):
            return "fp4"
        if weight.dtype is not torch.uint8 or not isinstance(checkpoint_tensors.get("weight_scale"), torch.Tensor):
            return None
        if isinstance(checkpoint_tensors.get("weight_scale_2"), torch.Tensor):
            return "fp4"

        quant_config = getattr(getattr(self.model, "config", None), "quantization_config", None)
        if isinstance(quant_config, dict):
            format_name = quant_config.get("format") or quant_config.get("quant_method")
        else:
            format_name = getattr(quant_config, "format", None) or getattr(quant_config, "quant_method", None)
        if str(format_name or "").strip().lower() in {"nvfp4", "fp4"}:
            return "fp4"
        return None

    def _build_decoder_forward_module(
        self,
        *,
        quant_source: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        """Clone the decoded quant source into a live forward module on ``device``."""

        forward_module = copy.deepcopy(quant_source)
        forward_module = forward_module.to(device=device)
        forward_module.eval()
        setattr(forward_module, "target_device", torch.device(device))
        return forward_module

    def _infer_fp8_forward_layout(
        self,
        *,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
    ) -> tuple[str, Optional[tuple[int, int]]]:
        """Infer the FP8 scale layout needed to rebuild a TorchFP8Linear wrapper."""

        if scale_inv.numel() == 1:
            return "tensor", None
        if scale_inv.ndim == 1 and scale_inv.shape[0] == weight.shape[0]:
            return "row", None
        return "block", infer_block_shape(tuple(weight.shape), scale_inv)

    def _infer_fp4_forward_block_size(
        self,
        *,
        target_submodule: nn.Module,
        scale: torch.Tensor,
    ) -> int:
        """Infer the NVFP4 block size used along the input-feature axis."""

        block_size = self._decoder_block_size()
        if block_size is not None and target_submodule.in_features % block_size[1] == 0:
            return int(block_size[1])

        if scale.ndim >= 1 and scale.shape[-1] > 0 and target_submodule.in_features % scale.shape[-1] == 0:
            return int(target_submodule.in_features // scale.shape[-1])

        raise ValueError(
            f"Cannot infer FP4 block size for in_features={target_submodule.in_features} "
            f"and scale shape={tuple(scale.shape)}."
        )

    def _build_fp8_forward_module(
        self,
        *,
        target_submodule: nn.Module,
        checkpoint_tensors: Dict[str, torch.Tensor],
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> Optional[nn.Module]:
        """Rebuild one linear submodule as a TorchFP8Linear forward wrapper."""

        if not isinstance(target_submodule, nn.Linear):
            return None

        weight = checkpoint_tensors.get("weight")
        if not isinstance(weight, torch.Tensor):
            return None

        scale_inv = self._decoder_scale_tensor(
            scale_tensor=checkpoint_tensors.get("weight_scale_inv"),
            result_shape=tuple(weight.shape),
        )
        if not isinstance(scale_inv, torch.Tensor):
            # ModelOpt-style FP8 checkpoints store direct scales instead of inverse scales;
            # normalize them here so TorchFP8Linear can use one consistent metadata form.
            scale = self._decoder_scale_tensor(
                scale_tensor=checkpoint_tensors.get("weight_scale"),
                result_shape=tuple(weight.shape),
            )
            if not isinstance(scale, torch.Tensor):
                return None
            scale = scale.to(torch.float32)
            tiny = torch.finfo(torch.float32).tiny
            scale_inv = torch.where(
                scale != 0,
                torch.reciprocal(scale),
                torch.full_like(scale, 1.0 / tiny),
            )

        format_name = str(weight.dtype).split(".")[-1]
        try:
            # Infer the wrapper layout from the normalized inverse-scale tensor so native
            # FP8 execution works for either checkpoint convention.
            weight_scale_method, weight_block_size = self._infer_fp8_forward_layout(
                weight=weight,
                scale_inv=scale_inv,
            )
            forward_module = TorchFP8Linear(
                bits=8,
                group_size=-1,
                desc_act=False,
                sym=True,
                in_features=target_submodule.in_features,
                out_features=target_submodule.out_features,
                bias=target_submodule.bias is not None,
                pack_dtype=torch.int32,
                format=format_name,
                weight_scale_method=weight_scale_method,
                weight_block_size=weight_block_size,
                register_buffers=False,
            ).to(device=device)
        except Exception:
            # Some checkpoints use padded or otherwise non-TorchFP8Linear layouts and must
            # fall back to the decoded dense path even on native-FP8-capable GPUs.
            return None
        forward_module.register_buffer("weight", weight.to(device=device))
        forward_module.register_buffer(
            "weight_scale_inv",
            scale_inv.to(device=device, dtype=torch.float32),
        )

        bias = checkpoint_tensors.get("bias")
        if isinstance(bias, torch.Tensor):
            forward_module.register_buffer(
                "bias",
                bias.to(device=device, dtype=target_dtype),
            )
        else:
            forward_module.bias = None

        forward_module.eval()
        setattr(forward_module, "target_device", torch.device(device))
        return forward_module

    def _build_fp4_forward_module(
        self,
        *,
        target_submodule: nn.Module,
        checkpoint_tensors: Dict[str, torch.Tensor],
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> Optional[nn.Module]:
        """Rebuild one linear submodule as a native NVFP4 forward wrapper."""

        if not isinstance(target_submodule, nn.Linear):
            return None

        weight = checkpoint_tensors.get("weight")
        scale = self._decoder_fp4_effective_scale(
            checkpoint_tensors=checkpoint_tensors,
            result_shape=(target_submodule.out_features, target_submodule.in_features),
        )
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            return None

        try:
            block_size = self._infer_fp4_forward_block_size(
                target_submodule=target_submodule,
                scale=scale,
            )
            forward_module = TorchFP4Linear(
                in_features=target_submodule.in_features,
                out_features=target_submodule.out_features,
                weight=weight.to(device=device),
                weight_scale=scale.to(device=device),
                weight_block_size=block_size,
                orig_dtype=target_dtype,
                bias=checkpoint_tensors.get("bias").to(device=device, dtype=target_dtype)
                if isinstance(checkpoint_tensors.get("bias"), torch.Tensor)
                else None,
            )
        except Exception:
            return None

        forward_module.eval()
        setattr(forward_module, "target_device", torch.device(device))
        return forward_module

    def _record_auto_module_decoder_event(
        self,
        *,
        named_module: "NamedModule",
        device: torch.device,
        forward_mode: str,
        source_dtype: torch.dtype,
        target_dtype: torch.dtype,
    ) -> None:
        """Store one auto-decoder decision so tests can assert the chosen path."""

        if named_module.state.get("_auto_module_decoder_event_recorded"):
            return

        self.auto_module_decoder_events.append(
            {
                "module": named_module.full_name,
                "device": str(device),
                "forward_mode": forward_mode,
                "source_dtype": str(source_dtype).split(".")[-1],
                "target_dtype": str(target_dtype).split(".")[-1],
            }
        )
        named_module.state["_auto_module_decoder_event_recorded"] = True

    def _prepare_auto_decoder_forward_module(
        self,
        *,
        target_submodule: nn.Module,
        device: torch.device,
        named_module: "NamedModule",
    ) -> nn.Module:
        """Swap one decoded shell module to an FP8 forward view when supported."""

        decoder_plan = named_module.state.get("auto_module_decoder")
        turtle_model = self.turtle_model
        if not isinstance(decoder_plan, dict) or turtle_model is None:
            return target_submodule

        checkpoint_tensors = turtle_model.checkpoint_tensors_for_submodule(
            target_model=self.model,
            target_submodule=target_submodule,
            recurse=False,
        )
        weight = checkpoint_tensors.get("weight")
        if not isinstance(weight, torch.Tensor):
            return target_submodule

        decoder_kind = self._decoder_weight_format(
            weight=weight,
            checkpoint_tensors=checkpoint_tensors,
        )
        if decoder_kind is None:
            return target_submodule

        target_dtype = decoder_plan.get("target_dtype", target_submodule.weight.dtype)
        forward_policy = str(decoder_plan.get("passthrough_forward_policy", "native")).strip().lower()
        if not isinstance(named_module.state.get("quant_source_module"), nn.Module):
            named_module.state["quant_source_module"] = self._build_decoder_quant_source_module(
                target_submodule,
                checkpoint_tensors=checkpoint_tensors,
                target_dtype=target_dtype,
            )

        forward_mode = "decode"
        replacement = target_submodule
        if forward_policy != "decode" and decoder_kind == "fp8" and device_supports_dtype(device, weight.dtype, require_validation=False):
            fp8_module = self._build_fp8_forward_module(
                target_submodule=target_submodule,
                checkpoint_tensors=checkpoint_tensors,
                device=device,
                target_dtype=target_dtype,
            )
            if fp8_module is not None:
                replacement = self._replace_live_submodule(target_submodule, fp8_module)
                forward_mode = "native"
        elif forward_policy != "decode" and decoder_kind == "fp4" and device_supports_native_fp4(device, require_validation=False):
            fp4_module = self._build_fp4_forward_module(
                target_submodule=target_submodule,
                checkpoint_tensors=checkpoint_tensors,
                device=device,
                target_dtype=target_dtype,
            )
            if fp4_module is not None:
                replacement = self._replace_live_submodule(target_submodule, fp4_module)
                forward_mode = "native"
        if forward_mode == "decode":
            decoded_forward = self._build_decoder_forward_module(
                quant_source=named_module.state["quant_source_module"],
                device=device,
            )
            replacement = self._replace_live_submodule(target_submodule, decoded_forward)

        named_module.state["auto_module_decoder_forward_mode"] = forward_mode
        self._record_auto_module_decoder_event(
            named_module=named_module,
            device=torch.device(device),
            forward_mode=forward_mode,
            source_dtype=weight.dtype,
            target_dtype=target_dtype,
        )
        return replacement

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
        if isinstance(module_kwargs, dict):
            per_feature_kwargs = module_kwargs.get("_awq_feature_kwargs", {})
            base_module_kwargs = {
                key: value
                for key, value in module_kwargs.items()
                if key != "_awq_feature_kwargs"
            }
        else:
            per_feature_kwargs = {}
            base_module_kwargs = module_kwargs

        if self.model.config is not None and self.dynamic_expert_index is not None:
            self.get_num_experts(self.model.config)

        def strip_non_quantize_flags(module_name):
            for flag in NON_QUANTIZE_FLAGS:
                if flag in module_name:
                    module_name = module_name.replace(flag, "")
            return module_name

        def _select_feature_name(names):
            """Return the first quantized child that has captured activations."""
            for raw in names:
                stripped = strip_non_quantize_flags(raw)
                if stripped in input_feat:
                    return stripped
            return strip_non_quantize_flags(names[0]) if names else None

        def _try_update_last_module(candidate_name: str) -> bool:
            nonlocal last_module, last_module_name, last_module_root

            resolved_module, _ = get_module_by_name_prefix(module, candidate_name)
            if resolved_module is None:
                log.debug(
                    "awq_get_modules_for_scaling: last-module candidate `%s` missing; retaining previous `%s`",
                    candidate_name,
                    last_module_name,
                )
                return False

            last_module = resolved_module
            last_module_name = candidate_name
            if "." in candidate_name:
                last_module_root = candidate_name.split(".", 1)[0]
            return True

        def _module_kwargs_for_feature(feature_name: str | None):
            kwargs_for_feature = dict(base_module_kwargs)
            if feature_name and isinstance(per_feature_kwargs, dict):
                feature_specific_kwargs = per_feature_kwargs.get(feature_name)
                if isinstance(feature_specific_kwargs, dict):
                    kwargs_for_feature.update(feature_specific_kwargs)
            return kwargs_for_feature

        full_layer_modules = self.full_layer_modules(
            self.model.config,
            is_awq_quantize=True,
            include_capture_only=True,
        )
        for i, block in enumerate(full_layer_modules):
            not_quantized = all(any(flag in name for flag in NON_QUANTIZE_FLAGS) for name in block)
            if not_quantized:
                # If both the current block and the previous one are marked as not quantized,
                # skip remembering the current block. This ensures that when two consecutive
                # blocks are not quantized, only the first one is remembered as last_module.
                if i > 0 and all(any(flag in name for flag in NON_QUANTIZE_FLAGS) for name in full_layer_modules[i - 1]):
                    continue

                # Remember the latest norm (use the last entry if multiple are present)
                candidate_name = strip_non_quantize_flags(block[-1])
                _try_update_last_module(candidate_name)
                continue

            is_moe_block = any(any(k in name for k in self.moe_expert_module_name_prefixes) for name in block)
            is_moe_down_block = is_moe_block and any("down" in name for name in block)
            is_moe_gate_up_block = is_moe_block and any("gate" in name for name in block) and any("up" in name for name in block)
            if is_moe_down_block and last_module is not None and last_module_name is not None:
                # mlp.experts.0.down_proj
                target_suffix = last_module_name.split(".")[-1]
                for name in block:
                    prev_op_name = ".".join(name.split(".")[:-1] + [target_suffix])
                    prev_op, _ = get_module_by_name_prefix(module, prev_op_name)
                    if prev_op is None or name not in input_feat:
                        log.debug("awq_get_modules_for_scaling: skipping expert `%s` due to missing prev_op or features", name)
                        continue

                    m, _ = get_module_by_name_prefix(module, name)
                    if m is None:
                        log.debug("awq_get_modules_for_scaling: skipping missing expert module `%s`", name)
                        continue
                    subset = [m]
                    feature_name = name
                    n, root = generate_node_for_awq_scaling(inp=input_feat[name], prev_op=prev_op,
                                                            module_kwargs=_module_kwargs_for_feature(feature_name), nodes_size=len(nodes),
                                                            subset=subset, module2inspect=None)
                    if root is not None and last_module_root != root:
                        last_module_root = root

                    nodes.append(n)
            else:
                # Normal execution subset
                subset = []  # preserve execution order while collecting quantizable modules
                skip = False
                for name in block:
                    if all(flag not in name for flag in NON_QUANTIZE_FLAGS):
                        m, _ = get_module_by_name_prefix(module, name)
                        # If the Model uses GQA (Grouped Query Attention), attention out will be skipped.
                        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                        if (self.awq_scale_optimize_shape_dependent_modules is not None
                                and name in self.awq_scale_optimize_shape_dependent_modules
                                and isinstance(last_module, nn.Linear)
                                and last_module.weight.shape != m.weight.shape):
                            # log.debug(f'"{name}" attention out skipped.')
                            skip = True

                        if m is None:
                            log.debug("awq_get_modules_for_scaling: skipping missing module `%s`", name)
                            skip = True
                            break
                        subset.append(m)

                if skip or not subset:
                    continue

                prev_op = last_module
                if prev_op is None:
                    log.debug("awq_get_modules_for_scaling: skipping block %s due to missing previous module", block)
                    continue

                # Match the activation bucket to the first quantized child in this block
                feature_name = _select_feature_name(block) or strip_non_quantize_flags(block[0])
                root_split = feature_name.split(".")
                module2inspect = None
                if len(root_split) >= 2:
                    root = root_split[0]
                    if root != last_module_root:
                        last_module_root = root
                        module2inspect, _ = get_module_by_name_prefix(module, root)

                # process ['mlp.experts.#.gate_proj', 'mlp.experts.#.gup_proj']
                if is_moe_gate_up_block and module2inspect is not None:
                    if last_module_root not in input_feat:
                        log.debug(
                            "awq_get_modules_for_scaling: missing input feature for `%s` while processing experts block (layer block size=%s)",
                            last_module_root,
                            len(block),
                        )
                    feature_name = last_module_root if last_module_root in input_feat else _select_feature_name(block)
                    inp = input_feat.get(last_module_root, input_feat.get(_select_feature_name(block)))
                else:
                    feature_name = _select_feature_name(block)
                    inp = input_feat.get(feature_name)

                if inp is None:
                    log.debug("awq_get_modules_for_scaling: skipping block %s due to missing input features", block)
                    continue

                n, root = generate_node_for_awq_scaling(inp=inp, prev_op=prev_op,
                                                        module_kwargs=_module_kwargs_for_feature(feature_name), nodes_size=len(nodes),
                                                        subset=subset, module2inspect=module2inspect)

                nodes.append(n)

            # Update tracker to the LAST item of this block
            if is_moe_gate_up_block:
                # Mixed MoE blocks can legitimately place shared-expert projections
                # before or after routed experts depending on real forward order.
                # For AWQ scaling, we still want the last routed expert gate/up proj
                # as the effective boundary for the expert segment in this block.
                gate_up_proj_indices = [
                    i for i, name in enumerate(block)
                    if any(k in name for k in self.moe_expert_module_name_prefixes) and ("gate" in name or "up" in name)
                ]

                # Use the last one if any exist
                assert len(gate_up_proj_indices) > 0, "No expert gate_proj/up_proj found in block."
                last_up_proj_index = gate_up_proj_indices[-1]

                candidate_name = strip_non_quantize_flags(block[last_up_proj_index])
                assert "gate" in candidate_name or "up" in candidate_name
            else:
                candidate_name = strip_non_quantize_flags(block[-1])
            _try_update_last_module(candidate_name)

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

    # Materialize the target shell module from the lazy turtle source on the requested device.
    def shell_module_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: torch.device,
            non_blocking: bool = False,
            role: str = "default",
            named_module: Optional["NamedModule"] = None,
    ) -> torch.nn.Module:
        with self._turtle_lock:
            if role == "quant_source" and named_module is not None:
                quant_source = named_module.state.get("quant_source_module")
                if not isinstance(quant_source, nn.Module):
                    decoder_plan = named_module.state.get("auto_module_decoder") or {}
                    target_dtype = decoder_plan.get(
                        "target_dtype",
                        getattr(getattr(target_submodule, "weight", None), "dtype", torch.float16),
                    )
                    checkpoint_tensors = None
                    if isinstance(self.turtle_model, LazyTurtle):
                        checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                            target_model=self.model,
                            target_submodule=target_submodule,
                            recurse=False,
                        )
                    quant_source = self._build_decoder_quant_source_module(
                        target_submodule,
                        checkpoint_tensors=checkpoint_tensors,
                        target_dtype=target_dtype,
                    )
                    named_module.state["quant_source_module"] = quant_source

                module = self._replace_live_submodule(target_submodule, quant_source)
                if get_device(module) != device:
                    module.to(device)
                return module

            turtle_model = self.turtle_model
            if role == "forward" and named_module is not None and isinstance(turtle_model, LazyTurtle):
                checkpoint_tensors = turtle_model.checkpoint_tensors_for_submodule(
                    target_model=self.model,
                    target_submodule=target_submodule,
                    recurse=False,
                )
                weight = checkpoint_tensors.get("weight")
                if isinstance(weight, torch.Tensor):
                    decoder_kind = self._decoder_weight_format(
                        weight=weight,
                        checkpoint_tensors=checkpoint_tensors,
                    )
                    if decoder_kind is not None:
                        # Packed floatx checkpoints can require decoder-specific
                        # materialization before any dense shell weight exists.
                        return self._prepare_auto_decoder_forward_module(
                            target_submodule=target_submodule,
                            device=torch.device(device),
                            named_module=named_module,
                        )

            if turtle_model is None:
                if get_device(target_submodule) != device:
                    target_submodule.to(device)
                module = target_submodule
            else:
                module = alias_from_turtle_for_submodule(
                    target_model=self.model,
                    turtle_model=turtle_model,
                    target_submodule=target_submodule,
                    device=device,
                )

            if role == "forward" and named_module is not None:
                module = self._prepare_auto_decoder_forward_module(
                    target_submodule=module,
                    device=torch.device(device),
                    named_module=named_module,
                )
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
    def build_layer_modules(cls, tree, include_capture_only: bool = False):
        """
        tree format:
          [<model_name>, <submodule>, "#", { parent_module: ( "child[:!][:grp]", ... ), ... }]
        Rules:
          - ':!' means participates in inference but is NOT quantized; keep this marker in output.
          - ':?' marks capture-only nodes; activations are recorded but the module is not quantized.
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
        alias_groups: Dict[tuple[str | None, int], List[tuple[str, bool, bool]]] = {}
        alias_meta: Dict[tuple[str | None, int], Dict[str, int]] = {}
        alias_seq = count()
        group_seq = count()

        def _parse_token(token: str) -> tuple[str, List[str]]:
            return cls._parse_module_flags(token)

        def _group_from_flags(flags: List[str]) -> int:
            for flag in flags:
                if flag.isdigit():
                    return int(flag)
            return 0

        def _has_numeric_flag(flags: List[str]) -> bool:
            return any(flag.isdigit() for flag in flags)

        def _get_scope(parent_name: str) -> str | None:
            if not parent_name:
                return None
            return parent_name.split(".", 1)[0]

        def process_entries(parent_token: str, entries, parent_group_offset: int = 0, scope_key: str | None = None):
            """Process entries recursively to handle nested dict structures for MoE"""
            groups: defaultdict[int, List[tuple]] = defaultdict(list)

            parent_name, parent_flags = _parse_token(parent_token)
            parent_rel_group = _group_from_flags(parent_flags)
            parent_group = parent_group_offset + parent_rel_group
            parent_has_bang = "!" in parent_flags
            parent_capture_only = "?" in parent_flags
            parent_has_numeric = _has_numeric_flag(parent_flags)

            scope = scope_key if scope_key is not None else _get_scope(parent_name)
            parent_alias_scope = scope if parent_has_numeric else parent_name

            def _make_entry(full_path: str, has_bang: bool, capture_only: bool, *, alias_base: int, alias_rel: int, alias_scope: str | None) -> tuple:
                return (full_path, has_bang, capture_only, alias_scope, (alias_base, alias_rel))

            child_group_offset = parent_group_offset
            add_parent = parent_has_bang or (parent_capture_only and include_capture_only)
            if add_parent:
                alias_base = parent_rel_group if parent_has_numeric else parent_group
                parent_entry_scope = f"{parent_alias_scope}.__parent__" if parent_alias_scope is not None else None
                groups[parent_group].append(
                    _make_entry(
                        parent_name,
                        parent_has_bang,
                        parent_capture_only,
                        alias_base=alias_base,
                        alias_rel=0,
                        alias_scope=parent_entry_scope,
                    )
                )
                child_group_offset = max(child_group_offset, parent_group + 1)

            # Handle tuple/list of strings (traditional format)
            if isinstance(entries, (tuple, list)):
                for ent in entries:
                    child_name, child_flags = _parse_token(ent)

                    has_bang = "!" in child_flags
                    capture_only = "?" in child_flags
                    # first numeric tag is the group id; default 0
                    child_rel_group = _group_from_flags(child_flags)
                    grp = child_group_offset + child_rel_group
                    # Apply parent group offset to avoid conflicts between different nesting levels
                    # Store the full path including parent for later use
                    if parent_name.endswith(f".{child_name}") or parent_name == child_name:
                        full_path = parent_name
                    elif parent_name:
                        full_path = f"{parent_name}.{child_name}"
                    else:
                        full_path = child_name

                    if capture_only and not include_capture_only:
                        continue
                    alias_scope = scope if parent_has_numeric else parent_name
                    alias_base = parent_rel_group if parent_has_numeric else grp
                    alias_rel = child_rel_group if parent_has_numeric else 0
                    groups[grp].append(
                        _make_entry(
                            full_path,
                            has_bang,
                            capture_only,
                            alias_base=alias_base,
                            alias_rel=alias_rel,
                            alias_scope=alias_scope,
                        )
                    )

            elif isinstance(entries, dict):
                # Calculate max group number used at current level to avoid conflicts
                max_current_group = 0
                for sub_parent, sub_entries in entries.items():
                    if isinstance(sub_entries, (tuple, list)):
                        for ent in sub_entries:
                            _, ent_flags = _parse_token(ent)
                            max_current_group = max(max_current_group, _group_from_flags(ent_flags))

                # Process nested entries with appropriate group offset
                current_offset = child_group_offset
                for sub_parent, sub_entries in entries.items():
                    if sub_parent == "#":
                        # Special case: "#" means expert index placeholder
                        # Create a template path that will be expanded later by simple_layer_modules
                        template_parent = (
                            f"{parent_name}.{EXPERT_INDEX_PLACEHOLDER}"
                            if parent_name else EXPERT_INDEX_PLACEHOLDER
                        )
                        template_parent_token = (
                            f"{template_parent}:{parent_rel_group}"
                            if parent_has_numeric
                            else template_parent
                        )
                        # Use a higher offset for expert modules to avoid conflicts with parent level
                        expert_offset = current_offset + max_current_group + 100  # Large offset to avoid conflicts

                        # Handle special case where sub_entries is ("#",) or "#" - this means use the parent path directly
                        if (isinstance(sub_entries, (tuple, list)) and len(sub_entries) == 1 and sub_entries[0] == "#") or sub_entries == "#":
                            # For ("#",) or "#" format, use the template_parent directly with default group 0
                            alias_scope = scope if parent_has_numeric else template_parent
                            alias_base = parent_rel_group if parent_has_numeric else expert_offset
                            groups[expert_offset].append(
                                _make_entry(
                                    template_parent,
                                    False,
                                    False,
                                    alias_base=alias_base,
                                    alias_rel=0,
                                    alias_scope=alias_scope,
                                )
                            )
                        else:
                            sub_groups = process_entries(template_parent_token, sub_entries, expert_offset, scope)
                            for grp, items in sub_groups.items():
                                groups[grp].extend(items)
                    else:
                        # Nested structure: process recursively with full path
                        # Special case: empty string key means use parent path directly
                        if sub_parent == "":
                            full_sub_parent = parent_name
                        else:
                            full_sub_parent = (
                                f"{parent_name}.{sub_parent}"
                                if parent_name else sub_parent
                            )
                        sub_groups = process_entries(full_sub_parent, sub_entries, current_offset, scope)
                        for grp, items in sub_groups.items():
                            groups[grp].extend(items)
                        # Update offset for next sibling to avoid conflicts
                        if sub_groups:
                            current_offset = max(sub_groups.keys()) + 1

            return groups

        def _register_alias(order_idx: int, item: tuple[str, bool, bool, str | None, tuple[int, int]]):
            full_path, has_bang, capture_only, scope, alias_parts = item
            if capture_only and not include_capture_only:
                return
            alias_scope = scope
            alias_base, alias_rel = alias_parts
            alias_index = alias_base + alias_rel
            key = (alias_scope, alias_index)
            meta = alias_meta.get(key)
            if meta is None:
                alias_meta[key] = {"order": order_idx, "seq": next(alias_seq)}
                alias_groups[key] = [(full_path, has_bang, capture_only)]
            else:
                meta["order"] = min(meta["order"], order_idx)
                alias_groups[key].append((full_path, has_bang, capture_only))

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)

            for g in sorted(groups):
                order_idx = next(group_seq)
                items = groups[g]
                for item in items:
                    if len(item) == 3:
                        full_path, has_bang, capture_only = item
                        scope = full_path
                        alias_parts = (g, 0)
                        _register_alias(order_idx, (full_path, has_bang, capture_only, scope, alias_parts))
                    else:
                        _register_alias(order_idx, item)

        for key in sorted(alias_groups.keys(), key=lambda k: (alias_meta[k]["order"], alias_meta[k]["seq"])):
            block = []
            for full_path, has_bang, capture_only in alias_groups[key]:
                name = full_path
                if has_bang:
                    name += NOT_QUANTIZE_FLAG
                if capture_only and include_capture_only:
                    name += CAPTURE_ONLY_FLAG
                block.append(name)
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
        root_path = [cls._parse_module_flags(node)[0] if isinstance(node, str) else node for node in tree[:sharp_idx-1]]

        out = []
        # Traverse each layer in root_path
        for i in range(len(root_path)):
            path = root_path[:i + 1]
            base = model
            exclude = cls._parse_module_flags(tree[len(path)])[0] if isinstance(tree[len(path)], str) else tree[len(path)]

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
          1. Drop dict entries where *all* values are ':!' or ':?' flagged.
          2. Remove ':!' / ':?' and ':<digit>' markers from strings.
        """

        # If it's a list, recurse into each element
        if isinstance(node, list):
            return [self.generate_layers_modules_tree_simple(x) for x in node]

        # If it's a dict, process each key -> value
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                clean_key = self._parse_module_flags(k)[0] if isinstance(k, str) else k
                # Expand tuple-of-strings blocks (special handling)
                if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                    # Rule 1: check if ALL entries are :!
                    if all(any(p in {"!", "?"} for p in x.split(":")[1:]) for x in v):
                        continue  # skip this parent entirely

                    # Rule 2: strip :! and :digit markers
                    cleaned = tuple(self._parse_module_flags(x)[0] for x in v)
                    new_dict[clean_key] = cleaned
                else:
                    # Recurse deeper
                    new_dict[clean_key] = self.generate_layers_modules_tree_simple(v)
            return new_dict

        # If it's a plain string (unlikely here), strip markers
        if isinstance(node, str):
            return self._parse_module_flags(node)[0]

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

    def _auto_detect_module_tree(self, model: PreTrainedModel, quant_method: METHOD):
        log.warn("Model not yet support, attempting Module Tree AutoCompat...")

        if quant_method not in {METHOD.GPTQ, METHOD.GGUF, METHOD.FP8, METHOD.BITSANDBYTES, METHOD.EXL3, METHOD.PARO}:
            log.warn(
                f"Module Tree AutoCompat: Failed, quant_method={quant_method}, "
                "only support GPTQ/GGUF/FP8/BITSANDBYTES/EXL3/PAROQUANT"
            )
            return None

        def _get(path):
            base = model
            for p in path.split("."):
                base = getattr(base, p, None)
                if base is None:
                    return None
            return base

        candidates = [
            "model.layers",
            "language_model.layers",
            "model.decoder.layers",
            "transformer.h",
            "transformer.blocks",
            "layers",
            "blocks",
            "model.blocks",
        ]

        chosen = None
        for c in candidates:
            m = _get(c)
            if isinstance(m, (nn.ModuleList, list, tuple)) and len(m) > 0 and isinstance(m[0], nn.Module):
                chosen = c
                log.warn(f"Module Tree AutoCompat: Matched candidate path '{c}', type={type(m).__name__}")
                break

        if chosen is None:
            log.warn("Module Tree AutoCompat: All candidate paths invalid, return None")
            return None

        layer0 = _get(chosen)[0]
        log.warn(f"Module Tree AutoCompat: Using layer0: {type(layer0).__name__}")

        def _linear_names(module):
            mods = find_modules(module, layers=[nn.Linear, nn.Conv1d, nn.Conv2d])
            log.warn(f"Module Tree AutoCompat: _linear_names: found {len(mods)} Linear/Conv modules in {type(module).__name__}")
            return list(mods.keys())

        all_linear = _linear_names(layer0)
        if len(all_linear)>0:
            log.warn(f"Module Tree AutoCompat: found {len(all_linear)} Linear/Conv modules in {type(layer0).__name__}: {all_linear}")
        else:
            log.warn("Module Tree AutoCompat: No Linear/Conv names in layer0, return None")
            return None

        mapping = {}

        def _find_parents(module, possible_names):
            found = set()
            for n, _ in module.named_children():
                l = n.lower()
                if any(k in l for k in possible_names):
                    found.add(n)
            return found

        def _leaf_tokens(prefix):
            return tuple(x.split(".")[-1] for x in all_linear if x.startswith(f"{prefix}."))

        possible_parent = ["attn", "attention", "self_attn", "mlp", "ffn", "feed", "dense"]

        found_parents = _find_parents(layer0, possible_parent)

        for p in found_parents:
            t = _leaf_tokens(p)
            if t:
                mapping[p] = t

        if not mapping:
            blocks = tuple(n.split(".")[-1] for n in all_linear)
            mapping[""] = blocks
            log.warn(f"Module Tree AutoCompat: Mapping empty, using all Linear as fallback: {blocks}")

        parts = chosen.split(".")
        tree = parts + ["#", mapping]
        log.warn(f"Module Tree AutoCompat: Final module_tree: {tree}")
        return tree

__all__ = ["BaseQModel"]

BaseQModel = ModelLoader(ModelWriter(BaseQModel))
