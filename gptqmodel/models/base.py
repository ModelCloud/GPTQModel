# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import count
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Tuple, Type, Union

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
    QuantizeEmbed,
    QuantizeEmbedConfig,
    ShardStrategy,
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
from ..utils.disk_telemetry import disk_telemetry
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import QuantizationRegionTimer, setup_logger
from ..utils.looper_helpers import normalize_device_like
from ..utils.model import (
    MODALITY,
    _module_has_meta_tensors,
    find_modules,
    get_layers_with_prefixes,
    get_module,
    get_module_by_name_prefix,
    get_module_name,
    move_to,
)
from ..utils.model_dequant import infer_block_shape
from ..utils.structure import (
    LazyTurtle,
    _get_parent_and_leaf_by_path,
    _get_qualified_name,
    alias_direct_meta_from_turtle_for_submodule,
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
from .loader import ModelLoader, _setup_rotation_online_had
from .writer import ModelWriter


if TYPE_CHECKING:
    try:
        from datasets import Dataset as HFDatasetType
        from datasets import IterableDataset as HFIterableDatasetType
    except Exception:  # pragma: no cover - optional dependency
        HFDatasetType = HFIterableDatasetType = object

    from ..looper.named_module import NamedModule


class _QuantizedCheckpointSource:
    """Minimal shard-map source over an on-disk quantized checkpoint.

    Provides the `_weight_map`/`model_local_path` interface the embedding
    replacement save path expects when no LazyTurtle model is available
    (e.g. models loaded via `from_quantized`).
    """

    def __init__(self, model_local_path: str):
        self.model_local_path = model_local_path
        index_path = os.path.join(model_local_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Checkpoint index at `{index_path}` has an empty or invalid `weight_map`.")
            self._weight_map = {str(k): str(v) for k, v in weight_map.items()}
        else:
            single_shard = os.path.join(model_local_path, "model.safetensors")
            if not os.path.exists(single_shard):
                raise FileNotFoundError(
                    f"No `model.safetensors.index.json` or `model.safetensors` found under `{model_local_path}`."
                )
            from safetensors import safe_open

            with safe_open(single_shard, framework="pt", device="cpu") as handler:
                self._weight_map = {name: "model.safetensors" for name in handler.keys()}


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

MODULE_TREE_FLAG_MOE = "moe"
MODULE_TREE_FLAG_ROUTED = "routed"
MODULE_TREE_FLAG_SHARED = "shared"
MODULE_TREE_FLAG_GATE = "gate"
MODULE_TREE_FLAG_UP = "up"
MODULE_TREE_FLAG_DOWN = "down"
MODULE_TREE_FLAG_Q = "q"
MODULE_TREE_FLAG_K = "k"
MODULE_TREE_FLAG_V = "v"
MODULE_TREE_FLAG_O = "o"
MODULE_TREE_ATTENTION_FLAGS = frozenset(
    {MODULE_TREE_FLAG_Q, MODULE_TREE_FLAG_K, MODULE_TREE_FLAG_V, MODULE_TREE_FLAG_O}
)
MODULE_TREE_EXPERT_FLAGS = frozenset({MODULE_TREE_FLAG_ROUTED, MODULE_TREE_FLAG_SHARED})
MODULE_TREE_MOE_FLAGS = frozenset({MODULE_TREE_FLAG_MOE, *MODULE_TREE_EXPERT_FLAGS})
MODULE_TREE_PROJECTION_FLAGS = frozenset(
    {MODULE_TREE_FLAG_GATE, MODULE_TREE_FLAG_UP, MODULE_TREE_FLAG_DOWN}
)


@dataclasses.dataclass(frozen=True)
class ModuleTreeMetadata:
    """Normalized semantic metadata emitted by one explicit module-tree declaration."""

    flags: frozenset[str] = frozenset()
    expert_group: Optional[str] = None


def normalize_module_tree_flags(flags) -> frozenset[str]:
    """Normalize structural tags and reject ambiguous expert declarations."""

    normalized = frozenset(flags or ())
    expert_roles = normalized & MODULE_TREE_EXPERT_FLAGS
    if len(expert_roles) > 1:
        raise ValueError(f"Module-tree node cannot be both routed and shared: {sorted(normalized)}")
    if expert_roles and MODULE_TREE_FLAG_MOE in normalized:
        normalized = normalized - {MODULE_TREE_FLAG_MOE}
    return normalized


def module_tree_flags_are_moe(flags) -> bool:
    """Return whether explicit structural tags place a module inside an MoE path."""

    return bool(frozenset(flags or ()) & MODULE_TREE_MOE_FLAGS)


def module_tree_flags_are_expert(flags) -> bool:
    """Return whether explicit structural tags identify a routed or shared expert."""

    return bool(frozenset(flags or ()) & MODULE_TREE_EXPERT_FLAGS)


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

    # Cache of role/semantic module-tree flags keyed by the path within a layer.
    # Populated by ``_build_layer_modules_for_tree`` and consulted by the looper
    # when constructing ``NamedModule`` wrappers.
    _module_tree_metadata_cache: ClassVar[Dict[Type["BaseQModel"], Dict[str, ModuleTreeMetadata]]] = {}

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

    # Whether this model should publish a layer's KV tuple into the shared
    # replay cache even when that same layer does not consume `kv_last_layer`.
    # Models like "hymba" need some layers to write KV for later layers even if
    # the current layer itself does not consume `kv_last_layer`.
    write_shared_kv_cache = False

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

    # Module paths that own meta params/buffers directly, not under child modules.
    modules_with_direct_meta_tensors: List[str] = []
    direct_meta_modules: List[str] = []  # Backward-compatible alias.

    server = None

    support_offload_to_disk = True
    # Optional runtime->checkpoint overrides for LazyTurtle. Prefer reversed
    # `WeightRenaming` entries; legacy runtime->checkpoint dicts are still accepted.
    HF_CONVERSION_MAP_REVERSED: Optional[Any] = None

    ATTENTION_MASKS_DTYPE = torch.bool # default to bool

    ATTENTION_MASKS_REQUIRED_FOR_INPUT: bool = False

    INPUT_EMBEDDING_EXTRA_ARGS = None

    # Some models (e.g. ovis2_6_moe) do not contain MoE layers directly.
    # The actual experts live inside submodules (e.g. Qwen3MoeModel.mlp.experts),
    # so `defuser_module_paths` is used to explicitly locate and defuse them.
    defuser_module_paths = None

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
        embeddings_module_quantized: bool = False,
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
        self.embeddings_module_quantized = embeddings_module_quantized
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
        It returns layer paths from each complete module_tree variant.
        Example:
            ["model", "layers", "#", {...}] -> ["model.layers"]
            [
                ["model", "L_module", "layers", "#", {...}],
                ["model", "H_module", "layers", "#", {...}],
            ]
                -> ["model.L_module.layers", "model.H_module.layers"]
        """
        paths = []
        for tree in cls._iter_module_tree_variants():
            for path in cls._expand_module_tree_prefixes(tree):
                if path not in paths:
                    paths.append(path)
        return paths

    @classmethod
    def _iter_module_tree_variants(cls, module_tree=None) -> List[List[Any]]:
        """Normalize module_tree into one or more complete tree variants."""

        tree = cls.module_tree if module_tree is None else module_tree
        if not isinstance(tree, list):
            return []

        if tree and all(isinstance(item, (list, tuple)) for item in tree):
            return [list(item) for item in tree]

        return [list(tree)]

    @classmethod
    def _expand_module_tree_prefixes(cls, tree=None) -> List[str]:
        """Return the module_tree prefix before `#` as a concrete path."""

        tree = cls.module_tree if tree is None else tree
        if tree is None:
            return []

        path = []
        for node in tree:
            if node == "#":
                break
            if not isinstance(node, str):
                break

            path.append(node.split(":", 1)[0])

        return [".".join(path)] if path else []

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
    def _set_module_tree_flags(
        cls,
        path: str,
        flags: frozenset,
        *,
        expert_group: Optional[str] = None,
    ) -> None:
        """Store role/semantic flags for a module-tree path (within a decoder layer)."""

        normalized = normalize_module_tree_flags(flags)
        if expert_group is not None and not module_tree_flags_are_expert(normalized):
            raise ValueError(
                f"Module-tree expert group `{expert_group}` requires an explicit routed/shared tag on `{path}`."
            )
        cache = cls._module_tree_metadata_cache.setdefault(cls, {})
        if path not in cache:
            cache[path] = ModuleTreeMetadata(flags=normalized, expert_group=expert_group)

    @classmethod
    def get_module_tree_metadata(cls, path: str) -> ModuleTreeMetadata:
        """Return normalized semantic metadata for ``path`` within a decoder layer."""

        cache = cls._module_tree_metadata_cache.get(cls, {})
        return cache.get(path, ModuleTreeMetadata())

    @classmethod
    def get_module_tree_flags(cls, path: str) -> frozenset:
        """Return the role/semantic flags associated with ``path`` in the module tree, if any."""

        return cls.get_module_tree_metadata(path).flags

    @classmethod
    def get_module_tree_expert_group(cls, path: str) -> Optional[str]:
        """Return the explicit module-tree expert group identity for ``path``."""

        return cls.get_module_tree_metadata(path).expert_group

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
        moe_modules = set()
        for tree in cls._iter_module_tree_variants():
            moe_modules.update(cls._collect_moe_modules_from_tree(tree))
        return moe_modules

    @classmethod
    def is_moe_module(cls, module_path: str) -> bool:
        """
        Check if a given module path is an MoE module based on :moe flags.

        Args:
            module_path: Full module path like "model.layers.0.mlp.experts.5.gate_proj"

        Returns:
            True if any parent in the path is marked with :moe flag
        """
        path_parts = tuple(module_path.split("."))
        for declared_path in cls.get_moe_modules():
            declared_parts = tuple(declared_path.split("."))
            declared_size = len(declared_parts)
            # Full runtime names include a model/layer prefix that is outside
            # module_tree. Match complete path components only; semantic names
            # such as "expert" never participate in classification.
            for offset in range(len(path_parts) - declared_size + 1):
                if path_parts[offset:offset + declared_size] == declared_parts:
                    return True
        return False

    @classmethod
    def get_moe_module_name(cls) -> Optional[List[str]]:
        """
        Get the name of the MoE module from module_tree.

        Each layer can have only ONE MoE module marked with :moe flag.
        For example:
        - GLM-4: "mlp:moe" -> returns "mlp"
        - MiniMax-M2: "block_sparse_moe:moe" -> returns "block_sparse_moe"

        Returns:
            A list of MoE module names (without flags), or an empty list if no MoE module is defined.
        """
        if cls.module_tree is None:
            return []

        found_names = []
        for tree in cls._iter_module_tree_variants():
            layer_structure = None
            found_hash = False
            for item in tree:
                if item == "#":
                    found_hash = True
                    continue
                if found_hash and isinstance(item, dict):
                    layer_structure = item
                    break

            if layer_structure is None:
                continue

            for key in layer_structure.keys():
                if cls.has_moe_flag(key):
                    module_name, _ = cls._parse_module_flags(key)
                    if module_name not in found_names:
                        found_names.append(module_name)

        return found_names

    @classmethod
    def build_moe_modules_if_need(cls, model_config, layer_modules, is_awq_quantize: bool = False):
        # MoE models
        if model_config is not None and cls.dynamic_expert_index is not None:
            num_experts = cls.get_num_experts(model_config)

            def _copy_expanded_metadata(template_path: str, expanded_path: str, index: int) -> None:
                """Copy template metadata while resolving its explicit expert-group placeholder."""

                metadata = cls.get_module_tree_metadata(template_path)
                expert_group = metadata.expert_group
                if expert_group is not None:
                    expert_group = expert_group.replace(EXPERT_INDEX_PLACEHOLDER, str(index))
                cls._set_module_tree_flags(
                    expanded_path,
                    metadata.flags,
                    expert_group=expert_group,
                )

            def _is_expert_gate_up_block(names: List[str]) -> bool:
                """Return whether explicit tags describe only expert gate/up projections."""

                metadata = [cls.get_module_tree_metadata(name) for name in names]
                if not metadata or not all(module_tree_flags_are_expert(item.flags) for item in metadata):
                    return False
                roles = set().union(*(item.flags & MODULE_TREE_PROJECTION_FLAGS for item in metadata))
                return roles == {MODULE_TREE_FLAG_GATE, MODULE_TREE_FLAG_UP}

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
                                expanded = n.replace(EXPERT_INDEX_PLACEHOLDER, str(index))
                                moe_simple[-1].append(expanded)
                                _copy_expanded_metadata(n, expanded, index)
                    # Currently, only need to add `capture_only_modules` to `['mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']`
                    # or ['mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    # or ['mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    add_capture_only_module = _is_expert_gate_up_block(names)
                    if add_capture_only_module and capture_only_modules:
                        # Extend all elements in capture_only_modules
                        moe_simple[-1].extend(capture_only_modules)
                else:
                    # result like: ['mlp.experts.0.gate_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.up_proj', ...]
                    for n in names:
                        for index in range(num_experts):
                            expanded = n.replace(EXPERT_INDEX_PLACEHOLDER, str(index))
                            moe_simple[-1].append(expanded)
                            _copy_expanded_metadata(n, expanded, index)

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

    @classmethod
    def should_quantize_layer(
        cls,
        layer: nn.Module,
        layer_name: str,
        layer_index: int | None,
        quantize_config: BaseQuantizeConfig,
    ) -> bool:
        """Return whether a decoder layer should be quantized during calibration."""
        return True

    @classmethod
    def should_quantize_module(
        cls,
        model: nn.Module,
        module_name: str,
        module: nn.Module,
        quantize_config: BaseQuantizeConfig,
    ) -> bool:
        """
        Return whether a concrete module should be replaced with a quantized module.

        `module_tree` describes reusable suffixes inside decoder layers, but some
        model families use different layer subclasses under the same layer list.
        Model definitions can override this hook when a suffix is quantizable only
        for specific layer subclasses.
        """
        return True

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
        yaqa_config = getattr(getattr(self, "quantize_config", None), "yaqa", None)
        return prepare_calibration_dataset(
            self,
            calibration_dataset=calibration_dataset,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            calibration_dataset_sort=calibration_dataset_sort,
            batch_size=batch_size,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            chat_template_config=getattr(yaqa_config, "chat_template", None),
            logger=log,
        )

    def quantize(
        self,
        calibration: Optional[Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]] = None,
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = "desc",
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        adapter: Adapter = None,
        adapter_calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]] = None,
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]] = None,
        embed_quant_mode: Optional[QuantizeEmbed] = None,
        validation_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        yaqa_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        module_replay_search_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        module_replay_confirmation_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        layer_scope: Optional[Union[int, slice, str, List[Union[int, str]]]] = None,
        freeze_others: bool = True,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Quantize the model, optionally limited to a subset of layers.

        `layer_scope` may be a layer index, slice, regex, or list of indices/regexes.
        Non-matching layers are frozen (excluded) so one layer can be quantized,
        saved, and later resumed via `requant` with a different config or calibration.

        For QVQ YAQA rounding, `yaqa_calibration` optionally supplies an independent
        dataset for the full-model Fisher/Sketch-B pass. If omitted, YAQA reuses
        `calibration`; the ordinary activation-Hessian and replay stream is never
        replaced by this YAQA-only dataset.

        QVQ module-granular replay requires explicit, disjoint
        `module_replay_search_calibration` and `module_replay_confirmation_calibration`
        streams. They are never inferred from ordinary or YAQA calibration.
        """

        # Layer-scope dynamic overrides are temporary. Snapshot the original map so
        # we can restore it before returning; this keeps saved checkpoints free of
        # transient per-call scope exclusions.
        if layer_scope is not None:
            original_dynamic = copy.deepcopy(self.quantize_config.dynamic)
        else:
            original_dynamic = None

        try:
            self._apply_layer_scope(layer_scope=layer_scope, freeze_others=freeze_others)
            result = self._quantize_impl(
                calibration=calibration,
                calibration_concat_size=calibration_concat_size,
                calibration_sort=calibration_sort,
                batch_size=batch_size,
                tokenizer=tokenizer,
                backend=backend,
                adapter=adapter,
                adapter_calibration_dataset=adapter_calibration_dataset,
                calibration_data_min_length=calibration_data_min_length,
                calibration_concat_separator=calibration_concat_separator,
                embed_quant_config=embed_quant_config,
                embed_quant_mode=embed_quant_mode,
                validation_calibration=validation_calibration,
                yaqa_calibration=yaqa_calibration,
                module_replay_search_calibration=module_replay_search_calibration,
                module_replay_confirmation_calibration=module_replay_confirmation_calibration,
                layer_scope=layer_scope,
                freeze_others=freeze_others,
            )
            return result
        finally:
            if layer_scope is not None:
                # Rebuild a clean dynamic map in precedence order:
                # 1. original negative exclusions first (they short-circuit everything),
                # 2. positive overrides for already-quantized layers whose effective
                #    config differs from the base (so they win over broad user positives),
                # 3. original positive user overrides last.
                # Any temporary negative scope exclusions we added are already absent
                # from `original_dynamic` and do not need to be stripped.
                final_dynamic: Dict[str, Any] = {}
                if original_dynamic is not None:
                    for pattern, override in original_dynamic.items():
                        if pattern.startswith("-:") or override is False:
                            final_dynamic[pattern] = override
                final_dynamic.update(self._capture_quantized_layer_dynamic())
                if original_dynamic is not None:
                    for pattern, override in original_dynamic.items():
                        if not (pattern.startswith("-:") or override is False):
                            final_dynamic[pattern] = override
                self.quantize_config._invalidate_dynamic_cache()
                self.quantize_config.dynamic = final_dynamic

    def _quantize_impl(
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
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]] = None,
        embed_quant_mode: Optional[QuantizeEmbed] = None,
        validation_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        yaqa_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        module_replay_search_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        module_replay_confirmation_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
        layer_scope: Optional[Union[int, slice, str, List[Union[int, str]]]] = None,
        freeze_others: bool = True,
    ) -> Dict[str, List[Dict[str, str]]]:
        embed_quant_config = self._normalize_embed_quant_config(
            embed_quant_config=embed_quant_config,
            embed_quant_mode=embed_quant_mode,
        )

        if self.quantize_config is None or not isinstance(self.quantize_config, BaseQuantizeConfig):
            raise AttributeError("`quantize_config` must be not None")

        if embed_quant_config is None and self.quantized and layer_scope is None:
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

        if validation_calibration is not None and self.quantize_config.method != METHOD.PARO:
            raise ValueError("`validation_calibration` is only supported for ParoQuant quantization.")

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
            quant_device = self.quantize_config.device
            if isinstance(quant_device, DEVICE):
                quant_device_type = quant_device.type
            elif isinstance(quant_device, torch.device):
                quant_device_type = quant_device.type
            else:
                quant_device_type = str(quant_device).split(":")[0].lower()

            if export_quant_method == METHOD.AWQ:
                if quant_device_type == "npu" and format_code != FORMAT.GEMM:
                    raise ValueError(
                        "NPU AWQ quantization requires FORMAT.GEMM so the AwqTorchLinear runtime can run on NPU; "
                        f"actual format is `{format_code}`."
                    )
                if format_code == FORMAT.GEMM:
                    # Weight-only RTN->AWQ export should stay on the portable torch kernel.
                    preferred_backend = (
                        BACKEND.AWQ_TORCH
                        if self.quantize_config.uses_weight_only_lifecycle() or quant_device_type == "npu"
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
                preferred_backend = BACKEND.QQQ_TORCH if quant_device_type == "npu" else BACKEND.QQQ
            elif self.quantize_config.method == METHOD.PARO:
                preferred_backend = BACKEND.PAROQUANT_CUDA
            elif self.quantize_config.method == METHOD.EXL3:
                preferred_backend = BACKEND.EXL3_EXLLAMA_V3
            elif self.quantize_config.method == METHOD.QVQ:
                preferred_backend = BACKEND.QVQ
            elif self.quantize_config.method == METHOD.GGUF:
                preferred_backend = BACKEND.AUTO
            elif self.quantize_config.method == METHOD.FP8:
                preferred_backend = BACKEND.FP8_TORCH
            elif self.quantize_config.method == METHOD.BITSANDBYTES:
                preferred_backend = BACKEND.BITSANDBYTES
            else:
                preferred_backend = BACKEND.GPTQ_TORCH

        if self.quantize_config.method == METHOD.QVQ:
            if preferred_backend not in (BACKEND.AUTO, BACKEND.QVQ):
                raise ValueError("QVQ quantization only supports BACKEND.AUTO or BACKEND.QVQ.")
        elif self.quantize_config.method == METHOD.EXL3:
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
        elif self.quantize_config.method == METHOD.QVQ:
            from ..nn_modules.qlinear.qvq import QVQLinear

            self.qlinear_kernel = QVQLinear
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

        if self.quantize_config.rotation:
            backend = normalize_backend(backend, quant_method=self.quantize_config.method)
            if backend == BACKEND.AUTO:
                backend = BACKEND.GPTQ_TORCH
            if backend not in (BACKEND.GPTQ_TORCH, BACKEND.GPTQ_TRITON):
                raise NotImplementedError(
                    f"`rotation` is only supported with `gptq_torch` or `gptq_triton` backend, got `{backend}`."
                )

        if self.quantize_config.uses_weight_only_lifecycle():
            result = self._quantize_weight_only(
                calibration=calibration,
                calibration_concat_size=calibration_concat_size,
                calibration_sort=calibration_sort,
                batch_size=batch_size,
                backend=backend,
                calibration_concat_separator=calibration_concat_separator,
                embed_quant_config=embed_quant_config,
            )
        else:
            if calibration is None:
                raise ValueError(
                    "Calibration dataset is required unless a weight-only quantize config is configured."
                )
            result = self._quantize_with_calibration(
                calibration=calibration,
                validation_calibration=validation_calibration,
                yaqa_calibration=yaqa_calibration,
                module_replay_search_calibration=module_replay_search_calibration,
                module_replay_confirmation_calibration=module_replay_confirmation_calibration,
                calibration_concat_size=calibration_concat_size,
                calibration_sort=calibration_sort,
                batch_size=batch_size,
                backend=backend,
                adapter_calibration_dataset=adapter_calibration_dataset,
                calibration_concat_separator=calibration_concat_separator,
                embed_quant_config=embed_quant_config,
            )

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.flush()

        disk_telemetry.log_summary(log, label="quantization complete", all_time=True)

        _setup_rotation_online_had(self.model, self.quantize_config.rotation)
        return result

    @staticmethod
    def _normalize_embed_quant_config(
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]],
        embed_quant_mode: Optional[QuantizeEmbed],
    ) -> Optional[QuantizeEmbedConfig]:
        """Normalize the current embedding config and the legacy mode argument."""
        if embed_quant_config is not None and embed_quant_mode is not None:
            raise ValueError("Pass only one of `embed_quant_config` or `embed_quant_mode`.")

        if isinstance(embed_quant_config, QuantizeEmbed):
            return QuantizeEmbedConfig(embed_quant_mode=embed_quant_config)

        if embed_quant_config is not None:
            if not isinstance(embed_quant_config, QuantizeEmbedConfig):
                raise TypeError(
                    "`embed_quant_config` must be a `QuantizeEmbedConfig` or `QuantizeEmbed` instance."
                )
            return embed_quant_config

        if embed_quant_mode is not None:
            if not isinstance(embed_quant_mode, QuantizeEmbed):
                raise TypeError("`embed_quant_mode` must be a `QuantizeEmbed` instance.")
            return QuantizeEmbedConfig(embed_quant_mode=embed_quant_mode)

        return None

    def requantize(
        self,
        calibration: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        embed_quant_config: Optional[Union[QuantizeEmbedConfig, QuantizeEmbed]] = None,
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = "desc",
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        adapter: Adapter = None,
        adapter_calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]] = None,
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
        embed_quant_mode: Optional[QuantizeEmbed] = None,
        validation_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        if not self.quantized:
            raise EnvironmentError("requantize() must be called on a model that has already been quantized.")

        embed_quant_config = self._normalize_embed_quant_config(
            embed_quant_config=embed_quant_config,
            embed_quant_mode=embed_quant_mode,
        )
        if embed_quant_config is None:
            raise ValueError("`requantize()` requires `embed_quant_config` or `embed_quant_mode`.")

        result = self.quantize(
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            batch_size=batch_size,
            tokenizer=tokenizer,
            backend=backend,
            adapter=adapter,
            adapter_calibration_dataset=adapter_calibration_dataset,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            embed_quant_config=embed_quant_config,
            validation_calibration=validation_calibration,
        )

        # Requantize only touches the embedding/lm_head modules. The live model tree
        # of a loaded quantized checkpoint is an inference structure (kernel-repacked
        # qweights, defused/fused MoE expert modules) that is not round-trip safe to
        # serialize, so save() must stream the source checkpoint shards and rewrite
        # only the shards containing the requantized embedding modules.
        prefixes: set = set(getattr(self, "_embedding_replacement_prefixes", set()))
        mode = embed_quant_config.embed_quant_mode
        if mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH):
            name = self.get_input_embeddings_name()
            if name:
                prefixes.add(name)
        if mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH):
            name = self.get_output_embeddings_name() or self.lm_head
            if name:
                prefixes.add(name)
        self._embedding_replacement_prefixes = prefixes

        if getattr(self, "load_quantized_model", False) and prefixes:
            if self.turtle_model is None:
                # Kept off `turtle_model` so shell materialization paths that expect a
                # LazyTurtle never see this shard-map-only stub; the embedding-only
                # save path resolves it as a fallback checkpoint source.
                self._embedding_replacement_source = _QuantizedCheckpointSource(str(self.model_local_path))
            self._model_free_weight_only_embeddings_only = True

        return result


    def _resolve_layer_scope_indices(
        self,
        layer_scope: Optional[Union[int, slice, str, List[Union[int, str]]]],
        layer_names: List[str],
        layer_count: int,
    ) -> Set[int]:
        """Normalize `layer_scope` into a set of concrete layer indices."""

        if layer_scope is None:
            return set(range(layer_count))

        if isinstance(layer_scope, int):
            if not (0 <= layer_scope < layer_count):
                raise IndexError(f"layer_scope index {layer_scope} out of range (0..{layer_count - 1})")
            return {layer_scope}

        if isinstance(layer_scope, slice):
            return set(range(layer_count)[layer_scope])

        if isinstance(layer_scope, str):
            compiled = re.compile(layer_scope)
            matched = {i for i, name in enumerate(layer_names) if compiled.match(name) or compiled.search(name)}
            if not matched:
                raise ValueError(f"layer_scope regex `{layer_scope}` did not match any of {layer_names}")
            return matched

        if isinstance(layer_scope, (list, tuple)):
            matched: Set[int] = set()
            for item in layer_scope:
                if isinstance(item, int):
                    if not (0 <= item < layer_count):
                        raise IndexError(f"layer_scope index {item} out of range (0..{layer_count - 1})")
                    matched.add(item)
                elif isinstance(item, str):
                    compiled = re.compile(item)
                    matched.update({i for i, name in enumerate(layer_names) if compiled.match(name) or compiled.search(name)})
                else:
                    raise TypeError(f"layer_scope items must be int or str, got {type(item)}")
            if not matched:
                raise ValueError(f"layer_scope list did not match any of {layer_names}")
            return matched

        raise TypeError(f"Unsupported layer_scope type: {type(layer_scope)}")

    def _get_layer_names(self) -> List[str]:
        """Return flat transformer layer names for the current model."""

        _, layer_names = get_layers_with_prefixes(self.model, self.extract_layers_node())
        return layer_names

    def _apply_layer_scope(
        self,
        layer_scope: Optional[Union[int, slice, str, List[Union[int, str]]]] = None,
        freeze_others: bool = True,
    ) -> Set[str]:
        """Add dynamic exclusions so only `layer_scope` layers are processed.

        When `freeze_others=True` (the default), we emit negative dynamic patterns
        for every non-scope transformer layer.  Scope layers keep the existing base
        config and any user-provided dynamic overrides.  When `freeze_others=False`,
        no negative exclusions are injected and the `layer_scope` is ignored by
        this helper; callers should use `freeze_others=False` only when they intend
        to quantize all layers.  Already-quantized modules are always skipped in
        `ModuleLooper.create_named_modules`.

        Returns the set of negative patterns that were added so callers can
        strip them after quantization finishes.
        """

        if layer_scope is None or not freeze_others:
            return set()

        layer_names = self._get_layer_names()
        layer_count = len(layer_names)
        if layer_count == 0:
            return set()

        scope_indices = self._resolve_layer_scope_indices(layer_scope, layer_names, layer_count)

        existing_dynamic = self.quantize_config.dynamic or {}
        new_dynamic: Dict[str, Any] = {}
        added: Set[str] = set()

        # Negative patterns first so they win over any broad positive user patterns.
        for idx, layer_name in enumerate(layer_names):
            if idx not in scope_indices:
                pattern = f"-:.*{re.escape(layer_name)}\\..*"
                new_dynamic[pattern] = False
                added.add(pattern)

        # Preserve existing dynamic overrides (positives or per-layer negatives).
        for pattern, override in existing_dynamic.items():
            if pattern not in new_dynamic:
                new_dynamic[pattern] = override

        self.quantize_config._invalidate_dynamic_cache()
        self.quantize_config.dynamic = new_dynamic
        return added

    def requant(
        self,
        calibration: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        quantize_config: Optional[BaseQuantizeConfig] = None,
        layer_scope: Optional[Union[int, slice, str, List[Union[int, str]]]] = None,
        freeze_others: bool = True,
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = "desc",
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        adapter: Adapter = None,
        adapter_calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]] = None,
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
        validation_calibration: Optional[
            Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]]
        ] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Continue quantization on a partially (or fully) quantized model.

        `quantize_config` replaces the current config for this step, letting each
        incremental layer use a different calibration dataset, bits, group_size, etc.
        Only layers matching `layer_scope` are processed; non-matched dense and
        already-quantized modules are left untouched.
        """

        if quantize_config is not None:
            if not isinstance(quantize_config, BaseQuantizeConfig):
                raise TypeError("`quantize_config` must be a `BaseQuantizeConfig` instance.")
            if quantize_config.method != self.quantize_config.method:
                raise ValueError(
                    f"`requant` cannot switch quantization method; "
                    f"current={self.quantize_config.method.value}, requested={quantize_config.method.value}. "
                    f"Mixed-method quantization is not yet supported."
                )
            # Merge the new algorithmic config into the existing config so runtime/
            # loading state (meta, adapter, existing per-layer dynamic overrides, and
            # the active offload temp directory) is not discarded.
            old_config = self.quantize_config
            new_config = quantize_config

            # Carry over device/offload state explicitly. Offload is special because
            # a new QuantizeConfig default (True) auto-creates a temp dir; we only
            # want to inherit the previous temp dir when the new config did not
            # explicitly provide its own path.
            if new_config.device is None:
                new_config.device = old_config.device
            new_offload_path = getattr(new_config, "offload_to_disk_path", None)
            new_temp_dir = getattr(new_config, "_offload_temp_dir", None)
            if new_config.offload_to_disk and new_offload_path is not None and new_temp_dir is None:
                # New config explicitly supplied an offload path; keep it.
                offload_to_disk = new_config.offload_to_disk
                offload_to_disk_path = new_offload_path
                offload_temp_dir = None
            else:
                # New config either disabled offload or used the default temp dir.
                # Inherit previous offload state so an in-use temp dir is not abandoned.
                offload_to_disk = new_config.offload_to_disk
                offload_to_disk_path = getattr(old_config, "offload_to_disk_path", None)
                offload_temp_dir = getattr(old_config, "_offload_temp_dir", None)
                # If offload is still enabled but no inherited path exists, fall back
                # to whatever the new config created.
                if offload_to_disk and offload_to_disk_path is None:
                    offload_to_disk_path = new_offload_path
                    offload_temp_dir = new_temp_dir

            # Copy all dataclass fields from the new config, then restore the
            # persisted/runtime fields that should be merged rather than replaced.
            old_dynamic = old_config.dynamic or {}
            new_dynamic = new_config.dynamic or {}
            merged_dynamic = dict(old_dynamic)
            if new_dynamic:
                merged_dynamic.update(new_dynamic)

            for field in dataclasses.fields(new_config):
                if field.name in (
                    "device",
                    "offload_to_disk",
                    "offload_to_disk_path",
                    "_offload_temp_dir",
                    "meta",
                    "adapter",
                    "dynamic",
                ):
                    continue
                setattr(old_config, field.name, getattr(new_config, field.name))

            old_config.device = new_config.device
            old_config.offload_to_disk = offload_to_disk
            old_config.offload_to_disk_path = offload_to_disk_path
            old_config._offload_temp_dir = offload_temp_dir
            old_config.meta = old_config.meta if new_config.meta is None else new_config.meta
            old_config.adapter = old_config.adapter if new_config.adapter is None else new_config.adapter
            old_config.dynamic = merged_dynamic

        if layer_scope is None:
            # Default to all layers that are not already quantized.
            layer_names = self._get_layer_names()
            quantized_indices = self._quantized_layer_indices()
            if len(quantized_indices) >= len(layer_names):
                raise ValueError("requant() called but all layers are already quantized.")
            layer_scope = [i for i in range(len(layer_names)) if i not in quantized_indices]

        return self.quantize(
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            batch_size=batch_size,
            tokenizer=tokenizer,
            backend=backend,
            adapter=adapter,
            adapter_calibration_dataset=adapter_calibration_dataset,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            validation_calibration=validation_calibration,
            layer_scope=layer_scope,
            freeze_others=freeze_others,
        )

    def _capture_quantized_layer_dynamic(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        r"""Build positive dynamic overrides for layers whose effective config differs from base.

        After a per-layer quantization step, the model may contain BaseQuantLinear
        modules quantized with different bits/group_size/sym/desc_act values. We
        read those values and emit `+:.*<layer>\..*` patterns so that subsequent
        save/load/requant operations reproduce the correct per-layer contract.
        """

        layer_names = self._get_layer_names()
        if not layer_names:
            return {}

        prefix_to_index = {name: idx for idx, name in enumerate(layer_names)}
        base = self.quantize_config
        positives: Dict[str, Dict[str, Any]] = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, BaseQuantLinear):
                continue

            # Map module to transformer layer prefix.
            matched_layer = None
            for layer_name, idx in prefix_to_index.items():
                if name.startswith(f"{layer_name}."):
                    matched_layer = layer_name
                    break
            if matched_layer is None:
                continue

            # Build effective config from the actual quantized module.
            override: Dict[str, Any] = {}
            # Preserve the installed module's public bit/rate contract exactly.
            # Most quantizers expose integer bit widths, but QVQ also supports
            # half-step rates (for example W2.5).  Coercing those rates to int
            # corrupts the dynamic metadata used to allocate checkpoint
            # buffers on reload.
            bits = getattr(module, "bits", base.bits)
            group_size = int(getattr(module, "requested_group_size", getattr(module, "group_size", base.group_size)))
            desc_act = bool(getattr(module, "desc_act", base.desc_act))
            sym = bool(getattr(module, "sym", base.sym))

            if bits != base.bits:
                override["bits"] = bits
            if group_size != base.group_size:
                override["group_size"] = group_size
            if desc_act != base.desc_act:
                override["desc_act"] = desc_act
            if sym != base.sym:
                override["sym"] = sym

            if override:
                # Emit one positive override per quantized module rather than one
                # per layer. A layer-wide pattern would shadow module-specific user
                # overrides and collapse modules within the same layer that were
                # quantized with different settings.
                pattern = f"+:.*{re.escape(name)}"
                positives[pattern] = override

        return positives

    def _quantized_layer_indices(self) -> Set[int]:
        """Return layer indices that already contain at least one BaseQuantLinear."""

        layer_names = self._get_layer_names()
        prefix_to_index = {name: idx for idx, name in enumerate(layer_names)}
        quantized: Set[int] = set()
        for name, module in self.model.named_modules():
            if isinstance(module, BaseQuantLinear):
                for layer_name, idx in prefix_to_index.items():
                    if name.startswith(f"{layer_name}."):
                        quantized.add(idx)
                        break
        return quantized

    def _quantize_with_calibration(
        self,
        *,
        calibration,
        validation_calibration,
        yaqa_calibration,
        module_replay_search_calibration,
        module_replay_confirmation_calibration,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        backend: Optional[BACKEND],
        adapter_calibration_dataset,
        calibration_concat_separator: Optional[str],
        embed_quant_config: Optional[QuantizeEmbedConfig] = None,
    ):
        from ..adapter.adapter import Lora
        from ..looper.analysis_processor import AnalysisProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper
        from ..looper.module_preprocessor import ModulePreProcessor
        from ..quantization.config import AnalysisConfig

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

        if yaqa_calibration is not None:
            if self.quantize_config.method != METHOD.QVQ:
                raise ValueError("`yaqa_calibration` is only supported for QVQ quantization.")
            if self.quantize_config.rounding != "yaqa":
                raise ValueError("`yaqa_calibration` requires QVQ `rounding='yaqa'`.")

        replay_config = getattr(self.quantize_config, "module_granular_replay", None)
        replay_streams = (module_replay_search_calibration, module_replay_confirmation_calibration)
        if replay_config is None and any(stream is not None for stream in replay_streams):
            raise ValueError(
                "Module replay calibration streams require QVQ `module_granular_replay` to be enabled."
            )
        if replay_config is not None and any(stream is None for stream in replay_streams):
            raise ValueError(
                "QVQ module-granular replay requires explicit search and confirmation calibration streams."
            )

        configured_preprocessors = getattr(self.quantize_config, "preprocessors", None) or []
        analysis_enabled = any(isinstance(item, AnalysisConfig) for item in configured_preprocessors)
        planning_preprocessors = [item for item in configured_preprocessors if not isinstance(item, AnalysisConfig)]

        preprocessors = []
        if planning_preprocessors:
            preprocessors.append(ModulePreProcessor(**args))
        if analysis_enabled:
            preprocessors.append(AnalysisProcessor(**args))

        if self.quantize_config.method == METHOD.QVQ:
            from ..looper.qvq_processor import QVQProcessor

            if needs_lora:
                raise NotImplementedError("QVQ quantization does not support adapter/EoRA generation.")
            qvq_args = {
                "tokenizer": self.tokenizer,
                "qcfg": self.quantize_config,
                "calibration": calibration,
                "prepare_dataset_func": self.prepare_dataset,
                "calibration_concat_size": calibration_concat_size,
                "calibration_sort": calibration_sort,
                "calibration_concat_separator": calibration_concat_separator,
                "batch_size": batch_size,
            }
            if yaqa_calibration is not None:
                qvq_args["yaqa_calibration"] = self.prepare_dataset(
                    calibration_dataset=yaqa_calibration,
                    calibration_dataset_concat_size=calibration_concat_size,
                    # YAQA batching is independent of ordinary calibration.
                    # Length bucketing removes padded model/Gram work without
                    # changing which independent Fisher rows are consumed.
                    calibration_dataset_sort=(
                        None
                        if self.quantize_config.yaqa.sequence_sort == "none"
                        else self.quantize_config.yaqa.sequence_sort
                    ),
                    # Sketch-B remains per-sequence: collating independent rows
                    # only amortizes model launches, Gram transfers, and MPS
                    # synchronization. Ordinary calibration keeps its own batch.
                    batch_size=self.quantize_config.yaqa.batch_size,
                    calibration_data_min_length=10,
                    calibration_concat_separator=calibration_concat_separator,
                )
            if replay_config is not None:
                qvq_args["module_replay_search_calibration"] = self.prepare_dataset(
                    calibration_dataset=module_replay_search_calibration,
                    calibration_dataset_concat_size=None,
                    calibration_dataset_sort=None,
                    batch_size=1,
                    calibration_data_min_length=10,
                    calibration_concat_separator=None,
                )
                qvq_args["module_replay_confirmation_calibration"] = self.prepare_dataset(
                    calibration_dataset=module_replay_confirmation_calibration,
                    calibration_dataset_concat_size=None,
                    calibration_dataset_sort=None,
                    batch_size=1,
                    calibration_data_min_length=10,
                    calibration_concat_separator=None,
                )
            qvq_processor = QVQProcessor(**qvq_args)
            qvq_processor.prepare_module_granular_replay(self)
            qvq_processor.prepare_yaqa(self)
            quantize_processor = preprocessors + [qvq_processor]
        elif self.quantize_config.method == METHOD.EXL3:
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
            paro_args["validation_calibration"] = validation_calibration

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

        module_looper = ModuleLooper(self, processors=processors, embed_quant_config=embed_quant_config)

        gc_context = (
            DEVICE_THREAD_POOL.no_auto_gc()
            if self.quantize_config.gc_mode == GcMode.ON_STAGE_END
            else nullcontext()
        )

        with gc_context:
            quant_log = module_looper.loop(
                backend=backend,
                fallback=self.quantize_config.fallback,
            )
            return quant_log

    def _quantize_weight_only(
        self,
        *,
        calibration,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        backend: Optional[BACKEND],
        calibration_concat_separator: Optional[str],
        embed_quant_config: Optional[QuantizeEmbedConfig] = None,
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
        module_looper = WeightOnlyLooper(model=self, processor=processor, embed_quant_config=embed_quant_config)

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
        from ..looper.analysis_processor import AnalysisProcessor
        from ..looper.dequantize_processor import DequantizeProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper
        from ..looper.module_preprocessor import ModulePreProcessor
        from ..quantization.config import AnalysisConfig

        self.quantize_config.adapter = adapter

        assert isinstance(self.quantize_config.adapter, Lora)

        # init processor with EoRA processor
        configured_preprocessors = getattr(self.quantize_config, "preprocessors", None) or []
        analysis_enabled = any(isinstance(item, AnalysisConfig) for item in configured_preprocessors)
        planning_preprocessors = [item for item in configured_preprocessors if not isinstance(item, AnalysisConfig)]

        processors = []
        if planning_preprocessors:
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
        if analysis_enabled:
            processors.append(
                AnalysisProcessor(
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
        input_ids = example.get("input_ids")
        if torch.is_tensor(input_ids):
            input_ids = self._sanitize_input_ids_for_embeddings(input_ids)
            if input_ids is not example.get("input_ids"):
                example = dict(example)
                example["input_ids"] = input_ids

        if self.INPUT_EMBEDDING_EXTRA_ARGS:
            return self.model.generate(
                **example,
                **self.INPUT_EMBEDDING_EXTRA_ARGS,
            )

        return self.model(**example, use_cache=use_cache)

    def _sanitize_input_ids_for_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            return input_ids

        if input_ids.numel() == 0:
            return input_ids

        embedder = getattr(self.model, "get_input_embeddings", None)
        if not callable(embedder):
            return input_ids

        # Some multimodal wrappers expose get_input_embeddings but intentionally
        # raise NotImplementedError at runtime; skip sanitization in that case.
        try:
            embedding = embedder()
        except NotImplementedError:
            return input_ids
        except Exception:
            return input_ids
        if embedding is None or not hasattr(embedding, "num_embeddings"):
            return input_ids

        vocab_size = int(getattr(embedding, "num_embeddings", 0) or 0)
        if vocab_size <= 0:
            return input_ids

        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if not bool(invalid_mask.any()):
            return input_ids

        tokenizer = getattr(self, "tokenizer", None)
        replacement_token_id = None
        for attr_name in ("unk_token_id", "pad_token_id", "eos_token_id", "bos_token_id"):
            candidate = getattr(tokenizer, attr_name, None) if tokenizer is not None else None
            if isinstance(candidate, int) and 0 <= candidate < vocab_size:
                replacement_token_id = candidate
                break
        if replacement_token_id is None:
            replacement_token_id = 0

        sanitized = input_ids.clone()
        sanitized[invalid_mask] = replacement_token_id

        invalid_count = int(invalid_mask.sum().item())
        log.warning(
            "Input capture: replaced %s out-of-range token ids with %s (valid range: [0, %s)).",
            invalid_count,
            replacement_token_id,
            vocab_size,
        )
        return sanitized

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

    def fuse(
        self,
        qkv: bool = True,
        gate_up: bool = True,
        free_original_weights: bool = True,
        gate_up_activation: bool = True,
    ) -> Dict[str, int]:
        """Fuse compatible quantized projection groups for inference.

        This is an opt-in step after `GPTQModel.load(...)`. It scans the
        underlying `transformers` model for `q_proj`/`k_proj`/`v_proj` and
        `gate_proj`/`up_proj` groups that share the same GPTQ backend,
        `bits`, `group_size`, `g_idx`, and device, and replaces their forward
        methods with a single concatenated GEMM. The fused state is kept only in
        memory and is not persisted to disk on `save()`.

        Args:
            qkv: Whether to fuse attention QKV projections.
            gate_up: Whether to fuse MLP gate/up projections.
            free_original_weights: Whether to delete the per-member packed
                weight buffers after the fused kernel owns a concatenated copy.
                This roughly halves the fused group's memory footprint but means
                `save()` cannot serialize the fused model; call `fuse()` again
                with `free_original_weights=False` if you need to save.
            gate_up_activation: Whether to also fuse the MLP activation
                (SiLU, GeLU, ReLU, tanh, etc.) and down projection into a single
                gate/up/down pass when the MLP structure can be detected. This
                further reduces Python launch overhead and intermediate memory.

        Returns:
            A mapping of fusion type to number of groups installed.
        """
        if not (self.quantized or self.load_quantized_model):
            log.warning(
                "BaseQModel.fuse() is intended for quantized inference models; "
                "skipping because neither `quantized` nor `load_quantized_model` is set."
            )
            return {"qkv": 0, "gate_up": 0}

        if free_original_weights:
            log.warn.once(
                "BaseQModel.fuse(free_original_weights=True) removes per-member "
                "packed weight buffers. `model.save()` and any code that reads "
                "member `.qweight`/`.scales` buffers will fail after this call. "
                "Pass free_original_weights=False if you need to save or inspect."
            )

        from ..nn_modules.fused_quant_linear import (
            get_module_tree_fusion_candidates,
            install_fused_gate_up,
            install_fused_qkv,
        )

        qkv_candidates = None
        gateup_candidates = None
        if getattr(self, "module_tree", None) is not None:
            try:
                qkv_candidates, gateup_candidates = get_module_tree_fusion_candidates(self.module_tree)
                if not qkv_candidates:
                    qkv_candidates = None
                if not gateup_candidates:
                    gateup_candidates = None
            except Exception:
                pass

        counts: Dict[str, int] = {}
        if qkv:
            counts["qkv"] = install_fused_qkv(
                self.model,
                candidates=qkv_candidates,
                free_original_weights=free_original_weights,
            )
        if gate_up:
            counts["gate_up"] = install_fused_gate_up(
                self.model,
                candidates=gateup_candidates,
                free_original_weights=free_original_weights,
                fuse_activation=gate_up_activation,
            )

        log.info(f"Fused {sum(counts.values())} quantized projection group(s): {counts}")
        return counts

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            split_by: Optional[str] = None,
            shard_strategy: Optional[Union[ShardStrategy, str]] = None,
            moe_modules_per_shard: int = 128,
            **kwargs,
    ):
        timer = getattr(self, "quant_region_timer", None)
        start_time = time.perf_counter() if timer else None

        try:
            if self.quantized:
                # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
                #untie_weights(self.model)

                if getattr(self, "_model_free_weight_only_embeddings_only", False):
                    self.save_quantized_embeddings(
                        save_dir=save_dir,
                        safetensors_metadata=safetensors_metadata,
                        max_shard_size=max_shard_size,
                        meta_quantizer=meta_quantizer,
                    )
                else:
                    self.save_quantized(
                        save_dir=save_dir,
                        safetensors_metadata=safetensors_metadata,
                        max_shard_size=max_shard_size,
                        meta_quantizer=meta_quantizer,
                        eora_path=eora_path,
                        split_by=split_by,
                        shard_strategy=shard_strategy,
                        moe_modules_per_shard=moe_modules_per_shard,
                    )

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

        if not TORCH_HAS_COMPILE:
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

    def update_layer_replay_kwargs_from_output(
        self,
        layer: nn.Module,
        layer_output: Any,
        layer_input_kwargs: Dict[str, Any],
        target_device: torch.device,
    ) -> Dict[str, Any]:
        """Allow model definitions to persist cross-layer replay state from one layer output."""

        return layer_input_kwargs

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.pre_lm_head_norm_module:
            norm, _ = get_module_by_name_prefix(self.model, [self.pre_lm_head_norm_module])
            norm = self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def pre_quantize(
        self,
        module: nn.Module,
        *,
        skip_module_names: Optional[set[str]] = None,
        defer_module_names: Optional[set[str]] = None,
        layer_name: str = "",
    ) -> nn.Module:
        timer = getattr(self, "quant_region_timer", None)
        src_device = get_device(module)
        base_device = normalize_device_like(self.quantize_config.device)
        if base_device is None:
            base_device = CPU

        if src_device == META or _module_has_meta_tensors(module):
            if (
                skip_module_names
                and isinstance(self.turtle_model, LazyTurtle)
                and self.turtle_model is not None
            ):
                start = time.perf_counter() if timer is not None else None
                self._pre_quantize_with_skip(
                    module,
                    skip_module_names=skip_module_names,
                    defer_module_names=defer_module_names,
                    base_device=base_device,
                    layer_name=layer_name,
                )
                if timer is not None and start is not None:
                    timer.record(
                        "module_load",
                        time.perf_counter() - start,
                        source=f"pre_quantize {layer_name} batch_load skip={len(skip_module_names)}",
                    )
                return module

            return self.shell_module_materialize(
                target_submodule=module,
                device=base_device,
            )
        elif src_device == CPU and base_device != CPU:
            start = time.perf_counter() if timer is not None else None
            result = move_to(module, device=base_device)
            if timer is not None and start is not None:
                timer.record(
                    "module_move",
                    time.perf_counter() - start,
                    source=f"pre_quantize {getattr(module, 'full_name', '')} {src_device}->{base_device}",
                )
            return result
        else:
            return module

    def _pre_quantize_with_skip(
        self,
        module: nn.Module,
        *,
        skip_module_names: set[str],
        defer_module_names: Optional[set[str]],
        base_device: torch.device,
        layer_name: str,
    ) -> None:
        """Load non-leaf modules directly and batch load leaf modules in parallel."""

        if not isinstance(self.turtle_model, LazyTurtle) or self.turtle_model is None:
            return

        # Fall back to the module's actual qualified path if the caller did not provide
        # a layer_name. This keeps checkpoint tensor resolution correct for split/multi-prefix
        # decoder architectures at the cost of one whole-model scan per layer.
        if not layer_name:
            layer_name = _get_qualified_name(self.model, module)

        skip_set = set(skip_module_names)

        def _is_skipped(rel_name: str) -> bool:
            if rel_name in skip_set:
                return True
            for skip in skip_set:
                if rel_name.startswith(skip + "."):
                    return True
            return False

        # Load non-quantized structural modules (norms, routers, rotary embeddings, etc.)
        # with recurse=False so we only touch their own parameters and leave the skipped
        # leaf projections for the batched grouped load below.
        for rel_name, sub in module.named_modules():
            if _is_skipped(rel_name):
                continue
            if rel_name:
                full_path = f"{layer_name}.{rel_name}" if layer_name else rel_name
            else:
                # The module itself (the decoder layer) may own direct parameters/buffers.
                full_path = layer_name
            self.turtle_model.materialize_submodule(
                target_model=self.model,
                target_submodule=sub,
                device=base_device,
                module_path=full_path,
                recurse=False,
                tie_weights=False,
                show_progress=False,
            )

        # Batch load all skipped leaf modules onto the layer device. The cross-submodule
        # grouped loader produces a single "LazyTurtle: loading N grouped tensors ...
        # [modules=...]" log line instead of one-per-expert "using 1 worker(s)" lines.
        deferred = defer_module_names or set()
        batch: List[Tuple[torch.nn.Module, str, torch.device]] = []
        for rel_name in sorted(skip_module_names):
            if rel_name in deferred:
                continue
            try:
                sub = module.get_submodule(rel_name)
            except AttributeError:
                continue
            if sub is None:
                continue
            full_path = f"{layer_name}.{rel_name}" if layer_name else rel_name
            batch.append((sub, full_path, base_device))

        if batch:
            self.turtle_model.materialize_submodules(
                target_model=self.model,
                submodules=batch,
                tie_weights=False,
                show_progress=False,
            )

        # Tie weights once for the whole layer rather than once per materialized submodule.
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def post_quantize(self, module: nn.Module) -> nn.Module:
        #return self.offload_to_disk(module=module)
        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            start = time.perf_counter()
            result = move_to(module, device=CPU)
            timer.record(
                "module_move",
                time.perf_counter() - start,
                source=f"post_quantize {getattr(module, 'full_name', '')}->{CPU}",
            )
            return result
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
        expert_prev_op_by_group: Dict[str, str] = {}
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

        def _metadata_for_name(raw_name: str) -> ModuleTreeMetadata:
            return self.get_module_tree_metadata(strip_non_quantize_flags(raw_name))

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

            block_metadata = [_metadata_for_name(name) for name in block]
            is_moe_block = any(module_tree_flags_are_expert(item.flags) for item in block_metadata)
            is_moe_down_block = is_moe_block and any(
                MODULE_TREE_FLAG_DOWN in item.flags for item in block_metadata
            )
            block_roles = set().union(*(item.flags for item in block_metadata))
            is_moe_gate_up_block = is_moe_block and {
                MODULE_TREE_FLAG_GATE,
                MODULE_TREE_FLAG_UP,
            }.issubset(block_roles)
            if is_moe_down_block and last_module is not None and last_module_name is not None:
                for name in block:
                    metadata = _metadata_for_name(name)
                    if MODULE_TREE_FLAG_DOWN not in metadata.flags or metadata.expert_group is None:
                        continue
                    prev_op_name = expert_prev_op_by_group.get(metadata.expert_group)
                    if prev_op_name is None:
                        log.debug(
                            "awq_get_modules_for_scaling: skipping expert `%s` because its module-tree group has no gate/up predecessor",
                            name,
                        )
                        continue
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
                gate_up_proj_indices = []
                routed_gate_up_indices = []
                for block_index, (name, metadata) in enumerate(zip(block, block_metadata)):
                    if not module_tree_flags_are_expert(metadata.flags):
                        continue
                    if not metadata.flags & {MODULE_TREE_FLAG_GATE, MODULE_TREE_FLAG_UP}:
                        continue
                    gate_up_proj_indices.append(block_index)
                    if metadata.expert_group is not None:
                        expert_prev_op_by_group[metadata.expert_group] = strip_non_quantize_flags(name)
                    if MODULE_TREE_FLAG_ROUTED in metadata.flags:
                        routed_gate_up_indices.append(block_index)

                # Use the last one if any exist
                assert len(gate_up_proj_indices) > 0, "No expert gate_proj/up_proj found in block."
                last_up_proj_index = (routed_gate_up_indices or gate_up_proj_indices)[-1]

                candidate_name = strip_non_quantize_flags(block[last_up_proj_index])
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

    def _is_lazy_turtle_loaded(self, tensor: torch.Tensor, device: torch.device) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.device.type == "meta" or tensor.device != device:
            return False
        turtle_model = getattr(self, "turtle_model", None)
        if isinstance(turtle_model, LazyTurtle):
            return id(tensor) in turtle_model._loaded_tensor_ids
        # Non-LazyTurtle sources materialize all at once; trust non-meta + device match.
        return True

    def _checkpoint_tensors_from_module(self, module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for name in ("weight", "bias", "weight_scale", "weight_scale_2", "weight_scale_inv"):
            t = getattr(module, name, None)
            if isinstance(t, torch.Tensor):
                tensors[name] = t
        return tensors

    # Materialize the target shell module from the lazy turtle source on the requested device.
    def shell_module_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: torch.device,
            non_blocking: bool = False,
            role: str = "default",
            named_module: Optional["NamedModule"] = None,
            module_path: Optional[str] = None,
            recurse: bool = True,
            show_progress: bool = True,
    ) -> torch.nn.Module:
        timer = getattr(self, "quant_region_timer", None)
        start = time.perf_counter() if timer is not None else None
        if module_path is None and named_module is not None:
            module_path = getattr(named_module, "full_name", None)
        try:
            with self._turtle_lock:
                if module_path is None and timer is not None:
                    module_path = _get_qualified_name(self.model, target_submodule)
                if role == "quant_source" and named_module is not None:
                    quant_source = named_module.state.get("quant_source_module")
                    if not isinstance(quant_source, nn.Module):
                        decoder_plan = named_module.state.get("auto_module_decoder") or {}
                        target_dtype = decoder_plan.get(
                            "target_dtype",
                            getattr(getattr(target_submodule, "weight", None), "dtype", torch.float16),
                        )
                        checkpoint_tensors = None
                        # If the target submodule has already been materialized (e.g. by a
                        # batch prefetch), build the quant source from its live tensors
                        # instead of re-reading the checkpoint from disk.
                        weight = getattr(target_submodule, "weight", None)
                        if isinstance(self.turtle_model, LazyTurtle) and not self._is_lazy_turtle_loaded(weight, device):
                            checkpoint_tensors = self.turtle_model.checkpoint_tensors_for_submodule(
                                target_model=self.model,
                                target_submodule=target_submodule,
                                recurse=False,
                                module_path=module_path,
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
                    weight = getattr(target_submodule, "weight", None)
                    checkpoint_tensors = None
                    if self._is_lazy_turtle_loaded(weight, device):
                        # Reuse already-materialized live tensors for the decoder probe.
                        checkpoint_tensors = self._checkpoint_tensors_from_module(target_submodule)
                    if checkpoint_tensors is None:
                        checkpoint_tensors = turtle_model.checkpoint_tensors_for_submodule(
                            target_model=self.model,
                            target_submodule=target_submodule,
                            recurse=False,
                            module_path=module_path,
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
                        module_path=module_path,
                        recurse=recurse,
                        show_progress=show_progress,
                    )

                if role == "forward" and named_module is not None:
                    module = self._prepare_auto_decoder_forward_module(
                        target_submodule=module,
                        device=torch.device(device),
                        named_module=named_module,
                    )
            return module
        finally:
            if timer is not None and start is not None:
                timer.record(
                    "module_load",
                    time.perf_counter() - start,
                    source=f"shell_module_materialize {module_path} -> {device}",
                )

    def shell_direct_meta_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: Optional[torch.device] = None,
            module_path: Optional[str] = None,
    ):
        timer = getattr(self, "quant_region_timer", None)
        start = time.perf_counter() if timer is not None else None
        try:
            with self._turtle_lock:
                if module_path is None and timer is not None:
                    module_path = _get_qualified_name(self.model, target_submodule)
                if self.turtle_model is None:
                    return None
                return alias_direct_meta_from_turtle_for_submodule(
                    target_model=self.model,
                    turtle_model=self.turtle_model,
                    target_submodule=target_submodule,
                    device=device,
                    module_path=module_path,
                )
        finally:
            if timer is not None and start is not None:
                timer.record(
                    "module_load",
                    time.perf_counter() - start,
                    source=f"shell_direct_meta_materialize {module_path} -> {device}",
                )

    def lazy_turtle_batch_materialize_submodules(
        self,
        submodules: List[Tuple[torch.nn.Module, str, torch.device]],
        non_blocking: bool = False,
    ) -> None:
        """Batch materialize many submodules from the LazyTurtle checkpoint source.

        `submodules` is a list of (target_submodule, module_path, target_device) tuples.
        This lets the grouped loader dispatch all requested tensors in one parallel
        pass instead of one tensor per submodule call.
        """

        turtle_model = getattr(self, "turtle_model", None)
        if not isinstance(turtle_model, LazyTurtle) or not submodules:
            return
        timer = getattr(self, "quant_region_timer", None)
        start = time.perf_counter() if timer is not None else None
        try:
            turtle_model.materialize_submodules(
                target_model=self.model,
                submodules=submodules,
                non_blocking=non_blocking,
            )
        finally:
            if timer is not None and start is not None:
                devices = sorted({str(device) for _, _, device in submodules})
                timer.record(
                    "module_load_batch",
                    time.perf_counter() - start,
                    source=f"LazyTurtle batch count={len(submodules)} devices={','.join(devices)}",
                )

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
    def _build_layer_modules_for_tree(cls, tree, include_capture_only: bool = False):
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
        alias_groups: Dict[tuple[str | None, int], List[tuple[str, bool, bool, List[str]]]] = {}
        alias_meta: Dict[tuple[str | None, int], Dict[str, int]] = {}
        alias_seq = count()
        group_seq = count()

        def _role_flags(flags: List[str]) -> List[str]:
            """Return role/semantic flags (exclude special/numeric markers)."""
            return sorted(f for f in flags if f and not f.isdigit() and f not in ("!", "?"))

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

        def process_entries(
            parent_token: str,
            entries,
            parent_group_offset: int = 0,
            scope_key: str | None = None,
            inherited_role_flags: frozenset[str] = frozenset(),
            inherited_expert_group: str | None = None,
        ):
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

            declared_parent_flags = frozenset(_role_flags(parent_flags))
            parent_role_flags = normalize_module_tree_flags(declared_parent_flags | inherited_role_flags)
            declared_expert_role = declared_parent_flags & MODULE_TREE_EXPERT_FLAGS
            parent_expert_group = parent_name if declared_expert_role else inherited_expert_group
            parent_extra_flags = sorted(parent_role_flags)

            def _make_entry(
                full_path: str,
                has_bang: bool,
                capture_only: bool,
                extra_flags: List[str],
                *,
                alias_base: int,
                alias_rel: int,
                alias_scope: str | None,
            ) -> tuple:
                return (full_path, has_bang, capture_only, extra_flags, alias_scope, (alias_base, alias_rel))

            child_group_offset = parent_group_offset
            add_parent = parent_has_bang or (parent_capture_only and include_capture_only)
            if add_parent:
                cls._set_module_tree_flags(
                    parent_name,
                    frozenset(parent_extra_flags),
                    expert_group=parent_expert_group,
                )
                alias_base = parent_rel_group if parent_has_numeric else parent_group
                parent_entry_scope = f"{parent_alias_scope}.__parent__" if parent_alias_scope is not None else None
                groups[parent_group].append(
                    _make_entry(
                        parent_name,
                        parent_has_bang,
                        parent_capture_only,
                        parent_extra_flags,
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
                    child_extra_flags = sorted(
                        normalize_module_tree_flags(frozenset(_role_flags(child_flags)) | parent_role_flags)
                    )
                    cls._set_module_tree_flags(
                        full_path,
                        frozenset(child_extra_flags),
                        expert_group=parent_expert_group,
                    )
                    groups[grp].append(
                        _make_entry(
                            full_path,
                            has_bang,
                            capture_only,
                            child_extra_flags,
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
                        template_expert_group = (
                            template_parent
                            if module_tree_flags_are_expert(parent_role_flags)
                            else parent_expert_group
                        )

                        # Handle special case where sub_entries is ("#",) or "#" - this means use the parent path directly
                        if (isinstance(sub_entries, (tuple, list)) and len(sub_entries) == 1 and sub_entries[0] == "#") or sub_entries == "#":
                            # For ("#",) or "#" format, use the template_parent directly with default group 0
                            alias_scope = scope if parent_has_numeric else template_parent
                            alias_base = parent_rel_group if parent_has_numeric else expert_offset
                            cls._set_module_tree_flags(
                                template_parent,
                                frozenset(parent_extra_flags),
                                expert_group=template_expert_group,
                            )
                            groups[expert_offset].append(
                                _make_entry(
                                    template_parent,
                                    False,
                                    False,
                                    parent_extra_flags,
                                    alias_base=alias_base,
                                    alias_rel=0,
                                    alias_scope=alias_scope,
                                )
                            )
                        else:
                            sub_groups = process_entries(
                                template_parent_token,
                                sub_entries,
                                expert_offset,
                                scope,
                                parent_role_flags,
                                template_expert_group,
                            )
                            for grp, items in sub_groups.items():
                                groups[grp].extend(items)
                    else:
                        # Nested structure: process recursively with full path
                        # Special case: empty string key means use parent path directly
                        # and drops the MoE container markers (e.g. the dense MLP fallback
                        # inside ``mlp:moe`` in Laguna's first layer).
                        if sub_parent == "":
                            full_sub_parent = parent_name
                            sub_role_flags = parent_role_flags - MODULE_TREE_MOE_FLAGS
                            sub_expert_group = None
                        else:
                            full_sub_parent = (
                                f"{parent_name}.{sub_parent}"
                                if parent_name else sub_parent
                            )
                            sub_role_flags = parent_role_flags
                            sub_expert_group = parent_expert_group
                        sub_groups = process_entries(
                            full_sub_parent,
                            sub_entries,
                            current_offset,
                            scope,
                            sub_role_flags,
                            sub_expert_group,
                        )
                        for grp, items in sub_groups.items():
                            groups[grp].extend(items)
                        # Update offset for next sibling to avoid conflicts
                        if sub_groups:
                            current_offset = max(sub_groups.keys()) + 1

            return groups

        def _register_alias(order_idx: int, item: tuple[str, bool, bool, List[str], str | None, tuple[int, int]]):
            full_path, has_bang, capture_only, extra_flags, scope, alias_parts = item
            if capture_only and not include_capture_only:
                return
            alias_scope = scope
            alias_base, alias_rel = alias_parts
            alias_index = alias_base + alias_rel
            key = (alias_scope, alias_index)
            meta = alias_meta.get(key)
            if meta is None:
                alias_meta[key] = {"order": order_idx, "seq": next(alias_seq)}
                alias_groups[key] = [(full_path, has_bang, capture_only, extra_flags)]
            else:
                meta["order"] = min(meta["order"], order_idx)
                alias_groups[key].append((full_path, has_bang, capture_only, extra_flags))

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)

            for g in sorted(groups):
                order_idx = next(group_seq)
                items = groups[g]
                for item in items:
                    _register_alias(order_idx, item)

        for key in sorted(alias_groups.keys(), key=lambda k: (alias_meta[k]["order"], alias_meta[k]["seq"])):
            block = []
            for full_path, has_bang, capture_only, _ in alias_groups[key]:
                name = full_path
                if has_bang:
                    name += NOT_QUANTIZE_FLAG
                if capture_only and include_capture_only:
                    name += CAPTURE_ONLY_FLAG
                # Role/semantic flags (e.g. :q, :gate, :up, :down) are stored in
                # ``NamedModule.state["module_tree_flags"]`` by the looper; keeping
                # them out of the emitted name string avoids breaking consumers that
                # resolve module paths from these tokens.
                block.append(name)
            out_blocks.append(block)

        return out_blocks

    @classmethod
    def build_layer_modules(cls, tree, include_capture_only: bool = False):
        blocks = []
        seen = set()
        for tree_variant in cls._iter_module_tree_variants(tree):
            for block in cls._build_layer_modules_for_tree(tree_variant, include_capture_only=include_capture_only):
                key = tuple(block)
                # Shared blocks across tree variants should keep first-seen order.
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(block)
        return blocks

    @classmethod
    def get_modules_with_direct_meta_tensors(cls, model: nn.Module):
        """
        Return module paths whose direct params/buffers are materialized outside
        normal layer/base-module traversal.
        """
        out = []
        module_names = cls.modules_with_direct_meta_tensors or cls.direct_meta_modules or []
        for module_name in module_names:
            module = get_module(model, module_name)
            if isinstance(module, nn.Module):
                out.append(module_name)
        return out

    @classmethod
    def get_direct_meta_modules(cls, model: nn.Module):
        return cls.get_modules_with_direct_meta_tensors(model)

    @classmethod
    def get_base_modules(cls, model: nn.Module):
        """
        Return list of base modules directly under the root path but not the layer container.
        """
        all_prefix_paths = []
        for tree in cls._iter_module_tree_variants():
            try:
                sharp_idx = tree.index("#")
            except ValueError:
                raise ValueError("module_tree must contain '#' to separate hierarchy")

            assert sharp_idx > 0, "failed to get_base_modules"
            for path in cls._expand_module_tree_prefixes(tree):
                prefix = tuple(path.split("."))
                if prefix not in all_prefix_paths:
                    all_prefix_paths.append(prefix)

        if not all_prefix_paths:
            return []

        exclude_by_parent: Dict[tuple[str, ...], set[str]] = {}
        for parts in all_prefix_paths:
            for i in range(len(parts) - 1):
                parent_path = parts[: i + 1]
                # Each tree reserves the next path segment for layer traversal.
                exclude_by_parent.setdefault(parent_path, set()).add(parts[i + 1])

        out = []
        seen = set()
        for parts in all_prefix_paths:
            for i in range(len(parts) - 1):
                path = parts[: i + 1]
                base = model
                exclude = exclude_by_parent.get(path, set())

                for node in path:
                    base = getattr(base, node, None)
                    if base is None:
                        break
                if base is None:
                    continue

                for name, _ in base.named_children():
                    # print("name", base, name, exclude)
                    if name in exclude:
                        continue
                    full_name = ".".join((*path, name))
                    if full_name in seen:
                        continue
                    seen.add(full_name)
                    out.append(full_name)
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

    def get_input_embeddings(self) -> Optional[nn.Module]:
        getter = getattr(self.model, "get_input_embeddings", None)
        return getter() if callable(getter) else None

    def get_input_embeddings_name(self) -> Optional[str]:
        module = self.get_input_embeddings()
        return get_module_name(self.model, module) if module is not None else None

    def get_output_embeddings(self) -> Optional[nn.Module]:
        getter = getattr(self.model, "get_output_embeddings", None)
        return getter() if callable(getter) else None

    def get_output_embeddings_name(self) -> Optional[str]:
        module = self.get_output_embeddings()
        return get_module_name(self.model, module) if module is not None else None

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

        if quant_method not in {
            METHOD.GPTQ,
            METHOD.GGUF,
            METHOD.FP8,
            METHOD.BITSANDBYTES,
            METHOD.EXL3,
            METHOD.QVQ,
            METHOD.PARO,
        }:
            log.warn(
                f"Module Tree AutoCompat: Failed, quant_method={quant_method}, "
                "only support GPTQ/GGUF/FP8/BITSANDBYTES/EXL3/QVQ/PAROQUANT"
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
                lowered_name = n.lower()
                if any(k in lowered_name for k in possible_names):
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
