# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import collections
import copy
import functools
import hashlib
import json
import math
import operator
import os
import re
import secrets
import shutil
import stat
import struct
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import accelerate
import pcre
import torch
import torch.nn as nn
import transformers
from packaging import version
from safetensors import safe_open
from torch.nn.modules.conv import _ConvNd
from transformers import PretrainedConfig
from transformers.pytorch_utils import id_tensor_storage
from transformers.utils.hub import cached_file

from ..adapter.adapter import LORA_MERGED_WEIGHT_PATHS, Adapter
from ..looper.named_module import NamedModule
from ..models._const import (
    CPU,
    DEVICE,
    EXPERT_INDEX_PLACEHOLDER,
    SUPPORTS_MODULE_TYPES,
)
from ..nn_modules.qlinear import (
    QZEROS_FORMAT_BUFFER,
    AWQuantLinear,
    BaseQuantLinear,
    GPTQQuantLinear,
)
from ..nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from ..nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
from ..nn_modules.qlinear.marlin import MarlinLinear
from ..nn_modules.qlinear.torch import TorchQuantEmbeddings
from ..quantization import FORMAT, QuantizeConfig
from ..quantization.config import (
    FORMAT_FIELD_CODE,
    METHOD,
    ResolvedModuleConfig,
    _normalize_quant_bits,
    dynamic_get,
    dynamic_matching_pattern,
    quant_bits_width,
    resolve_model_rotation,
    resolve_quant_format,
)
from ..quantization.module_config import resolve_module_constructor_config
from . import has_gil_disabled
from .backend import BACKEND, normalize_backend
from .checkpoint_paths import assert_auxiliary_copy_allowed
from .ctx import ctx
from .device import get_device
from .hf import get_hf_config_dtype, resolve_checkpoint_subfolder
from .hub import hf_hub_download, model_info
from .importer import select_quant_linear, validate_quant_linear
from .kernel import resolve_kernel_config, snapshot_kernel_config
from .logger import log_time_block, setup_logger
from .model_dequant import _correct_gptq_v1_qzeros, _revert_gptq_v1_qzeros_correction
from .p2p import safe_module_to, safe_move_to
from .packing import _auto_pack_impl
from .tensor import tensor_buffer
from .tied_weights import tied_weights_keys
from .torch import HAS_CUDA, as_torch_device, release_pinned_host_cache, torch_empty_cache


log = setup_logger()
_REQUIRES_VERSION_RE = pcre.compile(r"(<=|>=|==|<|>)\s*([\d\.]+)")


_DTYPE_SAFE_MAP = {
    torch.float32: ("F32", 4),
    torch.float16: ("F16", 2),
    torch.float64: ("F64", 8),
    torch.bfloat16: ("BF16", 2),
    torch.int64: ("I64", 8),
    torch.int32: ("I32", 4),
    torch.int16: ("I16", 2),
    torch.int8: ("I8", 1),
    torch.uint8: ("U8", 1),
    torch.bool: ("BOOL", 1),
}

# TODO: keep the float8 entries below in sync with ``_FLOAT8_DTYPE_NAMES``.
#
# Gotcha 1 — the header identifier is NOT ``"F8_" + name.upper()``. The
# safetensors JSON schema uses short names that drop the ``fn`` suffix and the
# ``fnu`` marker: ``torch.float8_e4m3fn`` -> ``F8_E4M3``,
# ``torch.float8_e5m2`` -> ``F8_E5M2``, ``torch.float8_e8m0fnu`` -> ``F8_E8M0``.
# Guessing the identifier (e.g. emitting ``F8_E4M3FN``) makes the reader reject the
# whole file with "unknown variant" while unpacking the header. Verified against
# safetensors 0.8.0. Add new entries by checking the accepted spelling, never by
# transforming the torch dtype name.
#
# Gotcha 2 — every supported float8 dtype needs an entry here as well as in
# ``_FLOAT8_DTYPE_NAMES``. The FNUZ variants use the explicit safetensors IDs
# ``F8_E4M3FNUZ`` and ``F8_E5M2FNUZ``; streaming export and reference offload
# both use this map when writing their headers.
#
# The failure mode when a dtype is missing here is a hard ``NotImplementedError``
# from ``_torch_dtype_to_safetensors`` at export time, not silent byte corruption.
# A widened ``_FLOAT8_DTYPE_NAMES`` does NOT automatically become exportable: the
# two tuples are intentionally separate (sizes vs. serializable identifiers), so
# adding a name in one place without the other silently leaves the dtype
# unexportable. This is a known wart; deriving both from a single
# ``(torch_name, safetensors_id, num_bytes)`` table would remove it.
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_SAFE_MAP[torch.float8_e4m3fn] = ("F8_E4M3", 1)
if hasattr(torch, "float8_e5m2"):
    _DTYPE_SAFE_MAP[torch.float8_e5m2] = ("F8_E5M2", 1)
if hasattr(torch, "float8_e4m3fnuz"):
    _DTYPE_SAFE_MAP[torch.float8_e4m3fnuz] = ("F8_E4M3FNUZ", 1)
if hasattr(torch, "float8_e5m2fnuz"):
    _DTYPE_SAFE_MAP[torch.float8_e5m2fnuz] = ("F8_E5M2FNUZ", 1)
if hasattr(torch, "float8_e8m0fnu"):
    _DTYPE_SAFE_MAP[torch.float8_e8m0fnu] = ("F8_E8M0", 1)

_FLOAT8_DTYPE_NAMES = tuple(
    name
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
)
_FLOAT4_PACKED_DTYPE_NAMES = tuple(
    name for name in ("float4_e2m1fn_x2",) if hasattr(torch, name)
)

# Byte-size fallbacks keep ancillary metadata math working for torch floatx
# dtypes even when the current safetensors header schema cannot serialize them.
_DTYPE_NUM_BYTES = dict.fromkeys((*[getattr(torch, name) for name in _FLOAT8_DTYPE_NAMES], *[getattr(torch, name) for name in _FLOAT4_PACKED_DTYPE_NAMES]), 1)


_DTYPE_STR_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "float64": torch.float64,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int": torch.int32,
    "int16": torch.int16,
    "short": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

for name in _FLOAT8_DTYPE_NAMES:
    dtype = getattr(torch, name)
    _DTYPE_STR_MAP[name] = dtype
    _DTYPE_STR_MAP[f"f8_{name.removeprefix('float8_')}"] = dtype

if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_STR_MAP["f8_e4m3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _DTYPE_STR_MAP["f8_e5m2"] = torch.float8_e5m2
if hasattr(torch, "float8_e8m0fnu"):
    _DTYPE_STR_MAP["float8_e8m0"] = torch.float8_e8m0fnu
    _DTYPE_STR_MAP["f8_e8m0"] = torch.float8_e8m0fnu

for name in _FLOAT4_PACKED_DTYPE_NAMES:
    dtype = getattr(torch, name)
    _DTYPE_STR_MAP[name] = dtype
    _DTYPE_STR_MAP[f"f4_{name.removeprefix('float4_')}"] = dtype

MoETopKState = List[Tuple[nn.Module, str, int]]

MOE_TOPK_FIELD_NAMES = [
    "top_k",
    "moe_k", # ernie4_5_vl_moe
    # K2 Horizon's MoVA router uses this name instead of ``top_k``.
    "num_experts_per_tok",
]

MOE_NUM_EXPERTS_FIELD_NAMES = [
    "num_experts",
    "moe_num_experts",  # ernie4_5_vl_moe
]


def _torch_dtype_num_bytes(dtype: torch.dtype) -> int:
    if dtype in _DTYPE_SAFE_MAP:
        return _DTYPE_SAFE_MAP[dtype][1]
    if dtype in _DTYPE_NUM_BYTES:
        return _DTYPE_NUM_BYTES[dtype]
    raise NotImplementedError(f"Unsupported dtype for safetensors export: {dtype}")


def _torch_dtype_to_safetensors(dtype: torch.dtype) -> str:
    if dtype not in _DTYPE_SAFE_MAP:
        raise NotImplementedError(f"Unsupported dtype for safetensors export: {dtype}")
    return _DTYPE_SAFE_MAP[dtype][0]


def _dtype_string_to_torch(dtype_str: Optional[str], fallback: torch.dtype) -> torch.dtype:
    if dtype_str is None:
        return fallback
    key = dtype_str.lower()
    return _DTYPE_STR_MAP.get(key, fallback)


@dataclass(frozen=True)
class OffloadTensorRef:
    path: str
    torch_dtype: torch.dtype
    shape: Tuple[int, ...]
    format: str  # 'dat' or 'safetensors'
    weight_name: Optional[str] = None
    data_offsets: Optional[Tuple[int, int]] = None
    sha256: Optional[str] = None

    @property
    def num_bytes(self) -> int:
        return _torch_dtype_num_bytes(self.torch_dtype) * math.prod(self.shape or (1,))


@dataclass
class TensorSource:
    name: str
    torch_dtype: torch.dtype
    shape: Tuple[int, ...]
    source: Union[torch.Tensor, OffloadTensorRef]

    @property
    def num_bytes(self) -> int:
        return _torch_dtype_num_bytes(self.torch_dtype) * math.prod(self.shape or (1,))

def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


def _module_has_meta_tensors(module: nn.Module) -> bool:
    for param in module.parameters(recurse=True):
        if getattr(param, "is_meta", False) or param.device.type == "meta":
            return True
    for buf in module.buffers(recurse=True):
        if getattr(buf, "is_meta", False) or buf.device.type == "meta":
            return True
    return False


def move_to(obj: torch.Tensor | nn.Module, device: torch.device, dtype: torch.dtype = None):
    device = as_torch_device(device)
    if isinstance(obj, nn.Module) and _module_has_meta_tensors(obj):
        if not accelerate.utils.has_offloaded_params(obj):
            raise NotImplementedError(
                "Cannot move a module that still contains meta tensors without offload hooks. "
                "Materialize it first before calling move_to()."
            )

        # Accelerate disk-offloaded modules keep meta placeholders until they are
        # explicitly restored, so materialize those leaves before the device move.
        from .offload import undo_offload_to_disk

        return undo_offload_to_disk(obj, device=device, dtype=dtype)

    if get_device(obj) != device or dtype is not None:
        # Cross-device moves must be direction-aware: on an asymmetric ROCm
        # topology a direct peer copy segfaults instead of raising.
        if isinstance(obj, nn.Module):
            obj = safe_module_to(obj, device, dtype=dtype)
        else:
            obj = safe_move_to(obj, device, dtype=dtype)

    return obj


def move_module_to_cpu_pinned(
    module: nn.Module,
    *,
    pin_min_bytes: int = 8 * 1024 * 1024,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> nn.Module:
    """Move a module to CPU with pinned staging for large accelerator tensors.

    Pageable D2H copies synchronously fault each output page and dominated
    layer offload. Large tensors keep pinned CPU storage so the device transfer
    needs no second staging copy. Small tensors stay pageable to avoid pinning
    allocator metadata and scalar state.
    """

    if pin_min_bytes < 0:
        raise ValueError("pin_min_bytes must be non-negative")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    streams: dict[torch.device, torch.cuda.Stream] = {}

    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "meta":
            # Meta tensors carry no storage; copying them raises.
            return tensor
        if tensor.device.type != "cuda":
            # XPU, MPS, NPU, XLA and CPU all take the ordinary path. Returning
            # the accelerator tensor unchanged would defeat layer offload.
            return tensor.to(device=CPU)
        if tensor.numel() * tensor.element_size() < pin_min_bytes:
            return tensor.to(device=CPU)

        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        try:
            result = torch.empty(
                source.shape,
                dtype=source.dtype,
                device=CPU,
                pin_memory=True,
            )
        except RuntimeError:
            return source.to(device=CPU)

        source_flat = source.view(-1)
        result_flat = result.view(-1)
        chunk_elements = max(1, chunk_bytes // source.element_size())
        streams.setdefault(
            source.device,
            torch.cuda.current_stream(source.device),
        )
        for start in range(0, source.numel(), chunk_elements):
            end = min(start + chunk_elements, source.numel())
            result_flat[start:end].copy_(
                source_flat[start:end],
                non_blocking=True,
            )
        return result

    moved = module._apply(to_cpu)
    for stream in streams.values():
        stream.synchronize()
    return moved


def nested_move_to(v, device, dtype: torch.dtype = None):
    if isinstance(v, torch.Tensor):
        return move_to(v, device=device, dtype=dtype)

    elif isinstance(v, dict):
        return {
            k: nested_move_to(val, device=device, dtype=dtype)
            for k, val in v.items()
        }

    elif isinstance(v, (list, tuple)):
        return type(v)(
            nested_move_to(e, device=device, dtype=dtype)
            for e in v
        )

    else:
        return v


def find_modules(module: nn.Module, layers=None, name: str="") -> Dict[str, nn.Module]:
    if not layers:
        layers = SUPPORTS_MODULE_TYPES

    if isinstance(module, tuple(layers)):
       return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_module_by_name(module, child_name):
    # get the child module by its name relative to the module
    for name, m in module.named_modules():
        if name == child_name:
            return m
    raise ValueError(f"Cannot find child_name {child_name} in module {module}")


def get_module_by_name_prefix(model, module_name: Union[List[str], str]):
    """Resolve the first path that is exactly ``prefix`` or a dotted descendant.

    A plain ``startswith`` test treats ``mlp_extra`` as ``mlp`` and can silently
    select a sibling instead of the declared module. Prefer an exact path, then
    reject ambiguous descendant matches instead of choosing among them.
    """
    module_name_list = module_name if isinstance(module_name, list) else [module_name]
    modules = dict(model.named_modules())
    for prefix in module_name_list:
        if isinstance(prefix, str) and prefix in modules:
            return modules[prefix], prefix
    for prefix in module_name_list:
        if not isinstance(prefix, str) or not prefix:
            continue
        matches = [
            (name, module)
            for name, module in modules.items()
            if name.startswith(f"{prefix}.")
        ]
        if len(matches) == 1:
            return matches[0][1], prefix
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous module prefix {prefix!r}: matches {[name for name, _ in matches[:8]]}"
            )

    return None, ""


def thinker_text_model(model: nn.Module) -> nn.Module:
    """Return the text decoder owned by a Qwen Omni ``thinker`` submodule.

    transformers 5.22 moved the decoder from ``thinker.model`` to
    ``thinker.model.language_model``; older releases keep ``embed_tokens``,
    ``layers``, ``norm`` and ``rotary_emb`` directly on ``thinker.model``.
    """

    thinker = getattr(model, "thinker", model)
    inner = getattr(thinker, "model", thinker)
    return getattr(inner, "language_model", inner)


def thinker_text_prefix(model: nn.Module) -> str:
    """Return the module path of the Qwen Omni thinker text decoder.

    The path is relative to ``model`` and mirrors what
    :func:`thinker_text_model` resolves, so ``model.get_submodule(path)`` is
    the decoder whether ``model`` is the outer Omni wrapper or a bare thinker.
    An empty string means ``model`` is already the decoder.
    """

    parts = []
    node = model
    if hasattr(node, "thinker"):
        parts.append("thinker")
        node = node.thinker
    if hasattr(node, "model"):
        parts.append("model")
        node = node.model
    if getattr(node, "language_model", None) is not None:
        parts.append("language_model")
    return ".".join(parts)


def thinker_multimodal_model(model: nn.Module) -> nn.Module:
    """Return the module that owns the thinker's visual and audio encoders.

    transformers 5.22 moved them from ``thinker`` to ``thinker.model``.
    """

    thinker = getattr(model, "thinker", model)
    inner = getattr(thinker, "model", thinker)
    return inner if hasattr(inner, "visual") else thinker


def get_layers_with_prefixes(model, module_name: Union[List[str], str]):
    """Resolve one or more layer containers into a flat layer list plus per-layer names.

    Existing model definitions often expose a single `layers` ModuleList, but some
    architectures split the decoder into multiple stacks that still need to be
    quantized as one ordered sequence. This helper flattens every matching layer
    container while preserving the source prefix for each layer.
    """

    module_name_list = module_name if isinstance(module_name, list) else [module_name]

    layers = []
    layer_names = []
    seen_container_ids = set()

    for prefix in module_name_list:
        if not prefix:
            continue
        try:
            module = get_module_by_name(model, prefix)
        except ValueError:
            module = None
        if module is None:
            continue

        container_id = id(module)
        if container_id in seen_container_ids:
            continue
        seen_container_ids.add(container_id)

        if isinstance(module, (nn.ModuleList, list, tuple)):
            for local_index, layer in enumerate(module):
                if isinstance(layer, nn.Module):
                    layers.append(layer)
                    layer_names.append(f"{prefix}.{local_index}")
        elif isinstance(module, nn.Module):
            layers.append(module)
            layer_names.append(prefix)

    if layers:
        return layers, layer_names

    module, prefix = get_module_by_name_prefix(model, module_name_list)
    if module is None:
        return None, []
    if isinstance(module, (nn.ModuleList, list, tuple)):
        return list(module), [f"{prefix}.{index}" for index in range(len(module))]
    return [module], [prefix]


def get_layer_name(layer_names: Union[List[str], str, None], layer_index: int) -> str:
    """Return the concrete full layer path for one flattened layer index."""

    if not layer_names:
        return ""
    if isinstance(layer_names, str):
        return layer_names
    if layer_index < len(layer_names):
        return layer_names[layer_index]
    return layer_names[-1]


def get_module_by_name_suffix(model, module_name: str):
    """Resolve a path suffix at a dotted segment boundary.

    Exact paths win. Multiple suffix matches are an error: returning the first
    one makes device placement and quantization depend on module registration
    order.
    """
    modules = dict(model.named_modules())
    exact = modules.get(module_name)
    if exact is not None:
        return exact
    matches = [
        (name, module)
        for name, module in modules.items()
        if name.endswith(f".{module_name}")
    ]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous module suffix {module_name!r}: matches {[name for name, _ in matches[:8]]}"
        )
    return None

def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module

def make_quant(
    module,
    qcfg: QuantizeConfig,
    quant_result: Dict[str, Dict[str, Any]],
    backend: BACKEND,
    lm_head_name: str,
    pack: bool = False,
    device: DEVICE = None,
    from_quantized: bool = False,
    dtype: Optional[torch.dtype] = None,
    bias_dtype: Optional[torch.dtype] = None,
    is_sharded: bool = False,
    cpu_offload: bool = False,
) -> Type[BaseQuantLinear]:

    bits = qcfg.runtime_bits
    group_size =qcfg.group_size
    extension = qcfg.adapter
    format = resolve_quant_format(qcfg.format, qcfg.method)
    desc_act = qcfg.desc_act
    sym = qcfg.sym
    dynamic = qcfg.dynamic
    pack_dtype = qcfg.pack_dtype
    init_kwargs = qcfg.quant_linear_init_kwargs()
    init_kwargs["kernel"] = snapshot_kernel_config(qcfg)

    export_quant_method = qcfg.export_quant_method()
    backend = normalize_backend(backend, quant_method=export_quant_method)

    # BitBLAS-native checkpoints can load directly. Other formats need a compatible preload kernel first.
    if not pack and backend in [BACKEND.GPTQ_BITBLAS, BACKEND.AWQ_BITBLAS]:
        if format in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
            backend = BACKEND.GPTQ_TORCH
        elif qcfg.quant_method == METHOD.AWQ and format == FORMAT.GEMM:
            backend = BACKEND.AWQ_TORCH

    # returns multiple validated kernels
    quant_linear_candidates = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=format,
        quant_method=export_quant_method,
        pack=pack,
        dynamic=dynamic,
        device=device,
        pack_dtype=pack_dtype,
        dtype=dtype,
        multi_select=True,
        adapter=extension,
        is_sharded=is_sharded,
        cpu_offload=cpu_offload,
    )

    log.info(f"Kernel: candidates -> `[{', '.join(cls.__name__ for cls in quant_linear_candidates)}]`")

    # Per-module kernel selection: each submodule picks the first candidate that
    # validates against its resolved (possibly dynamic-overridden) quant config.
    linear_cls = create_quant_layer(
        linear_candidates=quant_linear_candidates,
        bits=bits,
        desc_act=desc_act,
        dynamic=dynamic,
        group_size=group_size,
        module=module,
        sym=sym,
        device=device,
        quant_result=quant_result,
        lm_head_name=lm_head_name,
        pack_dtype=pack_dtype,
        backend=backend,
        adapter=qcfg.adapter,
        format=format,
        init_kwargs=init_kwargs,
        dtype=dtype,
        bias_dtype=bias_dtype,
    )
    log.info(f"Kernel: selected -> `{linear_cls.__name__}`.")
    return linear_cls

def create_quant_module(
    name: str,
    linear_cls: Type[BaseQuantLinear],
    bits,
    desc_act: bool,
    dynamic,
    group_size: int,
    module: nn.Module,
    submodule: nn.Module,
    sym: bool,
    device: DEVICE,
    lm_head_name: str,
    pack_dtype: torch.dtype,
    format: FORMAT = FORMAT.GPTQ,
    backend: BACKEND = BACKEND.AUTO,
    register_buffers: bool = True,
    adapter: Optional[Adapter] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    bias_dtype: Optional[torch.dtype] = None,
    resolved_module_config: Optional[ResolvedModuleConfig] = None,
    transfer_from: Optional[nn.Module] = None,

):
    # An adapter only covers the modules named by its PEFT target list and
    # actually present in its tensor file. Passing it to every quantized module
    # made a standard partial LoRA (q_proj/v_proj only) assert on the modules it
    # does not cover (issue #4210).
    if adapter is not None:
        if not adapter.targets_module(name):
            adapter = None
        elif not adapter.has_weights(name):
            log.warn(
                f"Adapter: no LoRA tensors for targeted module `{name}`; skipping."
            )
            adapter = None

    # unwrap named module
    if isinstance(submodule, NamedModule):
        # print(f"offloading named module: {module.full_name}")
        submodule = submodule.module

    if isinstance(submodule, nn.Embedding):
        linear_cls = TorchQuantEmbeddings

    # submodule may be BaseQuantLinear, and the next QuantLinear is selected because of in_features/out_features
    # mismatch and other reasons.
    # In this case, need to call list_buffer() to get the device.
    if not isinstance(submodule, BaseQuantLinear):
        ori_layer_device = next(submodule.parameters()).device
    else:
        ori_layer_device = submodule.list_buffers()[0].device

    # Transformers builds quantized models on the meta device before assigning
    # checkpoint tensors, so allow either CPU or meta here and keep the
    # replacement layer on the device the original layer occupied. A rebuild of
    # an already-loaded layer (`transfer_from`) legitimately sees a live device:
    # its checkpoint tensors are moved into the replacement below.
    if transfer_from is None and ori_layer_device.type not in {CPU.type, "meta"}:
        raise AssertionError(
            f"Expected `{name}` to reside on CPU or meta during quant module creation, "
            f"but found tensors on `{ori_layer_device}`."
        )

    if isinstance(submodule, NamedModule):
        in_features = submodule.state.get("in_features")
        out_features = submodule.state.get("out_features")
    elif isinstance(submodule, nn.Linear):
        in_features = submodule.in_features
        out_features = submodule.out_features
    elif isinstance(submodule, nn.Embedding):
        in_features = submodule.num_embeddings
        out_features = submodule.embedding_dim
    elif isinstance(submodule, _ConvNd):
        in_features = submodule.in_channels
        out_features = submodule.out_channels
    elif isinstance(submodule, transformers.Conv1D):
        in_features = submodule.weight.shape[0]
        out_features = submodule.weight.shape[1]
    elif isinstance(submodule, BaseQuantLinear):
        # if submodule is already a quant layer, we need to get in_features and out_features from the submodule
        in_features = submodule.in_features
        out_features = submodule.out_features
    else:
        raise NotImplementedError(f"Unsupported module {submodule}")

    bias = submodule.bias is not None if hasattr(submodule, "bias") else False

    if resolved_module_config is not None and resolved_module_config.module_name != name:
        raise ValueError("resolved module config does not match quantized module name")
    if resolved_module_config is not None:
        overrides = (
            False
            if resolved_module_config.excluded
            else resolved_module_config.overrides
        )
    elif dynamic is not None:
        overrides = dynamic_get(dynamic=dynamic, module_name=name)
    else:
        overrides = None

    # One owner resolves the module's effective constructor settings. Packing
    # and checkpoint restore both consume this config instead of reinterpreting
    # dynamic overrides independently.
    module_config = resolve_module_constructor_config(
        bits=_normalize_quant_bits(bits, format_value=format),
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        pack_dtype=pack_dtype,
        format=format,
        init_kwargs=init_kwargs,
        overrides=overrides,
    )
    if module_config is None:
        # negative module match, skip this module
        return
    tmp_bits = module_config.bits
    tmp_group_size = module_config.group_size
    tmp_desc_act = module_config.desc_act
    tmp_sym = module_config.sym
    tmp_pack_dtype = module_config.pack_dtype
    tmp_init_kwargs = module_config.options_dict()

    validate_bits = quant_bits_width(tmp_bits)
    constructor_bits = tmp_bits if getattr(linear_cls, "QUANT_TYPE", None) == "gguf" else validate_bits

    # GPTQ modules need the checkpoint format to select between the continuous
    # (gptq/gptq_v2) and planar (gptq_p) packed layouts.
    if issubclass(linear_cls, GPTQQuantLinear):
        tmp_init_kwargs.setdefault("format", format)

    # Keep concrete device ordinals through per-module validation.  The shared
    # selector cache validates every target in a multi-GPU selector and lets
    # device-sensitive kernels query the capability of that exact target.
    _, err = validate_quant_linear(
        linear_cls,
        bits=validate_bits,
        group_size=tmp_group_size,
        desc_act=tmp_desc_act,
        sym=tmp_sym,
        pack_dtype=tmp_pack_dtype,
        dtype=dtype,
        in_features=in_features,
        out_features=out_features,
        device=device,
        adapter=adapter, # TODO FIX ME..need to pass Eora if loaded
    )
    if err is not None:
        raise err

    new_layer = linear_cls(
        bits=constructor_bits,
        group_size=tmp_group_size,
        desc_act=tmp_desc_act,
        sym=tmp_sym,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=tmp_pack_dtype,
        bias=bias,
        dtype=dtype,
        #weight_dtype=submodule.qweight.dtype if isinstance(submodule, BaseQuantLinear) else submodule.weight.dtype,
        name=name,
        lm_head_name=lm_head_name,
        backend=backend,
        register_buffers=register_buffers,
        adapter=adapter,
        bias_dtype=bias_dtype,
        **tmp_init_kwargs,
    )
    new_layer.device = ori_layer_device
    if transfer_from is not None:
        # The caller already loaded the checkpoint tensors into the module this
        # one replaces (a load started before the native bridge was installed,
        # #4795). The replacement was constructed from the same serialized
        # config, so every persistent tensor keeps the checkpoint's shape and
        # only has to be copied over before the device move.
        source_state = transfer_from.state_dict()
        missing = sorted(set(new_layer.state_dict()) - set(source_state))
        if missing:
            sample = ", ".join(missing[:8])
            raise ValueError(
                f"Cannot rebuild `{name}` from its loaded tensors: the checkpoint "
                f"did not assign {sample}. Load this checkpoint with `GPTQModel.load`."
            )
        new_layer.load_state_dict(source_state, strict=False)
    recurse_setattr(module, name, new_layer.to(ori_layer_device))

def create_quant_layer(
        linear_candidates: List[Type[BaseQuantLinear]],
        bits,
        desc_act: bool,
        dynamic,
        group_size: int,
        quant_result: Dict[str, Dict[str, Any]],
        module,
        sym: bool,
        device: DEVICE,
        lm_head_name: str,
        pack_dtype: torch.dtype,
        backend: BACKEND,
        adapter: Optional[Adapter] = None,
        format: FORMAT = FORMAT.GPTQ,
        init_kwargs: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        bias_dtype: Optional[torch.dtype] = None,
        transfer_from: Optional[Dict[str, nn.Module]] = None,

) -> Type[BaseQuantLinear]:
    if any(isinstance(module, candidate) for candidate in linear_candidates):
        return type(module)

    selected_counts = dict.fromkeys(linear_candidates, 0)
    selected_counts[TorchQuantEmbeddings] = 0
    for name, submodule in module.named_modules():
        # skip non-quantized modules
        if name not in quant_result:
            continue

        candidates = [TorchQuantEmbeddings] if isinstance(submodule, nn.Embedding) else linear_candidates
        last_error = None
        for qlinear_cls in candidates:
            try:
                create_quant_module(
                    name=name,
                    linear_cls=qlinear_cls,
                    bits=bits,
                    desc_act=desc_act,
                    dynamic=dynamic,
                    group_size=group_size,
                    module=module,
                    submodule=submodule,
                    sym=sym,
                    device=device,
                    lm_head_name=lm_head_name,
                    pack_dtype=pack_dtype,
                    format=format,
                    backend=backend,
                    adapter=adapter,
                    init_kwargs=init_kwargs,
                    dtype=dtype,
                    bias_dtype=bias_dtype,
                    transfer_from=(
                        transfer_from.get(name) if transfer_from is not None else None
                    ),
                )
            except NotImplementedError as error:
                last_error = error
                if backend not in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
                    raise
                continue

            selected_counts[qlinear_cls] = selected_counts.get(qlinear_cls, 0) + 1
            break
        else:
            if last_error is not None:
                raise last_error
            raise ValueError(f"No compatible quant linear was found for module `{name}`.")

    selected_counts = {candidate: count for candidate, count in selected_counts.items() if count}
    if not selected_counts:
        raise ValueError(f"No compatible quant linear was found for this module: {module.__class__.__name__}")

    summary = ", ".join(f"{candidate.__name__}={count}" for candidate, count in selected_counts.items())
    log.info(f"Kernel: per-module selections -> `[{summary}]`")
    non_embedding_counts = {
        candidate: count
        for candidate, count in selected_counts.items()
        if candidate is not TorchQuantEmbeddings
    }
    return max(non_embedding_counts or selected_counts, key=(non_embedding_counts or selected_counts).get)

# public/stable api exposed to transformer/optimum
def hf_convert_gptq_v1_to_v2_format(
    model: nn.Module,
    bits: int,
    qlinear_kernel: Type[BaseQuantLinear],
    checkpoint_format: str,
    meta: Optional[Dict[str, any]],
) -> Tuple[nn.Module, bool]:
    try:
        source_format = resolve_quant_format(checkpoint_format, METHOD.GPTQ)
    except (TypeError, ValueError):
        source_format = None
    if source_format in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
        # Optimum calls this before `save_pretrained()` can run. Record the
        # on-disk layout here so the state-dict hook can undo the runtime v2
        # conversion for a v1 checkpoint.
        for module in model.modules():
            if isinstance(module, GPTQQuantLinear):
                module._checkpoint_format_on_disk = source_format

    if source_format == FORMAT.GPTQ:
        # Skip v1 to v2 conversion when no loaded quant module requires v2.
        if not any(
            isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
            for m in model.modules()
        ):
            return model, False

        cfg = QuantizeConfig(bits=bits)
        return convert_gptq_v1_to_v2_format(model, cfg, qlinear_kernel), True
    return model, False

def convert_gptq_v1_to_v2_format_module(module: BaseQuantLinear, bits: int, pack_dtype: torch.dtype) -> nn.Module:
    assert isinstance(module, BaseQuantLinear)

    log.info.once("Format: Converting GPTQ v1 to v2")

    # v1 stores a per-field (zero_point - 1): a packed-word `+= 1` carries
    # across fields when a stored value is maxed (zp=0 -> stored maxq), so the
    # correction must decode, shift in logical space, and repack.
    if bits in (2, 4, 8) or (bits == 3 and pack_dtype == torch.int32):
        module.qzeros.data.copy_(
            _correct_gptq_v1_qzeros(module.qzeros.data, bits, planar=getattr(module, "planar", False))
        )
    elif bits == 3:
        # Only int32 packing words are used for GPTQ INT3 in actual checkpoints.
        # Keep the legacy constant-offset path for synthetic smaller word sizes.
        # range 0 offset
        if pack_dtype == torch.int64:
            offset = 0b0010010010010010010010010010010000100100100100100100100100100100
        elif pack_dtype == torch.int16:
            offset = 0b0010010010010010
        elif pack_dtype == torch.int8:
            offset = 0b00100100

        module.qzeros.data[:, range(0, module.qzeros.data.shape[1], 3)] += (
            offset
        )

        # range 1 offset
        if pack_dtype == torch.int64:
            offset = 0b1001001001001001001001001001001010010010010010010010010010010010
        elif pack_dtype == torch.int16:
            offset = 0b1001001001001001
        elif pack_dtype == torch.int8:
            offset = 0b10010010

        module.qzeros.data[:, range(1, module.qzeros.data.shape[1], 3)] += (
            offset
        )

        # range 2 offset
        if pack_dtype == torch.int64:
            offset = 0b0100100100100100100100100100100101001001001001001001001001001001
        elif pack_dtype == torch.int16:
            offset = 0b0100100100100100
        elif pack_dtype == torch.int8:
            offset = 0b01001001

        module.qzeros.data[:, range(2, module.qzeros.data.shape[1], 3)] += (
            offset
        )
    elif bits in (5, 6, 7):
        if pack_dtype != torch.int32:
            raise NotImplementedError(
                f"Planar {bits}-bit GPTQ only supports 32-bit packing words, got pack_dtype={pack_dtype}."
            )
        # Planar layouts spread each zero-point across bit planes, so shift the
        # decoded logical values instead of adding a packed-word constant.
        module.qzeros.data.copy_(_correct_gptq_v1_qzeros(module.qzeros.data, bits, planar=True))
    else:
        raise NotImplementedError("Only 2,3,4,5,6,7,8 bits are supported.")

    # change format id
    module.qzero_format(format=2)

# Optionally convert weight from gptq_v1 to v2 format if Kernel is compatible with v2
@torch.inference_mode()
def convert_gptq_v1_to_v2_format(
    model,
    cfg: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):
    # skip v2 to v1 conversion when no loaded quant module requires v2
    if cfg.export_quant_method() == METHOD.GPTQ and not any(
        isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
        for m in model.modules()
    ):
        log.info(
            f"Format: Skipped v1 to v2 conversion; no selected kernel requires v2 (`{qlinear_kernel}`).")
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    # with tctl.threadpool_limits(limits=1):
    time.time()
    log.info(
        f"Format: Converting `{FORMAT_FIELD_CODE}` from `{FORMAT.GPTQ}` to internal `{FORMAT.GPTQ_V2}`.")

    for _, submodule in model.named_modules():
        # v1 checkpoint format used to do `qzeros = qzeros -= 1` before serialization, thus the
        # additions here do not overflow.
        # v1 checkpoint format with sym=False saved via convert_gptq_v2_to_v1_format() will
        # overflow ~<=13% based on testing
        if isinstance(submodule, BaseQuantLinear) and getattr(submodule, "REQUIRES_FORMAT_V2", False):
            convert_gptq_v1_to_v2_format_module(
                module=submodule,
                bits=getattr(submodule, "bits", cfg.bits),
                pack_dtype=getattr(submodule, "pack_dtype", cfg.pack_dtype),
            )

        #log.info(f"Format: Conversion complete: {time.time() - t}s")

    return model

# public/stable api exposed to transformer/optimum
def hf_convert_gptq_v2_to_v1_format(
    model: nn.Module,
    sym: bool,
    bits: int,
    qlinear_kernel: Type[BaseQuantLinear],
    checkpoint_format: str,
    meta: Optional[Dict[str, any]],
) -> Tuple[nn.Module, bool]:
    # note: sym=False is valid for gptq_v2 for all gptqmodel and gptq(v1) for gptqmodel >= `0.9.0`
    if sym and checkpoint_format == "gptq_v2":
        quantize_config = QuantizeConfig(bits=bits)
        return convert_gptq_v2_to_v1_format(model, quantize_config, qlinear_kernel), True
    else:
        return model, False

def convert_gptq_v2_to_v1_format_module(
    module: BaseQuantLinear,
    quantize_config: QuantizeConfig,
):
    assert isinstance(module, BaseQuantLinear)

    log.info.once("Format: Converting GPTQ v2 to v1")

    bits = getattr(module, "bits", quantize_config.bits)
    pack_dtype = getattr(module, "pack_dtype", quantize_config.pack_dtype)
    # v1 stores a per-field (zero_point - 1): a packed-word `-= 1` borrows
    # across fields when a logical zero point is 0, so the correction must
    # decode, shift in logical space, and repack.
    if bits in (2, 4, 8) or (bits == 3 and pack_dtype == torch.int32):
        module.qzeros.data.copy_(
            _revert_gptq_v1_qzeros_correction(module.qzeros.data, bits, planar=getattr(module, "planar", False))
        )
    elif bits == 3:
        module.qzeros.data[:, range(0, module.qzeros.data.shape[1], 3)] -= (
            0b00100100100100100100100100100100
        )
        module.qzeros.data[:, range(1, module.qzeros.data.shape[1], 3)] -= (
            0b10010010010010010010010010010010
        )
        module.qzeros.data[:, range(2, module.qzeros.data.shape[1], 3)] -= (
            0b01001001001001001001001001001001
        )
    elif bits in (5, 6, 7):
        if pack_dtype != torch.int32:
            raise NotImplementedError(
                f"Planar {bits}-bit GPTQ only supports 32-bit packing words, got pack_dtype={pack_dtype}."
            )
        module.qzeros.data.copy_(
            _revert_gptq_v1_qzeros_correction(module.qzeros.data, bits, planar=True)
        )
    else:
        raise NotImplementedError("Only 2,3,4,5,6,7,8 bits are supported.")

    module.qzero_format(format=1)

# Optionally convert weight from gptq_v2 to v1 export format if Kernel is compatible with v2
@torch.inference_mode()
def convert_gptq_v2_to_v1_format(
    model,
    quantize_config: QuantizeConfig,
    qlinear_kernel: Type[BaseQuantLinear],
):

    # skip v2 to v1 conversion when no loaded quant module requires v2
    if quantize_config.export_quant_method() == METHOD.GPTQ and not any(
        isinstance(m, BaseQuantLinear) and getattr(m, "REQUIRES_FORMAT_V2", False)
        for m in model.modules()
    ):
        return model

    # Limit thread usage to avoid auto-parallizataion regression
    # with tctl.threadpool_limits(limits=1):
    for _, submodule in model.named_modules():
        # sym=False has underflow probability of ~<=13% during testing. No underflow possible for sym=True.
        if isinstance(submodule, BaseQuantLinear) and getattr(submodule, "REQUIRES_FORMAT_V2", False):
            convert_gptq_v2_to_v1_format_module(module=submodule, quantize_config=quantize_config)

    return model


_RESAVE_CONVERTIBLE_GPTQ_FORMATS = (FORMAT.GPTQ, FORMAT.GPTQ_V2)
_RESAVE_LAYOUT_FIELDS = ("bits", "group_size", "sym", "desc_act", "pack_dtype")


def resolve_resave_format_conversion(quantize_config: QuantizeConfig) -> Optional[Tuple[FORMAT, FORMAT]]:
    """Return ``(on_disk_format, target_format)`` when a re-save converts qzeros.

    ``None`` means the checkpoint layout is unchanged.  A layout change that
    cannot be represented by the v1/v2 qzero shift is rejected here, before a
    writer emits a mislabelled checkpoint.
    """

    source_format = getattr(quantize_config, "_checkpoint_format_on_disk", None)
    if source_format is None:
        return None
    target_format = resolve_quant_format(quantize_config.format, quantize_config.method)
    if source_format == target_format:
        return None

    snapshot = getattr(quantize_config, "_checkpoint_layout_snapshot", None) or {}
    changed = [
        field
        for field in _RESAVE_LAYOUT_FIELDS
        if field in snapshot and getattr(quantize_config, field, None) != snapshot[field]
    ]
    if changed:
        raise ValueError(
            "Cannot change "
            + ", ".join(changed)
            + " while re-saving a loaded checkpoint: the packed weights would no longer "
            "match the config. Re-quantize from the dense source instead."
        )
    if (
        source_format in _RESAVE_CONVERTIBLE_GPTQ_FORMATS
        and target_format in _RESAVE_CONVERTIBLE_GPTQ_FORMATS
    ):
        return source_format, target_format
    raise ValueError(
        "Cannot re-save a loaded checkpoint with a different quantization format "
        f"(`{source_format}` -> `{target_format}`). Only GPTQ v1 <-> GPTQ v2 qzero "
        "conversion is exact; every other layout change requires re-quantization "
        "from the dense source checkpoint."
    )


def convert_loaded_qzeros_for_format_change(
    model: nn.Module,
    source_format: FORMAT,
    target_format: FORMAT,
) -> int:
    """Rewrite loaded GPTQ qzeros from ``source_format`` to ``target_format``.

    The source tensors are read raw from the checkpoint, so this is the only
    place a format change can be applied before the writer copies them.
    """

    if source_format == target_format:
        return 0
    if (
        source_format not in _RESAVE_CONVERTIBLE_GPTQ_FORMATS
        or target_format not in _RESAVE_CONVERTIBLE_GPTQ_FORMATS
    ):
        raise ValueError(
            f"Unsupported qzero format conversion: `{source_format}` -> `{target_format}`."
        )

    converted = 0
    for module in model.modules():
        if not isinstance(module, GPTQQuantLinear):
            continue
        qzeros = getattr(module, "qzeros", None)
        if qzeros is None or qzeros.numel() == 0:
            continue
        bits = getattr(module, "bits", 4)
        planar = bool(getattr(module, "planar", False))
        if target_format == FORMAT.GPTQ_V2:
            qzeros.data.copy_(_correct_gptq_v1_qzeros(qzeros.data, bits, planar=planar))
            module.qzero_format(format=2)
        else:
            qzeros.data.copy_(_revert_gptq_v1_qzeros_correction(qzeros.data, bits, planar=planar))
            module.qzero_format(format=1)
        converted += 1
    return converted


def resolve_pack_device(
    quantize_config: Optional[QuantizeConfig],
    device: Optional[Union[str, torch.device, DEVICE]] = None,
) -> Optional[torch.device]:
    """Resolve the accelerator a module should be packed on.

    ``QuantizeConfig.device`` is normalized to an index-less ``DEVICE`` enum by
    the loader, so on a multi-GPU host it would silently pin GPU packing to
    ``cuda:0`` even when the module was quantized on another GPU. Callers that
    know the assigned device pass it here; a CPU assignment keeps the historical
    auto-select behavior because ``pack_impl="auto"`` still prefers GPU packing
    when one is available.
    """

    cfg_device = getattr(quantize_config, "device", None)
    if device is not None:
        assigned = device.to_torch_device() if isinstance(device, DEVICE) else torch.device(device)
        if assigned.type != "cpu":
            cfg_device = assigned

    if isinstance(cfg_device, DEVICE):
        return cfg_device.to_torch_device()
    if isinstance(cfg_device, torch.device):
        return cfg_device
    if isinstance(cfg_device, str):
        try:
            return torch.device(cfg_device)
        except (RuntimeError, ValueError):
            log.warning(f"pack_module: unable to parse target device `{cfg_device}`; defaulting to CUDA auto-select.")
            return None
    return None


@torch.inference_mode()
def pack_module(
    name,
    qModules,
    q_scales,
    q_zeros,
    q_g_idx,
    layers,
    quant_linear_cls,
    lock: threading.Lock,
    q_scales_extra=None,
    quantize_config: Optional[QuantizeConfig] = None,
    quant_result: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    # Limit pack() thread usage to avoid auto-parallizataion regression
    # with ctx(tctl.threadpool_limits(limits=1), lock):
    layer = layers[name]
    module = qModules[name]

    assert get_device(module) == CPU
    assert get_device(layer) == CPU
    assert get_device(q_scales) == CPU
    assert get_device(q_zeros) == CPU

    # module = module.to(CPU)
    # layer = layer.to(CPU)
    # q_scales = q_scales.to(CPU)
    # q_zeros = q_zeros.to(CPU)

    if q_g_idx is not None:
        assert get_device(q_g_idx) == CPU
        #q_g_idx = q_g_idx.to(CPU)

    pack_impl = "original"
    target_device = None
    if quantize_config is not None:
        pack_impl = getattr(quantize_config, "pack_impl", "original") or "original"
        target_device = resolve_pack_device(quantize_config, device)

    packer_label = None

    if lock is not None:
        with lock:
            layers[name] = layer
            qModules[name] = module
    else:
        layers[name] = layer
        qModules[name] = module

    # Use the module's actual class when it is a real BaseQuantLinear; otherwise
    # fall back to the caller-supplied representative class (e.g. unit-test mocks).
    module_cls = type(module) if isinstance(module, BaseQuantLinear) else quant_linear_cls

    # TODO FIX ME..remove hard coded qqq pack
    if module_cls.QUANT_TYPE == "qqq":
        if q_scales_extra is not None:
            q_scales_extra = q_scales_extra.to(CPU)
        packer_label = "module.pack"
        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            module.pack(linear=layer, scales=q_scales, s_extra=q_scales_extra)
    elif module_cls.QUANT_TYPE.startswith("awq_") or module_cls.QUANT_TYPE == "llm-awq":
        packer_label = "module.pack"
        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            module.pack(
                linear=layer,
                scales=q_scales,
                zeros=q_zeros,
                g_idx=q_g_idx,
            )
    else:
        effective_impl = (pack_impl or "original").lower()

        if effective_impl in {"cpu", "block", "pack_block"}:
            effective_impl = "block"
        elif effective_impl in {"original", "pack_original"}:
            effective_impl = "original"
        elif effective_impl in {"auto", "default"}:
            effective_impl = _auto_pack_impl(module=module, linear=layer)
        elif effective_impl == "gpu":
            if not HAS_CUDA:
                log.warning("pack_module: GPU packing requested but CUDA is unavailable; falling back to original pack.")
                effective_impl = "original"
            elif not hasattr(module, "pack_gpu"):
                log.warning("pack_module: GPU packing requested but module lacks pack_gpu; falling back to original pack.")
                effective_impl = "original"
        elif effective_impl != "original":
            log.warning(
                "pack_module: Unknown pack_impl `%s`; defaulting to original pack.",
                pack_impl,
            )
            effective_impl = "original"

        label_map = {
            "gpu": "module.pack_gpu",
            "block": "module.pack_block",
            "original": "module.pack_original",
        }

        packer_label = label_map[effective_impl]

        with log_time_block(
            packer_label,
            logger=log,
            module_name=name,
        ):
            if effective_impl == "gpu":
                try:
                    module.pack_gpu(
                        linear=layer,
                        scales=q_scales,
                        zeros=q_zeros,
                        g_idx=q_g_idx,
                        device=target_device,
                    )
                except (ValueError, NotImplementedError):
                    module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)
            elif effective_impl == "block":
                try:
                    module.pack_block(
                        linear=layer,
                        scales=q_scales,
                        zeros=q_zeros,
                        g_idx=q_g_idx,
                    )
                except ValueError:
                    module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)
            else:
                module.pack_original(linear=layer, scales=q_scales, zeros=q_zeros, g_idx=q_g_idx)

        target_format = (
            resolve_quant_format(quantize_config.format, quantize_config.method)
            if quantize_config is not None
            else None
        )
        if (
            quantize_config is not None
            and quantize_config.export_quant_method() == METHOD.GPTQ
            and target_format in (FORMAT.GPTQ, FORMAT.GPTQ_V2)
        ):
            with log_time_block(
                "mark_checkpoint_qzero_format",
                logger=log,
                module_name=name,
            ):
                # Packing writes logical (v2) zero points, which is also what a
                # ``REQUIRES_FORMAT_V2`` kernel decodes. Convert only for
                # serialization: leaving the live buffer in the v1 ``zero - 1``
                # layout made every in-process forward read each zero point one
                # step low (issue #4063). ``checkpoint_layout_qzeros`` and the
                # writer emit the converted copy from this marker.
                module.qzero_format(format=2)
                if getattr(module_cls, "REQUIRES_FORMAT_V2", False):
                    module._checkpoint_format_on_disk = target_format

        # TODO: why move it back to gpu?
        # start = time.time()
        # qModules[name].to(layer_device)
        # log.info(f"Pack: moving module back to `{layer_device}` cost = {time.time()-start} seconds")

    return packer_label

def pack_model(
    model,
    quant_result: Dict[str, Dict[str, Any]],
    bits,
    group_size,
    backend: BACKEND,
    format: str | FORMAT,
    quant_method: str | METHOD,
    lm_head_name: str,
    desc_act=False,
    sym: bool = True,
    dynamic=None,
    pack_dtype: torch.dtype = None,
):
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        format=format,
        quant_method=quant_method,
        desc_act=desc_act,
        sym=sym,
        dynamic=dynamic,
        pack_dtype=pack_dtype,
    )

    model.to(CPU)

    log.info("Packing model...")

    modules = find_modules(model)

    modules = {n: modules[n] for n in quant_result}
    quant_linear_cls = make_quant(
        model,
        qcfg=qcfg,
        quant_result=quant_result,
        backend=backend,
        lm_head_name=lm_head_name,
        pack=True,
        device=DEVICE.CPU,
    )

    qModules = {
        name: quant_module
        for name, quant_module in find_modules(model, [BaseQuantLinear]).items()
        if name in quant_result
    }

    assert len(qModules) > 0, "No quantized modules found in the model."

    names = list(qModules.keys())
    lock = threading.Lock()

    if has_gil_disabled():
        from device_smi import Device
        cpu = Device("cpu")
        default_max_packers = min(8, max(2, cpu.count * cpu.cores))
    else:
        default_max_packers = 1 # due to gil, there is no point packing with more than 1 thread

    configured_max_packers = resolve_kernel_config(qcfg).max_packers
    max_packers = (
        configured_max_packers
        if configured_max_packers is not None
        else default_max_packers
    )

    with ctx(ThreadPoolExecutor(max_workers=max_packers), log.pb(names).manual()) as (executor, pb):
        def wrapper(name):
            # TODO FIX, thread pool executor does not advance iterator
            pb.next()
            pb.title(f"Packing {name}").draw()
            pack_module(
                name=name,
                qModules=qModules,
                quant_result=quant_result,
                layers=modules,
                quant_linear_cls=quant_linear_cls,
                lock=lock,
                quantize_config=qcfg,
            )

        for _ in executor.map(wrapper, names):
            pass

    log.info("Model packed.")
    return quant_linear_cls


def no_placement_module_names(model: nn.Module) -> set[str]:
    """Resolve Transformers no-placement parameter patterns to leaf modules."""

    patterns = getattr(model, "_no_placement_params", ()) or ()
    patterns = tuple(pattern for pattern in patterns if isinstance(pattern, str) and pattern)
    if not patterns:
        return set()

    names = set()
    tensors = (*model.named_parameters(), *model.named_buffers())
    for tensor_name, _ in tensors:
        if any(tensor_name == pattern or tensor_name.endswith(f".{pattern}") for pattern in patterns):
            names.add(tensor_name.rsplit(".", 1)[0])
    return names


def _remove_redundant_device_map_children(device_map: Dict[str, Union[str, int]]) -> None:
    """Remove child entries already covered by a same-device parent."""

    for name in sorted(device_map, key=lambda item: item.count(".")):
        if name not in device_map:
            continue
        prefix = f"{name}." if name else ""
        for child_name in list(device_map):
            if child_name != name and child_name.startswith(prefix) and device_map[child_name] == device_map[name]:
                device_map.pop(child_name)


def _split_device_map_around_module(
    model: nn.Module,
    device_map: Dict[str, Union[str, int]],
    ancestor_name: str,
    target_name: str,
    ancestor_device: Union[str, int],
) -> None:
    """Replace one parent mapping with non-overlapping branches around a target."""

    current_name = ancestor_name
    while current_name != target_name:
        current_module = model if not current_name else model.get_submodule(current_name)
        relative_target = target_name if not current_name else target_name[len(current_name) + 1:]
        path_child = relative_target.split(".", 1)[0]

        # Preserve direct tensors and sibling branches on the parent's device;
        # descend only through the branch containing the excluded module.
        for param_name, _ in (*current_module.named_parameters(recurse=False), *current_module.named_buffers(recurse=False)):
            full_name = f"{current_name}.{param_name}" if current_name else param_name
            device_map.setdefault(full_name, ancestor_device)
        for child_name, _ in current_module.named_children():
            full_name = f"{current_name}.{child_name}" if current_name else child_name
            if child_name != path_child:
                device_map.setdefault(full_name, ancestor_device)

        current_name = f"{current_name}.{path_child}" if current_name else path_child


def co_locate_shared_module_aliases(model: nn.Module, device_map: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
    """Keep shared tensor owners on their first registered owner's placement.

    This completes a generated layer-wise map; explicit user maps are not
    rewritten. No tensors move here, and device families/ordinals stay intact.
    """
    result = dict(device_map)
    modules = list(model.named_modules(remove_duplicate=False))
    parents = {id(module): id(module) for _, module in modules}

    def root(identity):
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    tensor_owners = {}
    for _, module in modules:
        for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
            previous = tensor_owners.setdefault(id(tensor), id(module))
            parents[root(id(module))] = root(previous)

    groups = {}
    for name, module in modules:
        # Container aliases are covered by their actual tensor-owning children.
        if not any(True for _ in module.parameters(recurse=False)) and not any(True for _ in module.buffers(recurse=False)):
            continue
        groups.setdefault(root(id(module)), []).append(name)

    def placement(name):
        candidates = [key for key in result if not key or key == name or name.startswith(f'{key}.')]
        if not candidates:
            return None, None
        key = max(candidates, key=len)
        return key, result[key]

    changed = False
    for names in groups.values():
        if len(names) < 2:
            continue
        selected = next((device for name in names if (device := placement(name)[1]) is not None), None)
        if selected is None:
            continue
        for name in names:
            ancestor, previous = placement(name)
            if previous == selected:
                continue
            if ancestor is not None and ancestor != name:
                _split_device_map_around_module(model, result, ancestor, name, previous)
                result.pop(ancestor, None)
            result[name] = selected
            changed = True
            log.info(f'Loader: shared owner alias `{name}` follows `{names[0]}` on {selected}.')
    if changed:
        _remove_redundant_device_map_children(result)
    return result


def apply_no_placement_to_device_map(model: nn.Module, device_map: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
    """Build a non-overlapping load map with excluded leaf modules on CPU."""

    result = dict(device_map)
    _remove_redundant_device_map_children(result)
    for module_name in no_placement_module_names(model):
        ancestors = [
            name
            for name in result
            if name != module_name and (not name or module_name.startswith(f"{name}."))
        ]
        for ancestor_name in sorted(ancestors, key=lambda item: item.count("."), reverse=True):
            ancestor_device = result.pop(ancestor_name)
            _split_device_map_around_module(
                model,
                result,
                ancestor_name,
                module_name,
                ancestor_device,
            )
        # The checkpoint preloader expands parent and child entries independently;
        # keep this map non-overlapping so the same tensor is not read on both devices.
        result[module_name] = "cpu"
    _remove_redundant_device_map_children(result)
    return result


def _single_resident_device(model: nn.Module, device_map: Dict[str, Union[str, int]]) -> Optional[torch.device]:
    """Return the map's one device when every tensor is already resident there.

    ``add_hook_to_module`` rebinds ``module.forward`` to ``partial(new_forward, module)``
    and keeps the previous bound method on ``module._old_forward``.  Both point back
    at the module, so a hooked model is a reference cycle: a GPU model keeps its VRAM
    allocated after the last user reference is dropped, until a full ``gc.collect()``
    (#4847).  A layer-wise map whose entries all name the same device does not need
    the hooks at all -- accelerate's own ``dispatch_model`` only attaches them when
    the map spans more than one device or offloads to disk -- so skip them whenever
    nothing is left to materialise or offload.
    """

    devices = {d for d in device_map.values() if d is not None}
    if len(devices) != 1:
        return None
    raw_device = next(iter(devices))
    if isinstance(raw_device, str) and raw_device in ("cpu", "mps", "disk"):
        return None
    try:
        device = torch.device(raw_device)
    except (TypeError, RuntimeError):
        return None
    for parameter in model.parameters():
        if parameter.device.type != device.type:
            return None
        if device.index is not None and parameter.device.index != device.index:
            return None
    # Buffers may legitimately stay on the CPU (a scalar format flag, a rotary
    # ``inv_freq`` cache), but a meta buffer still has to be materialised.
    if any(buffer.device.type == "meta" for buffer in model.buffers()):
        return None
    return device


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    device_map = dict(device_map)
    single_root = "" in device_map and len(device_map) == 1
    all_single_cpu_or_mps = all(
        d in ("cpu", "mps") for d in device_map.values()
    )
    # CPU offload is unnecessary for all-CPU/MPS device maps and must be skipped.
    if single_root or all_single_cpu_or_mps:
        d = next(iter(device_map.values()))
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    # One device and nothing to materialise: the per-entry hooks would only pin the
    # model in a reference cycle, so place nothing and keep the tensors where the
    # checkpoint loader already put them.
    if _single_resident_device(model, device_map) is not None:
        model.hf_device_map = device_map
        return model

    # Avoid eagerly materialising the full model on a single device when the map
    # provides finer-grained placements; rely on accelerate hooks instead.
    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    # These modules perform their own CPU lookup and must remain resident;
    # a normal CPU-offload hook would move their full weights back to the GPU.
    resident_cpu_modules = no_placement_module_names(model)
    module_names = dict(model.named_modules())
    cpu_offload_group = [
        (n, d)
        for n, d in device_map.items()
        if d == "cpu" and n not in resident_cpu_modules and n in module_names
    ]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        if n == "":
            continue
        m = module_names.get(n)
        if m is None:
            # Fine-grained maps can contain direct parameter entries.
            continue
        if d != "cpu":
            d = torch.device(d)
            has_other_device_child = any(
                child_name.startswith(f"{n}.") and child_device != device_map[n]
                for child_name, child_device in device_map.items()
            )
            # A mixed-device child means the parent hook may move inputs, but not descendants.
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=not has_other_device_child)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)

    model.hf_device_map = device_map

    return model


# public/stable api exposed to transformer/optimum
def hf_gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None,
                        max_input_length: Optional[int] = None):
    # Optimum calls this after ``convert_model``. Retrying here keeps the
    # package-init probe from caching a partially initialized ``optimum.gptq``
    # as "Optimum not installed" (#4749).
    from gptqmodel import _ensure_optimum_gptqmodel_load_bridge

    _ensure_optimum_gptqmodel_load_bridge()
    return gptqmodel_post_init(model, use_act_order, quantize_config, max_input_length)


# public/stable api exposed to transformers/optimum: native load bridge.
# Re-exported here so callers can keep importing from `gptqmodel.utils.model`.
# These names are intentionally not referenced in this module: they exist only to
# keep the `gptqmodel.utils.model` entry point stable for transformers/optimum,
# so the unused-import exemption is deliberate rather than dead code.
from .hf_load import (  # noqa: E402,F401
    HFGPTQModelLoadContext,
    hf_gptqmodel_post_init_for_load,
    hf_gptqmodel_prepare_model_for_load,
)


def _post_init_cpu_offloaded_marlin(module: MarlinLinear) -> None:
    """Refresh the writable CPU snapshot after Marlin changes its packed layout."""
    # Resident weights have no Accelerate snapshot to refresh.
    if not module.qweight.is_meta:
        module.post_init()
        return

    from accelerate.hooks import AlignDevicesHook, SequentialHook
    from accelerate.utils import PrefixedDataset

    # An offload hook may be wrapped by other hooks, but it must own this layer.
    attached = getattr(module, "_hf_hook", None)
    hooks = attached.hooks if isinstance(attached, SequentialHook) else (attached,)
    offload_hooks = [
        hook for hook in hooks if isinstance(hook, AlignDevicesHook) and hook.offload
    ]
    if len(offload_hooks) != 1 or offload_hooks[0].place_submodules:
        raise RuntimeError(
            "CPU-offloaded Marlin requires a layer-level Accelerate offload hook."
        )
    hook = offload_hooks[0]
    # CPU offload can keep layer entries in a shared dictionary under a prefix.
    weights_map = hook.weights_map
    if isinstance(weights_map, PrefixedDataset) and isinstance(
        weights_map.dataset, dict
    ):
        storage, prefix = weights_map.dataset, weights_map.prefix
    elif isinstance(weights_map, dict):
        storage, prefix = weights_map, ""
    else:
        raise RuntimeError(
            "CPU-offloaded Marlin requires a writable in-memory weights map."
        )

    # Save the original offloaded names before post_init changes tensor layouts.
    names = [
        name
        for name, tensor in module.named_parameters(recurse=False)
        if tensor.is_meta
    ]
    names.extend(
        name
        for name, tensor in module.named_buffers(recurse=False)
        if tensor.is_meta and name not in module._non_persistent_buffers_set
    )
    missing = [name for name in names if f"{prefix}{name}" not in storage]
    if missing:
        raise RuntimeError(
            f"CPU-offloaded Marlin is missing weights map entries: {missing}."
        )

    # Materialize the saved tensors, repack them, then return the layer to meta.
    updated = {}
    hook.pre_forward(module)
    try:
        module.post_init()
        for name in names:
            tensor = getattr(module, name, None)
            if tensor is None or tensor.is_meta:
                raise RuntimeError(
                    f"Marlin did not materialize offloaded tensor `{name}`."
                )
            updated[f"{prefix}{name}"] = tensor.detach().to("cpu").clone()
    finally:
        hook.post_forward(module, None)
    # Publish the packed snapshot only after post_forward has offloaded the layer.
    storage.update(updated)


def _dense_adapter_forward_hook(adapter: Adapter):
    """Return a forward hook that adds the adapter update to a dense module."""

    def hook(module, args, output):
        if not args or not isinstance(output, torch.Tensor):
            return output
        return adapter.apply(x=args[0], out=output)

    return hook


def apply_dense_module_adapters(model, adapter: Optional[Adapter]) -> int:
    """Attach a LoRA adapter to targeted modules that stayed dense.

    Quantized modules carry their adapter inside the kernel, but a module the
    checkpoint does not quantize (an unquantized ``lm_head``, a dynamic
    exclusion) stayed a plain ``nn.Linear`` and its adapter weights were
    silently dropped (issue #4210). Attach one forward hook per targeted dense
    module so the adapter contribution is added to the base output.

    Returns the number of modules that received an adapter.
    """

    if adapter is None:
        return 0
    if not getattr(adapter, "path", None) or adapter.path in LORA_MERGED_WEIGHT_PATHS:
        # Merged/EoRA adapters have no separate weights to apply here.
        return 0

    applied = 0
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear):
            continue
        if not isinstance(module, (nn.Linear, transformers.Conv1D)):
            continue
        if getattr(module, "_gptqmodel_dense_adapter", None) is not None:
            continue
        if not adapter.targets_module(name):
            continue
        if not adapter.has_weights(name):
            log.warn(
                f"Adapter: no LoRA tensors for targeted dense module `{name}`; skipping."
            )
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.device.type == "meta":
            continue
        dense_adapter = copy.deepcopy(adapter)
        dense_adapter.post_init(weight_key=name, device=weight.device)
        module._gptqmodel_dense_adapter = dense_adapter
        module.register_forward_hook(_dense_adapter_forward_hook(dense_adapter))
        applied += 1

    if applied:
        log.info(f"Adapter: applied to `{applied}` dense module(s).")
    return applied


def gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None,
                        max_input_length: Optional[int] = None,
                        awq_batch_size: Optional[int] = None):
    """
    Initialize model-persistent backend scratch buffers after quantized weights are loaded.

    ``awq_batch_size`` sizes the AWQ ExLlamaV2 scratch. It is an explicit
    argument rather than the ``AWQ_BATCH_SIZE`` environment variable, which a
    concurrent ``quantize()`` used to overwrite for every later load in the
    process (issue #4129).
    """
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for name, submodule in model.named_modules():
        if isinstance(submodule, ExllamaV2Linear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))
        elif isinstance(submodule, AwqExllamaV2Linear):
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed(
                max_input_len=max_input_length or 2048,
                max_batch_size=awq_batch_size if awq_batch_size is not None else 1,
            )
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        from ..utils.exllamav2 import ScratchSpace

        # we allocate a model-persistent scratch space for each device
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ScratchSpace(scratch_bytes=scratch_bytes, dev=device)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

    # The continuous and planar (gptq_p) 3-bit word layouts are not
    # interchangeable, and a 3-bit module is only planar when its constructor
    # received `format=gptq_p`. Fail loudly if any construction site dropped
    # the format instead of silently decoding the wrong layout.
    if quantize_config is not None:
        expect_planar3 = resolve_quant_format(quantize_config.format, quantize_config.method) == FORMAT.GPTQ_P
        for name, submodule in model.named_modules():
            if (
                isinstance(submodule, GPTQQuantLinear)
                and submodule.bits == 3
                and submodule.planar != expect_planar3
            ):
                raise ValueError(
                    f"`{name}`: 3-bit quant module was constructed with planar={submodule.planar} "
                    f"but the checkpoint format is `{quantize_config.format}`; the continuous and "
                    f"planar 3-bit layouts are not interchangeable. Pass `format=` when "
                    "constructing the module."
                )

    # The buffers need to have been initialized first before calling make_q4.
    for _, submodule in model.named_modules():
        if isinstance(submodule, (ExllamaV2Linear, AwqExllamaV2Linear)):
            device = submodule.qweight.device
            submodule.post_init(scratch_space=model.device_tensors[device])
        elif isinstance(submodule, MarlinLinear):
            _post_init_cpu_offloaded_marlin(submodule)
        elif isinstance(submodule, BaseQuantLinear):
            submodule.post_init()

    if quantize_config is not None and quantize_config.activation_mode in {"w4afp8", "w4a_nvfp4"}:
        from ..nn_modules.qlinear.w4a_llama_stream import install_w4a_llama_stream

        install_w4a_llama_stream(
            model, quantize_config.activation_mode, quantize_config.activation_recipe,
            global_scales=quantize_config.activation_global_scales,
            attention_mode=quantize_config.activation_attention_mode,
            attention_recipe=quantize_config.activation_attention_recipe,
            mlp_fp8_layers=quantize_config.activation_mlp_fp8_layers,
            fused_norms=bool(resolve_model_rotation(quantize_config)),
        )

    if quantize_config is not None:
        apply_dense_module_adapters(model, getattr(quantize_config, "adapter", None))

    torch_empty_cache()

    return model


def get_checkpoints(model_id_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs):
    """
    Retrives (and if necessary downloads from Hugging Face Hub) the model checkpoint. Sharding is supported. All the `possible_model_basenames` (e.g. `["model", "model-4bit-gptq"]`) will be explored over all `extensions` (e.g. `[".bin", ".safetensors"]`).
    """
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None

    if os.path.isdir(model_id_or_path):
        # A downloaded snapshot is a local directory, so the Hub branch below
        # never sees `subfolder`. Apply it here or a nested checkpoint would be
        # searched at the repository root.
        model_id_or_path = resolve_checkpoint_subfolder(
            model_id_or_path, cached_file_kwargs.get("subfolder")
        )
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_id_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    # The model is sharded over several checkpoints.
                    possible_model_basename = possible_index_file.replace(ext + ".index.json", "")
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_id_or_path, possible_model_basename)
                    searched_files.append(possible_model_basename + ext)
                    if os.path.isfile(model_save_name + ext):
                        resolved_archive_file = model_save_name + ext
                        return False, resolved_archive_file, possible_model_basename
    else:
        temp = None
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                shard_index = cached_file(
                    model_id_or_path,
                    shard_index_name,
                    **cached_file_kwargs,
                )
                searched_files.append(shard_index_name)
                if shard_index is not None:
                    # The model is sharded over several checkpoints.
                    with open(str(shard_index)) as f:
                        index_json = json.load(f)
                        # Download the shards from the index.json.
                        shards = list(set(index_json["weight_map"].values()))
                        for shard in shards:
                            resolved_archive_file = cached_file(
                                model_id_or_path,
                                shard,
                                **cached_file_kwargs,
                            )
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(
                        model_id_or_path,
                        possible_model_basename + ext,
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is None:
                        resolved_archive_file = temp
                    searched_files.append(possible_model_basename + ext)
                    if resolved_archive_file is not None:
                        temp = resolved_archive_file
                        return False, resolved_archive_file, possible_model_basename

    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find a model in {model_id_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
        )

    return False, resolved_archive_file, true_model_basename


# return the most stable tensor dtype for quantization while minimizing vram
def auto_dtype(config: PretrainedConfig,
               device: DEVICE,
               quant_inference: bool = False) -> torch.dtype:

    assert isinstance(device, DEVICE)

    # TODO: both MPS and XPU are locked to float16
    # XPU stack is missing bfloat16 (hardware supports it)
    # MPS stack has bfloat16 bugs in pytorch
    if device in [DEVICE.MPS, DEVICE.XPU]:
        log.info("Loader: Auto dtype (MPS or XPU): `torch.float16`")
        return torch.float16

    if device == DEVICE.XLA:
        # Tenstorrent PJRT currently rounds float16 transfers to bfloat16.
        # Select the native representation instead of silently losing bits.
        log.info("Loader: Auto dtype (XLA/Tenstorrent): `torch.bfloat16`")
        return torch.bfloat16

    # TODO: need to verify this
    # Torch 2.8 fused kernel for CPU is optimized for bfloat16
    if device in [DEVICE.CPU]:
        log.info("Loader: Auto dtype (CPU + Torch Fused): `torch.bfloat16`")
        return torch.bfloat16

    # Update: latest kernel accuracies have shown, with multiple ranges of shapes
    # There are no accuracy issues with bf16 vs fp16. The Marlin reduce-accumulation
    # path can use fp16, but it is disabled by default via
    # KernelConfig.marlin_fp32=True.
    # # for inference, always use FP16 for max accuracy
    # # check test_kernel_outputs for validation between fp16 and b16 in terms of kernel accuracy
    # if quant_inference:
    #     log.info("Loader: Auto dtype: `torch.float16` due to inference mode. If you wish to use `bfloat16`, please pass in `dtype` arg to `loader()`.")
    #     return torch.float16

    # get dtype from config
    dtype = get_hf_config_dtype(config)
    if dtype and not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype in config must be a torch.dtype, but got {dtype}")

    if dtype in [torch.float32, torch.float64]:
        log.info("Loader: Auto dtype (float32 down-cast): `torch.bfloat16`")
        return torch.bfloat16
    elif dtype == torch.float16:
        log.info("Loader: Auto dtype (native float16): `torch.float16`")
        return torch.float16
    elif dtype == torch.bfloat16:
        log.info("Loader: Auto dtype (native bfloat16): `torch.bfloat16`")
        return torch.bfloat16
    else:
        # TODO: extract weights from model file to check their original type, instead of forcing bfloat16
        # up/down-cast everything else to bfloat16 if not already in bfloat16
        log.info(f"Loader: Auto dtype (native = `{dtype}`): `torch.bfloat16`")
        return torch.bfloat16


# generate layer modules for moe models with experts
def get_moe_layer_modules(layer_modules: List, num_experts: int) -> List:
    new_inside_layer_modules = []
    for names in layer_modules:
        new_inside_layer_modules.append([])
        for n in names:
            if EXPERT_INDEX_PLACEHOLDER in n:
                for index in range(num_experts):
                    new_inside_layer_modules[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
            else:
                new_inside_layer_modules[-1].append(n)

    return new_inside_layer_modules


# Marker separating the routed-expert container from the expert index, e.g.
# `model.layers.0.mlp.experts.3.down_proj`. Shared experts (`shared_experts`)
# are not part of the fused group and do not contain this marker.
FUSED_EXPERT_GROUP_MARKER = ".experts."

# Dense projections that inference engines merge into one module: fused module
# suffix -> the GPT-QModel module suffixes it is built from.
FUSED_PROJECTION_UNITS = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
}

# `dynamic` fields that inference engines read when they resolve the single
# config of a fused module. Every other field only affects GPT-QModel's
# quantization step and is baked into the saved weights, so it may legitimately
# differ between the modules of one fused module.
FUSED_UNIT_ENGINE_FIELDS = ("bits", "group_size", "desc_act", "sym")


def fused_expert_group_name(module_name: str) -> Optional[str]:
    """Return the fused routed-expert group that `module_name` belongs to.

    Inference engines (vLLM, SGLang) fuse every routed expert of one layer into
    a single module (`<prefix>.experts`) with one precision and one
    quantization config, so `<prefix>.experts.<expert_id>.<projection>` modules
    must be quantized as a unit. Returns `None` for modules outside such a
    group, including shared experts and already fused expert containers.
    """

    if not module_name:
        return None

    marker_index = module_name.find(FUSED_EXPERT_GROUP_MARKER)
    if marker_index <= 0:
        return None

    expert_id, separator, _ = module_name[marker_index + len(FUSED_EXPERT_GROUP_MARKER):].partition(".")
    if not expert_id or not separator:
        return None

    return module_name[:marker_index + len(FUSED_EXPERT_GROUP_MARKER) - 1]


def fused_expert_group_members(module_names) -> Dict[str, List[str]]:
    """Group `module_names` by the fused routed-expert group they belong to."""

    groups: Dict[str, List[str]] = {}
    for module_name in module_names:
        group = fused_expert_group_name(module_name)
        if group is not None:
            groups.setdefault(group, []).append(module_name)
    return groups


def fused_unit_members(module_names) -> Dict[str, List[str]]:
    """Map every engine-fused module to the GPT-QModel modules it merges.

    Inference engines (vLLM, SGLang) merge the routed experts of one decoder
    layer into one module (`<prefix>.experts`) and the dense `gate_proj`/`up_proj`
    and `q_proj`/`k_proj`/`v_proj` shards into `gate_up_proj`/`qkv_proj`. They
    resolve one `dynamic` override per merged module from the *fused* name, so
    the modules that are merged must resolve the same engine-visible config.
    Modules without a fused partner are not reported.
    """

    names = set(module_names)
    units: Dict[str, List[str]] = {}
    for group, members in fused_expert_group_members(module_names).items():
        units[group] = list(members)

    for module_name in module_names:
        if fused_expert_group_name(module_name) is not None:
            # Routed experts are covered by their fused group above.
            continue
        prefix, separator, suffix = module_name.rpartition(".")
        if not separator:
            continue
        for fused_suffix, shard_suffixes in FUSED_PROJECTION_UNITS.items():
            if suffix not in shard_suffixes:
                continue
            members = [
                f"{prefix}.{shard_suffix}"
                for shard_suffix in shard_suffixes
                if f"{prefix}.{shard_suffix}" in names
            ]
            if len(members) > 1:
                units.setdefault(f"{prefix}.{fused_suffix}", members)
    return units


def fused_unit_engine_config(override, qcfg) -> tuple:
    """Return the config an inference engine resolves for one fused module.

    Engines read `bits`, `group_size`, `desc_act` and `sym` from the `dynamic`
    override and honor `-:` exclusions; the remaining fields only affect
    GPT-QModel's quantization step and are baked into the saved weights.
    """

    if override is False:
        return ("excluded",)
    if not isinstance(override, dict):
        override = {}
    return tuple(
        (field, override.get(field, getattr(qcfg, field, None)))
        for field in FUSED_UNIT_ENGINE_FIELDS
    )


@dataclass(frozen=True)
class FusedUnitDynamicConflict:
    """One fused module whose `dynamic` config the inference engines cannot honor."""

    unit: str
    engine_config: tuple
    member_configs: Tuple[Tuple[str, tuple], ...]

    def suggested_pattern(self) -> str:
        """Return a `dynamic` pattern that covers the fused module and its shards.

        The engines match the fused module name while GPT-QModel matches the
        unfused shards, so a rule must cover both. Routed experts live below
        their container, so one prefix pattern is enough; dense shards are
        siblings of the fused module and need an alternation.
        """

        nested = all(member.startswith(f"{self.unit}.") for member, _ in self.member_configs)
        if nested:
            return rf"{pcre.escape(self.unit)}(\.|$)"

        prefix, separator, fused_suffix = self.unit.rpartition(".")
        shards = [member.rsplit(".", 1)[-1] for member, _ in self.member_configs]
        alternation = "|".join(pcre.escape(name) for name in [*shards, fused_suffix])
        if not separator:
            return f"({alternation})$"
        return rf"{pcre.escape(prefix)}\.({alternation})$"


def fused_unit_dynamic_conflicts(module_names, qcfg) -> List[FusedUnitDynamicConflict]:
    """Report fused modules whose `dynamic` config the engines cannot honor.

    Inference engines resolve one `dynamic` override per fused module from the
    fused module name only, so a rule that matches just part of a fused module
    makes the engine disagree with the checkpoint GPT-QModel saves: the engine
    loads float shards into a quantized module (or quantized shards into a
    float module) and the model fails to load (issue #1455).
    """

    if qcfg is None or not getattr(qcfg, "dynamic", None):
        return []

    conflicts = []
    for unit, members in sorted(fused_unit_members(module_names).items()):
        engine_config = fused_unit_engine_config(qcfg.dynamic_get(unit, default=None), qcfg)
        member_configs = tuple(
            (member, fused_unit_engine_config(qcfg.dynamic_get(member, default=None), qcfg))
            for member in members
        )
        if all(config == engine_config for _, config in member_configs):
            continue
        conflicts.append(
            FusedUnitDynamicConflict(
                unit=unit,
                engine_config=engine_config,
                member_configs=member_configs,
            )
        )
    return conflicts


def _blamed_dynamic_patterns(dynamic, conflicts) -> List[str]:
    """Return the `dynamic` rules responsible for the fused-module conflicts."""

    blamed = set()
    for conflict in conflicts:
        member_names = [member for member, _ in conflict.member_configs]
        unit_pattern = dynamic_matching_pattern(dynamic, conflict.unit)
        for member, config in conflict.member_configs:
            if config == conflict.engine_config:
                continue
            pattern = dynamic_matching_pattern(dynamic, member)
            if pattern is not None:
                blamed.add(pattern)
        if unit_pattern is not None:
            # A rule on the fused name only matters when it does not cover every
            # shard of the fused module.
            raw = unit_pattern[2:] if unit_pattern.startswith(("-:", "+:")) else unit_pattern
            compiled = pcre.compile(raw)
            if not all(compiled.match(member) for member in member_names):
                blamed.add(unit_pattern)

    order = {pattern: index for index, pattern in enumerate(dynamic.keys())}
    return sorted(blamed, key=lambda pattern: order.get(pattern, len(order)))


def warn_fused_unit_dynamic_conflicts(
    module_names,
    qcfg,
    *,
    max_rules: int = 3,
) -> List[FusedUnitDynamicConflict]:
    """Warn when `dynamic` gives fused modules a config the engines cannot honor.

    Returns the detected conflicts so callers and tests can inspect them.
    """

    conflicts = fused_unit_dynamic_conflicts(module_names, qcfg)
    if not conflicts:
        return conflicts

    dynamic = qcfg.dynamic
    blamed = _blamed_dynamic_patterns(dynamic, conflicts)
    if blamed:
        rules = ", ".join(f"`{pattern}`" for pattern in blamed[:max_rules])
        if len(blamed) > max_rules:
            rules += f" (+{len(blamed) - max_rules} more)"
    else:
        rules = "a per-module `dynamic` rule"

    example = conflicts[0]

    log.warn(
        f"Dynamic: {rules} may prevent the saved checkpoint from loading in vLLM/SGLang: "
        f"fused modules such as `{example.unit}` resolve a different config than their shards, and those "
        "engines read `dynamic` from the fused name only. Match the fused name too, e.g. "
        f"`+:{example.suggested_pattern()}` or `-:` to skip it."
    )
    return conflicts


def fused_unit_check_module_names(model, layers_node) -> List[str]:
    """Return the module names to check for fused-module `dynamic` conflicts.

    Only the decoder layers are quantized, so modules outside them (vision
    towers, projectors, ...) cannot make the saved checkpoint disagree with the
    fused inference engines. Module names are read without materializing or
    type-filtering them, so the check also works on loaded quantized models.
    """

    names = [name for name, _ in model.named_modules() if name]
    _, layer_names = get_layers_with_prefixes(model, layers_node)
    prefixes = tuple(f"{layer_name}." for layer_name in layer_names if layer_name)
    if not prefixes:
        return names
    return [name for name in names if name.startswith(prefixes)]


def check_to_quantized(config):
    if isinstance(config, dict):
        if config["bits"] > 8 or "fp" in config["data_type"] or "float" in config["data_type"]:
            return False
        return True
    else:
        if config.bits > 8 or "fp" in config.data_type or "float" in config.data_type:
            return False
        return True


def copy_py_files(save_dir, file_extension=".py", model_id_or_path=""):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.isdir(model_id_or_path):
        py_files = [f for f in os.listdir(model_id_or_path) if f.endswith('.py')]
        for file in py_files:
            source_path = os.path.join(model_id_or_path, file)
            # `copy2` follows symlinks; a cloned repository must not smuggle a
            # file from outside its own tree into an output directory.
            assert_auxiliary_copy_allowed(model_id_or_path, source_path, kind="custom Python file")
            shutil.copy2(source_path, save_dir)
    else:
        remote_model_info = model_info(model_id_or_path)
        for file in remote_model_info.siblings:
            if file.rfilename.endswith(file_extension):
                _ = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename=file.rfilename,
                    local_dir=save_dir,
                )


def get_model_files_size(pre_quantized_model_path, file_extension=['.bin', '.safetensors', '.pth', '.pt', '.ckpt', '.h5', '.pb', '.onnx']):
    if os.path.isdir(pre_quantized_model_path):
        pre_quantized_size_bytes = sum(
            os.path.getsize(os.path.join(pre_quantized_model_path, f))
            for f in os.listdir(pre_quantized_model_path)
            if os.path.isfile(os.path.join(pre_quantized_model_path, f)) and os.path.splitext(f)[
                1] in file_extension
        )
    else:
        remote_model_info = model_info(pre_quantized_model_path, files_metadata=True)
        pre_quantized_size_bytes = sum(
            (file_data.size or 0)
            for file_data in remote_model_info.siblings
            if any(file_data.rfilename.endswith(ext) for ext in file_extension)
        )
    pre_quantized_size_mb = pre_quantized_size_bytes / (1024 * 1024)
    return pre_quantized_size_mb

def check_requires_version(requires_version, current_version):
    OPERATOR_MAP = {
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "<": operator.lt,
        ">": operator.gt,
    }
    match = _REQUIRES_VERSION_RE.match(requires_version)
    if match:
        op_symbol, required_version = match.groups()
        current_version = version.parse(current_version)
        required_version = version.parse(required_version)
        return OPERATOR_MAP[op_symbol](current_version, required_version)
    else:
        return None


class MODALITY(str, Enum):
    TEXT = "text"
    IMAGE_TO_TEXT = "image_to_text"
    VIDEO_TO_TEXT = "video_to_text"
    # TEXT_TO_IMAGE = "text_to_image"


def _split_parameter_path(full_name: str) -> Tuple[str, str]:
    if "." in full_name:
        module_path, leaf = full_name.rsplit(".", 1)
    else:
        module_path, leaf = "", full_name
    return module_path, leaf


def _generate_offload_search_paths(offload_root, module_path):
    """
    Generate fallback paths:
    'a.b.c' -> ['gptqmodel_offload/a.b.c', 'gptqmodel_offload/a.b', 'gptqmodel_offload/a', 'gptqmodel_offload/']
    """
    if not module_path:
        return [offload_root]

    parts = module_path.split(".")
    paths = []

    for i in range(len(parts), 0, -1):
        sub = ".".join(parts[:i])
        paths.append(os.path.join(offload_root, sub))

    paths.append(os.path.join(offload_root, ""))

    return paths


def _generate_entry_keys(
    module_path: str, leaf: str, index_dir: str, offload_root: str
) -> tuple[str, ...]:
    """Use the selected index's scope, then allow a fully qualified key."""
    full_name = f"{module_path}.{leaf}" if module_path else leaf
    index_scope = os.path.relpath(index_dir, offload_root)
    if index_scope == ".":
        return (full_name,)
    index_scope = index_scope.replace(os.sep, ".")
    if module_path != index_scope and not module_path.startswith(f"{index_scope}."):
        raise ValueError(f"Offload index scope '{index_scope}' does not contain '{full_name}'")
    relative_name = full_name[len(index_scope) + 1:]
    return (relative_name, full_name) if relative_name != full_name else (relative_name,)


def _resolve_offload_entry(
    offload_root: str,
    module_path: str,
    leaf: str,
    dtype: torch.dtype,
    shape_hint: Tuple[int, ...],
    index_cache: Dict[str, Optional[Dict]],
) -> Optional[OffloadTensorRef]:
    if not offload_root:
        return None

    search_paths = _generate_offload_search_paths(offload_root, module_path)
    index = None
    for module_dir in search_paths:
        index = index_cache.get(module_dir)
        if index is not None:
            break

        index_path = os.path.join(module_dir, "index.json")
        if not os.path.isfile(index_path):
            index_cache[module_dir] = None
            continue

        # load index.json
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)

        index_cache[module_dir] = index
        break

    if not index:
        return None

    keys = _generate_entry_keys(module_path, leaf, module_dir, offload_root)
    matches = []
    for key in keys:
        entry = index.get(key)
        if entry is not None:
            matches.append((key, entry))
    if not matches:
        return None
    if len(matches) > 1 and matches[0][1] != matches[1][1]:
        name = f"{module_path}.{leaf}" if module_path else leaf
        raise ValueError(
            f"Ambiguous offload index entries for tensor '{name}' in "
            f"'{os.path.join(module_dir, 'index.json')}': "
            f"'{matches[0][0]}' and '{matches[1][0]}' differ."
        )
    entry = matches[0][1]

    resolved_dtype = _dtype_string_to_torch(entry.get("dtype"), dtype)
    if "shape" in entry:
        shape = tuple(entry["shape"])
    else:
        shape = shape_hint

    if resolved_dtype != dtype or shape != shape_hint:
        name = f"{module_path}.{leaf}" if module_path else leaf
        index_path = os.path.join(module_dir, "index.json")
        raise ValueError(
            f"Offload metadata mismatch for tensor '{name}' in '{index_path}': "
            f"expected dtype {dtype} and shape {shape_hint}, "
            f"found dtype {resolved_dtype} and shape {shape}."
        )

    safetensors_file = entry.get("safetensors_file")
    if safetensors_file:
        path = safetensors_file
        if not os.path.isabs(path):
            path = os.path.join(module_dir, path)
        offsets = entry.get("data_offsets")
        if offsets is not None:
            offsets = tuple(int(x) for x in offsets)
        return OffloadTensorRef(
            path=os.path.abspath(path),
            torch_dtype=resolved_dtype,
            shape=shape,
            format="safetensors",
            weight_name=entry.get("weight_name", leaf),
            data_offsets=offsets,
            sha256=entry.get("sha256"),
        )

    filename = entry.get("filename")
    if filename:
        path = filename if os.path.isabs(filename) else os.path.join(module_dir, filename)
        start = int(entry.get("offset", 0))
        end = start + (_torch_dtype_num_bytes(resolved_dtype) * math.prod(shape or (1,)))
        return OffloadTensorRef(
            path=os.path.abspath(path),
            torch_dtype=resolved_dtype,
            shape=shape,
            format="dat",
            weight_name=None,
            data_offsets=(start, end),
        )

    data_path = os.path.join(module_dir, f"{leaf}.dat")
    if not os.path.isfile(data_path):
        return None

    return OffloadTensorRef(
        path=os.path.abspath(data_path),
        torch_dtype=resolved_dtype,
        shape=shape,
        format="dat",
        weight_name=None,
        data_offsets=None,
    )


def _collect_state_dict_with_offload(
    model: nn.Module,
    offload_root: str,
    include_prefixes: Optional[Tuple[str, ...]] = None,
    include_names: Optional[set[str]] = None,
) -> Dict[str, TensorSource]:
    state_dict: Dict[str, TensorSource] = collections.OrderedDict()
    index_cache: Dict[str, Optional[Dict]] = {}

    for name, param in model.named_parameters():
        if include_prefixes is not None and not name.startswith(include_prefixes):
            continue
        if include_names is not None and name not in include_names:
            continue
        module_path, leaf = _split_parameter_path(name)
        source = None
        if getattr(param, "is_meta", False) or param.device.type == "meta":
            source = _resolve_offload_entry(
                offload_root,
                module_path,
                leaf,
                param.dtype,
                tuple(param.shape),
                index_cache,
            )
            if source is None:
                raise FileNotFoundError(
                    f"Offloaded tensor '{name}' not found in offload directory '{offload_root}'."
                )
        else:
            source = param
        state_dict[name] = TensorSource(name=name, torch_dtype=param.dtype, shape=tuple(param.shape), source=source)

    # Collect persistent buffers in a single module-tree walk: each module owns
    # its own buffers via `named_buffers(recurse=False)`, and non-persistent ones
    # are skipped inline against `_non_persistent_buffers_set`.
    for module_name, module in model.named_modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", ())
        for buffer_name, buf in module.named_buffers(recurse=False):
            if buffer_name in non_persistent:
                continue
            if buffer_name == QZEROS_FORMAT_BUFFER:
                # State-dict transport metadata, not a checkpoint payload: the
                # writer resolves the qzero convention from `config.json`, so
                # emitting a second, possibly stale, label here could only
                # disagree with the words it writes.
                continue
            name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if include_prefixes is not None and not name.startswith(include_prefixes):
                continue
            if include_names is not None and name not in include_names:
                continue
            if name in state_dict:
                continue
            module_path, leaf = _split_parameter_path(name)
            if getattr(buf, "is_meta", False) or buf.device.type == "meta":
                source = _resolve_offload_entry(
                    offload_root,
                    module_path,
                    leaf,
                    buf.dtype,
                    tuple(buf.shape),
                    index_cache,
                )
                if source is None:
                    raise FileNotFoundError(
                        f"Offloaded buffer '{name}' not found in offload directory '{offload_root}'."
                    )
            else:
                source = _checkpoint_layout_buffer(module, buffer_name, buf)
            state_dict[name] = TensorSource(name=name, torch_dtype=buf.dtype, shape=tuple(buf.shape), source=source)

    return state_dict


def _restore_checkpoint_layout_for_export(model: nn.Module) -> int:
    """Return in-place transformed CPU layers to their checkpoint layout.

    The snapshot is restored explicitly here; ``clear_weight_cache`` only drops
    derived metadata and must keep the live packed buffers authoritative for
    moves and explicit clears. Modules without a snapshot (never transformed,
    or non-CPU backends) are untouched.
    """

    from ..nn_modules.qlinear._fused_train_state import restore_fused_train_state

    restored = 0
    for module in model.modules():
        if getattr(module, "_fused_train_state", None) is None:
            continue
        if restore_fused_train_state(module):
            clear = getattr(module, "clear_weight_cache", None)
            if callable(clear):
                clear()
            restored += 1
    return restored


def _checkpoint_layout_buffer(
    module: nn.Module,
    buffer_name: str,
    buffer: torch.Tensor,
) -> torch.Tensor:
    """Return the checkpoint-layout payload for one live quantized buffer.

    A kernel that requires the v2 qzero layout keeps logical zero points in
    memory while the checkpoint stores the v1 ``zero - 1`` words. The module
    owns the conversion so every serialization surface (this writer and
    ``state_dict()``) emits the same bytes without mutating the live buffer.
    """

    if buffer_name != "qzeros":
        return buffer
    converter = getattr(module, "checkpoint_layout_qzeros", None)
    if not callable(converter):
        return buffer
    converted = converter(buffer)
    return buffer if converted is None else converted


def get_state_dict_for_save(
    model: nn.Module,
    offload_root: Optional[str] = None,
    include_prefixes: Optional[Tuple[str, ...]] = None,
    include_names: Optional[set[str]] = None,
) -> Dict[str, TensorSource]:
    """
    Filter weight-sharing tensors.
    Referenced from transformers.modeling_utils.PreTrainedModel.save_pretrained.

    See https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L2369
    """
    # CPU kernels rewrite ``qweight``/``qzeros``/``scales`` (and widen
    # ``bias``) into their int4pack runtime layout *in place* on the first
    # inference forward. Serializing that layout writes checkpoint-incompatible
    # shapes under the original keys, so every transformed module is first put
    # back to the layout its checkpoint snapshot describes. This mutates only
    # packed runtime views, never a dense weight; the next forward re-runs the
    # same transform.
    _restore_checkpoint_layout_for_export(model)
    if offload_root:
        state_dict = _collect_state_dict_with_offload(
            model,
            offload_root,
            include_prefixes=include_prefixes,
            include_names=include_names,
        )
    else:
        state_dict = collections.OrderedDict()
        for name, param in model.named_parameters():
            if include_prefixes is not None and not name.startswith(include_prefixes):
                continue
            if include_names is not None and name not in include_names:
                continue
            state_dict[name] = TensorSource(name=name, torch_dtype=param.dtype, shape=tuple(param.shape), source=param)
        # Collect persistent buffers in a single module-tree walk, skipping
        # non-persistent buffers inline via `_non_persistent_buffers_set`.
        for module_name, module in model.named_modules():
            non_persistent = getattr(module, "_non_persistent_buffers_set", ())
            for buffer_name, buf in module.named_buffers(recurse=False):
                if buffer_name in non_persistent:
                    continue
                if buffer_name == QZEROS_FORMAT_BUFFER:
                    # See `get_state_dict_for_save`.
                    continue
                name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                if include_prefixes is not None and not name.startswith(include_prefixes):
                    continue
                if include_names is not None and name not in include_names:
                    continue
                if name in state_dict:
                    continue
                state_dict[name] = TensorSource(
                    name=name,
                    torch_dtype=buf.dtype,
                    shape=tuple(buf.shape),
                    source=_checkpoint_layout_buffer(module, buffer_name, buf),
                )

    ptrs = collections.defaultdict(list)
    for name, entry in state_dict.items():
        source = entry.source
        if isinstance(source, OffloadTensorRef):
            key = (
                "offload",
                source.path,
                source.weight_name or name,
                source.data_offsets,
                source.sha256,
            )
        elif isinstance(source, torch.Tensor):
            tensor = source
            if getattr(tensor, "is_meta", False) or tensor.device.type == "meta":
                key = ("meta", id(tensor))
            else:
                try:
                    # Distinct checkpoint weights may be views into one fused
                    # allocation. Only identical views are duplicate keys.
                    key = ("storage", id_tensor_storage(tensor), tensor.dtype, tensor.storage_offset(),
                           tuple(tensor.shape), tuple(tensor.stride()))
                except Exception:
                    key = ("tensor", id(tensor))
        else:
            key = ("other", id(source))
        ptrs[key].append(name)

    # Only a pair the checkpoint really ties may be collapsed to one tensor.
    # `_tied_weights_keys` is derived from the model config alone, and a
    # composite wrapper can leave its top-level `tie_word_embeddings` at the
    # default `True` while the checkpoint stores a distinct `lm_head` (#4799).
    # The stamped snapshot records what the checkpoint really ties.
    snapshot = tied_weights_keys(model)
    tied_targets = set(snapshot) if snapshot is not None else None

    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    separated_names = set()
    for names in shared_ptrs.values():
        # Removing the keys which are declared as known duplicates on
        # load. This allows to make sure the name which is kept is consistent.
        if tied_targets is not None:
            found = 0
            for name in sorted(names):
                if name in tied_targets and name in state_dict:
                    found += 1
                    if found < len(names):
                        del state_dict[name]
        elif model._tied_weights_keys is not None:
            found = 0
            for name in sorted(names):
                matches_pattern = any(pcre.compile(pat).search(name) for pat in model._tied_weights_keys)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        del state_dict[name]

        # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
        # If the link between tensors was done at runtime then `from_pretrained` will not get
        # the key back leading to random tensor. A proper warning will be shown
        # during reload (if applicable), but since the file is not necessarily compatible with
        # the config, better show a proper warning.
        found = 0
        for name in names:
            if name in state_dict:
                found += 1
                if found > 1:
                    entry = state_dict[name]
                    if (
                        tied_targets is not None
                        and name not in tied_targets
                        and isinstance(entry.source, torch.Tensor)
                    ):
                        # The checkpoint does not tie this tensor, so it must
                        # not be dropped: give it its own storage and keep it.
                        state_dict[name] = replace(
                            entry, source=entry.source.clone()
                        )
                        separated_names.add(name)
                        continue
                    del state_dict[name]
                    warn_names.add(name)
    if len(warn_names) > 0:
        log.warn.once(
            f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
        )
    if len(separated_names) > 0:
        log.warn.once(
            f"Separated shared storage for {separated_names} while saving: the checkpoint does "
            "not tie these tensors, so both are written.",
        )
    return state_dict


def _checkpoint_tensor_keys(
    checkpoint: str | os.PathLike, *, verify_shards: bool = False,
) -> Optional[set[str]]:
    # accelerate.load_checkpoint_in_model() does not return the checkpoint key
    # set. Read only metadata/index keys here so tie_weights() can distinguish
    # tensors that were truly absent from tensors that were loaded separately.
    checkpoint = os.fspath(checkpoint)

    if os.path.isfile(checkpoint):
        if checkpoint.endswith(".json"):
            with open(checkpoint, encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", index)
            if isinstance(weight_map, dict):
                if verify_shards:
                    keys = set()
                    for shard in set(weight_map.values()):
                        shard_keys = _checkpoint_tensor_keys(
                            os.path.join(os.path.dirname(checkpoint), shard),
                        )
                        if shard_keys is None:
                            return None
                        keys.update(shard_keys)
                    return keys
                return set(weight_map)
            return None

        if checkpoint.endswith(".safetensors"):
            with safe_open(checkpoint, framework="pt", device="cpu") as handler:
                return set(handler.keys())

        return None

    if not os.path.isdir(checkpoint):
        return None

    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    if os.path.isfile(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as handler:
            return set(handler.keys())

    index_files = [name for name in os.listdir(checkpoint) if name.endswith(".index.json")]
    if len(index_files) != 1:
        return None

    return _checkpoint_tensor_keys(
        os.path.join(checkpoint, index_files[0]), verify_shards=verify_shards,
    )


def validate_checkpoint_qweights(
    model: nn.Module, checkpoint: str | os.PathLike, format: FORMAT, key_converter=None,
) -> None:
    # AWQ producers (AutoAWQ, llm-compressor) and shared Marlin/BitBLAS
    # checkpoints expose the same `qweight` key as GPTQ.  Skipping validation
    # for them let a module whose checkpoint tensor was never loaded keep its
    # zero-filled placeholder and produce silent NaN output.
    validated_formats = {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.GPTQ_P,
        FORMAT.GEMM,
        FORMAT.GEMV,
        FORMAT.GEMV_FAST,
        FORMAT.LLM_AWQ,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
    }
    if format not in validated_formats:
        return

    checkpoint_keys = _checkpoint_tensor_keys(checkpoint, verify_shards=True)
    if checkpoint_keys is None:
        return

    # A shared quantized module can be saved under any one of its aliases.
    module_keys = collections.defaultdict(set)
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, (GPTQQuantLinear, AWQuantLinear)):
            tensor_name = f"{name}.qweight" if name else "qweight"
            module_keys[id(module)].add(tensor_name)
            if callable(key_converter):
                module_keys[id(module)].add(key_converter(tensor_name))
    missing = sorted(
        min(keys) for keys in module_keys.values() if not keys & checkpoint_keys
    )
    if missing:
        raise ValueError(
            f"Missing required quantized weights in checkpoint {os.fspath(checkpoint)!r}: "
            + ", ".join(missing)
        )


def _matches_ignore_pattern(name: str, pattern) -> bool:
    try:
        return re.search(pattern, name) is not None
    except (TypeError, re.error):
        return False


def _unmaterialized_checkpoint_parameters(
    model: nn.Module, checkpoint: str | os.PathLike | None,
) -> List[str]:
    """Non-quantized parameters that no checkpoint tensor was loaded into.

    A quantized checkpoint only stores the non-quantized tensors as plain
    dense entries. When a key cannot be mapped back onto the runtime module the
    parameter keeps the ``torch.empty`` storage allocated while the initializers
    were suspended: random values, sometimes NaN/inf, with no warning. Compare
    the checkpoint key set against the live parameters so such a load fails
    loudly instead of returning a model that runs on garbage.
    """

    if checkpoint is None:
        return []
    checkpoint_keys = _checkpoint_tensor_keys(checkpoint, verify_shards=True)
    if checkpoint_keys is None:
        # Pickle checkpoints cannot be inspected without loading them; leave
        # that path to Accelerate's own reporting.
        return []

    loaded_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if name in checkpoint_keys
    }

    quantized_prefixes: List[str] = []
    for name, module in model.named_modules():
        if getattr(module, "QUANT_TYPE", None) is not None:
            quantized_prefixes.append(f"{name}." if name else "")

    ignored_patterns = getattr(model, "_keys_to_ignore_on_load_missing", None) or ()

    missing: List[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in checkpoint_keys:
            continue
        if parameter.is_meta:
            # Meta parameters belong to a lazy shell; its own checks cover them.
            continue
        if id(parameter) in loaded_parameter_ids:
            # Tied to a parameter the checkpoint did populate.
            continue
        if any(name.startswith(prefix) for prefix in quantized_prefixes):
            continue
        if any(_matches_ignore_pattern(name, pattern) for pattern in ignored_patterns):
            continue
        missing.append(name)
    return missing


def _raise_on_unmaterialized_checkpoint_parameters(
    model: nn.Module, checkpoint: str | os.PathLike | None,
) -> None:
    """Raise when a load leaves non-quantized parameters uninitialized."""

    missing = _unmaterialized_checkpoint_parameters(model, checkpoint)
    if not missing:
        return
    preview = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
    raise RuntimeError(
        "Loader: the checkpoint "
        f"{os.fspath(checkpoint)!r} does not provide tensors for {len(missing)} "
        f"non-quantized parameter(s): {preview}. They would keep the "
        "`torch.empty` storage allocated while weight initialization was "
        "suspended, so inference would run on random (sometimes NaN/inf) "
        "values. This usually means the checkpoint was written in a module "
        "layout the loader cannot map back onto the runtime model; save it in "
        "the source HF layout or load a checkpoint that matches this model."
    )


def _tie_weights_after_checkpoint_load(
    model,
    checkpoint: str | os.PathLike | None,
    *,
    extra_keys: set[str] | frozenset[str] | None = None,
) -> None:
    # Match transformers.from_pretrained(): when both sides of a tied-weight
    # pair are present in the checkpoint, tie_weights(missing_keys=...) checks
    # whether their loaded values are equal and skips tying if they differ.
    # This preserves checkpoints whose config incorrectly advertises tied
    # embeddings while storing a distinct lm_head.
    missing_keys = None
    if checkpoint is not None:
        checkpoint_keys = _checkpoint_tensor_keys(checkpoint)
        if checkpoint_keys is not None:
            if extra_keys:
                # Accelerate may have received a temporary checkpoint without
                # some tensors the loader restored itself; they were present
                # in the source checkpoint, so do not treat them as missing.
                checkpoint_keys |= set(extra_keys)
            missing_keys = set(model.state_dict().keys()) - checkpoint_keys

    if missing_keys is None:
        model.tie_weights()
        return

    try:
        model.tie_weights(missing_keys=missing_keys)
    except TypeError:
        model.tie_weights()


def _convert_safetensors_header_keys(
    header: dict, key_converter, exclude_names=frozenset(),
) -> Tuple[dict, bool, bool]:
    converted_header = {}
    changed = False
    excluded = False
    for name, tensor_info in header.items():
        if name == "__metadata__":
            converted_header[name] = tensor_info
            continue
        converted_name = key_converter(name) if callable(key_converter) else name
        if converted_name in exclude_names:
            # The caller owns these tensors (for example packed FP8 weights)
            # and loads them directly from the source checkpoint. Keeping
            # them out of this file also keeps accelerate from reading the
            # packed bytes into its per-shard staging buffer.
            changed = True
            excluded = True
            continue
        if converted_name in converted_header:
            raise ValueError(f"Checkpoint key conversion produced duplicate tensor name: {converted_name}")
        converted_header[converted_name] = tensor_info
        changed |= converted_name != name
    return converted_header, changed, excluded


def _write_safetensors_with_converted_header(
    source_path: str, target_path: str, key_converter, exclude_names=frozenset(),
) -> bool:
    """Copy a safetensors file, rewriting its header keys.

    ``exclude_names`` drops converted tensors from the copy. A safetensors
    data section must be contiguous, so an exclusion rewrites the remaining
    payload with fresh offsets; without one the data section is copied
    verbatim. Returns whether the file changed.
    """
    with open(source_path, "rb") as source:
        header_size_bytes = source.read(8)
        if len(header_size_bytes) != 8:
            return False
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = json.loads(source.read(header_size))
        data_offset = 8 + header_size

    converted_header, changed, excluded = _convert_safetensors_header_keys(
        header, key_converter, exclude_names,
    )
    if not changed:
        return False

    if not excluded:
        encoded_header = json.dumps(converted_header, separators=(",", ":")).encode("utf-8")
        encoded_header += b" " * ((-len(encoded_header)) % 8)
        with open(target_path, "wb") as target, open(source_path, "rb") as source:
            target.write(struct.pack("<Q", len(encoded_header)))
            target.write(encoded_header)
            source.seek(data_offset)
            shutil.copyfileobj(source, target, length=32 * 1024 * 1024)
        return True

    # Compact the kept tensors so their offsets stay contiguous from zero.
    kept = []
    offset = 0
    for name, tensor_info in converted_header.items():
        if name == "__metadata__":
            continue
        start, end = tensor_info["data_offsets"]
        kept.append((start, end))
        tensor_info["data_offsets"] = [offset, offset + (end - start)]
        offset += end - start
    encoded_header = json.dumps(converted_header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * ((-len(encoded_header)) % 8)
    buffer = bytearray(4 * 1024 * 1024)
    view = memoryview(buffer)
    with open(target_path, "wb") as target, open(source_path, "rb") as source:
        target.write(struct.pack("<Q", len(encoded_header)))
        target.write(encoded_header)
        for start, end in kept:
            source.seek(data_offset + start)
            remaining = end - start
            while remaining > 0:
                chunk = min(len(buffer), remaining)
                read = source.readinto(view[:chunk])
                if not read:
                    raise IOError(f"Unexpected EOF while copying {source_path}")
                target.write(view[:read])
                remaining -= read
    return True


def _remap_safetensors_index(
    checkpoint: str, key_converter, exclude_names=frozenset(),
) -> Optional[str]:
    """Rewrite a sharded safetensors index and its shard headers into a temporary directory."""
    with open(checkpoint, encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return None

    checkpoint_dir = os.path.dirname(checkpoint)
    shard_names = sorted(set(weight_map.values()))
    shard_paths = {name: os.path.join(checkpoint_dir, name) for name in shard_names}
    if not all(
        os.path.isfile(path) and path.endswith(".safetensors")
        for path in shard_paths.values()
    ):
        # A non-safetensors shard cannot be rewritten in place; leave the
        # checkpoint untouched so the caller keeps the previous behaviour.
        return None

    converted_weight_map = {}
    changed = False
    for name, shard in weight_map.items():
        converted_name = key_converter(name) if callable(key_converter) else name
        if converted_name in exclude_names:
            changed = True
            continue
        if converted_name in converted_weight_map:
            raise ValueError(f"Checkpoint key conversion produced duplicate tensor name: {converted_name}")
        converted_weight_map[converted_name] = shard
        changed |= converted_name != name
    if not changed:
        return None

    try:
        remap_dir = tempfile.mkdtemp(
            prefix="gptqmodel-key-remap-", dir=_remap_temp_dir(checkpoint)
        )
    except OSError:
        remap_dir = tempfile.mkdtemp(prefix="gptqmodel-key-remap-")
    try:
        target_names = {}
        used_shards = set(converted_weight_map.values())
        for shard_name in shard_names:
            if shard_name not in used_shards:
                # Every tensor in this shard was excluded. Accelerate reads
                # the shard set from the rewritten index, so skip the copy.
                continue
            # Keep relative subdirectories, but never let a malformed index
            # escape the temporary directory.
            normalized = os.path.normpath(shard_name)
            if os.path.isabs(normalized) or normalized.startswith(".."):
                normalized = os.path.basename(normalized)
            target_names[shard_name] = normalized
            target_path = os.path.join(remap_dir, normalized)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if not _write_safetensors_with_converted_header(
                shard_paths[shard_name], target_path, key_converter, exclude_names,
            ):
                shutil.copyfile(shard_paths[shard_name], target_path)

        converted_index = dict(index)
        converted_index["weight_map"] = {
            name: target_names[shard] for name, shard in converted_weight_map.items()
        }
        converted_path = os.path.join(remap_dir, os.path.basename(checkpoint))
        with open(converted_path, "w", encoding="utf-8") as handle:
            json.dump(converted_index, handle)
        return converted_path
    except BaseException:
        shutil.rmtree(remap_dir, ignore_errors=True)
        raise


def _cleanup_remapped_checkpoint(checkpoint: Optional[str]) -> None:
    if not checkpoint:
        return
    checkpoint = os.fspath(checkpoint)
    parent = os.path.dirname(checkpoint)
    if (
        os.path.basename(checkpoint).endswith(".index.json")
        and os.path.basename(parent).startswith("gptqmodel-key-remap-")
    ):
        shutil.rmtree(parent, ignore_errors=True)
        return
    try:
        os.unlink(checkpoint)
    except OSError:
        # Best-effort cleanup: do not mask the original exception if temp file removal fails.
        pass


# Call tied_weights() after load_checkpoint_in_model() to have the weights tied correctly.
def _remap_temp_dir(checkpoint: str) -> Optional[str]:
    """Keep large temporary checkpoint copies off a RAM-backed temp dir.

    ``tempfile`` defaults to ``/tmp``, which is commonly tmpfs. An FP8 subset
    copy can be many GiB, so prefer the checkpoint's own filesystem and fall
    back to the platform default when it is not writable.
    """

    parent = os.path.dirname(os.path.abspath(checkpoint))
    try:
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            return parent
    except OSError:
        pass
    return None


def _remap_safetensors_checkpoint_keys(
    checkpoint: str | os.PathLike, key_converter, exclude_names=frozenset(),
) -> Optional[str]:
    checkpoint = os.fspath(checkpoint)
    if not callable(key_converter) and not exclude_names:
        return None
    if os.path.isdir(checkpoint):
        if not exclude_names:
            return None
        # Mirror Accelerate's file resolution. A pickle checkpoint cannot be
        # rewritten here, so leave it to the previous load path.
        if os.path.isfile(os.path.join(checkpoint, "pytorch_model.bin")):
            return None
        safetensors_path = os.path.join(checkpoint, "model.safetensors")
        if os.path.isfile(safetensors_path):
            checkpoint = safetensors_path
        else:
            indexes = [
                name for name in os.listdir(checkpoint) if name.endswith(".index.json")
            ]
            if len(indexes) != 1:
                return None
            checkpoint = os.path.join(checkpoint, indexes[0])
    if not os.path.isfile(checkpoint):
        return None
    if checkpoint.endswith(".index.json"):
        return _remap_safetensors_index(checkpoint, key_converter, exclude_names)
    if not checkpoint.endswith(".safetensors"):
        return None

    try:
        descriptor, converted_path = tempfile.mkstemp(
            prefix="gptqmodel-key-remap-",
            suffix=".safetensors",
            dir=_remap_temp_dir(checkpoint),
        )
    except OSError:
        descriptor, converted_path = tempfile.mkstemp(
            prefix="gptqmodel-key-remap-", suffix=".safetensors"
        )
    os.close(descriptor)
    try:
        if not _write_safetensors_with_converted_header(
            checkpoint, converted_path, key_converter, exclude_names,
        ):
            os.unlink(converted_path)
            return None
    except BaseException:
        try:
            os.unlink(converted_path)
        except OSError:
            # Best-effort cleanup: do not mask the original exception if temp file removal fails.
            pass
        raise
    return converted_path


def _checkpoint_tensor_shapes(checkpoint) -> Dict[str, Tuple[int, ...]]:
    """Read tensor shapes from a safetensors file, directory, or index.

    NF12 stores a data-dependent escape table. The module cannot know its
    row count from `in_features`/`out_features` alone, so the loader must size
    that buffer before accelerate assigns checkpoint tensors.
    """

    if checkpoint is None:
        return {}
    path = os.fspath(checkpoint)
    if isinstance(path, bytes):
        path = path.decode()
    files: List[str] = []
    if os.path.isdir(path):
        files = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".safetensors")
        ]
    elif path.endswith(".index.json"):
        try:
            with open(path, encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except (OSError, ValueError):
            return {}
        base = os.path.dirname(path)
        files = sorted({os.path.join(base, name) for name in weight_map.values()})
    elif path.endswith(".safetensors"):
        files = [path]

    shapes: Dict[str, Tuple[int, ...]] = {}
    for file in files:
        try:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    shapes[key] = tuple(handle.get_slice(key).get_shape())
        except Exception:
            # A malformed or non-safetensors shard is accelerate's problem to
            # report; shape discovery is best effort.
            continue
    return shapes


def _checkpoint_tensor_dtypes(checkpoint) -> Dict[str, torch.dtype]:
    """Read each safetensors tensor's stored dtype without loading its data.

    A re-save must publish the dtype the checkpoint actually stores, not the
    dtype the export shell happened to allocate. Header reads are cheap and do
    not materialize any payload.
    """

    if checkpoint is None:
        return {}
    path = os.fspath(checkpoint)
    if isinstance(path, bytes):
        path = path.decode()
    files: List[str] = []
    if os.path.isdir(path):
        files = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".safetensors")
        ]
    elif path.endswith(".index.json"):
        try:
            with open(path, encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except (OSError, ValueError):
            return {}
        base = os.path.dirname(path)
        files = sorted({os.path.join(base, name) for name in weight_map.values()})
    elif path.endswith(".safetensors"):
        files = [path]

    safe_name_to_dtype = {
        safe_name: dtype for dtype, (safe_name, _size) in _DTYPE_SAFE_MAP.items()
    }
    dtypes: Dict[str, torch.dtype] = {}
    for file in files:
        try:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    dtype = safe_name_to_dtype.get(handle.get_slice(key).get_dtype())
                    if dtype is not None:
                        dtypes[key] = dtype
        except Exception:
            # A malformed or non-safetensors shard is accelerate's problem to
            # report; dtype discovery is best effort.
            continue
    return dtypes


def _prepare_variable_quant_buffers(model, checkpoint) -> None:
    """Resize data-dependent quant buffers to the checkpoint's stored shapes."""

    shapes = _checkpoint_tensor_shapes(checkpoint)
    if not shapes:
        return
    # Only data-dependent buffers are listed here. `nf12_packed`, `nf12_exp_base`,
    # and `bias` shapes follow from the module geometry. BitsAndBytes stores its
    # 4-bit QuantState as JSON bytes whose length depends on the serialized
    # `nested_offset` float, so the module's template-sized buffer can be a few
    # bytes short. Accelerate assigns these tensors directly (it does not call
    # `_load_from_state_dict`), so they must be sized before the load.
    nf12_suffixes = (".nf12_escapes", ".nf12_exp_base")
    for key, shape in shapes.items():
        if key.endswith(".weight_quant_state"):
            suffix = ".weight_quant_state"
            expected_quant_type = "bitsandbytes"
        else:
            suffix = next((item for item in nf12_suffixes if key.endswith(item)), None)
            expected_quant_type = "nf12"
        if suffix is None:
            continue
        module_name = key[: -len(suffix)]
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            continue
        if getattr(module, "QUANT_TYPE", None) != expected_quant_type:
            continue
        buffer_name = suffix[1:]
        current = getattr(module, buffer_name, None)
        if not isinstance(current, torch.Tensor) or tuple(current.shape) == shape:
            continue
        dtype = current.dtype
        module.register_buffer(buffer_name, torch.empty(shape, dtype=dtype, device=current.device))


def _checkpoint_runtime_to_source_keys(
    checkpoint, key_converter,
) -> Optional[dict[str, str]]:
    """Map runtime tensor names to the names stored in the source checkpoint.

    The loader may hand Accelerate a temporary checkpoint whose keys were
    converted. Buffers restored from the source checkpoint still need the
    original names, so read the header keys and apply the converter once.
    """

    if not callable(key_converter):
        return None
    keys = _checkpoint_tensor_keys(checkpoint, verify_shards=True)
    if keys is None:
        return None
    mapping: dict[str, str] = {}
    for key in keys:
        runtime = key_converter(key)
        existing = mapping.get(runtime)
        if existing is not None and existing != key:
            raise ValueError(
                "Checkpoint key conversion produced duplicate tensor name: "
                f"{runtime}"
            )
        mapping[runtime] = key
    return mapping


def _restore_checkpoint_buffers(
    model, checkpoint, dtypes: dict[str, torch.dtype | None],
    *, exact_source_names: set[str] | frozenset[str] = frozenset(),
    name_map: dict[str, str] | None = None,
    device_map: dict | None = None,
    offload_folder: str | os.PathLike | None = None,
) -> None:
    """Restore buffers and dense parameters after Accelerate casts.

    ``None`` keeps the source dtype. Dense parameters are restored by the
    export shell or an explicit native FP32 policy; other dense tensors keep
    the requested model dtype.

    ``name_map`` maps a runtime tensor name to the name stored in the
    checkpoint. It lets the restore read from the original checkpoint when
    Accelerate was given a temporary, key-converted copy.

    ``device_map`` places a tensor that Accelerate never saw (packed FP8
    weights removed from its checkpoint) on the device the map assigns.
    Intentionally disk-offloaded slots stay meta; their payload and offload
    index are updated together instead of materializing them in the model.
    """
    names = set(dtypes)
    offload_index = {}
    source_name = (lambda name: name_map.get(name, name)) if name_map else (lambda name: name)

    def mapped_device(name: str, *, include_disk: bool = False):
        if not isinstance(device_map, dict):
            return None
        module_name = name
        while module_name and module_name not in device_map:
            module_name = ".".join(module_name.split(".")[:-1])
        if module_name == "" and "" not in device_map:
            return None
        target = device_map[module_name]
        if target == "meta" or (target == "disk" and not include_disk):
            return None
        return target

    checkpoint = os.fspath(checkpoint)
    if os.path.isdir(checkpoint):
        # Match Accelerate's directory resolution so the second read comes
        # from the same checkpoint that populated the rest of the model.
        for filename in ("pytorch_model.bin", "model.safetensors"):
            candidate = os.path.join(checkpoint, filename)
            if os.path.isfile(candidate):
                checkpoint = candidate
                break
        else:
            indexes = [name for name in os.listdir(checkpoint) if name.endswith(".index.json")]
            if len(indexes) != 1:
                raise ValueError(f"Cannot resolve checkpoint for buffer restoration: {checkpoint}")
            checkpoint = os.path.join(checkpoint, indexes[0])

    if checkpoint.endswith(".index.json"):
        with open(checkpoint, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        files: dict[str, dict[str, str]] = {}
        for name in names:
            source = source_name(name)
            shard = weight_map.get(source)
            if shard is not None:
                files.setdefault(
                    os.path.join(os.path.dirname(checkpoint), shard), {}
                )[source] = name
    else:
        files = {checkpoint: {source_name(name): name for name in names}}

    restored: set[str] = set()

    def restore_one(name: str, value: torch.Tensor) -> None:
        module_name, _, tensor_name = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        current = module._buffers.get(tensor_name)
        parameter = module._parameters.get(tensor_name)
        if current is None and parameter is None:
            raise RuntimeError(f"Cannot restore checkpoint tensor {name!r}: missing model slot")
        slot = current if current is not None else parameter
        if not isinstance(slot, torch.Tensor) or tuple(slot.shape) != tuple(value.shape):
            raise RuntimeError(f"Cannot restore checkpoint tensor {name!r}: missing or mismatched model slot")
        disk_offloaded = slot.device.type == "meta" and mapped_device(name, include_disk=True) == "disk"
        if slot.device.type == "meta" and not (disk_offloaded and offload_folder is not None):
            raise RuntimeError(f"Cannot restore checkpoint tensor {name!r} on a meta device")
        if name in exact_source_names and value.dtype != dtypes[name]:
            raise RuntimeError(
                f"Checkpoint tensor {name!r} must have packed dtype {dtypes[name]}, got {value.dtype}"
            )
        target_dtype = dtypes[name]
        if disk_offloaded:
            from accelerate.utils import offload_weight, set_module_tensor_to_device

            restored_value = value.to(device="cpu", dtype=value.dtype if target_dtype is None else target_dtype)
            offload_weight(restored_value, name, offload_folder, index=offload_index)
            set_module_tensor_to_device(model, name, "meta", dtype=restored_value.dtype)
            restored.add(name)
            return
        target_device = slot.device
        if name in exact_source_names:
            target_device = mapped_device(name) or target_device
        restored_value = value.to(
            device=target_device, dtype=value.dtype if target_dtype is None else target_dtype
        )
        if parameter is not None:
            # Keep the Parameter object so tied weights stay tied.
            parameter.data = restored_value
        else:
            module._buffers[tensor_name] = restored_value
        restored.add(name)

    for path, selected in files.items():
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for source in sorted(set(selected) & available):
                    restore_one(selected[source], handle.get_tensor(source))
        else:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError as error:
                if "mmap" not in str(error):
                    raise
                state = torch.load(path, map_location="cpu", weights_only=True)
            for source in sorted(set(selected) & set(state)):
                restore_one(selected[source], state[source])

    missing = names - restored
    if missing:
        raise RuntimeError(
            "Cannot restore checkpoint tensors missing from checkpoint: "
            + ", ".join(sorted(missing))
        )
    if offload_index:
        from accelerate.utils import save_offload_index

        # Accelerate merges these entries with unrelated offloaded tensors.
        save_offload_index(offload_index, offload_folder)


def _load_narrowed(checkpoint_dtype: torch.dtype, loaded_dtype: torch.dtype) -> bool:
    """Whether loading ``checkpoint_dtype`` values as ``loaded_dtype`` can round them.

    A load that widens (for example ``keep_in_fp32_modules`` placing a BF16
    tensor in FP32) keeps every value; only a narrowing or cross-format cast
    needs the checkpoint tensor restored.
    """

    if checkpoint_dtype == loaded_dtype:
        return False
    try:
        return torch.promote_types(checkpoint_dtype, loaded_dtype) != loaded_dtype
    except RuntimeError:
        # PyTorch does not promote float8 types; treat the cast as lossy.
        return True


def _restore_fp32_checkpoint_buffers(model, checkpoint, names: set[str]) -> None:
    """Restore FP32 buffers narrowed by Accelerate's BF16 load path."""
    _restore_checkpoint_buffers(model, checkpoint, dict.fromkeys(names, torch.float32))


def _native_fp32_checkpoint_names(model, dtype) -> set[str]:
    """Resolve Transformers' dtype policy on registered, unquantized slots.

    Transformers uses regex search after replacing ``*`` with ``.*``. Reuse
    its compiler: segment equality, substring checks and escaped globs each
    change real native policies. Quantized modules retain their own packed
    storage/buffer ABI even when a broad native pattern covers their names.
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."), None)
    patterns = []
    if dtype == torch.float16:
        patterns.extend(getattr(model, "_keep_in_fp32_modules", None) or ())
    if dtype in (torch.float16, torch.bfloat16):
        patterns.extend(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    if not patterns:
        return set()
    try:
        from transformers.core_model_loading import build_glob_alternation
    except ImportError as error:
        raise RuntimeError("Cannot resolve native FP32 policy: Transformers dtype-policy compiler unavailable") from error
    try:
        policy, _, _ = build_glob_alternation(patterns)
    except (re.error, TypeError, ValueError) as error:
        raise ValueError("Cannot resolve invalid native FP32 dtype policy") from error
    names = set()
    quantized_prefixes = set()
    for prefix, module in model.named_modules():
        parts = prefix.split(".")
        if quantized_prefixes and any(".".join(parts[:end]) in quantized_prefixes for end in range(len(parts) + 1)):
            continue
        if isinstance(module, BaseQuantLinear) or getattr(module, "MODEL_DTYPE_BUFFERS", None) is not None:
            quantized_prefixes.add(prefix)
            continue
        slots = dict(module._parameters)
        slots.update({name: value for name, value in module._buffers.items()
                      if name not in module._non_persistent_buffers_set})
        for name, tensor in slots.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and policy.search(full_name):
                names.add(full_name)
    return names


def load_checkpoint_in_model_then_tie_weights(model, *args, **kwargs):
    checkpoint = kwargs.get("checkpoint")
    if checkpoint is None and args:
        checkpoint = args[0]
    # Positional argument order is Accelerate's public checkpoint-loader ABI.
    dtype = kwargs.get("dtype", args[3] if len(args) > 3 else None)
    device_map = kwargs.get("device_map", args[1] if len(args) > 1 else None)
    offload_folder = kwargs.get("offload_folder", args[2] if len(args) > 2 else None)
    native_fp32_names = _native_fp32_checkpoint_names(model, dtype)
    key_converter = kwargs.pop("key_converter", None)
    fp32_bf16_buffer_names = set(kwargs.pop("fp32_bf16_buffer_names", ()))
    packed_fp8_buffer_dtypes = dict(kwargs.pop("packed_fp8_buffer_dtypes", {}))
    restore_checkpoint_buffer_dtypes = bool(
        kwargs.pop("restore_checkpoint_buffer_dtypes", False)
    )
    fp8_weight_names = set(packed_fp8_buffer_dtypes)
    converted_checkpoint = None
    restore_checkpoint = checkpoint
    restore_name_map = None
    if checkpoint is not None and (callable(key_converter) or fp8_weight_names):
        converted_checkpoint = _remap_safetensors_checkpoint_keys(
            checkpoint, key_converter, fp8_weight_names,
        )
        if converted_checkpoint is not None:
            if args:
                args = (converted_checkpoint, *args[1:])
            else:
                kwargs["checkpoint"] = converted_checkpoint
            if fp8_weight_names:
                # Packed FP8 weights are absent from the temporary checkpoint:
                # Accelerate would cast them to the model dtype (and stage the
                # whole shard on the target device) before the loader restored
                # the packed bytes. Read them from the source checkpoint
                # instead, and use the same source for every other restored
                # buffer so key conversion is applied only once.
                restore_checkpoint = checkpoint
                if callable(key_converter):
                    restore_name_map = _checkpoint_runtime_to_source_keys(
                        checkpoint, key_converter,
                    )
            else:
                restore_checkpoint = converted_checkpoint
    try:
        load_checkpoint = converted_checkpoint or checkpoint
        _prepare_variable_quant_buffers(
            model,
            load_checkpoint,
        )
        accelerate.load_checkpoint_in_model(model, *args, **kwargs)
        restore_dtypes = dict(packed_fp8_buffer_dtypes)
        if str(dtype) in {"torch.bfloat16", "bfloat16"}:
            restore_dtypes.update(dict.fromkeys(fp32_bf16_buffer_names, torch.float32))
        # Restore from the checkpoint, not the already narrowed model tensor.
        # Accelerate ignores BF16 keep rules and cannot match native suffix /
        # regex policies such as ERNIE's `gate.weight` under FP16 either.
        restore_dtypes.update(dict.fromkeys(native_fp32_names, torch.float32))
        # Accelerate casts every floating buffer to the requested model dtype.
        # A quantized module's buffers (scales, QQQ `s_group`/`s_channel`,
        # EXL3 sign vectors, ...) are quantization state, not activation
        # operands: narrowing them changes the dequantized weight (#4458).
        # Restore each one the load narrowed, except the activation-side
        # buffers the module declares in `MODEL_DTYPE_BUFFERS`. The re-save
        # shell (`restore_checkpoint_buffer_dtypes`) republishes every
        # floating buffer at exactly its stored dtype instead (#4379).
        checkpoint_dtypes = _checkpoint_tensor_dtypes(load_checkpoint)
        for name, module in model.named_modules():
            model_dtype_buffers = getattr(module, "MODEL_DTYPE_BUFFERS", None)
            if model_dtype_buffers is None and not restore_checkpoint_buffer_dtypes:
                continue
            prefix = f"{name}." if name else ""
            for buffer_name, current in module._buffers.items():
                full_name = f"{prefix}{buffer_name}"
                checkpoint_dtype = checkpoint_dtypes.get(full_name)
                if (
                    checkpoint_dtype is None
                    or full_name in restore_dtypes
                    or not isinstance(current, torch.Tensor)
                    or not current.is_floating_point()
                    or buffer_name in module._non_persistent_buffers_set
                ):
                    continue
                if restore_checkpoint_buffer_dtypes:
                    restore = current.dtype != checkpoint_dtype
                else:
                    restore = buffer_name not in model_dtype_buffers and _load_narrowed(
                        checkpoint_dtype, current.dtype
                    )
                if restore:
                    restore_dtypes[full_name] = None
            if restore_checkpoint_buffer_dtypes:
                # Dense parameters (norms, embeddings, unquantized heads and
                # projections) are allocated in the export shell's model dtype
                # too. With ``dtype=None`` Accelerate copies a checkpoint
                # tensor into that slot, so a mixed-dtype checkpoint was
                # narrowed on re-save (#4466). The normal load path passes an
                # explicit model dtype and must keep casting them.
                for parameter_name, current in module._parameters.items():
                    full_name = f"{prefix}{parameter_name}"
                    checkpoint_dtype = checkpoint_dtypes.get(full_name)
                    if (
                        checkpoint_dtype is None
                        or full_name in restore_dtypes
                        or not isinstance(current, torch.Tensor)
                        or not current.is_floating_point()
                        or current.dtype == checkpoint_dtype
                    ):
                        continue
                    restore_dtypes[full_name] = None
        # BitsAndBytes keeps its 4-bit quant metadata in state buffers whose
        # dtype is part of the kernel contract: the floating state tables must
        # stay FP32 and the packed map/QuantState bytes must stay byte-exact.
        # Accelerate casts floating buffers to the model dtype for every load
        # dtype (fp16 included), which makes the first M=1 `gemm_4bit` call
        # reject a non-FP32 absmax. Restore them from the checkpoint instead.
        if any(
            getattr(module, "QUANT_TYPE", None) == "bitsandbytes"
            and getattr(module, "is_4bit", False)
            for module in model.modules()
        ):
            from ..nn_modules.qlinear.bitsandbytes import (
                _BITSANDBYTES_4BIT_STATE_BUFFER_NAMES,
            )

            for name, module in model.named_modules():
                if getattr(module, "QUANT_TYPE", None) != "bitsandbytes":
                    continue
                if not getattr(module, "is_4bit", False):
                    continue
                prefix = f"{name}." if name else ""
                for buffer_name in _BITSANDBYTES_4BIT_STATE_BUFFER_NAMES:
                    if isinstance(module._buffers.get(buffer_name), torch.Tensor):
                        restore_dtypes[f"{prefix}{buffer_name}"] = None
        if restore_dtypes:
            _restore_checkpoint_buffers(
                model, restore_checkpoint, restore_dtypes,
                exact_source_names=set(packed_fp8_buffer_dtypes),
                name_map=restore_name_map,
                device_map=device_map,
                offload_folder=offload_folder,
            )
        _tie_weights_after_checkpoint_load(
            model,
            load_checkpoint,
            extra_keys=fp8_weight_names if converted_checkpoint is not None else None,
        )
        _raise_on_unmaterialized_checkpoint_parameters(
            model, kwargs.get("checkpoint", checkpoint)
        )
    finally:
        _cleanup_remapped_checkpoint(converted_checkpoint)
        # Accelerate stages offloaded checkpoint tensors through pinned host
        # memory. The staging copies are dead by now, but PyTorch keeps their
        # blocks cached for the life of the process.
        release_pinned_host_cache()


# 32MB read/write i/o buffer
_STREAM_BUFFER_SIZE = 32 * 1024 * 1024
_STREAM_BUFFER = memoryview(bytearray(_STREAM_BUFFER_SIZE))
_STREAM_BUFFER_LOCK = threading.Lock()

def _copy_file_stream(
    src_path: str,
    dst_fh,
    length: int,
    *,
    offset: int = 0,
    expected_sha256: str | None = None,
) -> None:
    if expected_sha256 is not None and (
        len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError(f"Invalid source tensor SHA-256 for {src_path}")
    digest = hashlib.sha256() if expected_sha256 is not None else None
    with ctx(open(src_path, "rb", buffering=0), _STREAM_BUFFER_LOCK) as (src, _):
        if offset:
            src.seek(offset)
        remaining = length
        while remaining > 0:
            chunk_size = min(_STREAM_BUFFER_SIZE, remaining)
            read = src.readinto(_STREAM_BUFFER[:chunk_size])
            if not read:
                raise IOError(f"Unexpected EOF while copying from {src_path}")
            dst_fh.write(_STREAM_BUFFER[:read])
            if digest is not None:
                digest.update(_STREAM_BUFFER[:read])
            remaining -= read
    if digest is not None and digest.hexdigest() != expected_sha256:
        raise IOError(f"Source tensor content digest differs for {src_path}")


def _write_tensor_bytes(out, tensor: torch.Tensor) -> None:
    # Write the exact storage payload. The raw uint8 view covers bfloat16 and
    # float8, which have no direct NumPy bridge, and needs no NumPy at all.
    _owner, buffer = tensor_buffer(tensor)
    out.write(buffer)


def _safetensors_header(
    entries: List[TensorSource], metadata: Dict[str, str]
) -> tuple[Dict[str, Any], bytes, int]:
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for entry in entries:
        header[entry.name] = {
            "dtype": _torch_dtype_to_safetensors(entry.torch_dtype),
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.num_bytes],
        }
        offset += entry.num_bytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Safetensors pads the JSON header to an 8-byte boundary.
    # Without that padding, some readers reject the file as malformed.
    header_padding = (-len(header_bytes)) % 8
    if header_padding:
        header_bytes += b" " * header_padding
    return header, header_bytes, offset


class _Sha256Writer:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, payload) -> int:
        self.digest.update(payload)
        return len(payload)


def _tensor_payload_sha256(entry: TensorSource) -> str | None:
    source = entry.source
    if isinstance(source, OffloadTensorRef):
        digest = source.sha256
        if digest is not None:
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                return None
            return digest

        # Older offload indexes do not carry per-tensor digests. Comparing the
        # completed shard against the exact current source bytes is equivalent
        # to rewriting it from those bytes, without paying the second write.
        if source.format == "dat":
            offset = source.data_offsets[0] if source.data_offsets is not None else 0
            writer = _Sha256Writer()
            _copy_file_stream(source.path, writer, entry.num_bytes, offset=offset)
            return writer.digest.hexdigest()
        if source.format == "safetensors" and source.data_offsets is not None:
            start, end = source.data_offsets
            if end - start != entry.num_bytes:
                return None
            writer = _Sha256Writer()
            _copy_file_stream(source.path, writer, entry.num_bytes, offset=start)
            return writer.digest.hexdigest()
        if source.format == "safetensors":
            with safe_open(source.path, framework="pt", device="cpu") as handler:
                tensor = handler.get_tensor(source.weight_name or entry.name)
            writer = _Sha256Writer()
            _write_tensor_bytes(writer, tensor.to(source.torch_dtype))
            return writer.digest.hexdigest()
        return None

    writer = _Sha256Writer()
    _write_tensor_bytes(writer, source.detach())
    return writer.digest.hexdigest()


def _existing_shard_matches(
    path: str, entries: List[TensorSource], metadata: Dict[str, str]
) -> bool:
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    _, expected_header_bytes, payload_size = _safetensors_header(entries, metadata)
    expected_size = 8 + len(expected_header_bytes) + payload_size
    try:
        if os.path.getsize(path) != expected_size:
            return False
        with ctx(open(path, "rb", buffering=0), _STREAM_BUFFER_LOCK) as (source, _):
            raw_header_size = source.read(8)
            if len(raw_header_size) != 8:
                return False
            header_size = struct.unpack("<Q", raw_header_size)[0]
            if header_size != len(expected_header_bytes):
                return False
            if source.read(header_size) != expected_header_bytes:
                return False

        expected_digests = [_tensor_payload_sha256(entry) for entry in entries]
        if any(digest is None for digest in expected_digests):
            return False

        with ctx(open(path, "rb", buffering=0), _STREAM_BUFFER_LOCK) as (source, _):
            source.seek(8 + len(expected_header_bytes))
            for entry, expected_sha256 in zip(entries, expected_digests):
                digest = hashlib.sha256()
                remaining = entry.num_bytes
                while remaining > 0:
                    chunk_size = min(_STREAM_BUFFER_SIZE, remaining)
                    read = source.readinto(_STREAM_BUFFER[:chunk_size])
                    if not read:
                        return False
                    digest.update(_STREAM_BUFFER[:read])
                    remaining -= read
                if digest.hexdigest() != expected_sha256:
                    return False
            return source.read(1) == b""
    except (OSError, TypeError, ValueError, struct.error):
        return False


@contextmanager
def _atomic_shard_path(path: str):
    """Stage a shard beside its destination and publish it only on success."""
    directory = os.path.dirname(os.path.abspath(path))
    try:
        existing_mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        existing_mode = None

    # os.open applies the process umask to a new file without changing it.
    # O_EXCL and a random name keep the temporary private to this export.
    for _ in range(10):
        temporary = os.path.join(
            directory, f".{os.path.basename(path)}.{secrets.token_hex(16)}.tmp"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666 if existing_mode is None else 0o600,
            )
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(f"Could not create a temporary shard beside {path}")

    try:
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        os.close(descriptor)
        descriptor = None
        yield temporary
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _atomic_shard_file(path: str):
    """Keep the current shard and any mapped source tensors alive until commit."""
    with _atomic_shard_path(path) as temporary:
        with open(temporary, "wb") as handle:
            yield handle


def _write_shard_file(path: str, entries: List[TensorSource], metadata: Dict[str, str]) -> int:
    _, header_bytes, _ = _safetensors_header(entries, metadata)

    with _atomic_shard_file(path) as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)

        for entry in entries:
            source = entry.source
            if isinstance(source, OffloadTensorRef):
                if source.format == "dat":
                    # print("offload tensor io buffered transfer DAT")
                    start = 0
                    if source.data_offsets is not None:
                        start = source.data_offsets[0]
                    _copy_file_stream(
                        source.path,
                        out,
                        entry.num_bytes,
                        offset=start,
                        expected_sha256=source.sha256,
                    )
                elif source.format == "safetensors" and source.data_offsets is not None:
                    # print("offload tensor io buffered transfer SAFETENSOR stream")
                    start, end = source.data_offsets
                    _copy_file_stream(
                        source.path,
                        out,
                        end - start,
                        offset=start,
                        expected_sha256=source.sha256,
                    )
                else:
                    # print("offload tensor slow tensor read")
                    with safe_open(source.path, framework="pt", device="cpu") as handler:
                        tensor = handler.get_tensor(source.weight_name or entry.name)
                    tensor = tensor.to(source.torch_dtype)
                    _write_tensor_bytes(out, tensor)
            else:
                tensor = source.detach()
                _write_tensor_bytes(out, tensor)
                del tensor

        file_size = out.tell()

    return file_size


def _plan_shards(entries: List[TensorSource], max_shard_size: Optional[int]) -> List[List[TensorSource]]:
    if not max_shard_size or max_shard_size <= 0:
        return [entries]

    shards: List[List[TensorSource]] = []
    current: List[TensorSource] = []
    current_size = 0

    for entry in entries:
        size = entry.num_bytes
        if size > max_shard_size:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([entry])
            continue
        if current_size + size > max_shard_size and current:
            shards.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += size

    if current:
        shards.append(current)

    return shards


def _validate_offload_sources_outside_save_dir(
    state_dict: Dict[str, TensorSource], save_dir: str
) -> None:
    """Keep source artifacts out of the export directory we may rewrite or clean."""
    save_roots = (os.path.abspath(save_dir), os.path.realpath(save_dir))
    for entry in state_dict.values():
        source = entry.source
        if not isinstance(source, OffloadTensorRef):
            continue
        source_paths = (os.path.abspath(source.path), os.path.realpath(source.path))
        for save_root, source_path in zip(save_roots, source_paths):
            try:
                if os.path.commonpath((save_root, source_path)) == save_root:
                    break
            except ValueError:  # Paths on different Windows drives cannot overlap.
                continue
        else:
            continue
        raise ValueError(
            f"Offloaded tensor '{entry.name}' references '{source.path}' inside "
            f"the export directory '{save_dir}'; choose a different save_dir "
            "so shard writes and cleanup cannot destroy the source."
        )


def streaming_state_dict_to_shards(
    state_dict: Dict[str, TensorSource],
    save_dir: str,
    model_base_name: str,
    single_file_name: str,
    metadata: Dict[str, str],
    max_shard_size: Optional[int],
) -> Tuple[List[str], Dict[str, str], int]:
    _validate_offload_sources_outside_save_dir(state_dict, save_dir)
    entries = list(state_dict.values())
    shards = _plan_shards(entries, max_shard_size)
    num_shards = len(shards)
    filenames: List[str] = []
    tensor_to_filename: Dict[str, str] = {}
    total_size = 0

    for idx, shard_entries in enumerate(shards, start=1):
        if num_shards == 1:
            filename = single_file_name
        else:
            filename = f"{model_base_name}-{idx:05d}-of-{num_shards:05d}.safetensors"

        path = os.path.join(save_dir, filename)
        if _existing_shard_matches(path, shard_entries, metadata):
            log.info(f"Reusing authenticated safetensors shard: {filename}")
        else:
            _write_shard_file(path, shard_entries, metadata)
        # Hugging Face's index describes the tensor payload, not the on-disk
        # file size. The latter includes the safetensors header and padding and
        # made `metadata.total_size` disagree with every other writer.
        total_size += sum(entry.num_bytes for entry in shard_entries)
        filenames.append(filename)
        for entry in shard_entries:
            tensor_to_filename[entry.name] = filename

    return filenames, tensor_to_filename, total_size


def find_config_seq_len(config_dict, target_keys):
    for k, v in config_dict.items():
        if k in target_keys:
            return v
        if isinstance(v, dict):
            found = find_config_seq_len(v, target_keys)
            if found is not None:
                return found
    return None


def get_module_name(module: nn.Module, child_module: nn.Module) -> str:
    for name, candidate in module.named_modules():
        if candidate is child_module:
            return name
    raise ValueError(f"Cannot find child_module {child_module} in module {module}")


def untie_word_embeddings(model: nn.Module) -> nn.Module:
    """Clone a tied output head so embedding endpoints can be quantized independently."""
    if not getattr(model.config, "tie_word_embeddings", False):
        return model

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("Cannot untie word embeddings without both input and output embedding modules.")

    model.config.tie_word_embeddings = False
    new_head = nn.Linear(
        input_embeddings.weight.shape[1],
        input_embeddings.weight.shape[0],
        bias=getattr(output_embeddings, "bias", None) is not None,
        device=input_embeddings.weight.device,
        dtype=input_embeddings.weight.dtype,
    )
    new_head.weight.data.copy_(input_embeddings.weight.data)
    if new_head.bias is not None and output_embeddings.bias is not None:
        new_head.bias.data.copy_(output_embeddings.bias.data)
    model.set_output_embeddings(new_head)
    return model


def check_module_quantized_in_keys(keys, module_name: str) -> bool:
    return any(
        key.startswith(module_name + ".")
        and (".qweight" in key or ".qzeros" in key or ".scales" in key)
        for key in keys
    )


def is_embeddings_module_quantized(
    model_dir: str,
    input_embed_name: Optional[str],
    output_embed_name: Optional[str],
) -> Tuple[bool, bool]:
    input_quantized = False
    output_quantized = False

    def inspect_keys(keys) -> Tuple[bool, bool]:
        return (
            bool(input_embed_name and check_module_quantized_in_keys(keys, input_embed_name)),
            bool(output_embed_name and check_module_quantized_in_keys(keys, output_embed_name)),
        )

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        return inspect_keys(index.get("weight_map", {}).keys())

    safetensor_files = [
        os.path.join(model_dir, filename)
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    ]
    for safefile in safetensor_files:
        try:
            with safe_open(safefile, framework="pt") as handle:
                found_input, found_output = inspect_keys(handle.keys())
                input_quantized = input_quantized or found_input
                output_quantized = output_quantized or found_output
                if input_quantized and output_quantized:
                    break
        except Exception as exc:
            log.warn(f"Failed to inspect {safefile}: {exc}")

    return input_quantized, output_quantized


def has_any_attr(obj, names):
    return any(hasattr(obj, name) for name in names)


def find_moe_routing_modules(model):
    modules = []
    for module in model.modules():
        if has_any_attr(module, MOE_TOPK_FIELD_NAMES) and _get_local_expert_count(module):
            modules.append(module)
    return modules


def _get_local_expert_count(module: nn.Module) -> Optional[int]:
    """Return the expert cardinality owned by one router module.

    Most MoE implementations expose ``num_experts`` directly.  K2 Horizon's
    MoVA value router instead owns an ``nn.ModuleList`` called ``v_experts``;
    deriving its local count is important because the model also contains a
    separate 100-expert MLP router.
    """
    for name in MOE_NUM_EXPERTS_FIELD_NAMES:
        value = getattr(module, name, None)
        if isinstance(value, int) and value > 0:
            return value

    experts = getattr(module, "v_experts", None)
    if experts is not None:
        try:
            count = len(experts)
        except TypeError:
            count = 0
        if count > 0:
            return count
    return None


def _install_phimoe_topk_override(router: nn.Module) -> None:
    """Make ``PhimoeTopKRouter`` honour a changed ``top_k``.

    Transformers 5.22 passes ``top_k`` into ``sparsemixer``, but that function
    always selects exactly two experts (the argmax and the argmax of the
    remainder).  Writing ``router.top_k`` alone is therefore a silent no-op.
    Keep the native sparse-mixer path for the model's configured value, and
    only replace the forward when the override actually changes it.
    """

    if getattr(router, "_gptqmodel_phimoe_topk_aware", False):
        return

    native_top_k = getattr(router, "top_k", None)
    if not isinstance(native_top_k, int):
        return

    original_forward = router.forward

    def forward_with_topk(hidden_states):
        if getattr(router, "top_k", None) == native_top_k:
            return original_forward(hidden_states)

        input_jitter_noise = getattr(router, "input_jitter_noise", 0.0)
        if router.training and input_jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - input_jitter_noise, 1.0 + input_jitter_noise
            )

        router_logits = nn.Linear.forward(router, hidden_states)
        top_k = int(router.top_k)
        num_experts = getattr(router, "num_experts", None)
        if isinstance(num_experts, int) and num_experts > 0:
            top_k = min(top_k, num_experts)
        top_k = max(1, top_k)
        routing_weights = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
        return router_logits, topk_weights, topk_indices

    router.forward = forward_with_topk
    router._gptqmodel_phimoe_topk_aware = True


def set_moe_topk(model: nn.Module, new_topk: int) -> MoETopKState:
    routers = find_moe_routing_modules(model)
    state: MoETopKState = []
    for r in routers:
        local_count = _get_local_expert_count(r)
        if local_count is None:
            continue
        if r.__class__.__name__ == "PhimoeTopKRouter":
            _install_phimoe_topk_override(r)
        for name in MOE_TOPK_FIELD_NAMES:
            if hasattr(r, name):
                old = getattr(r, name)
                assert isinstance(old, int)
                state.append((r, name, old))
                # A global override is resolved independently for every
                # router.  In particular, an override of 100 means all 100
                # MLP experts but only the 64 local MoVA value experts.
                setattr(r, name, min(new_topk, local_count))
                break
    return state


def restore_moe_topk(state: MoETopKState):
    for module, name, old in state:
        if hasattr(module, name):
            setattr(module, name, old)
