# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import importlib
import os
import pkgutil
from collections import OrderedDict
from typing import Dict, List, Optional, Type, Union

import torch

from gptqmodel.adapter.adapter import Adapter

from ..models._const import DEVICE, normalize_device
from ..nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ..quantization import FORMAT, METHOD
from ..utils.env import env_flag
from ..utils.logger import setup_logger
from . import BACKEND
from .rocm import IS_ROCM
from .torch import HAS_CUDA, HAS_MPS, HAS_XPU


ACCELERATE_DEVICE_MAP_KEYWORDS = {"auto", "balanced", "sequential"}
ACCELERATE_DEVICE_MAP_PREFIXES = ("balanced_low_",)
ACCELERATE_OFFLOAD_TARGETS = {"disk", "meta"}


message_logged = False
log = setup_logger()

def iter_quant_linear_kernels() -> List[Type[BaseQuantLinear]]:
    kernels = []
    seen = set()

    def _walk(cls):
        for subcls in cls.__subclasses__():
            if subcls in seen:
                continue
            seen.add(subcls)
            _walk(subcls)
            if "SUPPORTS_FORMATS" in subcls.__dict__:
                kernels.append(subcls)

    _walk(BaseQuantLinear)
    return kernels


def infer_quant_methods(cls: Type[BaseQuantLinear]) -> List[METHOD]:
    return [
        METHOD(method) if isinstance(method, METHOD) else METHOD(str(method).lower())
        for method in cls.SUPPORTS_METHODS
    ]


def get_kernel_backends(cls: Type[BaseQuantLinear]) -> List[BACKEND]:
    backends = []
    for backend in cls.SUPPORTS_BACKENDS:
        if isinstance(backend, BACKEND):
            backends.append(backend)
        else:
            backends.append(BACKEND(str(backend).lower()))
    return backends


def get_kernel_for_backend(backend: BACKEND, quant_method: METHOD, fmt: FORMAT) -> Type[BaseQuantLinear]:
    matches = []
    for cls in iter_quant_linear_kernels():
        if backend not in get_kernel_backends(cls):
            continue
        if quant_method not in cls.SUPPORTS_METHODS:
            continue
        if fmt not in cls.SUPPORTS_FORMATS:
            continue
        matches.append(cls)

    if not matches:
        raise ValueError(f"Unsupported backend: `{backend}` for `{quant_method}` with format `{fmt}`")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple kernels matched backend `{backend}` for `{quant_method}` with format `{fmt}`: "
            f"{', '.join(cls.__name__ for cls in matches)}"
        )
    return matches[0]


def _import_all_qlinear_kernels() -> None:
    from ..nn_modules import qlinear as qlinear_pkg

    for module_info in pkgutil.iter_modules(qlinear_pkg.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{qlinear_pkg.__name__}.{name}")
        except ImportError as exc:
            log.debug(f"Skipping qlinear module import `{name}`: {exc}")


def build_kernel_support_maps():
    _import_all_qlinear_kernels()
    # Build auto-select order and format support from kernel declarations.
    auto_entries = {}
    support_entries = {}

    for cls in iter_quant_linear_kernels():
        supports_formats = cls.SUPPORTS_FORMATS
        if not isinstance(supports_formats, dict):
            raise ValueError(f"{cls.__name__}.SUPPORTS_FORMATS must be a dict of FORMAT -> priority.")

        for backend in get_kernel_backends(cls):
            for method in infer_quant_methods(cls):
                for fmt, priority in supports_formats.items():
                    if not isinstance(fmt, FORMAT):
                        fmt = FORMAT(str(fmt).lower())
                    if not isinstance(priority, int):
                        raise ValueError(f"{cls.__name__}.SUPPORTS_FORMATS[{fmt}] priority must be an int.")

                    support_entries.setdefault(method, {}).setdefault(fmt, []).append((priority, backend))
                    # Priority <= 0 keeps format support but opts out of auto-selection.
                    if priority > 0:
                        auto_entries.setdefault(method, {}).setdefault(fmt, []).append((priority, backend, cls))

    supports_backend_map = {}
    auto_select_backend_order_map = {}

    for method, fmt_entries in support_entries.items():
        supports_backend_map[method] = {}
        for fmt, entries in fmt_entries.items():
            entries.sort(key=lambda item: (item[0], item[1].value), reverse=True)
            seen_backends = set()
            ordered_backends = []
            for _, backend in entries:
                if backend in seen_backends:
                    continue
                seen_backends.add(backend)
                ordered_backends.append(backend)
            supports_backend_map[method][fmt] = ordered_backends

    for method, fmt_entries in auto_entries.items():
        auto_select_backend_order_map[method] = {}
        for fmt, entries in fmt_entries.items():
            entries.sort(key=lambda item: (item[0], item[1].value), reverse=True)
            ordered = OrderedDict()
            for _, backend, cls in entries:
                if backend in ordered:
                    continue
                ordered[backend] = cls
            auto_select_backend_order_map[method][fmt] = ordered

    return auto_select_backend_order_map, supports_backend_map


AUTO_BACKEND_KERNEL_MAPPING, BACKEND_TO_METHOD_FORMAT_MAPPING = build_kernel_support_maps()


def debug_print_kernel_maps():
    def render_tree(title, tree):
        # Simple ANSI palette for depth coloring, aligned with print_module_tree.
        depth_colors = [
            "\033[36m",  # cyan
            "\033[33m",  # yellow
            "\033[35m",  # magenta
            "\033[32m",  # green
            "\033[34m",  # blue
            "\033[31m",  # red
        ]
        trunk_color = "\033[90m"
        reset = "\033[0m"

        lines = [title]
        methods = sorted(tree.keys(), key=lambda m: m.value)
        for mi, method in enumerate(methods):
            method_last = mi == len(methods) - 1
            method_prefix = "└─ " if method_last else "├─ "
            method_name = f"{depth_colors[0]}{method.value}{reset}"
            lines.append(f"{trunk_color}{method_prefix}{reset}{method_name}")
            fmt_prefix = "   " if method_last else "│  "
            formats = sorted(tree[method].keys(), key=lambda f: f.value)
            for fi, fmt in enumerate(formats):
                fmt_last = fi == len(formats) - 1
                fmt_trunk = "└─ " if fmt_last else "├─ "
                fmt_name = f"{depth_colors[1]}{fmt.value}{reset}"
                lines.append(f"{trunk_color}{fmt_prefix}{fmt_trunk}{reset}{fmt_name}")
                child_prefix = fmt_prefix + ("   " if fmt_last else "│  ")
                entries = tree[method][fmt]
                for bi, entry in enumerate(entries):
                    entry_last = bi == len(entries) - 1
                    entry_trunk = "└─ " if entry_last else "├─ "
                    entry_name = f"{depth_colors[2]}{entry}{reset}"
                    lines.append(f"{trunk_color}{child_prefix}{entry_trunk}{reset}{entry_name}")
        return "\n".join(lines)

    auto_tree = {}
    for method, fmt_map in AUTO_BACKEND_KERNEL_MAPPING.items():
        auto_tree[method] = {}
        for fmt, backend_map in fmt_map.items():
            entries = [f"{backend.value} -> {cls.__name__}" for backend, cls in backend_map.items()]
            auto_tree[method][fmt] = entries

    supports_tree = {}
    for method, fmt_map in BACKEND_TO_METHOD_FORMAT_MAPPING.items():
        supports_tree[method] = {}
        for fmt, backends in fmt_map.items():
            supports_tree[method][fmt] = [backend.value for backend in backends]

    print(render_tree("AUTO KERNEL SELECTION MAPPING", auto_tree))
    print(render_tree("KERNEL BACKEND to METHOD/FORMAT MAPPING", supports_tree))


if env_flag("DEBUG"):
    debug_print_kernel_maps()


def _is_accelerate_device_map_keyword(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ACCELERATE_DEVICE_MAP_KEYWORDS:
        return True
    return any(lowered.startswith(prefix) for prefix in ACCELERATE_DEVICE_MAP_PREFIXES)


def _is_accelerate_offload_target(value: str) -> bool:
    return value.strip().lower() in ACCELERATE_OFFLOAD_TARGETS


def normalize_device_device_map(device: Optional[Union[str, torch.device]], device_map: Optional[Union[str, Dict]]) -> Optional[DEVICE]:
    normalized_device = None
    accelerator = torch.accelerator.current_accelerator()
    if device is None:
        if device_map is not None:
            if isinstance(device_map, str):
                if _is_accelerate_device_map_keyword(device_map):
                    return DEVICE(accelerator.type) if accelerator is not None else DEVICE.CPU
                devices = {device_map}
            else:
                devices = set(device_map.values())
            normalized_devices = set()
            for device in devices:
                # Returning None means quant linear will be automatically selected.
                if device is None:
                    continue
                if isinstance(device, str):
                    if _is_accelerate_device_map_keyword(device) or device == "auto":
                        return DEVICE(accelerator.type) if accelerator is not None else DEVICE.CPU
                    if _is_accelerate_offload_target(device):
                        continue
                normalized_devices.add(normalize_device(device))
            if len(normalized_devices) == 1:
                d = normalized_devices.pop()
                if d in DEVICE:
                    normalized_device = d
            elif len(normalized_devices) > 1:
                normalized_devices.discard(DEVICE.CPU)
                normalized_device = normalized_devices.pop()
    else:
        if isinstance(device, str):
            normalized_device = normalize_device(device)
        elif isinstance(device, torch.device):
            normalized_device = DEVICE(device.type)
        else:
            raise ValueError(f"device must be a string or torch.device, got {type(device)}")

    # map fake cuda to actual rocm
    if normalized_device == DEVICE.CUDA and IS_ROCM:
        normalized_device = DEVICE.ROCM
    return normalized_device


def auto_select_device(device: Optional[DEVICE], backend: Optional[BACKEND]) -> DEVICE:
    assert device is None or isinstance(device, DEVICE)
    assert backend is None or isinstance(backend, BACKEND)

    if device is None:
        if HAS_CUDA:
            device = DEVICE.CUDA
        elif HAS_XPU:
            device = DEVICE.XPU
        elif HAS_MPS:
            device = DEVICE.MPS
        else:
            device = DEVICE.CPU
    return device

# public/stable api exposed to transformer/optimum
def hf_select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        checkpoint_format: str,
        meta: Optional[Dict[str, any]] = None,
        pack: Optional[bool] = True,
        device_map: Optional[Union[str, dict]] = None,
        backend: Optional[Union[str, BACKEND]] = None,
) -> Type[BaseQuantLinear]:
    # convert hf string backend to backend.enum
    if isinstance(backend, str):
        backend = BACKEND(backend.lower())

    if device_map is not None:
        device = normalize_device_device_map(None, device_map)
    else:
        device = DEVICE.CPU

    return select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        device=device,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack=pack,
        allow_marlin=True, # TODO: remove this after marlin padding is fixed
        dynamic=None,
        pack_dtype=torch.int32,
        adapter=None,
    )

# public/stable api exposed to transformer/optimum
def hf_select_quant_linear_v2(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        format: Union[str, FORMAT], # awq `version` should be pre-mapped to format
        quant_method: Union[str, METHOD], # awq llm-awq `version` should be pre-mapped to method
        zero_point: Optional[bool] = None, # awq only (True=asymmetric, False=symmetric)
        dtype: Optional[Union[str, torch.dtype]] = None,
        meta: Optional[Dict[str, any]] = None,
        pack: Optional[bool] = True,
        device_map: Optional[Union[str, dict]] = None,
        backend: Optional[Union[str, BACKEND]] = None,
) -> Type[BaseQuantLinear]:
    # convert hf string backend to backend.enum
    if isinstance(backend, str):
        backend = BACKEND(backend.lower())

    def _normalize_enum(value, enum_cls, field: str):
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value.lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported {field}: `{value}`") from exc
        raise ValueError(f"{field} must be a string or `{enum_cls.__name__}`, got `{type(value)}`")

    def _normalize_dtype(value: Optional[Union[str, torch.dtype]], field: str) -> Optional[torch.dtype]:
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            normalized = value.replace("torch.", "").lower()
            candidate = getattr(torch, normalized, None)
            if isinstance(candidate, torch.dtype):
                return candidate
        raise ValueError(f"Unsupported {field}: `{value}`")

    method = _normalize_enum(quant_method, METHOD, "quant_method")
    fmt = _normalize_enum(format, FORMAT, "format")
    normalized_dtype = _normalize_dtype(dtype, "dtype")

    pack_dtype_override = None
    if meta is not None:
        pack_dtype_override = meta.get("pack_dtype", None)
    # GEMV_FAST checkpoints are packed as int16; default to int32 otherwise.
    default_pack_dtype = torch.int16 if method == METHOD.AWQ and fmt == FORMAT.GEMV_FAST else torch.int32
    pack_dtype = _normalize_dtype(pack_dtype_override, "pack_dtype") if pack_dtype_override is not None else default_pack_dtype

    if device_map is not None:
        device = normalize_device_device_map(None, device_map)
    else:
        device = DEVICE.CPU

    if format == FORMAT.LLM_AWQ:
        # llm-awq uses torch.int16 to pack qweight
        pack_dtype = torch.int16

    effective_sym = sym
    if zero_point is not None:
        effective_sym = not bool(zero_point)

    return select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=effective_sym,
        backend=backend,
        device=device,
        format=fmt,
        quant_method=method,
        pack=pack,
        allow_marlin=True,  # TODO: remove this after marlin padding is fixed
        dynamic=None,
        pack_dtype=pack_dtype,
        dtype=normalized_dtype,
        adapter=None,
    )


# auto select the correct/optimal QuantLinear class
def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        device: DEVICE,
        backend: BACKEND = BACKEND.AUTO,
        format: FORMAT = FORMAT.GPTQ,
        quant_method: METHOD = METHOD.GPTQ,
        pack: bool = False,
        allow_marlin: bool = True,  # TODO: remove this after marlin padding is fixed
        dynamic=None,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        multi_select: bool = False, # return all valid kernels
        adapter: Optional[Adapter] = None,
) -> Union[Type[BaseQuantLinear], List[Type[BaseQuantLinear]]]:
    if isinstance(format, str):
        format = FORMAT(format.lower())
    if isinstance(quant_method, str):
        quant_method = METHOD(quant_method.lower())

    supported_formats = BACKEND_TO_METHOD_FORMAT_MAPPING.get(quant_method)
    if supported_formats is None:
        raise ValueError(f"Unsupported quantization method: `{quant_method}`")
    if format not in supported_formats:
        raise ValueError(f"Unsupported format: `{format}` for quantization method `{quant_method}`")

    backend = BACKEND.AUTO if backend is None else backend

    trainable = backend == BACKEND.AUTO_TRAINABLE

    validated_qlinears = []
    # Handle the case where backend is AUTO.
    if backend in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
        allow_quant_linears = list(AUTO_BACKEND_KERNEL_MAPPING[quant_method].get(format, {}).items())
        if not allow_quant_linears:
            raise ValueError(f"No auto-select kernels found for `{quant_method}` with format `{format}`.")

        err = None
        global message_logged
        # Suppose all quant linears in the model should have the same backend.
        for k, cls in allow_quant_linears:
            validate, err = cls.validate(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,
                sym=sym,
                pack_dtype=pack_dtype,
                dtype=dtype,
                dynamic=dynamic,
                device=device,
                trainable=trainable,
                adapter=adapter,
            )
            if os.environ.get("DEBUG") and not validate:
                log.info(f"skip {k} for {str(err)}")
            if validate:
                if pack:
                    check_pack_func = issubclass(cls, PackableQuantLinear) or (
                        hasattr(cls, "pack_block") and callable(getattr(cls, "pack_block"))
                    )
                    if check_pack_func:
                        #if not message_logged:
                        #    logger.info(f"Auto pick kernel based on compatibility: {cls}")
                        #    message_logged = True
                        log.info(f"{'Packing ' if pack else ''}Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                        validated_qlinears.append(cls)
                        if not multi_select:
                            log.info(f"Kernel: selected -> `{cls.__name__}`.")
                            return cls
                else:
                    #if not message_logged:
                    #    logger.info(f"Auto pick kernel based on compatibility: {cls}")
                    #    message_logged = True
                    log.info(f"{'Packing ' if pack else ''}Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                    validated_qlinears.append(cls)
                    if not multi_select:
                        log.info(f"Kernel: selected -> `{cls.__name__}`.")
                        return cls

        if err:
            raise err

        if len(validated_qlinears) == 0:
            raise ValueError("No valid quant linear")

        return validated_qlinears

    # TODO check AWQ format supports BACKEND

    # Handle the case where backend is not AUTO.
    qlinear = get_kernel_for_backend(backend, quant_method, format)

    validate, err = qlinear.validate(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        pack_dtype=pack_dtype,
        dtype=dtype,
        dynamic=dynamic,
        device=device,
        trainable=trainable,
    )

    log.info(f"{'Packing' if pack else ''} Kernel: selected: `{qlinear.__name__}`")
    if not validate:
        raise ValueError(err)
    else:
        if multi_select:
            return [qlinear]
        else:
            return qlinear
