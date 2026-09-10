# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import importlib
import os
import pkgutil
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Type, Union

import torch

from gptqmodel.adapter.adapter import Adapter

from ..models._const import DEVICE, normalize_device
from ..nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ..quantization import FORMAT, METHOD
from ..quantization.config import _normalize_quant_bits, quant_bits_width
from ..utils.env import env_flag
from ..utils.logger import setup_logger
from . import BACKEND
from .backend import normalize_backend
from .rocm import IS_ROCM
from .torch import HAS_CUDA, HAS_MPS, HAS_NPU, HAS_XPU


ACCELERATE_DEVICE_MAP_KEYWORDS = {"auto", "balanced", "sequential"}
ACCELERATE_DEVICE_MAP_PREFIXES = ("balanced_low_",)
ACCELERATE_OFFLOAD_TARGETS = {"disk", "meta"}


message_logged = False
log = setup_logger()


# A selector device is deliberately kept separate from ``DEVICE``.  DEVICE is
# the kernel declaration's *support family* (CUDA, CPU, ...), whereas a
# torch.device carries the concrete ordinal needed for capability checks and
# per-device module placement.
SelectorDevice = Union[DEVICE, torch.device]
SelectorDevices = Union[SelectorDevice, Sequence[SelectorDevice]]
_VALIDATION_CACHE_MAXSIZE = 1024
_VALIDATION_CACHE = OrderedDict()
_VALIDATION_CACHE_LOCK = threading.RLock()


def _device_family(device: SelectorDevice) -> DEVICE:
    if isinstance(device, DEVICE):
        return device
    device = torch.device(device)
    if IS_ROCM and device.type == "cuda":
        return DEVICE.ROCM
    try:
        return DEVICE(device.type)
    except ValueError as exc:
        raise ValueError(f"Unsupported selector device family `{device.type}`") from exc


def _as_selector_device(value) -> SelectorDevice:
    """Convert one public device value without discarding an ordinal."""
    if isinstance(value, DEVICE):
        if value == DEVICE.ALL:
            return value
        return value
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int):
        # Accelerate uses integer device-map values as CUDA ordinals.
        return torch.device(f"cuda:{value}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text == DEVICE.ROCM.value:
            return torch.device("cuda:0")
        try:
            return torch.device(text)
        except (RuntimeError, ValueError) as exc:
            # Preserve the old DEVICE error wording for symbolic families.
            try:
                return DEVICE(text)
            except ValueError:
                raise ValueError(f"Invalid device `{value}`") from exc
    raise ValueError(f"device must be a string, int, torch.device, or DEVICE, got {type(value)}")


def _selector_devices(device: Optional[SelectorDevices]) -> Optional[tuple[SelectorDevice, ...]]:
    if device is None:
        return None
    if isinstance(device, (str, int, torch.device, DEVICE)):
        values = (_as_selector_device(device),)
    else:
        if isinstance(device, (bytes, bytearray)):
            raise ValueError("device sequence must contain device values")
        values = tuple(_as_selector_device(item) for item in device)
    if not values:
        raise ValueError("device sequence must not be empty")

    # Keep first-seen ordering.  This makes multi-GPU selection deterministic
    # while still avoiding repeated capability probes for duplicate map values.
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _device_capability(device: SelectorDevice):
    """Return one device's capability, if its runtime exposes one."""
    if isinstance(device, DEVICE):
        if device not in (DEVICE.CUDA, DEVICE.ROCM):
            return None
        target = device.to_torch_device()
    else:
        target = torch.device(device)
    if target.type == "cuda":
        try:
            return tuple(torch.cuda.get_device_capability(target))
        except Exception:
            return None
    getter = getattr(getattr(torch, target.type, None), "get_device_capability", None)
    if getter is not None:
        try:
            return tuple(getter(target))
        except Exception:
            return None
    return None


def _selector_device_descriptor(device: SelectorDevice):
    if isinstance(device, DEVICE):
        target = device.to_torch_device() if device != DEVICE.ALL else None
        return (
            "family",
            device.value,
            None if target is None else target.type,
            None if target is None else target.index,
            _device_capability(device),
        )
    target = torch.device(device)
    return ("torch", target.type, target.index, _device_capability(target))


def _freeze_validation_value(value):
    if isinstance(value, (str, int, float, bool, type(None), torch.dtype)):
        return value
    if isinstance(value, (torch.device, DEVICE)):
        return _selector_device_descriptor(value)
    if isinstance(value, dict):
        return tuple(sorted((_freeze_validation_value(k), _freeze_validation_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        items = (_freeze_validation_value(item) for item in value)
        return tuple(sorted(items, key=repr)) if isinstance(value, (set, frozenset)) else tuple(items)
    if isinstance(value, Adapter):
        # Adapters can carry mutable runtime state.  Never retain one in the
        # cache value or use its mutable representation as a cache key.
        return (type(value), id(value))
    try:
        hash(value)
    except TypeError:
        return (type(value), id(value))
    return value


def _validation_method_identity(cls, name: str):
    for parent in cls.__mro__:
        descriptor = parent.__dict__.get(name)
        if descriptor is not None:
            return id(descriptor)
    return None


def _new_validation_error(error_type, message):
    if error_type is None:
        return None
    try:
        return error_type(message)
    except Exception:
        return ValueError(message)


def validate_quant_linear(cls: Type[BaseQuantLinear], **kwargs):
    """Validate one kernel contract with a bounded, device-aware cache.

    The cache stores only immutable status/error metadata.  In particular, an
    exception object (which carries traceback and mutable state) is recreated
    for every caller.
    """
    requested_device = kwargs.get("device")
    if isinstance(requested_device, (str, int)):
        kwargs = dict(kwargs)
        kwargs["device"] = _as_selector_device(requested_device)
        requested_device = kwargs["device"]
    if requested_device is not None and not isinstance(requested_device, (str, int, torch.device, DEVICE)):
        targets = _selector_devices(requested_device)
        assert targets is not None
        last_result = (True, None)
        for target in targets:
            target_kwargs = dict(kwargs)
            target_kwargs["device"] = target
            last_result = validate_quant_linear(cls, **target_kwargs)
            if not last_result[0]:
                return last_result
        return last_result

    key = (
        cls,
        _validation_method_identity(cls, "validate"),
        _validation_method_identity(cls, "cached_validate_once"),
        tuple(sorted((name, _freeze_validation_value(value)) for name, value in kwargs.items())),
    )
    with _VALIDATION_CACHE_LOCK:
        cached = _VALIDATION_CACHE.get(key)
        if cached is not None:
            _VALIDATION_CACHE.move_to_end(key)
            ok, error_type, error_message = cached
            return ok, _new_validation_error(error_type, error_message)

    try:
        result = cls.validate(**kwargs)
        ok, error = result
    except Exception as exc:
        ok, error = False, exc

    error_type = type(error) if isinstance(error, BaseException) else (ValueError if error is not None else None)
    error_message = str(error) if error is not None else None
    cached = (bool(ok), error_type, error_message)
    with _VALIDATION_CACHE_LOCK:
        _VALIDATION_CACHE[key] = cached
        _VALIDATION_CACHE.move_to_end(key)
        while len(_VALIDATION_CACHE) > _VALIDATION_CACHE_MAXSIZE:
            _VALIDATION_CACHE.popitem(last=False)
    return bool(ok), _new_validation_error(error_type, error_message)


def clear_validation_cache() -> None:
    with _VALIDATION_CACHE_LOCK:
        _VALIDATION_CACHE.clear()


def selector_device_family(device: Optional[SelectorDevices]) -> Optional[DEVICE]:
    """Return the support family for placement-only loader decisions."""
    targets = _selector_devices(device)
    if not targets:
        return None
    _validate_selector_device_families(targets)
    return _device_family(targets[0])


def expand_selector_device_family(device: SelectorDevices) -> SelectorDevice | tuple[SelectorDevice, ...]:
    """Expand one abstract accelerator family to all visible physical devices."""
    targets = _selector_devices(device)
    assert targets is not None
    if len(targets) != 1:
        return targets[0] if len(targets) == 1 else targets

    target = targets[0]
    if isinstance(target, DEVICE):
        family = target
    elif isinstance(target, torch.device) and target.index is None and target.type in {"cuda", "xpu", "npu"}:
        family = _device_family(target)
    else:
        return target

    if family not in (DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU):
        return family

    runtime_type = "cuda" if family == DEVICE.ROCM else family.type
    runtime = getattr(torch, runtime_type, None)
    count_getter = getattr(runtime, "device_count", None)
    if count_getter is None:
        return family
    try:
        count = count_getter()
    except Exception:
        return family
    if count < 1:
        return family

    devices = tuple(torch.device(f"{runtime_type}:{index}") for index in range(count))
    _validate_selector_device_families(devices)
    return devices[0] if len(devices) == 1 else devices


def _supports_pack_api(cls: Type[BaseQuantLinear]) -> bool:
    return (
        issubclass(cls, PackableQuantLinear)
        or (hasattr(cls, "pack") and callable(getattr(cls, "pack")))
        or (hasattr(cls, "pack_block") and callable(getattr(cls, "pack_block")))
    )


def _iter_dynamic_contracts(
    dynamic,
    bits: int,
    group_size: int,
    desc_act: bool,
    sym: bool,
    pack_dtype: torch.dtype,
    format_value: FORMAT,
):
    """Yield distinct effective quantization configs from base + optional dynamic overrides.

    When `dynamic` is provided, auto-selection tests each kernel against the union of
    effective (bits, group_size, desc_act, sym, pack_dtype) contracts. This lets a
    mixed-bitwidth model use a kernel that only supports 4-bit weights for its 4-bit
    layers even when the base `bits` is 3.
    """

    base_contract = {
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "sym": sym,
        "pack_dtype": pack_dtype,
    }
    if not dynamic:
        yield base_contract
        return

    seen = {(bits, group_size, desc_act, sym, pack_dtype)}
    yield base_contract

    for pattern, overrides in dynamic.items():
        if not isinstance(overrides, dict):
            continue
        if isinstance(pattern, str) and pattern.startswith("-"):
            continue
        if overrides is None:
            continue

        contract_bits = overrides.get("bits", bits)
        if contract_bits is not None:
            contract_bits = quant_bits_width(_normalize_quant_bits(contract_bits, format_value=format_value))
        else:
            contract_bits = bits

        contract = {
            "bits": contract_bits,
            "group_size": overrides.get("group_size", group_size),
            "desc_act": overrides.get("desc_act", desc_act),
            "sym": overrides.get("sym", sym),
            "pack_dtype": overrides.get("pack_dtype", pack_dtype),
        }
        key = (
            contract["bits"],
            contract["group_size"],
            contract["desc_act"],
            contract["sym"],
            contract["pack_dtype"],
        )
        if key in seen:
            continue
        seen.add(key)
        yield contract


def iter_quant_linear_kernels() -> List[Type[BaseQuantLinear]]:
    kernels = []
    seen = set()

    def _walk(cls):
        for subcls in cls.__subclasses__():
            if subcls in seen:
                continue
            seen.add(subcls)
            _walk(subcls)
            if (
                "SUPPORTS_FORMATS" in subcls.__dict__
                and getattr(subcls, "SUPPORTS_BACKEND_SELECTION", True)
            ):
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
    backend = normalize_backend(backend, quant_method=quant_method)
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
        except (ImportError, OSError) as exc:
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


def _validate_selector_device_families(devices: Sequence[SelectorDevice]) -> None:
    families = {_device_family(device) for device in devices if device != DEVICE.ALL}
    if len(families) > 1:
        raise ValueError(
            "Quantized kernel selection does not support a device map spanning "
            f"different accelerator families: {', '.join(sorted(f.value for f in families))}."
        )

    device_capabilities = [
        (device, _device_capability(device))
        for device in devices
        if _device_family(device) in (DEVICE.CUDA, DEVICE.ROCM)
    ]
    capabilities = {capability for _, capability in device_capabilities if capability is not None}
    if len(capabilities) > 1:
        rendered = ", ".join(
            f"{device}=" + ".".join(str(part) for part in capability)
            for device, capability in device_capabilities
            if capability is not None
        )
        raise ValueError(
            "Quantized kernel selection does not support a device map spanning "
            f"different GPU compute capabilities: {rendered}."
        )


def _accelerate_keyword_device(accelerator) -> SelectorDevice | tuple[SelectorDevice, ...]:
    if accelerator is None:
        return DEVICE.CPU

    family = _device_family(torch.device(accelerator.type))
    return expand_selector_device_family(family)


def hf_normalize_device_device_map(
    device: Optional[Union[str, torch.device]],
    device_map: Optional[Union[str, Dict]],
) -> SelectorDevice | tuple[SelectorDevice, ...]:
    return normalize_device_device_map(device=device, device_map=device_map, default=DEVICE.CPU)


def normalize_device_device_map(
    device: Optional[Union[str, int, torch.device, DEVICE]],
    device_map: Optional[Union[str, Dict]],
    default: Optional[SelectorDevice] = None,
) -> SelectorDevice | tuple[SelectorDevice, ...]:
    normalized_device: Optional[SelectorDevice] = default
    accelerator = torch.accelerator.current_accelerator()
    if device is None:
        if device_map is not None:
            if isinstance(device_map, str):
                if _is_accelerate_device_map_keyword(device_map):
                    return _accelerate_keyword_device(accelerator)
                devices = (device_map,)
            else:
                # Preserve map order and concrete ordinals.  A set here would
                # make cuda:0/cuda:1 indistinguishable from each other.
                devices = tuple(device_map.values())
            normalized_devices = []
            for map_device in devices:
                if map_device is None:
                    continue
                if isinstance(map_device, str):
                    if _is_accelerate_device_map_keyword(map_device) or map_device == "auto":
                        return _accelerate_keyword_device(accelerator)
                    if _is_accelerate_offload_target(map_device):
                        continue
                candidate = _as_selector_device(map_device)
                if candidate not in normalized_devices:
                    normalized_devices.append(candidate)

            # CPU offload entries do not determine the accelerator kernel.  If
            # all entries are CPU, retain CPU as the target.
            accelerator_devices = [
                candidate for candidate in normalized_devices
                if _device_family(candidate) != DEVICE.CPU
            ]
            selected_devices = accelerator_devices or normalized_devices
            if selected_devices:
                _validate_selector_device_families(selected_devices)
                normalized_device = (
                    selected_devices[0]
                    if len(selected_devices) == 1
                    else tuple(selected_devices)
                )
    else:
        if isinstance(device, int):
            # Public loader arguments use the historical active-accelerator
            # meaning for integers; only device-map values are CUDA ordinals.
            normalized_device = normalize_device(device)
        else:
            normalized = _selector_devices(device)
            assert normalized is not None
            _validate_selector_device_families(normalized)
            normalized_device = normalized[0] if len(normalized) == 1 else normalized

    # map fake cuda to actual rocm
    if normalized_device == DEVICE.CUDA and IS_ROCM:
        normalized_device = DEVICE.ROCM
    return normalized_device


def auto_select_device(
    device: Optional[SelectorDevices],
    backend: Optional[BACKEND],
) -> SelectorDevice | tuple[SelectorDevice, ...]:
    assert device is None or isinstance(device, (DEVICE, torch.device, str, int, tuple, list))
    assert backend is None or isinstance(backend, BACKEND)

    if device is None:
        # Backend-specific kernels should default to a compatible device class.
        if backend in (BACKEND.GPTQ_TORCH_FUSED, BACKEND.AWQ_TORCH_FUSED, BACKEND.TORCH_FUSED, BACKEND.TORCH_FUSED_AWQ):
            return DEVICE.XPU if HAS_XPU else DEVICE.CPU
        if HAS_CUDA:
            device = DEVICE.CUDA
        elif HAS_XPU:
            device = DEVICE.XPU
        elif HAS_NPU:
            device = DEVICE.NPU
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
    backend = normalize_backend(backend, quant_method=METHOD.GPTQ)

    if device_map is not None:
        device = hf_normalize_device_device_map(None, device_map)
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
    backend = normalize_backend(backend, quant_method=quant_method)

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
        device = hf_normalize_device_device_map(None, device_map)
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
        bits,
        group_size: int,
        desc_act: bool,
        sym: bool,
        device: Optional[SelectorDevices],
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
        is_sharded: bool = False,
) -> Union[Type[BaseQuantLinear], List[Type[BaseQuantLinear]]]:
    if isinstance(format, str):
        format = FORMAT(format.lower())
    if isinstance(quant_method, str):
        quant_method = METHOD(quant_method.lower())
    backend = normalize_backend(backend, quant_method=quant_method)
    if device is not None:
        if isinstance(device, int):
            device = normalize_device(device)
        else:
            targets = _selector_devices(device)
            assert targets is not None
            _validate_selector_device_families(targets)
            device = targets[0] if len(targets) == 1 else targets

    bits = quant_bits_width(_normalize_quant_bits(bits, format_value=format))

    supported_formats = BACKEND_TO_METHOD_FORMAT_MAPPING.get(quant_method)
    if supported_formats is None:
        raise ValueError(f"Unsupported quantization method: `{quant_method}`")
    if format not in supported_formats:
        raise ValueError(f"Unsupported format: `{format}` for quantization method `{quant_method}`")

    backend = BACKEND.AUTO if backend is None else backend

    trainable = backend == BACKEND.AUTO_TRAINABLE
    marlin_backends = {
        BACKEND.GPTQ_MARLIN,
        BACKEND.AWQ_MARLIN,
    }
    if not allow_marlin and backend in marlin_backends:
        raise ValueError("Marlin kernel selection was disabled by allow_marlin=False.")

    selector_targets = _selector_devices(device) if device is not None else None
    selector_families = (
        {_device_family(target) for target in selector_targets}
        if selector_targets is not None
        else set()
    )

    def validate_candidate(cls, **kwargs):
        # A kernel selected for a multi-GPU model must be valid on every target
        # ordinal.  Validate each exact target so device-sensitive kernels query
        # the correct capability rather than the process's current device.
        if selector_targets is None:
            return validate_quant_linear(cls, **kwargs, device=None)
        last_result = (True, None)
        for target in selector_targets:
            last_result = validate_quant_linear(cls, **kwargs, device=target)
            if not last_result[0]:
                return last_result
        return last_result

    validated_qlinears = []
    # Handle the case where backend is AUTO.
    if backend in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
        allow_quant_linears = list(AUTO_BACKEND_KERNEL_MAPPING[quant_method].get(format, {}).items())
        if not allow_quant_linears:
            raise ValueError(f"No auto-select kernels found for `{quant_method}` with format `{format}`.")

        last_err = None
        global message_logged
        # For multi-select (used by make_quant/create_quant_layer) we test each kernel
        # against the union of effective quant contracts (base + dynamic). This lets
        # mixed-bitwidth models pick different kernels per layer, e.g. a 4-bit-only
        # kernel for 4-bit dynamic layers while the base bits is 3. For single-select
        # we keep the original base contract with the full dynamic map so callers
        # asking for one representative kernel get a model-wide-compatible answer.
        if multi_select:
            contracts = list(_iter_dynamic_contracts(dynamic, bits, group_size, desc_act, sym, pack_dtype, format))
        for k, cls in allow_quant_linears:
            if not allow_marlin and marlin_backends.intersection(
                get_kernel_backends(cls) if getattr(cls, "SUPPORTS_BACKENDS", None) else ()
            ):
                if os.environ.get("DEBUG"):
                    log.info(f"skip {k} because Marlin selection is disabled")
                continue
            if (
                DEVICE.ALL not in cls.SUPPORTS_DEVICES
                and selector_families
                and not selector_families.intersection(set(cls.SUPPORTS_DEVICES))
            ):
                if os.environ.get("DEBUG"):
                    log.info(f"skip {k} for unsupported device `{device}`")
                continue
            supports_sharded_load = getattr(
                cls, "SUPPORTS_SHARDED_LOAD", getattr(cls, "SUPPORTS_SHARDS", True)
            )
            if is_sharded and not supports_sharded_load:
                if os.environ.get("DEBUG"):
                    log.info(f"skip {k} because sharded checkpoints are not supported")
                continue

            validated = False
            contract_err = None
            if multi_select:
                for contract in contracts:
                    validated, contract_err = validate_candidate(cls,
                        bits=contract["bits"],
                        group_size=contract["group_size"],
                        desc_act=contract["desc_act"],
                        sym=contract["sym"],
                        pack_dtype=contract["pack_dtype"],
                        dtype=dtype,
                        dynamic=None,
                        trainable=trainable,
                        adapter=adapter,
                    )
                    if validated:
                        break
            else:
                validated, contract_err = validate_candidate(cls,
                    bits=bits,
                    group_size=group_size,
                    desc_act=desc_act,
                    sym=sym,
                    pack_dtype=pack_dtype,
                    dtype=dtype,
                    dynamic=dynamic,
                    trainable=trainable,
                    adapter=adapter,
                )
            if not validated:
                last_err = contract_err
                if os.environ.get("DEBUG"):
                    log.info(f"skip {k} for {str(contract_err)}")
                continue

            if pack:
                if _supports_pack_api(cls):
                    log.info(f"{'Packing ' if pack else ''}Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                    validated_qlinears.append(cls)
                    if not multi_select:
                        log.info(f"Kernel: selected -> `{cls.__name__}`.")
                        return cls
            else:
                log.info(f"{'Packing ' if pack else ''}Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                validated_qlinears.append(cls)
                if not multi_select:
                    log.info(f"Kernel: selected -> `{cls.__name__}`.")
                    return cls

        if len(validated_qlinears) == 0:
            if last_err:
                raise last_err
            raise ValueError("No valid quant linear")

        return validated_qlinears

    # TODO check AWQ format supports BACKEND

    # Handle the case where backend is not AUTO.
    qlinear = get_kernel_for_backend(backend, quant_method, format)

    supports_sharded_load = getattr(
        qlinear, "SUPPORTS_SHARDED_LOAD", getattr(qlinear, "SUPPORTS_SHARDS", True)
    )
    if is_sharded and not supports_sharded_load:
        raise ValueError(f"Selected backend `{backend}` with kernel `{qlinear.__name__}` does not support sharded checkpoints.")

    validate, err = validate_candidate(qlinear,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        pack_dtype=pack_dtype,
        dtype=dtype,
        dynamic=dynamic,
        trainable=trainable,
    )

    log.info(f"{'Packing ' if pack else ''}Kernel: selected: `{qlinear.__name__}`")
    if not validate:
        raise ValueError(err)

    if pack:
        if not _supports_pack_api(qlinear):
            raise ValueError(
                f"Selected backend `{backend}` with kernel `{qlinear.__name__}` cannot pack quantized weights for format `{format}`."
            )

    if multi_select:
        return [qlinear]
    return qlinear
