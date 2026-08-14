# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Process-wide, cached NVML memory sampling without ``nvidia-smi`` subprocesses."""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Dict, Optional

import torch


_NVML_CACHE_SECONDS = 1.0
_NVML_LOCK = threading.Lock()
_NVML_LIBRARY = None
_NVML_INITIALIZED = False
_NVML_LAST_QUERY = float("-inf")
_NVML_LAST_RESULT: Optional[Dict[str, int]] = None


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load_nvml():
    global _NVML_INITIALIZED, _NVML_LIBRARY
    if _NVML_INITIALIZED:
        return _NVML_LIBRARY
    _NVML_INITIALIZED = True
    try:
        library = ctypes.CDLL("libnvidia-ml.so.1")
        library.nvmlInit_v2.restype = ctypes.c_int
        library.nvmlDeviceGetHandleByPciBusId_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        library.nvmlDeviceGetHandleByPciBusId_v2.restype = ctypes.c_int
        library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
        library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        if library.nvmlInit_v2() != 0:
            return None
        _NVML_LIBRARY = library
    except (AttributeError, OSError):
        _NVML_LIBRARY = None
    return _NVML_LIBRARY


def _cuda_pci_bus_id(index: int) -> bytes:
    properties = torch.cuda.get_device_properties(index)
    return (
        f"{properties.pci_domain_id:08X}:{properties.pci_bus_id:02X}:"
        f"{properties.pci_device_id:02X}.0"
    ).encode("ascii")


def cuda_memory_used_snapshot(*, cache_seconds: float = _NVML_CACHE_SECONDS) -> Optional[Dict[str, int]]:
    """Return logical CUDA device memory usage in bytes, cached process-wide.

    NVML is queried directly through its stable C ABI. The single global lock
    prevents concurrent refreshes under free-threaded Python, while the TTL
    makes every lifecycle processor reuse one host-wide sample.
    """

    global _NVML_LAST_QUERY, _NVML_LAST_RESULT
    if cache_seconds < 0:
        raise ValueError("NVML cache duration must be nonnegative")
    with _NVML_LOCK:
        now = time.monotonic()
        if now - _NVML_LAST_QUERY < cache_seconds:
            return None if _NVML_LAST_RESULT is None else dict(_NVML_LAST_RESULT)
        _NVML_LAST_QUERY = now
        library = _load_nvml()
        if library is None or not torch.cuda.is_available():
            _NVML_LAST_RESULT = None
            return None
        result: Dict[str, int] = {}
        try:
            for index in range(torch.cuda.device_count()):
                handle = ctypes.c_void_p()
                if library.nvmlDeviceGetHandleByPciBusId_v2(_cuda_pci_bus_id(index), ctypes.byref(handle)) != 0:
                    raise RuntimeError("NVML could not resolve a CUDA PCI bus id")
                memory = _NvmlMemory()
                if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                    raise RuntimeError("NVML memory query failed")
                result[f"cuda:{index}"] = int(memory.used)
        except (AttributeError, RuntimeError):
            _NVML_LAST_RESULT = None
            return None
        _NVML_LAST_RESULT = result
        return dict(result)


def _reset_nvml_cache_for_tests() -> None:
    global _NVML_LAST_QUERY, _NVML_LAST_RESULT
    with _NVML_LOCK:
        _NVML_LAST_QUERY = float("-inf")
        _NVML_LAST_RESULT = None


__all__ = ["cuda_memory_used_snapshot"]
