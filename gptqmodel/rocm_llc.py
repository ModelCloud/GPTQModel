# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Expose the AMD last-level cache (Infinity Cache / MALL) to PyTorch.

Why this exists
---------------
``torch.cuda.get_device_properties()`` reports only ``L2_cache_size``. On
RDNA3/RDNA4 that number is correct but incomplete: the R9700, for example, has
8 MiB of L2 *and* a 64 MiB memory-side last-level cache that AMD markets as
Infinity Cache (the LLVM/DRM name is MALL). HIP has no field for it --
``hipDeviceProp_t`` carries ``l2CacheSize`` and nothing else, and
``hipDeviceGetAttribute`` has no LLC enum -- so the value never reaches torch.

Anything that sizes a cache-flush buffer from ``L2_cache_size`` therefore
flushes 8 MiB against a 64 MiB cache and measures a warm cache. Inductor does
exactly that.

What this patches
-----------------
This is the Python half of https://github.com/pytorch/pytorch/pull/203965
("[ROCm] Expose last-level cache size and use read-based benchmark flushing"):

1. ``_CudaDeviceProperties.last_level_cache_size`` -- a new property that
   queries HSA's byte-valued cache array and falls back to ``L2_cache_size``.
   ``L2_cache_size`` itself is left untouched; it was never wrong.
2. ``InductorBenchmarker.get_device_cache_size`` -- use ``max(L2, LLC)`` so a
   flush buffer actually covers the outermost cache.
3. ``BenchmarkRequest.do_bench_with_cudagraphs`` -- same ``max(L2, LLC)`` when
   sizing the rotated CUDA-graph input pool.

The PR's fourth part (reading rather than writing the ROCm flush buffer) is not
back-ported: the installed Triton path already uses a 256 MiB flush on ROCm,
which covers the 64 MiB LLC, and the read-vs-write change only affects the
reported timing, not which kernel is selected.

The C++ property cannot be monkeypatched, so (1) attaches a Python property to
the pybind class. That is enough for every Python consumer, including (2) and
(3).

Usage
-----
    from gptqmodel.utils import rocm_llc

    rocm_llc.apply()                    # idempotent, never raises
    print(rocm_llc.cache_levels(0))     # (L1, L2, L3, L4) in bytes, or None
    print(rocm_llc.last_level_cache_size(0))

Or inspect the hardware without patching anything::

    python -m gptqmodel.rocm_llc

Everything here is a *hint*. A reported size is not a reservation, and MALL
residency is not pinnable; measured throughput still has to be verified.
"""

from __future__ import annotations

import ctypes
import inspect
import logging
import os
import re
import textwrap
import threading
from typing import Any

import torch


log = logging.getLogger(__name__)

# HSA constants (hsa.h / hsa_ext_amd.h).
_HSA_STATUS_SUCCESS = 0
_HSA_DEVICE_TYPE_GPU = 1
_HSA_AGENT_INFO_DEVICE = 17
_HSA_AGENT_INFO_CACHE_SIZE = 18
_HSA_AMD_AGENT_INFO_UUID = 0xA011

_UUID_PREFIX = b"GPU-"
_UUID_LEN = 16
_HEX = frozenset(b"0123456789abcdefABCDEF")
_UNKNOWN_UUID = b"0" * _UUID_LEN

_LIB_NAMES = ("libhsa-runtime64.so.1", "libhsa-runtime64.so", None)


class _HsaAgent(ctypes.Structure):
    """``hsa_agent_t`` is a single uint64 handle passed by value."""

    _fields_ = [("handle", ctypes.c_uint64)]


_ITERATE_CB = ctypes.CFUNCTYPE(ctypes.c_int, _HsaAgent, ctypes.c_void_p)


class _HsaApi:
    """Minimal dynamic binding to the HSA runtime HIP has already loaded."""

    def __init__(self) -> None:
        self._init = None
        for name in _LIB_NAMES:
            try:
                lib = ctypes.CDLL(name) if name is not None else ctypes.CDLL(None)
                init = lib.hsa_init
                shut_down = lib.hsa_shut_down
                iterate = lib.hsa_iterate_agents
                get_info = lib.hsa_agent_get_info
            except (OSError, AttributeError):
                continue
            init.restype = ctypes.c_int
            shut_down.restype = ctypes.c_int
            iterate.restype = ctypes.c_int
            iterate.argtypes = [_ITERATE_CB, ctypes.c_void_p]
            get_info.restype = ctypes.c_int
            get_info.argtypes = [_HsaAgent, ctypes.c_int, ctypes.c_void_p]
            self._init, self._shut_down = init, shut_down
            self._iterate, self._get_info = iterate, get_info
            # Keep the CDLL alive for the lifetime of the process.
            self._lib = lib
            break
        if self._init is None:
            raise OSError("libhsa-runtime64 is not loadable")

    def query(self, uuid: bytes) -> tuple[int, int, int, int] | None:
        """Return ``(L1, L2, L3, L4)`` bytes for the agent whose UUID matches.

        ``None`` means "not reported": an HSA failure, no match, an ambiguous
        match, or a UUID that is not the 16 hex characters HIP stores. The
        caller treats that as unknown rather than guessing a capacity.
        """

        if self._init() != _HSA_STATUS_SUCCESS:
            return None
        found: list[tuple[int, int, int, int]] = []
        prefix_len = len(_UUID_PREFIX)

        def visit(agent: _HsaAgent, _data: Any) -> int:
            device = ctypes.c_int(0)
            if self._get_info(agent, _HSA_AGENT_INFO_DEVICE, ctypes.byref(device)) != _HSA_STATUS_SUCCESS:
                return _HSA_STATUS_SUCCESS
            if device.value != _HSA_DEVICE_TYPE_GPU:
                return _HSA_STATUS_SUCCESS
            buf = ctypes.create_string_buffer(prefix_len + _UUID_LEN + 1)
            if self._get_info(agent, _HSA_AMD_AGENT_INFO_UUID, buf) != _HSA_STATUS_SUCCESS:
                return _HSA_STATUS_SUCCESS
            value = buf.value
            if not value.startswith(_UUID_PREFIX):
                return _HSA_STATUS_SUCCESS
            if value[prefix_len : prefix_len + _UUID_LEN] != uuid:
                return _HSA_STATUS_SUCCESS
            sizes = (ctypes.c_uint32 * 4)()
            if self._get_info(agent, _HSA_AGENT_INFO_CACHE_SIZE, sizes) != _HSA_STATUS_SUCCESS:
                return _HSA_STATUS_SUCCESS
            found.append(tuple(int(s) for s in sizes))  # type: ignore[arg-type]
            return _HSA_STATUS_SUCCESS

        try:
            status = self._iterate(_ITERATE_CB(visit), None)
        finally:
            # HSA init is reference counted; drop only this reference.
            self._shut_down()
        if status != _HSA_STATUS_SUCCESS or len(found) != 1:
            return None
        return found[0]


_api_lock = threading.Lock()
_api: _HsaApi | None = None
_api_error: str | None = None
_levels: dict[bytes, tuple[int, int, int, int] | None] = {}


def _get_api() -> _HsaApi | None:
    global _api, _api_error
    if _api is not None:
        return _api
    if _api_error is not None:
        return None
    with _api_lock:
        if _api is None and _api_error is None:
            try:
                _api = _HsaApi()
            except OSError as exc:
                _api_error = str(exc)
                log.debug("HSA runtime unavailable: %s", exc)
    return _api


def _device_uuid(device: int | str | torch.device | None) -> bytes | None:
    """The 16 hex characters HIP stores in ``hipUUID::bytes``."""

    try:
        props = torch.cuda.get_device_properties(device)
        raw = bytes(props.uuid.bytes)
    except Exception:  # noqa: BLE001 - a missing uuid must not be fatal
        return None
    if len(raw) != _UUID_LEN or not set(raw) <= _HEX or raw == _UNKNOWN_UUID:
        return None
    return raw


def cache_levels(device: int | str | torch.device | None = None) -> tuple[int, int, int, int] | None:
    """``(L1, L2, L3, L4)`` in bytes for ``device``, or ``None`` if unknown."""

    if not torch.version.hip:
        return None
    uuid = _device_uuid(device)
    if uuid is None:
        return None
    with _api_lock:
        if uuid in _levels:
            return _levels[uuid]
    api = _get_api()
    if api is None:
        return None
    # Query outside the lock; HSA is slow the first time and the cache write
    # below is idempotent if two threads race.
    result = api.query(uuid)
    with _api_lock:
        _levels.setdefault(uuid, result)
        return _levels[uuid]


def last_level_cache_size(device: int | str | torch.device | None = None) -> int:
    """Outermost reported cache in bytes, falling back to L2.

    Prefers L4 over L3 when both are reported, matching the upstream PR: the
    outermost level wins even if an inner level is larger.
    """

    try:
        props = torch.cuda.get_device_properties(device)
        l2 = int(props.L2_cache_size)
    except Exception:  # noqa: BLE001
        return 0
    levels = cache_levels(device)
    if not levels:
        return l2
    for size in (levels[3], levels[2]):
        if size > 0:
            return size
    return l2


# ---------------------------------------------------------------------------
# Monkeypatches
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {"applied": False, "restore": [], "status": {}}


def _patched_property() -> property:
    def fget(self: Any) -> int:
        try:
            return last_level_cache_size()
        except Exception:  # noqa: BLE001 - a property must never raise
            return int(getattr(self, "L2_cache_size", 0))

    return property(fget)


def _patch_props() -> str:
    cls = getattr(torch._C, "_CudaDeviceProperties", None)
    if cls is None:
        return "skipped: no _CudaDeviceProperties"
    try:
        had = "last_level_cache_size" in getattr(cls, "__dict__", {})
        previous = getattr(cls, "last_level_cache_size", None)

        def restore() -> None:
            if had:
                setattr(cls, "last_level_cache_size", previous)
            else:
                try:
                    delattr(cls, "last_level_cache_size")
                except AttributeError:
                    pass

        _STATE["restore"].append(restore)
        setattr(cls, "last_level_cache_size", _patched_property())
    except (AttributeError, TypeError) as exc:
        return f"skipped: {exc}"
    return "already provided by torch, overridden" if had else "applied"


def _rebind(cls: Any, name: str, replacement: Any, original: Any) -> None:
    _STATE["restore"].append(lambda: setattr(cls, name, original))
    setattr(cls, name, replacement)


def _patch_benchmarker() -> str:
    """Use ``max(L2, LLC)`` for the Inductor benchmark flush buffer.

    Replaced as a whole function rather than by rewriting source: the body is
    short and only depends on two helpers that have been stable for years.
    """

    from torch._dynamo.device_interface import get_interface_for_device

    try:
        from torch._inductor.runtime.benchmarking import (
            InductorBenchmarker,
            _normalize_gpu_device_type,
        )
    except Exception as exc:  # noqa: BLE001
        return f"skipped: {exc}"

    original = getattr(InductorBenchmarker, "get_device_cache_size", None)
    if original is None:
        return "skipped: get_device_cache_size not found"

    def get_device_cache_size(self: Any, device_type: Any = None) -> int:
        if "L2_cache_size" in self.__dict__:
            return self.__dict__["L2_cache_size"]
        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)
        device = device_interface.current_device()
        props = device_interface.get_device_properties(device)
        # The outermost cache need not have the largest capacity.
        cache_size = max(
            getattr(props, "L2_cache_size", 0) or 0,
            getattr(props, "last_level_cache_size", 0) or 0,
        )
        return cache_size or 256 * 1024 * 1024

    get_device_cache_size.__doc__ = original.__doc__
    _rebind(InductorBenchmarker, "get_device_cache_size", get_device_cache_size, original)
    return "applied"


_CUDAGRAPH_RE = re.compile(
    r"(?P<i>[ \t]*)cache_size = next\(\s*\(\s*"
    r'getattr\(props, attr\)\s*for attr in \("L2_cache_size", "last_level_cache_size"\)\s*'
    r"if getattr\(props, attr, None\)\s*\),\s*256 \* 1024 \* 1024,\s*\)",
    re.S,
)


def _cudagraph_replacement(match: re.Match[str]) -> str:
    i = match.group("i")
    return (
        f"{i}cache_size = (\n"
        f'{i}    max(\n'
        f'{i}        getattr(props, "L2_cache_size", 0) or 0,\n'
        f'{i}        getattr(props, "last_level_cache_size", 0) or 0,\n'
        f"{i}    )\n"
        f"{i}    or 256 * 1024 * 1024\n"
        f"{i})"
    )


def _patch_cudagraph() -> str:
    """Size the rotated CUDA-graph input pool from ``max(L2, LLC)``.

    The expression lives inside a ~150 line method, and the surrounding code is
    version specific, so this is a source rewrite. The owning class is located
    by content rather than by name: the public ``do_bench_with_cudagraphs`` on
    the request classes is an abstract stub that raises ``NotImplementedError``,
    while the implementation sits on ``GPUDeviceBenchmarkMixin`` and is shared
    by every GPU request subclass. Patching the mixin once covers them all.
    """

    try:
        import torch._inductor.autotune_process as module
    except Exception as exc:  # noqa: BLE001
        return f"skipped: {exc}"

    owners: list[tuple[Any, Any, str]] = []
    for obj in vars(module).values():
        if not inspect.isclass(obj):
            continue
        fn = obj.__dict__.get("do_bench_with_cudagraphs")
        if fn is None:
            continue
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            continue
        if "cache_size = next(" in src:
            owners.append((obj, fn, src))
    if not owners:
        return "skipped: torch source changed"

    namespace = dict(vars(module))
    patched_any = False
    for cls, original, src in owners:
        if src.lstrip().startswith("@"):
            continue
        patched, count = _CUDAGRAPH_RE.subn(_cudagraph_replacement, src)
        if count != 1:
            continue
        try:
            exec(compile(patched, f"<rocm_llc {module.__name__}.{cls.__name__}>", "exec"), namespace)
            replacement = namespace["do_bench_with_cudagraphs"]
        except Exception as exc:  # noqa: BLE001 - never break the caller
            log.debug("rocm_llc: cudagraph rewrite failed: %s", exc)
            continue
        _rebind(cls, "do_bench_with_cudagraphs", replacement, original)
        patched_any = True
    return "applied" if patched_any else "skipped: torch source changed"


def apply(*, benchmarker: bool = True, cudagraph: bool = True) -> dict[str, str]:
    """Install the back-port. Idempotent, cheap, and never raises.

    Returns a mapping of component to ``"applied"``, ``"already ..."`` or
    ``"skipped: <reason>"`` so a caller can log exactly what took effect.
    """

    if _STATE["applied"]:
        return dict(_STATE["status"])

    status: dict[str, str] = {}
    if os.environ.get("GPTQMODEL_NO_ROCM_LLC", "").strip().lower() in {"1", "true", "yes", "on"}:
        status["props"] = "skipped: GPTQMODEL_NO_ROCM_LLC"
        _STATE.update(applied=True, status=status)
        return status
    if not torch.version.hip:
        status["props"] = "skipped: not ROCm"
        _STATE.update(applied=True, status=status)
        return status

    status["props"] = _patch_props()
    if benchmarker:
        status["benchmarker"] = _patch_benchmarker()
    if cudagraph:
        status["cudagraph"] = _patch_cudagraph()

    _STATE.update(applied=True, status=status)
    for name, result in status.items():
        if result.startswith("skipped") and "not ROCm" not in result:
            log.debug("rocm_llc: %s %s", name, result)
    return status


def unapply() -> None:
    """Undo every patch. Intended for tests."""

    for restore in reversed(_STATE["restore"]):
        try:
            restore()
        except Exception:  # noqa: BLE001
            pass
    _STATE.update(applied=False, restore=[], status={})


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Report AMD cache geometry as torch sees it.")
    parser.add_argument("--apply", action="store_true", help="install the back-port and show the result")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible")
        return 1
    if torch.version.hip is None:
        print("not a ROCm build; last-level cache is reported as L2 by torch")
        return 0

    if args.apply:
        for name, result in apply().items():
            print(f"patch {name}: {result}")

    print(f"{'dev':>3} {'name':>28} {'L1':>9} {'L2':>10} {'L3 (MALL)':>11} {'L4':>10} {'torch LLC':>11}")
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        levels = cache_levels(device)
        shown = levels or (0, int(props.L2_cache_size), 0, 0)
        mib = [f"{value / 2**20:.0f} MiB" if value else "-" for value in shown]
        llc = getattr(props, "last_level_cache_size", None)
        llc_text = f"{int(llc) / 2**20:.0f} MiB" if llc else "missing"
        print(f"{device:>3} {props.name[:28]:>28} {mib[0]:>9} {mib[1]:>10} {mib[2]:>11} {mib[3]:>10} {llc_text:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
