# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Best-effort GPU peer-access (P2P) detection, enablement and safe copies.

PyTorch exposes ``torch.cuda.can_device_access_peer`` but no enablement API, so
enablement goes through ``ctypes``: ``hipDeviceEnablePeerAccess`` on ROCm and
``cuCtxEnablePeerAccess`` on CUDA.  ``hipErrorPeerAccessAlreadyEnabled`` /
``CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`` (704) is treated as success, which
makes repeated calls idempotent.

Peer access is *directed*: ``i -> j`` and ``j -> i`` are enabled separately, and
both directions are queried.

PyTorch already enables peer access lazily inside its own copy path
(``c10/cuda/PeerToPeerAccess.cpp`` -> ``CUDACachingAllocator::enablePeerAccess``),
so this module is **not** required for ``copy_``/``to`` to stay on the
interconnect.  Its job is to (a) enable the links before the first cross-device
copy so the very first transfer is direct, (b) route every cross-device move
through a direction-aware helper, and (c) optionally *diagnose* the links.
Enabling is cheap and idempotent; bandwidth probing moves real payload and is
opt-in through :class:`P2PConfig` (``probe=True``).

A capability bit is not proof of usable bandwidth, so an opt-in probe classifies
each direction as ``enabled``/``degraded``.  ``can_copy(src, dst)`` answers for
one direction only.  On some ROCm topologies the link is *asymmetric*: the
capability query reports ``0 -> 1`` unsupported while ``1 -> 0`` is enabled, and
PyTorch's direct ``tensor.to("cuda:1")`` then calls ``hipMemcpyPeerAsync`` and
segfaults the process instead of raising.  :func:`safe_move_to` therefore stages
that direction through pinned host memory, which is the only path the driver
does not crash on.  Never assert P2P is effective from a single direction or
from a global "some link works" verdict.

Behavior is controlled by :class:`P2PConfig`, normally supplied as
``BaseQuantizeConfig.p2p`` (or ``GPTQModel.load(..., p2p=...)`` for inference)
and installed with :func:`configure_p2p`.  Results are cached per device tuple
for the process.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import glob
import os
import site
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import torch

from .logger import setup_logger
from .torch import as_torch_device


log = setup_logger()

# "auto" enables P2P when the runtime reports a peer link; "on" forces the
# enablement attempt; "off" skips the pre-enablement call entirely (PyTorch's
# own lazy peer setup still runs inside its copy path).
_MODE_AUTO = "auto"
_MODE_ON = "on"
_MODE_OFF = "off"
_OFF_VALUES = frozenset({"0", "false", "off", "no", "disable", "disabled"})
_ON_VALUES = frozenset({"1", "true", "on", "yes", "enable", "enabled"})

DEFAULT_MIN_GBPS = 5.0
# A direct link must not be slower than the fallback. The reference is an
# explicit, correctly ordered ``src -> pinned CPU -> dst`` pipeline: it pays for
# *two* transfers (D2H then H2D) and is therefore strictly slower than a direct
# copy at the same link rate -- never at parity.
#
# Measured on this host (64 MiB payload, ordered staging, double-synchronized
# timer, median of 7 samples):
#
#   R9700 gfx1201 (2x AMD, PCIe)       direct 52.99 vs staged 27.35 GB/s (1.94x)
#   RTX 4090 sm_89  (2x NVIDIA, PCIe)  direct 26.23 vs staged 13.27 GB/s (1.98x)
#
# Earlier 0.96x "parity" numbers in this tree came from issuing the D2H and H2D
# legs on two device streams with no dependency, which measured the staged path
# at ~2x its real speed. With the legs ordered the direct path wins by ~2x, so a
# 1.0 gate keeps every healthy PCIe peer link direct while still rejecting a
# link that is genuinely slower than the fallback. ``P2PConfig.min_ratio``
# can be lowered for a topology that really does measure at parity.
DEFAULT_MIN_RATIO = 1.0


@dataclass
class P2PConfig:
    """Peer-to-peer policy for multi-GPU quantization and inference.

    Pass an instance as ``BaseQuantizeConfig(p2p=P2PConfig(...))``, or as
    ``GPTQModel.load(..., p2p=P2PConfig(...))`` when loading an already
    quantized model. A plain ``dict`` with the same keys is also accepted.

    ``mode``
        ``"auto"`` enables peer access when the runtime reports a peer link,
        ``"on"`` additionally attempts pairs whose capability bit is false, and
        ``"off"`` skips GPT-QModel's pre-enablement (PyTorch still enables
        peers lazily inside its own copy path).
    ``probe``
        Measure every directed pair and classify a link ``degraded`` when it is
        slower than ordered pinned-host staging. Diagnostic only; peer access
        stays enabled and no caller inserts a host relay because of it.
    ``probe_mib``
        Preferred probe payload size in MiB; the probe steps down under
        memory pressure.
    ``min_gbps`` / ``min_ratio``
        Diagnostic gates: absolute floor in GB/s, and the fraction of the
        ordered host-staging rate a direct copy must reach.
    ``use_torch_replicate``
        Allow ``torch.nn.parallel.replicate`` when the GIL is enabled. It is
        off by default and skipped on free-threaded builds regardless because
        it funnels into ``torch.cuda.comm.broadcast_coalesced``, which is not
        thread-safe.

    Peer access is an accelerator concept: this config only affects CUDA/HIP
    device-to-device copies. It is a no-op for CPU-only runs and for
    CPU<->CPU transfers, which never take the peer path (no warning is
    emitted, and the CPU move helpers fall back to their regular copy).
    """

    mode: str = _MODE_AUTO
    probe: bool = False
    probe_mib: int = 64
    min_gbps: float = DEFAULT_MIN_GBPS
    min_ratio: float = DEFAULT_MIN_RATIO
    use_torch_replicate: bool = False

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode in _OFF_VALUES:
            self.mode = _MODE_OFF
        elif mode in _ON_VALUES:
            self.mode = _MODE_ON
        elif mode in ("", "auto", "default", "none"):
            self.mode = _MODE_AUTO
        else:
            raise ValueError(
                f"P2PConfig.mode must be 'auto', 'on' or 'off'; got {self.mode!r}"
            )

        self.probe = bool(self.probe)
        self.use_torch_replicate = bool(self.use_torch_replicate)
        try:
            self.probe_mib = max(1, int(self.probe_mib))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"P2PConfig.probe_mib must be an integer; got {self.probe_mib!r}") from exc
        try:
            self.min_gbps = max(0.0, float(self.min_gbps))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"P2PConfig.min_gbps must be a number; got {self.min_gbps!r}") from exc
        try:
            self.min_ratio = max(0.0, float(self.min_ratio))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"P2PConfig.min_ratio must be a number; got {self.min_ratio!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_p2p_config(config: P2PConfig | Mapping[str, Any] | None) -> P2PConfig:
    if config is None:
        return P2PConfig()
    if isinstance(config, P2PConfig):
        return config
    if isinstance(config, Mapping):
        return P2PConfig(**dict(config))
    raise TypeError(
        "p2p config must be a P2PConfig, a mapping of its fields, or None; "
        f"got {type(config).__name__}"
    )


_CONFIG_LOCK = threading.Lock()
_ACTIVE_CONFIG: P2PConfig = P2PConfig()


def configure_p2p(config: P2PConfig | Mapping[str, Any] | None) -> P2PConfig:
    """Install the process-wide P2P policy used by every helper in this module.

    Quantization calls this from the loopers with ``BaseQuantizeConfig.p2p``;
    inference calls it with the ``p2p=`` argument of ``GPTQModel.load``. Calling
    it again replaces the policy and drops cached probe results. ``None``
    restores the defaults, so callers never need environment variables. The
    policy has no effect on CPU-only runs: CPU<->CPU transfers never use peer
    access.
    """

    global _ACTIVE_CONFIG, _WARNED_DEGRADED

    resolved = _coerce_p2p_config(config)
    with _CONFIG_LOCK:
        changed = resolved != _ACTIVE_CONFIG
        _ACTIVE_CONFIG = resolved
    if changed:
        # A new policy is a new run as far as the operator is concerned; let the
        # actionable warning fire again so a second configuration is not silent.
        with _WARN_LOCK:
            _WARNED_DEGRADED = False
        clear_p2p_cache()
    return resolved


def get_p2p_config() -> P2PConfig:
    """Return the active P2P policy (never ``None``)."""

    return _ACTIVE_CONFIG


STATE_ENABLED = "enabled"
STATE_UNSUPPORTED = "unsupported"
STATE_DEGRADED = "degraded"
# ``P2PConfig(mode='on')`` can make ``hipDeviceEnablePeerAccess`` report success for
# a direction the runtime capability query denies. That is not evidence the
# direction is usable -- on the asymmetric ROCm rig the enable call fails with
# 101, but a driver that accepted it would still crash a direct copy -- so such
# a pair is never routed directly.
STATE_UNVERIFIED = "unverified"

_PEER_ALREADY_ENABLED = 704

_STATUS_CACHE: dict[tuple[str, bool, tuple[int, ...]], "PeerStatus"] = {}
_BANDWIDTH_CACHE: dict[tuple[int, int], tuple[float | None, float | None]] = {}
# Result of actually exercising the runtime peer-copy path for a directed pair:
# ``(usable, direct_gbps, host_staged_gbps, reason)``. Capability bits and
# ``copy_`` speed are not evidence that ``cudaMemcpyPeerAsync`` /
# ``hipMemcpyPeerAsync`` works; this cache records a real copy-and-compare.
_PEER_VALIDATION_CACHE: dict[
    tuple[int, int], tuple[bool, float | None, float | None, str]
] = {}
# Raw ``can_device_access_peer`` answers, cached per directed pair.  This is the
# fallback used by :func:`safe_move_to` when no probe status covers the pair.
_CAPABILITY_CACHE: dict[tuple[int, int], bool] = {}
# One warning per unsupported direction, not one per tensor move.
_STAGED_DIRECTIONS: set[tuple[int, int]] = set()
# Most recent probe result. ``safe_move_to`` uses it when the caller does not
# pass a status, so a measured ``degraded`` verdict also selects host staging.
_LAST_STATUS: "PeerStatus | None" = None

# One actionable warning per process: a multi-GPU run that cannot enable every
# ordered pair must not repeat the same diagnosis for every layer/module.
_WARNED_DEGRADED = False
_WARN_LOCK = threading.Lock()

# Runtime peer-copy entry point, loaded once. ``peer_copy_batch`` runs on worker
# threads, so guard the one-time initialization instead of reloading the shared
# library on every batch.
_PEER_COPY_RUNTIME: tuple[ctypes.CDLL, str] | None = None
_PEER_COPY_LOCK = threading.Lock()


@dataclass
class PeerStatus:
    """Result of a peer-access probe; one entry per directed device pair."""

    mode: str = _MODE_AUTO
    devices: tuple[int, ...] = ()
    pairs: dict[tuple[int, int], str] = field(default_factory=dict)
    bandwidth_gbps: dict[tuple[int, int], float | None] = field(default_factory=dict)
    host_staged_gbps: dict[tuple[int, int], float | None] = field(default_factory=dict)
    # Enablement failures, kept so ``strict=True`` can still raise when the
    # best-effort result was cached by an earlier automatic call.
    errors: dict[tuple[int, int], str] = field(default_factory=dict)
    reason: str = ""

    def state(self, src: int, dst: int) -> str:
        """Return the per-direction state (defaults to ``unsupported``)."""
        return self.pairs.get((src, dst), STATE_UNSUPPORTED)

    def can_copy(self, src: int, dst: int) -> bool:
        """True only when ``src -> dst`` is a healthy peer link.

        This is deliberately per-direction: an asymmetric topology can have
        ``can_copy(0, 1) is False`` while ``can_copy(1, 0) is True``, and the
        false direction must never be used for a direct transfer.
        """
        return self.state(src, dst) == STATE_ENABLED

    def asymmetric_pairs(self) -> list[tuple[int, int]]:
        """Directed pairs that are not usable while the reverse is enabled.

        Both ``unsupported`` and ``unverified`` mean GPT-QModel must not issue a
        direct copy in that direction, while the reverse can still be usable.
        Such a topology is the one that crashes PyTorch's direct peer copy on
        ROCm; callers use this to avoid collective/replicate paths that assume
        a symmetric link as well as to log one actionable warning.
        """
        unusable = (STATE_UNSUPPORTED, STATE_UNVERIFIED)
        return [
            (src, dst)
            for (src, dst), state in sorted(self.pairs.items())
            if state in unusable and self.pairs.get((dst, src)) == STATE_ENABLED
        ]

    def unverified_pairs(self) -> list[tuple[int, int]]:
        """Directions forced on with ``P2PConfig(mode='on')`` that the runtime denies.

        Enablement reported success but the capability query says the direction
        is not accessible, so the link is never used for a direct copy.
        """
        return [
            (src, dst)
            for (src, dst), state in sorted(self.pairs.items())
            if state == STATE_UNVERIFIED
        ]

    def summary(self) -> str:
        counts = {
            STATE_ENABLED: 0,
            STATE_UNSUPPORTED: 0,
            STATE_DEGRADED: 0,
            STATE_UNVERIFIED: 0,
        }
        for state in self.pairs.values():
            counts[state] = counts.get(state, 0) + 1
        paired = ", ".join(
            f"{src}->{dst}:{state}" for (src, dst), state in sorted(self.pairs.items())
        ) or "none"
        reason = f"; reason={self.reason}" if self.reason else ""
        unverified = (
            f" unverified={counts[STATE_UNVERIFIED]}"
            if counts[STATE_UNVERIFIED]
            else ""
        )
        return (
            f"P2P mode={self.mode} devices={list(self.devices)} "
            f"enabled={counts[STATE_ENABLED]} unsupported={counts[STATE_UNSUPPORTED]} "
            f"degraded={counts[STATE_DEGRADED]}{unverified} [{paired}]{reason}"
        )


def clear_p2p_cache() -> None:
    """Drop cached probe/bandwidth results (tests, device hot-plug)."""
    global _LAST_STATUS
    _STATUS_CACHE.clear()
    _BANDWIDTH_CACHE.clear()
    _PEER_VALIDATION_CACHE.clear()
    _CAPABILITY_CACHE.clear()
    _STAGED_DIRECTIONS.clear()
    _LAST_STATUS = None


def _cache_namespace() -> bool:
    """Cache bucket: probing and non-probing results must not be shared."""
    return probe_enabled()


def p2p_mode() -> str:
    """Return the requested mode: ``auto``, ``on`` (force), or ``off``."""
    return _ACTIVE_CONFIG.mode


def _min_gbps() -> float:
    return _ACTIVE_CONFIG.min_gbps


def _min_ratio() -> float:
    return _ACTIVE_CONFIG.min_ratio


def probe_enabled() -> bool:
    """True when bandwidth diagnostics were requested via ``P2PConfig.probe``.

    Probing is off by default: PyTorch already keeps cross-device copies on the
    peer link, so measuring every directed pair only adds startup latency and
    payload traffic without changing the transfer path.
    """

    return _ACTIVE_CONFIG.probe


def _probe_mib() -> int:
    return _ACTIVE_CONFIG.probe_mib


def _normalize_indices(devices: Iterable | None) -> tuple[int, ...]:
    """Resolve device-like values to accelerator ordinals.

    An indexless value (``"cuda"`` or ``torch.device("cuda")``) means "the
    current device", matching ``torch.device`` semantics; it must never default
    to ordinal 0. Non-accelerator values are skipped rather than mapped to a
    bogus ordinal.
    """

    if devices is None:
        return tuple(range(torch.cuda.device_count()))
    indices: list[int] = []
    for candidate in devices:
        if isinstance(candidate, bool):
            raise TypeError("device must not be a bool")
        if isinstance(candidate, int):
            index = candidate
        else:
            try:
                device = candidate if isinstance(candidate, torch.device) else torch.device(str(candidate))
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid device {candidate!r}") from exc
            if device.type != "cuda":
                continue
            index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def accelerator_devices_from_device_map(device_map: dict | None) -> list[torch.device]:
    """Return the distinct accelerator devices referenced by a device map.

    Accelerate accepts strings (``"cuda:1"``), integers (bare accelerator
    ordinals) and ``torch.device`` values; all three must be normalized, or an
    integer map silently yields no devices at all.
    """

    devices: list[torch.device] = []
    for target in (device_map or {}).values():
        if target is None or isinstance(target, bool):
            continue
        try:
            if isinstance(target, int):
                device = torch.device(f"cuda:{target}")
            elif isinstance(target, torch.device):
                device = target
            elif isinstance(target, str):
                device = torch.device(target)
            else:
                continue
        except (TypeError, RuntimeError, ValueError):
            continue
        if device.type != "cuda":
            continue
        if device not in devices:
            devices.append(device)
    return devices


def peer_matrix(devices: Iterable | None = None) -> list[list[bool]]:
    """Return the directed peer matrix; ``matrix[i][j]`` is ``i -> j``.

    ``i == i`` is reported as available (a device can address itself) so the
    matrix is reflexive, and it is *not* assumed symmetric because HIP/CUDA peer
    links are directed.
    """

    indices = _normalize_indices(devices)
    matrix: list[list[bool]] = []
    for src in indices:
        row: list[bool] = []
        for dst in indices:
            if src == dst:
                row.append(True)
                continue
            try:
                row.append(bool(torch.cuda.can_device_access_peer(src, dst)))
            except Exception as exc:  # pragma: no cover - runtime dependent
                log.debug("can_device_access_peer(%s, %s) failed: %s", src, dst, exc)
                row.append(False)
        matrix.append(row)
    return matrix


def _loaded_hip_runtime_path() -> str | None:
    """Return the HIP runtime PyTorch already mapped, if it is discoverable.

    Loading a *second* HIP runtime and enabling peer access there does not
    affect the runtime PyTorch uses. ``hipMemcpyPeerAsync`` then returns
    success while copying nothing (or faults), which is exactly the failure
    seen on the R9700 rig. Prefer the already-loaded library.
    """

    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as maps:
            for line in maps:
                if "libamdhip64.so" not in line:
                    continue
                candidate = line.split()[-1]
                if os.path.isfile(candidate):
                    return candidate
    except OSError:
        return None
    return None


def _hip_runtime_candidates():
    """Yield HIP runtime paths, preferring the one PyTorch is using."""

    loaded = _loaded_hip_runtime_path()
    if loaded is not None:
        yield loaded

    roots: list[str] = []
    with contextlib.suppress(Exception):
        roots.extend(site.getsitepackages())
    with contextlib.suppress(Exception):
        roots.append(site.getusersitepackages())
    with contextlib.suppress(Exception):
        roots.append(os.path.dirname(os.path.dirname(torch.__file__)))

    for root in dict.fromkeys(roots):
        if not root or not os.path.isdir(root):
            continue
        # ROCm wheels ship the runtime in ``_rocm_sdk_core/lib``; some builds
        # place it under ``torch/lib`` instead.
        for pattern in ("_rocm_sdk_core/lib/libamdhip64.so*", "torch/lib/libamdhip64.so*"):
            yield from sorted(glob.glob(os.path.join(root, pattern)))

    yield "libamdhip64.so"
    yield "libamdhip64.so.7"
    yield "libamdhip64.so.6"


def _load_hip():
    for name in dict.fromkeys(_hip_runtime_candidates()):
        try:
            lib = ctypes.CDLL(name)
            lib.hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
            lib.hipDeviceEnablePeerAccess.restype = ctypes.c_int
            lib.hipGetLastError.argtypes = []
            lib.hipGetLastError.restype = ctypes.c_int
            lib.hipMemcpyPeerAsync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            lib.hipMemcpyPeerAsync.restype = ctypes.c_int
            return lib
        except OSError:
            continue
    return None


def _load_cuda():
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        lib.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
        lib.cuCtxEnablePeerAccess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        lib.cuCtxEnablePeerAccess.restype = ctypes.c_int
        return lib
    return None


def _load_cudart():
    """Load the CUDA *runtime*, which owns ``cudaMemcpyPeerAsync``.

    ``_load_cuda`` returns the driver library (``cuCtxEnablePeerAccess``); the
    async peer copy lives in the runtime, which is usually shipped inside the
    PyTorch wheel rather than installed system-wide.
    """

    candidates = [
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    with contextlib.suppress(Exception):
        discovered = ctypes.util.find_library("cudart")
        if discovered:
            candidates.append(discovered)

    roots: list[str] = []
    with contextlib.suppress(Exception):
        roots.extend(site.getsitepackages())
    with contextlib.suppress(Exception):
        roots.append(site.getusersitepackages())
    with contextlib.suppress(Exception):
        roots.append(os.path.dirname(os.path.dirname(torch.__file__)))

    for root in dict.fromkeys(roots):
        if not root or not os.path.isdir(root):
            continue
        candidates.extend(
            sorted(glob.glob(os.path.join(root, "nvidia", "**", "libcudart.so*"), recursive=True))
        )
        candidates.extend(sorted(glob.glob(os.path.join(root, "torch", "lib", "libcudart.so*"))))

    for candidate in dict.fromkeys(candidates):
        try:
            lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            lib.cudaGetLastError.argtypes = []
            lib.cudaGetLastError.restype = ctypes.c_int
            lib.cudaMemcpyPeerAsync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            lib.cudaMemcpyPeerAsync.restype = ctypes.c_int
        except (OSError, AttributeError):
            continue
        return lib
    return None


def _clear_hip_last_error(lib) -> None:
    """Drain the sticky HIP "last error" slot left by a ctypes call.

    HIP records errors in a thread-local slot that is *not* cleared by reading
    the return code of the C function we called through ``ctypes``.  A failed or
    redundant ``hipDeviceEnablePeerAccess`` (error 704, "peer access already
    enabled") therefore makes the *next* PyTorch HIP call raise
    ``CUDA error: peer access is already enabled``.  Measured on ROCm
    7.16.26385 / gfx1201: the first enable returns 0, a repeat returns 704, and
    an unrelated ``torch.ones(...).sum()`` then succeeds only after this drain.
    """
    getter = getattr(lib, "hipGetLastError", None)
    if getter is None:  # pragma: no cover - very old HIP
        return
    try:
        getter()
    except Exception as exc:  # pragma: no cover - never let cleanup raise
        log.debug("hipGetLastError drain failed: %s", exc)


def _clear_cuda_last_error() -> None:
    """Drain the CUDA runtime's sticky "last error" slot.

    Mirrors :func:`_clear_hip_last_error`. ``cudaGetLastError`` returns and
    clears the pending error; calling it through ``ctypes`` is enough to unstick
    the runtime's thread-local slot so subsequent PyTorch CUDA calls succeed.
    """

    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        getter = getattr(lib, "cudaGetLastError", None)
        if getter is None:
            continue
        try:
            getter.restype = ctypes.c_int
            getter.argtypes = []
            getter()
        except Exception as exc:  # pragma: no cover - never let cleanup raise
            log.debug("cudaGetLastError drain failed: %s", exc)
        return


def _enable_one(src: int, dst: int) -> tuple[bool, str]:
    """Enable ``src -> dst`` peer access with the caller's device set to ``src``."""
    try:
        torch.cuda.set_device(src)
    except Exception as exc:
        return False, f"set_device({src}) failed: {exc}"

    if torch.version.hip is not None:
        lib = _load_hip()
        if lib is None:
            return False, "HIP runtime library not found"
        result = int(lib.hipDeviceEnablePeerAccess(dst, 0))
        # Always drain: a 704 (or any failure) otherwise poisons the next torch
        # HIP call in this thread.
        _clear_hip_last_error(lib)
        if result == 0 or result == _PEER_ALREADY_ENABLED:
            return True, "hipDeviceEnablePeerAccess" + (" (already enabled)" if result else "")
        return False, f"hipDeviceEnablePeerAccess({dst}) failed: {result}"

    lib = _load_cuda()
    if lib is None:
        return False, "CUDA driver library not found"
    ctx = ctypes.c_void_p()
    result = int(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dst))
    if result != 0:
        return False, f"cuDevicePrimaryCtxRetain({dst}) failed: {result}"
    try:
        result = int(lib.cuCtxEnablePeerAccess(ctx, 0))
        if result == 0 or result == _PEER_ALREADY_ENABLED:
            return True, "cuCtxEnablePeerAccess" + (" (already enabled)" if result else "")
        return False, f"cuCtxEnablePeerAccess({dst}) failed: {result}"
    finally:
        lib.cuDevicePrimaryCtxRelease(dst)
        # CUDA has the same sticky "last error" slot as HIP: a 704 returned by
        # the ctypes call above stays latched and makes the *next* PyTorch CUDA
        # call raise ``CUDA error: peer access is already enabled``. Measured on
        # the 2x RTX 4090 / CUDA 13.2 rig: multi-GPU quantization aborted in the
        # first layer forward with exactly that error until the slot was
        # drained here. ``cudaGetLastError`` (runtime API) clears it.
        _clear_cuda_last_error()


def _timed_gbps(step, src: int, dst: int, bytes_moved: int, iterations: int) -> float | None:
    """Time ``step`` ``iterations`` times and return payload GB/s.

    Each iteration is timed individually with a host wall clock and the median
    is used. Two measured reasons for that shape:

    * CUDA-event pairs are unreliable here. Recording on the source and reading
      after a destination synchronize produced 0.1 GB/s (300+ ms) for a copy
      that takes 0.35 ms; on a staged copy it raises ``Both events must be
      completed``.
    * A single batched loop is not stable either: the first measured batch after
      allocation reported 13 GB/s where the steady state is 50 GB/s on gfx1201,
      even after warm-ups.

    Both devices are synchronized before and after every iteration so a sample
    covers the whole transfer.
    """

    import statistics
    import time

    # Several warm-ups: the first cross-device copy pays lazy HIP/CUDA context
    # creation and kernel module load, measured at ~80 ms / 0.9 GB/s on gfx1201
    # before settling at 0.35 ms / 50 GB/s.
    for _ in range(8):
        step()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        step()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        samples.append(time.perf_counter() - started)
    elapsed = statistics.median(samples)
    if elapsed <= 0:
        return None
    return bytes_moved / elapsed / 1e9


def _measure_gbps(src: int, dst: int, bytes_moved: int, iterations: int = 5) -> float | None:
    """Measure a direct device-to-device copy and return GB/s (None on failure)."""
    try:
        source = torch.empty(bytes_moved, dtype=torch.uint8, device=f"cuda:{src}")
        target = torch.empty(bytes_moved, dtype=torch.uint8, device=f"cuda:{dst}")
        return _timed_gbps(
            lambda: target.copy_(source, non_blocking=True), src, dst, bytes_moved, iterations,
        )
    except Exception as exc:
        log.debug("P2P bandwidth probe %s->%s failed: %s", src, dst, exc)
        return None


def _measure_host_staged_gbps(src: int, dst: int, bytes_moved: int, iterations: int = 5) -> float | None:
    """Return payload GB/s for an explicit ``src -> pinned CPU -> dst`` copy.

    This is the alternative the caller falls back to when a peer link is not
    usable, so it is the reference the direct measurement must beat. The rate
    is payload bytes over wall time, exactly like ``_measure_gbps``.
    """

    try:
        source = torch.empty(bytes_moved, dtype=torch.uint8, device=f"cuda:{src}")
        staged = torch.empty(bytes_moved, dtype=torch.uint8, pin_memory=True)
        target = torch.empty(bytes_moved, dtype=torch.uint8, device=f"cuda:{dst}")

        def step():
            # The D2H leg must complete before the H2D leg reads ``staged``;
            # otherwise the two device streams race and the measured rate is the
            # pipelined sum of both legs (~2x the real staging rate).
            staged.copy_(source, non_blocking=True)
            torch.cuda.synchronize(src)
            target.copy_(staged, non_blocking=True)

        return _timed_gbps(step, src, dst, bytes_moved, iterations)
    except Exception as exc:
        log.debug("P2P host-staged probe %s->%s failed: %s", src, dst, exc)
        return None


def _probe_bandwidth(src: int, dst: int) -> tuple[float | None, float | None]:
    """Return ``(direct_gbps, host_staged_gbps)`` for one directed pair.

    The probe must stay reliable when the devices are already holding a model,
    which is the normal case: the multi-device load enables peer access after the
    first shard is resident. Two measured behaviours shape this function:

    * 64 MiB reaches steady state on gfx1201, but under memory pressure the
      allocation can fail or be served from a fragmented pool and report a noisy
      rate. The payload therefore steps down until one can be measured.
    * A single sample is not stable enough to demote a link: a transient dip
      crossed the ratio boundary for one direction of a pair that measured
      healthy seconds earlier. Each direction is measured twice and the best
      sample is used, so one unlucky run cannot turn a healthy link into a
      ``degraded`` CPU bounce.

    ``P2PConfig.probe_mib`` sets the preferred payload; the ladder below is
    only used when that size cannot be measured.
    """

    key = (src, dst)
    if key in _BANDWIDTH_CACHE:
        return _BANDWIDTH_CACHE[key]

    preferred_mib = _probe_mib()
    ladder = [preferred_mib]
    for fallback in (32, 16, 4):
        if fallback < preferred_mib:
            ladder.append(fallback)

    direct = staged = None
    for mib in ladder:
        probe_bytes = mib * 1024 * 1024
        samples = []
        for _ in range(2):
            sample = _measure_gbps(src, dst, probe_bytes)
            if sample is None:
                samples = []
                break
            samples.append(sample)
        if not samples:
            continue
        direct = max(samples)
        staged = _measure_host_staged_gbps(src, dst, probe_bytes)
        if staged is not None:
            break

    _BANDWIDTH_CACHE[key] = (direct, staged)
    return direct, staged


_AUTO_VALIDATION_MIB = 4


def _verify_peer_payloads(sources, targets, dst: int) -> tuple[bool, str]:
    """Copy ``sources -> targets`` and verify exact size and random content."""

    for source, target in zip(sources, targets):
        if torch.equal(source.cpu(), target.cpu()):
            # CUDA/HIP RNG state is per device and starts from the same seed,
            # so independently generated buffers can be identical. A no-op
            # copy would then pass the comparison.
            return False, "validation payloads are identical before the copy"
    if not _peer_copy_batch_enqueue(sources, targets):
        return False, "runtime peer copy could not be enqueued"
    torch.cuda.current_stream(dst).synchronize()
    for source, target in zip(sources, targets):
        expected = source.numel() * source.element_size()
        actual = target.numel() * target.element_size()
        if actual != expected:
            return False, f"payload size mismatch ({actual} != {expected})"
        if not torch.equal(source.cpu(), target.cpu()):
            return False, "runtime peer copy returned mismatched bytes"
    return True, ""


def _validate_peer_copy_pair(
    src: int,
    dst: int,
    *,
    probe_mib: int | None = None,
) -> tuple[bool, float | None, float | None, str]:
    """Exercise the runtime peer-copy path and measure whether it helps.

    A capability bit and a successful return code are not proof that
    ``cudaMemcpyPeerAsync``/``hipMemcpyPeerAsync`` moved anything. On ROCm with
    ``PYTORCH_ALLOC_CONF=expandable_segments:True`` the call returns success but
    copies no data, so this function copies random buffers and compares the
    exact byte count and content before the pair is ever used. The destination
    is pre-filled with different random bytes, so a no-op copy cannot pass by
    accident.

    It also measures the validated path against ordered pinned-host staging, so
    ``auto`` mode can keep a pair enabled only when the direct copy is at least
    as fast as the fallback. Results are cached per directed pair.
    """

    key = (src, dst)
    cached = _PEER_VALIDATION_CACHE.get(key)
    if cached is not None:
        return cached

    mib = _AUTO_VALIDATION_MIB if probe_mib is None else min(int(probe_mib), _AUTO_VALIDATION_MIB)
    mib = max(1, mib)
    nbytes = mib * 1024 * 1024
    try:
        source = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=f"cuda:{src}")
        # Complement the destination so it can never equal the source, even
        # when both devices' RNGs produce the same sequence.
        target = torch.randint(0, 256, (nbytes,), dtype=torch.uint8,
                               device=f"cuda:{dst}").bitwise_not_()
        ok, reason = _verify_peer_payloads([source], [target], dst)

        # Also exercise the multi-tensor batch shape used by the module and
        # nested movers, with different payload sizes. The nested fast path is
        # disabled on ROCm by default, so this extra check runs only where the
        # shape is actually reachable.
        if ok and not _ROCM_NESTED_PEER_DISABLED:
            sizes = (64 * 1024, 256 * 1024, 1024 * 1024)
            batch_sources = [
                torch.randint(0, 256, (size,), dtype=torch.uint8,
                              device=f"cuda:{src}")
                for size in sizes
            ]
            batch_targets = [
                torch.randint(0, 256, (size,), dtype=torch.uint8,
                              device=f"cuda:{dst}").bitwise_not_()
                for size in sizes
            ]
            ok, reason = _verify_peer_payloads(batch_sources, batch_targets, dst)

        if not ok:
            result = (False, None, None, reason)
        else:
            direct = _timed_gbps(
                lambda: _peer_copy_batch_enqueue([source], [target]),
                src,
                dst,
                nbytes,
                iterations=3,
            )
            staged = _measure_host_staged_gbps(src, dst, nbytes, iterations=3)
            result = (True, direct, staged, "")
    except Exception as exc:  # pragma: no cover - runtime dependent
        result = (False, None, None, f"runtime peer copy validation failed: {exc}")

    _PEER_VALIDATION_CACHE[key] = result
    log.debug("P2P validation %s->%s: %s", src, dst, result[3] or "ok")
    return result


def _warn_degraded_once(status: PeerStatus) -> None:
    """Log one actionable warning when a multi-GPU run lacks full P2P.

    The wording is deliberately diagnostic: a missing directed pair is almost
    always a platform limitation, and the user needs to know the run still
    works but pays for host staging. ``_WARNED_DEGRADED`` makes it once per
    process; the lock keeps concurrent first calls from logging twice.
    """

    global _WARNED_DEGRADED

    if status.mode == _MODE_OFF or len(status.devices) < 2 or not status.pairs:
        return
    if all(state == STATE_ENABLED for state in status.pairs.values()):
        return

    with _WARN_LOCK:
        if _WARNED_DEGRADED:
            return
        _WARNED_DEGRADED = True

    enabled = sum(1 for state in status.pairs.values() if state == STATE_ENABLED)
    total = len(status.pairs)
    log.warning(
        "Multi-GPU peer-to-peer (P2P) access could not be fully enabled/validated "
        "for every ordered GPU pair (%d/%d directed pairs enabled). The GPUs do "
        "not have usable P2P either because of motherboard compatibility (PCIe "
        "slot wiring, ACS/IOMMU), BIOS compatibility (Above 4G decoding, "
        "Resizable BAR, PCIe generation), or hardware compatibility. Cross-GPU "
        "copies will fall back to slower host staging and multi-GPU "
        "quantization/inference speed may be sub-optimal. Set "
        "`p2p=P2PConfig(mode='on')` to force an enablement attempt; details: %s",
        enabled,
        total,
        status.summary(),
    )


def enable_peer_access(devices: Sequence | None = None, *, strict: bool = False) -> PeerStatus:
    """Enable peer access for every directed pair and classify each link.

    PyTorch enables peer access lazily on its first cross-device copy, so calling
    this beforehand only removes that first-copy latency; it is not what keeps
    copies on the interconnect.

    ``enabled``   - capability reported and enablement succeeded.
    ``unsupported`` - capability is false or enablement failed.

    When ``P2PConfig.probe`` is set a measured copy additionally downgrades a
    link to ``degraded`` if it is slower than ordered pinned-host staging. That
    is a diagnostic verdict: peer access stays enabled and no caller inserts a
    host relay because of it.

    With ``strict=True`` an enablement failure raises; the default is
    best-effort and only logs one summary line. A cached best-effort failure
    still raises for a later ``strict=True`` request.
    """

    global _LAST_STATUS

    mode = p2p_mode()
    indices = _normalize_indices(devices)
    status = PeerStatus(mode=mode, devices=indices)

    if mode == _MODE_OFF:
        status.reason = "P2PConfig.mode='off'"
        _LAST_STATUS = status
        log.debug("%s", status.summary())
        return status

    checking = _cache_namespace()
    cache_key = (mode, checking, indices)
    cached = _STATUS_CACHE.get(cache_key)
    if cached is not None:
        _warn_degraded_once(cached)
        _LAST_STATUS = cached
        if strict and cached.errors:
            (src, dst), detail = next(iter(cached.errors.items()))
            raise RuntimeError(f"P2P {src}->{dst} enablement failed: {detail}")
        return cached

    if len(indices) < 2 or not torch.cuda.is_available():
        status.reason = "fewer than two CUDA/HIP devices"
        _STATUS_CACHE[cache_key] = status
        _LAST_STATUS = status
        log.debug("%s", status.summary())
        return status

    threshold = _min_gbps()
    min_ratio = _min_ratio()
    diagnose = checking
    capability = peer_matrix(indices)
    try:
        original_device = torch.cuda.current_device()
    except Exception:
        original_device = None

    try:
        # Pass 1: enable every available directed pair before validating any of
        # them. A validation copy is issued on the destination device, so it
        # also needs the reverse direction enabled to read the source mapping.
        # Validating inside the enablement loop made the first direction look
        # broken on asymmetric topologies.
        pairs_to_validate: list[tuple[int, int]] = []
        for row, src in enumerate(indices):
            for col, dst in enumerate(indices):
                if src == dst:
                    continue
                # ``mode='on'`` is an explicit request to try the enablement
                # call even when the capability probe says no.
                forced = not capability[row][col]
                available = capability[row][col] or mode == _MODE_ON
                if not available:
                    status.pairs[(src, dst)] = STATE_UNSUPPORTED
                    continue
                ok, detail = _enable_one(src, dst)
                if not ok:
                    status.pairs[(src, dst)] = STATE_UNSUPPORTED
                    status.errors[(src, dst)] = detail
                    log.debug("P2P %s->%s not enabled: %s", src, dst, detail)
                    if strict:
                        raise RuntimeError(f"P2P {src}->{dst} enablement failed: {detail}")
                    continue

                if forced:
                    # The runtime denies this direction, so a success return
                    # code is not evidence of a usable link: a direct copy here
                    # can still segfault inside hipMemcpyPeerAsync. Never probe
                    # it either -- the probe performs a direct copy. Peer
                    # access stays enabled for PyTorch's own paths, but
                    # GPT-QModel moves route through host memory.
                    status.pairs[(src, dst)] = STATE_UNVERIFIED
                    log.debug(
                        "P2P %s->%s enablement reported success (%s) but the "
                        "capability query denies the direction; routing through "
                        "host memory",
                        src,
                        dst,
                        detail,
                    )
                    continue

                pairs_to_validate.append((src, dst))

        # Pass 2: validate and classify the pairs that were enabled.
        for src, dst in pairs_to_validate:
            # A capability bit and a successful enablement call do not prove
            # that the runtime peer copy moves bytes. Validate the actual
            # path before any caller can use it.
            usable, direct, staged, validation_reason = _validate_peer_copy_pair(
                src, dst
            )
            if not usable:
                status.pairs[(src, dst)] = STATE_DEGRADED
                log.debug(
                    "P2P %s->%s disabled: %s", src, dst, validation_reason
                )
                continue

            if mode == _MODE_AUTO and not diagnose:
                # ``auto`` must not enable a link that is no faster than the
                # regular copy path. This is the default, so it runs without
                # the opt-in full diagnostic.
                status.bandwidth_gbps[(src, dst)] = direct
                status.host_staged_gbps[(src, dst)] = staged
                reason = None
                if direct is None:
                    reason = "auto validation could not measure the direct path"
                elif direct < threshold:
                    reason = f"{direct:.2f} GB/s < {threshold:.2f} GB/s gate"
                elif staged and direct < staged * min_ratio:
                    reason = (
                        f"{direct:.2f} GB/s direct < {min_ratio:.2f}x host staging "
                        f"({staged:.2f} GB/s): the peer link is not faster than the fallback"
                    )
                if reason is not None:
                    status.pairs[(src, dst)] = STATE_DEGRADED
                    log.debug("P2P %s->%s degraded: %s", src, dst, reason)
                else:
                    status.pairs[(src, dst)] = STATE_ENABLED
                continue

            if not diagnose:
                # ``mode='on'`` is an explicit request to force the direct
                # path; correctness was validated above, but the relative
                # speed gate is intentionally skipped.
                status.pairs[(src, dst)] = STATE_ENABLED
                continue

            measured, staged = _probe_bandwidth(src, dst)
            status.bandwidth_gbps[(src, dst)] = measured
            status.host_staged_gbps[(src, dst)] = staged
            reason = None
            if measured is None:
                # An unmeasurable direct link is not evidence of a fast one.
                reason = "direct probe failed"
            elif measured < threshold:
                reason = f"{measured:.2f} GB/s < {threshold:.2f} GB/s gate"
            elif staged and mode != _MODE_ON and measured < staged * min_ratio:
                # ``mode='on'`` is an explicit request for the direct
                # path, so the relative comparison must not veto it. The
                # absolute floor above still catches a link that is not
                # moving data at all.
                reason = (
                    f"{measured:.2f} GB/s direct < {min_ratio:.2f}x host staging "
                    f"({staged:.2f} GB/s): the peer link is not faster than the fallback"
                )
            if reason is not None:
                status.pairs[(src, dst)] = STATE_DEGRADED
                log.debug("P2P %s->%s degraded: %s", src, dst, reason)
            else:
                status.pairs[(src, dst)] = STATE_ENABLED
    finally:
        if original_device is not None:
            try:
                torch.cuda.set_device(original_device)
            except Exception as exc:
                # Best-effort cleanup: never let a failed device restore mask the
                # original enablement error.
                log.debug("failed to restore device %s: %s", original_device, exc)

    _STATUS_CACHE[cache_key] = status
    _LAST_STATUS = status
    asymmetric = status.asymmetric_pairs()
    if asymmetric:
        pairs = ", ".join(f"{src}->{dst}" for src, dst in asymmetric)
        log.warning(
            "P2P topology is asymmetric (%s unsupported while the reverse is enabled); "
            "cross-device moves in those directions will be staged through host memory "
            "because a direct peer copy can crash the process.",
            pairs,
        )
    unverified = status.unverified_pairs()
    if unverified:
        pairs = ", ".join(f"{src}->{dst}" for src, dst in unverified)
        log.warning(
            "P2PConfig(mode='on') forced peer access on %s, but the capability query denies those "
            "directions; they are treated as unusable and staged through host "
            "memory because a direct peer copy can crash the process.",
            pairs,
        )
    log.info("%s", status.summary())
    _warn_degraded_once(status)
    return status


def _device_ordinal(device: torch.device) -> int | None:
    """Resolve an accelerator device to its ordinal (``None`` for CPU/other)."""
    if device.type != "cuda":
        return None
    if device.index is not None:
        return device.index
    try:
        return torch.cuda.current_device()
    except Exception:  # pragma: no cover - runtime dependent
        return 0


def _can_access_peer(src: int, dst: int) -> bool:
    """Cached ``can_device_access_peer`` answer for one directed pair."""
    key = (src, dst)
    cached = _CAPABILITY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        value = bool(torch.cuda.can_device_access_peer(src, dst))
    except Exception as exc:  # pragma: no cover - runtime dependent
        # An unanswerable capability query must not be treated as a healthy
        # link: a direct copy on an unsupported ROCm direction segfaults.
        log.debug("can_device_access_peer(%s, %s) failed: %s", src, dst, exc)
        value = False
    _CAPABILITY_CACHE[key] = value
    return value


def can_transfer_directly(
    src: torch.device | str,
    dst: torch.device | str,
    status: PeerStatus | None = None,
) -> bool:
    """True only when a direct ``src -> dst`` transfer is known to be safe.

    Unlike a global "P2P is available" check this is evaluated per direction.
    When ``status`` covers the pair its verdict wins (so ``degraded`` links stage
    through host memory, matching the probe that measured the fallback as
    faster); otherwise the raw capability query is used. Transfers that do not
    cross two accelerator devices are always safe.
    """
    src = torch.device(src)
    dst = torch.device(dst)
    src_index = _device_ordinal(src)
    dst_index = _device_ordinal(dst)
    if src_index is None or dst_index is None or src_index == dst_index:
        return True
    if status is None:
        status = _LAST_STATUS
    if status is not None and (src_index, dst_index) in status.pairs:
        return status.can_copy(src_index, dst_index)
    return _can_access_peer(src_index, dst_index)


def _warn_staged_once(src: int, dst: int, reason: str) -> None:
    key = (src, dst)
    if key in _STAGED_DIRECTIONS:
        return
    _STAGED_DIRECTIONS.add(key)
    log.warning(
        "P2P %s->%s is not usable (%s); cross-device moves in this direction "
        "are staged through pinned host memory to avoid a peer-copy crash.",
        src,
        dst,
        reason,
    )


def _stage_through_host(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move ``tensor`` to ``device`` via a pinned host buffer.

    The H2D leg is issued synchronously: a non-blocking copy from a temporary
    pinned buffer can outlive that buffer once this function returns, which
    corrupts the destination. Correctness beats the overlap here; this path is
    only used for directions the peer link cannot serve.
    """
    src_device = tensor.device
    src_index = _device_ordinal(src_device)
    dst_index = _device_ordinal(device)
    if src_index is not None and dst_index is not None:
        _warn_staged_once(src_index, dst_index, "direct peer copy not usable in this direction")

    try:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        with torch.no_grad():
            host.copy_(tensor.detach())
    except Exception:
        # ``pin_memory`` can be unavailable (no CUDA driver context, restricted
        # container); pageable staging still avoids the crashing peer copy.
        host = tensor.detach().to(device="cpu", copy=True)

    return host.to(device=device, dtype=dtype, non_blocking=False)


def safe_move_to(
    tensor: torch.Tensor,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
    status: PeerStatus | None = None,
    copy: bool = False,
) -> torch.Tensor:
    """Move ``tensor`` to ``device`` without an unsafe direct peer copy.

    ``tensor.to(device)`` on an asymmetric ROCm topology can segfault inside
    ``hipMemcpyPeerAsync`` instead of raising, so every cross-device move must
    go through this helper. When ``can_transfer_directly`` is false the tensor
    is staged ``src -> pinned CPU -> dst``, which was verified to work on the
    affected rig. ``status`` is optional; without it the cached capability query
    decides the direction.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"safe_move_to expects a Tensor, got {type(tensor).__name__}")

    device = as_torch_device(device)
    target_dtype = tensor.dtype if dtype is None else dtype
    ordinal = _device_ordinal(device)
    if ordinal is not None and device.index is None:
        device = torch.device(device.type, ordinal)

    if tensor.device == device and target_dtype == tensor.dtype:
        return tensor.clone() if copy else tensor

    if can_transfer_directly(tensor.device, device, status):
        return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

    # The staged result is always a fresh tensor, so ``copy=True`` is satisfied.
    return _stage_through_host(tensor, device, dtype=dtype)


def safe_module_to(
    module: torch.nn.Module,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
    status: PeerStatus | None = None,
) -> torch.nn.Module:
    """Move a module to ``device`` without a crashing direct peer copy.

    ``nn.Module.to`` cannot be redirected per tensor, so when any parameter or
    buffer would need an unsupported direction the module is first moved to the
    CPU and then to the destination. Both legs are safe and ``_apply`` still
    runs, which preserves dtype/format handling and cache invalidation in
    quantized subclasses.
    """
    device = as_torch_device(device)
    ordinal = _device_ordinal(device)
    if ordinal is not None and device.index is None:
        device = torch.device(device.type, ordinal)

    sources = {tensor.device for tensor in module.parameters()}
    sources.update(tensor.device for tensor in module.buffers())
    needs_staging = any(
        not can_transfer_directly(src, device, status)
        for src in sources
        if src.type == "cuda" and src != device
    )
    if needs_staging:
        module.to(device=torch.device("cpu"), dtype=dtype)
    return module.to(device=device, dtype=dtype)


def copy_tensor_to(dst: torch.Tensor, src: torch.Tensor, status: PeerStatus | None = None) -> torch.Tensor:
    """Copy ``src`` into ``dst``, staging through host memory when required.

    A plain ``dst.copy_(src)`` uses the peer path whenever the driver thinks it
    is available; on an asymmetric ROCm link that is exactly the call that
    segfaults. The direction is therefore checked first and an unsupported
    direction is relayed through pinned host memory. ``dst`` is returned.
    """
    if can_transfer_directly(src.device, dst.device, status):
        dst.copy_(src, non_blocking=True)
        return dst
    dst.copy_(safe_move_to(src, dst.device, status=status))
    return dst


def p2p_all_enabled(devices: Sequence | None = None) -> bool:
    """True when every directed pair between ``devices`` has peer access."""

    indices = _normalize_indices(devices)
    if len(indices) < 2:
        return False
    status = _cached_status_for(indices) or enable_peer_access(indices)
    return all(
        status.state(src, dst) == STATE_ENABLED
        for src in indices
        for dst in indices
        if src != dst
    )


def _cached_status_for(indices: tuple[int, ...]) -> PeerStatus | None:
    """Return a cached status whose device set covers ``indices``.

    ``enable_peer_access`` caches by the exact device tuple, so a status
    probed for ``[0, 1]`` would not answer a later ``[1, 0]`` query. The hot
    copy path asks for directed pairs in whatever order the caller uses, so
    reuse a superset instead of re-probing the same links.
    """

    mode = p2p_mode()
    checking = _cache_namespace()
    wanted = set(indices)
    for (cached_mode, cached_checking, _), status in list(_STATUS_CACHE.items()):
        if cached_mode != mode or cached_checking != checking:
            continue
        if wanted.issubset(status.devices):
            return status
    return None


def _can_peer_copy(src: int, dst: int) -> bool:
    """Return whether the runtime peer copy is usable for ``src -> dst``."""

    if src == dst:
        return False
    try:
        status = _cached_status_for((src, dst)) or enable_peer_access([src, dst])
        return status.can_copy(src, dst)
    except Exception as exc:  # pragma: no cover - runtime dependent
        log.debug("P2P status check %s->%s failed: %s", src, dst, exc)
        return False


def _clear_last_error_for_backend(lib=None) -> None:
    """Drain the sticky runtime error left behind by a failed peer copy."""

    if getattr(torch.version, "hip", None) is not None:
        if lib is not None:
            _clear_hip_last_error(lib)
        return
    _clear_cuda_last_error()


def _peer_copy_runtime() -> tuple[ctypes.CDLL, str] | None:
    """Return ``(runtime_library, symbol)`` for the async peer copy."""

    global _PEER_COPY_RUNTIME

    if _PEER_COPY_RUNTIME is not None:
        return _PEER_COPY_RUNTIME
    with _PEER_COPY_LOCK:
        if _PEER_COPY_RUNTIME is not None:
            return _PEER_COPY_RUNTIME
        if getattr(torch.version, "hip", None) is not None:
            lib = _load_hip()
            symbol = "hipMemcpyPeerAsync"
        else:
            lib = _load_cudart()
            symbol = "cudaMemcpyPeerAsync"
        if lib is None or not hasattr(lib, symbol):
            return None
        _PEER_COPY_RUNTIME = (lib, symbol)
        return _PEER_COPY_RUNTIME


def _peer_copy_batch_enqueue(
    sources: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
) -> bool:
    """Enqueue runtime peer-to-peer copies without consulting peer status.

    PyTorch's ``torch.cuda.comm.broadcast_coalesced`` funnels into
    ``torch._C._broadcast_coalesced``. That entry point is not safe to call
    from several threads on free-threaded (``PYTHON_GIL=0``) builds: it has
    been measured to hang and to publish torn copies on both ROCm and CUDA.
    The runtime's own ``cudaMemcpyPeerAsync``/``hipMemcpyPeerAsync`` is
    reached through ``ctypes``, which releases the GIL for the enqueue, so
    worker threads can stage replicas and calibration inputs concurrently.

    Every source must live on one device and every target on one other device.
    Copies are ordered after the source stream and are enqueued on the
    destination device's current stream, so consumers must use that stream
    (which normal PyTorch code on the destination device does).

    Returns ``True`` when all copies were enqueued. Returns ``False`` when the
    runtime is unavailable, a tensor is not contiguous, or the group is not a
    simple two-device batch. The caller is responsible for checking that peer
    access is enabled and validated for the pair.
    """

    if not sources or len(sources) != len(targets):
        return False

    source_device = sources[0].device
    target_device = targets[0].device
    if source_device.type != "cuda" or target_device.type != "cuda":
        return False
    if source_device == target_device:
        return False
    if source_device.index is None or target_device.index is None:
        return False
    if any(tensor.device != source_device for tensor in sources):
        return False
    if any(tensor.device != target_device for tensor in targets):
        return False
    runtime = _peer_copy_runtime()
    if runtime is None:
        return False
    lib, symbol = runtime
    peer_copy = getattr(lib, symbol)

    for source, target in zip(sources, targets):
        # The copy moves raw bytes: a non-contiguous view would be read in
        # storage order and silently corrupt the destination.
        if not source.is_contiguous() or not target.is_contiguous():
            return False
        if source.numel() * source.element_size() != target.numel() * target.element_size():
            return False

    # The runtime binds a stream to the *current* device's context, so the
    # destination stream must be used with the destination device current.
    # HIP does not do this implicitly: issuing the copy while the source device
    # is current places the work on the source device's queue, where the
    # destination stream -- and therefore ``Tensor.cpu()`` -- never waits for
    # it. Measured on ROCm as reads of uninitialised memory immediately after
    # ``p2p_move_nested``; ``torch.cuda.synchronize(target)`` alone did not fix
    # it while ``synchronize(source)`` did. CUDA orders the copy correctly
    # either way, but the explicit device scope is correct on both backends.
    try:
        with torch.cuda.device(target_device):
            target_stream = torch.cuda.current_stream(target_device)
            # Cross-device copies are not ordered against the producing stream
            # unless we join them explicitly; the calling thread's own stream
            # is what produced (or will consume) these tensors.
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(source_device))
            target_stream.wait_event(event)
            stream_handle = ctypes.c_void_p(target_stream.cuda_stream)

            for source, target in zip(sources, targets):
                byte_count = source.numel() * source.element_size()
                if byte_count == 0:
                    continue
                rc = peer_copy(
                    ctypes.c_void_p(target.data_ptr()),
                    int(target_device.index),
                    ctypes.c_void_p(source.data_ptr()),
                    int(source_device.index),
                    ctypes.c_size_t(byte_count),
                    stream_handle,
                )
                if rc != 0:
                    # A failed peer copy leaves a sticky runtime error that
                    # would otherwise surface from an unrelated PyTorch call.
                    _clear_last_error_for_backend(lib)
                    return False
                # The raw runtime copy is invisible to PyTorch's caching
                # allocator. Without this the source block can be freed and
                # reused on the source stream while the destination stream is
                # still reading it, which shows up as an async
                # HSA_STATUS_ERROR_MEMORY_FAULT on ROCm and torn bytes on
                # CUDA. Recording the target stream defers reuse until the
                # copy has completed.
                source.record_stream(target_stream)
                target.record_stream(target_stream)
    except Exception:
        _clear_last_error_for_backend(lib)
        return False

    return True


def peer_copy_batch(
    sources: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
) -> bool:
    """Enqueue validated runtime peer-to-peer copies for ``sources -> targets``.

    The pair must have been enabled and validated by :func:`enable_peer_access`
    (which ``auto`` mode does automatically). ``False`` tells the caller to use
    its regular ``Tensor.to`` path instead.
    """

    if not sources or len(sources) != len(targets):
        return False
    source_device = sources[0].device
    target_device = targets[0].device
    if (
        source_device.type != "cuda"
        or target_device.type != "cuda"
        or source_device.index is None
        or target_device.index is None
    ):
        return False
    if not _can_peer_copy(source_device.index, target_device.index):
        return False
    return _peer_copy_batch_enqueue(sources, targets)


def _contiguous_like(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Allocate a contiguous destination for a peer copy."""

    return torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=device,
        layout=tensor.layout,
    )


# The peer-copy path only wins when per-leaf dispatch overhead dominates the
# transfer itself. Measured on 2x RTX 4090 (within-process, median of 3 runs x
# 10 roundtrips, 12 module shapes): at 64 KiB per leaf it is 3-14% faster than
# the regular ``Tensor.to`` path, while at 512 KiB and above it is 0-6% slower
# because each leaf needs an extra allocation plus an event join. 256 KiB sits
# between the measured win and loss regions.
_PEER_COPY_MAX_AVG_LEAF_BYTES = 256 * 1024

# ROCm's HIP runtime can fault the GPU (``HSA_STATUS_ERROR_MEMORY_FAULT``) and
# wedge the process when a small VMM peer copy targets an expandable-segment
# mapping that the executing device was never granted. The per-tensor nested
# path is the only caller that hits that shape; the module path copies larger,
# validated ranges. Disable the nested fast path on ROCm until the runtime VMM
# peer-access fix is available. Set ``GPTQMODEL_P2P_NESTED_ROCM=1`` to opt in
# on a runtime that carries the fix.
_ROCM_NESTED_PEER_DISABLED = torch.version.hip is not None and os.environ.get(
    "GPTQMODEL_P2P_NESTED_ROCM", ""
).strip().lower() not in ("1", "true", "yes", "on")


def broadcast_module_to_device(module: torch.nn.Module, device: torch.device) -> bool:
    """Move all CUDA leaves of ``module`` to ``device`` with runtime peer copies.

    Returns ``True`` when the peer-copy path handled the move. Callers must
    still realign Accelerate hooks afterwards; this helper only replaces the
    many small ``tensor.to()`` calls that otherwise dominate cross-GPU module
    moves. Returns ``False`` for CPU/meta/offloaded/mixed-device modules or
    non-contiguous leaves so the caller can fall back to the regular move path.
    """

    target = device if isinstance(device, torch.device) else torch.device(str(device))
    if target.type != "cuda" or target.index is None:
        return False

    leaves: list[tuple[torch.nn.Module, str, torch.Tensor, bool]] = []
    source_devices = set()
    seen_tensors = set()
    for sub in module.modules():
        for name, parameter in list(getattr(sub, "_parameters", {}).items()):
            if parameter is None:
                continue
            if parameter.device.type != "cuda" or parameter.is_meta:
                return False
            # A shared (tied) parameter appears under more than one name. The
            # peer path replaces parameter objects, so it must fall back rather
            # than silently untie the weights.
            if id(parameter) in seen_tensors:
                return False
            seen_tensors.add(id(parameter))
            leaves.append((sub, name, parameter, True))
            source_devices.add(parameter.device)
        for name, buffer in list(getattr(sub, "_buffers", {}).items()):
            if buffer is None or not isinstance(buffer, torch.Tensor):
                continue
            if buffer.device.type != "cuda" or buffer.is_meta:
                return False
            if id(buffer) in seen_tensors:
                return False
            seen_tensors.add(id(buffer))
            leaves.append((sub, name, buffer, False))
            source_devices.add(buffer.device)

    if not leaves:
        return False
    if len(source_devices) != 1:
        return False

    source = next(iter(source_devices))
    if source == target:
        return True
    if source.index is None or not _can_peer_copy(source.index, target.index):
        return False

    pending = [(sub, name, tensor, is_param) for sub, name, tensor, is_param in leaves if tensor.device != target]
    if not pending:
        return True

    # The peer-copy path moves raw storage, so every leaf must be contiguous.
    # A non-contiguous parameter would be copied in storage order and silently
    # produce wrong values, so fall back to the regular move instead.
    non_contiguous = [entry for entry in pending if not entry[2].is_contiguous()]
    if non_contiguous:
        # Observable, not silent: the regular path is correct but skips the
        # peer-copy fast path, and a non-contiguous leaf usually means a new
        # load path or kernel skipped its ``.contiguous()``.
        log.debug(
            "P2P peer copy skipped for %d non-contiguous leaf/leaves: %s",
            len(non_contiguous),
            ", ".join(
                f"{type(sub).__name__}.{name}{tuple(tensor.shape)} stride={tuple(tensor.stride())}"
                for sub, name, tensor, _ in non_contiguous[:5]
            ),
        )
        return False

    # Leave large-leaf modules to the regular move path: the peer path adds an
    # allocation and an event join per leaf, which only pays for itself when
    # the leaves are small.
    total_bytes = sum(entry[2].numel() * entry[2].element_size() for entry in pending)
    if total_bytes > _PEER_COPY_MAX_AVG_LEAF_BYTES * len(pending):
        return False

    try:
        moved = [_contiguous_like(entry[2], target) for entry in pending]
    except Exception:
        return False
    if not peer_copy_batch([entry[2] for entry in pending], moved):
        return False

    # Unlike a calibration input, a module move is structural: the new weights
    # must be readable from any stream as soon as this returns, which is what
    # ``Tensor.to`` guarantees. Wait for the copies we just enqueued.
    torch.cuda.current_stream(target).synchronize()

    for (sub, name, old, is_param), new in zip(pending, moved):
        if is_param:
            sub._parameters[name] = torch.nn.Parameter(new, requires_grad=old.requires_grad)
        else:
            if old.requires_grad:
                new.requires_grad_(True)
            sub._buffers[name] = new

    return True


_P2P_NOT_APPLICABLE = object()
# Public alias for callers that need to distinguish "fast path handled it" from
# "fall back to the regular move helper".
P2P_NOT_APPLICABLE = _P2P_NOT_APPLICABLE


def p2p_move_nested(value: Any, device: torch.device, *, min_bytes: int = 1024 * 1024) -> Any:
    """Move a nested tensor structure to ``device`` using runtime peer copies.

    Returns ``P2P_NOT_APPLICABLE`` when the fast path cannot be used so the
    caller can fall back to its regular move helper. The fast path only
    triggers for at least two tensors living on one CUDA source device, which
    is exactly the multi-batch/multi-kwarg case. Each leaf still costs one
    enqueue, so this is on par with, not faster than, ``Tensor.to``; it exists
    so the copies do not go through ``torch.cuda.comm``.
    """

    target = device if isinstance(device, torch.device) else torch.device(str(device))
    if (
        target.type != "cuda"
        or target.index is None
        or not isinstance(value, (torch.Tensor, dict, list, tuple))
    ):
        return _P2P_NOT_APPLICABLE

    if _ROCM_NESTED_PEER_DISABLED:
        return _P2P_NOT_APPLICABLE

    tensors: list[torch.Tensor] = []

    def _walk(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                _walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                _walk(child)

    _walk(value)
    if not tensors:
        return _P2P_NOT_APPLICABLE

    if all(tensor.device == target for tensor in tensors):
        return value

    pending = [tensor for tensor in tensors if tensor.device != target]
    if len(pending) < 2:
        return _P2P_NOT_APPLICABLE

    # The peer-copy fast path always produces contiguous leaves and does not
    # preserve autograd history. Both are fine for calibration/inference
    # activations, but fall back to ``.to()`` when a caller relies on either.
    if any(tensor.requires_grad for tensor in pending):
        return _P2P_NOT_APPLICABLE
    if not all(tensor.is_contiguous() for tensor in pending):
        return _P2P_NOT_APPLICABLE
    source_devices = {tensor.device for tensor in pending}
    if len(source_devices) != 1:
        return _P2P_NOT_APPLICABLE

    source = next(iter(source_devices))
    if source.type != "cuda" or source.index is None:
        return _P2P_NOT_APPLICABLE

    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in pending)
    if total_bytes < min_bytes:
        return _P2P_NOT_APPLICABLE

    # Same size gate as the module path: a peer copy needs an extra allocation
    # and event join per leaf, which only pays off for small leaves. Typical
    # calibration batches have multi-MiB activations and stay on ``.to()``.
    if total_bytes > _PEER_COPY_MAX_AVG_LEAF_BYTES * len(pending):
        return _P2P_NOT_APPLICABLE

    if not _can_peer_copy(source.index, target.index):
        return _P2P_NOT_APPLICABLE

    staged = [_contiguous_like(tensor, target) for tensor in pending]
    if not peer_copy_batch(pending, staged):
        return _P2P_NOT_APPLICABLE

    replacements = {id(tensor): moved for tensor, moved in zip(pending, staged)}

    def _rebuild(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return replacements.get(id(item), item)
        if isinstance(item, dict):
            return {key: _rebuild(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(_rebuild(child) for child in item)
        return item

    return _rebuild(value)


def enable_for_devices(devices: Sequence | None = None, *, strict: bool = False) -> PeerStatus:
    """Convenience wrapper used by the quantization loops.

    Default is best effort: a probe failure is logged and reported in
    ``PeerStatus.reason`` instead of blocking quantization. With ``strict=True``
    the failure is re-raised.
    """

    try:
        return enable_peer_access(devices, strict=strict)
    except Exception as exc:
        if strict:
            raise
        status = PeerStatus(mode=p2p_mode(), devices=_normalize_indices(devices), reason=str(exc))
        log.info("%s", status.summary())
        return status
