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
copy so the very first transfer is direct, and (b) route every cross-device
move through a direction-aware helper.  Enabling is cheap and idempotent;
bandwidth probing moves real payload and is opt-in via
``GPTQMODEL_P2P_PROBE=1``.

A capability bit is not proof of usable bandwidth, so an opt-in probe classifies
each direction as ``enabled``/``degraded``.  ``can_copy(src, dst)`` answers for
one direction only.  On some ROCm topologies the link is *asymmetric*: the
capability query reports ``0 -> 1`` unsupported while ``1 -> 0`` is enabled, and
PyTorch's direct ``tensor.to("cuda:1")`` then calls ``hipMemcpyPeerAsync`` and
segfaults the process instead of raising.  :func:`safe_move_to` therefore stages
that direction through pinned host memory, which is the only path the driver
does not crash on.  Never assert P2P is effective from a single direction or
from a global "some link works" verdict.

The module is opt-out via ``GPTQMODEL_P2P=0`` and force-on via ``GPTQMODEL_P2P=1``
(default ``auto``).  Results are cached per device tuple for the process.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from .logger import setup_logger
from .torch import as_torch_device


log = setup_logger()

ENV_P2P = "GPTQMODEL_P2P"
ENV_P2P_MIN_GBPS = "GPTQMODEL_P2P_MIN_GBPS"
ENV_P2P_MIN_RATIO = "GPTQMODEL_P2P_MIN_RATIO"
ENV_P2P_PROBE = "GPTQMODEL_P2P_PROBE"
ENV_P2P_PROBE_MIB = "GPTQMODEL_P2P_PROBE_MIB"

# "auto" enables P2P when the runtime reports a peer link; "1" forces the
# enablement attempt; "0" skips the pre-enablement call entirely (PyTorch's own
# lazy peer setup still runs inside its copy path).
_MODE_AUTO = "auto"
_MODE_ON = "1"
_MODE_OFF = "0"
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
# link that is genuinely slower than the fallback. ``GPTQMODEL_P2P_MIN_RATIO``
# can be lowered for a topology that really does measure at parity.
DEFAULT_MIN_RATIO = 1.0

STATE_ENABLED = "enabled"
STATE_UNSUPPORTED = "unsupported"
STATE_DEGRADED = "degraded"
# ``GPTQMODEL_P2P=1`` can make ``hipDeviceEnablePeerAccess`` report success for
# a direction the runtime capability query denies. That is not evidence the
# direction is usable -- on the asymmetric ROCm rig the enable call fails with
# 101, but a driver that accepted it would still crash a direct copy -- so such
# a pair is never routed directly.
STATE_UNVERIFIED = "unverified"

_PEER_ALREADY_ENABLED = 704

_STATUS_CACHE: dict[tuple[str, bool, tuple[int, ...]], "PeerStatus"] = {}
_BANDWIDTH_CACHE: dict[tuple[int, int], tuple[float | None, float | None]] = {}
# Raw ``can_device_access_peer`` answers, cached per directed pair.  This is the
# fallback used by :func:`safe_move_to` when no probe status covers the pair.
_CAPABILITY_CACHE: dict[tuple[int, int], bool] = {}
# One warning per unsupported direction, not one per tensor move.
_STAGED_DIRECTIONS: set[tuple[int, int]] = set()
# Most recent probe result. ``safe_move_to`` uses it when the caller does not
# pass a status, so a measured ``degraded`` verdict also selects host staging.
_LAST_STATUS: "PeerStatus | None" = None


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
        """Directions forced on with ``GPTQMODEL_P2P=1`` that the runtime denies.

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
    _CAPABILITY_CACHE.clear()
    _STAGED_DIRECTIONS.clear()
    _LAST_STATUS = None


def _cache_namespace() -> bool:
    """Cache bucket: probing and non-probing results must not be shared."""
    return probe_enabled()


def p2p_mode() -> str:
    """Return the requested mode: ``auto``, ``1`` (force), or ``0`` (off)."""
    raw = os.getenv(ENV_P2P, _MODE_AUTO).strip().lower()
    if raw in _OFF_VALUES:
        return _MODE_OFF
    if raw in _ON_VALUES:
        return _MODE_ON
    return _MODE_AUTO


def _min_gbps() -> float:
    try:
        return max(0.0, float(os.getenv(ENV_P2P_MIN_GBPS, DEFAULT_MIN_GBPS)))
    except ValueError:
        return DEFAULT_MIN_GBPS


def _min_ratio() -> float:
    try:
        return max(0.0, float(os.getenv(ENV_P2P_MIN_RATIO, DEFAULT_MIN_RATIO)))
    except ValueError:
        return DEFAULT_MIN_RATIO


def probe_enabled() -> bool:
    """True when bandwidth diagnostics were requested via ``GPTQMODEL_P2P_PROBE``.

    Probing is off by default: PyTorch already keeps cross-device copies on the
    peer link, so measuring every directed pair only adds startup latency and
    payload traffic without changing the transfer path.
    """

    return os.getenv(ENV_P2P_PROBE, "").strip().lower() in _ON_VALUES


def _probe_mib() -> int:
    try:
        return max(1, int(os.getenv(ENV_P2P_PROBE_MIB, "64")))
    except (TypeError, ValueError):
        log.debug("%s=%r is not an integer; using 64 MiB", ENV_P2P_PROBE_MIB, os.getenv(ENV_P2P_PROBE_MIB))
        return 64


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


def _load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            lib = ctypes.CDLL(name)
            lib.hipDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
            lib.hipDeviceEnablePeerAccess.restype = ctypes.c_int
            lib.hipGetLastError.argtypes = []
            lib.hipGetLastError.restype = ctypes.c_int
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

    ``GPTQMODEL_P2P_PROBE_MIB`` sets the preferred payload; the ladder below is
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


def enable_peer_access(devices: Sequence | None = None, *, strict: bool = False) -> PeerStatus:
    """Enable peer access for every directed pair and classify each link.

    PyTorch enables peer access lazily on its first cross-device copy, so calling
    this beforehand only removes that first-copy latency; it is not what keeps
    copies on the interconnect.

    ``enabled``   - capability reported and enablement succeeded.
    ``unsupported`` - capability is false or enablement failed.

    When ``GPTQMODEL_P2P_PROBE=1`` a measured copy additionally downgrades a
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
        status.reason = f"{ENV_P2P}=0"
        _LAST_STATUS = status
        log.debug("%s", status.summary())
        return status

    checking = _cache_namespace()
    cache_key = (mode, checking, indices)
    cached = _STATUS_CACHE.get(cache_key)
    if cached is not None:
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
        for row, src in enumerate(indices):
            for col, dst in enumerate(indices):
                if src == dst:
                    continue
                # ``GPTQMODEL_P2P=1`` is an explicit request to try the
                # enablement call even when the capability probe says no.
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

                if not diagnose:
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
                    # ``GPTQMODEL_P2P=1`` is an explicit request for the direct
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
            "%s forced peer access on %s, but the capability query denies those "
            "directions; they are treated as unusable and staged through host "
            "memory because a direct peer copy can crash the process.",
            ENV_P2P,
            pairs,
        )
    log.info("%s", status.summary())
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
