# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Best-effort GPU peer-access (P2P) detection, enablement and direct copy.

PyTorch exposes ``torch.cuda.can_device_access_peer`` but no enablement API, so
enablement goes through ``ctypes``: ``hipDeviceEnablePeerAccess`` on ROCm and
``cuCtxEnablePeerAccess`` on CUDA.  ``hipErrorPeerAccessAlreadyEnabled`` /
``CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`` (704) is treated as success, which
makes repeated calls idempotent.

Peer access is *directed*: ``i -> j`` and ``j -> i`` are enabled separately, and
both directions are queried.

A capability bit is not proof of usable bandwidth.  Consumer cards (for example
RTX 4090) commonly report ``can_device_access_peer == True`` while throughput is
below a host-staged copy.  Such pairs are therefore classified ``degraded``
after a measured bandwidth gate and forced onto host staging.  Never assert P2P
is effective from the call stack or from the capability bit alone.

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


log = setup_logger()

ENV_P2P = "GPTQMODEL_P2P"
ENV_P2P_MIN_GBPS = "GPTQMODEL_P2P_MIN_GBPS"
ENV_P2P_MIN_RATIO = "GPTQMODEL_P2P_MIN_RATIO"

# "auto" enables P2P when the runtime reports a peer link; "1" forces the
# enablement attempt; "0" disables P2P entirely (host staging everywhere).
_MODE_AUTO = "auto"
_MODE_ON = "1"
_MODE_OFF = "0"
_OFF_VALUES = frozenset({"0", "false", "off", "no", "disable", "disabled"})
_ON_VALUES = frozenset({"1", "true", "on", "yes", "enable", "enabled"})

DEFAULT_MIN_GBPS = 5.0
# A direct link must also beat host staging by this ratio. An absolute GB/s
# floor alone cannot detect "the capability bit is set but the transfer is not
# actually direct": measured on 2x RTX 4090 (0.998x) and 4x R9700 / gfx1201
# (0.954x-0.99x) with a 128 MiB payload, the direct copy ran at 0.95x-1.00x of a
# pinned-host staged copy, i.e. it was no faster than the fallback (and usually
# slower) while still clearing any low absolute threshold.
DEFAULT_MIN_RATIO = 1.0

STATE_ENABLED = "enabled"
STATE_UNSUPPORTED = "unsupported"
STATE_DEGRADED = "degraded"

_PEER_ALREADY_ENABLED = 704

_STATUS_CACHE: dict[tuple[str, tuple[int, ...]], "PeerStatus"] = {}
_BANDWIDTH_CACHE: dict[tuple[int, int], tuple[float | None, float | None]] = {}


@dataclass
class PeerStatus:
    """Result of a peer-access probe; one entry per directed device pair."""

    mode: str = _MODE_AUTO
    devices: tuple[int, ...] = ()
    pairs: dict[tuple[int, int], str] = field(default_factory=dict)
    bandwidth_gbps: dict[tuple[int, int], float | None] = field(default_factory=dict)
    host_staged_gbps: dict[tuple[int, int], float | None] = field(default_factory=dict)
    reason: str = ""

    def state(self, src: int, dst: int) -> str:
        """Return the per-direction state (defaults to ``unsupported``)."""
        return self.pairs.get((src, dst), STATE_UNSUPPORTED)

    def direct_copy_allowed(self) -> bool:
        """True when at least one directed pair is enabled and none degraded."""
        return any(state == STATE_ENABLED for state in self.pairs.values()) and not any(
            state == STATE_DEGRADED for state in self.pairs.values()
        )

    def can_copy(self, src: int, dst: int) -> bool:
        """True only when ``src -> dst`` measured as a healthy peer link."""
        return self.state(src, dst) == STATE_ENABLED

    def summary(self) -> str:
        counts = {STATE_ENABLED: 0, STATE_UNSUPPORTED: 0, STATE_DEGRADED: 0}
        for state in self.pairs.values():
            counts[state] = counts.get(state, 0) + 1
        paired = ", ".join(
            f"{src}->{dst}:{state}" for (src, dst), state in sorted(self.pairs.items())
        ) or "none"
        reason = f"; reason={self.reason}" if self.reason else ""
        return (
            f"P2P mode={self.mode} devices={list(self.devices)} "
            f"enabled={counts[STATE_ENABLED]} unsupported={counts[STATE_UNSUPPORTED]} "
            f"degraded={counts[STATE_DEGRADED]} [{paired}]{reason}"
        )


def clear_p2p_cache() -> None:
    """Drop cached probe/bandwidth results (tests, device hot-plug)."""
    _STATUS_CACHE.clear()
    _BANDWIDTH_CACHE.clear()


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


def _normalize_indices(devices: Iterable | None) -> tuple[int, ...]:
    if devices is None:
        count = torch.cuda.device_count()
        return tuple(range(count))
    indices: list[int] = []
    for candidate in devices:
        if isinstance(candidate, int):
            index = candidate
        elif isinstance(candidate, torch.device):
            index = candidate.index if candidate.index is not None else torch.cuda.current_device()
        else:
            index = torch.device(str(candidate)).index or 0
        if index not in indices:
            indices.append(index)
    return tuple(indices)


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
    except Exception:  # pragma: no cover - never let cleanup raise
        pass


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
            staged.copy_(source, non_blocking=True)
            target.copy_(staged, non_blocking=True)

        return _timed_gbps(step, src, dst, bytes_moved, iterations)
    except Exception as exc:
        log.debug("P2P host-staged probe %s->%s failed: %s", src, dst, exc)
        return None


def _probe_bandwidth(src: int, dst: int) -> tuple[float | None, float | None]:
    """Return ``(direct_gbps, host_staged_gbps)`` for one directed pair."""
    key = (src, dst)
    if key in _BANDWIDTH_CACHE:
        return _BANDWIDTH_CACHE[key]
    gib = int(os.getenv("GPTQMODEL_P2P_PROBE_MIB", "16")) * 1024 * 1024
    direct = _measure_gbps(src, dst, gib)
    staged = _measure_host_staged_gbps(src, dst, gib) if direct is not None else None
    _BANDWIDTH_CACHE[key] = (direct, staged)
    return direct, staged


def enable_peer_access(devices: Sequence | None = None, *, strict: bool = False) -> PeerStatus:
    """Enable peer access for every directed pair and classify each link.

    ``enabled``   - capability reported and measured bandwidth passed the gate.
    ``unsupported`` - capability is false or enablement failed.
    ``degraded``  - capability is true but measured bandwidth is below the gate;
                    callers must use host staging for this direction.

    With ``strict=True`` an enablement failure raises; the default is
    best-effort and only logs one summary line.
    """

    mode = p2p_mode()
    indices = _normalize_indices(devices)
    status = PeerStatus(mode=mode, devices=indices)

    if mode == _MODE_OFF:
        status.reason = f"{ENV_P2P}=0"
        log.debug("%s", status.summary())
        return status

    cache_key = (mode, indices)
    if cache_key in _STATUS_CACHE:
        return _STATUS_CACHE[cache_key]

    if len(indices) < 2 or not torch.cuda.is_available():
        status.reason = "fewer than two CUDA/HIP devices"
        _STATUS_CACHE[cache_key] = status
        log.debug("%s", status.summary())
        return status

    threshold = _min_gbps()
    min_ratio = _min_ratio()
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
                available = capability[row][col] or mode == _MODE_ON
                if not available:
                    status.pairs[(src, dst)] = STATE_UNSUPPORTED
                    continue
                ok, detail = _enable_one(src, dst)
                if not ok:
                    status.pairs[(src, dst)] = STATE_UNSUPPORTED
                    log.debug("P2P %s->%s not enabled: %s", src, dst, detail)
                    if strict:
                        raise RuntimeError(f"P2P {src}->{dst} enablement failed: {detail}")
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
                elif staged and measured < staged * min_ratio:
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
            except Exception:
                pass

    _STATUS_CACHE[cache_key] = status
    log.info("%s", status.summary())
    return status


def copy_tensor_to(dst: torch.Tensor, src: torch.Tensor, status: PeerStatus | None = None) -> torch.Tensor:
    """Copy ``src`` into ``dst`` directly when the measured P2P link is healthy.

    Falls back to host staging (which is what torch does for a non-peer copy)
    for ``unsupported``/``degraded`` directions.  ``dst`` is returned.
    """

    if status is None or dst.device.type != "cuda" or src.device.type != "cuda":
        dst.copy_(src, non_blocking=True)
        return dst

    src_index = src.device.index if src.device.index is not None else torch.cuda.current_device()
    dst_index = dst.device.index if dst.device.index is not None else torch.cuda.current_device()
    if src.device == dst.device or status.can_copy(src_index, dst_index):
        dst.copy_(src, non_blocking=True)
        return dst

    staged = src.to("cpu", non_blocking=True)
    dst.copy_(staged, non_blocking=True)
    return dst


def enable_for_devices(devices: Sequence | None = None, *, strict: bool = False) -> PeerStatus:
    """Convenience wrapper used by the quantization loops (best effort)."""
    try:
        return enable_peer_access(devices, strict=strict)
    except Exception as exc:  # pragma: no cover - defensive
        status = PeerStatus(mode=p2p_mode(), devices=_normalize_indices(devices), reason=str(exc))
        log.info("%s", status.summary())
        return status
