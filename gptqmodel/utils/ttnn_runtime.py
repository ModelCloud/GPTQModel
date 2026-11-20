# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Process-wide native TTNN device ownership for Tenstorrent kernels.

Torch/XLA (``XLALinear``) reaches the chips through the PJRT plugin, which owns
its own ``MeshDevice``.  A native TTNN kernel talks to ``ttnn.MeshDevice``
directly, and TT-Metal permits exactly one owner per chip, so the two paths can
never be open at the same time.  This module owns that singleton, records which
backend holds it, and refuses to hand out a mismatched device instead of
deadlocking inside the driver.

Fabric is optional: tensor-parallel collectives such as ``all_gather`` require
``set_fabric_config`` before the mesh is opened.  ``open_mesh`` takes that
decision from the caller so a single-chip run does not pay the fabric bring-up
cost, and records the applied configuration so it can be released on close.
"""

from __future__ import annotations

import importlib
import sys
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .logger import setup_logger
from .tt_thermal import after_device_close, before_device_open, park_idle_clock


log = setup_logger()

_LOCK = threading.RLock()
_MESH: Optional["TTNNMesh"] = None
_MESH_GENERATION = 0
# Live users of the open mesh.  Layers register when they upload their packed
# tensors and drop the lease in ``release_device``; the mesh closes only after
# the last lease is gone.  A plain module attribute would keep released layers
# alive, so the registry holds weak references.
_LEASES: "weakref.WeakSet[Any]" = weakref.WeakSet()


class TTNNUnavailable(RuntimeError):
    """Raised when the native TTNN stack cannot satisfy a request."""


@dataclass(frozen=True)
class TTNNMesh:
    """An open ``ttnn.MeshDevice`` plus the request it was opened for.

    ``device_id`` and ``trace_region_size`` are recorded next to the fabric mode
    because they are part of the placement the caller asked for: a later
    ``open_mesh`` may only reuse this handle when every one of them matches.
    ``mesh_shape == ()`` means a single chip, so it cannot carry the ordinal;
    ``device_id`` is where that chip identity lives.
    """

    device: Any
    mesh_shape: tuple[int, ...]
    fabric: str
    owner: str
    device_id: int = 0
    trace_region_size: int = 0
    #: Monotonic per-process open counter. Device-resident caches include it so
    #: a buffer built for a closed mesh can never match a later handle, even if
    #: Python recycles the device object's address.
    generation: int = 0

    @property
    def num_devices(self) -> int:
        return 1 if not self.mesh_shape else _prod(self.mesh_shape)

    def close(self) -> None:
        ttnn = import_ttnn()
        try:
            if self.mesh_shape:
                ttnn.close_mesh_device(self.device)
            else:
                ttnn.close_device(self.device)
        finally:
            if self.fabric != "DISABLED":
                try:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                except Exception:  # pragma: no cover - teardown must not mask errors
                    log.warning("TTNN: fabric config was not restored on close.")


def _prod(values) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def import_ttnn():
    """Import ``ttnn`` or raise an actionable error.

    ``ttnn`` is only installed by the pinned Tenstorrent environment, and a
    failed import is not a kernel bug, so the message names the requirement
    file instead of surfacing a bare ``ModuleNotFoundError``.
    """

    try:
        return importlib.import_module("ttnn")
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise TTNNUnavailable(
            "the native Tenstorrent TTNN runtime is not importable; install the "
            "pinned environment from `requirements/tenstorrent-py312.txt`"
        ) from exc


def is_ttnn_available() -> bool:
    """Probe TTNN importability without opening a device.

    Device discovery touches the driver and can reset chips, so availability
    checks must not query the cluster.  Callers that need a device count use
    :func:`open_mesh`, which is explicit about taking ownership.
    """

    try:
        import_ttnn()
    except TTNNUnavailable:
        return False
    return True


def ttnn_device_count() -> int:
    """Return the visible chip count without keeping a device open."""

    ttnn = import_ttnn()
    try:
        # ``get_num_pcie_devices`` reads the cluster description only; it does
        # not open a mesh, so it is safe next to a PJRT-owned runtime.
        return int(ttnn.get_num_pcie_devices())
    except Exception as exc:
        try:
            return int(ttnn.GetNumPCIeDevices())
        except Exception:
            raise _translate_open_failure(exc) from exc


_OWNER_CONFLICT_MARKERS = (
    "num_initialized_routers",
    "has not been initialized",
    "already owned",
    "Cannot close device",
    "context_id",
    "is invalid",
)


def _translate_open_failure(exc: BaseException) -> BaseException:
    """Turn a TT-Metal ownership/init failure into an actionable error.

    TT-Metal enforces one runtime owner per chip, but reports a violation as a
    bare ``TT_FATAL`` from the fabric router initializer.  Re-raising that text
    leaves the user with an assert address and no cause.  The markers below are
    the shapes that failure takes when another runtime (normally the Torch/XLA
    PJRT plugin) already owns the devices.
    """

    text = str(exc)
    if any(marker in text for marker in _OWNER_CONFLICT_MARKERS):
        return TTNNUnavailable(
            "TT-Metal could not initialize the Tenstorrent devices because they "
            "are already owned by another runtime in this process, normally the "
            "Torch/XLA PJRT plugin. One runtime may own a chip at a time: run the "
            "native TTNN backend in a process that does not load the XLA path "
            f"(or close it first). Original error: {text.splitlines()[0]}"
        )
    return exc


def open_mesh(
    *,
    device_count: int = 1,
    fabric: str = "DISABLED",
    owner: str = "gptqmodel",
    trace_region_size: int = 0,
    device_id: int = 0,
) -> TTNNMesh:
    """Open (or reuse) the process-wide mesh for the native TTNN backend.

    ``device_count == 1`` opens a single ``ttnn.Device``; larger counts open a
    one-dimensional ``MeshShape(1, N)`` mesh, which is the layout the
    tensor-parallel GPTQ kernel shards ``out_features`` across.  Only one
    backend may own the chips at a time, so a PJRT-owned runtime is rejected
    rather than left to fail inside TT-Metal.

    ``trace_region_size`` reserves device memory for captured traces (a traced
    decode step needs one).  ``device_id`` picks the chip for a single-device
    open, which is how one process per card is run.

    A live handle is reused only when it was opened for the same owner, chip
    count, chip ordinal, fabric mode and trace reservation.  Anything else
    raises :class:`TTNNUnavailable` instead of returning a device with different
    placement than the caller asked for; changing placement requires an
    explicit :func:`close_mesh` first.
    """

    global _MESH, _MESH_GENERATION

    ttnn = import_ttnn()
    if device_count < 1:
        raise ValueError(f"device_count must be >= 1, got {device_count}")
    device_count = int(device_count)
    device_id = int(device_id)
    trace_region_size = int(trace_region_size)
    # Normalise and validate the request before touching process state, so an
    # unknown fabric or a mesh offset fails even when a handle already exists.
    fabric = (fabric or "DISABLED").upper()
    configured_fabric = None
    if fabric != "DISABLED":
        configured_fabric = getattr(ttnn.FabricConfig, fabric, None)
        if configured_fabric is None:
            raise ValueError(f"unknown TTNN fabric config {fabric!r}")
    if device_count > 1 and device_id:
        raise ValueError("device_id selects a single chip; it cannot offset a mesh")

    with _LOCK:
        if _MESH is not None:
            if _MESH.owner != owner:
                raise TTNNUnavailable(
                    f"the Tenstorrent chips are already owned by {_MESH.owner!r}; "
                    f"{owner!r} cannot open a second runtime in this process"
                )
            mismatched = []
            if _MESH.num_devices != device_count:
                mismatched.append(f"device_count={device_count} (open: {_MESH.num_devices})")
            if _MESH.device_id != device_id:
                mismatched.append(f"device_id={device_id} (open: {_MESH.device_id})")
            if _MESH.fabric != fabric:
                mismatched.append(f"fabric={fabric} (open: {_MESH.fabric})")
            if _MESH.trace_region_size != trace_region_size:
                mismatched.append(
                    f"trace_region_size={trace_region_size} "
                    f"(open: {_MESH.trace_region_size})"
                )
            if mismatched:
                raise TTNNUnavailable(
                    "the open TTNN device cannot serve this request: "
                    + ", ".join(mismatched)
                    + "; close it and reopen explicitly to change placement"
                )
            return _MESH

        _reject_pjrt_owner()

        # The chips are idle here by construction: no TT-Metal runtime owns them,
        # so PCI telemetry is safe to read. Pace the open when the previous model
        # left the GDDR hot, and give the parts a fixed idle window otherwise.
        before_device_open(owner=owner)

        applied_fabric = "DISABLED"
        if configured_fabric is not None:
            ttnn.set_fabric_config(configured_fabric)
            applied_fabric = fabric

        mesh_shape: tuple[int, ...] = ()
        device = None
        try:
            extra = {"trace_region_size": trace_region_size} if trace_region_size else {}
            if device_count == 1:
                device = ttnn.open_device(device_id=device_id, **extra)
            else:
                mesh_shape = (1, device_count)
                device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape), **extra)
        except Exception as exc:
            if applied_fabric != "DISABLED":
                try:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                except Exception as cleanup_exc:
                    # The open already failed, so this cleanup must not mask the
                    # real error. Record it and let the translated open failure
                    # below reach the caller.
                    log.debug(f"TTNN: fabric reset failed during cleanup: {cleanup_exc}")
            raise _translate_open_failure(exc) from exc

        _MESH_GENERATION += 1
        _MESH = TTNNMesh(
            device=device,
            mesh_shape=mesh_shape,
            fabric=applied_fabric,
            owner=owner,
            device_id=device_id,
            trace_region_size=trace_region_size,
            generation=_MESH_GENERATION,
        )
        log.info(
            f"TTNN: opened "
            f"{'mesh ' + str(mesh_shape) if mesh_shape else f'device {device_id}'} "
            f"for {owner!r} (fabric={applied_fabric})."
        )
        return _MESH


def close_mesh(*, owner: Optional[str] = None) -> None:
    """Close the process-wide mesh, optionally only when ``owner`` matches."""

    global _MESH

    with _LOCK:
        if _MESH is None:
            return
        if owner is not None and _MESH.owner != owner:
            raise TTNNUnavailable(
                f"refusing to close the Tenstorrent runtime owned by {_MESH.owner!r} "
                f"from {owner!r}"
            )
        mesh, _MESH = _MESH, None
        _LEASES.clear()
        mesh.close()
        # Every cached grid operand is device-resident, so it died with this
        # device. Dropping the entries here also keeps a recycled object id from
        # ever matching a buffer that belonged to the closed device.
        try:
            from .tt_int4_kernel import clear_grid_weight_cache

            clear_grid_weight_cache()
        except Exception as exc:  # pragma: no cover - teardown must not mask errors
            log.debug(f"TTNN: could not clear the grid weight cache on close: {exc}")
        # Closing the device does not stop the card generating heat: measured on
        # a p300c, AICLK stayed at 1350 MHz and ~50 W per board for over 50 s
        # after ``close_device``, with GDDR still climbing. The firmware drops
        # to its 800 MHz long-idle clock only when told to, so ask for it before
        # the idle window or the cooldown is not actually idle.
        try:
            park_idle_clock()
        except Exception as exc:  # pragma: no cover - teardown must not mask errors
            log.debug(f"TTNN: could not park the idle clock on close: {exc}")
        # Pace the next open only after the device is genuinely closed, so the
        # idle window is not spent with the chips still running.
        after_device_close(owner=mesh.owner)


def current_mesh() -> Optional[TTNNMesh]:
    """Return the open mesh without taking ownership or opening anything."""

    with _LOCK:
        return _MESH


def hold_mesh_lease(owner: Any) -> None:
    """Record that ``owner`` (normally a kernel module) is using the mesh.

    Every upload path calls this after ``open_mesh`` succeeds, so the model
    wrapper can release only devices its own kernels hold.  A repeated call is
    a no-op, which keeps a cache rebuild from inflating the lease count.
    """

    with _LOCK:
        if _MESH is None:
            raise TTNNUnavailable("cannot hold a lease on a closed TTNN mesh")
        _LEASES.add(owner)


def release_mesh_lease(owner: Any) -> bool:
    """Drop ``owner``'s mesh lease.

    Returns ``True`` only when ``owner`` held the last live lease, which is the
    caller's signal to close the shared mesh.  Releasing an owner that never
    acquired one returns ``False`` so an unleased or already-released handle is
    left untouched.
    """

    with _LOCK:
        if owner not in _LEASES:
            return False
        _LEASES.discard(owner)
        return not _LEASES


def _reject_pjrt_owner() -> None:
    """Detect an already-initialized PJRT runtime without initializing one.

    ``torch_xla.runtime.addressable_device_count()`` *initializes* PJRT and
    claims every chip, so calling it to check for an owner would itself become
    the conflicting owner.  The probe therefore only inspects what the import
    left behind: whether ``torch_xla`` is already in ``sys.modules`` *and* the
    process already holds XLA device state.  That keeps the common case (a
    fresh native TTNN process) free of any PJRT interaction.

    TT-Metal remains the real enforcement point: it fails closed if two owners
    race.  This check only turns the common mistake into an actionable message.
    """

    if "torch_xla" not in sys.modules:
        return
    try:
        import torch_xla
    except Exception:
        return
    # Importing is inert; only a bound XLA device means the runtime is live.
    if getattr(torch_xla, "_gptqmodel_device_claimed", False):
        raise TTNNUnavailable(
            "a Torch/XLA PJRT runtime has already claimed the Tenstorrent "
            "devices in this process; TT-Metal allows one owner per chip, so "
            "close the XLA path (or start a fresh process) before using TTNNLinear."
        )


def mesh_device_ids(mesh: TTNNMesh) -> Tuple[int, ...]:
    """Chip ordinals this mesh occupies.

    A single-chip handle records the requested ordinal itself; a multi-chip
    mesh exposes the physical chip ids it was opened on (for example ``(0, 3)``
    for two chips in a four-chip cluster).
    """

    if int(getattr(mesh, "num_devices", 1) or 1) <= 1:
        return (int(getattr(mesh, "device_id", 0) or 0),)
    device = getattr(mesh, "device", None)
    ids = device.get_device_ids() if device is not None else ()
    return tuple(int(index) for index in ids)


def dram_allocated_bytes(device) -> int:
    """(measured) DRAM bytes one chip's allocator currently holds.

    ``get_memory_view`` reports per-bank totals, so the chip total is
    ``num_banks * total_bytes_allocated_per_bank``. On a mesh the view is per
    chip, not summed across chips: a replicated tensor reports the full tensor
    on each chip and a shard reports only its own slice.

    Reading the allocator includes what the tensor payload alone hides: tile
    padding, per-page rounding and replicated operands (#5810).
    """

    ttnn = import_ttnn()
    ttnn.synchronize_device(device)
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return int(view.num_banks) * int(view.total_bytes_allocated_per_bank)


def to_host_replica(ttnn, tensor, mesh):
    """Copy a mesh tensor to host memory, keeping one replica.

    A tensor produced on a multi-chip mesh is sharded across devices, so the
    conversion needs a composer: the pinned runtime rejects a multi-shard
    tensor without one. After an ``all_gather`` the shards are identical, so
    the concatenation is trimmed back to a single replica.

    Shared by the ordinary host forward and by ``TTDeviceTensor.readback`` so
    the two cannot disagree about multi-chip conversion.
    """

    count = int(getattr(mesh, "num_devices", 1) or 1)
    if count <= 1:
        return ttnn.to_torch(tensor)
    composed = ttnn.to_torch(
        tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh.device, dim=0)
    )
    return composed[: composed.shape[0] // count]


__all__ = [
    "TTNNMesh",
    "TTNNUnavailable",
    "close_mesh",
    "current_mesh",
    "dram_allocated_bytes",
    "hold_mesh_lease",
    "import_ttnn",
    "is_ttnn_available",
    "mesh_device_ids",
    "open_mesh",
    "release_mesh_lease",
    "to_host_replica",
    "ttnn_device_count",
]
