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
from dataclasses import dataclass
from typing import Any, Optional

from .logger import setup_logger


log = setup_logger()

_LOCK = threading.RLock()
_MESH: Optional["TTNNMesh"] = None


class TTNNUnavailable(RuntimeError):
    """Raised when the native TTNN stack cannot satisfy a request."""


@dataclass(frozen=True)
class TTNNMesh:
    """An open ``ttnn.MeshDevice`` plus the fabric mode it was opened with."""

    device: Any
    mesh_shape: tuple[int, ...]
    fabric: str
    owner: str

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
) -> TTNNMesh:
    """Open (or reuse) the process-wide mesh for the native TTNN backend.

    ``device_count == 1`` opens a single ``ttnn.Device``; larger counts open a
    one-dimensional ``MeshShape(1, N)`` mesh, which is the layout the
    tensor-parallel GPTQ kernel shards ``out_features`` across.  Only one
    backend may own the chips at a time, so a PJRT-owned runtime is rejected
    rather than left to fail inside TT-Metal.
    """

    global _MESH

    ttnn = import_ttnn()
    if device_count < 1:
        raise ValueError(f"device_count must be >= 1, got {device_count}")

    with _LOCK:
        if _MESH is not None:
            if _MESH.owner != owner:
                raise TTNNUnavailable(
                    f"the Tenstorrent chips are already owned by {_MESH.owner!r}; "
                    f"{owner!r} cannot open a second runtime in this process"
                )
            if _MESH.mesh_shape and _prod(_MESH.mesh_shape) != device_count:
                raise TTNNUnavailable(
                    f"an existing {_MESH.mesh_shape} TTNN mesh cannot serve "
                    f"device_count={device_count}"
                )
            return _MESH

        _reject_pjrt_owner()

        fabric = (fabric or "DISABLED").upper()
        applied_fabric = "DISABLED"
        if fabric != "DISABLED":
            configured = getattr(ttnn.FabricConfig, fabric, None)
            if configured is None:
                raise ValueError(f"unknown TTNN fabric config {fabric!r}")
            ttnn.set_fabric_config(configured)
            applied_fabric = fabric

        mesh_shape: tuple[int, ...] = ()
        device = None
        try:
            if device_count == 1:
                device = ttnn.open_device(device_id=0)
            else:
                mesh_shape = (1, device_count)
                device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
        except Exception as exc:
            if applied_fabric != "DISABLED":
                try:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                except Exception:
                    pass
            raise _translate_open_failure(exc) from exc

        _MESH = TTNNMesh(
            device=device,
            mesh_shape=mesh_shape,
            fabric=applied_fabric,
            owner=owner,
        )
        log.info(
            f"TTNN: opened {'mesh ' + str(mesh_shape) if mesh_shape else 'device 0'} "
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
        mesh.close()


def current_mesh() -> Optional[TTNNMesh]:
    """Return the open mesh without taking ownership or opening anything."""

    with _LOCK:
        return _MESH


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


__all__ = [
    "TTNNMesh",
    "TTNNUnavailable",
    "close_mesh",
    "current_mesh",
    "import_ttnn",
    "is_ttnn_available",
    "open_mesh",
    "ttnn_device_count",
]
