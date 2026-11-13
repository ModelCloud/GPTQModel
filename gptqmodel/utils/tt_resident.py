# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Device-resident activations for the native TTNN backend.

On CUDA the model is moved to the device once and every op in between stays
there. The native TTNN backend cannot do that implicitly, because only the
quantized linear layers run on the Tensix cores: everything between them
(attention matmuls, softmax, norms, residual adds) is ordinary PyTorch on the
host. A ``forward`` that takes a CPU tensor and returns a CPU tensor therefore
crosses PCIe **twice per layer** — ~144 crossings per token for OPT-125M.

Measure carefully before optimising those crossings: on a p300c one is about
0.12 ms at decode sizes, while re-expanding the packed weights costs ~1.27 ms
per layer. Residency removes the former (measured 1.13x on a warmed chain), so
it is worth having for the contract and the upload count, but the dequant is
where the time is.

This module gives the native path the missing piece: an activation that lives
on the device across several layers. :class:`TTDeviceTensor` is an opaque
handle to a ``ttnn`` tensor, and :func:`run_resident` executes a chain of
resident-aware layers with **one upload and one readback for the whole chain**
instead of two per layer.

Scope, stated plainly: this removes the crossings *inside* a chain. It cannot
remove the crossings around host-side ops, because a chain is broken wherever
the graph needs a host op (attention, softmax, a norm). Full single-upload
inference needs those ops on device too, which is a model runner rather than a
linear replacement; :func:`resident_boundaries` reports how many crossings a
given layer sequence still costs so the remaining gap is measurable instead of
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from .logger import setup_logger


log = setup_logger()

#: Activation names that can be applied on device between resident layers.
DEVICE_ACTIVATIONS = ("relu", "gelu", "silu", "none")

_TRANSFERS = {"uploads": 0, "readbacks": 0, "bytes_up": 0, "bytes_down": 0}


def transfer_stats() -> dict:
    """Return cumulative activation transfer counters for this process."""

    return dict(_TRANSFERS)


def reset_transfer_stats() -> None:
    """Reset the transfer counters (tests and benchmarks)."""

    for key in _TRANSFERS:
        _TRANSFERS[key] = 0


def _charge(direction: str, numel: int, itemsize: int = 2) -> None:
    _TRANSFERS[direction] += 1
    _TRANSFERS["bytes_up" if direction == "uploads" else "bytes_down"] += numel * itemsize


@dataclass
class TTDeviceTensor:
    """An activation that is already resident on the Tenstorrent device.

    Holds a ``ttnn`` tensor plus the mesh that owns it. ``batch_shape`` keeps
    the leading dimensions the caller saw so a resident chain can hand back a
    correctly shaped host tensor at the end without the layers in between
    needing to care about anything but ``[rows, features]``.
    """

    tensor: Any
    mesh: Any
    batch_shape: tuple
    features: int
    layout_owner: str = "ttnn"

    @property
    def rows(self) -> int:
        total = 1
        for dim in self.batch_shape:
            total *= int(dim)
        return total

    def readback(self) -> torch.Tensor:
        """Copy the activation to host memory and release the device copy."""

        from .ttnn_runtime import import_ttnn, to_host_replica

        ttnn = import_ttnn()
        try:
            # A resident chain on a multi-chip mesh ends with a sharded tensor,
            # so the conversion needs the same composer the ordinary forward
            # uses; without it the runtime rejects the multi-shard tensor.
            host = to_host_replica(ttnn, self.tensor, self.mesh)
        finally:
            self.release()
        _charge("readbacks", host.numel())
        return host.reshape(tuple(self.batch_shape) + (self.features,))

    def release(self) -> None:
        """Free the device copy, if it is still live."""

        tensor, self.tensor = self.tensor, None
        if tensor is None:
            return
        try:
            from .ttnn_runtime import import_ttnn

            import_ttnn().deallocate(tensor)
        except Exception as exc:  # pragma: no cover - teardown must not raise
            log.debug(f"TTResident: device activation release failed: {exc}")


def is_device_tensor(value: Any) -> bool:
    """Return whether ``value`` is a resident activation owned by this module."""

    return isinstance(value, TTDeviceTensor)


def upload(host: torch.Tensor, *, mesh, features: Optional[int] = None) -> TTDeviceTensor:
    """Move a host activation onto the device once, for reuse by several layers."""

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    dev = mesh.device
    batch_shape = tuple(host.shape[:-1])
    width = int(features if features is not None else host.shape[-1])
    flat = host.reshape(-1, width).contiguous()
    if mesh.num_devices == 1:
        tensor = ttnn.to_device(
            ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dev
        )
    else:
        tensor = ttnn.to_device(
            ttnn.from_torch(
                flat,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
            ),
            dev,
        )
    _charge("uploads", flat.numel())
    return TTDeviceTensor(
        tensor=tensor, mesh=mesh, batch_shape=batch_shape, features=width
    )


def apply_activation(value: TTDeviceTensor, name: str) -> TTDeviceTensor:
    """Apply a device-side activation, keeping the result resident."""

    if name in (None, "", "none"):
        return value
    if name not in DEVICE_ACTIVATIONS:
        raise NotImplementedError(
            f"no device-side activation for {name!r}; supported: {DEVICE_ACTIVATIONS}"
        )
    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    previous = value.tensor
    value.tensor = getattr(ttnn, name)(previous)
    try:
        ttnn.deallocate(previous)
    except Exception:  # pragma: no cover - best-effort release
        pass
    return value


def add_residual(value: TTDeviceTensor, residual: TTDeviceTensor) -> TTDeviceTensor:
    """Add two resident activations on device."""

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    value.tensor = ttnn.add(value.tensor, residual.tensor)
    return value


def run_resident(
    modules: Sequence[Any],
    activation: TTDeviceTensor,
    *,
    activation_fn: str = "none",
    residual: Optional[TTDeviceTensor] = None,
) -> TTDeviceTensor:
    """Run consecutive resident-aware layers with the activation staying on device.

    ``modules`` must be resident-aware (``forward_resident``), which for GPTQ
    means :class:`~gptqmodel.nn_modules.qlinear.ttnn.TTNNLinear`. The activation
    is uploaded once by :func:`upload` and read back once by
    :meth:`TTDeviceTensor.readback`, regardless of how many layers run between.
    """

    current = activation
    for index, module in enumerate(modules):
        current = module.forward_resident(current)
        if activation_fn not in (None, "", "none") and index < len(modules) - 1:
            current = apply_activation(current, activation_fn)
    if residual is not None:
        current = add_residual(current, residual)
    return current


def run_resident_host(
    modules: Sequence[Any],
    host: torch.Tensor,
    *,
    mesh,
    activation_fn: str = "none",
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience wrapper: upload once, run the chain, read back once."""

    resident = upload(host, mesh=mesh)
    resident_residual = upload(residual, mesh=mesh) if residual is not None else None
    try:
        out = run_resident(
            modules, resident, activation_fn=activation_fn, residual=resident_residual
        )
        return out.readback()
    finally:
        if resident_residual is not None:
            resident_residual.release()


def resident_boundaries(sequence: Sequence[bool]) -> int:
    """Count host<->device crossings a layer sequence still costs.

    ``sequence`` marks each op as device-resident (``True``) or host-side
    (``False``). Every ``False`` breaks the chain, so the cost is two crossings
    per maximal resident run. This makes the remaining gap explicit: a model
    whose attention and norms stay on the host cannot reach one upload.
    """

    crossings = 0
    in_run = False
    for resident in sequence:
        if resident and not in_run:
            crossings += 2  # upload + readback for this run
            in_run = True
        elif not resident:
            in_run = False
    return crossings


__all__ = [
    "DEVICE_ACTIVATIONS",
    "TTDeviceTensor",
    "add_residual",
    "apply_activation",
    "is_device_tensor",
    "reset_transfer_stats",
    "resident_boundaries",
    "run_resident",
    "run_resident_host",
    "transfer_stats",
    "upload",
]
