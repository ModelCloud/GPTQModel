# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pure-Torch grouped P32 semantics for the A41/R0 execution plan.

This module deliberately sits above the native QVQ kernels.  It describes
which already-quantized children may share the input transform and provides a
slow, FP32 implementation that is used as the semantic oracle for future
Ampere and Hopper grouped kernels.  Checkpoint payloads stay canonical; the
grouped payload is a transient, lossless concatenation along the output-tile
axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .qvq import (
    pack_qvq_binary_bank_ids,
    reconstruct_qvq_inner_weight,
    unpack_qvq_binary_bank_ids,
)
from .qvq_rates import qvq_words_per_tile
from .rotation.hadamard_utils import matmul_hadU


@dataclass(frozen=True)
class QVQExecutionDescriptor:
    """Execution metadata emitted by an architecture implementor.

    ``input_basis_id`` identifies the activation basis shared by sibling
    projections.  The scale vectors remain part of each :class:`QVQLinear`
    child; a group is legal only when the input ``SU`` vectors are identical.
    ``output_hadamard`` may differ between children because output recovery is
    always child-local.
    """

    module_name: str
    input_basis_id: str
    input_hadamard: bool = True
    output_hadamard: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str) or not self.module_name:
            raise ValueError("QVQ execution descriptor module_name must be non-empty")
        if not isinstance(self.input_basis_id, str) or not self.input_basis_id:
            raise ValueError("QVQ execution descriptor input_basis_id must be non-empty")
        if not isinstance(self.input_hadamard, bool) or not isinstance(self.output_hadamard, bool):
            raise TypeError("QVQ execution descriptor Hadamard flags must be bool")


@dataclass(frozen=True)
class QVQGroupedP32Spec:
    """The ordered members of one legal grouped execution."""

    basis_id: str
    ordered_members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.basis_id, str) or not self.basis_id:
            raise ValueError("QVQ grouped P32 basis_id must be non-empty")
        if not self.ordered_members:
            raise ValueError("QVQ grouped P32 requires at least one member")
        if any(not isinstance(name, str) or not name for name in self.ordered_members):
            raise ValueError("QVQ grouped P32 member names must be non-empty strings")
        if len(set(self.ordered_members)) != len(self.ordered_members):
            raise ValueError("QVQ grouped P32 member names must be unique")


@dataclass(frozen=True)
class QVQCanonicalP32Payload:
    """One child’s canonical, serialized P32 payload.

    ``bank_ids`` is normalized to one packed byte per K16xN16 tile.  The
    tensor is never modified by grouping; callers can use
    :func:`ungroup_canonical_p32_payload` to recover an exact packed copy.
    """

    trellis: torch.Tensor
    bank_ids: torch.Tensor
    bank_alt_id: torch.Tensor
    bits: float
    in_features: int
    out_features: int
    codebook_version: str
    vector_size: int = 2
    trellis_window: int = 16


@dataclass(frozen=True)
class QVQGroupedP32Segment:
    """Output-tile range and bank metadata for one grouped child."""

    module_name: str
    output_tile_start: int
    output_tile_count: int
    out_features: int
    bank_alt_id: torch.Tensor


@dataclass(frozen=True)
class QVQGroupedP32Payload:
    """Transient lossless concatenation of canonical child payloads."""

    trellis: torch.Tensor
    bank_ids: torch.Tensor
    segments: tuple[QVQGroupedP32Segment, ...]
    bits: float
    in_features: int
    codebook_version: str
    vector_size: int = 2
    trellis_window: int = 16

    @property
    def out_features(self) -> int:
        return sum(segment.out_features for segment in self.segments)


def _require_qvq_linear(child: object) -> None:
    # Import lazily so this semantic module does not participate in the
    # qlinear package's import cycle.
    from ..nn_modules.qlinear.qvq import QVQLinear

    if not isinstance(child, QVQLinear):
        raise TypeError(f"QVQ grouped P32 children must be QVQLinear, got {type(child).__name__}")


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and left.device == right.device
        and torch.equal(left, right)
    )


def _packed_binary_bank_ids(child: object) -> torch.Tensor:
    """Return the canonical packed selector byte for each P32 tile."""

    bank_ids = getattr(child, "bank_ids", None)
    if bank_ids is None:
        raise ValueError(f"QVQ P32 child `{getattr(child, 'name', '<unnamed>')}` has no bank_ids")
    tile_count = (int(child.in_features) // 16) * (int(child.out_features) // 16)
    selector_count = tile_count * 8
    if bank_ids.ndim != 1:
        raise ValueError("QVQ P32 bank_ids must be one-dimensional")
    if bank_ids.numel() == tile_count:
        # Validate every packed byte by round-tripping its eight binary
        # selectors.  The clone prevents a caller mutation from changing a
        # grouped payload after construction.
        unpack_qvq_binary_bank_ids(bank_ids, selector_count)
        return bank_ids.detach().clone().contiguous().to(torch.uint8)
    if bank_ids.numel() == selector_count:
        return pack_qvq_binary_bank_ids(bank_ids.detach()).contiguous()
    raise ValueError(
        f"QVQ P32 bank_ids must contain {tile_count} packed bytes or {selector_count} selectors, "
        f"got {bank_ids.numel()}"
    )


def _canonical_child_payload(child: object) -> QVQCanonicalP32Payload:
    _require_qvq_linear(child)
    if not getattr(child, "v2b2_p32", False):
        raise ValueError("A41/R0 Phase 1 supports only V2B2-P32 children")
    if int(child.vector_size) != 2 or int(child.trellis_window) != 16:
        raise ValueError("A41/R0 Phase 1 requires vector_size=2 and trellis_window=16")
    if int(child.bank_count) != 2:
        raise ValueError("A41/R0 Phase 1 requires two-bank V2B2-P32 children")
    if child.trellis.device.type == "meta":
        raise ValueError("A41/R0 payload grouping requires concrete tensors, not meta tensors")
    if child.bank_alt_id is None or tuple(child.bank_alt_id.shape) != (1,):
        raise ValueError("A41/R0 V2B2-P32 children require one bank_alt_id value")
    alt_id = child.bank_alt_id.detach().clone().contiguous()
    if alt_id.device != child.trellis.device:
        raise ValueError("QVQ P32 trellis, bank_ids, and bank_alt_id must share a device")
    if not bool(((alt_id >= 1) & (alt_id <= 3)).all()):
        raise ValueError("QVQ P32 bank_alt_id must be in [1, 3]")
    if child.trellis.dtype != torch.int32:
        raise TypeError(f"QVQ P32 trellis must use torch.int32, got {child.trellis.dtype}")
    k_tiles = int(child.in_features) // 16
    n_tiles = int(child.out_features) // 16
    if k_tiles < 1 or n_tiles < 1:
        raise ValueError("QVQ P32 child dimensions must be positive multiples of 16")
    expected_words = qvq_words_per_tile(
        float(child.bits), weight_count=16 * 16, vector_size=2
    )
    if tuple(child.trellis.shape) != (k_tiles * n_tiles, expected_words):
        raise ValueError(
            "QVQ P32 child trellis must have shape "
            f"`{(k_tiles * n_tiles, expected_words)}`, got `{tuple(child.trellis.shape)}`"
        )
    if child.bank_ids.device != child.trellis.device:
        raise ValueError("QVQ P32 trellis and bank_ids must share a device")
    return QVQCanonicalP32Payload(
        trellis=child.trellis.detach().clone().contiguous(),
        bank_ids=_packed_binary_bank_ids(child),
        bank_alt_id=alt_id,
        bits=float(child.bits),
        in_features=int(child.in_features),
        out_features=int(child.out_features),
        codebook_version=str(child.codebook_version).strip().lower(),
    )


def validate_group(
    children: Sequence[object],
    descriptors: Sequence[QVQExecutionDescriptor],
    spec: QVQGroupedP32Spec | None = None,
) -> QVQGroupedP32Spec:
    """Validate the exact R0 legality contract and return the group spec.

    A failed validation is a normal fallback condition for R0: callers should
    run ordinary per-module P32 rather than approximating or mutating a child.
    """

    if not children:
        raise ValueError("QVQ grouped P32 requires at least one child")
    if len(children) != len(descriptors):
        raise ValueError("QVQ grouped P32 children and descriptors must have equal length")
    payloads = [_canonical_child_payload(child) for child in children]
    names = tuple(descriptor.module_name for descriptor in descriptors)
    basis_id = descriptors[0].input_basis_id
    if any(descriptor.input_basis_id != basis_id for descriptor in descriptors):
        raise ValueError("QVQ grouped P32 members must use one input_basis_id")
    if any(descriptor.input_hadamard != descriptors[0].input_hadamard for descriptor in descriptors):
        raise ValueError("QVQ grouped P32 members must agree on input_hadamard")
    if len(set(names)) != len(names):
        raise ValueError("QVQ grouped P32 member names must be unique")
    resolved = spec or QVQGroupedP32Spec(basis_id=basis_id, ordered_members=names)
    if resolved.basis_id != basis_id or resolved.ordered_members != names:
        raise ValueError("QVQ grouped P32 spec does not match descriptor order")

    reference = payloads[0]
    reference_child = children[0]
    for child, payload in zip(children[1:], payloads[1:], strict=True):
        if payload.in_features != reference.in_features:
            raise ValueError("QVQ grouped P32 members must have the same K")
        if payload.bits != reference.bits:
            raise ValueError("QVQ grouped P32 members must use the same rate")
        if payload.codebook_version != reference.codebook_version:
            raise ValueError("QVQ grouped P32 members must use the same codebook")
        if payload.vector_size != reference.vector_size or payload.trellis_window != reference.trellis_window:
            raise ValueError("QVQ grouped P32 members must use the same trellis geometry")
        if payload.trellis.device != reference.trellis.device:
            raise ValueError("QVQ grouped P32 members must share a device")
        if not _bitwise_equal(reference_child.SU, child.SU):
            raise ValueError("QVQ grouped P32 members require bit-identical SU vectors")
    return resolved


def canonical_child_p32_payload(child: object) -> QVQCanonicalP32Payload:
    """Return a detached canonical payload snapshot for one child."""

    return _canonical_child_payload(child)


def group_canonical_p32_payloads(
    children: Sequence[object],
    descriptors: Sequence[QVQExecutionDescriptor],
    spec: QVQGroupedP32Spec | None = None,
) -> QVQGroupedP32Payload:
    """Concatenate legal child payloads along the N16-tile dimension."""

    validate_group(children, descriptors, spec)
    payloads = [_canonical_child_payload(child) for child in children]
    reference = payloads[0]
    k_tiles = reference.in_features // 16
    trellis_parts = []
    bank_parts = []
    segments = []
    output_tile_start = 0
    for descriptor, payload in zip(descriptors, payloads, strict=True):
        n_tiles = payload.out_features // 16
        trellis_parts.append(payload.trellis.reshape(k_tiles, n_tiles, -1))
        bank_parts.append(payload.bank_ids.reshape(k_tiles, n_tiles))
        segments.append(
            QVQGroupedP32Segment(
                module_name=descriptor.module_name,
                output_tile_start=output_tile_start,
                output_tile_count=n_tiles,
                out_features=payload.out_features,
                bank_alt_id=payload.bank_alt_id.detach().clone().contiguous(),
            )
        )
        output_tile_start += n_tiles
    return QVQGroupedP32Payload(
        trellis=torch.cat(trellis_parts, dim=1).reshape(-1, reference.trellis.shape[-1]).contiguous(),
        bank_ids=torch.cat(bank_parts, dim=1).reshape(-1).contiguous(),
        segments=tuple(segments),
        bits=reference.bits,
        in_features=reference.in_features,
        codebook_version=reference.codebook_version,
    )


def ungroup_canonical_p32_payload(
    grouped: QVQGroupedP32Payload,
    member_index: int,
) -> QVQCanonicalP32Payload:
    """Recover one child payload from a grouped payload exactly."""

    if not isinstance(member_index, int) or isinstance(member_index, bool):
        raise TypeError("member_index must be an integer")
    if member_index < 0 or member_index >= len(grouped.segments):
        raise IndexError("grouped P32 member_index is out of range")
    segment = grouped.segments[member_index]
    k_tiles = grouped.in_features // 16
    n_tiles = segment.output_tile_count
    trellis = grouped.trellis.reshape(k_tiles, -1, grouped.trellis.shape[-1])
    bank_ids = grouped.bank_ids.reshape(k_tiles, -1)
    start = segment.output_tile_start
    return QVQCanonicalP32Payload(
        trellis=trellis[:, start : start + n_tiles].reshape(-1, grouped.trellis.shape[-1]).contiguous(),
        bank_ids=bank_ids[:, start : start + n_tiles].reshape(-1).contiguous(),
        bank_alt_id=segment.bank_alt_id.detach().clone().contiguous(),
        bits=grouped.bits,
        in_features=grouped.in_features,
        out_features=segment.out_features,
        codebook_version=grouped.codebook_version,
    )


def _inner_from_payload(payload: QVQCanonicalP32Payload, device: torch.device) -> torch.Tensor:
    return reconstruct_qvq_inner_weight(
        payload.trellis.to(device=device),
        bits=payload.bits,
        vector_size=payload.vector_size,
        trellis_window=payload.trellis_window,
        in_features=payload.in_features,
        out_features=payload.out_features,
        codebook_version=payload.codebook_version,
        bank_ids=payload.bank_ids.to(device=device),
        v2b2_p32=True,
        bank_alt_id=payload.bank_alt_id.to(device=device),
    ).to(dtype=torch.float32)


def _validate_oracle_input(child: object, x: torch.Tensor) -> None:
    _require_qvq_linear(child)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"QVQ oracle input must be a torch.Tensor, got {type(x).__name__}")
    if x.requires_grad:
        raise RuntimeError("QVQ grouped P32 oracle does not accept input that requires gradients")
    if x.shape[-1] != int(child.in_features):
        raise ValueError(f"QVQ oracle input width must be {child.in_features}, got {x.shape[-1]}")


def _apply_child_oracle_epilogue(
    transformed: torch.Tensor,
    child: object,
    descriptor: QVQExecutionDescriptor,
    *,
    device: torch.device,
) -> torch.Tensor:
    output = transformed
    if descriptor.output_hadamard:
        output = matmul_hadU(output)
    output = output * child.SV.to(device=device, dtype=torch.float32)
    if child.bias is not None:
        output = output + child.bias.to(device=device, dtype=torch.float32)
    return output


def qvq_torch_child_oracle(
    child: object,
    x: torch.Tensor,
    descriptor: QVQExecutionDescriptor | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Evaluate one P32 child using the explicit FP32 A41/R0 semantics."""

    _validate_oracle_input(child, x)
    descriptor = descriptor or QVQExecutionDescriptor(
        module_name=str(child.name), input_basis_id=str(child.name)
    )
    resolved_device = torch.device(x.device if device is None else device)
    payload = _canonical_child_payload(child)
    with torch.inference_mode():
        x_2d = x.to(device=resolved_device, dtype=torch.float32).reshape(-1, int(child.in_features))
        transformed = x_2d * child.SU.to(device=resolved_device, dtype=torch.float32)
        if descriptor.input_hadamard:
            transformed = matmul_hadU(transformed)
        output = _apply_child_oracle_epilogue(
            transformed @ _inner_from_payload(payload, resolved_device),
            child,
            descriptor,
            device=resolved_device,
        )
        return output.reshape(*x.shape[:-1], int(child.out_features)).detach()


def qvq_torch_group_oracle(
    children: Sequence[object],
    x: torch.Tensor,
    descriptors: Sequence[QVQExecutionDescriptor],
    spec: QVQGroupedP32Spec | None = None,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Evaluate a legal group with exactly one shared input transform.

    Inner matrices and output recovery remain separate per member.  This
    intentionally avoids a concatenated GEMM so numerical differences from a
    different GEMM tiling cannot hide an A41 semantic error.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"QVQ oracle input must be a torch.Tensor, got {type(x).__name__}")
    if x.requires_grad:
        raise RuntimeError("QVQ grouped P32 oracle does not accept input that requires gradients")
    resolved_spec = validate_group(children, descriptors, spec)
    del resolved_spec
    for child in children:
        _validate_oracle_input(child, x)
    grouped = group_canonical_p32_payloads(children, descriptors, spec)
    resolved_device = torch.device(x.device if device is None else device)
    with torch.inference_mode():
        x_2d = x.to(device=resolved_device, dtype=torch.float32).reshape(-1, grouped.in_features)
        shared = x_2d * children[0].SU.to(device=resolved_device, dtype=torch.float32)
        if descriptors[0].input_hadamard:
            shared = matmul_hadU(shared)
        outputs = []
        for index, (child, descriptor) in enumerate(zip(children, descriptors, strict=True)):
            payload = ungroup_canonical_p32_payload(grouped, index)
            inner = _inner_from_payload(payload, resolved_device)
            transformed = shared @ inner
            outputs.append(
                _apply_child_oracle_epilogue(
                    transformed, child, descriptor, device=resolved_device
                ).reshape(*x.shape[:-1], int(child.out_features)).detach()
            )
        return tuple(outputs)


class QVQGroupedRuntimeOracle:
    """Fail-closed sibling-call adapter for an existing module graph.

    Hugging Face attention blocks commonly invoke ``q_proj``, ``k_proj``, and
    ``v_proj`` as separate Python calls.  This adapter embeds the stateless
    grouped oracle into that call pattern without making the oracle itself
    stateful.  A cycle must consume every member in descriptor order and must
    use the identical input tensor object and mutation version for every call.
    """

    def __init__(
        self,
        children: Sequence[object],
        descriptors: Sequence[QVQExecutionDescriptor],
        spec: QVQGroupedP32Spec | None = None,
    ) -> None:
        self.children = tuple(children)
        self.descriptors = tuple(descriptors)
        self.spec = validate_group(self.children, self.descriptors, spec)
        self._member_indices = {
            descriptor.module_name: index
            for index, descriptor in enumerate(self.descriptors)
        }
        self._input: torch.Tensor | None = None
        self._input_version: int | None = None
        self._outputs: tuple[torch.Tensor, ...] | None = None
        self._next_index = 0

    @property
    def active(self) -> bool:
        """Whether a sibling cycle is waiting for more consumers."""

        return self._outputs is not None

    def _clear(self) -> None:
        self._input = None
        self._input_version = None
        self._outputs = None
        self._next_index = 0

    def consume(self, module_name: str, x: torch.Tensor) -> torch.Tensor:
        """Return one ordered sibling output or fail closed.

        The first call must name the first descriptor member.  Any duplicate,
        out-of-order, different-object, or mutated-input call clears the cycle
        and raises rather than returning a potentially stale output.
        """

        if module_name not in self._member_indices:
            raise KeyError(f"unknown grouped P32 member `{module_name}`")
        member_index = self._member_indices[module_name]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"QVQ grouped P32 input must be a torch.Tensor, got {type(x).__name__}")
        if self._outputs is None:
            if member_index != 0:
                raise RuntimeError("QVQ grouped P32 sibling consumer arrived out of order")
            outputs = qvq_torch_group_oracle(
                self.children, x, self.descriptors, self.spec
            )
            self._input = x
            self._input_version = x._version
            self._outputs = outputs
        else:
            if self._input is not x:
                self._clear()
                raise RuntimeError("QVQ grouped P32 sibling input tensor object changed")
            if self._input_version != x._version:
                self._clear()
                raise RuntimeError("QVQ grouped P32 sibling input tensor was mutated")
        if member_index != self._next_index:
            self._clear()
            raise RuntimeError("QVQ grouped P32 sibling consumer arrived out of order or duplicated")
        assert self._outputs is not None
        output = self._outputs[member_index]
        self._next_index += 1
        return output

    def finish_cycle(self) -> None:
        """End a cycle after every member has been consumed."""

        if self._outputs is None:
            raise RuntimeError("QVQ grouped P32 sibling cycle is not active")
        if self._next_index != len(self.children):
            self._clear()
            raise RuntimeError(
                "QVQ grouped P32 sibling cycle is incomplete: not every member was consumed"
            )
        self._clear()


__all__ = [
    "QVQCanonicalP32Payload",
    "QVQExecutionDescriptor",
    "QVQGroupedP32Payload",
    "QVQGroupedP32Segment",
    "QVQGroupedP32Spec",
    "QVQGroupedRuntimeOracle",
    "canonical_child_p32_payload",
    "group_canonical_p32_payloads",
    "qvq_torch_child_oracle",
    "qvq_torch_group_oracle",
    "ungroup_canonical_p32_payload",
    "validate_group",
]
