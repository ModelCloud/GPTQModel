# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tensor layout invariants for model init and post-init.

GPT-QModel kernels and the P2P peer-copy path both assume that every parameter
and buffer is contiguous. The peer-copy path enforces that assumption by
refusing non-contiguous leaves (a raw byte copy of a strided view would silently
produce wrong values), and every quantized kernel is written against contiguous
packed weights.

That invariant is currently maintained by construction: load paths call
``.contiguous()`` after transposes, and packing produces fresh contiguous
buffers. This module makes it checkable, so a new kernel or load path that
breaks it fails loudly instead of silently taking a slow or incorrect path.

Note that contiguity is *not* the same as adjacency. Two contiguous tensors
still live in separate allocations, so they cannot be covered by one memcpy
without a packing step. Do not use this check to reason about whether a group
of tensors can be copied in a single transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class NonContiguousTensor:
    """One parameter or buffer whose memory layout is not contiguous."""

    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.kind}, shape={self.shape}, "
            f"dtype={self.dtype}, stride={self.stride})"
        )


def iter_model_tensors(
    module: torch.nn.Module,
) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Yield ``(name, kind, tensor)`` for every parameter and buffer.

    Tensors shared by identity (tied weights) are yielded once, under their
    first observed name. Meta tensors are skipped: they have no storage, so
    contiguity is meaningless for them.
    """

    seen: set[int] = set()
    for module_name, submodule in module.named_modules():
        for kind, table in (("parameter", submodule._parameters), ("buffer", submodule._buffers)):
            for leaf_name, tensor in table.items():
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.is_meta:
                    continue
                if id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                name = f"{module_name}.{leaf_name}" if module_name else leaf_name
                yield name, kind, tensor


def find_non_contiguous_tensors(
    module: torch.nn.Module,
) -> list[NonContiguousTensor]:
    """Return every non-contiguous parameter or buffer in ``module``."""

    return [
        NonContiguousTensor(
            name=name,
            kind=kind,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(tensor.stride()),
        )
        for name, kind, tensor in iter_model_tensors(module)
        if not tensor.is_contiguous()
    ]


def assert_model_tensors_contiguous(
    module: torch.nn.Module,
    *,
    label: str = "model",
    max_report: int = 10,
) -> None:
    """Raise ``ValueError`` when any parameter or buffer is non-contiguous.

    ``label`` only names the checked state in the error message, for example
    ``"post-quantize"`` or ``"after reload"``.
    """

    offenders = find_non_contiguous_tensors(module)
    if not offenders:
        return

    shown = ", ".join(str(item) for item in offenders[:max_report])
    remainder = len(offenders) - min(len(offenders), max_report)
    suffix = f" (and {remainder} more)" if remainder else ""
    raise ValueError(
        f"{label}: {len(offenders)} non-contiguous parameter/buffer(s); "
        f"GPT-QModel kernels and the P2P peer-copy path require contiguous tensors: "
        f"{shown}{suffix}"
    )
