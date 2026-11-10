# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Functional API over ``torch.ops.nf12`` (SPEC.md section 8.2)."""

from __future__ import annotations

import dataclasses
from typing import Mapping

import torch

from ._library import load


PACKED_KEY = "nf12_packed"
ESCAPES_KEY = "nf12_escapes"
EXP_BASE_KEY = "nf12_exp_base"


def check_weight_shape(n: int, k: int) -> None:
    if n <= 0 or n % 16:
        raise ValueError(f"nf12: out_features (N) must be a positive multiple of 16, got {n}")
    if k <= 0 or k % 64:
        raise ValueError(f"nf12: in_features (K) must be a positive multiple of 64, got {k}")


def _check_cuda(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"nf12: {name} must be a CUDA tensor, got device {t.device}")


def _check_bf16_matrix(w: torch.Tensor, name: str) -> None:
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise ValueError(f"nf12: {name} must be a 2-D bfloat16 tensor, got {w.dtype} with shape {tuple(w.shape)}")
    _check_cuda(w, name)
    check_weight_shape(*w.shape)


@dataclasses.dataclass(eq=False)
class NF12Weight:
    """An NF12-encoded ``[N, K]`` BF16 matrix (SPEC.md section 5 tensor set)."""

    packed: torch.Tensor  # uint8 [N, K*3/2], tiled layout
    escapes: torch.Tensor  # int16 [E, 8], BF16 bit patterns of escaped groups
    exp_base: int  # 0..120
    shape: tuple[int, int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        p, e = self.packed, self.escapes
        if p.dtype != torch.uint8 or p.dim() != 2 or p.shape[1] % 3:
            raise ValueError(f"nf12: packed must be uint8 [N, K*3/2], got {p.dtype} {tuple(p.shape)}")
        if e.dtype != torch.int16 or e.dim() != 2 or e.shape[1] != 8:
            raise ValueError(f"nf12: escapes must be int16 [E, 8], got {e.dtype} {tuple(e.shape)}")
        if e.device != p.device:
            raise ValueError(f"nf12: packed ({p.device}) and escapes ({e.device}) must share a device")
        self.exp_base = int(self.exp_base)
        if not 0 <= self.exp_base <= 120:
            raise ValueError(f"nf12: exp_base must be in [0, 120], got {self.exp_base}")
        self.shape = (p.shape[0], p.shape[1] * 2 // 3)
        check_weight_shape(*self.shape)

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def num_escapes(self) -> int:
        return self.escapes.shape[0]

    @property
    def nbytes(self) -> int:
        """Bytes of encoded storage (packed + escapes)."""
        return self.packed.nbytes + self.escapes.nbytes

    def decode(self) -> torch.Tensor:
        return decode(self)

    def to(self, device: torch.device | str | int) -> NF12Weight:
        return NF12Weight(self.packed.to(device), self.escapes.to(device), self.exp_base)

    def narrow_rows(self, start: int, length: int) -> NF12Weight:
        """View an aligned output-row slice without decoding or repacking.

        Escape indices in the packed stream refer to the original table, so
        the view deliberately retains the whole escape table.
        """
        if start < 0 or length <= 0 or start % 16 or length % 16 or start + length > self.shape[0]:
            raise ValueError("NF12 row slices need positive 16-row-aligned bounds inside the weight")
        return NF12Weight(self.packed.narrow(0, start, length), self.escapes, self.exp_base)

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Tensors under SPEC.md section 5 names. Use ``prefix="W."`` for weight ``W``."""
        return {
            prefix + PACKED_KEY: self.packed,
            prefix + ESCAPES_KEY: self.escapes,
            prefix + EXP_BASE_KEY: torch.tensor([self.exp_base], dtype=torch.uint8),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor], prefix: str = "") -> NF12Weight:
        exp_base = state_dict[prefix + EXP_BASE_KEY]
        if exp_base.dtype != torch.uint8 or exp_base.numel() != 1:
            raise ValueError(f"nf12: {prefix + EXP_BASE_KEY} must be a uint8 [1] tensor")
        return cls(state_dict[prefix + PACKED_KEY], state_dict[prefix + ESCAPES_KEY], int(exp_base.item()))


def encode(w: torch.Tensor) -> NF12Weight:
    """Losslessly encodes a ``[N, K]`` bf16 CUDA matrix. Synchronizes the current stream."""
    _check_bf16_matrix(w, "weight")
    load()
    packed, escapes, exp_base = torch.ops.nf12.encode(w)
    return NF12Weight(packed, escapes, exp_base)


def decode(weight: NF12Weight) -> torch.Tensor:
    """Recovers the original ``[N, K]`` BF16 matrix."""
    _check_cuda(weight.packed, "weight")
    load()
    return torch.ops.nf12.decode(weight.packed, weight.escapes, weight.exp_base)


def _check_input(x: torch.Tensor, k: int) -> None:
    _check_cuda(x, "x")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"nf12: x must be bfloat16, got {x.dtype}")
    if x.dim() == 0 or x.shape[-1] != k:
        raise ValueError(f"nf12: x must have shape [..., {k}], got {tuple(x.shape)}")


def linear(x: torch.Tensor, weight: NF12Weight, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``y = x @ W.T (+ bias)`` maps bf16 ``x`` of shape ``[..., K]`` to ``[..., N]``.

    At the per-architecture M threshold (SPEC.md 6.5), row slices are decoded
    and multiplied by cuBLAS, matching the same calls on :func:`decode`.
    Below that threshold the C API ``nf12_linear`` runs."""
    _check_input(x, weight.shape[1])
    load()
    return torch.ops.nf12.linear(x, weight.packed, weight.escapes, weight.exp_base, bias)


def linear_fp32(x: torch.Tensor, weight: NF12Weight, bias: torch.Tensor | None = None,
                rows: torch.Tensor | None = None) -> torch.Tensor:
    """Run an FP32 projection, optionally on selected output rows, from NF12."""
    from ._fp32 import linear_fp32 as impl

    return impl(x, weight, bias, rows)


def linear_bf16_fp32(x: torch.Tensor, weight: NF12Weight,
                     bias: torch.Tensor | None = None) -> torch.Tensor:
    """BF16 activation and packed weight, with FP32 accumulation and output."""
    from ._fp32 import linear_bf16_fp32 as impl

    return impl(x, weight, bias)


def embedding(ids: torch.Tensor, weight: NF12Weight) -> torch.Tensor:
    """Embedding lookup ``W[ids]`` (SPEC.md section 10): int32 or int64 ``ids`` of any shape -> bf16
    ``[*ids.shape, K]``, preserving the table values. IDs outside ``[0, N)`` give ``+0.0`` rows."""
    _check_cuda(ids, "ids")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"nf12: ids must be int32 or int64, got {ids.dtype}")
    load()
    return torch.ops.nf12.embedding(ids, weight.packed, weight.escapes, weight.exp_base)


def pack_bf16t(w: torch.Tensor) -> torch.Tensor:
    """Packs ``[N, K]`` bf16 into the uncompressed bf16t twin layout, uint8 ``[N, 2K]`` (SPEC.md 4.4)."""
    _check_bf16_matrix(w, "weight")
    load()
    return torch.ops.nf12.pack_bf16t(w)


def linear_bf16t(x: torch.Tensor, w_bf16t: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Linear on a bf16t-packed weight, matching C API ``nf12_linear`` on NF12.

    Also matches :func:`linear` below the decode + cuBLAS crossover (SPEC.md 6.5)."""
    _check_input(x, w_bf16t.shape[1] // 2)
    load()
    return torch.ops.nf12.linear_bf16t(x, w_bf16t, bias)
