# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Mixture-of-experts API over ``torch.ops.nf12`` (SPEC.md section 9).

A typical layer is ``moe_align`` -> ``moe_linear`` (w13, gathered from tokens
into expert-sorted order) -> activation -> ``moe_linear`` (w2, scattered back to
assignment order and scaled by the router weights) -> ``moe_combine``.
:func:`fused_moe` uses a direct two-kernel route for supported single-token
shapes and composes these operations otherwise. Nothing here synchronizes
except encoding and the explicitly noted utilities, so the layer captures
into CUDA graphs and compiles under ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import torch

from ._library import load
from .api import ESCAPES_KEY, EXP_BASE_KEY, PACKED_KEY, _check_cuda, check_weight_shape


ESCAPE_OFFSETS_KEY = "nf12_escape_offsets"


@dataclasses.dataclass(eq=False)
class NF12MoEWeight:
    """``E`` NF12-encoded ``[N, K]`` expert matrices (SPEC.md section 9.1 tensor set).

    Expert ``e`` is a standalone NF12 matrix with its own ``exp_base[e]`` and
    escape rows ``escapes[escape_offsets[e]:escape_offsets[e + 1]]`` (local
    indices).
    """

    packed: torch.Tensor  # uint8 [E, N, K*3/2]
    escapes: torch.Tensor  # int16 [S, 8], expert-major
    escape_offsets: torch.Tensor  # int64 [E + 1]
    exp_base: torch.Tensor  # uint8 [E]
    shape: tuple[int, int, int] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        p, e, off, b = self.packed, self.escapes, self.escape_offsets, self.exp_base
        if p.dtype != torch.uint8 or p.dim() != 3 or p.shape[0] < 1 or p.shape[2] % 3:
            raise ValueError(f"nf12: packed must be uint8 [E, N, K*3/2] with E >= 1, got {p.dtype} {tuple(p.shape)}")
        if e.dtype != torch.int16 or e.dim() != 2 or e.shape[1] != 8:
            raise ValueError(f"nf12: escapes must be int16 [S, 8], got {e.dtype} {tuple(e.shape)}")
        num_experts = p.shape[0]
        if off.dtype != torch.int64 or tuple(off.shape) != (num_experts + 1,):
            raise ValueError(f"nf12: escape_offsets must be int64 [{num_experts + 1}], got {off.dtype} {tuple(off.shape)}")
        if b.dtype != torch.uint8 or tuple(b.shape) != (num_experts,):
            raise ValueError(f"nf12: exp_base must be uint8 [{num_experts}], got {b.dtype} {tuple(b.shape)}")
        for name, t in (("escapes", e), ("escape_offsets", off), ("exp_base", b)):
            if t.device != p.device:
                raise ValueError(f"nf12: packed ({p.device}) and {name} ({t.device}) must share a device")
        self.shape = (num_experts, p.shape[1], p.shape[2] * 2 // 3)
        check_weight_shape(*self.shape[1:])

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def num_experts(self) -> int:
        return self.shape[0]

    @property
    def num_escapes(self) -> int:
        """Total escape rows over all experts."""
        return self.escapes.shape[0]

    @property
    def nbytes(self) -> int:
        """Bytes of encoded storage (packed, escapes, escape offsets, exponent bases)."""
        return self.packed.nbytes + self.escapes.nbytes + self.escape_offsets.nbytes + self.exp_base.nbytes

    def decode(self) -> torch.Tensor:
        """Recovers the original ``[E, N, K]`` BF16 expert weights."""
        _check_cuda(self.packed, "weight")
        load()
        return torch.ops.nf12.moe_decode(self.packed, self.escapes, self.escape_offsets, self.exp_base)

    def to(self, device: torch.device | str | int) -> NF12MoEWeight:
        return NF12MoEWeight(self.packed.to(device), self.escapes.to(device), self.escape_offsets.to(device),
                             self.exp_base.to(device))

    def narrow(self, start: int, length: int) -> NF12MoEWeight:
        """Experts ``[start, start + length)`` as a standalone weight, e.g. one
        expert-parallel rank's local experts. Packed bytes and escapes are
        views, with escape offsets rebased to 0. Synchronizes to read two offsets."""
        if not (0 <= start and length >= 1 and start + length <= self.num_experts):
            raise ValueError(f"nf12: experts [{start}, {start + length}) out of range for {self.num_experts}")
        offsets = self.escape_offsets[start:start + length + 1]
        first, last = (int(v) for v in offsets[[0, -1]].tolist())
        return NF12MoEWeight(self.packed[start:start + length], self.escapes[first:last], offsets - first,
                             self.exp_base[start:start + length])

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Tensors under SPEC.md section 9.1 names. Use ``prefix="w13."`` for weight ``w13``."""
        return {
            prefix + PACKED_KEY: self.packed,
            prefix + ESCAPES_KEY: self.escapes,
            prefix + ESCAPE_OFFSETS_KEY: self.escape_offsets,
            prefix + EXP_BASE_KEY: self.exp_base,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, torch.Tensor], prefix: str = "") -> NF12MoEWeight:
        return cls(state_dict[prefix + PACKED_KEY], state_dict[prefix + ESCAPES_KEY],
                   state_dict[prefix + ESCAPE_OFFSETS_KEY], state_dict[prefix + EXP_BASE_KEY])


def _check_experts(w: torch.Tensor) -> None:
    if w.dtype != torch.bfloat16 or w.dim() != 3 or w.shape[0] < 1:
        raise ValueError(f"nf12: experts must be a bfloat16 [E, N, K] tensor, got {w.dtype} {tuple(w.shape)}")
    _check_cuda(w, "weight")
    check_weight_shape(*w.shape[1:])


def encode_moe(w: torch.Tensor) -> NF12MoEWeight:
    """Losslessly encodes stacked ``[E, N, K]`` bf16 CUDA experts, each exactly as
    :func:`nf12.encode` would encode it alone. Synchronizes the current stream."""
    _check_experts(w)
    load()
    return NF12MoEWeight(*torch.ops.nf12.encode_moe(w))


def pack_bf16t_moe(w: torch.Tensor) -> torch.Tensor:
    """Packs ``[E, N, K]`` bf16 experts into the bf16t twin layout, uint8 ``[E, N, 2K]``."""
    _check_experts(w)
    load()
    e, n, k = w.shape
    # N % 16 == 0, so expert e's super-atoms are rows [e*N, (e+1)*N) of the stacked matrix.
    return torch.ops.nf12.pack_bf16t(w.reshape(e * n, k)).view(e, n, 2 * k)


def _prepare_topk_ids(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    _check_cuda(topk_ids, "topk_ids")
    if topk_ids.dim() != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"nf12: topk_ids must be an int32/int64 [T, topk] tensor, got {topk_ids.dtype} "
                         f"{tuple(topk_ids.shape)}")
    if topk_ids.dtype == torch.int64:
        # Drop before narrowing: 2**32 must not wrap into a valid expert.
        topk_ids = torch.where((topk_ids >= 0) & (topk_ids < num_experts), topk_ids, -1)
    return topk_ids.to(torch.int32)


def moe_align(topk_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorts assignment ids ``t * topk + j`` stably by expert ``topk_ids[t, j]``.

    Returns ``(sorted int32 [T*topk], expert_offsets int64 [E + 1])``: expert
    ``e`` owns sorted positions ``[off[e], off[e+1])``. Ids outside ``[0, E)``
    are dropped, ``off[E]`` counts the rest and later entries of ``sorted`` are
    unspecified.
    """
    topk_ids = _prepare_topk_ids(topk_ids, num_experts)
    load()
    return torch.ops.nf12.moe_align(topk_ids, num_experts)


def _check_moe_input(x: torch.Tensor, k: int) -> None:
    _check_cuda(x, "x")
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != k:
        raise ValueError(f"nf12: x must be a bfloat16 [rows, {k}] tensor, got {x.dtype} {tuple(x.shape)}")


def moe_linear(x: torch.Tensor, weight: NF12MoEWeight | torch.Tensor, expert_offsets: torch.Tensor, *,
               gather: torch.Tensor | None = None, gather_div: int = 1, scatter: torch.Tensor | None = None,
               row_scale: torch.Tensor | None = None, zero_fill: bool = False) -> torch.Tensor:
    """``y[out(p)] = x[in(p)] @ W_e.T (* row_scale[out(p)])`` for every valid sorted position ``p`` of expert ``e``.

    ``in(p) = gather[p] // gather_div`` (else ``p``), ``out(p) = scatter[p]``
    (else ``p``). Returns bf16 ``[A, N]`` with ``A`` the length of ``gather``,
    else ``scatter``, else the rows of ``x``. Rows no valid position writes are
    uninitialized unless ``zero_fill``. ``weight`` is an :class:`NF12MoEWeight`
    or a bf16t twin from :func:`pack_bf16t_moe`. Both produce the same results
    except when NF12 without ``scatter``/``row_scale``/``zero_fill`` takes the
    large-T decode + ``torch._grouped_mm`` route of SPEC.md 9.4. That route
    matches those grouped GEMMs on the decoded experts.
    """
    if isinstance(weight, NF12MoEWeight):
        _check_moe_input(x, weight.shape[2])
        load()
        return torch.ops.nf12.moe_linear(x, weight.packed, weight.escapes, weight.escape_offsets, weight.exp_base,
                                         expert_offsets, gather, gather_div, scatter, row_scale, zero_fill)
    return moe_linear_bf16t(x, weight, expert_offsets, gather=gather, gather_div=gather_div, scatter=scatter,
                            row_scale=row_scale, zero_fill=zero_fill)


def moe_linear_bf16t(x: torch.Tensor, w_bf16t: torch.Tensor, expert_offsets: torch.Tensor, *,
                     gather: torch.Tensor | None = None, gather_div: int = 1, scatter: torch.Tensor | None = None,
                     row_scale: torch.Tensor | None = None, zero_fill: bool = False) -> torch.Tensor:
    """:func:`moe_linear` on a bf16t twin ``uint8 [E, N, 2K]``."""
    if w_bf16t.dtype != torch.uint8 or w_bf16t.dim() != 3:
        raise ValueError(f"nf12: w_bf16t must be uint8 [E, N, 2K], got {w_bf16t.dtype} {tuple(w_bf16t.shape)}")
    _check_moe_input(x, w_bf16t.shape[2] // 2)
    load()
    return torch.ops.nf12.moe_linear_bf16t(x, w_bf16t, expert_offsets, gather, gather_div, scatter, row_scale,
                                           zero_fill)


def moe_combine(y: torch.Tensor, topk: int) -> torch.Tensor:
    """``out[t] = sum_j y[t * topk + j]`` in FP32 in increasing ``j``, returning bf16 ``[T, N]``."""
    _check_cuda(y, "y")
    load()
    return torch.ops.nf12.moe_combine(y, topk)


def moe_silu_mul(h: torch.Tensor) -> torch.Tensor:
    """SiLU gating of ``h = [gate | up]`` (bf16 ``[A, 2I]``) -> bf16 ``[A, I]``.

    ``out = bf16(s * u)`` with ``s = g / (1 + exp(-g))`` in FP32 (one kernel,
    one BF16 rounding).
    """
    _check_cuda(h, "h")
    load()
    return torch.ops.nf12.moe_silu_mul(h)



def _out_features(w: NF12MoEWeight | torch.Tensor) -> tuple[int, int, int]:
    """(E, N, K) of an NF12 weight or bf16t twin."""
    if isinstance(w, NF12MoEWeight):
        return w.shape
    if w.dtype != torch.uint8 or w.dim() != 3:
        raise ValueError(f"nf12: expected an NF12MoEWeight or a bf16t uint8 [E, N, 2K], got {w.dtype} "
                         f"{tuple(w.shape)}")
    return w.shape[0], w.shape[1], w.shape[2] // 2


def _map_experts(topk_ids: torch.Tensor, expert_map: torch.Tensor) -> torch.Tensor:
    """``expert_map[topk_ids]``, with ids outside ``[0, global_E)`` mapped to -1
    (dropped) instead of wrapping (negative) or faulting (too large)."""
    if expert_map.dim() != 1 or expert_map.numel() == 0:
        raise ValueError(f"nf12: expert_map must be a non-empty 1-D tensor, got {tuple(expert_map.shape)}")
    valid = (topk_ids >= 0) & (topk_ids < expert_map.shape[0])
    return torch.where(valid, expert_map[torch.where(valid, topk_ids, 0)], -1)


def _moe_weight_args(w: NF12MoEWeight | torch.Tensor) -> tuple:
    if isinstance(w, NF12MoEWeight):
        return w.packed, w.escapes, w.escape_offsets, w.exp_base
    return w, None, None, None


def fused_moe(x: torch.Tensor, w13: NF12MoEWeight | torch.Tensor, w2: NF12MoEWeight | torch.Tensor,
              topk_weights: torch.Tensor, topk_ids: torch.Tensor, activation: str = "silu", *,
              expert_map: torch.Tensor | None = None) -> torch.Tensor:
    """A gated MoE MLP: ``sum_j topk_weights[t, j] * W2_e (act(gate) * up)`` with ``[gate; up] = W13_e x[t]``.

    ``x``: bf16 ``[T, K]``; ``w13``: ``E`` experts of ``[2I, K]`` (gate rows
    first); ``w2``: ``E`` experts of ``[K2, I]``; ``topk_weights``/``topk_ids``:
    ``[T, topk]``. Weights are :class:`NF12MoEWeight` or bf16t twins.

    Expert ids must be in ``[0, E)``. For expert parallelism pass
    ``expert_map`` (int ``[global_E]``: global id -> local id, or -1 for
    experts on other ranks). Dropped assignments contribute zero, so the
    result is this rank's partial sum.
    """
    if activation != "silu":
        raise ValueError(f"nf12: unsupported activation {activation!r} (only 'silu')")
    e13, n13, k13 = _out_features(w13)
    e2, n2, k2 = _out_features(w2)
    if e13 != e2 or n13 != 2 * k2:
        raise ValueError(f"nf12: w13 {(e13, n13, k13)} and w2 {(e2, n2, k2)} need the same E and N13 == 2 * K2")
    if x.dim() != 2 or topk_ids.dim() != 2 or topk_weights.shape != topk_ids.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError(f"nf12: need x [T, K] and topk_weights/topk_ids [T, topk], got {tuple(x.shape)}, "
                         f"{tuple(topk_weights.shape)}, {tuple(topk_ids.shape)}")
    if expert_map is not None:
        topk_ids = _map_experts(topk_ids, expert_map)
    topk_ids = _prepare_topk_ids(topk_ids, e13)
    _check_moe_input(x, k13)
    load()
    return torch.ops.nf12._fused_moe(x, *_moe_weight_args(w13), *_moe_weight_args(w2),
                                    topk_weights.float(), topk_ids, expert_map is not None)
