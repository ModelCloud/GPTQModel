# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-FileCopyrightText: 2026 the MACAQ authors
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# MACaQ (Model Agnostic Calibrated Quantization) weight quantizer.
#
# This module is a clean-room Torch port of the MACaQ MXFP4 quantization recipe
# published by Huawei CSL:
#
#   https://github.com/huawei-csl/MACaQ        (Apache License 2.0)
#   "MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision
#    Weights and Activations", Muller et al., arXiv:2611.10649
#
# The reference implementation files this port was derived from are
# `quantize/macaq_mxfp4.py`, `quantize/macaq_balance.py` and
# `quantize/mxfp4_helper.py` in the upstream repository.  The numerical
# recipe is preserved:
#
#   1. activation-aware diagonal pre-scale D1
#        p_j = w_std_j^alpha / act_rms_j^(1 - alpha)
#   2. block-Hadamard rotation (32-wide blocks, the MXFP4 group)
#   3. OCP MXFP4 round-to-nearest (E2M1 values, one E8M0 scale per 32 values)
#   4. Hessian-weighted per-(row, block) E8M0 exponent screen
#   5. greedy single-level "flip" refinement in the fixed Hessian metric
#
# It intentionally does NOT subclass `gptqmodel.quantization.gptq.GPTQ`: MACaQ
# is not GPTQ.  It shares Hessian capture and the per-Linear calibration
# lifecycle, but replaces GPTQ's sequential inverse-Hessian error propagation
# and affine INT4 grid with a rotation + MXFP4 block scale + local-search
# refinement, and it stores nibble-packed E2M1 codes instead of int32-packed
# affine weights.  Subclassing GPTQ would carry Cholesky/group-search/desc_act
# machinery that MACaQ never executes.  See `docs/macaq.md` for the comparison.

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from ..looper.named_module import NamedModule
from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM


log = setup_logger()

MXFP4_BLOCK = 32
_HESSIAN_DTYPE = torch.float32

# Signed E2M1 grid: 15 distinct levels (0 is shared by both signs), matching
# the OCP MXFP4 element set.  Index 7 is the zero level.
E2M1_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
# Index into the unsigned magnitude code (0..7) for each signed-grid index.
E2M1_MAGNITUDE_INDEX = torch.tensor(
    [7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8
)
# Sign bit for each signed-grid index.
E2M1_SIGN_BIT = torch.tensor(
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8
)
# E2M1 magnitude for the 8 unsigned codes.
E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _device_const(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor if tensor.device == device else tensor.to(device)


# ---------------------------------------------------------------------------
# Hadamard rotation
# ---------------------------------------------------------------------------
def make_hadamard(n: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the normalized ``n x n`` Sylvester Hadamard matrix.

    ``H[i, j] = (-1) ** popcount(i & j) / sqrt(n)`` is orthogonal
    (``H @ H.T == I``).  MACaQ applies it block-diagonally over 32-wide
    MXFP4 groups so the rotation never mixes two scale groups.
    """

    if n <= 0 or n & (n - 1):
        raise ValueError(f"MACaQ: Hadamard size must be a positive power of two, got {n}.")
    rows = [[(-1.0) ** (i & j).bit_count() for j in range(n)] for i in range(n)]
    return torch.tensor(rows, dtype=dtype, device=device) / math.sqrt(n)


def rotate_blocks(x: torch.Tensor, bh: torch.Tensor) -> torch.Tensor:
    """Right-multiply ``x`` by the block-diagonal Hadamard matrix on the last axis."""

    block = bh.shape[0]
    if x.shape[-1] % block:
        raise ValueError(
            f"MACaQ: last dimension {x.shape[-1]} is not divisible by the Hadamard block {block}."
        )
    lead = x.shape[:-1]
    return (x.reshape(*lead, x.shape[-1] // block, block) @ bh.to(x.dtype)).reshape(*lead, x.shape[-1])


# ---------------------------------------------------------------------------
# D1 activation-aware diagonal pre-scale
# ---------------------------------------------------------------------------
def balance_pre_scale_from_diag(
    weight: torch.Tensor,
    sum_x2: torch.Tensor,
    n_tokens: float,
    alpha: float = 0.5,
) -> torch.Tensor:
    """MACaQ D1 pre-scale from the activation diagonal ``diag(H) = sum x^2``.

    ``p_j = w_std_j^alpha / act_rms_j^(1 - alpha)`` with
    ``act_rms_j = sqrt(sum_x2_j / T)``.  Folding ``p`` into the input and
    ``1/p`` into the weight cancels in the GEMM.
    """

    w_std = weight.float().std(dim=0).clamp_min(1e-8)
    act_rms = (sum_x2.float() / max(float(n_tokens), 1.0)).clamp_min(1e-12).sqrt()
    return (w_std.pow(alpha) / act_rms.pow(1.0 - alpha)).clamp(1e-4, 1e4)


def balance_pre_scale(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    n_tokens: float,
    alpha: float = 0.5,
) -> torch.Tensor:
    return balance_pre_scale_from_diag(weight, hessian.diagonal(), n_tokens, alpha)


# ---------------------------------------------------------------------------
# OCP MXFP4 round-to-nearest (E2M1 + E8M0)
# ---------------------------------------------------------------------------
def _e2m1_code_from_scaled(abs_scaled: torch.Tensor, *, tie_high: bool = False) -> torch.Tensor:
    """Nearest E2M1 magnitude code (0..7) for values already divided by the scale.

    ``tie_high=False`` uses strict ``>`` comparisons, matching the MACaQ weight
    quantizer.  The MACaQ activation kernels compare with ``>=``; pass
    ``tie_high=True`` to reproduce that tie behavior bit-for-bit.
    """

    def above(value: float) -> torch.Tensor:
        return abs_scaled >= value if tie_high else abs_scaled > value

    return torch.where(
        above(5.0), 7,
        torch.where(
            above(3.5), 6,
            torch.where(
                above(2.5), 5,
                torch.where(
                    above(1.75), 4,
                    torch.where(
                        above(1.25), 3,
                        torch.where(above(0.75), 2, torch.where(above(0.25), 1, 0)),
                    ),
                ),
            ),
        ),
    ).long()


@torch.no_grad()
def mxfp4_rtn_signed_index(
    weight: torch.Tensor,
    block_size: int = MXFP4_BLOCK,
    exp_offset: int = 0,
    *,
    tie_high: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RTN a ``[out, in]`` matrix to OCP MXFP4 signed-grid indices.

    Returns ``(signed_index, log2_scale, scale_element)`` where
    ``signed_index`` indexes :data:`E2M1_VALUES`, ``log2_scale`` is
    ``[out, in // block]`` and ``scale_element`` is the broadcast
    ``[out, in]`` power-of-two scale.  The E8M0 exponent is
    ``floor(log2(amax)) - 2 + exp_offset`` (``-2`` is the exponent of the
    largest E2M1 magnitude, ``6 = 1.5 * 2^2``).
    """

    out_features, in_features = weight.shape
    if in_features % block_size:
        raise ValueError(
            f"MACaQ: in_features {in_features} must be divisible by the MXFP4 block {block_size}."
        )
    groups = in_features // block_size
    grouped = weight.float().view(out_features, groups, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    log2_scale = torch.floor(torch.log2(amax)) - 2.0 + float(exp_offset)
    scale = torch.exp2(log2_scale)

    abs_scaled = (grouped / scale).abs()
    code = _e2m1_code_from_scaled(abs_scaled, tie_high=tie_high)
    negative = grouped < 0
    signed_index = torch.where(
        code == 0,
        torch.full_like(code, 7),
        torch.where(negative, 7 - code, 7 + code),
    ).view(out_features, in_features)
    scale_element = scale.expand(out_features, groups, block_size).reshape(out_features, in_features).contiguous()
    return signed_index, log2_scale.squeeze(-1), scale_element


@torch.no_grad()
def screen_block_exponents(
    weight: torch.Tensor,
    hessian_diagonal: torch.Tensor,
    block_size: int = MXFP4_BLOCK,
    offsets: Tuple[int, ...] = (1, 0, -1),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the H-weighted best E8M0 exponent per ``(row, block)``.

    Each candidate offset is a different power-of-two base for the block.  The
    objective is the diagonal-Hessian-weighted squared error
    ``sum_{j in block} H_jj (w_j - deq_j)^2``; because the metric is
    diagonal, the screen is separable per block.  Off-diagonal coupling is
    handled afterwards by :func:`flip_refine_levels`, with the chosen
    exponents frozen.
    """

    out_features, in_features = weight.shape
    groups = in_features // block_size
    device = weight.device
    grid = _device_const(E2M1_VALUES, device)
    hdiag = hessian_diagonal.float().view(1, groups, block_size)

    best_error = best_index = best_log2 = best_scale = None
    for offset in offsets:
        signed_index, log2_scale, scale_element = mxfp4_rtn_signed_index(weight, block_size, offset)
        dequant = grid[signed_index] * scale_element
        error = ((weight.float() - dequant).pow(2).view(out_features, groups, block_size) * hdiag).sum(-1)
        if best_error is None:
            best_error, best_index, best_log2, best_scale = error, signed_index, log2_scale, scale_element
            continue
        better = error < best_error
        expand = better.unsqueeze(-1)
        best_error = torch.where(better, error, best_error)
        best_log2 = torch.where(better, log2_scale, best_log2)
        best_index = torch.where(
            expand,
            signed_index.view(out_features, groups, block_size),
            best_index.view(out_features, groups, block_size),
        ).view(out_features, in_features)
        best_scale = torch.where(
            expand,
            scale_element.view(out_features, groups, block_size),
            best_scale.view(out_features, groups, block_size),
        ).reshape(out_features, in_features)
    return best_index.contiguous(), best_log2, best_scale.contiguous()


@torch.no_grad()
def pack_mxfp4_from_signed_index(
    signed_index: torch.Tensor,
    log2_scale: torch.Tensor,
    block_size: int = MXFP4_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack signed-grid indices and frozen exponents into MXFP4 storage.

    Returns ``(packed, scales)`` where ``packed`` is ``uint8 [out, in // 2]``
    (low nibble = even column, high nibble = odd column) and ``scales`` is an
    ``E8M0`` view of ``uint8 [out, in // block]``.  The scale exponent is
    ``log2_scale + 127`` and the value convention is
    ``value = code * 2 ** (e8m0 - 127)``.
    """

    out_features, in_features = signed_index.shape
    groups = in_features // block_size
    device = signed_index.device
    magnitude = _device_const(E2M1_MAGNITUDE_INDEX, device)[signed_index]
    sign = _device_const(E2M1_SIGN_BIT, device)[signed_index]
    nibbles = (sign.to(torch.uint8) << 3) | magnitude.to(torch.uint8)
    packed = (nibbles[:, 1::2] << 4) | nibbles[:, 0::2]
    biased = (log2_scale + 127).clamp(0, 254).to(torch.int32)
    scales = biased.to(torch.uint8).contiguous().view(torch.float8_e8m0fnu)
    if scales.shape != (out_features, groups):
        raise AssertionError("MACaQ: packed scale shape mismatch")
    return packed.contiguous(), scales


@torch.no_grad()
def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    in_features: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode packed MXFP4 storage back to a dense ``[out, in_features]`` tensor.

    ``scales`` may be an ``E8M0`` tensor or its ``uint8`` view, and may carry
    kernel padding; only the first ``in_features // 32`` columns are read.
    """

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("MACaQ: `packed` must be a 2-D uint8 tensor.")
    out_features = packed.shape[0]
    groups = in_features // MXFP4_BLOCK
    if packed.shape[1] * 2 < in_features:
        raise ValueError("MACaQ: `packed` holds fewer than `in_features` values.")
    scale_bytes = scales.view(torch.uint8)[:out_features, :groups].to(torch.float32)
    scale = torch.exp2(scale_bytes - 127.0)
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    code = torch.stack([lo, hi], dim=-1).reshape(out_features, packed.shape[1] * 2)[:, :in_features]
    magnitude = _device_const(E2M1_MAGNITUDES, packed.device)[code & 7]
    sign = torch.where((code >> 3) & 1 == 1, -1.0, 1.0)
    values = sign * magnitude
    values = values.view(out_features, groups, MXFP4_BLOCK)
    values = values * scale.unsqueeze(-1)
    return values.reshape(out_features, in_features).to(dtype)


# ---------------------------------------------------------------------------
# Greedy flip refinement
# ---------------------------------------------------------------------------
def _flip_sweep(
    signed_index: torch.Tensor,
    scale_element: torch.Tensor,
    hessian_weight_target: torch.Tensor,
    hessian_weight_current: torch.Tensor,
    rows: torch.Tensor,
    grid: torch.Tensor,
    inf: torch.Tensor,
    hessian_diagonal: torch.Tensor,
    hessian: torch.Tensor,
    *,
    active: bool,
) -> torch.Tensor:
    """One greedy sweep; mutates ``signed_index`` / ``hessian_weight_current``."""

    if active:
        selected_index = signed_index[rows]
        selected_scale = scale_element[rows]
        gradient = (hessian_weight_target[:, rows] - hessian_weight_current[:, rows]).T
    else:
        selected_index = signed_index
        selected_scale = scale_element
        gradient = (hessian_weight_target - hessian_weight_current).T

    up_ok = selected_index < 14
    c_up = (grid[torch.where(up_ok, selected_index + 1, selected_index)] - grid[selected_index]) * selected_scale
    delta_up = torch.where(up_ok, c_up * c_up * hessian_diagonal - 2 * c_up * gradient, inf)
    down_ok = selected_index > 0
    c_down = (grid[torch.where(down_ok, selected_index - 1, selected_index)] - grid[selected_index]) * selected_scale
    delta_down = torch.where(down_ok, c_down * c_down * hessian_diagonal - 2 * c_down * gradient, inf)

    up_better = delta_up <= delta_down
    delta = torch.where(up_better, delta_up, delta_down)
    best_delta, best_column = delta.min(dim=1)
    improving = best_delta < -1e-12
    if not improving.any():
        return torch.empty(0, dtype=torch.long, device=signed_index.device)

    local_rows = torch.arange(rows.shape[0], device=signed_index.device)
    up_selected = up_better[local_rows, best_column]
    up_mask = up_selected & improving
    down_mask = (~up_selected) & improving
    delta_selected = torch.where(up_selected, c_up[local_rows, best_column], c_down[local_rows, best_column])

    if active:
        signed_index[rows[up_mask], best_column[up_mask]] += 1
        signed_index[rows[down_mask], best_column[down_mask]] -= 1
        improved_rows = rows[improving]
        hessian_weight_current[:, improved_rows] = (
            hessian_weight_current[:, improved_rows]
            + hessian.index_select(1, best_column[improving]) * delta_selected[improving].unsqueeze(0)
        )
        return improved_rows

    signed_index[rows[up_mask], best_column[up_mask]] += 1
    signed_index[rows[down_mask], best_column[down_mask]] -= 1
    delta_selected = torch.where(improving, delta_selected, torch.zeros_like(delta_selected))
    hessian_weight_current.add_(hessian.index_select(1, best_column) * delta_selected.unsqueeze(0))
    return improving


_COMPILED_FLIP_LOCK = threading.RLock()
_COMPILED_FLIP_SWEEP = None


def _compiled_flip_sweep(*args, **kwargs):
    """Serialize Inductor's first-use autotuning across quantization workers."""

    global _COMPILED_FLIP_SWEEP
    with _COMPILED_FLIP_LOCK:
        if _COMPILED_FLIP_SWEEP is None:
            _COMPILED_FLIP_SWEEP = torch.compile(_flip_sweep, dynamic=True)
        return _COMPILED_FLIP_SWEEP(*args, **kwargs)


@torch.no_grad()
def flip_refine_levels(
    signed_index: torch.Tensor,
    scale_element: torch.Tensor,
    weight_target: torch.Tensor,
    hessian: torch.Tensor,
    n_iters: int,
    *,
    fast: bool = True,
    compile_flip: bool = False,
) -> None:
    """Greedy fixed-Hessian refinement of the E2M1 level codes, in place.

    For a single code change ``c`` the Hessian-weighted objective changes by
    ``delta = c^2 H_jj - 2 c g_j`` with ``g = H (w_target - w_current)``.  Each
    row takes its best improving single-level move per sweep.  ``fast=True``
    keeps an active set and drops rows that have converged; rows are decoupled
    here, so this is bit-identical to the dense sweep.

    ``compile_flip`` fuses each repeated sweep through ``torch.compile`` on
    NVIDIA CUDA with the GIL enabled. Other devices and free-threaded PyTorch
    use the eager sweep; concurrent compiler calls are not qualified there.
    """

    out_features, in_features = signed_index.shape
    device = signed_index.device
    grid = _device_const(E2M1_VALUES, device)
    hessian_diagonal = hessian.diagonal()
    inf = torch.tensor(float("inf"), device=device)
    weight_current = grid[signed_index] * scale_element
    hessian_weight_target = hessian @ weight_target.T
    hessian_weight_current = hessian @ weight_current.T
    sweep = (
        _compiled_flip_sweep
        if (
            compile_flip and device.type == "cuda" and not IS_ROCM
            and getattr(sys, "_is_gil_enabled", lambda: True)()
        )
        else _flip_sweep
    )

    if not fast:
        rows = torch.arange(out_features, device=device)
        for _ in range(n_iters):
            improving = sweep(
                signed_index, scale_element, hessian_weight_target, hessian_weight_current,
                rows, grid, inf, hessian_diagonal, hessian, active=False,
            )
            if not improving.any():
                break
        return

    active_rows = torch.arange(out_features, device=device)
    for _ in range(n_iters):
        active_rows = sweep(
            signed_index, scale_element, hessian_weight_target, hessian_weight_current,
            active_rows, grid, inf, hessian_diagonal, hessian, active=True,
        )
        if active_rows.numel() == 0:
            break


# ---------------------------------------------------------------------------
# Full per-Linear quantization
# ---------------------------------------------------------------------------
@dataclass
class MacaqQuantResult:
    """Packed artifacts for one MACaQ-quantized Linear."""

    W_packed: torch.Tensor
    W_scales: torch.Tensor
    BH: torch.Tensor
    in_features: int
    out_features: int
    block_size: int = MXFP4_BLOCK
    pre_scale: Optional[torch.Tensor] = None
    dequantized_weight: Optional[torch.Tensor] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def macaq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    n_tokens: float,
    *,
    block_size: int = MXFP4_BLOCK,
    had_block: int = MXFP4_BLOCK,
    n_flip_iters: int = 256,
    ridge: float = 1.0,
    do_flip: bool = True,
    pre_mode: str = "balance",
    pre_alpha: float = 0.75,
    exp_screen: bool = True,
    exp_offsets: Tuple[int, ...] = (1, 0, -1),
    fast_flip: bool = True,
    compile_flip: bool = False,
    return_dequantized: bool = False,
) -> MacaqQuantResult:
    """Quantize one ``[out, in]`` weight matrix with the MACaQ recipe."""

    device = weight.device
    weight = weight.float()
    hessian = hessian.float()
    out_features, in_features = weight.shape
    if in_features % block_size or in_features % had_block:
        raise ValueError(
            f"MACaQ: in_features {in_features} must be divisible by block_size={block_size} "
            f"and had_block={had_block}."
        )

    pre_scale = None
    if pre_mode != "none":
        pre_scale = balance_pre_scale(weight, hessian, n_tokens, alpha=pre_alpha)
        weight = weight / pre_scale.unsqueeze(0)
        hessian = hessian * pre_scale.unsqueeze(0) * pre_scale.unsqueeze(1)

    bh = make_hadamard(had_block, device=device, dtype=torch.float32)
    bh_bf16 = bh.to(torch.bfloat16)
    rotated_weight = rotate_blocks(weight, bh_bf16)
    rotated_hessian = rotate_blocks(rotate_blocks(hessian, bh_bf16).T, bh_bf16).T

    flip_hessian = None
    if do_flip or exp_screen:
        flip_hessian = rotated_hessian.clone()
        if ridge > 0:
            flip_hessian.diagonal().add_(ridge * flip_hessian.diagonal().mean())

    if exp_screen and flip_hessian is not None and len(exp_offsets) > 1:
        signed_index, log2_scale, scale_element = screen_block_exponents(
            rotated_weight, flip_hessian.diagonal(), block_size, exp_offsets
        )
    else:
        signed_index, log2_scale, scale_element = mxfp4_rtn_signed_index(
            rotated_weight, block_size
        )

    if do_flip:
        flip_refine_levels(
            signed_index, scale_element, rotated_weight, flip_hessian, n_flip_iters,
            fast=fast_flip, compile_flip=compile_flip,
        )

    packed, scales = pack_mxfp4_from_signed_index(signed_index, log2_scale, block_size)
    result = MacaqQuantResult(
        W_packed=packed,
        W_scales=scales,
        BH=bh_bf16,
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        pre_scale=None if pre_scale is None else pre_scale.to(torch.bfloat16),
    )
    if return_dequantized:
        grid = _device_const(E2M1_VALUES, device)
        result.dequantized_weight = (grid[signed_index] * scale_element).contiguous()
    return result


@torch.no_grad()
def macaq_quantize_activations(
    x: torch.Tensor,
    pre_scale: Optional[torch.Tensor] = None,
    bh: Optional[torch.Tensor] = None,
    block_size: int = MXFP4_BLOCK,
) -> torch.Tensor:
    """Torch reference W4A4 activation path: ``MXFP4((x * p) @ blkdiag(BH))``.

    Returns the dequantized activation, matching the Triton fused kernel's
    numerical contract closely enough for parity tests (the kernel rounds the
    rotated value through bfloat16).
    """

    lead = x.shape[:-1]
    k = x.shape[-1]
    work = x.reshape(-1, k).float()
    if pre_scale is not None:
        work = work * pre_scale.float()
    # The Triton kernel stores the scaled activation as bfloat16 before the
    # block rotation and rounds the rotated value back through bfloat16.
    work = work.to(torch.bfloat16).float()
    if bh is not None:
        rotated = rotate_blocks(work.to(torch.bfloat16).float(), bh.float())
        work = rotated.to(torch.bfloat16).float()
    signed_index, log2_scale, _ = mxfp4_rtn_signed_index(work, block_size, tie_high=True)
    packed, scales = pack_mxfp4_from_signed_index(signed_index, log2_scale, block_size)
    return dequantize_mxfp4(packed, scales, k, dtype=x.dtype).reshape(*lead, k)


class Macaq:
    """Torch MACaQ quantizer for one ``nn.Linear`` / ``transformers.Conv1D`` module."""

    def __init__(self, module: nn.Module, qcfg=None):
        from .config import MacaqConfig

        self.qcfg = qcfg if qcfg is not None else MacaqConfig()

        if isinstance(module, NamedModule):
            self._named_module = module
            self.layer = module.module
            self.name = module.name
        else:
            self._named_module = None
            self.layer = module
            self.name = getattr(module, "name", module.__class__.__name__)

        if not isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            raise ValueError(
                f"MACaQ supports only `nn.Linear` and `transformers.Conv1D` layers, got "
                f"`{self.layer.__class__.__name__}`."
            )

        weight = self.layer.weight
        if isinstance(self.layer, transformers.Conv1D):
            rows, columns = weight.shape[1], weight.shape[0]
        else:
            rows, columns = weight.shape[0], weight.shape[1]
        self.rows = int(rows)
        self.columns = int(columns)

        self.dev = weight.device
        self.nsamples = 0
        self.fwd_counter = 0
        self.lock = threading.Lock()
        self.H2: Optional[torch.Tensor] = None
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self.quant_result: Optional[MacaqQuantResult] = None

    def free(self) -> None:
        self.H2 = None
        self._device_hessian_partials.clear()
        self._device_sample_counts.clear()

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor] = None) -> None:
        """Accumulate ``H = sum(x^T x)`` for one calibration batch."""

        del out
        if inp.dim() == 0:
            raise ValueError("Macaq.add_batch expects an input with at least one dimension.")
        if inp.shape[-1] != self.columns:
            raise ValueError(
                f"Macaq.add_batch expected {self.columns} input features, got {inp.shape[-1]}."
            )
        inp = inp.reshape(-1, self.columns).t()
        device = inp.device
        tokens = inp.shape[1]
        inp = inp.to(dtype=_HESSIAN_DTYPE)

        with self.lock:
            self.fwd_counter += 1
            partial = self._device_hessian_partials.get(device)
            if partial is None:
                partial = torch.zeros((self.columns, self.columns), dtype=_HESSIAN_DTYPE, device=device)
                self._device_hessian_partials[device] = partial
            partial.addmm_(inp, inp.t(), beta=1.0, alpha=1.0)
            self._device_sample_counts[device] = self._device_sample_counts.get(device, 0) + tokens
            self.nsamples += tokens

    def materialize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """Merge per-device partials into the summed Hessian on ``target_device``."""

        if target_device is None:
            target_device = self.dev
        target_device = torch.device(target_device)
        if not self._device_hessian_partials:
            self.H2 = torch.zeros(
                (self.columns, self.columns), dtype=_HESSIAN_DTYPE, device=target_device
            )
            return self.H2

        merged = torch.zeros((self.columns, self.columns), dtype=_HESSIAN_DTYPE, device=target_device)
        while self._device_hessian_partials:
            device, partial = self._device_hessian_partials.popitem()
            if partial.device == target_device:
                merged.add_(partial)
            else:
                merged.add_(partial.to(target_device))
            del partial
        self.H2 = merged
        return self.H2

    def _layer_weight(self) -> torch.Tensor:
        weight = self.layer.weight.data
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        return weight.float()

    @torch.no_grad()
    def quantize(self) -> Tuple[MacaqQuantResult, float, float]:
        """Run MACaQ and return ``(result, duration_seconds, hessian_weighted_loss)``."""

        start = time.time()
        hessian = self.materialize_hessian(self.dev)
        weight = self._layer_weight().to(self.dev)
        if hessian.shape != (self.columns, self.columns):
            raise ValueError(
                f"MACaQ: Hessian shape {tuple(hessian.shape)} does not match "
                f"{self.columns} input features."
            )
        if not torch.isfinite(hessian).all() or hessian.diagonal().abs().sum() == 0:
            raise ValueError(
                f"MACaQ: Hessian for `{self.name}` has no usable calibration signal; "
                "increase calibration data or enable fallback."
            )

        result = macaq_quantize_weight(
            weight,
            hessian,
            n_tokens=max(self.nsamples, 1),
            block_size=int(self.qcfg.block_size),
            had_block=int(self.qcfg.had_block),
            n_flip_iters=int(self.qcfg.n_flip_iters),
            ridge=float(self.qcfg.ridge),
            do_flip=bool(self.qcfg.do_flip),
            pre_mode=str(self.qcfg.pre_mode),
            pre_alpha=float(self.qcfg.pre_alpha),
            exp_screen=bool(self.qcfg.exp_screen),
            exp_offsets=tuple(self.qcfg.exp_offsets),
            fast_flip=bool(self.qcfg.fast_flip),
            compile_flip=bool(self.qcfg.compile_flip),
            return_dequantized=True,
        )

        loss = self.hessian_weighted_error(
            weight_original=weight,
            dequantized=result.dequantized_weight,
            hessian=hessian,
            pre_scale=result.pre_scale,
            bh=result.BH,
        )
        duration = time.time() - start
        result.stats.update(loss=loss, nsamples=self.nsamples, duration=duration)
        self.quant_result = result
        return result, duration, loss

    @staticmethod
    @torch.no_grad()
    def hessian_weighted_error(
        *,
        weight_original: torch.Tensor,
        dequantized: torch.Tensor,
        hessian: torch.Tensor,
        pre_scale: Optional[torch.Tensor] = None,
        bh: Optional[torch.Tensor] = None,
    ) -> float:
        """H-weighted reconstruction error in the original weight frame."""

        # Compare in the original weight frame so the metric is the original
        # calibration Hessian. `dequantized` is stored in the (D1, Hadamard)
        # frame, so undo that transform first.
        weight = weight_original.float()
        restored = dequantized.float()
        if bh is not None:
            restored = rotate_blocks(restored, bh.float().T)
        if pre_scale is not None:
            restored = restored * pre_scale.float().unsqueeze(0)
        delta = restored.to(torch.float64) - weight.to(torch.float64)
        metric = hessian.to(torch.float64)
        error = (delta @ metric * delta).sum()
        denominator = (weight.to(torch.float64) @ metric * weight.to(torch.float64)).sum()
        return float((error / denominator.clamp_min(1e-12)).item())
