# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant optimization implementation adapted from the ParoQuant paper and
# public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""ParoQuant calibration-time optimization utilities.

This module implements the paper's transformed-domain PTQ lifecycle in a
direct way:
1. learn channel scales and Givens-rotation angles on calibration activations
2. initialize and optimize quantization parameters in the transformed domain
3. export packed runtime tensors that reproduce the pseudo-quantized layer
"""

from __future__ import annotations

import math
import random
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...utils.env import env_flag
from ...utils.paroquant import apply_paroquant_rotation_autograd, build_identity_rotation_buffers


_PAROQUANT_STAGE_PAIR_IMPLS: tuple[str, ...] = ("fast", "reference")
_PAROQUANT_QUANTIZER_IMPLS: tuple[str, ...] = ("fast", "reference")
_PAIR_CACHE_LOCK = threading.Lock()


def _normalize_opt_impl(name: str, *, field: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in _PAROQUANT_STAGE_PAIR_IMPLS:
        raise ValueError(
            f"ParoQuant optimization: `{field}` must be one of {_PAROQUANT_STAGE_PAIR_IMPLS}, got `{name}`."
        )
    return normalized


def _normalize_quantizer_impl(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in _PAROQUANT_QUANTIZER_IMPLS:
        raise ValueError(
            "ParoQuant optimization: `quantizer_impl` must be one of "
            f"{_PAROQUANT_QUANTIZER_IMPLS}, got `{name}`."
        )
    return normalized


def _quantizer_sym_for_impl(sym: bool, quantizer_impl: str) -> bool:
    impl = _normalize_quantizer_impl(quantizer_impl)
    if impl == "reference":
        return False
    return bool(sym)


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Apply a straight-through round so gradients flow through quantization."""
    return (x.round() - x).detach() + x


def _clamp_ste(
    x: torch.Tensor,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
) -> torch.Tensor:
    """Clamp with a straight-through estimator to stabilize learned qparams."""
    return (x.clamp(min_value, max_value) - x).detach() + x


def _normalize_group_size(group_size: int, in_features: int) -> int:
    """Validate and normalize a ParoQuant group size for a given hidden width."""
    normalized = in_features if group_size == -1 else int(group_size)
    if normalized <= 0:
        raise ValueError(f"ParoQuant optimization: invalid group_size `{group_size}` for in_features={in_features}.")
    if in_features % normalized != 0:
        raise ValueError(
            f"ParoQuant optimization: in_features ({in_features}) must be divisible by group_size ({normalized})."
        )
    if normalized % 2 != 0:
        raise ValueError(f"ParoQuant optimization: group_size ({normalized}) must be even.")
    return normalized


def _require_paroquant_sym(sym: bool) -> None:
    """Reject asymmetric ParoQuant configurations in this implementation."""
    if sym is not True:
        raise ValueError("ParoQuant optimization: `sym=False` is disabled; use `sym=True`.")


def _select_independent_pairs(
    all_pairs: Sequence[tuple[int, int]],
    dim: int,
    num_rotations: int,
    num_pairs_each: int,
) -> list[list[tuple[int, int]]]:
    """Choose non-overlapping channel pairs for each rotation step."""
    available = torch.ones(dim, dim, dtype=torch.bool)
    available.fill_diagonal_(False)
    rotations: list[list[tuple[int, int]]] = []

    for _ in range(num_rotations):
        available_in_rotation = available.clone()
        selected: list[tuple[int, int]] = []

        for i, j in all_pairs:
            if len(selected) >= num_pairs_each:
                break
            if not bool(available_in_rotation[i, j]):
                continue

            selected.append((i, j))
            available_in_rotation[i, :] = False
            available_in_rotation[j, :] = False
            available_in_rotation[:, i] = False
            available_in_rotation[:, j] = False
            available[i, j] = False
            available[j, i] = False

        rotations.append(selected)

    return rotations


def _pad_rotation_group(
    selected_pairs: Sequence[tuple[int, int]],
    group_size: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a sparse rotation schedule with dummy identity pairs."""
    half_group = group_size // 2
    pairs = torch.zeros((half_group, 2), dtype=torch.int16, device=device)
    mask = torch.zeros((half_group,), dtype=torch.bool, device=device)
    used = torch.zeros((group_size,), dtype=torch.bool, device=device)

    count = 0
    for i, j in selected_pairs:
        if count >= half_group:
            break
        pairs[count, 0] = int(i)
        pairs[count, 1] = int(j)
        used[i] = True
        used[j] = True
        count += 1

    if count == half_group:
        return pairs, mask

    remaining = [idx for idx in range(group_size) if not bool(used[idx])]
    if len(remaining) % 2 != 0:
        raise ValueError(f"ParoQuant optimization: unable to pad group of size {group_size}.")

    remaining_iter = iter(remaining)
    while count < half_group:
        try:
            i = next(remaining_iter)
            j = next(remaining_iter)
        except StopIteration as exc:
            raise ValueError(f"ParoQuant optimization: incomplete dummy-pair padding for group size {group_size}.") from exc
        pairs[count, 0] = int(i)
        pairs[count, 1] = int(j)
        mask[count] = True
        count += 1

    return pairs, mask


@lru_cache(maxsize=128)
def _build_random_rotation_buffers_cached_cpu(
    in_features: int,
    group_size: int,
    krot: int,
    pair_ratio: float,
    seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    return _build_random_rotation_buffers_cpu(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=pair_ratio,
        seed=seed,
    )


def _clear_random_rotation_buffers_cache() -> None:
    """Clear cached rotation buffers under a lock."""
    with _PAIR_CACHE_LOCK:
        _build_random_rotation_buffers_cached_cpu.cache_clear()


def _warm_random_rotation_buffers_cache(
    *,
    in_features: int,
    group_size: int,
    krot: int,
    pair_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Populate the cache under a lock and return the cached buffers."""
    with _PAIR_CACHE_LOCK:
        return _build_random_rotation_buffers_cached_cpu(
            in_features=in_features,
            group_size=group_size,
            krot=krot,
            pair_ratio=pair_ratio,
            seed=seed,
        )


def _build_random_rotation_buffers_cpu(
    *,
    in_features: int,
    group_size: int,
    krot: int,
    pair_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_group_size = _normalize_group_size(group_size, in_features)
    if krot <= 0:
        raise ValueError(f"ParoQuant optimization: `krot` must be positive, got {krot}.")
    if not (0.0 < float(pair_ratio) <= 0.5):
        raise ValueError("ParoQuant optimization: `pair_ratio` must be in the interval (0, 0.5].")

    rng = random.Random(int(seed))
    num_groups = in_features // normalized_group_size
    num_pairs_each = max(1, int(normalized_group_size * float(pair_ratio)))
    num_pairs_each = min(num_pairs_each, normalized_group_size // 2)

    rotation_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []

    for _ in range(krot):
        rotation_rows.append(torch.empty(0, dtype=torch.int16, device=torch.device("cpu")))
        mask_rows.append(torch.empty(0, dtype=torch.bool, device=torch.device("cpu")))

    for _ in range(num_groups):
        group_pairs = [(i, j) for i in range(normalized_group_size) for j in range(i + 1, normalized_group_size)]
        rng.shuffle(group_pairs)
        selected_per_rotation = _select_independent_pairs(
            group_pairs,
            normalized_group_size,
            krot,
            num_pairs_each,
        )

        for rot_idx in range(krot):
            padded_pairs, mask = _pad_rotation_group(
                selected_per_rotation[rot_idx],
                normalized_group_size,
                device=torch.device("cpu"),
            )
            rotation_rows[rot_idx] = torch.cat((rotation_rows[rot_idx], padded_pairs.reshape(-1)), dim=0)
            mask_rows[rot_idx] = torch.cat((mask_rows[rot_idx], mask), dim=0)

    pairs = torch.stack(rotation_rows, dim=0).contiguous()
    masks = torch.stack(mask_rows, dim=0).contiguous()
    return pairs, masks


def build_random_rotation_buffers(
    *,
    in_features: int,
    group_size: int,
    krot: int,
    pair_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build randomized pair schedules and masks for ParoQuant angle learning."""
    if krot <= 0:
        raise ValueError(f"ParoQuant optimization: `krot` must be positive, got {krot}.")
    if not (0.0 < float(pair_ratio) <= 0.5):
        raise ValueError("ParoQuant optimization: `pair_ratio` must be in the interval (0, 0.5].")

    pairs_cpu, masks_cpu = _build_random_rotation_buffers_cached_cpu(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=float(pair_ratio),
        seed=int(seed),
    )

    if torch.device(device).type == "cuda":
        return pairs_cpu.to(device=device), masks_cpu.to(device=device)
    return pairs_cpu, masks_cpu


def _get_independent_channel_pairs_reference(
    pairs: torch.Tensor,
    dim: int,
    num_rotations: int,
    num_pairs_each: int,
) -> list[list[tuple[int, int]]]:
    pairs_cpu = pairs.cpu().tolist()
    rotations_pairs: list[list[tuple[int, int]]] = []
    available = torch.ones(dim, dim)
    available.fill_diagonal_(0)

    for _ in range(num_rotations):
        independent_pairs: list[tuple[int, int]] = []
        available_in_rotation = available.clone()
        for i, j in pairs_cpu:
            if len(independent_pairs) == num_pairs_each:
                break
            if available_in_rotation[i, j] == 0:
                continue
            independent_pairs.append((i, j))
            available_in_rotation[i, :] = 0
            available_in_rotation[j, :] = 0
            available_in_rotation[:, i] = 0
            available_in_rotation[:, j] = 0
            available[i, j] = 0
            available[j, i] = 0
        rotations_pairs.append(independent_pairs)
    return rotations_pairs


def _align_pairs_to_kernel_shape_reference(
    pair: torch.Tensor,
    angle: torch.Tensor,
    *,
    group_size: int,
    include_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if pair.size(0) != angle.size(0) or pair.size(1) != 2:
        raise ValueError("ParoQuant optimization(reference): pair/angle shape mismatch.")

    group_idx = 0
    pair_ptr = 0
    pair_groups: list[torch.Tensor] = []
    angle_groups: list[torch.Tensor] = []
    mask_groups: list[torch.Tensor] = []

    while True:
        if pair_ptr >= pair.size(0):
            break
        occupied = torch.zeros((group_size), dtype=torch.int32)
        count = 0
        temp_pairs = torch.zeros((group_size // 2, 2), dtype=torch.int32, device=pair.device)
        temp_angle = torch.zeros((group_size // 2), dtype=torch.float, device=angle.device)
        temp_mask = torch.zeros((group_size // 2), dtype=torch.int32, device=angle.device)
        while count < group_size // 2:
            if (
                pair_ptr < pair.size(0)
                and pair[pair_ptr, 0] - group_idx * group_size < group_size
                and pair[pair_ptr, 1] - group_idx * group_size < group_size
            ):
                temp_pairs[count, :] = pair[pair_ptr, :]
                temp_angle[count] = angle[pair_ptr]
                if occupied[pair[pair_ptr, 0] % group_size] == 1 or occupied[pair[pair_ptr, 1] % group_size] == 1:
                    raise ValueError("ParoQuant optimization(reference): illegal pair.")
                occupied[pair[pair_ptr, :] % group_size] = 1
                pair_ptr += 1
            else:
                t_pair = torch.tensor([-1, -1])
                for i in range(group_size):
                    if occupied[i] == 0:
                        t_pair[0] = i
                        occupied[i] = 1
                        break
                for i in range(group_size):
                    if occupied[i] == 0:
                        t_pair[1] = i
                        occupied[i] = 1
                        break
                if t_pair[0] == -1 or t_pair[1] == -1:
                    raise ValueError("ParoQuant optimization(reference): unable to find dummy pair.")
                temp_pairs[count, :] = t_pair
                temp_angle[count] = float(0)
                temp_mask[count] = 1
            count += 1
        group_idx += 1
        pair_groups.append(temp_pairs)
        angle_groups.append(temp_angle)
        mask_groups.append(temp_mask)

    rotation_pairs = torch.cat(pair_groups, dim=0).view(-1).contiguous() % group_size
    angles = torch.cat(angle_groups, dim=0)
    masks = torch.cat(mask_groups, dim=0)
    if include_mask:
        return rotation_pairs, angles, masks
    return rotation_pairs, angles, None


def build_random_rotation_buffers_reference(
    *,
    in_features: int,
    group_size: int,
    krot: int,
    pair_ratio: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_group_size = _normalize_group_size(group_size, in_features)
    if krot <= 0:
        raise ValueError(f"ParoQuant optimization(reference): `krot` must be positive, got {krot}.")
    if not (0.0 < float(pair_ratio) <= 0.5):
        raise ValueError("ParoQuant optimization(reference): `pair_ratio` must be in the interval (0, 0.5].")

    rng = random.Random(int(seed))
    group_num = in_features // normalized_group_size
    num_pairs_per_group = int(normalized_group_size * float(pair_ratio))
    pairs_by_rotation: list[list[tuple[int, int]]] = [[] for _ in range(krot)]

    for group_idx in range(group_num):
        all_pairs = [(i, j) for i in range(normalized_group_size) for j in range(i + 1, normalized_group_size)]
        rng.shuffle(all_pairs)
        selected_by_rotation = _get_independent_channel_pairs_reference(
            torch.tensor(all_pairs),
            normalized_group_size,
            krot,
            num_pairs_per_group,
        )
        offset = group_idx * normalized_group_size
        for rotation_idx in range(krot):
            for col1, col2 in selected_by_rotation[rotation_idx]:
                pairs_by_rotation[rotation_idx].append((col1 + offset, col2 + offset))

    pair_tensors = [torch.tensor(pairs, dtype=torch.int32, device=device) for pairs in pairs_by_rotation]
    angle_tensors = [torch.zeros((pairs.shape[0],), dtype=torch.float32, device=device) for pairs in pair_tensors]

    aligned_pairs: list[torch.Tensor] = []
    aligned_masks: list[torch.Tensor] = []
    for pair_tensor, angle_tensor in zip(pair_tensors, angle_tensors):
        pair, _angle, mask = _align_pairs_to_kernel_shape_reference(
            pair_tensor,
            angle_tensor,
            group_size=normalized_group_size,
            include_mask=True,
        )
        aligned_pairs.append(pair.to(dtype=torch.int16))
        aligned_masks.append(mask.to(dtype=torch.bool))

    return torch.stack(aligned_pairs, dim=0).contiguous(), torch.stack(aligned_masks, dim=0).contiguous()


def _sample_activation_rows(inputs: torch.Tensor, max_rows: int) -> torch.Tensor:
    """Downsample calibration activations to a bounded replay set."""
    rows = inputs.reshape(-1, inputs.shape[-1])
    if rows.shape[0] <= max_rows:
        return rows
    indices = torch.linspace(0, rows.shape[0] - 1, steps=max_rows, device=rows.device)
    indices = torch.round(indices).to(dtype=torch.long)
    return rows.index_select(0, indices)


def _apply_rotation(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    *,
    scales: Optional[torch.Tensor],
    group_size: int,
    fused_rotation: Optional[bool] = None,
) -> torch.Tensor:
    """Apply the forward ParoQuant transform in the optimization domain."""
    if x.dim() != 2:
        raise ValueError(f"ParoQuant optimization expects a rank-2 tensor, got {tuple(x.shape)}.")

    use_fused_rotation = (
        env_flag("GPTQMODEL_PAROQUANT_OPT_FUSED_ROTATION", default=True)
        if fused_rotation is None
        else bool(fused_rotation)
    )

    if use_fused_rotation:
        scale_tensor = None if scales is None else scales.view(1, -1)
        return apply_paroquant_rotation_autograd(
            x,
            pairs,
            theta,
            scales=scale_tensor,
            group_size=group_size,
        )

    out = x
    if scales is not None:
        out = out * scales.view(1, -1)

    hidden = out.shape[-1]
    normalized_group_size = _normalize_group_size(group_size, hidden)
    num_groups = hidden // normalized_group_size
    half_group = normalized_group_size // 2
    offsets = torch.arange(num_groups, device=out.device, dtype=torch.long).unsqueeze(1) * normalized_group_size
    pair_view = pairs.to(device=out.device, dtype=torch.long).view(pairs.shape[0], num_groups, half_group, 2)
    theta_view = theta.to(device=out.device, dtype=out.dtype).view(theta.shape[0], num_groups, half_group)

    for rot_idx in range(pair_view.shape[0]):
        idx_i = (pair_view[rot_idx, :, :, 0] + offsets).reshape(-1)
        idx_j = (pair_view[rot_idx, :, :, 1] + offsets).reshape(-1)

        xi = out[:, idx_i].view(out.shape[0], num_groups, half_group)
        xj = out[:, idx_j].view(out.shape[0], num_groups, half_group)
        cos_t = torch.cos(theta_view[rot_idx]).unsqueeze(0)
        sin_t = torch.sin(theta_view[rot_idx]).unsqueeze(0)

        next_out = out.clone()
        next_out[:, idx_i] = (xi * cos_t + xj * sin_t).reshape(out.shape[0], -1)
        next_out[:, idx_j] = (-xi * sin_t + xj * cos_t).reshape(out.shape[0], -1)
        out = next_out

    return out


def _apply_inverse_rotation(
    x: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    *,
    group_size: int,
    fused_rotation: Optional[bool] = None,
) -> torch.Tensor:
    """Apply the inverse transform that maps export-domain weights back to input space."""
    if pairs.shape[0] == 0:
        return x
    return _apply_rotation(
        x,
        pairs.flip(0),
        -theta.flip(0),
        scales=None,
        group_size=group_size,
        fused_rotation=fused_rotation,
    )


def _reshape_group_params(weight: torch.Tensor, group_size: int, values: torch.Tensor) -> torch.Tensor:
    """Restore flat per-group quantizer values to the packed weight layout."""
    groups = weight.shape[1] // group_size
    return values.view(weight.shape[0], groups)


def _calc_affine_qparams(
    weight: torch.Tensor,
    *,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute affine quantizer parameters from transformed weights."""
    view = weight.reshape(-1, group_size)
    min_val = view.amin(dim=1, keepdim=True)
    max_val = view.amax(dim=1, keepdim=True)
    qmax = 2**bits - 1
    scale = (max_val - min_val).clamp(min=1e-5) / qmax
    zero_point_float = min_val / scale
    return scale, zero_point_float


class GroupLinearQuantizer(nn.Module):
    """Learnable per-group quantizer matching ParoQuant's transformed-domain packing."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        sym: bool,
    ) -> None:
        """Initialize either symmetric or affine groupwise quantization parameters."""
        super().__init__()
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.sym = bool(sym)

        if self.sym:
            view = weight.reshape(-1, self.group_size)
            qmax = 2 ** (self.bits - 1) - 1
            scale = view.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / qmax
            self.scale = nn.Parameter(scale)
            self.zero_point_float = None
        else:
            scale, zero_point_float = _calc_affine_qparams(weight, group_size=self.group_size, bits=self.bits)
            self.scale = nn.Parameter(scale)
            self.zero_point_float = nn.Parameter(zero_point_float)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Pseudo-quantize a transformed weight tensor with STE-enabled qparams."""
        return pseudo_quantize_dequant(
            weight,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            scale=self.scale,
            zero_point_float=self.zero_point_float,
            use_ste=True,
        )

    def optim_params(self) -> list[nn.Parameter]:
        """Return only the quantizer parameters that should receive optimizer updates."""
        params = [self.scale]
        if self.zero_point_float is not None:
            params.append(self.zero_point_float)
        return params

    def pack_params(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert learned qparams into the runtime scale/zero-point tensors."""
        scales = _reshape_group_params(weight, self.group_size, self.scale.detach())
        if self.sym:
            zeros = torch.full_like(scales, 2 ** (self.bits - 1))
        else:
            qmax = 2**self.bits - 1
            zeros = _clamp_ste(-self.zero_point_float.detach().round(), 0, qmax)
            zeros = _reshape_group_params(weight, self.group_size, zeros)
        return scales, zeros


def pseudo_quantize_dequant(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
    scale: Optional[torch.Tensor] = None,
    zero_point_float: Optional[torch.Tensor] = None,
    use_ste: bool,
) -> torch.Tensor:
    """Reference pseudo-quantization path shared by optimization and export tests."""
    dtype = weight.dtype
    weight_view = weight.reshape(-1, group_size)

    if sym:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        if scale is None:
            scale = weight_view.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / qmax
        if use_ste:
            scale = _clamp_ste(scale, min_value=1e-5, max_value=1e5)
            quant = _clamp_ste(_round_ste(weight_view / scale), qmin, qmax)
        else:
            scale = scale.clamp(min=1e-5, max=1e5)
            quant = torch.clamp(torch.round(weight_view / scale), qmin, qmax)
        dequant = quant * scale
    else:
        qmin = 0
        qmax = 2**bits - 1
        if scale is None or zero_point_float is None:
            scale, zero_point_float = _calc_affine_qparams(weight, group_size=group_size, bits=bits)
        if use_ste:
            scale = _clamp_ste(scale, min_value=1e-5, max_value=1e5)
            round_zero_point = _clamp_ste(-_round_ste(zero_point_float), qmin, qmax)
            quant = _round_ste(weight_view / scale) + round_zero_point
            quant = _clamp_ste(quant, qmin, qmax)
        else:
            scale = scale.clamp(min=1e-5, max=1e5)
            round_zero_point = torch.clamp(-torch.round(zero_point_float), qmin, qmax)
            quant = torch.round(weight_view / scale) + round_zero_point
            quant = torch.clamp(quant, qmin, qmax)
        dequant = (quant - round_zero_point) * scale

    return dequant.reshape_as(weight).to(dtype)


class _ParoQuantOptimLinear(nn.Module):
    """Minimal layer wrapper used during ParoQuant calibration optimization."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        bits: int,
        group_size: int,
        quantizer_sym: bool,
        pairs: torch.Tensor,
        theta_mask: torch.Tensor,
        fused_rotation: Optional[bool] = None,
    ) -> None:
        """Materialize a replayable linear layer in the original input domain."""
        super().__init__()
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.quantizer_sym = bool(quantizer_sym)
        self.register_buffer("pairs", pairs)
        self.register_buffer("theta_mask", theta_mask)
        self.weight = nn.Parameter(weight.clone())
        self.bias = None if bias is None else nn.Parameter(bias.clone())
        self.theta = nn.Parameter(torch.zeros((pairs.shape[0], weight.shape[1] // 2), device=weight.device, dtype=weight.dtype))
        self.channel_scales_opt = nn.Parameter(torch.ones((weight.shape[1],), device=weight.device, dtype=weight.dtype))
        self.quantizer: Optional[GroupLinearQuantizer] = None
        self.fused_rotation = fused_rotation

    def transformed_weight(self) -> torch.Tensor:
        """Project the learnable weight into ParoQuant's transformed domain."""
        scaled_weight = self.weight * self.channel_scales_opt.view(1, -1)
        return _apply_rotation(
            scaled_weight,
            self.pairs,
            self.theta,
            scales=None,
            group_size=self.group_size,
            fused_rotation=self.fused_rotation,
        )

    def quantized_transformed_weight(self) -> torch.Tensor:
        """Pseudo-quantize the transformed weight using current learned qparams."""
        transformed = self.transformed_weight()
        if self.quantizer is None:
            return pseudo_quantize_dequant(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.quantizer_sym,
                use_ste=True,
            )
        return self.quantizer(transformed)

    def pseudo_weight(self) -> torch.Tensor:
        """Map the transformed-domain quantized weight back to runtime input space."""
        quantized = self.quantized_transformed_weight()
        quantized = _apply_inverse_rotation(
            quantized,
            self.pairs,
            self.theta,
            group_size=self.group_size,
            fused_rotation=self.fused_rotation,
        )
        channel_scales = self.channel_scales_opt.view(1, -1).clamp(min=1e-5)
        return quantized / channel_scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replay calibration activations through the current pseudo-quantized layer."""
        return F.linear(x, self.pseudo_weight(), self.bias)

    def reset_masked_angles(self) -> None:
        """Force dummy padded pairs to stay at zero angle during optimization."""
        with torch.no_grad():
            self.theta.masked_fill_(self.theta_mask, 0)

    def init_quantizer(self) -> None:
        """Bootstrap the transformed-domain quantizer from the current rotated weight."""
        transformed = self.transformed_weight().detach()
        self.quantizer = GroupLinearQuantizer(
            transformed,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.quantizer_sym,
        )

    def export_pack_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Export runtime tensors that should match the pseudo-quantized layer exactly."""
        transformed = self.transformed_weight().detach()
        if self.quantizer is None:
            quantizer = GroupLinearQuantizer(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.quantizer_sym,
            )
            pack_scales, pack_zeros = quantizer.pack_params(transformed)
            quantized = pseudo_quantize_dequant(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.quantizer_sym,
                scale=quantizer.scale.detach(),
                zero_point_float=None if quantizer.zero_point_float is None else quantizer.zero_point_float.detach(),
                use_ste=False,
            )
        else:
            pack_scales, pack_zeros = self.quantizer.pack_params(transformed)
            quantized = pseudo_quantize_dequant(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.quantizer_sym,
                scale=self.quantizer.scale.detach(),
                zero_point_float=(
                    None if self.quantizer.zero_point_float is None else self.quantizer.zero_point_float.detach()
                ),
                use_ste=False,
            )

        theta = self.theta.detach().masked_fill(self.theta_mask, 0)
        runtime_channel_scales = self.channel_scales_opt.detach().clamp(min=1e-5).reciprocal().view(1, -1)
        return quantized, pack_scales, pack_zeros, theta, runtime_channel_scales


@dataclass
class ParoQuantOptimizationResult:
    """All tensors and diagnostics produced by one ParoQuant layer optimization run."""

    pseudo_weight: torch.Tensor
    pack_weight: torch.Tensor
    q_scales: torch.Tensor
    q_zeros: torch.Tensor
    pairs: torch.Tensor
    theta: torch.Tensor
    channel_scales: torch.Tensor
    train_loss: float
    val_loss: float
    used_identity: bool


def _chunk_rows(rows: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    """Yield contiguous mini-batches from flattened calibration activations."""
    for start in range(0, rows.shape[0], batch_size):
        yield rows[start:start + batch_size]


def _evaluate_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_amp: bool = False,
) -> float:
    """Measure replay error for early stopping and stage selection."""
    if inputs.numel() == 0:
        return 0.0
    with torch.no_grad():
        autocast_ctx = torch.amp.autocast("cuda") if use_amp and inputs.device.type == "cuda" else nullcontext()
        with autocast_ctx:
            preds = model(inputs)
            loss = F.smooth_l1_loss(preds, targets)
        return float(loss.item())


def _run_stage_gptqmodel(
    *,
    model: nn.Module,
    inputs_train: torch.Tensor,
    targets_train: torch.Tensor,
    inputs_val: torch.Tensor,
    targets_val: torch.Tensor,
    param_groups: Sequence[dict[str, object]],
    epochs: int,
    batch_size: int,
    ) -> tuple[float, float]:
    """Run one optimization stage with validation-based best-state selection."""
    normalized_groups = []
    for param_group in param_groups:
        params = [param for param in param_group.get("params", []) if param.requires_grad]
        if not params:
            continue
        normalized_groups.append(
            {
                "params": params,
                "lr": float(param_group["lr"]),
                "weight_decay": float(param_group.get("weight_decay", 0.01)),
                "betas": tuple(param_group.get("betas", (0.9, 0.95))),
                "eps": float(param_group.get("eps", 1e-10)),
            }
        )

    use_amp = inputs_train.device.type == "cuda"
    if epochs <= 0 or not normalized_groups:
        train_loss = _evaluate_model(model, inputs_train, targets_train, use_amp=use_amp)
        val_loss = _evaluate_model(model, inputs_val, targets_val, use_amp=use_amp)
        return train_loss, val_loss

    optimizer = torch.optim.AdamW(normalized_groups)
    steps_per_epoch = max(1, math.ceil(max(1, inputs_train.shape[0]) / max(1, batch_size)))
    total_steps = max(1, epochs * steps_per_epoch)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler(enabled=use_amp)
    global_step = 0

    best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    best_val_loss = float("inf")
    last_train_loss = _evaluate_model(model, inputs_train, targets_train, use_amp=use_amp)

    for _epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for input_batch, target_batch in zip(_chunk_rows(inputs_train, batch_size), _chunk_rows(targets_train, batch_size)):
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with autocast_ctx:
                preds = model(input_batch)
                loss = F.smooth_l1_loss(preds, target_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * min(global_step, total_steps) / total_steps))
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = (base_lr / 20.0) + ((base_lr - (base_lr / 20.0)) * cosine_ratio)

            model.reset_masked_angles()
            epoch_loss += float(loss.item())
            batch_count += 1

        last_train_loss = epoch_loss / max(1, batch_count)
        val_loss = _evaluate_model(model, inputs_val, targets_val, use_amp=use_amp)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    model.reset_masked_angles()
    return last_train_loss, best_val_loss


def _run_stage_reference(
    *,
    model: nn.Module,
    inputs_train: torch.Tensor,
    targets_train: torch.Tensor,
    inputs_val: torch.Tensor,
    targets_val: torch.Tensor,
    param_groups: Sequence[dict[str, object]],
    epochs: int,
    batch_size: int,
) -> tuple[float, float]:
    """Official-parity stage runner: AMP + GradScaler + cosine LR update."""
    normalized_groups = []
    for param_group in param_groups:
        params = [param for param in param_group.get("params", []) if param.requires_grad]
        if not params:
            continue
        normalized_groups.append(
            {
                "params": params,
                "lr": float(param_group["lr"]),
                "weight_decay": float(param_group.get("weight_decay", 0.01)),
                "betas": tuple(param_group.get("betas", (0.9, 0.95))),
                "eps": float(param_group.get("eps", 1e-10)),
            }
        )

    use_amp = inputs_train.device.type == "cuda"
    if epochs <= 0 or not normalized_groups:
        train_loss = _evaluate_model(model, inputs_train, targets_train, use_amp=use_amp)
        val_loss = _evaluate_model(model, inputs_val, targets_val, use_amp=use_amp)
        return train_loss, val_loss

    optimizer = torch.optim.AdamW(normalized_groups)
    steps_per_epoch = max(1, math.ceil(max(1, inputs_train.shape[0]) / max(1, batch_size)))
    total_steps = max(1, epochs * steps_per_epoch)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler(enabled=use_amp)
    global_step = 0

    best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    best_val_loss = _evaluate_model(model, inputs_val, targets_val, use_amp=use_amp)
    last_train_loss = _evaluate_model(model, inputs_train, targets_train, use_amp=use_amp)

    for _ in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)

        for input_batch, target_batch in zip(_chunk_rows(inputs_train, batch_size), _chunk_rows(targets_train, batch_size)):
            autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with autocast_ctx:
                preds = model(input_batch)
                loss = F.smooth_l1_loss(preds, target_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * min(global_step, total_steps) / total_steps))
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = (base_lr / 20.0) + ((base_lr - (base_lr / 20.0)) * cosine_ratio)

            model.reset_masked_angles()
            epoch_loss += float(loss.item())
            batch_count += 1

        last_train_loss = epoch_loss / max(1, batch_count)
        val_loss = _evaluate_model(model, inputs_val, targets_val, use_amp=use_amp)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    model.reset_masked_angles()
    return last_train_loss, best_val_loss


def _run_stage(
    *,
    model: nn.Module,
    inputs_train: torch.Tensor,
    targets_train: torch.Tensor,
    inputs_val: torch.Tensor,
    targets_val: torch.Tensor,
    param_groups: Sequence[dict[str, object]],
    epochs: int,
    batch_size: int,
    stage_impl: str,
) -> tuple[float, float]:
    impl = _normalize_opt_impl(stage_impl, field="stage_impl")
    if impl == "reference":
        return _run_stage_reference(
            model=model,
            inputs_train=inputs_train,
            targets_train=targets_train,
            inputs_val=inputs_val,
            targets_val=targets_val,
            param_groups=param_groups,
            epochs=epochs,
            batch_size=batch_size,
        )
    return _run_stage_gptqmodel(
        model=model,
        inputs_train=inputs_train,
        targets_train=targets_train,
        inputs_val=inputs_val,
        targets_val=targets_val,
        param_groups=param_groups,
        epochs=epochs,
        batch_size=batch_size,
    )


def _result_from_model(
    model: _ParoQuantOptimLinear,
    *,
    train_loss: float,
    val_loss: float,
    used_identity: bool,
) -> ParoQuantOptimizationResult:
    """Export one optimized linear replay module into the runtime tensor contract."""
    pseudo_weight = model.pseudo_weight().detach()
    pack_weight, q_scales, q_zeros, theta, channel_scales = model.export_pack_state()
    return ParoQuantOptimizationResult(
        pseudo_weight=pseudo_weight,
        pack_weight=pack_weight.detach(),
        q_scales=q_scales.detach(),
        q_zeros=q_zeros.detach(),
        pairs=model.pairs.detach(),
        theta=theta.detach(),
        channel_scales=channel_scales.detach(),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        used_identity=used_identity,
    )


def _identity_result(
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    quantizer_sym: bool,
    krot: int,
) -> ParoQuantOptimizationResult:
    """Return the no-optimization fallback used when calibration activations are missing."""
    del bias
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=weight.shape[1],
        group_size=group_size,
        krot=krot,
        device=weight.device,
        dtype=weight.dtype,
    )
    quantizer = GroupLinearQuantizer(
        weight,
        bits=bits,
        group_size=_normalize_group_size(group_size, weight.shape[1]),
        sym=quantizer_sym,
    )
    q_scales, q_zeros = quantizer.pack_params(weight)
    pack_weight = pseudo_quantize_dequant(
        weight,
        bits=bits,
        group_size=_normalize_group_size(group_size, weight.shape[1]),
        sym=quantizer_sym,
        scale=quantizer.scale.detach(),
        zero_point_float=None if quantizer.zero_point_float is None else quantizer.zero_point_float.detach(),
        use_ste=False,
    )
    return ParoQuantOptimizationResult(
        pseudo_weight=pack_weight.detach(),
        pack_weight=pack_weight.detach(),
        q_scales=q_scales.detach(),
        q_zeros=q_zeros.detach(),
        pairs=pairs.detach(),
        theta=theta.detach(),
        channel_scales=channel_scales.detach(),
        train_loss=0.0,
        val_loss=0.0,
        used_identity=True,
    )


def optimize_paroquant_linear(
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    inputs: torch.Tensor,
    bits: int,
    group_size: int,
    sym: bool,
    krot: int,
    pair_ratio: float,
    train_rows: int,
    val_rows: int,
    batch_size: int,
    rotation_epochs: int,
    finetune_epochs: int,
    rotation_lr: float,
    weight_lr: float,
    quantizer_lr: float,
    seed: int,
    fused_rotation: Optional[bool] = None,
    stage_impl: Literal["fast", "reference"] = "fast",
    pair_impl: Literal["fast", "reference"] = "fast",
    quantizer_impl: Literal["fast", "reference"] = "fast",
) -> ParoQuantOptimizationResult:
    """Optimize one linear layer following the paper's two-stage PTQ schedule."""
    _require_paroquant_sym(sym)
    if weight.dim() != 2:
        raise ValueError(f"ParoQuant optimization expects rank-2 weights, got {tuple(weight.shape)}.")

    normalized_group_size = _normalize_group_size(group_size, weight.shape[1])
    quantizer_sym = _quantizer_sym_for_impl(sym, quantizer_impl)
    rows = _sample_activation_rows(inputs, max_rows=max(1, int(train_rows) + int(val_rows)))
    if rows.numel() == 0:
        return _identity_result(
            weight=weight,
            bias=bias,
            bits=bits,
            group_size=normalized_group_size,
            quantizer_sym=quantizer_sym,
            krot=krot,
        )

    opt_device = weight.device
    opt_dtype = torch.float32
    weight_opt = weight.detach().to(device=opt_device, dtype=opt_dtype)
    bias_opt = None if bias is None else bias.detach().to(device=opt_device, dtype=opt_dtype)
    rows = rows.to(device=opt_device, dtype=opt_dtype)

    targets = F.linear(rows, weight_opt, bias_opt)
    train_count = min(rows.shape[0], max(1, int(train_rows)))
    val_count = min(max(1, int(val_rows)), max(1, rows.shape[0] - train_count))
    inputs_train = rows[:train_count]
    targets_train = targets[:train_count]
    inputs_val = rows[-val_count:]
    targets_val = targets[-val_count:]

    if inputs_train.numel() == 0 or targets_train.numel() == 0:
        raise ValueError("ParoQuant optimization requires non-empty training activations.")

    normalized_pair_impl = _normalize_opt_impl(pair_impl, field="pair_impl")
    normalized_stage_impl = _normalize_opt_impl(stage_impl, field="stage_impl")
    _normalize_quantizer_impl(quantizer_impl)
    if normalized_pair_impl == "reference":
        pairs, theta_mask = build_random_rotation_buffers_reference(
            in_features=weight_opt.shape[1],
            group_size=normalized_group_size,
            krot=krot,
            pair_ratio=pair_ratio,
            seed=seed,
            device=opt_device,
        )
    else:
        pairs, theta_mask = build_random_rotation_buffers(
            in_features=weight_opt.shape[1],
            group_size=normalized_group_size,
            krot=krot,
            pair_ratio=pair_ratio,
            seed=seed,
            device=opt_device,
        )
    model = _ParoQuantOptimLinear(
        weight_opt,
        bias_opt,
        bits=bits,
        group_size=normalized_group_size,
        quantizer_sym=quantizer_sym,
        pairs=pairs,
        theta_mask=theta_mask,
        fused_rotation=fused_rotation,
    ).to(device=opt_device, dtype=opt_dtype)
    model.reset_masked_angles()

    _, _ = _run_stage(
        model=model,
        inputs_train=inputs_train,
        targets_train=targets_train,
        inputs_val=inputs_val,
        targets_val=targets_val,
        param_groups=[
            {"params": [model.channel_scales_opt], "lr": rotation_lr},
            {"params": [model.theta], "lr": rotation_lr},
        ],
        epochs=rotation_epochs,
        batch_size=batch_size,
        stage_impl=normalized_stage_impl,
    )

    model.init_quantizer()
    train_loss, val_loss = _run_stage(
        model=model,
        inputs_train=inputs_train,
        targets_train=targets_train,
        inputs_val=inputs_val,
        targets_val=targets_val,
        param_groups=[
            {"params": [model.weight], "lr": weight_lr},
            {"params": model.quantizer.optim_params(), "lr": quantizer_lr},
        ],
        epochs=finetune_epochs,
        batch_size=batch_size,
        stage_impl=normalized_stage_impl,
    )

    return _result_from_model(
        model,
        train_loss=train_loss,
        val_loss=val_loss,
        used_identity=False,
    )


__all__ = [
    "ParoQuantOptimizationResult",
    "build_random_rotation_buffers",
    "optimize_paroquant_linear",
    "pseudo_quantize_dequant",
]
