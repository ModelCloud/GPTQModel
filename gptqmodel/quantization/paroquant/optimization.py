# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...utils.paroquant import build_identity_rotation_buffers


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _clamp_ste(
    x: torch.Tensor,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
) -> torch.Tensor:
    return (x.clamp(min_value, max_value) - x).detach() + x


def _normalize_group_size(group_size: int, in_features: int) -> int:
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


def _select_independent_pairs(
    all_pairs: Sequence[tuple[int, int]],
    dim: int,
    num_rotations: int,
    num_pairs_each: int,
) -> list[list[tuple[int, int]]]:
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


def build_random_rotation_buffers(
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
        rotation_rows.append(torch.empty(0, dtype=torch.int16, device=device))
        mask_rows.append(torch.empty(0, dtype=torch.bool, device=device))

    for _group_index in range(num_groups):
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
                device=device,
            )
            rotation_rows[rot_idx] = torch.cat((rotation_rows[rot_idx], padded_pairs.reshape(-1)), dim=0)
            mask_rows[rot_idx] = torch.cat((mask_rows[rot_idx], mask), dim=0)

    return torch.stack(rotation_rows, dim=0).contiguous(), torch.stack(mask_rows, dim=0).contiguous()


def _sample_activation_rows(inputs: torch.Tensor, max_rows: int) -> torch.Tensor:
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
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"ParoQuant optimization expects a rank-2 tensor, got {tuple(x.shape)}.")

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
) -> torch.Tensor:
    if pairs.shape[0] == 0:
        return x
    return _apply_rotation(
        x,
        pairs.flip(0),
        -theta.flip(0),
        scales=None,
        group_size=group_size,
    )


def _reshape_group_params(weight: torch.Tensor, group_size: int, values: torch.Tensor) -> torch.Tensor:
    groups = weight.shape[1] // group_size
    return values.view(weight.shape[0], groups)


def _calc_affine_qparams(
    weight: torch.Tensor,
    *,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    view = weight.reshape(-1, group_size)
    min_val = view.amin(dim=1, keepdim=True)
    max_val = view.amax(dim=1, keepdim=True)
    qmax = 2**bits - 1
    scale = (max_val - min_val).clamp(min=1e-5) / qmax
    zero_point_float = min_val / scale
    return scale, zero_point_float


class GroupLinearQuantizer(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        bits: int,
        group_size: int,
        sym: bool,
    ) -> None:
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
        params = [self.scale]
        if self.zero_point_float is not None:
            params.append(self.zero_point_float)
        return params

    def pack_params(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        bits: int,
        group_size: int,
        sym: bool,
        pairs: torch.Tensor,
        theta_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.sym = bool(sym)
        self.register_buffer("pairs", pairs)
        self.register_buffer("theta_mask", theta_mask)
        self.weight = nn.Parameter(weight.clone())
        self.bias = None if bias is None else nn.Parameter(bias.clone())
        self.theta = nn.Parameter(torch.zeros((pairs.shape[0], weight.shape[1] // 2), device=weight.device, dtype=weight.dtype))
        self.channel_scales_opt = nn.Parameter(torch.ones((weight.shape[1],), device=weight.device, dtype=weight.dtype))
        self.quantizer: Optional[GroupLinearQuantizer] = None

    def transformed_weight(self) -> torch.Tensor:
        scaled_weight = self.weight * self.channel_scales_opt.view(1, -1)
        return _apply_rotation(
            scaled_weight,
            self.pairs,
            self.theta,
            scales=None,
            group_size=self.group_size,
        )

    def quantized_transformed_weight(self) -> torch.Tensor:
        transformed = self.transformed_weight()
        if self.quantizer is None:
            return pseudo_quantize_dequant(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.sym,
                use_ste=True,
            )
        return self.quantizer(transformed)

    def pseudo_weight(self) -> torch.Tensor:
        quantized = self.quantized_transformed_weight()
        quantized = _apply_inverse_rotation(
            quantized,
            self.pairs,
            self.theta,
            group_size=self.group_size,
        )
        channel_scales = self.channel_scales_opt.view(1, -1).clamp(min=1e-5)
        return quantized / channel_scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.pseudo_weight(), self.bias)

    def reset_masked_angles(self) -> None:
        with torch.no_grad():
            self.theta.masked_fill_(self.theta_mask, 0)

    def init_quantizer(self) -> None:
        transformed = self.transformed_weight().detach()
        self.quantizer = GroupLinearQuantizer(
            transformed,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
        )

    def export_pack_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed = self.transformed_weight().detach()
        if self.quantizer is None:
            quantizer = GroupLinearQuantizer(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.sym,
            )
            pack_scales, pack_zeros = quantizer.pack_params(transformed)
            quantized = pseudo_quantize_dequant(
                transformed,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.sym,
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
                sym=self.sym,
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
    for start in range(0, rows.shape[0], batch_size):
        yield rows[start:start + batch_size]


def _evaluate_model(model: _ParoQuantOptimLinear, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    if inputs.numel() == 0:
        return 0.0
    with torch.no_grad():
        preds = model(inputs)
        return float(F.smooth_l1_loss(preds, targets).item())


def _run_stage(
    *,
    model: _ParoQuantOptimLinear,
    inputs_train: torch.Tensor,
    targets_train: torch.Tensor,
    inputs_val: torch.Tensor,
    targets_val: torch.Tensor,
    param_groups: Sequence[dict[str, object]],
    epochs: int,
    batch_size: int,
) -> tuple[float, float]:
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

    if epochs <= 0 or not normalized_groups:
        train_loss = _evaluate_model(model, inputs_train, targets_train)
        val_loss = _evaluate_model(model, inputs_val, targets_val)
        return train_loss, val_loss

    optimizer = torch.optim.AdamW(normalized_groups)
    steps_per_epoch = max(1, math.ceil(max(1, inputs_train.shape[0]) / max(1, batch_size)))
    total_steps = max(1, epochs * steps_per_epoch)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    global_step = 0

    best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    best_val_loss = float("inf")
    last_train_loss = _evaluate_model(model, inputs_train, targets_train)

    for _epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for input_batch, target_batch in zip(_chunk_rows(inputs_train, batch_size), _chunk_rows(targets_train, batch_size)):
            optimizer.zero_grad(set_to_none=True)
            preds = model(input_batch)
            loss = F.smooth_l1_loss(preds, target_batch)
            loss.backward()
            optimizer.step()
            global_step += 1
            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * min(global_step, total_steps) / total_steps))
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = (base_lr / 20.0) + ((base_lr - (base_lr / 20.0)) * cosine_ratio)

            model.reset_masked_angles()
            epoch_loss += float(loss.item())
            batch_count += 1

        last_train_loss = epoch_loss / max(1, batch_count)
        val_loss = _evaluate_model(model, inputs_val, targets_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    model.reset_masked_angles()
    return last_train_loss, best_val_loss


def _identity_result(
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    sym: bool,
    krot: int,
) -> ParoQuantOptimizationResult:
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
        sym=sym,
    )
    q_scales, q_zeros = quantizer.pack_params(weight)
    pack_weight = pseudo_quantize_dequant(
        weight,
        bits=bits,
        group_size=_normalize_group_size(group_size, weight.shape[1]),
        sym=sym,
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
) -> ParoQuantOptimizationResult:
    if weight.dim() != 2:
        raise ValueError(f"ParoQuant optimization expects rank-2 weights, got {tuple(weight.shape)}.")

    normalized_group_size = _normalize_group_size(group_size, weight.shape[1])
    rows = _sample_activation_rows(inputs, max_rows=max(1, int(train_rows) + int(val_rows)))
    if rows.numel() == 0:
        return _identity_result(
            weight=weight,
            bias=bias,
            bits=bits,
            group_size=normalized_group_size,
            sym=sym,
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
        sym=sym,
        pairs=pairs,
        theta_mask=theta_mask,
    ).to(device=opt_device, dtype=opt_dtype)
    model.reset_masked_angles()

    train_loss, val_loss = _run_stage(
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
    )

    pseudo_weight = model.pseudo_weight().detach()
    pack_weight, q_scales, q_zeros, theta, channel_scales = model.export_pack_state()

    return ParoQuantOptimizationResult(
        pseudo_weight=pseudo_weight,
        pack_weight=pack_weight.detach(),
        q_scales=q_scales.detach(),
        q_zeros=q_zeros.detach(),
        pairs=pairs.detach(),
        theta=theta.detach(),
        channel_scales=channel_scales.detach(),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        used_identity=False,
    )


__all__ = [
    "ParoQuantOptimizationResult",
    "build_random_rotation_buffers",
    "optimize_paroquant_linear",
    "pseudo_quantize_dequant",
]
