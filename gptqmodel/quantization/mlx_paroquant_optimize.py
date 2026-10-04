# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MLX module-scope optimization for ParoQuant."""

import math
import platform
import sys
from functools import lru_cache

import numpy as np
import torch

from .mlx_native import _accurate_matmul_mlx
from .mlx_paroquant_quant import paroquant_quantize_weight_mlx
from .mlx_paroquant_rotation import _rotation_metadata, paroquant_rotate_mlx


@lru_cache(maxsize=1)
def paroquant_mlx_optimization_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _differentiable_rotation_metadata(pairs, *, columns: int, group_size: int):
    partners, pair_indices, sine_signs = _rotation_metadata(
        pairs,
        columns=columns,
        group_size=group_size,
    )
    groups = columns // group_size
    offsets = np.repeat(np.arange(groups, dtype=np.int32) * group_size, group_size)
    pair_offsets = np.repeat(
        np.arange(groups, dtype=np.int32) * (group_size // 2),
        group_size,
    )
    return (
        partners + offsets[None, :],
        pair_indices + pair_offsets[None, :],
        sine_signs,
    )


def _rotate_differentiable(
    value,
    theta,
    metadata,
    *,
    scales=None,
    inverse: bool = False,
):
    import mlx.core as mx

    partners, pair_indices, sine_signs = metadata
    result = value if scales is None or inverse else value * scales.reshape(1, -1)
    stages = range(theta.shape[0] - 1, -1, -1) if inverse else range(theta.shape[0])
    for stage in stages:
        angles = theta[stage].reshape(-1)[pair_indices[stage]]
        if inverse:
            angles = -angles
        result = (
            result * mx.cos(angles)[None, :]
            + result[:, partners[stage]]
            * mx.sin(angles)[None, :]
            * sine_signs[stage][None, :]
        )
    if scales is not None and inverse:
        result = result / scales.reshape(1, -1)
    return result


def _round_ste(value):
    import mlx.core as mx

    return value + mx.stop_gradient(mx.round(value) - value)


def _clip_ste(value, lower, upper):
    import mlx.core as mx

    return value + mx.stop_gradient(mx.clip(value, lower, upper) - value)


def _quantizer_parameters(weight, *, bits: int, group_size: int, symmetric: bool):
    import mlx.core as mx

    grouped = weight.reshape(-1, group_size)
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        scale = mx.maximum(mx.max(mx.abs(grouped), axis=1, keepdims=True), 1e-5) / qmax
        return scale, None
    qmax = (1 << bits) - 1
    minimum = mx.min(grouped, axis=1, keepdims=True)
    maximum = mx.max(grouped, axis=1, keepdims=True)
    scale = mx.maximum(maximum - minimum, 1e-5) / qmax
    return scale, minimum / scale


def _pseudo_quantize(
    weight,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    scale=None,
    zero_point=None,
    ste: bool,
):
    import mlx.core as mx

    grouped = weight.reshape(-1, group_size)
    if scale is None:
        scale, zero_point = _quantizer_parameters(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
    clip = _clip_ste if ste else mx.clip
    rounding = _round_ste if ste else mx.round
    scale = clip(scale, 1e-5, 1e5)
    if symmetric:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        codes = clip(rounding(grouped / scale), qmin, qmax)
        result = codes * scale
    else:
        qmax = (1 << bits) - 1
        rounded_zero = clip(-rounding(zero_point), 0, qmax)
        codes = clip(rounding(grouped / scale) + rounded_zero, 0, qmax)
        result = (codes - rounded_zero) * scale
    return result.reshape(weight.shape)


def _smooth_l1(prediction, target):
    import mlx.core as mx

    difference = mx.abs(prediction - target)
    return mx.mean(mx.where(difference < 1.0, 0.5 * difference.square(), difference - 0.5))


def _forward(
    parameters,
    inputs,
    bias,
    metadata,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    scale_clamp_min: float,
    scale_clamp_max: float,
    initialized_quantizer: bool,
):
    import mlx.core as mx

    channel_scales = _clip_ste(
        parameters["channel_scales"],
        scale_clamp_min,
        scale_clamp_max,
    )
    rotated_inputs = _rotate_differentiable(
        inputs,
        parameters["theta"],
        metadata,
        scales=mx.reciprocal(channel_scales),
    )
    transformed = _rotate_differentiable(
        parameters["weight"] * channel_scales.reshape(1, -1),
        parameters["theta"],
        metadata,
    )
    quantized = _pseudo_quantize(
        transformed,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        scale=parameters.get("quant_scale") if initialized_quantizer else None,
        zero_point=parameters.get("quant_zero") if initialized_quantizer else None,
        ste=True,
    )
    output = _accurate_matmul_mlx(rotated_inputs, quantized.T)
    return output if bias is None else output + bias


def _optimizer_step(
    parameters,
    gradients,
    state,
    *,
    learning_rates,
    optimizer_name: str,
    weight_decay: float,
    betas,
    eps: float,
    amsgrad: bool,
    momentum: float,
    dampening: float,
    nesterov: bool,
):
    import mlx.core as mx

    step = state["step"] + 1
    updated = dict(parameters)
    beta1, beta2 = betas
    for name, gradient in gradients.items():
        parameter = parameters[name]
        learning_rate = learning_rates[name]
        if optimizer_name == "sgd":
            if weight_decay:
                gradient = gradient + weight_decay * parameter
            if momentum:
                if state["step"] == 0:
                    velocity = gradient
                else:
                    velocity = momentum * state["velocity"][name] + (1 - dampening) * gradient
                state["velocity"][name] = velocity
                gradient = gradient + momentum * velocity if nesterov else velocity
            updated[name] = parameter - learning_rate * gradient
            continue

        if optimizer_name == "adam" and weight_decay:
            gradient = gradient + weight_decay * parameter
        first = beta1 * state["first"][name] + (1 - beta1) * gradient
        second = beta2 * state["second"][name] + (1 - beta2) * gradient.square()
        state["first"][name] = first
        state["second"][name] = second
        denominator_second = second
        if amsgrad:
            maximum = mx.maximum(state["maximum"][name], second)
            state["maximum"][name] = maximum
            denominator_second = maximum
        if optimizer_name == "adamw" and weight_decay:
            parameter = parameter * (1 - learning_rate * weight_decay)
        step_size = learning_rate / (1 - beta1**step)
        denominator = mx.sqrt(denominator_second) / math.sqrt(1 - beta2**step) + eps
        updated[name] = parameter - step_size * first / denominator
    state["step"] = step
    return updated


def _new_optimizer_state(parameters, names, *, optimizer_name: str, amsgrad: bool):
    import mlx.core as mx

    state = {"step": 0}
    if optimizer_name == "sgd":
        state["velocity"] = {name: mx.zeros_like(parameters[name]) for name in names}
    else:
        state["first"] = {name: mx.zeros_like(parameters[name]) for name in names}
        state["second"] = {name: mx.zeros_like(parameters[name]) for name in names}
        if amsgrad:
            state["maximum"] = {name: mx.zeros_like(parameters[name]) for name in names}
    return state


def _snapshot(parameters, *, dtype):
    import mlx.core as mx

    result = {name: value.astype(dtype) for name, value in parameters.items()}
    mx.eval(result)
    return result


def _run_stage(
    parameters,
    active_names,
    inputs_train,
    targets_train,
    inputs_val,
    targets_val,
    bias,
    metadata,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    scale_clamp_min: float,
    scale_clamp_max: float,
    initialized_quantizer: bool,
    epochs: int,
    batch_size: int,
    base_learning_rates,
    optimizer_name: str,
    weight_decay: float,
    betas,
    eps: float,
    amsgrad: bool,
    momentum: float,
    dampening: float,
    nesterov: bool,
    snapshot_dtype,
):
    import mlx.core as mx

    def predict(values, inputs):
        merged = dict(parameters)
        merged.update(values)
        return _forward(
            merged,
            inputs,
            bias,
            metadata,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            scale_clamp_min=scale_clamp_min,
            scale_clamp_max=scale_clamp_max,
            initialized_quantizer=initialized_quantizer,
        )

    def loss(values, inputs, targets):
        return _smooth_l1(predict(values, inputs), targets)

    active = {name: parameters[name] for name in active_names}
    def evaluate(values, inputs, targets):
        return float(loss(values, inputs, targets).item())

    if epochs <= 0:
        return parameters, evaluate(active, inputs_train, targets_train), evaluate(active, inputs_val, targets_val)

    value_and_grad = mx.value_and_grad(loss)
    optimizer_state = _new_optimizer_state(
        parameters,
        active_names,
        optimizer_name=optimizer_name,
        amsgrad=amsgrad,
    )
    steps_per_epoch = max(1, math.ceil(inputs_train.shape[0] / batch_size))
    total_steps = max(1, epochs * steps_per_epoch)
    current_learning_rates = dict(base_learning_rates)
    global_step = 0
    best_state = _snapshot(active, dtype=snapshot_dtype)
    best_validation = float("inf")
    last_train = evaluate(active, inputs_train, targets_train)

    for _ in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, inputs_train.shape[0], batch_size):
            stop = min(start + batch_size, inputs_train.shape[0])
            loss_value, gradients = value_and_grad(
                active,
                inputs_train[start:stop],
                targets_train[start:stop],
            )
            active = _optimizer_step(
                active,
                gradients,
                optimizer_state,
                learning_rates=current_learning_rates,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                amsgrad=amsgrad,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
            )
            if "theta" in active:
                active["theta"] = mx.where(
                    parameters["theta_mask"],
                    mx.zeros_like(active["theta"]),
                    active["theta"],
                )
            global_step += 1
            cosine_ratio = 0.5 * (1 + math.cos(math.pi * min(global_step, total_steps) / total_steps))
            current_learning_rates = {
                name: (rate / 20) + ((rate - rate / 20) * cosine_ratio)
                for name, rate in base_learning_rates.items()
            }
            mx.eval(active, optimizer_state)
            epoch_loss += float(loss_value.item())
            batch_count += 1
        last_train = epoch_loss / max(1, batch_count)
        validation = evaluate(active, inputs_val, targets_val)
        if validation < best_validation:
            best_validation = validation
            best_state = _snapshot(active, dtype=snapshot_dtype)

    parameters.update({name: value.astype(mx.float32) for name, value in best_state.items()})
    mx.eval(parameters)
    return parameters, last_train, best_validation


def optimize_paroquant_linear_mlx(
    weight,
    bias,
    inputs,
    pairs,
    theta_mask,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    train_rows: int,
    val_rows: int,
    batch_size: int,
    rotation_epochs: int,
    finetune_epochs: int,
    rotation_lr: float,
    weight_lr: float,
    quantizer_lr: float,
    optimizer_name: str,
    optimizer_weight_decay: float,
    optimizer_betas,
    optimizer_eps: float,
    optimizer_amsgrad: bool,
    sgd_momentum: float,
    sgd_dampening: float,
    sgd_nesterov: bool,
    best_state_dtype: str,
    scale_clamp_min: float,
    scale_clamp_max: float,
):
    """Optimize one ParoQuant linear layer using MLX autodiff."""
    import mlx.core as mx

    weight = mx.array(weight).astype(mx.float32)
    inputs = mx.array(inputs).astype(mx.float32)
    bias = None if bias is None else mx.array(bias).astype(mx.float32)
    if weight.ndim != 2 or inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("weight and inputs must be compatible rank-two arrays")
    out_features, in_features = weight.shape
    if group_size == -1:
        group_size = in_features
    if group_size <= 0 or group_size % 2 or in_features % group_size:
        raise ValueError("group_size must be positive, even, and divide input width")
    if bits < 2 or bits > 8:
        raise ValueError("bits must be between 2 and 8")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError("bias must match the output width")
    if inputs.shape[0] == 0:
        raise ValueError("MLX optimization requires calibration rows")
    if optimizer_name not in {"adamw", "adam", "sgd"}:
        raise ValueError("optimizer_name must be adamw, adam, or sgd")

    pair_values = np.asarray(pairs)
    mask_values = np.asarray(theta_mask, dtype=np.bool_)
    metadata_values = _differentiable_rotation_metadata(
        pair_values,
        columns=in_features,
        group_size=group_size,
    )
    metadata = tuple(mx.array(value) for value in metadata_values)
    theta_shape = (pair_values.shape[0], in_features // 2)
    if mask_values.shape != theta_shape:
        raise ValueError("theta_mask must have shape (rotations, input_features // 2)")
    maximum_rows = max(1, train_rows + val_rows)
    if inputs.shape[0] > maximum_rows:
        indices = mx.round(mx.linspace(0, inputs.shape[0] - 1, maximum_rows)).astype(mx.int32)
        inputs = inputs[indices]
    targets = _accurate_matmul_mlx(inputs, weight.T)
    if bias is not None:
        targets = targets + bias
    train_count = min(inputs.shape[0], max(1, train_rows))
    val_count = min(max(1, val_rows), max(1, inputs.shape[0] - train_count))
    inputs_train = inputs[:train_count]
    targets_train = targets[:train_count]
    inputs_val = inputs[-val_count:]
    targets_val = targets[-val_count:]
    parameters = {
        "weight": weight,
        "theta": mx.zeros(theta_shape, dtype=mx.float32),
        "theta_mask": mx.array(mask_values),
        "channel_scales": mx.ones((in_features,), dtype=mx.float32),
    }
    snapshot_dtype = {
        "fp16": mx.float16,
        "bf16": mx.bfloat16,
        "fp32": mx.float32,
    }[best_state_dtype]
    common = {
        "bias": bias,
        "metadata": metadata,
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "scale_clamp_min": scale_clamp_min,
        "scale_clamp_max": scale_clamp_max,
        "batch_size": batch_size,
        "optimizer_name": optimizer_name,
        "weight_decay": optimizer_weight_decay,
        "betas": optimizer_betas,
        "eps": optimizer_eps,
        "amsgrad": optimizer_amsgrad,
        "momentum": sgd_momentum,
        "dampening": sgd_dampening,
        "nesterov": sgd_nesterov,
        "snapshot_dtype": snapshot_dtype,
    }
    parameters, _, _ = _run_stage(
        parameters,
        ("channel_scales", "theta"),
        inputs_train,
        targets_train,
        inputs_val,
        targets_val,
        initialized_quantizer=False,
        epochs=rotation_epochs,
        base_learning_rates={"channel_scales": rotation_lr, "theta": rotation_lr},
        **common,
    )
    transformed = _rotate_differentiable(
        parameters["weight"]
        * mx.clip(parameters["channel_scales"], scale_clamp_min, scale_clamp_max).reshape(1, -1),
        parameters["theta"],
        metadata,
    )
    quant_scale, quant_zero = _quantizer_parameters(
        transformed,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    parameters["quant_scale"] = quant_scale
    active = ["weight", "quant_scale"]
    learning_rates = {"weight": weight_lr, "quant_scale": quantizer_lr}
    if quant_zero is not None:
        parameters["quant_zero"] = quant_zero
        active.append("quant_zero")
        learning_rates["quant_zero"] = quantizer_lr
    parameters, train_loss, val_loss = _run_stage(
        parameters,
        tuple(active),
        inputs_train,
        targets_train,
        inputs_val,
        targets_val,
        initialized_quantizer=True,
        epochs=finetune_epochs,
        base_learning_rates=learning_rates,
        **common,
    )

    theta = mx.where(
        parameters["theta_mask"],
        mx.zeros_like(parameters["theta"]),
        parameters["theta"],
    )
    bounded_scales = mx.clip(
        parameters["channel_scales"],
        scale_clamp_min,
        scale_clamp_max,
    )
    runtime_channel_scales = mx.reciprocal(bounded_scales).reshape(1, -1)
    transformed = paroquant_rotate_mlx(
        parameters["weight"] * bounded_scales.reshape(1, -1),
        pair_values,
        theta,
        group_size=group_size,
    )
    quantized = paroquant_quantize_weight_mlx(
        transformed,
        parameters["quant_scale"],
        bits=bits,
        group_size=group_size,
        sym=symmetric,
        zero_point_float=parameters.get("quant_zero"),
    )
    pseudo_weight = paroquant_rotate_mlx(
        quantized,
        pair_values,
        theta,
        group_size=group_size,
        inverse=True,
    ) * runtime_channel_scales
    groups = in_features // group_size
    q_scales = parameters["quant_scale"].reshape(out_features, groups)
    if symmetric:
        q_zeros = mx.full(q_scales.shape, 1 << (bits - 1), dtype=mx.float32)
    else:
        q_zeros = mx.clip(-mx.round(parameters["quant_zero"]), 0, (1 << bits) - 1).reshape(
            out_features,
            groups,
        )
    outputs = (
        pseudo_weight,
        quantized,
        q_scales,
        q_zeros,
        theta,
        runtime_channel_scales,
    )
    mx.eval(*outputs)
    return (*outputs, train_loss, val_loss)


def optimize_paroquant_linear_mlx_to_torch(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    pairs: torch.Tensor,
    theta_mask: torch.Tensor,
    **kwargs,
):
    """Bridge processor tensors through MLX and return the Torch result contract."""
    import mlx.core as mx

    from .paroquant.optimization import ParoQuantOptimizationResult

    if weight.device.type == "mps":
        torch.mps.synchronize()
    input_rows = inputs.detach().float().reshape(-1, inputs.shape[-1]).contiguous()
    result = optimize_paroquant_linear_mlx(
        mx.from_dlpack(weight.detach().float().contiguous()),
        None if bias is None else mx.from_dlpack(bias.detach().float().contiguous()),
        mx.from_dlpack(input_rows),
        pairs.detach().cpu().numpy(),
        theta_mask.detach().cpu().numpy(),
        **kwargs,
    )

    def to_torch(value):
        return torch.from_dlpack(value).to(device=weight.device).contiguous()

    pseudo_weight, pack_weight, q_scales, q_zeros, theta, channel_scales = (
        to_torch(value) for value in result[:6]
    )
    return ParoQuantOptimizationResult(
        pseudo_weight=pseudo_weight,
        pack_weight=pack_weight,
        q_scales=q_scales,
        q_zeros=q_zeros,
        pairs=pairs.to(device=weight.device),
        theta=theta,
        channel_scales=channel_scales,
        train_loss=float(result[6]),
        val_loss=float(result[7]),
        used_identity=False,
    )


__all__ = [
    "optimize_paroquant_linear_mlx",
    "optimize_paroquant_linear_mlx_to_torch",
    "paroquant_mlx_optimization_available",
]
