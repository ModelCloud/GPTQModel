# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX-LM supplies model loading, native AWQ quantization, and checkpoint saving.

"""Native MLX GPTQ/AWQ quantization for Apple silicon.

The GPTQ group update is fused into one Metal dispatch per weight group. MLX
still performs the Hessian Cholesky factorization on its CPU stream.
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _gptq_group_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_group_update",
        input_names=["weights", "hinv", "scales", "biases"],
        output_names=["packed", "errors"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
            }
            float scale = scales[row];
            float bias = biases[row];
            uint word = 0;
            for (int k = 0; k < G; ++k) {
                float code = scale == 0.0f ? 0.0f :
                    metal::clamp(metal::rint((values[k] - bias) / scale),
                                 0.0f, float((1 << BITS) - 1));
                float quantized = code * scale + bias;
                float error = (values[k] - quantized) / hinv[k * G + k];
                word |= uint(code) << ((k % PACK) * BITS);
                if (k % PACK == PACK - 1) {
                    packed[row * (G / PACK) + k / PACK] = word;
                    word = 0;
                }
                errors[row * G + k] = error;
                for (int j = k + 1; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j];
                }
            }
        """,
    )


def inverse_hessian_mlx(hessian, damp_percent: float = 0.01):
    """Return the upper Cholesky factor of the damped inverse Hessian."""
    import mlx.core as mx

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix")
    if not 0 < damp_percent < 1:
        raise ValueError("damp_percent must be between 0 and 1")
    with mx.stream(mx.cpu):
        h = hessian.astype(mx.float32)
        diagonal = mx.diag(h)
        damp = mx.maximum(
            damp_percent * mx.mean(diagonal),
            mx.maximum(mx.max(diagonal) * 1e-6, 1e-12),
        )
        h = h + mx.eye(h.shape[0], dtype=mx.float32) * damp
        factor = mx.linalg.cholesky(h)
        inverse = mx.linalg.cholesky_inv(factor)
        result = mx.linalg.cholesky(inverse, upper=True)
        mx.eval(result)
    return result


def _hessian_partial_mlx(activations):
    """Accumulate an accurate float32 Hessian with MLX GPU matmuls."""
    import mlx.core as mx

    flat = activations.reshape(-1, activations.shape[-1]).astype(mx.float32)
    # Splitting into BF16 high and residual parts avoids the reduced input
    # precision of a single GPU float32 matmul while staying on the GPU.
    high = flat.astype(mx.bfloat16).astype(mx.float32)
    low = flat - high
    cross = high.T @ low
    return high.T @ high + cross + cross.T + low.T @ low


def _accurate_matmul_mlx(left, right):
    """Multiply float32 arrays on the GPU with BF16 residual correction."""
    import mlx.core as mx

    left_high = left.astype(mx.bfloat16).astype(mx.float32)
    right_high = right.astype(mx.bfloat16).astype(mx.float32)
    left_low = left - left_high
    right_low = right - right_high
    return (
        left_high @ right_high
        + left_high @ right_low
        + left_low @ right_high
        + left_low @ right_low
    )


def gptq_quantize_weight_mlx(
    weight, inverse_hessian, bits: int = 4, group_size: int = 64
):
    """Quantize one [output, input] weight matrix with fused GPTQ updates.

    Returns MLX affine ``(packed_weight, scales, biases)`` tensors.
    """
    import mlx.core as mx

    if bits not in (2, 4, 8):
        raise ValueError("GPTQ MLX supports 2, 4, or 8 bits")
    if group_size not in (32, 64, 128):
        raise ValueError("MLX affine quantization requires group_size 32, 64, or 128")
    if weight.ndim not in (2, 3) or weight.shape[-1] % group_size:
        raise ValueError(
            "weight must be 2D or 3D with input width divisible by group_size"
        )
    columns = weight.shape[-1]
    prefix = weight.shape[:-1]
    rows = weight.size // columns
    if rows == 0:
        raise ValueError("weight must have at least one output row")
    if inverse_hessian.shape != (columns, columns):
        raise ValueError("inverse_hessian shape must match the input width")

    kernel = _gptq_group_kernel()
    remaining = weight.reshape(rows, columns).astype(mx.float32)
    packed_groups, all_scales, all_biases = [], [], []
    values_per_word = 32 // bits
    for start in range(0, columns, group_size):
        end = start + group_size
        group = remaining[:, :group_size]
        _, scales, biases = mx.quantize(group, bits=bits, group_size=group_size)
        packed_group, errors = kernel(
            inputs=[group, inverse_hessian[start:end, start:end], scales, biases],
            template=[("G", group_size), ("BITS", bits), ("PACK", values_per_word)],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[(rows, group_size // values_per_word), (rows, group_size)],
            output_dtypes=[mx.uint32, mx.float32],
        )
        packed_groups.append(packed_group)
        all_scales.append(scales)
        all_biases.append(biases)
        if end < columns:
            remaining = (
                remaining[:, group_size:]
                - _accurate_matmul_mlx(errors, inverse_hessian[start:end, end:])
            )
            mx.eval(remaining)

    packed = mx.concatenate(packed_groups, axis=1).reshape(
        *prefix, columns // values_per_word
    )
    scales = mx.concatenate(all_scales, axis=1).reshape(*prefix, columns // group_size)
    biases = mx.concatenate(all_biases, axis=1).reshape(*prefix, columns // group_size)
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def _effective_group_size(width: int, requested: int):
    return next(
        (size for size in (128, 64, 32) if size <= requested and width % size == 0),
        None,
    )


def gptq_quantize_model_mlx(
    model,
    data,
    bits: int = 4,
    group_size: int = 64,
    fallback_bits: int = 6,
    fallback_group_size: int = 64,
    batch_size: int = 8,
    damp_percent: float = 0.01,
):
    """Quantize an MLX-LM model in place, returning its quantization config."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.models.switch_layers import SwitchLinear

    if bits not in (2, 4, 8):
        raise ValueError("GPTQ MLX supports 2, 4, or 8 bits")
    if group_size not in (32, 64, 128):
        raise ValueError("MLX affine quantization requires group_size 32, 64, or 128")
    if fallback_group_size not in (32, 64, 128):
        raise ValueError("fallback_group_size must be 32, 64, or 128")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.H = None

        def __call__(self, x, *args, **kwargs):
            partial = _hessian_partial_mlx(x)
            self.H = partial if self.H is None else self.H + partial
            return self.module(x, *args, **kwargs)

    layers = [
        (key, Catcher(layer))
        for key, layer in tree_flatten(
            model.leaf_modules(), is_leaf=nn.Module.is_module
        )
        if type(layer) in (nn.Linear, SwitchLinear)
        and _effective_group_size(layer.weight.shape[-1], group_size) is not None
    ]
    model.update_modules(tree_unflatten(layers))
    for start in range(0, len(data), batch_size):
        model(data[start : start + batch_size])
        mx.eval([layer.H for _, layer in layers if layer.H is not None])

    quantized = []
    for key, catcher in layers:
        if catcher.H is None:
            raise ValueError(f"No calibration data reached MLX layer {key}")
        hinv = inverse_hessian_mlx(catcher.H, damp_percent=damp_percent)
        layer = catcher.module
        orig_dtype = layer.weight.dtype
        layer_group_size = _effective_group_size(layer.weight.shape[-1], group_size)
        packed, scales, biases = gptq_quantize_weight_mlx(
            layer.weight,
            hinv,
            bits=bits,
            group_size=layer_group_size,
        )
        qlayer = layer.to_quantized(bits=bits, group_size=layer_group_size)
        qlayer.weight, qlayer.scales, qlayer.biases = packed, scales, biases
        qlayer.set_dtype(orig_dtype)
        mx.eval(qlayer)
        quantized.append((key, qlayer))
    model.update_modules(tree_unflatten(quantized))

    config = {"bits": bits, "group_size": group_size}
    for key, qlayer in quantized:
        if qlayer.group_size != group_size:
            config[key] = {"bits": bits, "group_size": qlayer.group_size}
    other_layers = []
    for key, layer in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if hasattr(layer, "to_quantized"):
            layer_group_size = _effective_group_size(
                layer.weight.shape[-1], fallback_group_size
            )
            if layer_group_size is None:
                config[key] = False
                continue
            layer_fallback = {"bits": fallback_bits, "group_size": layer_group_size}
            config[key] = layer_fallback
            other_layers.append((key, layer.to_quantized(**layer_fallback)))
    if other_layers:
        model.update_modules(tree_unflatten(other_layers))
    return model, config


def quantize_mlx(
    model_id_or_path: str,
    output_path: str,
    method: str = "gptq",
    bits: int = 4,
    group_size: int = 64,
    num_samples: int = 128,
    sequence_length: int = 512,
    trust_remote_code: bool = False,
    calibration_data=None,
    seed: int = 123,
    **kwargs,
):
    """Load, calibrate, quantize, and save a model using native MLX operations."""
    import mlx.core as mx
    from mlx_lm.quant.utils import load_data
    from mlx_lm.utils import load, save

    method = method.lower()
    if method not in ("gptq", "awq"):
        raise ValueError("method must be 'gptq' or 'awq'")
    mx.random.seed(seed)
    load_kwargs = {"lazy": True, "return_config": True}
    if trust_remote_code:
        from inspect import signature

        if "trust_remote_code" not in signature(load).parameters:
            raise ValueError("The installed MLX-LM does not support trust_remote_code")
        load_kwargs["trust_remote_code"] = True
    model, tokenizer, config = load(model_id_or_path, **load_kwargs)
    calibration = (
        load_data(tokenizer, num_samples, sequence_length)
        if calibration_data is None
        else calibration_data
    )
    if calibration.ndim != 2 or calibration.shape[0] == 0:
        raise ValueError("calibration_data must contain a nonempty 2D token array")
    if method == "gptq":
        model, config["quantization"] = gptq_quantize_model_mlx(
            model,
            calibration,
            bits=bits,
            group_size=group_size,
            **kwargs,
        )
    else:
        from mlx_lm.quant.awq import (
            AWQ_MODEL_CONFIGS,
            awq_quantize,
            dist_split,
            update_config,
        )

        awq_config = AWQ_MODEL_CONFIGS.get(config["model_type"])
        if awq_config is None:
            raise ValueError(
                f"MLX AWQ does not support model type {config['model_type']}"
            )
        calibration = dist_split(calibration, mx.distributed.init())
        awq_quantize(
            model,
            calibration,
            awq_config,
            bits=bits,
            group_size=group_size,
            **kwargs,
        )
        config = update_config(model, config)
    source_path = Path(model_id_or_path)
    if not source_path.is_dir():
        from huggingface_hub import hf_hub_download

        source_path = Path(
            hf_hub_download(model_id_or_path, "config.json", local_files_only=True)
        ).parent
    save(output_path, source_path, model, tokenizer, config)
    return output_path
