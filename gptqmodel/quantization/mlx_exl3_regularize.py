# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Native MLX fused channel scaling and Hadamard regularization for EXL3."""

import math
from functools import lru_cache

from .mlx_exl3_rms import exl3_block_rms_mlx

_EXL3_ZERO_THRESHOLD = 1e-30
_EXL3_OUTPUT_SKEW_THRESHOLD = 0.15


@lru_cache(maxsize=1)
def _exl3_regularize_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_regularize",
        input_names=["weight", "signs", "rms", "mean"],
        output_names=["transformed", "channel_scales"],
        source="""
            threadgroup float current[128];

            uint lane = thread_position_in_threadgroup.x;
            uint vector = threadgroup_position_in_grid.x;
            uint row;
            uint column;
            if (INPUT_MODE) {
                row = (vector / COLUMNS) * 128u + lane;
                column = vector % COLUMNS;
            } else {
                row = vector / COLUMN_BLOCKS;
                column = (vector % COLUMN_BLOCKS) * 128u + lane;
            }

            uint channel = INPUT_MODE ? row : column;
            float root_mean_square = rms[channel];
            float scale;
            bool zero_channel;
            if (INPUT_MODE) {
                zero_channel = metal::abs(root_mean_square) < 1.0e-30f;
                float effective_rms = zero_channel ? 0.1f : root_mean_square;
                scale = signs[channel] * effective_rms / -1.24371088f + 1.0e-10f;
            } else {
                float normalized_rms = HAS_MEAN
                    ? root_mean_square / mean[0]
                    : root_mean_square;
                zero_channel = metal::abs(normalized_rms) < 1.0e-30f;
                if (APPLY_OUTPUT_SCALES) {
                    float effective_rms = zero_channel ? 0.1f : normalized_rms;
                    scale = signs[channel] * effective_rms + 1.0e-10f;
                } else {
                    scale = signs[channel];
                }
            }

            uint element = row * COLUMNS + column;
            current[lane] = weight[element] / scale;
            if ((!INPUT_MODE && row == 0u) || (INPUT_MODE && column == 0u)) {
                channel_scales[channel] = (!INPUT_MODE && zero_channel)
                    ? 0.0f
                    : scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total = 0.0f;
            for (uint source = 0u; source < 128u; ++source) {
                uint bits = lane & source;
                bits ^= bits >> 4u;
                bits ^= bits >> 2u;
                bits ^= bits >> 1u;
                float coefficient = (bits & 1u)
                    ? -0.08838834764831845f
                    : 0.08838834764831845f;
                total = metal::fma(current[source], coefficient, total);
            }
            transformed[element] = total;
        """,
    )


def _validate_inputs(weight, signs, rms, *, axis: int):
    import mlx.core as mx

    weight = mx.array(weight)
    signs = mx.array(signs)
    rms = mx.array(rms)
    if weight.ndim != 2 or any(dimension == 0 for dimension in weight.shape):
        raise ValueError("weight must be a nonempty rank-two array")
    if weight.dtype != mx.float32:
        raise ValueError("weight must have float32 dtype")
    if weight.shape[axis] % 128:
        raise ValueError("the transformed dimension must be divisible by 128")
    expected_shape = (weight.shape[0], 1) if axis == 0 else (1, weight.shape[1])
    if signs.shape != expected_shape or rms.shape != expected_shape:
        raise ValueError(f"signs and rms must have shape {expected_shape}")
    if signs.dtype != mx.float32 or rms.dtype != mx.float32:
        raise ValueError("signs and rms must have float32 dtype")
    return weight, signs, rms


def _regularize(weight, signs, rms, *, input_mode: bool, mean, has_mean, apply_scales):
    import mlx.core as mx

    rows, columns = weight.shape
    column_blocks = max(1, columns // 128)
    vector_count = (rows // 128) * columns if input_mode else rows * column_blocks
    outputs = _exl3_regularize_kernel()(
        inputs=[mx.contiguous(weight), mx.contiguous(signs), mx.contiguous(rms), mean],
        template=[
            ("INPUT_MODE", input_mode),
            ("APPLY_OUTPUT_SCALES", apply_scales),
            ("HAS_MEAN", has_mean),
            ("COLUMNS", columns),
            ("COLUMN_BLOCKS", column_blocks),
        ],
        grid=(vector_count * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[weight.shape, signs.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    mx.eval(*outputs)
    return tuple(outputs)


def exl3_output_regularize_mlx(
    weight,
    signs,
    rms,
    *,
    mean: float,
    apply_scales: bool,
):
    """Apply output-channel scales and the right EXL3 Hadamard transform.

    Inputs are expected to come from EXL3's validated float32 RMS/Hadamard and
    random-sign stages. Returns ``(transformed_weight, stored_scales, all_zero)``.
    """
    import mlx.core as mx

    weight, signs, rms = _validate_inputs(weight, signs, rms, axis=1)
    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
        raise TypeError("mean must be a finite nonnegative number")
    mean = float(mean)
    if not math.isfinite(mean) or mean < 0:
        raise ValueError("mean must be a finite nonnegative number")
    if not isinstance(apply_scales, bool):
        raise TypeError("apply_scales must be a boolean")

    transformed, stored_scales = _regularize(
        weight,
        signs,
        rms,
        input_mode=False,
        mean=mx.array([mean], dtype=mx.float32),
        has_mean=mean > _EXL3_ZERO_THRESHOLD,
        apply_scales=apply_scales,
    )
    return transformed, stored_scales, mean <= _EXL3_ZERO_THRESHOLD


def exl3_input_regularize_mlx(weight, signs, rms):
    """Apply input-channel/codebook scales and the left EXL3 Hadamard transform."""
    import mlx.core as mx

    weight, signs, rms = _validate_inputs(weight, signs, rms, axis=0)
    return _regularize(
        weight,
        signs,
        rms,
        input_mode=True,
        mean=mx.array([1.0], dtype=mx.float32),
        has_mean=False,
        apply_scales=False,
    )


def exl3_regularize_transforms_mlx(
    weight,
    input_signs,
    output_signs,
    *,
    force_output_scales=None,
    hessian_diagonal=None,
    fallback: bool = False,
):
    """Apply EXL3's complete two-sided regularization transform on MLX.

    This composes the native block-RMS reductions, channel scaling, and
    Hadamard kernels while keeping the weight resident in MLX. It intentionally
    stops before EXL3's separately exposed global-scale search.

    Returns ``(apply_output_scales, transformed_weight, input_scales,
    output_scales)``. ``input_signs`` and ``output_signs`` use shapes
    ``(input_features, 1)`` and ``(1, output_features)`` respectively.
    """
    import mlx.core as mx

    weight = mx.array(weight)
    input_signs = mx.array(input_signs)
    output_signs = mx.array(output_signs)
    if force_output_scales is not None and not isinstance(force_output_scales, bool):
        raise TypeError("force_output_scales must be a boolean or None")
    if not isinstance(fallback, bool):
        raise TypeError("fallback must be a boolean")

    if hessian_diagonal is not None:
        hessian_diagonal = mx.array(hessian_diagonal)
        expected_shape = (weight.shape[0],) if weight.ndim == 2 else None
        if hessian_diagonal.shape != expected_shape:
            raise ValueError(
                "hessian_diagonal must have one value per input feature"
            )
        if hessian_diagonal.dtype != mx.float32:
            raise ValueError("hessian_diagonal must have float32 dtype")
        valid_diagonal = mx.all(
            mx.isfinite(hessian_diagonal) & (hessian_diagonal >= 0)
        )
        mx.eval(valid_diagonal)
        if not bool(valid_diagonal.item()):
            raise ValueError(
                "hessian_diagonal must contain finite nonnegative values"
            )

    if not fallback and hessian_diagonal is not None:
        diagonal = mx.sort(mx.sqrt(hessian_diagonal))[::-1]
        cutoff = diagonal.shape[0] // 50
        skew = mx.sum(diagonal[:cutoff]) / mx.sum(diagonal)
        mx.eval(skew)
        apply_output_scales = (
            float(skew.item()) < _EXL3_OUTPUT_SKEW_THRESHOLD
            if force_output_scales is None
            else force_output_scales
        )
    else:
        apply_output_scales = (
            True if force_output_scales is None else force_output_scales
        )
    if fallback:
        apply_output_scales = force_output_scales

    output_rms = exl3_block_rms_mlx(weight, axis=0)
    output_mean_array = mx.mean(output_rms)
    mx.eval(output_mean_array)
    output_mean = float(output_mean_array.item())
    if output_mean <= _EXL3_ZERO_THRESHOLD and force_output_scales is not None:
        apply_output_scales = True

    transformed, output_scales, _ = exl3_output_regularize_mlx(
        weight,
        output_signs,
        output_rms,
        mean=output_mean,
        apply_scales=bool(apply_output_scales),
    )
    input_rms = exl3_block_rms_mlx(transformed, axis=1)
    transformed, input_scales = exl3_input_regularize_mlx(
        transformed,
        input_signs,
        input_rms,
    )
    return apply_output_scales, transformed, input_scales, output_scales


__all__ = [
    "exl3_input_regularize_mlx",
    "exl3_output_regularize_mlx",
    "exl3_regularize_transforms_mlx",
]
