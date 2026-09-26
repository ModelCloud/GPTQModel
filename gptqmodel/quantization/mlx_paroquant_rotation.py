# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation math: Z Lab, MIT, https://github.com/z-lab/paroquant
# MLX runtime license: MIT, https://github.com/ml-explore/mlx

"""Native MLX rotations for ParoQuant's transformed-domain weights."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _rotation_kernel(group_size: int, columns: int, groups: int, krot: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="gptqmodel_paroquant_weight_rotation",
        input_names=[
            "weight",
            "partners",
            "pair_indices",
            "sine_signs",
            "theta",
            "scales",
        ],
        output_names=["rotated"],
        source="""
            threadgroup float current[GROUP_SIZE];
            threadgroup float next_values[GROUP_SIZE];

            uint lane = thread_position_in_threadgroup.x;
            uint row_group = threadgroup_position_in_grid.x;
            uint row = row_group / GROUPS;
            uint group = row_group % GROUPS;
            uint column = group * GROUP_SIZE + lane;
            uint output_index = row * COLUMNS + column;

            current[lane] = weight[output_index];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (!INVERSE && HAS_SCALES) {
                current[lane] *= scales[column];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint iteration = 0; iteration < KROT; ++iteration) {
                uint stage = INVERSE ? KROT - 1 - iteration : iteration;
                uint metadata_index = stage * COLUMNS + column;
                uint partner = partners[metadata_index];
                uint pair_index = pair_indices[metadata_index];
                uint theta_index = stage * (COLUMNS / 2)
                    + group * (GROUP_SIZE / 2) + pair_index;
                float angle = theta[theta_index];
                if (INVERSE) {
                    angle = -angle;
                }
                float cosine = metal::cos(angle);
                float sine = metal::sin(angle)
                    * sine_signs[metadata_index];
                float left = current[lane];
                float right = current[partner];
                next_values[lane] = left * cosine + right * sine;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                current[lane] = next_values[lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (INVERSE && HAS_SCALES) {
                current[lane] /= scales[column];
            }
            rotated[output_index] = current[lane];
        """,
    )


def _rotation_metadata(pairs, *, columns: int, group_size: int):
    """Build a partner lookup from ParoQuant's per-group pair ordering."""
    pair_values = np.asarray(pairs)
    if pair_values.ndim != 2 or pair_values.shape[1] != columns:
        raise ValueError("pairs must have shape (rotations, input_features)")
    if not np.issubdtype(pair_values.dtype, np.integer) or np.issubdtype(
        pair_values.dtype, np.bool_
    ):
        raise TypeError("pairs must contain integer channel indices")

    krot = pair_values.shape[0]
    if krot == 0:
        raise ValueError("pairs must include at least one rotation stage")
    groups = columns // group_size
    half_group = group_size // 2
    pair_groups = pair_values.astype(np.int64, copy=False).reshape(
        krot, groups, group_size
    )
    expected = np.arange(group_size)
    partners = np.empty((krot, columns), dtype=np.int32)
    pair_indices = np.empty((krot, columns), dtype=np.int32)
    sine_signs = np.empty((krot, columns), dtype=np.float32)

    for stage in range(krot):
        for group in range(groups):
            members = pair_groups[stage, group]
            if not np.array_equal(np.sort(members), expected):
                raise ValueError(
                    "each ParoQuant group must contain every local channel index once"
                )
            for pair_index, (left, right) in enumerate(members.reshape(half_group, 2)):
                left_column = group * group_size + int(left)
                right_column = group * group_size + int(right)
                partners[stage, left_column] = int(right)
                partners[stage, right_column] = int(left)
                pair_indices[stage, left_column] = pair_index
                pair_indices[stage, right_column] = pair_index
                sine_signs[stage, left_column] = 1.0
                sine_signs[stage, right_column] = -1.0

    return partners, pair_indices, sine_signs


def paroquant_rotate_mlx(
    weight,
    pairs,
    theta,
    *,
    group_size: int = 128,
    channel_scales=None,
    inverse: bool = False,
):
    """Apply ParoQuant's grouped Givens rotations to a weight matrix on MLX.

    ``weight`` has shape ``(..., input_features)`` and may be float16,
    bfloat16, or float32. ``pairs`` is integer metadata shaped
    ``(rotations, input_features)``; each consecutive pair of entries names
    two distinct local channels within its group. ``theta`` is float16,
    bfloat16, or float32 with shape ``(rotations, input_features // 2)``.
    Optional ``channel_scales`` is a positive floating vector with one value
    per input feature. The forward transform multiplies by these scales before
    rotation; ``inverse=True`` applies the reverse rotations and divides by the
    scales afterward. The result is always float32 for stable quantization
    arithmetic.

    This standalone primitive does not update ParoQuant's autograd optimizer
    or replace its Torch processor.
    """
    import mlx.core as mx

    weight = mx.array(weight)
    theta = mx.array(theta)
    if weight.ndim < 2 or any(dim == 0 for dim in weight.shape):
        raise ValueError("weight must have rank at least two and nonempty dimensions")
    if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("weight must have float16, bfloat16, or float32 dtype")
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise TypeError("group_size must be an integer")
    if group_size not in (16, 32, 64, 128):
        raise ValueError("group_size must be one of 16, 32, 64, or 128")
    if weight.shape[-1] % group_size:
        raise ValueError("input_features must be divisible by group_size")
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be bool")

    columns = weight.shape[-1]
    pair_values = np.asarray(pairs)
    if pair_values.ndim != 2 or pair_values.shape[1] != columns:
        raise ValueError("pairs must have shape (rotations, input_features)")
    krot = pair_values.shape[0]
    if theta.shape != (krot, columns // 2):
        raise ValueError("theta must have shape (rotations, input_features // 2)")
    if theta.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("theta must have float16, bfloat16, or float32 dtype")
    if not np.issubdtype(pair_values.dtype, np.integer) or np.issubdtype(
        pair_values.dtype, np.bool_
    ):
        raise TypeError("pairs must contain integer channel indices")

    scales = None
    if channel_scales is not None:
        scales = mx.array(channel_scales)
        if scales.shape not in ((columns,), (1, columns)):
            raise ValueError("channel_scales must have shape (input_features,)")
        if scales.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("channel_scales must have a floating dtype")
        scales = scales.astype(mx.float32).reshape((columns,))
        valid_scales = mx.all(mx.isfinite(scales) & (scales > 0))
        mx.eval(valid_scales)
        if not bool(valid_scales.item()):
            raise ValueError("channel_scales must be finite and positive")
    else:
        scales = mx.ones((columns,), dtype=mx.float32)

    theta = theta.astype(mx.float32)
    valid_weight = mx.all(mx.isfinite(weight))
    valid_theta = mx.all(mx.isfinite(theta))
    mx.eval(valid_weight, valid_theta)
    if not bool(valid_weight.item()):
        raise ValueError("weight must be finite")
    if not bool(valid_theta.item()):
        raise ValueError("theta must be finite")

    partners, pair_indices, sine_signs = _rotation_metadata(
        pair_values, columns=columns, group_size=group_size
    )
    groups = columns // group_size
    rows = weight.size // columns
    kernel = _rotation_kernel(group_size, columns, groups, krot)
    result = kernel(
        inputs=[
            mx.contiguous(weight.astype(mx.float32).reshape(rows, columns)),
            mx.array(partners.reshape(-1)),
            mx.array(pair_indices.reshape(-1)),
            mx.array(sine_signs.reshape(-1)),
            mx.contiguous(theta.reshape(-1)),
            scales,
        ],
        template=[
            ("GROUP_SIZE", group_size),
            ("COLUMNS", columns),
            ("GROUPS", groups),
            ("KROT", krot),
            ("INVERSE", inverse),
            ("HAS_SCALES", channel_scales is not None),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(group_size, 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(result)
    return result


__all__ = ["paroquant_rotate_mlx"]
