# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation reference: Z Lab, MIT, https://github.com/z-lab/paroquant
# MLX array operations: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""ParoQuant rotations followed by a packed MLX affine matrix product."""

from functools import lru_cache

import numpy as np
import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=1)
def _rotation_stage_kernel():
    """Apply one pairwise rotation stage in one Metal dispatch."""
    return mx.fast.metal_kernel(
        name="gptqmodel_paro_rotation",
        input_names=["x", "partner", "cosine", "sine", "channel_scales"],
        output_names=["rotated"],
        source="""
            uint index = thread_position_in_grid.x;
            uint column = index % K;
            uint paired = partner[column];
            uint row_offset = index - column;
            float left = float(x[index]);
            float right = float(x[row_offset + paired]);
            if (FIRST) {
                left *= float(channel_scales[column]);
                right *= float(channel_scales[paired]);
            }
            rotated[index] = left * cosine[column] + right * sine[column];
        """,
    )


def _rotate_stage(x, partner, cosine, sine, channel_scales, first):
    # Preserve float32 values between stages; repeated FP16 rounding was the
    # dominant error against the independent Torch rotation oracle.
    return _rotation_stage_kernel()(
        inputs=[x, partner, cosine, sine, channel_scales],
        template=[("K", x.shape[-1]), ("FIRST", first)],
        grid=(x.size, 1, 1), threadgroup=(min(x.size, 256), 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _fused_rotation_kernel():
    """Apply every pairwise rotation stage inside one Metal dispatch."""
    return mx.fast.metal_kernel(
        name="gptqmodel_paro_fused_rotation",
        input_names=["x", "partner", "cosine", "sine", "channel_scales"],
        output_names=["rotated"],
        source="""
            threadgroup float current[GROUP_SIZE];
            threadgroup float next_values[GROUP_SIZE];

            uint lane = thread_position_in_threadgroup.x;
            uint row_group = threadgroup_position_in_grid.x;
            uint row = row_group / GROUPS;
            uint group = row_group % GROUPS;
            uint column = group * GROUP_SIZE + lane;
            uint index = row * K + column;

            current[lane] = float(x[index]) * float(channel_scales[column]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stage = 0; stage < KROT; ++stage) {
                uint metadata_index = stage * K + column;
                next_values[lane] = metal::fma(
                    current[partner[metadata_index]],
                    sine[metadata_index],
                    current[lane] * cosine[metadata_index]
                );
                threadgroup_barrier(mem_flags::mem_threadgroup);
                current[lane] = next_values[lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            rotated[index] = current[lane];
        """,
    )


def _rotate_fused(x, partner, cosine, sine, channel_scales, group_size, krot):
    rows = x.size // x.shape[-1]
    groups = x.shape[-1] // group_size
    return _fused_rotation_kernel()(
        inputs=[x, partner, cosine, sine, channel_scales],
        template=[("K", x.shape[-1]), ("GROUP_SIZE", group_size),
                  ("GROUPS", groups), ("KROT", krot)],
        grid=(rows * groups * group_size, 1, 1),
        threadgroup=(group_size, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )[0]


class MlxParoLinear(nn.Module):
    """Apply the checkpoint's learned pairwise rotations before packed matmul."""

    def __init__(self, linear, pairs, theta, channel_scales, group_size):
        super().__init__()
        self.linear = linear
        krot, input_dims = pairs.shape
        groups = input_dims // group_size
        pair_rows = pairs.astype(np.int64).reshape(krot, groups, group_size // 2, 2)
        angle_rows = theta.astype(np.float32).reshape(krot, groups, group_size // 2)
        offsets = (np.arange(groups) * group_size)[None, :, None]
        first = (pair_rows[..., 0] + offsets).reshape(krot, -1)
        second = (pair_rows[..., 1] + offsets).reshape(krot, -1)
        partner = np.empty((krot, input_dims), dtype=np.int32)
        cosine = np.empty((krot, input_dims), dtype=np.float32)
        sine = np.empty((krot, input_dims), dtype=np.float32)
        for stage in range(krot):
            c = np.cos(angle_rows[stage]).reshape(-1)
            s = np.sin(angle_rows[stage]).reshape(-1)
            partner[stage, first[stage]] = second[stage]
            partner[stage, second[stage]] = first[stage]
            cosine[stage, first[stage]] = c
            cosine[stage, second[stage]] = c
            sine[stage, first[stage]] = s
            sine[stage, second[stage]] = -s
        self.group_size = int(group_size)
        self.krot = int(krot)
        self.channel_scales = mx.array(channel_scales.astype(np.float16))
        self.partner = tuple(mx.array(row) for row in partner)
        self.cosine = tuple(mx.array(row) for row in cosine)
        self.sine = tuple(mx.array(row) for row in sine)
        if self.group_size <= 128:
            local_partner = partner % self.group_size
            # Tuples keep immutable derived metadata out of MLX's loadable
            # parameter tree, as with the per-stage metadata above.
            self.fused_partner = (mx.array(local_partner.reshape(-1)),)
            self.fused_cosine = (mx.array(cosine.reshape(-1)),)
            self.fused_sine = (mx.array(sine.reshape(-1)),)
        self.identity = bool(np.all(theta == 0) and np.all(channel_scales == 1))
        self.freeze()

    def _forward_unrounded(self, x):
        if not self.identity:
            if (x.size and self.group_size <= 128
                    and (x.dtype == mx.bfloat16 or self.krot > 1)):
                x = _rotate_fused(
                    x, self.fused_partner[0], self.fused_cosine[0], self.fused_sine[0],
                    self.channel_scales, self.group_size, self.krot,
                )
            elif x.dtype == mx.float16 and x.size:
                for stage, (partner, cosine, sine) in enumerate(zip(self.partner, self.cosine, self.sine)):
                    x = _rotate_stage(x, partner, cosine, sine, self.channel_scales, stage == 0)
            else:
                x = x * self.channel_scales
                for partner, cosine, sine in zip(self.partner, self.cosine, self.sine):
                    old = x
                    x = old * cosine + mx.take(old, partner, axis=-1) * sine
        return self.linear(x)

    def __call__(self, x):
        # Rotation and the affine matmul use FP32 intermediates, but a
        # quantized projection keeps the model's activation dtype.
        return self._forward_unrounded(x).astype(x.dtype)
