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
def _rotation_kernel():
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
    return _rotation_kernel()(
        inputs=[x, partner, cosine, sine, channel_scales],
        template=[("K", x.shape[-1]), ("FIRST", first)],
        grid=(x.size, 1, 1), threadgroup=(min(x.size, 256), 1, 1),
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
        self.channel_scales = mx.array(channel_scales.astype(np.float16))
        self.partner = tuple(mx.array(row) for row in partner)
        self.cosine = tuple(mx.array(row) for row in cosine)
        self.sine = tuple(mx.array(row) for row in sine)
        self.identity = bool(np.all(theta == 0) and np.all(channel_scales == 1))
        self.freeze()

    def __call__(self, x):
        if not self.identity:
            if x.dtype == mx.float16 and x.size:
                for stage, (partner, cosine, sine) in enumerate(zip(self.partner, self.cosine, self.sine)):
                    x = _rotate_stage(x, partner, cosine, sine, self.channel_scales, stage == 0)
            else:
                x = x * self.channel_scales
                for partner, cosine, sine in zip(self.partner, self.cosine, self.sine):
                    old = x
                    x = old * cosine + mx.take(old, partner, axis=-1) * sine
        return self.linear(x)
