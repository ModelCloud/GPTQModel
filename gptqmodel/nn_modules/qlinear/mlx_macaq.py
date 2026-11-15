# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MACaQ reference: Huawei CSL contributors, Apache-2.0, https://github.com/huawei-csl/MACaQ
"""Packed MACaQ MXFP4 (W4A4/W4A16) inference for MLX Metal.

MACaQ stores OCP MXFP4 exactly as MLX's ``mxfp4`` quantization mode does: one
E2M1 nibble per value (low nibble first, ``sign << 3 | magnitude``) and one
E8M0 scale byte per 32 values. The checkpoint bytes therefore load straight
into MLX's native MXFP4 matmul, so no repack and no dense expansion happens at
runtime; only the 32-wide block Hadamard rotation and the MACaQ diagonal
pre-scale run in the forward, on the rotated-frame activations the quantizer
already assumed.
"""

import mlx.core as mx
import mlx.nn as nn


_MXFP4_GROUP = 32
# Row count where MLX's fused activation quantization stops winning.
_W4A4_FUSED_ROWS = 16


def _rotate_blocks(x, bh):
    """Right-multiply the last axis by the block-diagonal Hadamard matrix."""

    block = bh.shape[0]
    lead = x.shape[:-1]
    width = x.shape[-1]
    if width % block:
        raise ValueError(f"MACaQ: last dimension {width} is not divisible by the Hadamard block {block}")
    return (x.reshape(*lead, width // block, block) @ bh).reshape(*lead, width)


class MlxMACaQLinear(nn.Module):
    """Apply the MACaQ pre-scale and block rotation, then an MXFP4 matmul."""

    def __init__(
        self,
        *,
        in_features,
        out_features,
        had_block=32,
        activation_precision="w4a4",
        compute_dtype=mx.bfloat16,
        bias=False,
    ):
        super().__init__()
        in_features = int(in_features)
        out_features = int(out_features)
        had_block = int(had_block)
        if in_features % _MXFP4_GROUP or in_features % had_block:
            raise ValueError("MACaQ MLX inference needs in_features divisible by the MXFP4 group")
        if activation_precision not in ("w4a4", "w4a16"):
            raise ValueError("MACaQ MLX inference needs `w4a4` or `w4a16` activation precision")
        self.in_features = in_features
        self.out_features = out_features
        self.had_block = had_block
        self.activation_precision = activation_precision
        self.compute_dtype = compute_dtype
        # Checkpoint bytes: [out, in / 8] uint32 nibbles and one E8M0 per group.
        self.weight = mx.zeros((out_features, in_features // 8), dtype=mx.uint32)
        self.scales = mx.zeros((out_features, in_features // _MXFP4_GROUP), dtype=mx.uint8)
        self.pre_scale = mx.ones((in_features,), dtype=compute_dtype)
        self.BH = mx.zeros((had_block, had_block), dtype=compute_dtype)
        # A zero bias keeps the epilogue uniform for biased and unbiased layers.
        self.bias = mx.zeros((out_features,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        rows = x.size // self.in_features
        work = x.reshape(rows, self.in_features).astype(self.compute_dtype)
        work = work * self.pre_scale
        work = _rotate_blocks(work, self.BH)
        if self.activation_precision == "w4a4":
            # One fused dispatch quantizes the rotated activation to MXFP4 and
            # multiplies it against the packed weight. MLX's fused path loses
            # to an explicit quantize/dequantize pair once the token block is
            # wide enough for the extra passes to amortize (measured on this
            # host: equal through 16 rows, 5x slower at 512).
            if rows <= _W4A4_FUSED_ROWS:
                output = mx.qqmm(
                    work, self.weight, self.scales,
                    group_size=_MXFP4_GROUP, bits=4, mode="mxfp4",
                )
            else:
                quantized, scales = mx.quantize(
                    work, group_size=_MXFP4_GROUP, bits=4, mode="mxfp4",
                )
                work = mx.dequantize(
                    quantized, scales, group_size=_MXFP4_GROUP, bits=4, mode="mxfp4",
                )
                output = mx.quantized_matmul(
                    work, self.weight, self.scales,
                    transpose=True, group_size=_MXFP4_GROUP, bits=4, mode="mxfp4",
                )
        else:
            output = mx.quantized_matmul(
                work, self.weight, self.scales,
                transpose=True, group_size=_MXFP4_GROUP, bits=4, mode="mxfp4",
            )
        return (output + self.bias).astype(x.dtype).reshape(output_shape)


__all__ = ["MlxMACaQLinear"]
