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
already assumed. A layer quantized without the rotation and without the
pre-scale skips that pass and runs MLX's MXFP4 matmul directly.
"""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


_MXFP4_GROUP = 32
# Row count where MLX's fused activation quantization stops winning.
_W4A4_FUSED_ROWS = 16
_PRE_ROTATE_THREADS = 128


@lru_cache(maxsize=3)
def _pre_rotate_kernel(output_dtype):
    """Apply the diagonal pre-scale and the block Hadamard in one dispatch.

    The block rotation is a per-token ``[K / block, block] @ [block, block]``
    product. A multiply plus a batched matmul needs two dispatches and an
    intermediate activation buffer; at the token counts that dominate decode and
    small-batch prefill (4..16) that overhead is a visible share of the layer, so
    the scale and the rotation are folded into one pass over the activation.
    """

    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported MACaQ activation dtype: {output_dtype}") from exc
    return mx.fast.metal_kernel(
        name=f"gptqmodel_macaq_pre_rotate_{output_suffix}",
        input_names=["x", "pre_scale", "bh"],
        output_names=["output"],
        source="""
            uint column = thread_position_in_grid.x;
            uint row = thread_position_in_grid.y;
            uint block = column / BLOCK;
            uint offset = column % BLOCK;
            uint base = block * BLOCK;
            float sum = 0.0f;
            for (uint index = 0; index < BLOCK; ++index) {
                sum = metal::fma(
                    float(x[row * K + base + index]) * float(pre_scale[base + index]),
                    float(bh[index * BLOCK + offset]),
                    sum
                );
            }
            output[row * K + column] = OUTPUT_TYPE(sum);
        """.replace("OUTPUT_TYPE", output_type),
    )


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
        apply_rotation=True,
        apply_pre_scale=True,
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
        # The loaded BH / pre_scale are identities when these are False, so
        # skipping them is exact.
        self.apply_rotation = bool(apply_rotation)
        self.apply_pre_scale = bool(apply_pre_scale)
        # Checkpoint bytes: [out, in / 8] uint32 nibbles and one E8M0 per group.
        self.weight = mx.zeros((out_features, in_features // 8), dtype=mx.uint32)
        self.scales = mx.zeros((out_features, in_features // _MXFP4_GROUP), dtype=mx.uint8)
        self.pre_scale = mx.ones((in_features,), dtype=compute_dtype)
        self.BH = mx.zeros((had_block, had_block), dtype=compute_dtype)
        # A zero bias keeps the epilogue uniform for biased and unbiased layers.
        self.bias = mx.zeros((out_features,), dtype=mx.float32)
        self.freeze()

    def _rotate(self, work):
        """Pre-scale and rotate a ``[rows, in_features]`` activation block."""
        if not self.apply_rotation:
            return work * self.pre_scale if self.apply_pre_scale else work
        return _pre_rotate_kernel(self.compute_dtype)(
            inputs=[work, self.pre_scale, self.BH],
            template=[("K", self.in_features), ("BLOCK", self.had_block)],
            grid=(self.in_features, work.shape[0], 1),
            threadgroup=(_PRE_ROTATE_THREADS, 1, 1),
            output_shapes=[(work.shape[0], self.in_features)],
            output_dtypes=[self.compute_dtype],
        )[0]

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        rows = x.size // self.in_features
        work = self._rotate(x.reshape(rows, self.in_features).astype(self.compute_dtype))
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
