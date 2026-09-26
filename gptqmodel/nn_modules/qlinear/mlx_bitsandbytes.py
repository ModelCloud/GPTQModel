# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX Metal kernel API: Apple Inc., MIT, https://github.com/ml-explore/mlx
# bitsandbytes formats: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
"""Packed BitsAndBytes NF4, FP4, and INT8 inference kernels for MLX."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np


_THREADS = 32


def repack_int8_affine(weight, scales, in_features, out_features):
    """Map signed BNB INT8 rows exactly to MLX unsigned affine groups."""
    group_size = next((size for size in (128, 64, 32) if in_features % size == 0), 0)
    if not group_size:
        return None
    signed = np.asarray(weight, dtype=np.int8).reshape(out_features, in_features)
    unsigned = (signed.astype(np.int16) + 128).astype(np.uint8)
    packed = np.ascontiguousarray(unsigned).view(np.uint32).reshape(out_features, -1)
    row_scales = np.asarray(scales, dtype=np.float32).reshape(out_features, 1) / np.float32(127)
    affine_scales = np.repeat(row_scales, in_features // group_size, axis=1)
    affine_biases = affine_scales * np.float32(-128)
    return packed, affine_scales, affine_biases, group_size


@lru_cache(maxsize=1)
def _four_bit_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_4bit_matmul",
        input_names=["x", "weight", "scales", "codebook", "bias"],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            uint row_base = threadgroup_position_in_grid.z * RTILE;
            threadgroup float partials[RTILE * GROUPS];
            float sums[RTILE];
            for (uint r = 0; r < RTILE; ++r) sums[r] = 0.0f;
            uint weight_offset = column * K;
            if (EVEN) {
                for (uint k = lane * 2; k < K; k += THREADS * 2) {
                    uint index = weight_offset + k;
                    uchar packed = weight[index >> 1];
                    float scale = scales[index / BLOCK];
                    float first = float(half(codebook[uint(packed >> 4)] * scale));
                    float second = float(half(codebook[uint(packed & 15)] * scale));
                    for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                        uint input_offset = (row_base + r) * K + k;
                        sums[r] = metal::fma(float(x[input_offset]), first, sums[r]);
                        sums[r] = metal::fma(float(x[input_offset + 1]), second, sums[r]);
                    }
                }
            } else {
                for (uint k = lane; k < K; k += THREADS) {
                    uint index = weight_offset + k;
                    uchar packed = weight[index >> 1];
                    uint code = (index & 1) ? uint(packed & 15) : uint(packed >> 4);
                    float value = float(half(codebook[code] * scales[index / BLOCK]));
                    for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r)
                        sums[r] = metal::fma(float(x[(row_base + r) * K + k]), value, sums[r]);
                }
            }
            for (uint r = 0; r < RTILE; ++r) {
                float reduced = simd_sum(sums[r]);
                if ((lane & 31) == 0) partials[r * GROUPS + (lane >> 5)] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                float reduced = lane < GROUPS ? partials[r * GROUPS + lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0) output[(row_base + r) * N + column] = reduced + bias[column];
            }
        """,
    )


@lru_cache(maxsize=1)
def _int8_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_int8_matmul",
        input_names=["x", "weight", "scales", "bias"],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            uint row_base = threadgroup_position_in_grid.z * RTILE;
            threadgroup float partials[RTILE * GROUPS];
            float sums[RTILE];
            for (uint r = 0; r < RTILE; ++r) sums[r] = 0.0f;
            uint weight_offset = column * K;
            float scale = scales[column] * (1.0f / 127.0f);
            for (uint k = lane; k < K; k += THREADS) {
                float value = float(half(float(weight[weight_offset + k]) * scale));
                for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r)
                    sums[r] = metal::fma(float(x[(row_base + r) * K + k]), value, sums[r]);
            }
            for (uint r = 0; r < RTILE; ++r) {
                float reduced = simd_sum(sums[r]);
                if ((lane & 31) == 0) partials[r * GROUPS + (lane >> 5)] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                float reduced = lane < GROUPS ? partials[r * GROUPS + lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0) output[(row_base + r) * N + column] = reduced + bias[column];
            }
        """,
    )


class MlxBitsAndBytesLinear(nn.Module):
    """Multiply activations by packed BitsAndBytes weights without dense expansion."""

    def __init__(
        self,
        weight,
        scales,
        *,
        in_features,
        out_features,
        bits,
        block_size=64,
        codebook=None,
        affine_biases=None,
        bias=None,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("BitsAndBytes MLX inference supports 4-bit and 8-bit weights")
        if bits == 4 and codebook is None:
            raise ValueError("4-bit BitsAndBytes MLX inference requires a codebook")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.block_size = int(block_size)
        self.weight = mx.array(weight).reshape(-1)
        self.scales = mx.array(scales).astype(mx.float32)
        if codebook is not None:
            self.codebook = mx.array(codebook).astype(mx.float32).reshape(-1)
        if affine_biases is not None:
            self.weight = self.weight.reshape(self.out_features, -1)
            self.scales = self.scales.reshape(self.out_features, -1)
            self.affine_biases = mx.array(affine_biases).astype(mx.float32).reshape(self.out_features, -1)
        self.bias = mx.zeros((self.out_features,), dtype=mx.float32) if bias is None else mx.array(bias).astype(mx.float32)
        self.freeze()

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        if self.bits == 8 and "affine_biases" in self:
            output = mx.quantized_matmul(
                x, self.weight, scales=self.scales, biases=self.affine_biases,
                transpose=True, group_size=self.block_size, bits=8,
            )
            return (output + self.bias).astype(x.dtype)
        rows = x.size // self.in_features
        row_tile = 1 if rows == 1 else min(8, rows)
        common = {
            "grid": (_THREADS, self.out_features, (rows + row_tile - 1) // row_tile),
            "threadgroup": (_THREADS, 1, 1),
            "output_shapes": [(rows, self.out_features)],
            "output_dtypes": [mx.float32],
        }
        if self.bits == 4:
            output = _four_bit_kernel()(
                inputs=[x, self.weight, self.scales, self.codebook, self.bias],
                template=[("K", self.in_features), ("N", self.out_features), ("BLOCK", self.block_size),
                          ("ROWS", rows), ("RTILE", row_tile), ("THREADS", _THREADS),
                          ("EVEN", self.in_features % 2 == 0),
                          ("GROUPS", _THREADS // 32)],
                **common,
            )[0]
        else:
            output = _int8_kernel()(
                inputs=[x, self.weight, self.scales, self.bias],
                template=[("K", self.in_features), ("N", self.out_features),
                          ("ROWS", rows), ("RTILE", row_tile), ("THREADS", _THREADS),
                          ("GROUPS", _THREADS // 32)],
                **common,
            )[0]
        return output.reshape(output_shape).astype(x.dtype)
