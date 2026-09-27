# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 encoding layouts: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
"""Packed FP8 inference kernels for MLX with FP16/BF16 outputs."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


class MlxFP8Linear(nn.Module):
    """Apply checkpoint scales in FP32 and preserve the activation dtype."""

    def __init__(self, in_features, out_features, output_scale, bias=None):
        super().__init__()
        self.linear = nn.QuantizedLinear(
            in_features, out_features, bias=False, group_size=32, bits=8, mode="mxfp8",
        )
        self.output_scale = mx.array(output_scale)
        if bias is not None:
            self.bias = mx.array(bias)
        self.freeze()

    def _unscaled_dot(self, x):
        # The unscaled E4M3 dot product can exceed FP16 even when the final
        # checkpoint output is finite.
        return self.linear(x.astype(mx.float32))

    def __call__(self, x):
        result = self._unscaled_dot(x) * self.output_scale
        if "bias" in self:
            result = result + self.bias
        return result.astype(x.dtype)


@lru_cache(maxsize=3)
def _packed_fp8_kernel(output_dtype):
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported FP8 activation dtype: {output_dtype}") from exc
    return mx.fast.metal_kernel(
        name=f"gptqmodel_fp8_packed_matmul_{output_suffix}",
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
            float output_scale = MODE == 0 ? 1.0f / scales[0] :
                (MODE == 1 ? 1.0f / scales[column] : 1.0f);
            if ((K & 3) == 0) {
                for (uint k = lane * 4u; k < K; k += THREADS * 4u) {
                    uint packed = *reinterpret_cast<const device uint *>(
                        weight + weight_offset + k
                    );
                    for (uint item = 0; item < 4u; ++item) {
                        uint input_column = k + item;
                        uint scale_index = (column / BLOCK_ROWS) * SCALE_COLS
                            + input_column / BLOCK_COLS;
                        float scale = MODE == 2 ? 1.0f / scales[scale_index]
                            : output_scale;
                        float value = codebook[(packed >> (item * 8u)) & 255u] * scale;
                        for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                            sums[r] = metal::fma(
                                float(x[(row_base + r) * K + input_column]),
                                value, sums[r]
                            );
                        }
                    }
                }
            } else {
                for (uint k = lane; k < K; k += THREADS) {
                    uint scale_index = (column / BLOCK_ROWS) * SCALE_COLS
                        + k / BLOCK_COLS;
                    float scale = MODE == 2 ? 1.0f / scales[scale_index]
                        : output_scale;
                    float value = codebook[uint(weight[weight_offset + k])] * scale;
                    for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                        sums[r] = metal::fma(
                            float(x[(row_base + r) * K + k]), value, sums[r]
                        );
                    }
                }
            }
            for (uint r = 0; r < RTILE; ++r) {
                float reduced = simd_sum(sums[r]);
                if ((lane & 31u) == 0)
                    partials[r * GROUPS + (lane >> 5)] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                float reduced = lane < GROUPS ? partials[r * GROUPS + lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0)
                    output[(row_base + r) * N + column] = OUTPUT_TYPE(
                        reduced + bias[column]
                    );
            }
        """.replace("OUTPUT_TYPE", output_type),
    )


@lru_cache(maxsize=1)
def _packed_fp8_prefill_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_fp8_packed_matmul_r16",
        input_names=["x", "weight", "scales", "codebook"],
        output_names=["output"],
        header="#include <metal_simdgroup_matrix>\n",
        source="""
            uint physical = thread_position_in_threadgroup.x;
            uint simdgroup = physical >> 5;
            uint output_tile = threadgroup_position_in_grid.y;
            uint row_base = threadgroup_position_in_grid.z * 16u;
            threadgroup float tile_weight[1024];
            threadgroup float tile_input[1024];
            simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8>(0.0f);

            for (uint input_tile = 0; input_tile < K_TILES; ++input_tile) {
                for (uint position = physical; position < 1024u; position += THREADS) {
                    uint local_k = position >> 4;
                    uint local_n = position & 15u;
                    uint column = output_tile * 16u + local_n;
                    uint k = input_tile * 64u + local_k;
                    uint scale_index = (column / BLOCK_ROWS) * SCALE_COLS
                        + k / BLOCK_COLS;
                    float scale = MODE == 0 ? 1.0f / scales[0] :
                        (MODE == 1 ? 1.0f / scales[column]
                        : 1.0f / scales[scale_index]);
                    tile_weight[position] =
                        codebook[uint(weight[column * K + k])] * scale;

                    uint input_row = position >> 6;
                    uint input_k = input_tile * 64u + (position & 63u);
                    tile_input[position] = row_base + input_row < ROWS
                        ? float(x[(row_base + input_row) * K + input_k]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simdgroup < 4u) {
                    uint input_row = (simdgroup >> 1) * 512u;
                    uint weight_column = (simdgroup & 1u) * 8u;
                    simdgroup_float8x8 input_fragment;
                    simdgroup_float8x8 weight_fragment;
                    for (uint part = 0; part < 8u; ++part) {
                        simdgroup_load(
                            input_fragment, tile_input + input_row + part * 8u, 64u
                        );
                        simdgroup_load(
                            weight_fragment,
                            tile_weight + part * 128u + weight_column,
                            16u
                        );
                        simdgroup_multiply_accumulate(
                            accumulator, input_fragment, weight_fragment, accumulator
                        );
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (simdgroup < 4u) {
                simdgroup_store(
                    accumulator,
                    output + (row_base + (simdgroup >> 1) * 8u) * N
                        + output_tile * 16u + (simdgroup & 1u) * 8u,
                    N
                );
            }
        """,
    )


class MlxFP8PackedLinear(nn.Module):
    """Multiply activations by FP8 checkpoint bytes without dense expansion."""

    def __init__(
        self,
        weight,
        scales,
        codebook,
        *,
        in_features,
        out_features,
        scale_method,
        block_size=None,
        bias=None,
    ):
        super().__init__()
        if scale_method not in {"tensor", "row", "block"}:
            raise ValueError(f"Unsupported FP8 scale method: {scale_method}")
        if scale_method == "block" and block_size is None:
            raise ValueError("FP8 block scaling requires a block size")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale_method = scale_method
        self.block_size = (1, 1) if block_size is None else tuple(map(int, block_size))
        self.weight = (
            mx.zeros((self.out_features, self.in_features), dtype=mx.uint8)
            if weight is None else mx.array(weight).astype(mx.uint8).reshape(
                self.out_features, self.in_features,
            )
        )
        if scales is None:
            block_rows, block_cols = self.block_size
            if scale_method == "tensor":
                scale_count = 1
            elif scale_method == "row":
                scale_count = self.out_features
            else:
                scale_count = (
                    (self.out_features // block_rows)
                    * (self.in_features // block_cols)
                )
            self.scales = mx.zeros((scale_count,), dtype=mx.float32)
        else:
            self.scales = mx.array(scales).astype(mx.float32).reshape(-1)
        self.codebook = (
            mx.zeros((256,), dtype=mx.float32)
            if codebook is None else mx.array(codebook).astype(mx.float32).reshape(256)
        )
        self.bias = (
            mx.zeros((self.out_features,), dtype=mx.float32)
            if bias is None else mx.array(bias).astype(mx.float32)
        )
        self.freeze()

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported FP8 activation dtype: {x.dtype}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        rows = x.size // self.in_features
        block_rows, block_cols = self.block_size
        mode = {"tensor": 0, "row": 1, "block": 2}[self.scale_method]
        if (rows != 1 and self.in_features <= 8192 and self.out_features >= 8192
                and self.in_features % 64 == 0 and self.out_features % 16 == 0):
            padded_rows = ((rows + 15) // 16) * 16
            output = _packed_fp8_prefill_kernel()(
                inputs=[x, self.weight, self.scales, self.codebook],
                template=[
                    ("K", self.in_features), ("N", self.out_features),
                    ("ROWS", rows), ("K_TILES", self.in_features // 64),
                    ("THREADS", 128), ("MODE", mode),
                    ("BLOCK_ROWS", block_rows), ("BLOCK_COLS", block_cols),
                    ("SCALE_COLS", self.in_features // block_cols),
                ],
                grid=(128, self.out_features // 16, padded_rows // 16),
                threadgroup=(128, 1, 1),
                output_shapes=[(padded_rows, self.out_features)],
                output_dtypes=[mx.float32],
            )[0][:rows]
            return (output + self.bias).reshape(output_shape).astype(x.dtype)
        row_tile = 1 if rows == 1 else min(rows, 16)
        threads = 64
        output = _packed_fp8_kernel(x.dtype)(
            inputs=[x, self.weight, self.scales, self.codebook, self.bias],
            template=[
                ("K", self.in_features), ("N", self.out_features),
                ("ROWS", rows), ("RTILE", row_tile),
                ("THREADS", threads), ("GROUPS", threads // 32),
                ("MODE", mode),
                ("BLOCK_ROWS", block_rows), ("BLOCK_COLS", block_cols),
                ("SCALE_COLS", self.in_features // block_cols),
            ],
            grid=(threads, self.out_features, (rows + row_tile - 1) // row_tile),
            threadgroup=(threads, 1, 1),
            output_shapes=[(rows, self.out_features)],
            output_dtypes=[x.dtype],
        )[0]
        return output.reshape(output_shape)


__all__ = ["MlxFP8Linear", "MlxFP8PackedLinear"]
