# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
"""Packed EXL3 inference kernels for MLX with FP16/BF16 outputs."""

from functools import lru_cache

import mlx.core as mx
from mlx import nn

from ...quantization.mlx_exl3_hadamard import _exl3_hadamard_kernel

_CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}
_ROW_THREADS = 64
_FUSED_ROW_THREADS = 128
_PREFILL_THREADS = 128


def _hadamard_128(matrix):
    """Apply the normalized block-128 Hadamard without synchronizing the stream."""
    rows, columns = matrix.shape
    column_blocks = columns // 128
    return _exl3_hadamard_kernel(columns, 1)(
        inputs=[mx.contiguous(matrix)],
        template=[("AXIS", 1), ("COLUMNS", columns), ("COLUMN_BLOCKS", column_blocks)],
        grid=(rows * column_blocks * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[matrix.shape],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=120)
def _exl3_row_matmul_kernel(
    bits: int, codebook_id: int, fused_output: int = 0, output_dtype=mx.float32,
):
    output_types = {
        mx.float16: ("fp16", "half"),
        mx.bfloat16: ("bf16", "bfloat16_t"),
        mx.float32: ("fp32", "float"),
    }
    try:
        output_suffix, output_type = output_types[output_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported EXL3 activation dtype: {output_dtype}") from exc
    return mx.fast.metal_kernel(
        name=(f"gptqmodel_exl3_row_matmul_b{bits}_c{codebook_id}"
              f"_f{int(fused_output)}_{output_suffix}"),
        input_names=["x", "trellis", "svh", "bias"],
        output_names=["output"],
        header=r"""
            inline uint exl3_row_state(
                const device short* packed,
                uint tile_base,
                uint position,
                uint bits
            ) {
                uint stream_bits = 256u * bits;
                uint end_bit = (position + 1u) * bits;
                uint start_bit = (end_bit + stream_bits - 16u) % stream_bits;
                uint logical_word = start_bit >> 4;
                uint bit_in_word = start_bit & 15u;
                uint first = uint(as_type<ushort>(packed[tile_base + (logical_word ^ 1u)]));
                if (bit_in_word == 0u) return first;
                uint next_word = (logical_word + 1u) % (16u * bits);
                uint second = uint(as_type<ushort>(packed[tile_base + (next_word ^ 1u)]));
                return ((first << bit_in_word) | (second >> (16u - bit_in_word))) & 0xffffu;
            }

            inline half exl3_row_decode(uint state, uint codebook) {
                if (codebook == 0) {
                    uint raw = state * 89226354u + 64248484u;
                    raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
                    return as_type<half>(ushort(raw & 0xffffu))
                        + as_type<half>(ushort(raw >> 16));
                }
                if (codebook == 1) {
                    uint raw = state * 0xcbac1fedu;
                    raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
                    return as_type<half>(ushort(raw & 0xffffu))
                        + as_type<half>(ushort(raw >> 16));
                }
                uint raw = state * 0x83dcd12du;
                uint byte_sum = (raw & 0xffu) + ((raw >> 8) & 0xffu)
                    + ((raw >> 16) & 0xffu) + ((raw >> 24) & 0xffu);
                half accumulator = as_type<half>(ushort(byte_sum + 0x6400u));
                return metal::fma(
                    accumulator,
                    as_type<half>(ushort(0x1eeeu)),
                    as_type<half>(ushort(0xc931u))
                );
            }

        """,
        source=r"""
            uint column = thread_position_in_grid.x;
            uint lane = thread_position_in_threadgroup.x;
            uint local_column = column & 15u;
            uint output_tile = column >> 4;
            threadgroup float current[FUSE_OUTPUT ? 128 : 1];
            threadgroup float next_values[FUSE_OUTPUT ? 128 : 1];
            float sum = 0.0f;
            for (uint input_tile = 0; input_tile < K_TILES; ++input_tile) {
                uint tile_base = (input_tile * N_TILES + output_tile) * PACKED_WORDS;
                for (uint local_row = 0; local_row < 16u; ++local_row) {
                    uint tensor_thread = (local_column & 7u) * 4u + ((local_row & 7u) >> 1);
                    uint tensor_item = (local_row & 1u)
                        | (local_row >= 8u ? 2u : 0u)
                        | (local_column >= 8u ? 4u : 0u);
                    uint position = tensor_thread * 8u + tensor_item;
                    uint state = exl3_row_state(trellis, tile_base, position, BITS);
                    float weight = float(exl3_row_decode(state, CODEBOOK));
                    sum = metal::fma(x[input_tile * 16u + local_row], weight, sum);
                }
            }
            if (FUSE_OUTPUT) {
                for (uint stride = 1u; stride < 32u; stride <<= 1u) {
                    float other = simd_shuffle_xor(sum, stride);
                    sum = (lane & stride) == 0u ? sum + other : other - sum;
                }
                current[lane] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = 32u; stride < 128u; stride <<= 1u) {
                    uint partner = lane ^ stride;
                    float own = current[lane];
                    float other = current[partner];
                    next_values[lane] = (lane & stride) == 0u ? own + other : other - own;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    current[lane] = next_values[lane];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float value = current[lane] * 0.08838834764831845f * svh[column];
                if (FUSE_OUTPUT == 2) value += bias[column];
                output[column] = OUTPUT_TYPE(value);
            } else {
                output[column] = OUTPUT_TYPE(sum);
            }
        """.replace("OUTPUT_TYPE", output_type),
    )


@lru_cache(maxsize=24)
def _exl3_matmul_kernel(bits: int, codebook_id: int):
    return mx.fast.metal_kernel(
        name=f"gptqmodel_exl3_matmul_b{bits}_c{codebook_id}_r16",
        input_names=["x", "trellis"],
        output_names=["output"],
        header=r"""
            #include <metal_simdgroup_matrix>

            inline uint exl3_state(
                const device short* packed,
                uint tile_base,
                uint position,
                uint bits
            ) {
                // A trellis state is exactly the trailing 16-bit window of
                // the cyclic symbol stream at this matrix position.
                uint stream_bits = 256u * bits;
                uint end_bit = (position + 1u) * bits;
                uint start_bit = (end_bit + stream_bits - 16u) % stream_bits;
                uint logical_word = start_bit >> 4;
                uint bit_in_word = start_bit & 15u;
                uint first = uint(as_type<ushort>(packed[tile_base + (logical_word ^ 1u)]));
                if (bit_in_word == 0u) return first;
                uint next_word = (logical_word + 1u) % (16u * bits);
                uint second = uint(as_type<ushort>(packed[tile_base + (next_word ^ 1u)]));
                return ((first << bit_in_word) | (second >> (16u - bit_in_word))) & 0xffffu;
            }

            inline half exl3_decode(uint state, uint codebook) {
                if (codebook == 0) {
                    uint raw = state * 89226354u + 64248484u;
                    raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
                    return as_type<half>(ushort(raw & 0xffffu))
                        + as_type<half>(ushort(raw >> 16));
                }
                if (codebook == 1) {
                    uint raw = state * 0xcbac1fedu;
                    raw = 0x3b603b60u ^ (raw & 0x8fff8fffu);
                    return as_type<half>(ushort(raw & 0xffffu))
                        + as_type<half>(ushort(raw >> 16));
                }
                uint raw = state * 0x83dcd12du;
                uint byte_sum = (raw & 0xffu) + ((raw >> 8) & 0xffu)
                    + ((raw >> 16) & 0xffu) + ((raw >> 24) & 0xffu);
                half accumulator = as_type<half>(ushort(byte_sum + 0x6400u));
                return metal::fma(
                    accumulator,
                    as_type<half>(ushort(0x1eeeu)),
                    as_type<half>(ushort(0xc931u))
                );
            }
        """,
        source=r"""
            uint physical = thread_position_in_threadgroup.x;
            uint simdgroup = physical >> 5;
            uint output_tile = threadgroup_position_in_grid.y;
            uint row_base = threadgroup_position_in_grid.z * 16u;
            threadgroup float tile_weight[256];
            threadgroup float tile_input[256];
            simdgroup_float8x8 accumulator = make_filled_simdgroup_matrix<float, 8>(0.0f);

            for (uint input_tile = 0; input_tile < K_TILES; ++input_tile) {
                uint tile = input_tile * N_TILES + output_tile;
                uint tile_base = tile * PACKED_WORDS;
                for (uint position = physical; position < 256u; position += THREADS) {
                    // EXL3 stores each 16x16 tile in NVIDIA tensor-core lane order.
                    uint tensor_thread = position >> 3;
                    uint tensor_item = position & 7u;
                    uint local_row = (tensor_thread & 3u) * 2u + (tensor_item & 1u)
                        + ((tensor_item & 2u) ? 8u : 0u);
                    uint matrix_column = (tensor_thread >> 2)
                        + ((tensor_item & 4u) ? 8u : 0u);
                    uint state = exl3_state(trellis, tile_base, position, BITS);
                    tile_weight[local_row * 16u + matrix_column] = float(
                        exl3_decode(state, CODEBOOK)
                    );
                }
                for (uint position = physical; position < 256u; position += THREADS) {
                    uint input_row = position >> 4;
                    uint input_column = input_tile * 16u + (position & 15u);
                    tile_input[position] = row_base + input_row < ROWS
                        ? x[(row_base + input_row) * K + input_column]
                        : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simdgroup < 4u) {
                    uint input_row = (simdgroup >> 1) * 128u;
                    uint weight_column = (simdgroup & 1u) * 8u;
                    simdgroup_float8x8 input_fragment;
                    simdgroup_float8x8 weight_fragment;
                    simdgroup_load(input_fragment, tile_input + input_row, 16u);
                    simdgroup_load(
                        weight_fragment,
                        tile_weight + weight_column,
                        16u
                    );
                    simdgroup_multiply_accumulate(
                        accumulator, input_fragment, weight_fragment, accumulator
                    );
                    simdgroup_load(input_fragment, tile_input + input_row + 8u, 16u);
                    simdgroup_load(
                        weight_fragment,
                        tile_weight + 128u + weight_column,
                        16u
                    );
                    simdgroup_multiply_accumulate(
                        accumulator, input_fragment, weight_fragment, accumulator
                    );
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


class MlxEXL3Linear(nn.Module):
    """Multiply activations by EXL3 trellises without a dense weight fallback."""

    def __init__(self, in_features, out_features, bits, codebook="3inst", bias=False):
        super().__init__()
        if bits not in range(1, 9):
            raise ValueError("EXL3 MLX inference supports 1 through 8 bits")
        if codebook not in _CODEBOOK_IDS:
            raise ValueError(f"Unsupported EXL3 codebook: {codebook}")
        if in_features <= 0 or in_features % 128:
            raise ValueError("EXL3 input features must be a positive multiple of 128")
        if out_features <= 0 or out_features % 128:
            raise ValueError("EXL3 output features must be a positive multiple of 128")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.codebook = codebook
        self.trellis = mx.zeros(
            (in_features // 16, out_features // 16, bits * 16), dtype=mx.int16,
        )
        self.suh = mx.ones((in_features,), dtype=mx.float16)
        self.svh = mx.ones((out_features,), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float16)
        self.freeze()

    def __call__(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        output_shape = (*x.shape[:-1], self.out_features)
        if x.size == 0:
            return mx.zeros(output_shape, dtype=x.dtype)
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported EXL3 activation dtype: {x.dtype}")

        rows = x.size // self.in_features
        padded_rows = ((rows + 15) // 16) * 16
        fused_row = rows == 1 and self.out_features > self.in_features
        fused_mode = (2 if "bias" in self else 1) if fused_row else 0
        transformed = _hadamard_128(
            x.reshape(rows, self.in_features).astype(mx.float32) * self.suh,
        )
        kernel = (
            _exl3_row_matmul_kernel(
                self.bits, _CODEBOOK_IDS[self.codebook], fused_mode,
                x.dtype if fused_row else mx.float32,
            )
            if rows == 1
            else _exl3_matmul_kernel(self.bits, _CODEBOOK_IDS[self.codebook])
        )
        inner = kernel(
            inputs=([transformed, self.trellis, self.svh,
                     self.bias if "bias" in self else self.svh]
                    if rows == 1 else [transformed, self.trellis]),
            template=[
                ("BITS", self.bits),
                ("CODEBOOK", _CODEBOOK_IDS[self.codebook]),
                ("FUSE_OUTPUT", fused_mode),
                ("PACKED_WORDS", 16 * self.bits),
                ("THREADS", (_FUSED_ROW_THREADS if fused_row else _ROW_THREADS)
                            if rows == 1 else _PREFILL_THREADS),
                ("K", self.in_features), ("N", self.out_features),
                ("K_TILES", self.in_features // 16),
                ("N_TILES", self.out_features // 16),
                ("ROWS", rows),
            ],
            grid=(self.out_features, 1, 1) if rows == 1 else (
                _PREFILL_THREADS, self.out_features // 16, padded_rows // 16,
            ),
            threadgroup=((_FUSED_ROW_THREADS if fused_row else _ROW_THREADS)
                         if rows == 1 else _PREFILL_THREADS, 1, 1),
            output_shapes=[
                (1, self.out_features) if rows == 1 else (padded_rows, self.out_features),
            ],
            output_dtypes=[x.dtype if fused_row else mx.float32],
        )[0]
        inner = inner[:rows]
        output = inner if fused_row else _hadamard_128(inner) * self.svh
        if "bias" in self and not fused_row:
            output = output + self.bias
        return output.reshape(output_shape).astype(x.dtype)


__all__ = ["MlxEXL3Linear"]
