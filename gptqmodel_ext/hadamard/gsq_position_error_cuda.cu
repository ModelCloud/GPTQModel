// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__global__ void gsq_position_error_kernel(
    const float *__restrict__ probabilities,
    const __nv_bfloat16 *__restrict__ baseline,
    const uint8_t *__restrict__ position_indices,
    const uint8_t *__restrict__ position_choices,
    const __nv_bfloat16 *__restrict__ position_deltas,
    const __nv_bfloat16 *__restrict__ target,
    __nv_bfloat16 *__restrict__ error,
    int tile_count, int n, int output_tiles, int choices) {
    const int thread = threadIdx.x;
    const int dense_tile = blockIdx.x * 2 + thread / 128;
    const int pair = thread & 127;
    if (dense_tile < tile_count) {
        const int scalar = pair * 2;
        const int tile_row = dense_tile / output_tiles;
        const int tile_col = dense_tile - tile_row * output_tiles;
        const int row = scalar / 16;
        const int column = scalar - row * 16;
        const int matrix_offset = (tile_row * 16 + row) * n + tile_col * 16 + column;
        const auto base = *reinterpret_cast<const __nv_bfloat162 *>(
            baseline + dense_tile * 256 + scalar);
        const auto teacher = *reinterpret_cast<const __nv_bfloat162 *>(target + matrix_offset);
        *reinterpret_cast<__nv_bfloat162 *>(error + matrix_offset) = __hsub2(base, teacher);
    }
    __syncthreads();

    if (thread >= 128) return;
    const int tile = blockIdx.x * 2 + thread / 64;
    const int position = thread & 63;
    if (tile >= tile_count) return;
    const int compact_offset = (tile * 64 + position) * 3;
    if (position_choices[compact_offset] == 0) return;
    const int scalar = position_indices[tile * 64 + position];
    const int tile_row = tile / output_tiles;
    const int tile_col = tile - tile_row * output_tiles;
    const int row = scalar / 16;
    const int column = scalar - row * 16;
    const int matrix_offset = (tile_row * 16 + row) * n + tile_col * 16 + column;
    const int probability_offset = tile * choices;
    const int choice0 = position_choices[compact_offset];
    const int choice1 = position_choices[compact_offset + 1];
    const int choice2 = position_choices[compact_offset + 2];
    const float delta0 = __bfloat162float(position_deltas[compact_offset]);
    const float delta1 = __bfloat162float(position_deltas[compact_offset + 1]);
    const float delta2 = __bfloat162float(position_deltas[compact_offset + 2]);
    float contribution = __fmul_rn(probabilities[probability_offset + choice1], delta1);
    contribution = __fmaf_rn(probabilities[probability_offset + choice0], delta0, contribution);
    contribution = __fmaf_rn(probabilities[probability_offset + choice2], delta2, contribution);
    const float base_error = __fsub_rn(
        __bfloat162float(baseline[tile * 256 + scalar]),
        __bfloat162float(target[matrix_offset]));
    error[matrix_offset] = __float2bfloat16_rn(__fadd_rn(contribution, base_error));
}

}  // namespace

void gsq_position_error_cuda(
    const at::Tensor probabilities, const at::Tensor baseline,
    const at::Tensor position_indices, const at::Tensor position_choices,
    const at::Tensor position_deltas, const at::Tensor target,
    at::Tensor error, cudaStream_t stream) {
    const int tile_count = probabilities.size(0);
    const int n = target.size(1);
    const int output_tiles = n / 16;
    const int choices = probabilities.size(1);
    gsq_position_error_kernel<<<(tile_count + 1) / 2, 256, 0, stream>>>(
        probabilities.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16 *>(baseline.data_ptr<at::BFloat16>()),
        position_indices.data_ptr<uint8_t>(), position_choices.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(position_deltas.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(target.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(error.data_ptr<at::BFloat16>()),
        tile_count, n, output_tiles, choices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
