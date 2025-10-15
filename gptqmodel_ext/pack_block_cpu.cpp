// SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
// SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
// SPDX-License-Identifier: Apache-2.0
// Contact: qubitium@modelcloud.ai, x.com/qubitium

#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace gptqmodel {

std::tuple<at::Tensor, at::Tensor> pack_block_cpu(
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& zeros,
    const at::Tensor& g_idx,
    int64_t bits,
    int64_t word_bits,
    int64_t block_in,
    int64_t threads
) {
    TORCH_CHECK(weight.device().is_cpu(), "weight must reside on CPU");
    TORCH_CHECK(scales.device().is_cpu(), "scales must reside on CPU");
    TORCH_CHECK(zeros.device().is_cpu(), "zeros must reside on CPU");
    TORCH_CHECK(g_idx.device().is_cpu(), "g_idx must reside on CPU");

    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out, in]");
    TORCH_CHECK(scales.dim() == 2, "scales must be 2D [groups, out]");
    TORCH_CHECK(zeros.dim() == 2, "zeros must be 2D [groups, out]");
    TORCH_CHECK(g_idx.dim() == 1, "g_idx must be 1D [in]");

    TORCH_CHECK(word_bits == 32, "Only 32-bit packing supported");

    at::Tensor weight_f = weight.contiguous().to(at::kFloat);
    at::Tensor scales_f = scales.contiguous().to(at::kFloat);
    at::Tensor zeros_i32 = zeros.contiguous().to(at::kInt);
    at::Tensor g_idx_i32 = g_idx.contiguous().to(at::kInt);

    at::Tensor scale_zeros = zeros_i32.to(at::kFloat) * scales_f;

    const int64_t out_features = weight_f.size(0);
    const int64_t in_features = weight_f.size(1);
    TORCH_CHECK(g_idx_i32.size(0) == in_features, "g_idx length mismatch");
    TORCH_CHECK(in_features % word_bits == 0, "in_features must be divisible by word_bits");

    const int64_t groups = scales_f.size(0);
    TORCH_CHECK(scales_f.size(1) == out_features, "scales shape mismatch");
    TORCH_CHECK(zeros_i32.size(0) == groups && zeros_i32.size(1) == out_features, "zeros shape mismatch");

    if (block_in <= 0) {
        block_in = word_bits;
    }
    block_in = std::max<int64_t>(word_bits, (block_in / word_bits) * word_bits);
    if (block_in == 0) {
        block_in = word_bits;
    }

    const int rows_per_group = (bits == 3) ? 3 : static_cast<int>(bits);
    const int64_t num_blocks = in_features / word_bits;

    auto q_options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    at::Tensor qweight = at::empty({num_blocks * rows_per_group, out_features}, q_options);

    int pack_factor = 0;
    if (bits == 2 || bits == 4 || bits == 8) {
        pack_factor = static_cast<int>(word_bits / bits);
    } else if (bits != 3) {
        TORCH_CHECK(false, "Unsupported bits value", bits);
    }

    const int max_q = (1 << bits) - 1;
    const float* weight_ptr = weight_f.const_data_ptr<float>();
    const float* scales_ptr = scales_f.const_data_ptr<float>();
    const float* scale_zeros_ptr = scale_zeros.const_data_ptr<float>();
    const int32_t* gidx_ptr = g_idx_i32.const_data_ptr<int32_t>();
    int32_t* qweight_ptr = qweight.data_ptr<int32_t>();

    const int64_t out_stride = in_features;
    const int64_t scales_stride = out_features;

    int64_t grain_size = block_in / word_bits;
    if (grain_size <= 0) {
        grain_size = 1;
    }
    if (threads > 0) {
        // Limit the number of parallel chunks to roughly `threads` without
        // mutating the global ATen thread configuration, keeping the kernel reentrant.
        int64_t target_chunk = (num_blocks + threads - 1) / threads;
        if (target_chunk <= 0) {
            target_chunk = 1;
        }
        grain_size = std::max<int64_t>(grain_size, target_chunk);
    }

    at::parallel_for(0, num_blocks, grain_size, [&](int64_t block_begin, int64_t block_end) {
        std::array<int32_t, 32> qvals{};
        for (int64_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
            const int64_t base_input = block_idx * word_bits;
            const int row_base = static_cast<int>(block_idx * rows_per_group);

            for (int out = 0; out < out_features; ++out) {
                for (int lane = 0; lane < word_bits; ++lane) {
                    const int64_t input_idx = base_input + lane;
                    const int32_t raw_group = gidx_ptr[input_idx];
                    int32_t group = raw_group;
                    if (group < 0) {
                        group += static_cast<int32_t>(groups);
                    }
                    TORCH_CHECK(
                        group >= 0 && group < groups,
                        "pack_block_cpu: g_idx[",
                        input_idx,
                        "]=",
                        raw_group,
                        " is out of range for groups=",
                        groups
                    );
                    float scale = scales_ptr[static_cast<int64_t>(group) * scales_stride + out];
                    float offset = scale_zeros_ptr[static_cast<int64_t>(group) * scales_stride + out];
                    float w = weight_ptr[out * out_stride + input_idx];
                    if (scale == 0.0f) {
                        scale = 1e-6f;
                    }
                    float qf = std::nearbyint((w + offset) / scale);
                    int32_t q = static_cast<int32_t>(qf);
                    q = std::max<int32_t>(0, std::min<int32_t>(q, max_q));
                    qvals[lane] = q;
                }

               if (bits == 3) {
                   int64_t A = 0;
                   for (int j = 0; j < 10; ++j) {
                       A |= static_cast<int64_t>(qvals[j]) << (3 * j);
                   }
                    A |= static_cast<int64_t>(qvals[10]) << 30;

                    int64_t B = static_cast<int64_t>((qvals[10] >> 2) & 0x1);
                    for (int j = 0; j < 10; ++j) {
                        B |= static_cast<int64_t>(qvals[11 + j]) << (3 * j + 1);
                    }
                    B |= static_cast<int64_t>(qvals[21]) << 31;

                    int64_t C = static_cast<int64_t>((qvals[21] >> 1) & 0x3);
                    for (int j = 0; j < 10; ++j) {
                        C |= static_cast<int64_t>(qvals[22 + j]) << (3 * j + 2);
                    }

                    qweight_ptr[(row_base + 0) * out_features + out] = static_cast<int32_t>(A & 0xFFFFFFFF);
                    qweight_ptr[(row_base + 1) * out_features + out] = static_cast<int32_t>(B & 0xFFFFFFFF);
                    qweight_ptr[(row_base + 2) * out_features + out] = static_cast<int32_t>(C & 0xFFFFFFFF);
               } else {
                   for (int bit_plane = 0; bit_plane < bits; ++bit_plane) {
                       int64_t packed = 0;
                       for (int pf = 0; pf < pack_factor; ++pf) {
                            int idx = bit_plane * pack_factor + pf;
                            packed |= static_cast<int64_t>(qvals[idx]) << (bits * pf);
                        }
                        qweight_ptr[(row_base + bit_plane) * out_features + out] = static_cast<int32_t>(packed & 0xFFFFFFFF);
                    }
                }
            }
        }
    });

    at::Tensor zeros_i32_contig = zeros_i32.contiguous();
    const int32_t* zeros_ptr = zeros_i32_contig.const_data_ptr<int32_t>();
    const int64_t zeros_stride = zeros_i32_contig.size(1);

    int64_t qzeros_cols = (out_features / word_bits) * bits;
    at::Tensor qzeros = at::zeros({groups, qzeros_cols}, q_options);
    int32_t* qzeros_ptr = qzeros.data_ptr<int32_t>();

    if (bits == 2 || bits == 4 || bits == 8) {
        int pack_factor_local = pack_factor;
        for (int64_t g = 0; g < groups; ++g) {
            const int32_t* zeros_row = zeros_ptr + g * zeros_stride;
            int32_t* dst_row = qzeros_ptr + g * qzeros_cols;
            for (int64_t col = 0; col < qzeros_cols; ++col) {
                int64_t base = col * pack_factor_local;
                int64_t packed = 0;
                for (int pf = 0; pf < pack_factor_local; ++pf) {
                    packed |= static_cast<int64_t>(zeros_row[base + pf]) << (bits * pf);
                }
                dst_row[col] = static_cast<int32_t>(packed & 0xFFFFFFFF);
            }
        }
    } else if (bits == 3) {
        for (int64_t g = 0; g < groups; ++g) {
            const int32_t* zeros_row = zeros_ptr + g * zeros_stride;
            int32_t* dst_row = qzeros_ptr + g * qzeros_cols;
            int idx = 0;
            for (int64_t col = 0; col < qzeros_cols;) {
                int64_t A = 0;
                for (int j = 0; j < 10; ++j) {
                    A |= static_cast<int64_t>(zeros_row[idx + j]) << (3 * j);
                }
                A |= static_cast<int64_t>(zeros_row[idx + 10]) << 30;
                dst_row[col++] = static_cast<int32_t>(A & 0xFFFFFFFF);

                int64_t B = static_cast<int64_t>((zeros_row[idx + 10] >> 2) & 0x1);
                for (int j = 0; j < 10; ++j) {
                    B |= static_cast<int64_t>(zeros_row[idx + 11 + j]) << (3 * j + 1);
                }
                B |= static_cast<int64_t>(zeros_row[idx + 21]) << 31;
                dst_row[col++] = static_cast<int32_t>(B & 0xFFFFFFFF);

                int64_t C = static_cast<int64_t>((zeros_row[idx + 21] >> 1) & 0x3);
                for (int j = 0; j < 10; ++j) {
                    C |= static_cast<int64_t>(zeros_row[idx + 22 + j]) << (3 * j + 2);
                }
                dst_row[col++] = static_cast<int32_t>(C & 0xFFFFFFFF);

                idx += 32;
            }
        }
    } else {
        TORCH_CHECK(false, "Unsupported bits for qzeros packing");
    }

    return {qweight, qzeros};
}

} // namespace gptqmodel

TORCH_LIBRARY(gptqmodel, m) {
    m.def(
        "pack_block_cpu(Tensor weight, Tensor scales, Tensor zeros, Tensor g_idx, int bits, int word_bits, int block_in, int threads) -> (Tensor, Tensor)"
    );
    m.impl(
        "pack_block_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel::pack_block_cpu)
    );
}
