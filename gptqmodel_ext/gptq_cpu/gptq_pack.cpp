// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "gptq_pack.hpp"

#include <ATen/Parallel.h>


namespace gptq_cpu
{

    namespace
    {
        inline uint8_t get_nibble(uint32_t word, int64_t nibble)
        {
            return static_cast<uint8_t>((word >> (4 * nibble)) & 0xF);
        }
    } // namespace

    std::vector<torch::Tensor> pack_gptq_int4(
        const torch::Tensor &qweight,
        const torch::Tensor &qzeros,
        const torch::Tensor &ret_idx,
        int64_t out_features,
        int64_t in_features)
    {
        TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight must be int32");
        TORCH_CHECK(qzeros.scalar_type() == at::kInt, "qzeros must be int32");
        TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D");
        TORCH_CHECK(qzeros.dim() == 2, "qzeros must be 2D");
        TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
        TORCH_CHECK(qzeros.is_contiguous(), "qzeros must be contiguous");
        TORCH_CHECK(out_features % 32 == 0, "out_features must be divisible by 32");
        TORCH_CHECK(in_features % 8 == 0, "in_features must be divisible by 8");
        TORCH_CHECK(
            qweight.size(0) * 8 == in_features,
            "qweight feature dimension does not match in_features");
        TORCH_CHECK(
            qweight.size(1) == out_features,
            "qweight output dimension does not match out_features");
        TORCH_CHECK(
            qzeros.size(1) * 8 == out_features,
            "qzeros output dimension does not match out_features");
        TORCH_CHECK(qzeros.size(0) > 0, "qzeros group count must be > 0");
        TORCH_CHECK(
            in_features % qzeros.size(0) == 0,
            "qzeros group count must divide in_features");

        const bool reorder = ret_idx.defined() && ret_idx.numel() != 0;
        if (reorder)
        {
            TORCH_CHECK(ret_idx.scalar_type() == at::kInt, "ret_idx must be int32");
            TORCH_CHECK(ret_idx.dim() == 1, "ret_idx must be 1D");
            TORCH_CHECK(ret_idx.numel() == in_features, "ret_idx size must match in_features");
            TORCH_CHECK(ret_idx.is_contiguous(), "ret_idx must be contiguous");
            TORCH_CHECK(
                ret_idx.min().item<int32_t>() >= 0 &&
                    ret_idx.max().item<int32_t>() < in_features,
                "ret_idx values must be in [0, in_features)");
        }

        const int64_t groups = out_features / 32;
        const int64_t k_half = in_features / 2;
        auto packed = torch::empty(
            {out_features, k_half},
            qweight.options().dtype(at::kByte));
        auto zero_codes = torch::empty(
            {qzeros.size(0), out_features},
            qzeros.options().dtype(at::kByte));

        const int32_t *qweight_ptr = qweight.data_ptr<int32_t>();
        const int32_t *qzeros_ptr = qzeros.data_ptr<int32_t>();
        const int32_t *ret_idx_ptr = reorder ? ret_idx.data_ptr<int32_t>() : nullptr;
        uint8_t *packed_ptr = packed.data_ptr<uint8_t>();
        uint8_t *zeros_ptr = zero_codes.data_ptr<uint8_t>();

        if (!reorder)
        {
            const int64_t k_words = in_features / 8;
            at::parallel_for(0, k_words, 1, [&](int64_t begin, int64_t end)
            {
                for (int64_t k_word = begin; k_word < end; ++k_word)
                {
                    const int32_t *word_row = qweight_ptr + k_word * out_features;
                    for (int64_t group = 0; group < groups; ++group)
                    {
                        uint8_t *out_group = packed_ptr + group * k_half * 32;
                        for (int64_t row = 0; row < 16; ++row)
                        {
                            const int64_t low_row = group * 32 + row;
                            const uint32_t low_word = static_cast<uint32_t>(word_row[low_row]);
                            const uint32_t high_word = static_cast<uint32_t>(word_row[low_row + 16]);
                            uint8_t *out = out_group + (4 * k_word) * 32 + row * 2;
                            for (int64_t nibble = 0; nibble < 8; ++nibble)
                            {
                                const uint8_t low_code = get_nibble(low_word, nibble);
                                const uint8_t high_code = get_nibble(high_word, nibble);
                                out[(nibble >> 1) * 32 + (nibble & 1)] =
                                    low_code | (high_code << 4);
                            }
                        }
                    }
                }
            });
        }
        else
        {
            at::parallel_for(0, groups, 1, [&](int64_t begin, int64_t end)
            {
                for (int64_t group = begin; group < end; ++group)
                {
                    const int64_t row_base = group * 32;
                    uint8_t *out_group = packed_ptr + group * k_half * 32;

                    for (int64_t k_half_idx = 0; k_half_idx < k_half; ++k_half_idx)
                    {
                        uint8_t *out_k = out_group + k_half_idx * 32;
                        for (int64_t row = 0; row < 16; ++row)
                        {
                            const int64_t low_row = row_base + row;
                            const int64_t high_row = low_row + 16;
                            for (int64_t pair = 0; pair < 2; ++pair)
                            {
                                const int64_t k_out = 2 * k_half_idx + pair;
                                const int64_t k_in = ret_idx_ptr[k_out];
                                const uint32_t shift = static_cast<uint32_t>(4 * (k_in & 7));
                                const int64_t word_base = (k_in >> 3) * out_features;
                                const uint8_t low_code = static_cast<uint8_t>(
                                    (static_cast<uint32_t>(qweight_ptr[word_base + low_row]) >> shift) & 0xF);
                                const uint8_t high_code = static_cast<uint8_t>(
                                    (static_cast<uint32_t>(qweight_ptr[word_base + high_row]) >> shift) & 0xF);
                                out_k[row * 2 + pair] = low_code | (high_code << 4);
                            }
                        }
                    }
                }
            });
        }

        at::parallel_for(0, zero_codes.size(0), 1, [&](int64_t begin, int64_t end)
        {
            for (int64_t group = begin; group < end; ++group)
            {
                const int64_t zero_word_base = group * (out_features / 8);
                uint8_t *out_group = zeros_ptr + group * out_features;
                for (int64_t word_idx = 0; word_idx < out_features / 8; ++word_idx)
                {
                    const uint32_t word = static_cast<uint32_t>(qzeros_ptr[zero_word_base + word_idx]);
                    for (int64_t nibble = 0; nibble < 8; ++nibble)
                    {
                        out_group[word_idx * 8 + nibble] = get_nibble(word, nibble);
                    }
                }
            }
        });

        return {packed, zero_codes};
    }

} // namespace gptq_cpu
