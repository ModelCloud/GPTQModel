#include "cpu_features.hpp"
#include "int4_qzero_group_scaled_avx512.hpp"
#include <ATen/ATen.h>
#include <ATen/native/CPUBlas.h>
#include <torch/library.h>

namespace {

bool qqq_brgemm_available() {
    if (!at::hasMKLDNN() || !at::globalContext().userEnabledMkldnn() ||
        !__builtin_cpu_supports("avx512vnni")) {
        return false;
    }
    static const bool available = [] {
        alignas(64) uint8_t input[4] = {};
        alignas(64) int8_t weight[4 * 32] = {};
        alignas(64) int32_t output[32] = {};
        try {
            at::native::cpublas::brgemm(1, 32, 4, 4, 32, 32, false, input, weight, output);
            at::native::cpublas::brgemm_release();
            return true;
        } catch (const c10::Error&) {
            at::native::cpublas::brgemm_release();
            return false;
        }
    }();
    return available;
}

__m512i qqq_weight_pair(__m256i codes, const float* scales) {
    const __m512i centered = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(codes), _mm512_set1_epi16(8));
    if (scales == nullptr) {
        return _mm512_slli_epi16(centered, 4);
    }
    const __m512 scale = _mm512_loadu_ps(scales);
    const __m512 scale_low = _mm512_permutexvar_ps(
        _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7), scale);
    const __m512 scale_high = _mm512_permutexvar_ps(
        _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15), scale);
    const auto requantize = [](__m256i values, __m512 group_scale) {
        const __m512 product = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(values)), group_scale);
        const __m512 clamped = _mm512_min_ps(
            _mm512_max_ps(product, _mm512_set1_ps(-128.0f)), _mm512_set1_ps(127.0f));
        const __m512i rounded = _mm512_cvt_roundps_epi32(
            clamped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm512_cvtepi32_epi16(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(product, product, _CMP_ORD_Q), rounded));
    };
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(requantize(_mm512_castsi512_si256(centered), scale_low)),
        requantize(_mm512_extracti64x4_epi64(centered, 1), scale_high), 1);
}

template <int Rows>
void qqq_tile(const int8_t* input, const uint8_t* weight, const float* scales,
              float* output, int64_t in_features, int64_t out_features, int64_t group_size) {
    __m512i accum[Rows * 2];
    for (auto& value : accum) {
        value = _mm512_setzero_si512();
    }
    const __m256i mask = _mm256_set1_epi8(15);
    for (int64_t group_start = 0; group_start < in_features; group_start += group_size) {
        const float* group_scales = scales == nullptr ? nullptr : scales + group_start / group_size * out_features;
        for (int64_t input_index = group_start; input_index < group_start + group_size; input_index += 2) {
            const __m256i packed = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(weight + input_index * 16));
            const __m512i weight_low = qqq_weight_pair(_mm256_and_si256(packed, mask), group_scales);
            const __m512i weight_high = qqq_weight_pair(
                _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask),
                group_scales == nullptr ? nullptr : group_scales + 16);
            int4_qzero_group_scaled_cpu::avx512::Unroll<Rows>{}([&](auto row) {
                const int8_t* row_input = input + row * in_features + input_index;
                const uint32_t pair = static_cast<uint16_t>(static_cast<int16_t>(row_input[0])) |
                    (static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(row_input[1]))) << 16);
                const __m512i activation = _mm512_set1_epi32(static_cast<int32_t>(pair));
                accum[row * 2] = _mm512_add_epi32(accum[row * 2], _mm512_madd_epi16(activation, weight_low));
                accum[row * 2 + 1] = _mm512_add_epi32(accum[row * 2 + 1], _mm512_madd_epi16(activation, weight_high));
            });
        }
    }
    for (int row = 0; row < Rows; ++row) {
        _mm512_storeu_ps(output + row * out_features, _mm512_cvtepi32_ps(accum[row * 2]));
        _mm512_storeu_ps(output + row * out_features + 16, _mm512_cvtepi32_ps(accum[row * 2 + 1]));
    }
}

__m512i qqq_int8_quad(__m512i first, __m512i second) {
    const __m256i first_bytes = _mm512_cvtepi16_epi8(first);
    const __m256i second_bytes = _mm512_cvtepi16_epi8(second);
    const __m512i interleaved = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm256_unpacklo_epi16(first_bytes, second_bytes)),
        _mm256_unpackhi_epi16(first_bytes, second_bytes), 1);
    return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), interleaved);
}

void qqq_brgemm_tile(const uint8_t* input, const uint8_t* weight, const float* scales,
                     float* output, int64_t rows, int64_t in_features, int64_t out_features, int64_t group_size) {
    alignas(64) int8_t unpacked[128 * 32];
    alignas(64) int32_t partial[128 * 32];
    alignas(64) int32_t accum[128 * 32] = {};
    const __m256i mask = _mm256_set1_epi8(15);
    for (int64_t group_start = 0; group_start < in_features; group_start += group_size) {
        const float* group_scales = scales == nullptr ? nullptr : scales + group_start / group_size * out_features;
        for (int64_t chunk_start = group_start; chunk_start < group_start + group_size; chunk_start += 128) {
            const int64_t chunk_size = std::min<int64_t>(128, group_start + group_size - chunk_start);
            __m512i correction_low = _mm512_setzero_si512();
            __m512i correction_high = _mm512_setzero_si512();
            const __m512i ones = _mm512_set1_epi16(1);
            for (int64_t offset = 0; offset < chunk_size; offset += 4) {
                const __m256i first = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weight + (chunk_start + offset) * 16));
                const __m256i second = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weight + (chunk_start + offset + 2) * 16));
                const __m512i first_low = qqq_weight_pair(_mm256_and_si256(first, mask), group_scales);
                const __m512i second_low = qqq_weight_pair(_mm256_and_si256(second, mask), group_scales);
                const __m512i first_high = qqq_weight_pair(
                    _mm256_and_si256(_mm256_srli_epi16(first, 4), mask),
                    group_scales == nullptr ? nullptr : group_scales + 16);
                const __m512i second_high = qqq_weight_pair(
                    _mm256_and_si256(_mm256_srli_epi16(second, 4), mask),
                    group_scales == nullptr ? nullptr : group_scales + 16);
                _mm512_store_si512(unpacked + offset * 32, qqq_int8_quad(first_low, second_low));
                _mm512_store_si512(unpacked + offset * 32 + 64, qqq_int8_quad(first_high, second_high));
                correction_low = _mm512_add_epi32(correction_low,
                    _mm512_madd_epi16(_mm512_add_epi16(first_low, second_low), ones));
                correction_high = _mm512_add_epi32(correction_high,
                    _mm512_madd_epi16(_mm512_add_epi16(first_high, second_high), ones));
            }
            at::native::cpublas::brgemm(rows, 32, chunk_size, in_features, 32, 32, false,
                                      input + chunk_start, unpacked, partial);
            correction_low = _mm512_slli_epi32(correction_low, 7);
            correction_high = _mm512_slli_epi32(correction_high, 7);
            for (int64_t row = 0; row < rows; ++row) {
                const auto add = [&](int64_t col, __m512i correction) {
                    const __m512i current = _mm512_sub_epi32(_mm512_load_si512(partial + row * 32 + col), correction);
                    _mm512_store_si512(accum + row * 32 + col,
                        _mm512_add_epi32(_mm512_load_si512(accum + row * 32 + col), current));
                };
                add(0, correction_low);
                add(16, correction_high);
            }
        }
    }
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < 32; col += 16) {
            _mm512_storeu_ps(output + row * out_features + col,
                            _mm512_cvtepi32_ps(_mm512_load_si512(accum + row * 32 + col)));
        }
    }
}

at::Tensor qqq_int4_cpu(const at::Tensor& input, const at::Tensor& weight,
                        const at::Tensor& scales, int64_t group_size) {
    TORCH_CHECK(int4_qzero_group_scaled_cpu::CPUFeatures::hasAVX512BF16(), "QQQ CPU requires AVX512-BF16");
    TORCH_CHECK(input.device().is_cpu() && weight.device().is_cpu() && scales.device().is_cpu(),
                "QQQ tensors must be on CPU");
    TORCH_CHECK(input.scalar_type() == at::kChar && weight.scalar_type() == at::kByte &&
                scales.scalar_type() == at::kFloat, "QQQ requires int8 input, uint8 weight, float32 scales");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "QQQ input and weight must be 2D");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && scales.is_contiguous(),
                "QQQ tensors must be contiguous");
    const int64_t rows = input.size(0);
    const int64_t in_features = input.size(1);
    const int64_t out_features = weight.size(0);
    TORCH_CHECK(in_features > 0 && in_features <= 131071 && in_features % 2 == 0,
                "QQQ input features must be positive, even, and <= 131071");
    TORCH_CHECK(out_features > 0 && out_features % 32 == 0 && weight.size(1) * 2 == in_features,
                "QQQ packed weight shape mismatch");
    TORCH_CHECK(group_size > 0 && group_size % 2 == 0 && in_features % group_size == 0,
                "QQQ group size must be positive, even, and divide input features");
    TORCH_CHECK((scales.numel() == 0 && group_size == in_features) ||
                (scales.dim() == 2 && scales.size(0) == in_features / group_size && scales.size(1) == out_features),
                "QQQ group scales shape mismatch");
    auto output = at::empty({rows, out_features}, input.options().dtype(at::kFloat));
    if (rows == 0) {
        return output;
    }
    const auto* activation = input.data_ptr<int8_t>();
    const auto* packed = weight.data_ptr<uint8_t>();
    const auto* group_scales = scales.numel() == 0 ? nullptr : scales.data_ptr<float>();
    auto* result = output.data_ptr<float>();
    const bool use_brgemm = rows > 4 && group_size % 4 == 0 && qqq_brgemm_available();
    const int64_t row_tile = use_brgemm ? (out_features / 32 >= at::get_num_threads() ? 128 : 32) : 4;
    const auto unsigned_input = use_brgemm ? input.to(at::kByte).add_(128) : at::Tensor();
    int4_qzero_group_scaled_cpu::avx512::parallel_2d(
        (rows + row_tile - 1) / row_tile, out_features / 32,
        [&](int64_t row_begin, int64_t row_end, int64_t col_begin, int64_t col_end) {
            for (int64_t col_block = col_begin; col_block < col_end; ++col_block) {
                const int64_t col = col_block * 32;
                for (int64_t row_block = row_begin; row_block < row_end; ++row_block) {
                    const int64_t row = row_block * row_tile;
                    if (use_brgemm) {
                        qqq_brgemm_tile(unsigned_input.data_ptr<uint8_t>() + row * in_features,
                            packed + col * in_features / 2, group_scales == nullptr ? nullptr : group_scales + col,
                            result + row * out_features + col, std::min<int64_t>(row_tile, rows - row),
                            in_features, out_features, group_size);
                        continue;
                    }
                    const auto launch = [&](auto count) {
                        qqq_tile<count>(activation + row * in_features, packed + col * in_features / 2,
                            group_scales == nullptr ? nullptr : group_scales + col,
                            result + row * out_features + col, in_features, out_features, group_size);
                    };
                    switch (std::min<int64_t>(4, rows - row)) {
                        case 1: launch(std::integral_constant<int, 1>{}); break;
                        case 2: launch(std::integral_constant<int, 2>{}); break;
                        case 3: launch(std::integral_constant<int, 3>{}); break;
                        case 4: launch(std::integral_constant<int, 4>{}); break;
                    }
                }
            }
            if (use_brgemm) {
                at::native::cpublas::brgemm_release();
            }
        });
    return output;
}

}

TORCH_LIBRARY_FRAGMENT(gptqmodel_int4_qzero_group_scaled_cpu, ops) {
    ops.def("qqq_int4(Tensor input, Tensor weight, Tensor scales, int group_size) -> Tensor");
    ops.impl("qqq_int4", c10::DispatchKey::CPU, &qqq_int4_cpu);
}
