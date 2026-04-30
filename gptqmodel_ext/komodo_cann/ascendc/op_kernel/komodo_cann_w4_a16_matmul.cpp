#include "kernel_operator.h"

using namespace AscendC;

namespace {
class KomodoCannW4A16ScalarKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR packed_weight,
        GM_ADDR scales,
        GM_ADDR offsets,
        GM_ADDR bias,
        GM_ADDR y,
        const KomodoCannW4A16MatmulTilingData* tiling)
    {
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
        packed_weight_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packed_weight));
        scales_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scales));
        offsets_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(offsets));
        if (tiling->has_bias != 0) {
            bias_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bias));
        }
        y_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        const uint32_t rows = tiling_->rows;
        const uint32_t in_features = tiling_->in_features;
        const uint32_t out_features = tiling_->out_features;
        const uint32_t group_size = tiling_->group_size;
        const uint32_t has_bias = tiling_->has_bias;
        const uint32_t zero_offsets = tiling_->zero_offsets;
        const uint32_t physical_core_idx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t scheduled_blocks = static_cast<uint32_t>(GetBlockNum());
        const uint32_t block_dim = tiling_->block_dim != 0 ? tiling_->block_dim : scheduled_blocks;
        if (block_dim == 0) {
            return;
        }
        const uint32_t core_idx = physical_core_idx % block_dim;

        const uint32_t packed_stride = out_features >> 3;
        const uint32_t packed_per_core = (packed_stride + block_dim - 1) / block_dim;
        const uint32_t packed_begin = core_idx * packed_per_core;
        if (packed_begin >= packed_stride) {
            return;
        }
        const uint32_t packed_end_candidate = packed_begin + packed_per_core;
        const uint32_t packed_end = packed_end_candidate < packed_stride ? packed_end_candidate : packed_stride;
        uint32_t m = 0;
        for (; m + 7 < rows; m += 8) {
            ProcessRowOct(m, in_features, out_features, group_size, has_bias, zero_offsets, packed_begin, packed_end);
        }
        for (; m + 3 < rows; m += 4) {
            ProcessRowQuad(m, in_features, out_features, group_size, has_bias, packed_begin, packed_end);
        }
        for (; m + 1 < rows; m += 2) {
            ProcessRowPair(m, in_features, out_features, group_size, has_bias, packed_begin, packed_end);
        }
        if (m < rows) {
            ProcessSingleRow(m, in_features, out_features, group_size, has_bias, packed_begin, packed_end);
        }
    }

private:
    __aicore__ inline void ProcessSingleRow(
        uint32_t m,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t has_bias,
        uint32_t packed_begin,
        uint32_t packed_end)
    {
        const uint32_t packed_stride = out_features >> 3;
        const uint32_t row_offset = m * out_features;
        const uint32_t x_offset = m * in_features;
        uint32_t packed_col = packed_begin;
        for (; packed_col + 1 < packed_end; packed_col += 2) {
            const uint32_t n_base0 = packed_col << 3;
            const uint32_t n_base1 = n_base0 + 8;
            float acc00 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0)) : 0.0f;
            float acc01 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 1)) : 0.0f;
            float acc02 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 2)) : 0.0f;
            float acc03 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 3)) : 0.0f;
            float acc04 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 4)) : 0.0f;
            float acc05 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 5)) : 0.0f;
            float acc06 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 6)) : 0.0f;
            float acc07 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 7)) : 0.0f;
            float acc10 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1)) : 0.0f;
            float acc11 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 1)) : 0.0f;
            float acc12 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 2)) : 0.0f;
            float acc13 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 3)) : 0.0f;
            float acc14 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 4)) : 0.0f;
            float acc15 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 5)) : 0.0f;
            float acc16 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 6)) : 0.0f;
            float acc17 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 7)) : 0.0f;

            const uint32_t groups = group_size == 0 ? 1 : in_features / group_size;
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base0 = group * out_features + n_base0;
                const uint32_t scale_base1 = scale_base0 + 8;
                const float scale00 = static_cast<float>(scales_gm_.GetValue(scale_base0));
                const float scale01 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 1));
                const float scale02 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 2));
                const float scale03 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 3));
                const float scale04 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 4));
                const float scale05 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 5));
                const float scale06 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 6));
                const float scale07 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 7));
                const float scale10 = static_cast<float>(scales_gm_.GetValue(scale_base1));
                const float scale11 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 1));
                const float scale12 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 2));
                const float scale13 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 3));
                const float scale14 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 4));
                const float scale15 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 5));
                const float scale16 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 6));
                const float scale17 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 7));
                const float offset00 = static_cast<float>(offsets_gm_.GetValue(scale_base0));
                const float offset01 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 1));
                const float offset02 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 2));
                const float offset03 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 3));
                const float offset04 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 4));
                const float offset05 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 5));
                const float offset06 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 6));
                const float offset07 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 7));
                const float offset10 = static_cast<float>(offsets_gm_.GetValue(scale_base1));
                const float offset11 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 1));
                const float offset12 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 2));
                const float offset13 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 3));
                const float offset14 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 4));
                const float offset15 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 5));
                const float offset16 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 6));
                const float offset17 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 7));

                // Offset is constant within a quant group, so M1 can apply its
                // contribution once after accumulating the signed INT4 lanes.
                float x_sum = 0.0f;
                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value = static_cast<float>(x_gm_.GetValue(x_offset + k));
                    x_sum += x_value;
                    const uint32_t word0 =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t word1 =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col + 1));
                    acc00 += x_value * DequantLaneNoOffset(word0, 0, scale00);
                    acc01 += x_value * DequantLaneNoOffset(word0, 1, scale01);
                    acc02 += x_value * DequantLaneNoOffset(word0, 2, scale02);
                    acc03 += x_value * DequantLaneNoOffset(word0, 3, scale03);
                    acc04 += x_value * DequantLaneNoOffset(word0, 4, scale04);
                    acc05 += x_value * DequantLaneNoOffset(word0, 5, scale05);
                    acc06 += x_value * DequantLaneNoOffset(word0, 6, scale06);
                    acc07 += x_value * DequantLaneNoOffset(word0, 7, scale07);
                    acc10 += x_value * DequantLaneNoOffset(word1, 0, scale10);
                    acc11 += x_value * DequantLaneNoOffset(word1, 1, scale11);
                    acc12 += x_value * DequantLaneNoOffset(word1, 2, scale12);
                    acc13 += x_value * DequantLaneNoOffset(word1, 3, scale13);
                    acc14 += x_value * DequantLaneNoOffset(word1, 4, scale14);
                    acc15 += x_value * DequantLaneNoOffset(word1, 5, scale15);
                    acc16 += x_value * DequantLaneNoOffset(word1, 6, scale16);
                    acc17 += x_value * DequantLaneNoOffset(word1, 7, scale17);
                }
                acc00 += x_sum * offset00 * scale00;
                acc01 += x_sum * offset01 * scale01;
                acc02 += x_sum * offset02 * scale02;
                acc03 += x_sum * offset03 * scale03;
                acc04 += x_sum * offset04 * scale04;
                acc05 += x_sum * offset05 * scale05;
                acc06 += x_sum * offset06 * scale06;
                acc07 += x_sum * offset07 * scale07;
                acc10 += x_sum * offset10 * scale10;
                acc11 += x_sum * offset11 * scale11;
                acc12 += x_sum * offset12 * scale12;
                acc13 += x_sum * offset13 * scale13;
                acc14 += x_sum * offset14 * scale14;
                acc15 += x_sum * offset15 * scale15;
                acc16 += x_sum * offset16 * scale16;
                acc17 += x_sum * offset17 * scale17;
            }

            StoreRow(row_offset + n_base0, acc00, acc01, acc02, acc03, acc04, acc05, acc06, acc07);
            StoreRow(row_offset + n_base1, acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17);
        }
        for (; packed_col < packed_end; ++packed_col) {
            const uint32_t n_base = packed_col << 3;
            float acc0 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base)) : 0.0f;
            float acc1 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 1)) : 0.0f;
            float acc2 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 2)) : 0.0f;
            float acc3 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 3)) : 0.0f;
            float acc4 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 4)) : 0.0f;
            float acc5 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 5)) : 0.0f;
            float acc6 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 6)) : 0.0f;
            float acc7 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 7)) : 0.0f;

            const uint32_t groups = group_size == 0 ? 1 : in_features / group_size;
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base = group * out_features + n_base;
                const float scale0 = static_cast<float>(scales_gm_.GetValue(scale_base));
                const float scale1 = static_cast<float>(scales_gm_.GetValue(scale_base + 1));
                const float scale2 = static_cast<float>(scales_gm_.GetValue(scale_base + 2));
                const float scale3 = static_cast<float>(scales_gm_.GetValue(scale_base + 3));
                const float scale4 = static_cast<float>(scales_gm_.GetValue(scale_base + 4));
                const float scale5 = static_cast<float>(scales_gm_.GetValue(scale_base + 5));
                const float scale6 = static_cast<float>(scales_gm_.GetValue(scale_base + 6));
                const float scale7 = static_cast<float>(scales_gm_.GetValue(scale_base + 7));
                const float offset0 = static_cast<float>(offsets_gm_.GetValue(scale_base));
                const float offset1 = static_cast<float>(offsets_gm_.GetValue(scale_base + 1));
                const float offset2 = static_cast<float>(offsets_gm_.GetValue(scale_base + 2));
                const float offset3 = static_cast<float>(offsets_gm_.GetValue(scale_base + 3));
                const float offset4 = static_cast<float>(offsets_gm_.GetValue(scale_base + 4));
                const float offset5 = static_cast<float>(offsets_gm_.GetValue(scale_base + 5));
                const float offset6 = static_cast<float>(offsets_gm_.GetValue(scale_base + 6));
                const float offset7 = static_cast<float>(offsets_gm_.GetValue(scale_base + 7));

                // Offset is constant within a quant group, so M1 can apply its
                // contribution once after accumulating the signed INT4 lanes.
                float x_sum = 0.0f;
                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value = static_cast<float>(x_gm_.GetValue(x_offset + k));
                    x_sum += x_value;
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    acc0 += x_value * DequantLaneNoOffset(word, 0, scale0);
                    acc1 += x_value * DequantLaneNoOffset(word, 1, scale1);
                    acc2 += x_value * DequantLaneNoOffset(word, 2, scale2);
                    acc3 += x_value * DequantLaneNoOffset(word, 3, scale3);
                    acc4 += x_value * DequantLaneNoOffset(word, 4, scale4);
                    acc5 += x_value * DequantLaneNoOffset(word, 5, scale5);
                    acc6 += x_value * DequantLaneNoOffset(word, 6, scale6);
                    acc7 += x_value * DequantLaneNoOffset(word, 7, scale7);
                }
                acc0 += x_sum * offset0 * scale0;
                acc1 += x_sum * offset1 * scale1;
                acc2 += x_sum * offset2 * scale2;
                acc3 += x_sum * offset3 * scale3;
                acc4 += x_sum * offset4 * scale4;
                acc5 += x_sum * offset5 * scale5;
                acc6 += x_sum * offset6 * scale6;
                acc7 += x_sum * offset7 * scale7;
            }

            StoreRow(row_offset + n_base, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        }
    }

    __aicore__ inline void ProcessRowOct(
        uint32_t m,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t has_bias,
        uint32_t zero_offsets,
        uint32_t packed_begin,
        uint32_t packed_end)
    {
        const uint32_t packed_stride = out_features >> 3;
        const uint32_t row_offset0 = m * out_features;
        const uint32_t row_offset1 = row_offset0 + out_features;
        const uint32_t row_offset2 = row_offset1 + out_features;
        const uint32_t row_offset3 = row_offset2 + out_features;
        const uint32_t row_offset4 = row_offset3 + out_features;
        const uint32_t row_offset5 = row_offset4 + out_features;
        const uint32_t row_offset6 = row_offset5 + out_features;
        const uint32_t row_offset7 = row_offset6 + out_features;
        const uint32_t x_offset0 = m * in_features;
        const uint32_t x_offset1 = x_offset0 + in_features;
        const uint32_t x_offset2 = x_offset1 + in_features;
        const uint32_t x_offset3 = x_offset2 + in_features;
        const uint32_t x_offset4 = x_offset3 + in_features;
        const uint32_t x_offset5 = x_offset4 + in_features;
        const uint32_t x_offset6 = x_offset5 + in_features;
        const uint32_t x_offset7 = x_offset6 + in_features;

#define KOMODO_INIT_ACC(prefix) \
        float prefix##0 = bias0; \
        float prefix##1 = bias1; \
        float prefix##2 = bias2; \
        float prefix##3 = bias3; \
        float prefix##4 = bias4; \
        float prefix##5 = bias5; \
        float prefix##6 = bias6; \
        float prefix##7 = bias7

#define KOMODO_ACCUM_ROW(prefix, x_value) \
        prefix##0 += (x_value) * deq0; \
        prefix##1 += (x_value) * deq1; \
        prefix##2 += (x_value) * deq2; \
        prefix##3 += (x_value) * deq3; \
        prefix##4 += (x_value) * deq4; \
        prefix##5 += (x_value) * deq5; \
        prefix##6 += (x_value) * deq6; \
        prefix##7 += (x_value) * deq7

        for (uint32_t packed_col = packed_begin; packed_col < packed_end; ++packed_col) {
            const uint32_t n_base = packed_col << 3;
            const float bias0 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base)) : 0.0f;
            const float bias1 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 1)) : 0.0f;
            const float bias2 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 2)) : 0.0f;
            const float bias3 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 3)) : 0.0f;
            const float bias4 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 4)) : 0.0f;
            const float bias5 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 5)) : 0.0f;
            const float bias6 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 6)) : 0.0f;
            const float bias7 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 7)) : 0.0f;
            KOMODO_INIT_ACC(acc0);
            KOMODO_INIT_ACC(acc1);
            KOMODO_INIT_ACC(acc2);
            KOMODO_INIT_ACC(acc3);
            KOMODO_INIT_ACC(acc4);
            KOMODO_INIT_ACC(acc5);
            KOMODO_INIT_ACC(acc6);
            KOMODO_INIT_ACC(acc7);

            const uint32_t groups = group_size == 0 ? 1 : in_features / group_size;
            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base = group * out_features + n_base;
                const float scale0 = static_cast<float>(scales_gm_.GetValue(scale_base));
                const float scale1 = static_cast<float>(scales_gm_.GetValue(scale_base + 1));
                const float scale2 = static_cast<float>(scales_gm_.GetValue(scale_base + 2));
                const float scale3 = static_cast<float>(scales_gm_.GetValue(scale_base + 3));
                const float scale4 = static_cast<float>(scales_gm_.GetValue(scale_base + 4));
                const float scale5 = static_cast<float>(scales_gm_.GetValue(scale_base + 5));
                const float scale6 = static_cast<float>(scales_gm_.GetValue(scale_base + 6));
                const float scale7 = static_cast<float>(scales_gm_.GetValue(scale_base + 7));
                const float offset0 = OffsetValue(scale_base, zero_offsets);
                const float offset1 = OffsetValue(scale_base + 1, zero_offsets);
                const float offset2 = OffsetValue(scale_base + 2, zero_offsets);
                const float offset3 = OffsetValue(scale_base + 3, zero_offsets);
                const float offset4 = OffsetValue(scale_base + 4, zero_offsets);
                const float offset5 = OffsetValue(scale_base + 5, zero_offsets);
                const float offset6 = OffsetValue(scale_base + 6, zero_offsets);
                const float offset7 = OffsetValue(scale_base + 7, zero_offsets);

                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                    const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
                    const float x_value2 = static_cast<float>(x_gm_.GetValue(x_offset2 + k));
                    const float x_value3 = static_cast<float>(x_gm_.GetValue(x_offset3 + k));
                    const float x_value4 = static_cast<float>(x_gm_.GetValue(x_offset4 + k));
                    const float x_value5 = static_cast<float>(x_gm_.GetValue(x_offset5 + k));
                    const float x_value6 = static_cast<float>(x_gm_.GetValue(x_offset6 + k));
                    const float x_value7 = static_cast<float>(x_gm_.GetValue(x_offset7 + k));
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const float deq0 = DequantLane(word, 0, scale0, offset0);
                    const float deq1 = DequantLane(word, 1, scale1, offset1);
                    const float deq2 = DequantLane(word, 2, scale2, offset2);
                    const float deq3 = DequantLane(word, 3, scale3, offset3);
                    const float deq4 = DequantLane(word, 4, scale4, offset4);
                    const float deq5 = DequantLane(word, 5, scale5, offset5);
                    const float deq6 = DequantLane(word, 6, scale6, offset6);
                    const float deq7 = DequantLane(word, 7, scale7, offset7);
                    KOMODO_ACCUM_ROW(acc0, x_value0);
                    KOMODO_ACCUM_ROW(acc1, x_value1);
                    KOMODO_ACCUM_ROW(acc2, x_value2);
                    KOMODO_ACCUM_ROW(acc3, x_value3);
                    KOMODO_ACCUM_ROW(acc4, x_value4);
                    KOMODO_ACCUM_ROW(acc5, x_value5);
                    KOMODO_ACCUM_ROW(acc6, x_value6);
                    KOMODO_ACCUM_ROW(acc7, x_value7);
                }
            }

            StoreRow(row_offset0 + n_base, acc00, acc01, acc02, acc03, acc04, acc05, acc06, acc07);
            StoreRow(row_offset1 + n_base, acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17);
            StoreRow(row_offset2 + n_base, acc20, acc21, acc22, acc23, acc24, acc25, acc26, acc27);
            StoreRow(row_offset3 + n_base, acc30, acc31, acc32, acc33, acc34, acc35, acc36, acc37);
            StoreRow(row_offset4 + n_base, acc40, acc41, acc42, acc43, acc44, acc45, acc46, acc47);
            StoreRow(row_offset5 + n_base, acc50, acc51, acc52, acc53, acc54, acc55, acc56, acc57);
            StoreRow(row_offset6 + n_base, acc60, acc61, acc62, acc63, acc64, acc65, acc66, acc67);
            StoreRow(row_offset7 + n_base, acc70, acc71, acc72, acc73, acc74, acc75, acc76, acc77);
        }

#undef KOMODO_ACCUM_ROW
#undef KOMODO_INIT_ACC
    }

    __aicore__ inline void ProcessRowQuad(
        uint32_t m,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t has_bias,
        uint32_t packed_begin,
        uint32_t packed_end)
    {
        const uint32_t packed_stride = out_features >> 3;
        const uint32_t row_offset0 = m * out_features;
        const uint32_t row_offset1 = row_offset0 + out_features;
        const uint32_t row_offset2 = row_offset1 + out_features;
        const uint32_t row_offset3 = row_offset2 + out_features;
        const uint32_t x_offset0 = m * in_features;
        const uint32_t x_offset1 = x_offset0 + in_features;
        const uint32_t x_offset2 = x_offset1 + in_features;
        const uint32_t x_offset3 = x_offset2 + in_features;

#define KOMODO_INIT_ACC8(prefix, b0, b1, b2, b3, b4, b5, b6, b7) \
        float prefix##0 = b0; \
        float prefix##1 = b1; \
        float prefix##2 = b2; \
        float prefix##3 = b3; \
        float prefix##4 = b4; \
        float prefix##5 = b5; \
        float prefix##6 = b6; \
        float prefix##7 = b7

#define KOMODO_ACCUM8(prefix, x_value, d0, d1, d2, d3, d4, d5, d6, d7) \
        prefix##0 += (x_value) * d0; \
        prefix##1 += (x_value) * d1; \
        prefix##2 += (x_value) * d2; \
        prefix##3 += (x_value) * d3; \
        prefix##4 += (x_value) * d4; \
        prefix##5 += (x_value) * d5; \
        prefix##6 += (x_value) * d6; \
        prefix##7 += (x_value) * d7

        const uint32_t groups = group_size == 0 ? 1 : in_features / group_size;
        uint32_t packed_col = packed_begin;
        for (; packed_col + 1 < packed_end; packed_col += 2) {
            const uint32_t n_base0 = packed_col << 3;
            const uint32_t n_base1 = n_base0 + 8;
            const float bias00 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0)) : 0.0f;
            const float bias01 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 1)) : 0.0f;
            const float bias02 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 2)) : 0.0f;
            const float bias03 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 3)) : 0.0f;
            const float bias04 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 4)) : 0.0f;
            const float bias05 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 5)) : 0.0f;
            const float bias06 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 6)) : 0.0f;
            const float bias07 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 7)) : 0.0f;
            const float bias10 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1)) : 0.0f;
            const float bias11 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 1)) : 0.0f;
            const float bias12 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 2)) : 0.0f;
            const float bias13 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 3)) : 0.0f;
            const float bias14 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 4)) : 0.0f;
            const float bias15 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 5)) : 0.0f;
            const float bias16 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 6)) : 0.0f;
            const float bias17 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 7)) : 0.0f;
            KOMODO_INIT_ACC8(acc00, bias00, bias01, bias02, bias03, bias04, bias05, bias06, bias07);
            KOMODO_INIT_ACC8(acc01, bias10, bias11, bias12, bias13, bias14, bias15, bias16, bias17);
            KOMODO_INIT_ACC8(acc10, bias00, bias01, bias02, bias03, bias04, bias05, bias06, bias07);
            KOMODO_INIT_ACC8(acc11, bias10, bias11, bias12, bias13, bias14, bias15, bias16, bias17);
            KOMODO_INIT_ACC8(acc20, bias00, bias01, bias02, bias03, bias04, bias05, bias06, bias07);
            KOMODO_INIT_ACC8(acc21, bias10, bias11, bias12, bias13, bias14, bias15, bias16, bias17);
            KOMODO_INIT_ACC8(acc30, bias00, bias01, bias02, bias03, bias04, bias05, bias06, bias07);
            KOMODO_INIT_ACC8(acc31, bias10, bias11, bias12, bias13, bias14, bias15, bias16, bias17);

            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base0 = group * out_features + n_base0;
                const uint32_t scale_base1 = scale_base0 + 8;
                const float scale00 = static_cast<float>(scales_gm_.GetValue(scale_base0));
                const float scale01 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 1));
                const float scale02 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 2));
                const float scale03 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 3));
                const float scale04 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 4));
                const float scale05 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 5));
                const float scale06 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 6));
                const float scale07 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 7));
                const float scale10 = static_cast<float>(scales_gm_.GetValue(scale_base1));
                const float scale11 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 1));
                const float scale12 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 2));
                const float scale13 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 3));
                const float scale14 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 4));
                const float scale15 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 5));
                const float scale16 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 6));
                const float scale17 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 7));
                const float offset00 = static_cast<float>(offsets_gm_.GetValue(scale_base0));
                const float offset01 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 1));
                const float offset02 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 2));
                const float offset03 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 3));
                const float offset04 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 4));
                const float offset05 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 5));
                const float offset06 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 6));
                const float offset07 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 7));
                const float offset10 = static_cast<float>(offsets_gm_.GetValue(scale_base1));
                const float offset11 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 1));
                const float offset12 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 2));
                const float offset13 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 3));
                const float offset14 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 4));
                const float offset15 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 5));
                const float offset16 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 6));
                const float offset17 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 7));

                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                    const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
                    const float x_value2 = static_cast<float>(x_gm_.GetValue(x_offset2 + k));
                    const float x_value3 = static_cast<float>(x_gm_.GetValue(x_offset3 + k));
                    const uint32_t word0 =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t word1 =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col + 1));
                    const float deq00 = DequantLane(word0, 0, scale00, offset00);
                    const float deq01 = DequantLane(word0, 1, scale01, offset01);
                    const float deq02 = DequantLane(word0, 2, scale02, offset02);
                    const float deq03 = DequantLane(word0, 3, scale03, offset03);
                    const float deq04 = DequantLane(word0, 4, scale04, offset04);
                    const float deq05 = DequantLane(word0, 5, scale05, offset05);
                    const float deq06 = DequantLane(word0, 6, scale06, offset06);
                    const float deq07 = DequantLane(word0, 7, scale07, offset07);
                    const float deq10 = DequantLane(word1, 0, scale10, offset10);
                    const float deq11 = DequantLane(word1, 1, scale11, offset11);
                    const float deq12 = DequantLane(word1, 2, scale12, offset12);
                    const float deq13 = DequantLane(word1, 3, scale13, offset13);
                    const float deq14 = DequantLane(word1, 4, scale14, offset14);
                    const float deq15 = DequantLane(word1, 5, scale15, offset15);
                    const float deq16 = DequantLane(word1, 6, scale16, offset16);
                    const float deq17 = DequantLane(word1, 7, scale17, offset17);
                    KOMODO_ACCUM8(acc00, x_value0, deq00, deq01, deq02, deq03, deq04, deq05, deq06, deq07);
                    KOMODO_ACCUM8(acc01, x_value0, deq10, deq11, deq12, deq13, deq14, deq15, deq16, deq17);
                    KOMODO_ACCUM8(acc10, x_value1, deq00, deq01, deq02, deq03, deq04, deq05, deq06, deq07);
                    KOMODO_ACCUM8(acc11, x_value1, deq10, deq11, deq12, deq13, deq14, deq15, deq16, deq17);
                    KOMODO_ACCUM8(acc20, x_value2, deq00, deq01, deq02, deq03, deq04, deq05, deq06, deq07);
                    KOMODO_ACCUM8(acc21, x_value2, deq10, deq11, deq12, deq13, deq14, deq15, deq16, deq17);
                    KOMODO_ACCUM8(acc30, x_value3, deq00, deq01, deq02, deq03, deq04, deq05, deq06, deq07);
                    KOMODO_ACCUM8(acc31, x_value3, deq10, deq11, deq12, deq13, deq14, deq15, deq16, deq17);
                }
            }

            StoreRow(row_offset0 + n_base0, acc000, acc001, acc002, acc003, acc004, acc005, acc006, acc007);
            StoreRow(row_offset0 + n_base1, acc010, acc011, acc012, acc013, acc014, acc015, acc016, acc017);
            StoreRow(row_offset1 + n_base0, acc100, acc101, acc102, acc103, acc104, acc105, acc106, acc107);
            StoreRow(row_offset1 + n_base1, acc110, acc111, acc112, acc113, acc114, acc115, acc116, acc117);
            StoreRow(row_offset2 + n_base0, acc200, acc201, acc202, acc203, acc204, acc205, acc206, acc207);
            StoreRow(row_offset2 + n_base1, acc210, acc211, acc212, acc213, acc214, acc215, acc216, acc217);
            StoreRow(row_offset3 + n_base0, acc300, acc301, acc302, acc303, acc304, acc305, acc306, acc307);
            StoreRow(row_offset3 + n_base1, acc310, acc311, acc312, acc313, acc314, acc315, acc316, acc317);
        }

        for (; packed_col < packed_end; ++packed_col) {
            const uint32_t n_base = packed_col << 3;
            const float bias0 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base)) : 0.0f;
            const float bias1 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 1)) : 0.0f;
            const float bias2 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 2)) : 0.0f;
            const float bias3 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 3)) : 0.0f;
            const float bias4 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 4)) : 0.0f;
            const float bias5 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 5)) : 0.0f;
            const float bias6 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 6)) : 0.0f;
            const float bias7 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 7)) : 0.0f;
            float acc00 = bias0;
            float acc01 = bias1;
            float acc02 = bias2;
            float acc03 = bias3;
            float acc04 = bias4;
            float acc05 = bias5;
            float acc06 = bias6;
            float acc07 = bias7;
            float acc10 = bias0;
            float acc11 = bias1;
            float acc12 = bias2;
            float acc13 = bias3;
            float acc14 = bias4;
            float acc15 = bias5;
            float acc16 = bias6;
            float acc17 = bias7;
            float acc20 = bias0;
            float acc21 = bias1;
            float acc22 = bias2;
            float acc23 = bias3;
            float acc24 = bias4;
            float acc25 = bias5;
            float acc26 = bias6;
            float acc27 = bias7;
            float acc30 = bias0;
            float acc31 = bias1;
            float acc32 = bias2;
            float acc33 = bias3;
            float acc34 = bias4;
            float acc35 = bias5;
            float acc36 = bias6;
            float acc37 = bias7;

            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base = group * out_features + n_base;
                const float scale0 = static_cast<float>(scales_gm_.GetValue(scale_base));
                const float scale1 = static_cast<float>(scales_gm_.GetValue(scale_base + 1));
                const float scale2 = static_cast<float>(scales_gm_.GetValue(scale_base + 2));
                const float scale3 = static_cast<float>(scales_gm_.GetValue(scale_base + 3));
                const float scale4 = static_cast<float>(scales_gm_.GetValue(scale_base + 4));
                const float scale5 = static_cast<float>(scales_gm_.GetValue(scale_base + 5));
                const float scale6 = static_cast<float>(scales_gm_.GetValue(scale_base + 6));
                const float scale7 = static_cast<float>(scales_gm_.GetValue(scale_base + 7));
                const float offset0 = static_cast<float>(offsets_gm_.GetValue(scale_base));
                const float offset1 = static_cast<float>(offsets_gm_.GetValue(scale_base + 1));
                const float offset2 = static_cast<float>(offsets_gm_.GetValue(scale_base + 2));
                const float offset3 = static_cast<float>(offsets_gm_.GetValue(scale_base + 3));
                const float offset4 = static_cast<float>(offsets_gm_.GetValue(scale_base + 4));
                const float offset5 = static_cast<float>(offsets_gm_.GetValue(scale_base + 5));
                const float offset6 = static_cast<float>(offsets_gm_.GetValue(scale_base + 6));
                const float offset7 = static_cast<float>(offsets_gm_.GetValue(scale_base + 7));

                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                    const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
                    const float x_value2 = static_cast<float>(x_gm_.GetValue(x_offset2 + k));
                    const float x_value3 = static_cast<float>(x_gm_.GetValue(x_offset3 + k));
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const float deq0 = DequantLane(word, 0, scale0, offset0);
                    const float deq1 = DequantLane(word, 1, scale1, offset1);
                    const float deq2 = DequantLane(word, 2, scale2, offset2);
                    const float deq3 = DequantLane(word, 3, scale3, offset3);
                    const float deq4 = DequantLane(word, 4, scale4, offset4);
                    const float deq5 = DequantLane(word, 5, scale5, offset5);
                    const float deq6 = DequantLane(word, 6, scale6, offset6);
                    const float deq7 = DequantLane(word, 7, scale7, offset7);
                    acc00 += x_value0 * deq0;
                    acc01 += x_value0 * deq1;
                    acc02 += x_value0 * deq2;
                    acc03 += x_value0 * deq3;
                    acc04 += x_value0 * deq4;
                    acc05 += x_value0 * deq5;
                    acc06 += x_value0 * deq6;
                    acc07 += x_value0 * deq7;
                    acc10 += x_value1 * deq0;
                    acc11 += x_value1 * deq1;
                    acc12 += x_value1 * deq2;
                    acc13 += x_value1 * deq3;
                    acc14 += x_value1 * deq4;
                    acc15 += x_value1 * deq5;
                    acc16 += x_value1 * deq6;
                    acc17 += x_value1 * deq7;
                    acc20 += x_value2 * deq0;
                    acc21 += x_value2 * deq1;
                    acc22 += x_value2 * deq2;
                    acc23 += x_value2 * deq3;
                    acc24 += x_value2 * deq4;
                    acc25 += x_value2 * deq5;
                    acc26 += x_value2 * deq6;
                    acc27 += x_value2 * deq7;
                    acc30 += x_value3 * deq0;
                    acc31 += x_value3 * deq1;
                    acc32 += x_value3 * deq2;
                    acc33 += x_value3 * deq3;
                    acc34 += x_value3 * deq4;
                    acc35 += x_value3 * deq5;
                    acc36 += x_value3 * deq6;
                    acc37 += x_value3 * deq7;
                }
            }

            StoreRow(row_offset0 + n_base, acc00, acc01, acc02, acc03, acc04, acc05, acc06, acc07);
            StoreRow(row_offset1 + n_base, acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17);
            StoreRow(row_offset2 + n_base, acc20, acc21, acc22, acc23, acc24, acc25, acc26, acc27);
            StoreRow(row_offset3 + n_base, acc30, acc31, acc32, acc33, acc34, acc35, acc36, acc37);
        }

#undef KOMODO_ACCUM8
#undef KOMODO_INIT_ACC8
    }

    __aicore__ inline void ProcessRowPair(
        uint32_t m,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t has_bias,
        uint32_t packed_begin,
        uint32_t packed_end)
    {
        const uint32_t packed_stride = out_features >> 3;
        const uint32_t row_offset0 = m * out_features;
        const uint32_t row_offset1 = row_offset0 + out_features;
        const uint32_t x_offset0 = m * in_features;
        const uint32_t x_offset1 = x_offset0 + in_features;
        const uint32_t groups = group_size == 0 ? 1 : in_features / group_size;
        uint32_t packed_col = packed_begin;
        for (; packed_col + 1 < packed_end; packed_col += 2) {
            const uint32_t n_base0 = packed_col << 3;
            const uint32_t n_base1 = n_base0 + 8;
            const float bias00 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0)) : 0.0f;
            const float bias01 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 1)) : 0.0f;
            const float bias02 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 2)) : 0.0f;
            const float bias03 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 3)) : 0.0f;
            const float bias04 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 4)) : 0.0f;
            const float bias05 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 5)) : 0.0f;
            const float bias06 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 6)) : 0.0f;
            const float bias07 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base0 + 7)) : 0.0f;
            const float bias10 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1)) : 0.0f;
            const float bias11 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 1)) : 0.0f;
            const float bias12 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 2)) : 0.0f;
            const float bias13 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 3)) : 0.0f;
            const float bias14 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 4)) : 0.0f;
            const float bias15 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 5)) : 0.0f;
            const float bias16 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 6)) : 0.0f;
            const float bias17 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base1 + 7)) : 0.0f;
            float acc000 = bias00;
            float acc001 = bias01;
            float acc002 = bias02;
            float acc003 = bias03;
            float acc004 = bias04;
            float acc005 = bias05;
            float acc006 = bias06;
            float acc007 = bias07;
            float acc010 = bias10;
            float acc011 = bias11;
            float acc012 = bias12;
            float acc013 = bias13;
            float acc014 = bias14;
            float acc015 = bias15;
            float acc016 = bias16;
            float acc017 = bias17;
            float acc100 = bias00;
            float acc101 = bias01;
            float acc102 = bias02;
            float acc103 = bias03;
            float acc104 = bias04;
            float acc105 = bias05;
            float acc106 = bias06;
            float acc107 = bias07;
            float acc110 = bias10;
            float acc111 = bias11;
            float acc112 = bias12;
            float acc113 = bias13;
            float acc114 = bias14;
            float acc115 = bias15;
            float acc116 = bias16;
            float acc117 = bias17;

            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base0 = group * out_features + n_base0;
                const uint32_t scale_base1 = scale_base0 + 8;
                const float scale00 = static_cast<float>(scales_gm_.GetValue(scale_base0));
                const float scale01 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 1));
                const float scale02 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 2));
                const float scale03 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 3));
                const float scale04 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 4));
                const float scale05 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 5));
                const float scale06 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 6));
                const float scale07 = static_cast<float>(scales_gm_.GetValue(scale_base0 + 7));
                const float scale10 = static_cast<float>(scales_gm_.GetValue(scale_base1));
                const float scale11 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 1));
                const float scale12 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 2));
                const float scale13 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 3));
                const float scale14 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 4));
                const float scale15 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 5));
                const float scale16 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 6));
                const float scale17 = static_cast<float>(scales_gm_.GetValue(scale_base1 + 7));
                const float offset00 = static_cast<float>(offsets_gm_.GetValue(scale_base0));
                const float offset01 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 1));
                const float offset02 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 2));
                const float offset03 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 3));
                const float offset04 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 4));
                const float offset05 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 5));
                const float offset06 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 6));
                const float offset07 = static_cast<float>(offsets_gm_.GetValue(scale_base0 + 7));
                const float offset10 = static_cast<float>(offsets_gm_.GetValue(scale_base1));
                const float offset11 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 1));
                const float offset12 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 2));
                const float offset13 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 3));
                const float offset14 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 4));
                const float offset15 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 5));
                const float offset16 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 6));
                const float offset17 = static_cast<float>(offsets_gm_.GetValue(scale_base1 + 7));

                float x_sum0 = 0.0f;
                float x_sum1 = 0.0f;
                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                    const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
                    x_sum0 += x_value0;
                    x_sum1 += x_value1;
                    const uint32_t word0 =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t word1 =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col + 1));
                    const float deq00 = DequantLaneNoOffset(word0, 0, scale00);
                    const float deq01 = DequantLaneNoOffset(word0, 1, scale01);
                    const float deq02 = DequantLaneNoOffset(word0, 2, scale02);
                    const float deq03 = DequantLaneNoOffset(word0, 3, scale03);
                    const float deq04 = DequantLaneNoOffset(word0, 4, scale04);
                    const float deq05 = DequantLaneNoOffset(word0, 5, scale05);
                    const float deq06 = DequantLaneNoOffset(word0, 6, scale06);
                    const float deq07 = DequantLaneNoOffset(word0, 7, scale07);
                    const float deq10 = DequantLaneNoOffset(word1, 0, scale10);
                    const float deq11 = DequantLaneNoOffset(word1, 1, scale11);
                    const float deq12 = DequantLaneNoOffset(word1, 2, scale12);
                    const float deq13 = DequantLaneNoOffset(word1, 3, scale13);
                    const float deq14 = DequantLaneNoOffset(word1, 4, scale14);
                    const float deq15 = DequantLaneNoOffset(word1, 5, scale15);
                    const float deq16 = DequantLaneNoOffset(word1, 6, scale16);
                    const float deq17 = DequantLaneNoOffset(word1, 7, scale17);
                    acc000 += x_value0 * deq00;
                    acc001 += x_value0 * deq01;
                    acc002 += x_value0 * deq02;
                    acc003 += x_value0 * deq03;
                    acc004 += x_value0 * deq04;
                    acc005 += x_value0 * deq05;
                    acc006 += x_value0 * deq06;
                    acc007 += x_value0 * deq07;
                    acc010 += x_value0 * deq10;
                    acc011 += x_value0 * deq11;
                    acc012 += x_value0 * deq12;
                    acc013 += x_value0 * deq13;
                    acc014 += x_value0 * deq14;
                    acc015 += x_value0 * deq15;
                    acc016 += x_value0 * deq16;
                    acc017 += x_value0 * deq17;
                    acc100 += x_value1 * deq00;
                    acc101 += x_value1 * deq01;
                    acc102 += x_value1 * deq02;
                    acc103 += x_value1 * deq03;
                    acc104 += x_value1 * deq04;
                    acc105 += x_value1 * deq05;
                    acc106 += x_value1 * deq06;
                    acc107 += x_value1 * deq07;
                    acc110 += x_value1 * deq10;
                    acc111 += x_value1 * deq11;
                    acc112 += x_value1 * deq12;
                    acc113 += x_value1 * deq13;
                    acc114 += x_value1 * deq14;
                    acc115 += x_value1 * deq15;
                    acc116 += x_value1 * deq16;
                    acc117 += x_value1 * deq17;
                }
                acc000 += x_sum0 * offset00 * scale00;
                acc001 += x_sum0 * offset01 * scale01;
                acc002 += x_sum0 * offset02 * scale02;
                acc003 += x_sum0 * offset03 * scale03;
                acc004 += x_sum0 * offset04 * scale04;
                acc005 += x_sum0 * offset05 * scale05;
                acc006 += x_sum0 * offset06 * scale06;
                acc007 += x_sum0 * offset07 * scale07;
                acc010 += x_sum0 * offset10 * scale10;
                acc011 += x_sum0 * offset11 * scale11;
                acc012 += x_sum0 * offset12 * scale12;
                acc013 += x_sum0 * offset13 * scale13;
                acc014 += x_sum0 * offset14 * scale14;
                acc015 += x_sum0 * offset15 * scale15;
                acc016 += x_sum0 * offset16 * scale16;
                acc017 += x_sum0 * offset17 * scale17;
                acc100 += x_sum1 * offset00 * scale00;
                acc101 += x_sum1 * offset01 * scale01;
                acc102 += x_sum1 * offset02 * scale02;
                acc103 += x_sum1 * offset03 * scale03;
                acc104 += x_sum1 * offset04 * scale04;
                acc105 += x_sum1 * offset05 * scale05;
                acc106 += x_sum1 * offset06 * scale06;
                acc107 += x_sum1 * offset07 * scale07;
                acc110 += x_sum1 * offset10 * scale10;
                acc111 += x_sum1 * offset11 * scale11;
                acc112 += x_sum1 * offset12 * scale12;
                acc113 += x_sum1 * offset13 * scale13;
                acc114 += x_sum1 * offset14 * scale14;
                acc115 += x_sum1 * offset15 * scale15;
                acc116 += x_sum1 * offset16 * scale16;
                acc117 += x_sum1 * offset17 * scale17;
            }

            StoreRow(row_offset0 + n_base0, acc000, acc001, acc002, acc003, acc004, acc005, acc006, acc007);
            StoreRow(row_offset0 + n_base1, acc010, acc011, acc012, acc013, acc014, acc015, acc016, acc017);
            StoreRow(row_offset1 + n_base0, acc100, acc101, acc102, acc103, acc104, acc105, acc106, acc107);
            StoreRow(row_offset1 + n_base1, acc110, acc111, acc112, acc113, acc114, acc115, acc116, acc117);
        }
        for (; packed_col < packed_end; ++packed_col) {
            const uint32_t n_base = packed_col << 3;
            const float bias0 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base)) : 0.0f;
            const float bias1 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 1)) : 0.0f;
            const float bias2 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 2)) : 0.0f;
            const float bias3 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 3)) : 0.0f;
            const float bias4 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 4)) : 0.0f;
            const float bias5 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 5)) : 0.0f;
            const float bias6 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 6)) : 0.0f;
            const float bias7 = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n_base + 7)) : 0.0f;
            float acc00 = bias0;
            float acc01 = bias1;
            float acc02 = bias2;
            float acc03 = bias3;
            float acc04 = bias4;
            float acc05 = bias5;
            float acc06 = bias6;
            float acc07 = bias7;
            float acc10 = bias0;
            float acc11 = bias1;
            float acc12 = bias2;
            float acc13 = bias3;
            float acc14 = bias4;
            float acc15 = bias5;
            float acc16 = bias6;
            float acc17 = bias7;

            for (uint32_t group = 0; group < groups; ++group) {
                const uint32_t k_begin = group_size == 0 ? 0 : group * group_size;
                const uint32_t k_end = group_size == 0 ? in_features : k_begin + group_size;
                const uint32_t scale_base = group * out_features + n_base;
                const float scale0 = static_cast<float>(scales_gm_.GetValue(scale_base));
                const float scale1 = static_cast<float>(scales_gm_.GetValue(scale_base + 1));
                const float scale2 = static_cast<float>(scales_gm_.GetValue(scale_base + 2));
                const float scale3 = static_cast<float>(scales_gm_.GetValue(scale_base + 3));
                const float scale4 = static_cast<float>(scales_gm_.GetValue(scale_base + 4));
                const float scale5 = static_cast<float>(scales_gm_.GetValue(scale_base + 5));
                const float scale6 = static_cast<float>(scales_gm_.GetValue(scale_base + 6));
                const float scale7 = static_cast<float>(scales_gm_.GetValue(scale_base + 7));
                const float offset0 = static_cast<float>(offsets_gm_.GetValue(scale_base));
                const float offset1 = static_cast<float>(offsets_gm_.GetValue(scale_base + 1));
                const float offset2 = static_cast<float>(offsets_gm_.GetValue(scale_base + 2));
                const float offset3 = static_cast<float>(offsets_gm_.GetValue(scale_base + 3));
                const float offset4 = static_cast<float>(offsets_gm_.GetValue(scale_base + 4));
                const float offset5 = static_cast<float>(offsets_gm_.GetValue(scale_base + 5));
                const float offset6 = static_cast<float>(offsets_gm_.GetValue(scale_base + 6));
                const float offset7 = static_cast<float>(offsets_gm_.GetValue(scale_base + 7));

                float x_sum0 = 0.0f;
                float x_sum1 = 0.0f;
                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                    const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
                    x_sum0 += x_value0;
                    x_sum1 += x_value1;
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const float deq0 = DequantLaneNoOffset(word, 0, scale0);
                    const float deq1 = DequantLaneNoOffset(word, 1, scale1);
                    const float deq2 = DequantLaneNoOffset(word, 2, scale2);
                    const float deq3 = DequantLaneNoOffset(word, 3, scale3);
                    const float deq4 = DequantLaneNoOffset(word, 4, scale4);
                    const float deq5 = DequantLaneNoOffset(word, 5, scale5);
                    const float deq6 = DequantLaneNoOffset(word, 6, scale6);
                    const float deq7 = DequantLaneNoOffset(word, 7, scale7);
                    acc00 += x_value0 * deq0;
                    acc01 += x_value0 * deq1;
                    acc02 += x_value0 * deq2;
                    acc03 += x_value0 * deq3;
                    acc04 += x_value0 * deq4;
                    acc05 += x_value0 * deq5;
                    acc06 += x_value0 * deq6;
                    acc07 += x_value0 * deq7;
                    acc10 += x_value1 * deq0;
                    acc11 += x_value1 * deq1;
                    acc12 += x_value1 * deq2;
                    acc13 += x_value1 * deq3;
                    acc14 += x_value1 * deq4;
                    acc15 += x_value1 * deq5;
                    acc16 += x_value1 * deq6;
                    acc17 += x_value1 * deq7;
                }
                acc00 += x_sum0 * offset0 * scale0;
                acc01 += x_sum0 * offset1 * scale1;
                acc02 += x_sum0 * offset2 * scale2;
                acc03 += x_sum0 * offset3 * scale3;
                acc04 += x_sum0 * offset4 * scale4;
                acc05 += x_sum0 * offset5 * scale5;
                acc06 += x_sum0 * offset6 * scale6;
                acc07 += x_sum0 * offset7 * scale7;
                acc10 += x_sum1 * offset0 * scale0;
                acc11 += x_sum1 * offset1 * scale1;
                acc12 += x_sum1 * offset2 * scale2;
                acc13 += x_sum1 * offset3 * scale3;
                acc14 += x_sum1 * offset4 * scale4;
                acc15 += x_sum1 * offset5 * scale5;
                acc16 += x_sum1 * offset6 * scale6;
                acc17 += x_sum1 * offset7 * scale7;
            }

            StoreRow(row_offset0 + n_base, acc00, acc01, acc02, acc03, acc04, acc05, acc06, acc07);
            StoreRow(row_offset1 + n_base, acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17);
        }
    }

    __aicore__ inline void StoreRow(
        uint32_t y_base,
        float acc0,
        float acc1,
        float acc2,
        float acc3,
        float acc4,
        float acc5,
        float acc6,
        float acc7)
    {
        y_gm_.SetValue(y_base, static_cast<half>(acc0));
        y_gm_.SetValue(y_base + 1, static_cast<half>(acc1));
        y_gm_.SetValue(y_base + 2, static_cast<half>(acc2));
        y_gm_.SetValue(y_base + 3, static_cast<half>(acc3));
        y_gm_.SetValue(y_base + 4, static_cast<half>(acc4));
        y_gm_.SetValue(y_base + 5, static_cast<half>(acc5));
        y_gm_.SetValue(y_base + 6, static_cast<half>(acc6));
        y_gm_.SetValue(y_base + 7, static_cast<half>(acc7));
    }

    __aicore__ inline float OffsetValue(uint32_t offset, uint32_t zero_offsets)
    {
        return zero_offsets != 0 ? 0.0f : static_cast<float>(offsets_gm_.GetValue(offset));
    }

    __aicore__ inline float DequantLane(uint32_t word, uint32_t lane, float scale, float offset)
    {
        const uint32_t shift = lane << 2;
        const int32_t raw = static_cast<int32_t>((word >> shift) & 0xFU);
        const int32_t signed_w = (raw ^ 0x8) - 0x8;
        return (static_cast<float>(signed_w) + offset) * scale;
    }

    __aicore__ inline float DequantLaneNoOffset(uint32_t word, uint32_t lane, float scale)
    {
        const uint32_t shift = lane << 2;
        const int32_t raw = static_cast<int32_t>((word >> shift) & 0xFU);
        const int32_t signed_w = (raw ^ 0x8) - 0x8;
        return static_cast<float>(signed_w) * scale;
    }

    GlobalTensor<half> x_gm_;
    GlobalTensor<int32_t> packed_weight_gm_;
    GlobalTensor<half> scales_gm_;
    GlobalTensor<half> offsets_gm_;
    GlobalTensor<half> bias_gm_;
    GlobalTensor<half> y_gm_;
    const KomodoCannW4A16MatmulTilingData* tiling_;
};
}  // namespace

extern "C" __global__ __aicore__ void komodo_cann_w4_a16_matmul(
    GM_ADDR x,
    GM_ADDR packed_weight,
    GM_ADDR scales,
    GM_ADDR offsets,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    KomodoCannW4A16ScalarKernel op;
    op.Init(x, packed_weight, scales, offsets, bias, y, &tiling_data);
    op.Process();
}
