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
        const uint32_t core_idx = GetBlockIdx();
        if (core_idx != 0) {
            return;
        }

        const uint32_t packed_stride = out_features >> 3;
        uint32_t m = 0;
        for (; m + 1 < rows; m += 2) {
            ProcessRowPair(m, in_features, out_features, group_size, has_bias, packed_stride);
        }
        if (m < rows) {
            ProcessSingleRow(m, in_features, out_features, group_size, has_bias, packed_stride);
        }
    }

private:
    __aicore__ inline void ProcessSingleRow(
        uint32_t m,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t has_bias,
        uint32_t packed_stride)
    {
        const uint32_t row_offset = m * out_features;
        const uint32_t x_offset = m * in_features;
        for (uint32_t packed_col = 0; packed_col < packed_stride; ++packed_col) {
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

                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value = static_cast<float>(x_gm_.GetValue(x_offset + k));
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    acc0 += x_value * DequantLane(word, 0, scale0, offset0);
                    acc1 += x_value * DequantLane(word, 1, scale1, offset1);
                    acc2 += x_value * DequantLane(word, 2, scale2, offset2);
                    acc3 += x_value * DequantLane(word, 3, scale3, offset3);
                    acc4 += x_value * DequantLane(word, 4, scale4, offset4);
                    acc5 += x_value * DequantLane(word, 5, scale5, offset5);
                    acc6 += x_value * DequantLane(word, 6, scale6, offset6);
                    acc7 += x_value * DequantLane(word, 7, scale7, offset7);
                }
            }

            StoreRow(row_offset + n_base, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        }
    }

    __aicore__ inline void ProcessRowPair(
        uint32_t m,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t has_bias,
        uint32_t packed_stride)
    {
        const uint32_t row_offset0 = m * out_features;
        const uint32_t row_offset1 = row_offset0 + out_features;
        const uint32_t x_offset0 = m * in_features;
        const uint32_t x_offset1 = x_offset0 + in_features;
        for (uint32_t packed_col = 0; packed_col < packed_stride; ++packed_col) {
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

                for (uint32_t k = k_begin; k < k_end; ++k) {
                    const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                    const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
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
                }
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

    __aicore__ inline float DequantLane(uint32_t word, uint32_t lane, float scale, float offset)
    {
        const uint32_t shift = lane << 2;
        const int32_t raw = static_cast<int32_t>((word >> shift) & 0xFU);
        const int32_t signed_w = raw >= 8 ? raw - 16 : raw;
        return (static_cast<float>(signed_w) + offset) * scale;
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
    GET_TILING_DATA(tiling_data, tiling);
    KomodoCannW4A16ScalarKernel op;
    op.Init(x, packed_weight, scales, offsets, bias, y, &tiling_data);
    op.Process();
}
