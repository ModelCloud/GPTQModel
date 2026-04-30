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
        for (uint32_t m = 0; m < rows; ++m) {
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

                for (uint32_t k = 0; k < in_features; ++k) {
                    const float x_value = static_cast<float>(x_gm_.GetValue(x_offset + k));
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t group = group_size == 0 ? 0 : k / group_size;
                    const uint32_t scale_base = group * out_features + n_base;
                    acc0 += x_value * DequantLane(word, 0, scale_base);
                    acc1 += x_value * DequantLane(word, 1, scale_base + 1);
                    acc2 += x_value * DequantLane(word, 2, scale_base + 2);
                    acc3 += x_value * DequantLane(word, 3, scale_base + 3);
                    acc4 += x_value * DequantLane(word, 4, scale_base + 4);
                    acc5 += x_value * DequantLane(word, 5, scale_base + 5);
                    acc6 += x_value * DequantLane(word, 6, scale_base + 6);
                    acc7 += x_value * DequantLane(word, 7, scale_base + 7);
                }

                const uint32_t y_base = row_offset + n_base;
                y_gm_.SetValue(y_base, static_cast<half>(acc0));
                y_gm_.SetValue(y_base + 1, static_cast<half>(acc1));
                y_gm_.SetValue(y_base + 2, static_cast<half>(acc2));
                y_gm_.SetValue(y_base + 3, static_cast<half>(acc3));
                y_gm_.SetValue(y_base + 4, static_cast<half>(acc4));
                y_gm_.SetValue(y_base + 5, static_cast<half>(acc5));
                y_gm_.SetValue(y_base + 6, static_cast<half>(acc6));
                y_gm_.SetValue(y_base + 7, static_cast<half>(acc7));
            }
        }
    }

private:
    __aicore__ inline float DequantLane(uint32_t word, uint32_t lane, uint32_t scale_idx)
    {
        const uint32_t shift = lane << 2;
        const int32_t raw = static_cast<int32_t>((word >> shift) & 0xFU);
        const int32_t signed_w = raw >= 8 ? raw - 16 : raw;
        const float scale = static_cast<float>(scales_gm_.GetValue(scale_idx));
        const float offset = static_cast<float>(offsets_gm_.GetValue(scale_idx));
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
