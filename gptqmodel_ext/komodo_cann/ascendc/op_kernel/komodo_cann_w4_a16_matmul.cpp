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
        pipe_.InitBuffer(dequant_tile_buf_, kDequantTileValues * sizeof(half));
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        const uint32_t in_features = tiling_->in_features;
        const uint32_t out_features = tiling_->out_features;
        const uint32_t group_size = tiling_->group_size;
        const uint32_t has_bias = tiling_->has_bias;
        const uint32_t total_outputs = tiling_->total_outputs;
        const uint32_t core_idx = GetBlockIdx();
        if (core_idx != 0) {
            return;
        }
        constexpr uint32_t block_dim = 1;

        for (uint32_t out_index = core_idx; out_index < total_outputs; out_index += block_dim) {
            const uint32_t m = out_index / out_features;
            const uint32_t n = out_index - m * out_features;
            float acc = has_bias != 0 ? static_cast<float>(bias_gm_.GetValue(n)) : 0.0f;

            LocalTensor<half> dequant_tile = dequant_tile_buf_.Get<half>();
            for (uint32_t k_tile = 0; k_tile < in_features; k_tile += kDequantTileValues) {
                const uint32_t tile_len =
                    k_tile + kDequantTileValues <= in_features ? kDequantTileValues : in_features - k_tile;
                StageDequantTile(dequant_tile, k_tile, tile_len, n, out_features, group_size);
                for (uint32_t k_inner = 0; k_inner < tile_len; ++k_inner) {
                    const uint32_t k = k_tile + k_inner;
                    const float x_value = static_cast<float>(x_gm_.GetValue(m * in_features + k));
                    const float dequant_w = static_cast<float>(dequant_tile.GetValue(k_inner));
                    acc += x_value * dequant_w;
                }
            }

            y_gm_.SetValue(out_index, static_cast<half>(acc));
        }
    }

private:
    __aicore__ inline void StageDequantTile(
        LocalTensor<half>& dequant_tile,
        uint32_t k_tile,
        uint32_t tile_len,
        uint32_t n,
        uint32_t out_features,
        uint32_t group_size)
    {
        const uint32_t packed_col = n >> 3;
        const uint32_t packed_stride = out_features >> 3;
        const uint32_t shift = (n & 7U) << 2;
        for (uint32_t k_inner = 0; k_inner < tile_len; ++k_inner) {
            const uint32_t k = k_tile + k_inner;
            const uint32_t packed_idx = k * packed_stride + packed_col;
            const uint32_t word = static_cast<uint32_t>(packed_weight_gm_.GetValue(packed_idx));
            const int32_t raw = static_cast<int32_t>((word >> shift) & 0xFU);
            const int32_t signed_w = raw >= 8 ? raw - 16 : raw;
            const uint32_t group = group_size == 0 ? 0 : k / group_size;
            const uint32_t scale_idx = group * out_features + n;
            const float scale = static_cast<float>(scales_gm_.GetValue(scale_idx));
            const float offset = static_cast<float>(offsets_gm_.GetValue(scale_idx));
            const float dequant_w = (static_cast<float>(signed_w) + offset) * scale;
            dequant_tile.SetValue(k_inner, static_cast<half>(dequant_w));
        }
    }

    static constexpr uint32_t kDequantTileValues = 64;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> dequant_tile_buf_;
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
