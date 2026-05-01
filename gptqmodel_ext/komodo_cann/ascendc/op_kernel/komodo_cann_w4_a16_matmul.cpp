#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if defined(KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT)
#include "c_api/asc_simd.h"
#endif
#include "komodo_cann_w4_a16_matmul_tiling_key.h"

using namespace AscendC;
using namespace matmul;

#if defined(KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT)
using KomodoCannCapiInt4 = ::int4b_t;
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH) && !defined(KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER) && \
    !defined(KOMODO_CANN_EXPERIMENTAL_MIXED_AIV_BASELINE)
#error "KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH requires KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER) && defined(KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER)
#error "KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER and KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER are mutually exclusive"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF) && \
    (!defined(KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER) || !defined(KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH) || \
     !defined(KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT))
#error "KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF requires staged dequant, TSCM consumer, and mixed launch"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT) && \
    (!defined(KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF) || \
     !defined(KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT))
#error "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT requires CANN9 vector dequant and TSCM runtime handoff"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK) && \
    !defined(KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT)
#error "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK requires TSCM direct dequant"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF) && \
    (!defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER) || !defined(KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH) || \
     !defined(KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT) || !defined(KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT))
#error "KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF requires staged dequant, CANN9 vector dequant, VECOUT consumer, and mixed launch"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A) && !defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
#error "KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A requires VecOut runtime handoff"
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT) || \
    defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
#define KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_DEQUANT 1
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK) || \
    defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
#define KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_MULTIK 1
#endif

namespace {
#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
constexpr uint32_t kKernelModeStagedDequant = 1;
#endif

__aicore__ inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

#ifdef KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER
class KomodoCannW4A16CubeConsumerProbe {
public:
#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
    using AType = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
#else
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
#endif
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER
    using BType = MatmulType<TPosition::TSCM, CubeFormat::NZ, half, true>;
#elif defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER)
    using BType = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
#else
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
#endif
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    Matmul<AType, BType, CType, BiasType> mm;

#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER
    __aicore__ inline void SetTensorBLocalProbe(const LocalTensor<half>& b_tile)
    {
        mm.SetTensorB(b_tile);
    }
#endif

#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER
    __aicore__ inline bool InitTscmBTile(TPipe& pipe, const KomodoCannW4A16MatmulTilingData* tiling)
    {
        const uint32_t base_k = tiling->base_k;
        const uint32_t base_n = tiling->base_n;
        if (base_k == 0 || base_n == 0) {
            return false;
        }
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK
        constexpr uint32_t kTscmSlots = 2;
#else
        constexpr uint32_t kTscmSlots = 1;
#endif
        tscm_ready_ = pipe.InitBuffer(b_tscm_, kTscmSlots, base_k * base_n * sizeof(half));
#ifdef KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_DEQUANT
        direct_dequant_ready_ = pipe.InitBuffer(b_ub_, base_k * base_n * sizeof(half));
#endif
        return tscm_ready_;
    }

    __aicore__ inline LocalTensor<half> LoadStagedBTileToTscm(
        GlobalTensor<half>& staged_weight,
        uint32_t stage_offset,
        const KomodoCannW4A16MatmulTilingData* tiling)
    {
        b_tscm_local_ = b_tscm_.AllocTensor<half>();
        Nd2NzParams trans_param = {
            1,
            static_cast<uint16_t>(tiling->base_k),
            static_cast<uint16_t>(tiling->base_n),
            0,
            static_cast<uint16_t>(tiling->base_n),
            static_cast<uint16_t>(CeilDivU32(tiling->base_k, 16) * 16),
            1,
            0,
        };
        DataCopy(b_tscm_local_, staged_weight[stage_offset], trans_param);
        b_tscm_.EnQue(b_tscm_local_);
        b_tscm_.DeQue();
        return b_tscm_local_;
    }

#ifdef KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_DEQUANT
    __aicore__ inline LocalTensor<half> GetDirectDequantTile(const KomodoCannW4A16MatmulTilingData* tiling)
    {
        return b_ub_.Get<half>(tiling->base_k * tiling->base_n);
    }

#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT
    __aicore__ inline LocalTensor<half> LoadDirectBTileToTscm(
        const LocalTensor<half>& b_tile,
        const KomodoCannW4A16MatmulTilingData* tiling)
    {
        b_tscm_local_ = b_tscm_.AllocTensor<half>();
        Nd2NzParams trans_param = {
            1,
            static_cast<uint16_t>(tiling->base_k),
            static_cast<uint16_t>(tiling->base_n),
            0,
            static_cast<uint16_t>(tiling->base_n),
            static_cast<uint16_t>(CeilDivU32(tiling->base_k, 16) * 16),
            1,
            0,
        };
        PipeBarrier<PIPE_ALL>();
        DataCopy(b_tscm_local_, b_tile, trans_param);
        b_tscm_.EnQue(b_tscm_local_);
        b_tscm_.DeQue();
        return b_tscm_local_;
    }
#endif

    __aicore__ inline bool DirectDequantReady() const
    {
        return direct_dequant_ready_;
    }
#endif

    __aicore__ inline void SetTensorBTscmProbe(const LocalTensor<half>& b_tile)
    {
        mm.SetTensorB(b_tile, true);
    }

    __aicore__ inline void FreeTscmBTile()
    {
        b_tscm_.FreeTensor(b_tscm_local_);
    }

#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK
    __aicore__ inline void FreeTscmBTile(LocalTensor<half>& b_tile)
    {
        b_tscm_.FreeTensor(b_tile);
    }
#endif

    __aicore__ inline bool TscmReady() const
    {
        return tscm_ready_;
    }

private:
    TSCM<TPosition::GM> b_tscm_;
    LocalTensor<half> b_tscm_local_;
    bool tscm_ready_ = false;
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT
    TBuf<TPosition::VECCALC> b_ub_;
    bool direct_dequant_ready_ = false;
#endif
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF) && \
    !defined(KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER)
    __aicore__ inline bool InitVecoutBTile(TPipe& pipe, const KomodoCannW4A16MatmulTilingData* tiling)
    {
        const uint32_t base_k = tiling->base_k;
        const uint32_t base_n = tiling->base_n;
        if (base_k == 0 || base_n == 0) {
            return false;
        }
        direct_dequant_ready_ = pipe.InitBuffer(b_ub_, base_k * base_n * sizeof(half));
#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
        local_a_ready_ = pipe.InitBuffer(a_ub_, tiling->base_m * base_k * sizeof(half));
        packed_b_ready_ = pipe.InitBuffer(packed_b_ub_, base_k * (base_n >> 3) * sizeof(int32_t));
        return direct_dequant_ready_ && local_a_ready_ && packed_b_ready_;
#else
        return direct_dequant_ready_;
#endif
    }

    __aicore__ inline LocalTensor<half> GetDirectDequantTile(const KomodoCannW4A16MatmulTilingData* tiling)
    {
        return b_ub_.Get<half>(tiling->base_k * tiling->base_n);
    }

#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
    __aicore__ inline LocalTensor<half> GetLocalATile(const KomodoCannW4A16MatmulTilingData* tiling)
    {
        return a_ub_.Get<half>(tiling->base_m * tiling->base_k);
    }

    __aicore__ inline LocalTensor<int32_t> GetPackedBTile(const KomodoCannW4A16MatmulTilingData* tiling)
    {
        return packed_b_ub_.Get<int32_t>(tiling->base_k * (tiling->base_n >> 3));
    }

    __aicore__ inline bool LocalAReady() const
    {
        return local_a_ready_ && packed_b_ready_;
    }
#endif

    __aicore__ inline bool DirectDequantReady() const
    {
        return direct_dequant_ready_;
    }

private:
#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
    TBuf<TPosition::VECOUT> a_ub_;
    TBuf<TPosition::VECCALC> packed_b_ub_;
    bool local_a_ready_ = false;
    bool packed_b_ready_ = false;
#endif
    TBuf<TPosition::VECOUT> b_ub_;
    bool direct_dequant_ready_ = false;
#endif
};

__aicore__ inline TCubeTiling MakeCubeConsumerTiling(const KomodoCannW4A16MatmulTilingData* tiling)
{
    TCubeTiling cube_tiling;
    cube_tiling.usedCoreNum = static_cast<int32_t>(tiling->block_dim);
    cube_tiling.M = static_cast<int32_t>(tiling->rows);
    cube_tiling.N = static_cast<int32_t>(tiling->out_features);
#ifdef KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_MULTIK
    const int32_t cube_k = static_cast<int32_t>(tiling->base_k);
#else
    const int32_t cube_k = static_cast<int32_t>(tiling->in_features);
#endif
    cube_tiling.Ka = cube_k;
    cube_tiling.Kb = cube_k;
    cube_tiling.singleCoreM = static_cast<int32_t>(tiling->base_m);
    cube_tiling.singleCoreN = static_cast<int32_t>(tiling->base_n);
    cube_tiling.singleCoreK = static_cast<int32_t>(tiling->base_k);
    cube_tiling.baseM = static_cast<int32_t>(tiling->base_m);
    cube_tiling.baseN = static_cast<int32_t>(tiling->base_n);
    cube_tiling.baseK = static_cast<int32_t>(tiling->base_k);
    cube_tiling.depthA1 = 1;
    cube_tiling.depthB1 = 1;
    cube_tiling.stepM = 1;
    cube_tiling.stepN = 1;
    cube_tiling.isBias = tiling->has_bias != 0 ? 1 : 0;
    cube_tiling.stepKa = 1;
    cube_tiling.stepKb = 1;
    cube_tiling.dbL0A = 1;
    cube_tiling.dbL0B = 1;
    cube_tiling.dbL0C = 1;
    cube_tiling.BatchNum = 1;
    return cube_tiling;
}
#endif

class KomodoCannW4A16ScalarKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR packed_weight,
        GM_ADDR scales,
        GM_ADDR offsets,
        GM_ADDR bias,
        GM_ADDR y,
        GM_ADDR user_workspace,
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
#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
        if (tiling->kernel_mode == kKernelModeStagedDequant && tiling->staging_workspace_bytes != 0) {
            __gm__ uint8_t* workspace_bytes = reinterpret_cast<__gm__ uint8_t*>(user_workspace);
            staged_weight_gm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(workspace_bytes + tiling->staging_workspace_offset));
#ifdef KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT
            vector_dequant_ready_ =
                pipe_.InitBuffer(vector_packed_ub_, kCann9VectorDequantPackedBytes) &&
                pipe_.InitBuffer(vector_half_ub_, kCann9VectorDequantHalfBytes);
#endif
        }
#else
        (void)user_workspace;
#endif
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
        uint32_t physical_core_idx = static_cast<uint32_t>(GetBlockIdx());
        if ASCEND_IS_AIV {
            const uint32_t task_ratio = static_cast<uint32_t>(GetTaskRation());
            if (task_ratio > 1) {
                physical_core_idx /= task_ratio;
            }
        }
        const uint32_t scheduled_blocks = static_cast<uint32_t>(GetBlockNum());
        const uint32_t block_dim = tiling_->block_dim != 0 ? tiling_->block_dim : scheduled_blocks;
        if (block_dim == 0) {
            return;
        }
        const uint32_t core_idx = physical_core_idx % block_dim;

#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
        if (tiling_->kernel_mode == kKernelModeStagedDequant) {
            StageWeightTiles(core_idx, in_features, out_features, group_size, zero_offsets);
        }
#endif

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
            ProcessRowQuad(
                m, in_features, out_features, group_size, has_bias, zero_offsets, packed_begin, packed_end);
        }
        for (; m + 1 < rows; m += 2) {
            ProcessRowPair(m, in_features, out_features, group_size, has_bias, zero_offsets, packed_begin, packed_end);
        }
        if (m < rows) {
            ProcessSingleRow(
                m, in_features, out_features, group_size, has_bias, zero_offsets, packed_begin, packed_end);
        }
    }

#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF)
    __aicore__ inline bool TryProcessSingleKTileTscmHandoff(KomodoCannW4A16CubeConsumerProbe& cube_probe)
    {
        if (!cube_probe.TscmReady() || tiling_->kernel_mode != kKernelModeStagedDequant || tiling_->base_k == 0 ||
            tiling_->base_n == 0 || tiling_->out_features % tiling_->base_n != 0) {
            return false;
        }
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK
        if (tiling_->in_features % tiling_->base_k != 0) {
            return false;
        }
#else
        if (tiling_->in_features != tiling_->base_k) {
            return false;
        }
#endif

        const uint32_t rows = tiling_->rows;
        const uint32_t in_features = tiling_->in_features;
        const uint32_t out_features = tiling_->out_features;
        const uint32_t group_size = tiling_->group_size;
        const uint32_t zero_offsets = tiling_->zero_offsets;
        const uint32_t base_m = tiling_->base_m;
        const uint32_t base_n = tiling_->base_n;
        const uint32_t staging_blocks = tiling_->staging_blocks;
        const uint32_t staging_slots = tiling_->staging_slots;
        if (rows == 0 || out_features == 0 || base_m == 0 || staging_blocks == 0 || staging_slots == 0 ||
            tiling_->staging_tile_bytes == 0 || tiling_->block_dim == 0) {
            return false;
        }
        if (GetSubBlockIdx() != 0) {
            return true;
        }

        uint32_t physical_core_idx = static_cast<uint32_t>(GetBlockIdx());
        if ASCEND_IS_AIV {
            const uint32_t task_ratio = static_cast<uint32_t>(GetTaskRation());
            if (task_ratio > 1) {
                physical_core_idx /= task_ratio;
            }
        }
        const uint32_t core_idx = physical_core_idx % tiling_->block_dim;
        if (core_idx >= staging_blocks) {
            return true;
        }

        const uint32_t n_tiles = out_features / base_n;
        const uint32_t tile_elements = tiling_->staging_tile_bytes / sizeof(half);
        const uint32_t packed_stride = out_features >> 3;
        const uint32_t m_tiles = CeilDivU32(rows, base_m);
        for (uint32_t n_tile = core_idx; n_tile < n_tiles; n_tile += staging_blocks) {
            const uint32_t slot = (n_tile / staging_blocks) % staging_slots;
            const uint32_t n_begin = n_tile * base_n;
            const uint32_t n_end = n_begin + base_n;
            const uint32_t packed_begin = n_begin >> 3;
            const uint32_t packed_end = n_end >> 3;
            const uint32_t stage_base = (core_idx * staging_slots + slot) * tile_elements;

#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT
            if (!cube_probe.DirectDequantReady()) {
                return false;
            }
            LocalTensor<half> direct_b_tile = cube_probe.GetDirectDequantTile(tiling_);
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK
            const uint32_t k_tiles = in_features / tiling_->base_k;
            const bool use_pipelined_fill = tiling_->base_k >= 128 || k_tiles >= 4;
            if (use_pipelined_fill) {
                FillDirectBTileKTile(direct_b_tile, 0, n_begin, packed_begin, packed_end, packed_stride, zero_offsets);
                LocalTensor<half> b_tscm_tile = cube_probe.LoadDirectBTileToTscm(direct_b_tile, tiling_);
                for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    const uint32_t k_begin = k_tile * tiling_->base_k;
                    LocalTensor<half> next_b_tscm_tile;
                    const bool has_next_k_tile = k_tile + 1 < k_tiles;
                    for (uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                        const uint32_t m_begin = m_tile * base_m;
                        const uint32_t m_len_candidate = rows - m_begin;
                        const uint32_t m_len = m_len_candidate < base_m ? m_len_candidate : base_m;
                        cube_probe.mm.SetTensorA(x_gm_[m_begin * in_features + k_begin]);
                        cube_probe.mm.SetTensorB(b_tscm_tile, true);
                        cube_probe.mm.SetTail(static_cast<int32_t>(m_len), static_cast<int32_t>(base_n));
                        ConfigureCubeBiasForKTile(cube_probe, k_tile, n_begin);
                        cube_probe.mm.IterateAll<false>(
                            y_gm_[m_begin * out_features + n_begin], k_tile != 0, false, true);
                        if (m_tile == 0 && has_next_k_tile) {
                            FillDirectBTileKTile(
                                direct_b_tile,
                                k_tile + 1,
                                n_begin,
                                packed_begin,
                                packed_end,
                                packed_stride,
                                zero_offsets);
                            next_b_tscm_tile = cube_probe.LoadDirectBTileToTscm(direct_b_tile, tiling_);
                        }
                        cube_probe.mm.WaitIterateAll();
                    }
                    cube_probe.FreeTscmBTile(b_tscm_tile);
                    if (has_next_k_tile) {
                        b_tscm_tile = next_b_tscm_tile;
                    }
                }
            } else {
                for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    const uint32_t k_begin = k_tile * tiling_->base_k;
                    for (uint32_t tile_k = 0; tile_k < tiling_->base_k; ++tile_k) {
                        const uint32_t k = k_begin + tile_k;
                        const uint32_t group = group_size == 0 ? 0 : k / group_size;
                        for (uint32_t packed_col = packed_begin; packed_col < packed_end; ++packed_col) {
                            const uint32_t word =
                                static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                            const uint32_t n_base = packed_col << 3;
                            const uint32_t scale_base = group * out_features + n_base;
                            const uint32_t tile_n = n_base - n_begin;
                            FillDirectBTileWord(
                                direct_b_tile, tile_k * base_n + tile_n, word, scale_base, zero_offsets);
                        }
                    }
                    LocalTensor<half> b_tscm_tile = cube_probe.LoadDirectBTileToTscm(direct_b_tile, tiling_);
                    for (uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                        const uint32_t m_begin = m_tile * base_m;
                        const uint32_t m_len_candidate = rows - m_begin;
                        const uint32_t m_len = m_len_candidate < base_m ? m_len_candidate : base_m;
                        cube_probe.mm.SetTensorA(x_gm_[m_begin * in_features + k_begin]);
                        cube_probe.mm.SetTensorB(b_tscm_tile, true);
                        cube_probe.mm.SetTail(static_cast<int32_t>(m_len), static_cast<int32_t>(base_n));
                        ConfigureCubeBiasForKTile(cube_probe, k_tile, n_begin);
                        cube_probe.mm.IterateAll<false>(
                            y_gm_[m_begin * out_features + n_begin], k_tile != 0, false, true);
                        cube_probe.mm.WaitIterateAll();
                    }
                    cube_probe.FreeTscmBTile(b_tscm_tile);
                }
            }
            continue;
#else
            for (uint32_t k = 0; k < in_features; ++k) {
                const uint32_t group = group_size == 0 ? 0 : k / group_size;
                for (uint32_t packed_col = packed_begin; packed_col < packed_end; ++packed_col) {
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t n_base = packed_col << 3;
                    const uint32_t scale_base = group * out_features + n_base;
                    const uint32_t tile_n = n_base - n_begin;
                    FillDirectBTileWord(direct_b_tile, k * base_n + tile_n, word, scale_base, zero_offsets);
                }
            }
            LocalTensor<half> b_tscm_tile = cube_probe.LoadDirectBTileToTscm(direct_b_tile, tiling_);
#endif
#else
            for (uint32_t k = 0; k < in_features; ++k) {
                const uint32_t group = group_size == 0 ? 0 : k / group_size;
                for (uint32_t packed_col = packed_begin; packed_col < packed_end; ++packed_col) {
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t n_base = packed_col << 3;
                    const uint32_t scale_base = group * out_features + n_base;
                    const uint32_t tile_n = n_base - n_begin;
                    StagePackedWord(stage_base + k * base_n + tile_n, word, scale_base, zero_offsets);
                }
            }
            LocalTensor<half> b_tscm_tile = cube_probe.LoadStagedBTileToTscm(staged_weight_gm_, stage_base, tiling_);
#endif
#ifndef KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK
            for (uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                const uint32_t m_begin = m_tile * base_m;
                const uint32_t m_len_candidate = rows - m_begin;
                const uint32_t m_len = m_len_candidate < base_m ? m_len_candidate : base_m;
                cube_probe.mm.SetTensorA(x_gm_[m_begin * in_features]);
                cube_probe.mm.SetTensorB(b_tscm_tile, true);
                cube_probe.mm.SetTail(static_cast<int32_t>(m_len), static_cast<int32_t>(base_n));
                ConfigureCubeBiasForKTile(cube_probe, 0, n_begin);
                cube_probe.mm.IterateAll<false>(y_gm_[m_begin * out_features + n_begin], false, false, true);
                cube_probe.mm.WaitIterateAll();
            }
            cube_probe.FreeTscmBTile();
#endif
        }
        return true;
    }
#endif

#if defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
    __aicore__ inline bool TryProcessVecoutHandoff(KomodoCannW4A16CubeConsumerProbe& cube_probe)
    {
        if (!cube_probe.DirectDequantReady() || tiling_->kernel_mode != kKernelModeStagedDequant ||
            tiling_->base_k == 0 || tiling_->base_n == 0 || tiling_->out_features % tiling_->base_n != 0 ||
            tiling_->in_features % tiling_->base_k != 0) {
            return false;
        }

        const uint32_t rows = tiling_->rows;
        const uint32_t in_features = tiling_->in_features;
        const uint32_t out_features = tiling_->out_features;
        const uint32_t base_m = tiling_->base_m;
        const uint32_t base_n = tiling_->base_n;
        const uint32_t staging_blocks = tiling_->staging_blocks;
        if (rows == 0 || out_features == 0 || base_m == 0 || staging_blocks == 0 || tiling_->block_dim == 0) {
            return false;
        }
        if (GetSubBlockIdx() != 0) {
            return true;
        }

        uint32_t physical_core_idx = static_cast<uint32_t>(GetBlockIdx());
        if ASCEND_IS_AIV {
            const uint32_t task_ratio = static_cast<uint32_t>(GetTaskRation());
            if (task_ratio > 1) {
                physical_core_idx /= task_ratio;
            }
        }
        const uint32_t core_idx = physical_core_idx % tiling_->block_dim;
        if (core_idx >= staging_blocks) {
            return true;
        }

        const uint32_t n_tiles = out_features / base_n;
        const uint32_t packed_stride = out_features >> 3;
#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
        if (!cube_probe.LocalAReady()) {
            return false;
        }
        const uint32_t m_tiles = CeilDivU32(rows, base_m);
#else
        const uint32_t m_tiles = rows;
#endif
        const uint32_t k_tiles = in_features / tiling_->base_k;
        const uint32_t zero_offsets = tiling_->zero_offsets;
        LocalTensor<half> direct_b_tile = cube_probe.GetDirectDequantTile(tiling_);
#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
        LocalTensor<half> direct_a_tile = cube_probe.GetLocalATile(tiling_);
        LocalTensor<int32_t> packed_b_tile = cube_probe.GetPackedBTile(tiling_);
        for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
            const uint32_t k_begin = k_tile * tiling_->base_k;
            if (m_tiles <= 1) {
                const uint32_t m_begin = 0;
                const uint32_t m_len = rows < base_m ? rows : base_m;
                FillDirectATile(direct_a_tile, m_begin, m_len, k_begin, tiling_->base_k, in_features);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t n_tile = core_idx; n_tile < n_tiles; n_tile += staging_blocks) {
                    const uint32_t n_begin = n_tile * base_n;
                    const uint32_t n_end = n_begin + base_n;
                    const uint32_t packed_begin = n_begin >> 3;
                    const uint32_t packed_end = n_end >> 3;
                    FillDirectBTileKTileFromPackedTile(
                        direct_b_tile,
                        packed_b_tile,
                        k_tile,
                        n_begin,
                        packed_begin,
                        packed_end,
                        packed_stride,
                        zero_offsets);
                    PipeBarrier<PIPE_V>();
                    cube_probe.mm.SetTensorA(direct_a_tile);
                    cube_probe.mm.SetTensorB(direct_b_tile);
                    cube_probe.mm.SetTail(static_cast<int32_t>(m_len), static_cast<int32_t>(base_n));
                    ConfigureCubeBiasForKTile(cube_probe, k_tile, n_begin);
                    cube_probe.mm.IterateAll<false>(
                        y_gm_[m_begin * out_features + n_begin], k_tile != 0, false, true);
                    cube_probe.mm.WaitIterateAll();
                }
                continue;
            }
            for (uint32_t n_tile = core_idx; n_tile < n_tiles; n_tile += staging_blocks) {
                const uint32_t n_begin = n_tile * base_n;
                const uint32_t n_end = n_begin + base_n;
                const uint32_t packed_begin = n_begin >> 3;
                const uint32_t packed_end = n_end >> 3;
                FillDirectBTileKTileFromPackedTile(
                    direct_b_tile,
                    packed_b_tile,
                    k_tile,
                    n_begin,
                    packed_begin,
                    packed_end,
                    packed_stride,
                    zero_offsets);
                PipeBarrier<PIPE_V>();
                for (uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                    const uint32_t m_begin = m_tile * base_m;
                    const uint32_t m_len_candidate = rows - m_begin;
                    const uint32_t m_len = m_len_candidate < base_m ? m_len_candidate : base_m;
                    FillDirectATile(direct_a_tile, m_begin, m_len, k_begin, tiling_->base_k, in_features);
                    PipeBarrier<PIPE_ALL>();
                    cube_probe.mm.SetTensorA(direct_a_tile);
                    cube_probe.mm.SetTensorB(direct_b_tile);
                    cube_probe.mm.SetTail(static_cast<int32_t>(m_len), static_cast<int32_t>(base_n));
                    ConfigureCubeBiasForKTile(cube_probe, k_tile, n_begin);
                    cube_probe.mm.IterateAll<false>(
                        y_gm_[m_begin * out_features + n_begin], k_tile != 0, false, true);
                    cube_probe.mm.WaitIterateAll();
                }
            }
        }
#else
        for (uint32_t n_tile = core_idx; n_tile < n_tiles; n_tile += staging_blocks) {
            const uint32_t n_begin = n_tile * base_n;
            const uint32_t n_end = n_begin + base_n;
            const uint32_t packed_begin = n_begin >> 3;
            const uint32_t packed_end = n_end >> 3;
            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const uint32_t k_begin = k_tile * tiling_->base_k;
                FillDirectBTileKTile(direct_b_tile, k_tile, n_begin, packed_begin, packed_end, packed_stride, zero_offsets);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                    const uint32_t m_begin = m_tile;
                    const uint32_t m_len = 1;
                    cube_probe.mm.SetTensorA(x_gm_[m_begin * in_features + k_begin]);
                    cube_probe.mm.SetTensorB(direct_b_tile);
                    cube_probe.mm.SetTail(static_cast<int32_t>(m_len), static_cast<int32_t>(base_n));
                    ConfigureCubeBiasForKTile(cube_probe, k_tile, n_begin);
                    cube_probe.mm.IterateAll<false>(
                        y_gm_[m_begin * out_features + n_begin], k_tile != 0, false, true);
                    cube_probe.mm.WaitIterateAll();
                }
            }
        }
#endif
        return true;
    }
#endif

private:
#if defined(KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF) || \
    defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
    __aicore__ inline void ConfigureCubeBiasForKTile(
        KomodoCannW4A16CubeConsumerProbe& cube_probe,
        uint32_t k_tile,
        uint32_t n_begin)
    {
        if (tiling_->has_bias == 0) {
            return;
        }
        if (k_tile == 0) {
            cube_probe.mm.SetBias(bias_gm_[n_begin]);
        } else {
            cube_probe.mm.ClearBias();
        }
    }
#endif

#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
#ifdef KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_DEQUANT
#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
    __aicore__ inline void FillDirectATile(
        LocalTensor<half>& a_tile,
        uint32_t m_begin,
        uint32_t m_len,
        uint32_t k_begin,
        uint32_t base_k,
        uint32_t in_features)
    {
        // Row-wise DataCopy wins on medium/larger local-A tiles and row-8 large-K tiles.
        if (tiling_->rows >= 16 || (tiling_->rows == 8 && in_features >= 1024)) {
            for (uint32_t m = 0; m < m_len; ++m) {
                const uint32_t src_base = (m_begin + m) * in_features + k_begin;
                const uint32_t dst_base = m * base_k;
                DataCopy(a_tile[dst_base], x_gm_[src_base], base_k);
            }
            return;
        }
        for (uint32_t m = 0; m < m_len; ++m) {
            const uint32_t src_base = (m_begin + m) * in_features + k_begin;
            const uint32_t dst_base = m * base_k;
            for (uint32_t k = 0; k < base_k; ++k) {
                a_tile.SetValue(dst_base + k, x_gm_.GetValue(src_base + k));
            }
        }
    }
#endif

#ifdef KOMODO_CANN_EXPERIMENTAL_VECOUT_LOCAL_A
    __aicore__ inline void FillDirectBTileKTileFromPackedTile(
        LocalTensor<half>& b_tile,
        LocalTensor<int32_t>& packed_tile,
        uint32_t k_tile,
        uint32_t n_begin,
        uint32_t packed_begin,
        uint32_t packed_end,
        uint32_t packed_stride,
        uint32_t zero_offsets)
    {
        const uint32_t base_k = tiling_->base_k;
        const uint32_t packed_cols = packed_end - packed_begin;
        const uint32_t k_begin = k_tile * base_k;
        for (uint32_t tile_k = 0; tile_k < base_k; ++tile_k) {
            const uint32_t k = k_begin + tile_k;
            DataCopy(
                packed_tile[tile_k * packed_cols],
                packed_weight_gm_[k * packed_stride + packed_begin],
                packed_cols);
        }
        PipeBarrier<PIPE_ALL>();
        FillDirectBTileKTileFromPackedValues(
            b_tile, packed_tile, k_tile, n_begin, packed_begin, packed_end, packed_cols, zero_offsets);
    }

    __aicore__ inline void FillDirectBTileKTileFromPackedValues(
        LocalTensor<half>& b_tile,
        LocalTensor<int32_t>& packed_tile,
        uint32_t k_tile,
        uint32_t n_begin,
        uint32_t packed_begin,
        uint32_t packed_end,
        uint32_t packed_cols,
        uint32_t zero_offsets)
    {
        const uint32_t base_n = tiling_->base_n;
        const uint32_t k_begin = k_tile * tiling_->base_k;
        const uint32_t k_end = k_begin + tiling_->base_k;
        const uint32_t first_group = tiling_->group_size == 0 ? 0 : k_begin / tiling_->group_size;
        const uint32_t last_group = tiling_->group_size == 0 ? 0 : (k_end - 1) / tiling_->group_size;
        for (uint32_t group = first_group; group <= last_group; ++group) {
            const uint32_t group_k_begin_candidate = tiling_->group_size == 0 ? k_begin : group * tiling_->group_size;
            const uint32_t group_k_end_candidate =
                tiling_->group_size == 0 ? k_end : group_k_begin_candidate + tiling_->group_size;
            const uint32_t group_k_begin = group_k_begin_candidate < k_begin ? k_begin : group_k_begin_candidate;
            const uint32_t group_k_end = group_k_end_candidate > k_end ? k_end : group_k_end_candidate;
            for (uint32_t packed_col = packed_begin; packed_col < packed_end;) {
                if (zero_offsets != 0 && packed_col + 3 < packed_end) {
                    const uint32_t n_base0 = packed_col << 3;
                    const uint32_t n_base1 = n_base0 + 8;
                    const uint32_t n_base2 = n_base1 + 8;
                    const uint32_t n_base3 = n_base2 + 8;
                    const uint32_t scale_base0 = group * tiling_->out_features + n_base0;
                    const uint32_t scale_base1 = scale_base0 + 8;
                    const uint32_t scale_base2 = scale_base1 + 8;
                    const uint32_t scale_base3 = scale_base2 + 8;
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
                    const float scale20 = static_cast<float>(scales_gm_.GetValue(scale_base2));
                    const float scale21 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 1));
                    const float scale22 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 2));
                    const float scale23 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 3));
                    const float scale24 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 4));
                    const float scale25 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 5));
                    const float scale26 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 6));
                    const float scale27 = static_cast<float>(scales_gm_.GetValue(scale_base2 + 7));
                    const float scale30 = static_cast<float>(scales_gm_.GetValue(scale_base3));
                    const float scale31 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 1));
                    const float scale32 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 2));
                    const float scale33 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 3));
                    const float scale34 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 4));
                    const float scale35 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 5));
                    const float scale36 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 6));
                    const float scale37 = static_cast<float>(scales_gm_.GetValue(scale_base3 + 7));
                    const uint32_t packed_offset0 = packed_col - packed_begin;
                    const uint32_t packed_offset1 = packed_offset0 + 1;
                    const uint32_t packed_offset2 = packed_offset1 + 1;
                    const uint32_t packed_offset3 = packed_offset2 + 1;
                    const uint32_t tile_n0 = n_base0 - n_begin;
                    const uint32_t tile_n1 = n_base1 - n_begin;
                    const uint32_t tile_n2 = n_base2 - n_begin;
                    const uint32_t tile_n3 = n_base3 - n_begin;
                    for (uint32_t k = group_k_begin; k < group_k_end; ++k) {
                        const uint32_t tile_k = k - k_begin;
                        const uint32_t row_offset = tile_k * packed_cols;
                        const uint32_t word0 =
                            static_cast<uint32_t>(packed_tile.GetValue(row_offset + packed_offset0));
                        const uint32_t word1 =
                            static_cast<uint32_t>(packed_tile.GetValue(row_offset + packed_offset1));
                        const uint32_t word2 =
                            static_cast<uint32_t>(packed_tile.GetValue(row_offset + packed_offset2));
                        const uint32_t word3 =
                            static_cast<uint32_t>(packed_tile.GetValue(row_offset + packed_offset3));
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n0,
                            word0,
                            scale00,
                            scale01,
                            scale02,
                            scale03,
                            scale04,
                            scale05,
                            scale06,
                            scale07);
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n1,
                            word1,
                            scale10,
                            scale11,
                            scale12,
                            scale13,
                            scale14,
                            scale15,
                            scale16,
                            scale17);
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n2,
                            word2,
                            scale20,
                            scale21,
                            scale22,
                            scale23,
                            scale24,
                            scale25,
                            scale26,
                            scale27);
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n3,
                            word3,
                            scale30,
                            scale31,
                            scale32,
                            scale33,
                            scale34,
                            scale35,
                            scale36,
                            scale37);
                    }
                    packed_col += 4;
                    continue;
                }
                if (zero_offsets != 0 && packed_col + 1 < packed_end) {
                    const uint32_t n_base0 = packed_col << 3;
                    const uint32_t n_base1 = n_base0 + 8;
                    const uint32_t scale_base0 = group * tiling_->out_features + n_base0;
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
                    const uint32_t packed_offset0 = packed_col - packed_begin;
                    const uint32_t packed_offset1 = packed_offset0 + 1;
                    const uint32_t tile_n0 = n_base0 - n_begin;
                    const uint32_t tile_n1 = n_base1 - n_begin;
                    for (uint32_t k = group_k_begin; k < group_k_end; ++k) {
                        const uint32_t tile_k = k - k_begin;
                        const uint32_t row_offset = tile_k * packed_cols;
                        const uint32_t word0 =
                            static_cast<uint32_t>(packed_tile.GetValue(row_offset + packed_offset0));
                        const uint32_t word1 =
                            static_cast<uint32_t>(packed_tile.GetValue(row_offset + packed_offset1));
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n0,
                            word0,
                            scale00,
                            scale01,
                            scale02,
                            scale03,
                            scale04,
                            scale05,
                            scale06,
                            scale07);
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n1,
                            word1,
                            scale10,
                            scale11,
                            scale12,
                            scale13,
                            scale14,
                            scale15,
                            scale16,
                            scale17);
                    }
                    packed_col += 2;
                    continue;
                }
                const uint32_t n_base = packed_col << 3;
                const uint32_t scale_base = group * tiling_->out_features + n_base;
                const float scale0 = static_cast<float>(scales_gm_.GetValue(scale_base));
                const float scale1 = static_cast<float>(scales_gm_.GetValue(scale_base + 1));
                const float scale2 = static_cast<float>(scales_gm_.GetValue(scale_base + 2));
                const float scale3 = static_cast<float>(scales_gm_.GetValue(scale_base + 3));
                const float scale4 = static_cast<float>(scales_gm_.GetValue(scale_base + 4));
                const float scale5 = static_cast<float>(scales_gm_.GetValue(scale_base + 5));
                const float scale6 = static_cast<float>(scales_gm_.GetValue(scale_base + 6));
                const float scale7 = static_cast<float>(scales_gm_.GetValue(scale_base + 7));
                const uint32_t packed_offset = packed_col - packed_begin;
                const uint32_t tile_n = n_base - n_begin;
                if (zero_offsets != 0) {
                    for (uint32_t k = group_k_begin; k < group_k_end; ++k) {
                        const uint32_t tile_k = k - k_begin;
                        const uint32_t word =
                            static_cast<uint32_t>(packed_tile.GetValue(tile_k * packed_cols + packed_offset));
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n,
                            word,
                            scale0,
                            scale1,
                            scale2,
                            scale3,
                            scale4,
                            scale5,
                            scale6,
                            scale7);
                    }
                } else {
                    const float offset0 = static_cast<float>(offsets_gm_.GetValue(scale_base));
                    const float offset1 = static_cast<float>(offsets_gm_.GetValue(scale_base + 1));
                    const float offset2 = static_cast<float>(offsets_gm_.GetValue(scale_base + 2));
                    const float offset3 = static_cast<float>(offsets_gm_.GetValue(scale_base + 3));
                    const float offset4 = static_cast<float>(offsets_gm_.GetValue(scale_base + 4));
                    const float offset5 = static_cast<float>(offsets_gm_.GetValue(scale_base + 5));
                    const float offset6 = static_cast<float>(offsets_gm_.GetValue(scale_base + 6));
                    const float offset7 = static_cast<float>(offsets_gm_.GetValue(scale_base + 7));
                    for (uint32_t k = group_k_begin; k < group_k_end; ++k) {
                        const uint32_t tile_k = k - k_begin;
                        const uint32_t word =
                            static_cast<uint32_t>(packed_tile.GetValue(tile_k * packed_cols + packed_offset));
                        FillDirectBTileWordValues(
                            b_tile,
                            tile_k * base_n + tile_n,
                            word,
                            scale0,
                            scale1,
                            scale2,
                            scale3,
                            scale4,
                            scale5,
                            scale6,
                            scale7,
                            offset0,
                            offset1,
                            offset2,
                            offset3,
                            offset4,
                            offset5,
                            offset6,
                            offset7);
                    }
                }
                ++packed_col;
            }
        }
    }
#endif

#ifdef KOMODO_CANN_EXPERIMENTAL_LOCAL_DIRECT_MULTIK
    __aicore__ inline void FillDirectBTileKTile(
        LocalTensor<half>& b_tile,
        uint32_t k_tile,
        uint32_t n_begin,
        uint32_t packed_begin,
        uint32_t packed_end,
        uint32_t packed_stride,
        uint32_t zero_offsets)
    {
        const uint32_t base_n = tiling_->base_n;
        const uint32_t k_begin = k_tile * tiling_->base_k;
        const uint32_t k_end = k_begin + tiling_->base_k;
        const uint32_t first_group = tiling_->group_size == 0 ? 0 : k_begin / tiling_->group_size;
        const uint32_t last_group = tiling_->group_size == 0 ? 0 : (k_end - 1) / tiling_->group_size;
        for (uint32_t group = first_group; group <= last_group; ++group) {
            const uint32_t group_k_begin_candidate = tiling_->group_size == 0 ? k_begin : group * tiling_->group_size;
            const uint32_t group_k_end_candidate =
                tiling_->group_size == 0 ? k_end : group_k_begin_candidate + tiling_->group_size;
            const uint32_t group_k_begin = group_k_begin_candidate < k_begin ? k_begin : group_k_begin_candidate;
            const uint32_t group_k_end = group_k_end_candidate > k_end ? k_end : group_k_end_candidate;
            for (uint32_t packed_col = packed_begin; packed_col < packed_end; ++packed_col) {
                const uint32_t n_base = packed_col << 3;
                const uint32_t scale_base = group * tiling_->out_features + n_base;
                const float scale0 = static_cast<float>(scales_gm_.GetValue(scale_base));
                const float scale1 = static_cast<float>(scales_gm_.GetValue(scale_base + 1));
                const float scale2 = static_cast<float>(scales_gm_.GetValue(scale_base + 2));
                const float scale3 = static_cast<float>(scales_gm_.GetValue(scale_base + 3));
                const float scale4 = static_cast<float>(scales_gm_.GetValue(scale_base + 4));
                const float scale5 = static_cast<float>(scales_gm_.GetValue(scale_base + 5));
                const float scale6 = static_cast<float>(scales_gm_.GetValue(scale_base + 6));
                const float scale7 = static_cast<float>(scales_gm_.GetValue(scale_base + 7));
                const uint32_t tile_n = n_base - n_begin;
                if (zero_offsets != 0) {
                    for (uint32_t k = group_k_begin; k < group_k_end; ++k) {
                        const uint32_t tile_k = k - k_begin;
                        const uint32_t word =
                            static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                        FillDirectBTileWordValuesNoOffset(
                            b_tile,
                            tile_k * base_n + tile_n,
                            word,
                            scale0,
                            scale1,
                            scale2,
                            scale3,
                            scale4,
                            scale5,
                            scale6,
                            scale7);
                    }
                } else {
                    const float offset0 = static_cast<float>(offsets_gm_.GetValue(scale_base));
                    const float offset1 = static_cast<float>(offsets_gm_.GetValue(scale_base + 1));
                    const float offset2 = static_cast<float>(offsets_gm_.GetValue(scale_base + 2));
                    const float offset3 = static_cast<float>(offsets_gm_.GetValue(scale_base + 3));
                    const float offset4 = static_cast<float>(offsets_gm_.GetValue(scale_base + 4));
                    const float offset5 = static_cast<float>(offsets_gm_.GetValue(scale_base + 5));
                    const float offset6 = static_cast<float>(offsets_gm_.GetValue(scale_base + 6));
                    const float offset7 = static_cast<float>(offsets_gm_.GetValue(scale_base + 7));
                    for (uint32_t k = group_k_begin; k < group_k_end; ++k) {
                        const uint32_t tile_k = k - k_begin;
                        const uint32_t word =
                            static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                        FillDirectBTileWordValues(
                            b_tile,
                            tile_k * base_n + tile_n,
                            word,
                            scale0,
                            scale1,
                            scale2,
                            scale3,
                            scale4,
                            scale5,
                            scale6,
                            scale7,
                            offset0,
                            offset1,
                            offset2,
                            offset3,
                            offset4,
                            offset5,
                            offset6,
                            offset7);
                    }
                }
            }
        }
    }
#endif

    __aicore__ inline void FillDirectBTileWord(
        LocalTensor<half>& b_tile,
        uint32_t tile_offset,
        uint32_t word,
        uint32_t scale_base,
        uint32_t zero_offsets)
    {
        if (vector_dequant_ready_) {
            FillDirectBTileWordVector(b_tile, tile_offset, word, scale_base, zero_offsets);
            return;
        }
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
        FillDirectBTileWordValues(
            b_tile,
            tile_offset,
            word,
            scale0,
            scale1,
            scale2,
            scale3,
            scale4,
            scale5,
            scale6,
            scale7,
            offset0,
            offset1,
            offset2,
            offset3,
            offset4,
            offset5,
            offset6,
            offset7);
    }

    __aicore__ inline void FillDirectBTileWordValues(
        LocalTensor<half>& b_tile,
        uint32_t tile_offset,
        uint32_t word,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        float scale4,
        float scale5,
        float scale6,
        float scale7,
        float offset0,
        float offset1,
        float offset2,
        float offset3,
        float offset4,
        float offset5,
        float offset6,
        float offset7)
    {
#ifndef KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF
        if (vector_dequant_ready_) {
            FillDirectBTileWordVectorValues(
                b_tile,
                tile_offset,
                word,
                scale0,
                scale1,
                scale2,
                scale3,
                scale4,
                scale5,
                scale6,
                scale7,
                offset0,
                offset1,
                offset2,
                offset3,
                offset4,
                offset5,
                offset6,
                offset7);
            return;
        }
#endif
        b_tile.SetValue(tile_offset, static_cast<half>(DequantLane(word, 0, scale0, offset0)));
        b_tile.SetValue(tile_offset + 1, static_cast<half>(DequantLane(word, 1, scale1, offset1)));
        b_tile.SetValue(tile_offset + 2, static_cast<half>(DequantLane(word, 2, scale2, offset2)));
        b_tile.SetValue(tile_offset + 3, static_cast<half>(DequantLane(word, 3, scale3, offset3)));
        b_tile.SetValue(tile_offset + 4, static_cast<half>(DequantLane(word, 4, scale4, offset4)));
        b_tile.SetValue(tile_offset + 5, static_cast<half>(DequantLane(word, 5, scale5, offset5)));
        b_tile.SetValue(tile_offset + 6, static_cast<half>(DequantLane(word, 6, scale6, offset6)));
        b_tile.SetValue(tile_offset + 7, static_cast<half>(DequantLane(word, 7, scale7, offset7)));
    }

    __aicore__ inline void FillDirectBTileWordValuesNoOffset(
        LocalTensor<half>& b_tile,
        uint32_t tile_offset,
        uint32_t word,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        float scale4,
        float scale5,
        float scale6,
        float scale7)
    {
        b_tile.SetValue(tile_offset, static_cast<half>(DequantLaneNoOffset(word, 0, scale0)));
        b_tile.SetValue(tile_offset + 1, static_cast<half>(DequantLaneNoOffset(word, 1, scale1)));
        b_tile.SetValue(tile_offset + 2, static_cast<half>(DequantLaneNoOffset(word, 2, scale2)));
        b_tile.SetValue(tile_offset + 3, static_cast<half>(DequantLaneNoOffset(word, 3, scale3)));
        b_tile.SetValue(tile_offset + 4, static_cast<half>(DequantLaneNoOffset(word, 4, scale4)));
        b_tile.SetValue(tile_offset + 5, static_cast<half>(DequantLaneNoOffset(word, 5, scale5)));
        b_tile.SetValue(tile_offset + 6, static_cast<half>(DequantLaneNoOffset(word, 6, scale6)));
        b_tile.SetValue(tile_offset + 7, static_cast<half>(DequantLaneNoOffset(word, 7, scale7)));
    }

    __aicore__ inline void FillDirectBTileWordVector(
        LocalTensor<half>& b_tile,
        uint32_t tile_offset,
        uint32_t word,
        uint32_t scale_base,
        uint32_t zero_offsets)
    {
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
        FillDirectBTileWordVectorValues(
            b_tile,
            tile_offset,
            word,
            scale0,
            scale1,
            scale2,
            scale3,
            scale4,
            scale5,
            scale6,
            scale7,
            offset0,
            offset1,
            offset2,
            offset3,
            offset4,
            offset5,
            offset6,
            offset7);
    }

    __aicore__ inline void FillDirectBTileWordVectorValues(
        LocalTensor<half>& b_tile,
        uint32_t tile_offset,
        uint32_t word,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        float scale4,
        float scale5,
        float scale6,
        float scale7,
        float offset0,
        float offset1,
        float offset2,
        float offset3,
        float offset4,
        float offset5,
        float offset6,
        float offset7)
    {
        LocalTensor<uint32_t> packed = vector_packed_ub_.Get<uint32_t>(kCann9VectorDequantPackedWords);
        LocalTensor<half> dequant = vector_half_ub_.Get<half>(kCann9VectorDequantLanes);
        packed.SetValue(0, word);
        asc_int42half_sync(
            reinterpret_cast<__ubuf__ half*>(dequant.GetPhyAddr()),
            reinterpret_cast<__ubuf__ KomodoCannCapiInt4*>(packed.GetPhyAddr()),
            static_cast<uint32_t>(kCann9VectorDequantLanes));

        b_tile.SetValue(tile_offset, static_cast<half>((static_cast<float>(dequant.GetValue(0)) + offset0) * scale0));
        b_tile.SetValue(
            tile_offset + 1, static_cast<half>((static_cast<float>(dequant.GetValue(1)) + offset1) * scale1));
        b_tile.SetValue(
            tile_offset + 2, static_cast<half>((static_cast<float>(dequant.GetValue(2)) + offset2) * scale2));
        b_tile.SetValue(
            tile_offset + 3, static_cast<half>((static_cast<float>(dequant.GetValue(3)) + offset3) * scale3));
        b_tile.SetValue(
            tile_offset + 4, static_cast<half>((static_cast<float>(dequant.GetValue(4)) + offset4) * scale4));
        b_tile.SetValue(
            tile_offset + 5, static_cast<half>((static_cast<float>(dequant.GetValue(5)) + offset5) * scale5));
        b_tile.SetValue(
            tile_offset + 6, static_cast<half>((static_cast<float>(dequant.GetValue(6)) + offset6) * scale6));
        b_tile.SetValue(
            tile_offset + 7, static_cast<half>((static_cast<float>(dequant.GetValue(7)) + offset7) * scale7));
    }
#endif

    __aicore__ inline void StageWeightTiles(
        uint32_t core_idx,
        uint32_t in_features,
        uint32_t out_features,
        uint32_t group_size,
        uint32_t zero_offsets)
    {
        const uint32_t staging_blocks = tiling_->staging_blocks;
        const uint32_t staging_slots = tiling_->staging_slots;
        const uint32_t base_k = tiling_->base_k;
        const uint32_t base_n = tiling_->base_n;
        if (tiling_->staging_workspace_bytes == 0 || tiling_->staging_tile_bytes == 0 || staging_blocks == 0 ||
            staging_slots == 0 || base_k == 0 || base_n == 0 || core_idx >= staging_blocks) {
            return;
        }

        const uint32_t k_tiles = CeilDiv(in_features, base_k);
        const uint32_t n_tiles = CeilDiv(out_features, base_n);
        const uint32_t total_tasks = k_tiles * n_tiles;
        const uint32_t tile_elements = tiling_->staging_tile_bytes / sizeof(half);
        const uint32_t packed_stride = out_features >> 3;
        for (uint32_t task = core_idx; task < total_tasks; task += staging_blocks) {
            const uint32_t slot = (task / staging_blocks) % staging_slots;
            const uint32_t k_tile = task / n_tiles;
            const uint32_t n_tile = task - k_tile * n_tiles;
            const uint32_t k_begin = k_tile * base_k;
            const uint32_t k_end_candidate = k_begin + base_k;
            const uint32_t k_end = k_end_candidate < in_features ? k_end_candidate : in_features;
            const uint32_t n_begin = n_tile * base_n;
            const uint32_t n_end_candidate = n_begin + base_n;
            const uint32_t n_end = n_end_candidate < out_features ? n_end_candidate : out_features;
            const uint32_t packed_begin = n_begin >> 3;
            const uint32_t packed_end = (n_end + 7) >> 3;
            const uint32_t stage_base = (core_idx * staging_slots + slot) * tile_elements;

            for (uint32_t k = k_begin; k < k_end; ++k) {
                const uint32_t tile_k = k - k_begin;
                const uint32_t group = group_size == 0 ? 0 : k / group_size;
                for (uint32_t packed_col = packed_begin; packed_col < packed_end; ++packed_col) {
                    const uint32_t word =
                        static_cast<uint32_t>(packed_weight_gm_.GetValue(k * packed_stride + packed_col));
                    const uint32_t n_base = packed_col << 3;
                    const uint32_t scale_base = group * out_features + n_base;
                    const uint32_t tile_n = n_base - n_begin;
                    StagePackedWord(stage_base + tile_k * base_n + tile_n, word, scale_base, zero_offsets);
                }
            }
        }
    }

    __aicore__ inline void StagePackedWord(uint32_t stage_offset, uint32_t word, uint32_t scale_base, uint32_t zero_offsets)
    {
#ifdef KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT
        if (vector_dequant_ready_) {
            StagePackedWordVector(stage_offset, word, scale_base, zero_offsets);
            return;
        }
#endif
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
        staged_weight_gm_.SetValue(stage_offset, static_cast<half>(DequantLane(word, 0, scale0, offset0)));
        staged_weight_gm_.SetValue(stage_offset + 1, static_cast<half>(DequantLane(word, 1, scale1, offset1)));
        staged_weight_gm_.SetValue(stage_offset + 2, static_cast<half>(DequantLane(word, 2, scale2, offset2)));
        staged_weight_gm_.SetValue(stage_offset + 3, static_cast<half>(DequantLane(word, 3, scale3, offset3)));
        staged_weight_gm_.SetValue(stage_offset + 4, static_cast<half>(DequantLane(word, 4, scale4, offset4)));
        staged_weight_gm_.SetValue(stage_offset + 5, static_cast<half>(DequantLane(word, 5, scale5, offset5)));
        staged_weight_gm_.SetValue(stage_offset + 6, static_cast<half>(DequantLane(word, 6, scale6, offset6)));
        staged_weight_gm_.SetValue(stage_offset + 7, static_cast<half>(DequantLane(word, 7, scale7, offset7)));
    }

#ifdef KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT
    __aicore__ inline void StagePackedWordVector(
        uint32_t stage_offset,
        uint32_t word,
        uint32_t scale_base,
        uint32_t zero_offsets)
    {
        LocalTensor<uint32_t> packed = vector_packed_ub_.Get<uint32_t>(kCann9VectorDequantPackedWords);
        LocalTensor<half> dequant = vector_half_ub_.Get<half>(kCann9VectorDequantLanes);
        packed.SetValue(0, word);
        asc_int42half_sync(
            reinterpret_cast<__ubuf__ half*>(dequant.GetPhyAddr()),
            reinterpret_cast<__ubuf__ KomodoCannCapiInt4*>(packed.GetPhyAddr()),
            static_cast<uint32_t>(kCann9VectorDequantLanes));

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
        staged_weight_gm_.SetValue(
            stage_offset, static_cast<half>((static_cast<float>(dequant.GetValue(0)) + offset0) * scale0));
        staged_weight_gm_.SetValue(
            stage_offset + 1, static_cast<half>((static_cast<float>(dequant.GetValue(1)) + offset1) * scale1));
        staged_weight_gm_.SetValue(
            stage_offset + 2, static_cast<half>((static_cast<float>(dequant.GetValue(2)) + offset2) * scale2));
        staged_weight_gm_.SetValue(
            stage_offset + 3, static_cast<half>((static_cast<float>(dequant.GetValue(3)) + offset3) * scale3));
        staged_weight_gm_.SetValue(
            stage_offset + 4, static_cast<half>((static_cast<float>(dequant.GetValue(4)) + offset4) * scale4));
        staged_weight_gm_.SetValue(
            stage_offset + 5, static_cast<half>((static_cast<float>(dequant.GetValue(5)) + offset5) * scale5));
        staged_weight_gm_.SetValue(
            stage_offset + 6, static_cast<half>((static_cast<float>(dequant.GetValue(6)) + offset6) * scale6));
        staged_weight_gm_.SetValue(
            stage_offset + 7, static_cast<half>((static_cast<float>(dequant.GetValue(7)) + offset7) * scale7));
    }
#endif
#endif

    __aicore__ inline void ProcessSingleRow(
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
                const float offset00 = OffsetValue(scale_base0, zero_offsets);
                const float offset01 = OffsetValue(scale_base0 + 1, zero_offsets);
                const float offset02 = OffsetValue(scale_base0 + 2, zero_offsets);
                const float offset03 = OffsetValue(scale_base0 + 3, zero_offsets);
                const float offset04 = OffsetValue(scale_base0 + 4, zero_offsets);
                const float offset05 = OffsetValue(scale_base0 + 5, zero_offsets);
                const float offset06 = OffsetValue(scale_base0 + 6, zero_offsets);
                const float offset07 = OffsetValue(scale_base0 + 7, zero_offsets);
                const float offset10 = OffsetValue(scale_base1, zero_offsets);
                const float offset11 = OffsetValue(scale_base1 + 1, zero_offsets);
                const float offset12 = OffsetValue(scale_base1 + 2, zero_offsets);
                const float offset13 = OffsetValue(scale_base1 + 3, zero_offsets);
                const float offset14 = OffsetValue(scale_base1 + 4, zero_offsets);
                const float offset15 = OffsetValue(scale_base1 + 5, zero_offsets);
                const float offset16 = OffsetValue(scale_base1 + 6, zero_offsets);
                const float offset17 = OffsetValue(scale_base1 + 7, zero_offsets);

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
                const float offset0 = OffsetValue(scale_base, zero_offsets);
                const float offset1 = OffsetValue(scale_base + 1, zero_offsets);
                const float offset2 = OffsetValue(scale_base + 2, zero_offsets);
                const float offset3 = OffsetValue(scale_base + 3, zero_offsets);
                const float offset4 = OffsetValue(scale_base + 4, zero_offsets);
                const float offset5 = OffsetValue(scale_base + 5, zero_offsets);
                const float offset6 = OffsetValue(scale_base + 6, zero_offsets);
                const float offset7 = OffsetValue(scale_base + 7, zero_offsets);

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

#define KOMODO_APPLY_OFFSET_ROW(prefix, x_sum) \
        prefix##0 += (x_sum) * offset0 * scale0; \
        prefix##1 += (x_sum) * offset1 * scale1; \
        prefix##2 += (x_sum) * offset2 * scale2; \
        prefix##3 += (x_sum) * offset3 * scale3; \
        prefix##4 += (x_sum) * offset4 * scale4; \
        prefix##5 += (x_sum) * offset5 * scale5; \
        prefix##6 += (x_sum) * offset6 * scale6; \
        prefix##7 += (x_sum) * offset7 * scale7

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

                if (zero_offsets != 0) {
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
                        const float deq0 = DequantLaneNoOffset(word, 0, scale0);
                        const float deq1 = DequantLaneNoOffset(word, 1, scale1);
                        const float deq2 = DequantLaneNoOffset(word, 2, scale2);
                        const float deq3 = DequantLaneNoOffset(word, 3, scale3);
                        const float deq4 = DequantLaneNoOffset(word, 4, scale4);
                        const float deq5 = DequantLaneNoOffset(word, 5, scale5);
                        const float deq6 = DequantLaneNoOffset(word, 6, scale6);
                        const float deq7 = DequantLaneNoOffset(word, 7, scale7);
                        KOMODO_ACCUM_ROW(acc0, x_value0);
                        KOMODO_ACCUM_ROW(acc1, x_value1);
                        KOMODO_ACCUM_ROW(acc2, x_value2);
                        KOMODO_ACCUM_ROW(acc3, x_value3);
                        KOMODO_ACCUM_ROW(acc4, x_value4);
                        KOMODO_ACCUM_ROW(acc5, x_value5);
                        KOMODO_ACCUM_ROW(acc6, x_value6);
                        KOMODO_ACCUM_ROW(acc7, x_value7);
                    }
                } else {
                    const float offset0 = OffsetValue(scale_base, zero_offsets);
                    const float offset1 = OffsetValue(scale_base + 1, zero_offsets);
                    const float offset2 = OffsetValue(scale_base + 2, zero_offsets);
                    const float offset3 = OffsetValue(scale_base + 3, zero_offsets);
                    const float offset4 = OffsetValue(scale_base + 4, zero_offsets);
                    const float offset5 = OffsetValue(scale_base + 5, zero_offsets);
                    const float offset6 = OffsetValue(scale_base + 6, zero_offsets);
                    const float offset7 = OffsetValue(scale_base + 7, zero_offsets);
                    float x_sum0 = 0.0f;
                    float x_sum1 = 0.0f;
                    float x_sum2 = 0.0f;
                    float x_sum3 = 0.0f;
                    float x_sum4 = 0.0f;
                    float x_sum5 = 0.0f;
                    float x_sum6 = 0.0f;
                    float x_sum7 = 0.0f;
                    for (uint32_t k = k_begin; k < k_end; ++k) {
                        const float x_value0 = static_cast<float>(x_gm_.GetValue(x_offset0 + k));
                        const float x_value1 = static_cast<float>(x_gm_.GetValue(x_offset1 + k));
                        const float x_value2 = static_cast<float>(x_gm_.GetValue(x_offset2 + k));
                        const float x_value3 = static_cast<float>(x_gm_.GetValue(x_offset3 + k));
                        const float x_value4 = static_cast<float>(x_gm_.GetValue(x_offset4 + k));
                        const float x_value5 = static_cast<float>(x_gm_.GetValue(x_offset5 + k));
                        const float x_value6 = static_cast<float>(x_gm_.GetValue(x_offset6 + k));
                        const float x_value7 = static_cast<float>(x_gm_.GetValue(x_offset7 + k));
                        x_sum0 += x_value0;
                        x_sum1 += x_value1;
                        x_sum2 += x_value2;
                        x_sum3 += x_value3;
                        x_sum4 += x_value4;
                        x_sum5 += x_value5;
                        x_sum6 += x_value6;
                        x_sum7 += x_value7;
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
                        KOMODO_ACCUM_ROW(acc0, x_value0);
                        KOMODO_ACCUM_ROW(acc1, x_value1);
                        KOMODO_ACCUM_ROW(acc2, x_value2);
                        KOMODO_ACCUM_ROW(acc3, x_value3);
                        KOMODO_ACCUM_ROW(acc4, x_value4);
                        KOMODO_ACCUM_ROW(acc5, x_value5);
                        KOMODO_ACCUM_ROW(acc6, x_value6);
                        KOMODO_ACCUM_ROW(acc7, x_value7);
                    }
                    KOMODO_APPLY_OFFSET_ROW(acc0, x_sum0);
                    KOMODO_APPLY_OFFSET_ROW(acc1, x_sum1);
                    KOMODO_APPLY_OFFSET_ROW(acc2, x_sum2);
                    KOMODO_APPLY_OFFSET_ROW(acc3, x_sum3);
                    KOMODO_APPLY_OFFSET_ROW(acc4, x_sum4);
                    KOMODO_APPLY_OFFSET_ROW(acc5, x_sum5);
                    KOMODO_APPLY_OFFSET_ROW(acc6, x_sum6);
                    KOMODO_APPLY_OFFSET_ROW(acc7, x_sum7);
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
#undef KOMODO_APPLY_OFFSET_ROW
#undef KOMODO_INIT_ACC
    }

    __aicore__ inline void ProcessRowQuad(
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
                const float offset00 = OffsetValue(scale_base0, zero_offsets);
                const float offset01 = OffsetValue(scale_base0 + 1, zero_offsets);
                const float offset02 = OffsetValue(scale_base0 + 2, zero_offsets);
                const float offset03 = OffsetValue(scale_base0 + 3, zero_offsets);
                const float offset04 = OffsetValue(scale_base0 + 4, zero_offsets);
                const float offset05 = OffsetValue(scale_base0 + 5, zero_offsets);
                const float offset06 = OffsetValue(scale_base0 + 6, zero_offsets);
                const float offset07 = OffsetValue(scale_base0 + 7, zero_offsets);
                const float offset10 = OffsetValue(scale_base1, zero_offsets);
                const float offset11 = OffsetValue(scale_base1 + 1, zero_offsets);
                const float offset12 = OffsetValue(scale_base1 + 2, zero_offsets);
                const float offset13 = OffsetValue(scale_base1 + 3, zero_offsets);
                const float offset14 = OffsetValue(scale_base1 + 4, zero_offsets);
                const float offset15 = OffsetValue(scale_base1 + 5, zero_offsets);
                const float offset16 = OffsetValue(scale_base1 + 6, zero_offsets);
                const float offset17 = OffsetValue(scale_base1 + 7, zero_offsets);

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
        uint32_t zero_offsets,
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
                const float offset00 = OffsetValue(scale_base0, zero_offsets);
                const float offset01 = OffsetValue(scale_base0 + 1, zero_offsets);
                const float offset02 = OffsetValue(scale_base0 + 2, zero_offsets);
                const float offset03 = OffsetValue(scale_base0 + 3, zero_offsets);
                const float offset04 = OffsetValue(scale_base0 + 4, zero_offsets);
                const float offset05 = OffsetValue(scale_base0 + 5, zero_offsets);
                const float offset06 = OffsetValue(scale_base0 + 6, zero_offsets);
                const float offset07 = OffsetValue(scale_base0 + 7, zero_offsets);
                const float offset10 = OffsetValue(scale_base1, zero_offsets);
                const float offset11 = OffsetValue(scale_base1 + 1, zero_offsets);
                const float offset12 = OffsetValue(scale_base1 + 2, zero_offsets);
                const float offset13 = OffsetValue(scale_base1 + 3, zero_offsets);
                const float offset14 = OffsetValue(scale_base1 + 4, zero_offsets);
                const float offset15 = OffsetValue(scale_base1 + 5, zero_offsets);
                const float offset16 = OffsetValue(scale_base1 + 6, zero_offsets);
                const float offset17 = OffsetValue(scale_base1 + 7, zero_offsets);

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
                const float offset0 = OffsetValue(scale_base, zero_offsets);
                const float offset1 = OffsetValue(scale_base + 1, zero_offsets);
                const float offset2 = OffsetValue(scale_base + 2, zero_offsets);
                const float offset3 = OffsetValue(scale_base + 3, zero_offsets);
                const float offset4 = OffsetValue(scale_base + 4, zero_offsets);
                const float offset5 = OffsetValue(scale_base + 5, zero_offsets);
                const float offset6 = OffsetValue(scale_base + 6, zero_offsets);
                const float offset7 = OffsetValue(scale_base + 7, zero_offsets);

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
        const uint32_t raw = (word >> shift) & 0xFU;
        const int32_t signed_w = raw < 8U ? static_cast<int32_t>(raw) : static_cast<int32_t>(raw) - 16;
        return (static_cast<float>(signed_w) + offset) * scale;
    }

    __aicore__ inline float DequantLaneNoOffset(uint32_t word, uint32_t lane, float scale)
    {
        const uint32_t shift = lane << 2;
        const uint32_t raw = (word >> shift) & 0xFU;
        const int32_t signed_w = raw < 8U ? static_cast<int32_t>(raw) : static_cast<int32_t>(raw) - 16;
        return static_cast<float>(signed_w) * scale;
    }

#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
    }
#endif

    GlobalTensor<half> x_gm_;
    GlobalTensor<int32_t> packed_weight_gm_;
    GlobalTensor<half> scales_gm_;
    GlobalTensor<half> offsets_gm_;
    GlobalTensor<half> bias_gm_;
    GlobalTensor<half> y_gm_;
#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
    GlobalTensor<half> staged_weight_gm_;
#ifdef KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT
    static constexpr uint32_t kCann9VectorDequantLanes = 8;
    static constexpr uint32_t kCann9VectorDequantPackedWords = 1;
    static constexpr uint32_t kCann9VectorDequantPackedBytes = 32;
    static constexpr uint32_t kCann9VectorDequantHalfBytes = 32;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> vector_packed_ub_;
    TBuf<TPosition::VECCALC> vector_half_ub_;
    bool vector_dequant_ready_ = false;
#endif
#endif
    const KomodoCannW4A16MatmulTilingData* tiling_;
};
}  // namespace

template <int LAUNCH_MODE>
__global__ __aicore__ void komodo_cann_w4_a16_matmul(
    GM_ADDR x,
    GM_ADDR packed_weight,
    GM_ADDR scales,
    GM_ADDR offsets,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
#ifdef KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GET_TILING_DATA(tiling_data, tiling);
#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
    GM_ADDR user_workspace = workspace;
    if (tiling_data.kernel_mode == kKernelModeStagedDequant && tiling_data.staging_workspace_bytes != 0) {
        if (workspace == nullptr) {
            return;
        }
        AscendC::SetSysWorkspaceForce(workspace);
        user_workspace = AscendC::GetUserWorkspace(workspace);
    }
#endif
#ifdef KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER
#ifdef KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH
    if (workspace == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(workspace);
    AscendC::clearWorkspace(reinterpret_cast<__gm__ uint8_t*>(workspace));
    TPipe cube_pipe;
    KomodoCannW4A16CubeConsumerProbe cube_probe;
    TCubeTiling cube_tiling = MakeCubeConsumerTiling(&tiling_data);
    REGIST_MATMUL_OBJ(&cube_pipe, GetSysWorkSpacePtr(), cube_probe.mm, &cube_tiling);
    if ASCEND_IS_AIC {
        return;
    }
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF
    cube_probe.InitTscmBTile(cube_pipe, &tiling_data);
#elif defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
    cube_probe.InitVecoutBTile(cube_pipe, &tiling_data);
#endif
#else
    if ASCEND_IS_AIC {
        if (workspace == nullptr) {
            return;
        }
        AscendC::SetSysWorkspaceForce(workspace);
        TPipe cube_pipe;
        KomodoCannW4A16CubeConsumerProbe cube_probe;
        TCubeTiling cube_tiling = MakeCubeConsumerTiling(&tiling_data);
        REGIST_MATMUL_OBJ(&cube_pipe, GetSysWorkSpacePtr(), cube_probe.mm, &cube_tiling);
        return;
    }
#endif
#else
    if ASCEND_IS_AIC {
        return;
    }
#endif
    KomodoCannW4A16ScalarKernel op;
#ifdef KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT
    op.Init(x, packed_weight, scales, offsets, bias, y, user_workspace, &tiling_data);
#else
    op.Init(x, packed_weight, scales, offsets, bias, y, workspace, &tiling_data);
#endif
#ifdef KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF
    if (op.TryProcessSingleKTileTscmHandoff(cube_probe)) {
        return;
    }
#elif defined(KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF)
    if (op.TryProcessVecoutHandoff(cube_probe)) {
        return;
    }
#endif
    op.Process();
}
