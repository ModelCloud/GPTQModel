// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_devctx.cuh"
#include <map>
#include <mutex>
#include <c10/cuda/CUDAException.h>

// Small-batch FP16 GEMV with the upstream shape heuristic.
// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
static int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int narrow_coresident)
{
    if (K < 2 || K > 4) return -1;
    if (K != 4 && cb == 0) return -1;
    if (size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here

    // The narrow config wins (up to ~30%) whenever its grid fits in a single co-resident wave;
    // in the 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small (small k). The wide config covers a band of large-n shapes with
    // small-to-mid k. Everything else runs the regular block-pipelined kernel.
    // Per-bits envelopes: 2 bpw is decode-bound and won at every measured shape on both archs;
    // 3 bpw wins everywhere on Ada but only in the narrow envelope on Ampere
    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

void* exl3_gemv_select_kernel_half(int bits, bool c_fp32, int mmode, int cfg, bool smem);
void* exl3_gemv_select_kernel_q4nx(int bits, bool half_k, int mmode, int cfg);

// bound: the narrow config's two-blocks-per-SM instance (the wide config has one form)
static void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem, bool bound)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (cfg_ == 0 && bound) ? (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_, false, cfg_ == 0> \
                                        : (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}

bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream
)
{
    const bool q4nx = cb == EXL3_Q4NX_CODEBOOK;
    if (q4nx)
    {
        if (!c_fp32 || !q4nx_supports_step_bits(4 * K + 2 * half_k)) return false;
    }
    else if (half_k)
    {
        if (K < 1 || K > 3 || cb != 2) return false;
    }
    else
    {
        if (K < 2 || K > 4) return false;
        if (K != 4 && cb == 0) return false;
    }
    if (size_m > EXL3_GEMV_MAX_M) return false;
    if (size_k % 128 || size_n % 128) return false;

    int cc = DevCtx::instance().get_cc(device);
    // Q4NX small-batch decode measured on the RTX 4090 (sm_89, 128 SMs, m 1..8 x k 2048..8192 x
    // n 2048..16384 x 2.5 / 3.0 / 3.5 bpw, tests/benchmark_exl3_q4nx_gemv_ada.py): the GEMV is
    // 1.03-1.72x the block-pipelined kernel cold and 1.00-1.72x warm (geomean 1.19x / 1.30x), so
    // Ada joins Ampere. The same matrix on an RTX 5090 (sm_120, 170 SMs, 96 MiB L2,
    // tests/benchmark_exl3_q4nx_gemv_blackwell.py) keeps Blackwell out: the sm_120 GEMV is 0.16-0.82
    // relative off the reconstructed FP32 weight in all 180 cells, mul1 W3 and mcg W2/W4 miss by
    // 0.26-0.43 on the same device, and the speedup is only 1.06x cold / 1.18x warm with 61
    // regressing cells. sm_120 runs the block-pipelined kernel.
    if (q4nx && cc != CC_AMPERE && cc != CC_ADA) return false;
    int mmode = size_m == 1 ? 0 : 1;
    int num_sms = DevCtx::instance().get_num_sms(device);

    // Cooperative launch: grids are capped at full co-residency (cached per kernel), and the
    // narrow config's co-residency also feeds the shape heuristic
    static std::map<void*, int> occ_cache[MAX_DEVICES];
    static std::mutex occ_mutex;
    auto& cache = occ_cache[device];
    auto occupancy = [&] (void* kernel, int block_dim) -> int
    {
        const std::lock_guard<std::mutex> guard(occ_mutex);
        auto it = cache.find(kernel);
        if (it != cache.end()) return it->second;
        int blocks_per_sm;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
        cache[kernel] = blocks_per_sm;
        return blocks_per_sm;
    };

    // Each warp loads packed words into registers and shuffles the words it needs.
    constexpr bool smem = false;

    auto select = [&] (int cfg_, bool bound) -> void*
    {
        if (q4nx) return exl3_gemv_select_kernel_q4nx(K, half_k, mmode, cfg_);
        return half_k ? exl3_gemv_select_kernel_half(K, c_fp32, mmode, cfg_, smem)
                      : exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg_, smem, bound);
    };
    void* narrow_kernel = select(0, true);
    if (!narrow_kernel) return false;
    int narrow_coresident = occupancy(narrow_kernel, 512) * num_sms;

    // Shape heuristic: a half-integer rate K + 0.5 is handled like the integer rate above it (its tile is
    // between the two in bytes; unmeasured, so it inherits the K + 1 envelope)
    // Q4NX envelope on Ada, from the same sweep. The narrow config (one block per 32 output
    // columns) wins at every measured shape whose grid fits one co-resident wave, including the
    // single-wave n <= 2048 grids where it still fills only half the SMs. Two measured exceptions
    // shape the envelope:
    // - a 3.0 bpw tile of >= 64 Mi elements (k = n = 8192) is 1.03-1.06x cold but 0.87-0.90x warm:
    //   at that size the packed kernel keeps the trellis in L2 and wins, so Ada falls back;
    // - past the narrow grid (n > 8192) the wide config only wins for small k, where its 64-column
    //   tiles still cover the SMs: 1.05-1.20x for 2.5 / 3.5 bpw at k <= 2048, but 0.73-1.06x
    //   elsewhere, and 3.0 bpw is 0.98x warm there. Ada enables it for 2.5 / 3.5 bpw with
    //   k <= 2048 / n >= 12288 only; Ampere keeps the unconditional wide-config fallback.
    int cfg;
    if (q4nx)
    {
        const int step_bits = 4 * K + 2 * half_k;
        const bool narrow_fits = size_n / 32 <= narrow_coresident;
        const bool ada_packed_l2_win =
            cc == CC_ADA && step_bits == 12 && (int64_t) size_k * size_n >= 64 * 1024 * 1024;
        const bool ada_wide = cc == CC_ADA && step_bits != 12 && size_k <= 2048 && size_n >= 12288;
        if (narrow_fits && !ada_packed_l2_win) cfg = 0;
        else if (cc == CC_ADA) cfg = ada_wide ? 1 : -1;
        else cfg = 1;
    }
    else
        cfg = exl3_gemv_cfg(cc, size_m, size_k, size_n, half_k ? K + 1 : K, cb, narrow_coresident);
    if (cfg < 0) return false;

    void* kernel = cfg == 0 ? narrow_kernel : select(cfg, false);
    if (!kernel) return false;

    // A narrow grid that fits in one wave without the bound runs the unbounded integer instance
    if (cfg == 0 && !half_k && !q4nx)
    {
        void* unbounded = select(0, false);
        if (unbounded && size_n / 32 <= occupancy(unbounded, 512) * num_sms) kernel = unbounded;
    }

    int block_dim = cfg == 0 ? 512 : 256;
    int cols = cfg == 0 ? 32 : 64;

    int max_blocks = occupancy(kernel, block_dim) * num_sms;
    if (q4nx && size_n / cols > max_blocks) return false;
    int grid = MIN(size_n / cols, max_blocks);
    if (grid < 1) return false;

    auto status = cudaLaunchCooperativeKernel
    (
        kernel,
        dim3(grid),
        dim3(block_dim),
        kernel_args,
        0,
        stream
    );
    if (status == cudaErrorNotSupported || status == cudaErrorCooperativeLaunchTooLarge)
    {
        cudaGetLastError();
        return false;
    }
    C10_CUDA_CHECK(status);
    return true;
}
#endif
