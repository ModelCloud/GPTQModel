#include "../../torch_stable_compat.h"
#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <algorithm>
#include <limits>
#include <set>
#include <shared_mutex>
#include "../hgemm.cuh"
#include "../util.h"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "exl3_gemv.cuh"
#include "hadamard.cuh"
#include "reconstruct.cuh"

namespace {
std::set<void*> kernel_attr_set[MAX_DEVICES];
std::shared_mutex kernel_attr_mutex;
}

int exl3_gemm(
    const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
    const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
    bool mcg, bool mul1, bool fp16_gemv, int64_t int8_gemv, bool fused_gemm, bool q4nx)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    TORCH_CHECK(false,
        "ExLlamaV3 tensor-core GEMM is not available in this build (ROCm/HIP); "
        "use the reconstruct + hgemm path (torch_mode) instead.");
#else
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = A.get_device();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    const bool c_fp32 = C.scalar_type() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(int(mcg) + int(mul1) + int(q4nx) <= 1, "conflicting EXL3 codebooks");
    TORCH_CHECK(int8_gemv >= 0 && int8_gemv <= 2, "EXL3 INT8 mode must be 0, 1 or 2");

    const int64_t words_per_tile = B.size(2);
    TORCH_CHECK(words_per_tile >= 16 && words_per_tile <= 128, "invalid EXL3 trellis width");
    const int K = words_per_tile / 16;
    const bool half_k = words_per_tile % 16 == 8;
    if (q4nx)
    {
        TORCH_CHECK(words_per_tile % 4 == 0 && q4nx_supports_step_bits(words_per_tile / 4),
                    "Q4NX requires 2.5, 3.0, or 3.5 bpw");
    }
    else
    {
        TORCH_CHECK(words_per_tile % 16 == 0 || (half_k && K <= 3 && mul1),
                    "EXL3 half-integer bitrates require mul1 at 1.5, 2.5 or 3.5 bpw");
    }

    if (C.numel() == 0) return 0;
    TORCH_CHECK(A.size(-1) > 0 && A.size(-1) % 128 == 0 && C.size(-1) % 128 == 0,
                "EXL3 fused transforms require K and N divisible by 128");
    const int64_t rows = A.numel() / A.size(-1);
    TORCH_CHECK(rows == C.numel() / C.size(-1), "inconsistent EXL3 input and output rows");
    TORCH_CHECK(std::max(A.numel(), C.numel()) <= std::numeric_limits<int>::max(),
                "EXL3 kernel supports up to INT32_MAX activation elements");
    const int size_m = rows;
    const int size_k = A.size(-1);
    const int size_n = C.size(-1);
    const int cb = q4nx ? EXL3_Q4NX_CODEBOOK : mcg ? 1 : mul1 ? 2 : 0;
    const int num_sms = DevCtx::instance().get_num_sms(device);
    const int cc = DevCtx::instance().get_cc(device);

    // On the measured Ampere prefill shapes, decoding once and reusing the
    // weights across rows is faster than decoding for each packed row tile.
    // Cap the per-call expanded matrix at 48 MiB and require more rows for
    // the wider projections. Keep the HGEMM result wide through normalization.
    const int64_t weight_elements = (int64_t) size_k * size_n;
    if (q4nx && c_fp32 && cc == CC_AMPERE && !fused_gemm &&
        weight_elements >= 4 * 1024 * 1024 && weight_elements <= 24 * 1024 * 1024 &&
        size_m >= (weight_elements <= 12 * 1024 * 1024 ? 64 : 128))
    {
        try
        {
            auto weight = at::empty({size_k, size_n}, at::options(A));
            reconstruct_q4nx(weight, B, K, half_k);
            auto x = A.view({size_m, size_k});
            auto xh = A_had.view({size_m, size_k});
            auto y = C.view({size_m, size_n});
            had_r_128(x, xh, suh, c10::nullopt, 1.0f);
            hgemm(xh, weight, y);
            had_r_128(y, y, c10::nullopt, svh, EXL3_Q4NX_ALPHA);
            return 92;
        }
        catch (const c10::OutOfMemoryError&)
        {
            // The expanded weight and HGEMM workspace are optional scratch.
        }
    }

    if (int8_gemv && mul1 && size_m <= (int8_gemv == 1 ? 1 : 2)
        && exl3_gemv_int8(A, B, C, suh, A_had, svh, int8_gemv, stream))
        return 91;

    const half* A_ptr = reinterpret_cast<const half*>(A.const_data_ptr());
    const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(B.const_data_ptr());
    void* C_ptr = C.data_ptr();
    const half* su = reinterpret_cast<const half*>(suh.const_data_ptr());
    half* xh = reinterpret_cast<half*>(A_had.data_ptr());
    const half* sv = reinterpret_cast<const half*>(svh.const_data_ptr());
    int* locks = nullptr;
    void* args[] = {
        (void*)&A_ptr, (void*)&B_ptr, (void*)&C_ptr, (void*)&size_m, (void*)&size_k,
        (void*)&size_n, (void*)&locks, (void*)&su, (void*)&xh, (void*)&sv
    };
    if (fp16_gemv && exl3_gemv_try_launch(args, size_m, size_k, size_n, K, half_k, cb, c_fp32, device, stream))
        return 90;

    const bool use_fused = fused_gemm || half_k || q4nx;
    const int smem_max = std::min<int>(SMEM_MAX, at::cuda::getDeviceProperties(device)->sharedMemPerBlockOptin);
    const TResult* config = select_exl3_gemm_kernel_tuned(
        cc, size_k, size_n, K, c_fp32, -1, 0, cb, half_k, use_fused, size_m, smem_max);
    TORCH_CHECK(config && config->kernel, "no EXL3 GEMM kernel for this shape");
    void* kernel = reinterpret_cast<void*>(config->kernel);

    // The split-K counters belong to this launch (#4619). The caching
    // allocator preserves captured addresses and keeps streams isolated.
    const int lock_rows = use_fused ? 1 : CEIL_DIVIDE(size_m, EXL3_GEMM_TILESIZE_M);
    auto locks_tensor = at::empty({(int64_t) lock_rows * (size_n / 16)},
                                  at::options(A).dtype(at::kInt));
    locks = locks_tensor.mutable_data_ptr<int>();
    C10_CUDA_CHECK(cudaMemsetAsync(
        locks, 0,
        static_cast<size_t>(locks_tensor.numel()) * locks_tensor.element_size(),
        stream));

    bool attr_ready;
    {
        const std::shared_lock<std::shared_mutex> guard(kernel_attr_mutex);
        attr_ready = kernel_attr_set[device].find(kernel) != kernel_attr_set[device].end();
    }
    if (!attr_ready)
    {
        const std::unique_lock<std::shared_mutex> guard(kernel_attr_mutex);
        if (kernel_attr_set[device].find(kernel) == kernel_attr_set[device].end())
        {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max));
            kernel_attr_set[device].insert(kernel);
        }
    }

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        kernel, MIN(num_sms, config->num_sms), config->block_dim, args, smem_max, stream));
    return config->shape_idx;
#endif
}
