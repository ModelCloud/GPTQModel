// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#include <cuda_fp16.h>
#include "exl3_gemv.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "comp_units/exl3_gemv_int8_instances.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"
#include "hadamard_inner.cuh"
#include <cooperative_groups.h>
#include <mutex>
#include <c10/cuda/CUDAException.h>
#include <set>
#include <map>


// The function attribute and occupancy tables contain host metadata only.
static std::mutex gemv_launch_cache_mutex;
static std::set<void*> gemv_attr_set[MAX_DEVICES];
static std::map<std::pair<void*, size_t>, int> gemv_occ_cache[MAX_DEVICES];

typedef void (*gemv_int8_coop_fn)
    (const half*, const uint16_t*, void*, int, int, int, int*, const half*, half*, const half*);

static void* select_gemv_int8_kernel(int K, bool half_k, bool c_fp32, bool residual)
{
    if (half_k) switch (K)
    {
        case 1: return exl3_gemv_int8_coop_sel_h1(c_fp32, residual);
        case 2: return exl3_gemv_int8_coop_sel_h2(c_fp32, residual);
        case 3: return exl3_gemv_int8_coop_sel_h3(c_fp32, residual);
        default: return nullptr;
    }
    switch (K)
    {
        case 1: return exl3_gemv_int8_coop_sel_k1(c_fp32, residual);
        case 2: return exl3_gemv_int8_coop_sel_k2(c_fp32, residual);
        case 3: return exl3_gemv_int8_coop_sel_k3(c_fp32, residual);
        case 4: return exl3_gemv_int8_coop_sel_k4(c_fp32, residual);
        case 5: return exl3_gemv_int8_coop_sel_k5(c_fp32, residual);
        case 6: return exl3_gemv_int8_coop_sel_k6(c_fp32, residual);
        case 7: return exl3_gemv_int8_coop_sel_k7(c_fp32, residual);
        case 8: return exl3_gemv_int8_coop_sel_k8(c_fp32, residual);
    }
    return nullptr;
}

static void* select_gemv_int8_sq_kernel(int K, bool half_k, int M, bool c_fp32, bool residual)
{
    if (half_k) switch (K)
    {
        case 1: return exl3_gemv_int8_sq_sel_h1(M, c_fp32, residual);
        case 2: return exl3_gemv_int8_sq_sel_h2(M, c_fp32, residual);
        case 3: return exl3_gemv_int8_sq_sel_h3(M, c_fp32, residual);
        default: return nullptr;
    }
    switch (K)
    {
        case 1: return exl3_gemv_int8_sq_sel_k1(M, c_fp32, residual);
        case 2: return exl3_gemv_int8_sq_sel_k2(M, c_fp32, residual);
        case 3: return exl3_gemv_int8_sq_sel_k3(M, c_fp32, residual);
        case 4: return exl3_gemv_int8_sq_sel_k4(M, c_fp32, residual);
        case 5: return exl3_gemv_int8_sq_sel_k5(M, c_fp32, residual);
        case 6: return exl3_gemv_int8_sq_sel_k6(M, c_fp32, residual);
    }
    return nullptr;
}

// Cap temporary storage to the upstream workspace bound. Each invocation owns
// its counters and partial sums, so graph replays on different streams cannot
// alias. The caching allocator keeps captured addresses stable.
constexpr size_t GEMV_INT8_WS_INTS = 16 * 1024 * 1024 / sizeof(int);
// m == 1 fast path: per-slice-scale kernel, regular launch. Returns false to fall through to the
// cooperative kernel (and from there to the regular fp16 kernel).
static bool exl3_gemv_int8_sq
(
    const half* A_ptr, const uint16_t* B_ptr, void* C_ptr,
    int size_m, int size_k, int size_n, int K, bool half_k, bool c_fp32, bool residual,
    const half* suh_ptr, half* A_had_ptr, const half* svh_ptr,
    const at::TensorOptions& options, int device, int num_sms, cudaStream_t stream
)
{
    if (size_m > 2) return false;
    int M = size_m;
    void* fn = select_gemv_int8_sq_kernel(K, half_k, M, c_fp32, residual);
    if (!fn) return false;

    int rows_max = gemv_int8_sq_rows_max(M, residual);

    // Mirror of the kernel's work decomposition (single-wave rule with a half-wave floor)
    auto decomp = [&] (int grid_, int& ksplit, int& rows_per)
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int r = CEIL_DIVIDE(rows_total * nb256, grid_);
        rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = (size_t) gemv_int8_stage_bytes(K, half_k);
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M * (residual ? 2 : 1)
               + stage + (size_t) 2 * M * 128 * 4;
    };

    int maxb;
    int ksplit, rows_per;
    decomp(6 * num_sms, ksplit, rows_per);
    const size_t smem_guess = smem_for(rows_per);
    {
        const std::lock_guard<std::mutex> guard(gemv_launch_cache_mutex);
        if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
        {
            C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_for(rows_max)));
            // Match the tensor-core kernels' carveout to avoid SM reconfiguration
            // on a transition between adjacent layers during graph replay.
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
            gemv_attr_set[device].insert(fn);
        }
        auto occ_key = std::make_pair(fn, smem_guess);
        auto occ_it = gemv_occ_cache[device].find(occ_key);
        if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
        else
        {
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess));
            gemv_occ_cache[device][occ_key] = maxb;
        }
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);
    if (ksplit > SQ_KSPLIT_CAP) return false;
    if (size_n / 256 > SQ_COUNTERS_CAP) return false;

    int pstride = size_n * (residual ? 2 : 1);
    const size_t ws_ints = SQ_WS_RESERVED + (size_t) ksplit * M * pstride;
    if (ws_ints > GEMV_INT8_WS_INTS) return false;
    auto ws = at::empty({(int64_t) ws_ints}, options.dtype(at::kInt));
    int* ws_ptr = ws.data_ptr<int>();
    // The first region is the completion counters. Partial sums and scales
    // have exclusive or idempotent producers before the CTA that consumes them.
    C10_CUDA_CHECK(cudaMemsetAsync(ws_ptr, 0, sizeof(int) * (size_n / 256), stream));

    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr
    };

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), kernelArgs, smem, stream);
    if (err == cudaErrorNotSupported || err == cudaErrorCooperativeLaunchTooLarge)
    {
        // The matmul was declined. The caller can use the FP16 kernel.
        cudaGetLastError();
        return false;
    }
    C10_CUDA_CHECK(err);
    return true;
}

bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    int mode,
    cudaStream_t stream
)
{

    // 16 * K uint16 per tile, 16 * K + 8 at the half-integer rates (mul1 only, which this path is anyway)
    const int tile_u16 = B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    if (half_k && (tile_u16 % 16 != 8 || K > 3)) return false;
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int size_m = A.numel() / size_k;
    if (size_n % 256) return false;
    if (size_k % 128) return false;

    int device;
    cudaGetDevice(&device);
    // Half-integer rates take the gate of the integer rate above them
    int cc = DevCtx::instance().get_cc(device);
    if (K < 1 || K + (half_k ? 1 : 0) > (cc >= CC_HOPPER ? 6 : 5)) return false;

    // Both int8 kernels below size their shared memory against a fixed ~80 KB budget baked into
    // gemv_int8_sq_rows_max() and the coop path's 768-row bound, and - critically - the kernels
    // recompute rows_per from those same constants on the device, so the host cannot simply ask
    // for less without desyncing the two. On Turing (64 KB) the cudaFuncSetAttribute would fail
    // and the following cuda_check would abort the process. This path is a decode-time
    // optimization over the regular fp16 tensor-core kernel, which handles the same work, so
    // declining is a performance loss and nothing more.
    if (at::cuda::getDeviceProperties(device)->sharedMemPerBlockOptin < 80 * 1024) return false;

    int num_sms = DevCtx::instance().get_num_sms(device);
    bool c_fp32 = C.dtype() == at::kFloat;
    bool residual = mode == 1;

    // Per-slice-scale kernel: m == 1, plus m == 2 in plain int8 mode (rows share the decoded
    // weights and the B stream; measured on 3090, larger m and batched residual lose to the fp16
    // tensor-core kernel). Falls through to the cooperative kernel on a constraint miss at m == 1;
    // batched rows beyond the gate go straight to the regular kernel.
    if (size_m <= (residual ? 1 : 2) && exl3_gemv_int8_sq(
        (const half*) A.data_ptr(), (const uint16_t*) B.data_ptr(), C.data_ptr(),
        size_m, size_k, size_n, K, half_k, c_fp32, residual,
        (const half*) suh.data_ptr(), (half*) A_had.data_ptr(), (const half*) svh.data_ptr(),
        A.options(), device, num_sms, stream))
        return true;
    if (size_m > 1) return false;

    void* fn = select_gemv_int8_kernel(K, half_k, c_fp32, residual);
    if (!fn) return false;

    // Mirror the kernel's work decomposition for the shared memory size; grid = max co-resident
    // blocks (natural register allocation measures faster than forcing higher occupancy)
    auto smem_for_grid = [&] (int grid_) -> size_t
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int smem_rows_max = residual ? 384 : 768;
        int ksplit = CEIL_DIVIDE(4 * grid_, nb256);
        ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
        ksplit = MIN(ksplit, rows_total);
        int rows_per = CEIL_DIVIDE(rows_total, ksplit);
        size_t stage = (size_t) gemv_int8_stage_bytes(K, half_k);
        return MAX((size_t) rows_per * 16 * 4 * (residual ? 2 : 1) + stage, (size_t) 8 * 128 * 4);
    };

    int maxb;
    const size_t smem_guess = smem_for_grid(6 * num_sms);
    {
        const std::lock_guard<std::mutex> guard(gemv_launch_cache_mutex);
        if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
        {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 768 * 16 * 4 + GEMV_STAGE_MAX_BYTES));
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
            gemv_attr_set[device].insert(fn);
        }
        auto occ_key = std::make_pair(fn, smem_guess);
        auto occ_it = gemv_occ_cache[device].find(occ_key);
        if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
        else
        {
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess));
            gemv_occ_cache[device][occ_key] = maxb;
        }
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    size_t smem = smem_for_grid(grid);

    // The cooperative kernel initializes the accumulators before its first grid barrier.
    const size_t ws_ints = (size_t) 2 * size_n + 4 * size_m + 1024;
    if (ws_ints > GEMV_INT8_WS_INTS) return false;
    auto ws = at::empty({(int64_t) ws_ints}, A.options().dtype(at::kInt));
    int* ws_ptr = ws.data_ptr<int>();

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = C.data_ptr();
    const half* suh_ptr = (const half*) suh.data_ptr();
    half* A_had_ptr = (half*) A_had.data_ptr();   // scratch; used through a raw half* like the regular kernel
    const half* svh_ptr = (const half*) svh.data_ptr();

    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr
    };

    cudaError_t err = cudaLaunchCooperativeKernel(fn, grid, NUM_THREADS, kernelArgs, smem, stream);
    if (err == cudaErrorNotSupported || err == cudaErrorCooperativeLaunchTooLarge)
    {
        // The caller can use the FP16 kernel.
        cudaGetLastError();
        return false;
    }
    C10_CUDA_CHECK(err);
    return true;
}

#else  // __HIP_PLATFORM_AMD__ || USE_ROCM

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/ops/empty.h>
#include <cstdint>

#include "exl3_gemv.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "../util.h"
#include "../util.cuh"

// ROCm int8-activation GEMV for mul1 (cb 2) trellises. The CUDA kernel is one cooperative
// launch; HIP has no grid barrier, so the header's HIP entry point runs the same phases as four
// launches on the caller's stream, exactly like the fp16 HIP GEMV. Every decline returns false
// before the first launch, so the caller can fall through to the fp16 HIP GEMV (and from there
// to reconstruct + hgemm) without any buffer having been touched.
bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    int mode,
    cudaStream_t stream
)
{
    // 16 * K uint16 per tile; the half-integer rates have no int8 instantiation
    const int64_t tile_u16 = B.size(2);
    if (tile_u16 % 16 != 0) return false;
    const int bits = int(tile_u16 / 16);
    if (bits < 2 || bits > 4) return false;
    if (mode != 1 && mode != 2) return false;
    const bool residual = mode == 1;

    const int size_k = int(A.size(-1));
    const int size_n = int(B.size(1) * 16);
    if (size_k <= 0 || size_n <= 0) return false;
    const int size_m = int(A.numel() / size_k);
    // Same gate as the CUDA path: residual mode is m == 1, plain int8 mode m <= 2
    if (size_m < 1 || size_m > (residual ? 1 : 2)) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (C.dtype() != at::kFloat) return false;

    const int device = A.get_device();
    if (!exl3_gemv_int8_hip_host::arch_ok(device)) return false;
    // A_had is the CUDA kernel's scratch; the HIP pipeline stages the input Hadamard in shared
    // memory instead, so the caller's buffer is left untouched.
    (void) A_had;

    const auto quant_options = A.options().dtype(at::kChar);
    at::Tensor qa = at::empty({(int64_t) size_m * size_k}, quant_options);
    at::Tensor qb = at::empty({(int64_t) size_m * size_k}, quant_options);
    at::Tensor qscale = at::empty({size_m}, A.options().dtype(at::kFloat));
    at::Tensor qsum = at::empty({size_m}, A.options().dtype(at::kInt));
    at::Tensor qsum2 = at::empty({size_m}, A.options().dtype(at::kInt));

    const int status = exl3_gemv_int8_hip_launch
    (
        reinterpret_cast<const half*>(A.const_data_ptr()),
        reinterpret_cast<const uint16_t*>(B.const_data_ptr()),
        C.data_ptr(),
        reinterpret_cast<const half*>(suh.const_data_ptr()),
        reinterpret_cast<const half*>(svh.const_data_ptr()),
        reinterpret_cast<int8_t*>(qa.data_ptr()),
        reinterpret_cast<int8_t*>(qb.data_ptr()),
        reinterpret_cast<float*>(qscale.data_ptr()),
        reinterpret_cast<int*>(qsum.data_ptr()),
        reinterpret_cast<int*>(qsum2.data_ptr()),
        size_m, size_k, size_n, bits, residual, device, stream
    );
    if (status < 0) C10_CUDA_KERNEL_LAUNCH_CHECK();
    return status > 0;
}

#endif
