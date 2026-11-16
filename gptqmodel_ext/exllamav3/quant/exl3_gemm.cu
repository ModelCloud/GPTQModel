#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include <set>
#include <shared_mutex>

constexpr int EXL3_GEMM_SMEM_MAX = 90 * 1024;

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

// One entry per (device, kernel) records that the shared-memory attribute was
// already raised.  The set is process-wide and is filled on the first launch of
// each kernel variant, which GIL=0 callers can reach concurrently; an
// unsynchronized find/insert on a std::set is a data race.  The steady state
// only reads, so a shared lock keeps concurrent launches parallel and the
// exclusive lock is taken only the first time a kernel variant is seen on a
// device.
std::set<void*> kernel_attr_set[MAX_DEVICES] = {};
std::shared_mutex kernel_attr_mutex;

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    // TORCH_CHECK_SHAPES(A, 0, C, 0, 1);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    // Get SU, optionally
    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = nullptr;
    if (suh_ptr)
    {
        // TORCH_CHECK_SHAPES(suh.value(), 0, A, 1, 1);
        A_had_ptr = (half*) OPTPTR(A_had);
        // TORCH_CHECK(A_had_ptr, "Must supply A_had with suh");
        // TORCH_CHECK_SHAPES_FULL(A_had.value(), A);
    }

    // Get SV, optionally
    const half* svh_ptr = (const half*) OPTPTR(svh);
    // if (svh_ptr)
        // TORCH_CHECK_SHAPES(svh.value(), 0, B, 1, 16);

    // Device properties
    int device;
    cudaGetDevice(&device);
    // Per-device tables below are MAX_DEVICES entries. Reject an ordinal that
    // would index past them instead of writing host memory out of bounds.
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES,
                "EXL3 device ordinal ", device, " exceeds MAX_DEVICES=", MAX_DEVICES);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    int K = B.size(2) / 16;
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel;

    TResult* tr = select_exl3_gemm_kernel_tuned(cc, size_k, size_n, K, c_fp32, force_shape_idx, force_num_sms, cb);
    if (!tr) return 0;
    num_sms = MIN(num_sms, tr->num_sms);
    kernel = tr->kernel;
    block_dim = tr->block_dim;
    shape_idx = tr->shape_idx;

    // On ROCm the tensor-core GEMM kernels are not implemented, so the kernel
    // tables are populated with null entries (see exl3_kernel_map.cuh). Fail
    // with an actionable message instead of launching a null pointer.
    TORCH_CHECK(
        kernel != nullptr,
        "ExLlamaV3 tensor-core GEMM is not available in this build (ROCm/HIP); "
        "use the reconstruct + hgemm path (torch_mode) instead."
    );

    // Launch
    bool attr_ready;
    {
        std::shared_lock<std::shared_mutex> guard(kernel_attr_mutex);
        attr_ready = kernel_attr_set[device].find((void*) kernel) != kernel_attr_set[device].end();
    }
    if (!attr_ready)
    {
        // First launch of this kernel variant on this device.  Re-check under the
        // exclusive lock: another thread may have raised the attribute while this
        // one waited for it.
        std::unique_lock<std::shared_mutex> guard(kernel_attr_mutex);
        if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
        {
            // HIP only provides hipFuncSetAttribute(const void*, ...); the CUDA
            // template overload that accepts a function pointer directly has no HIP
            // equivalent, so cast to the void* form that both runtimes expose.
            cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, EXL3_GEMM_SMEM_MAX);
            cuda_check(cudaPeekAtLastError());
            // Publish only after the attribute call succeeded, so a failed setup
            // is retried by the next launch instead of being treated as done.
            kernel_attr_set[device].insert((void*) kernel);
        }
    }
    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        EXL3_GEMM_SMEM_MAX,
        stream
    );

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}
