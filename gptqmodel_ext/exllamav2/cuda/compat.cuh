#ifndef _compat_cuh
#define _compat_cuh

// atomicAdd for half types, to support CC < 7.x
//
// HIP already declares `atomicAdd(half*, half)` and `atomicAdd(half2*, half2)`
// in hip_fp16.h. Re-declaring them here with a `void` return type is an
// "overload only by return type" error on current HIP (and makes the call
// sites in the GEMM kernels ambiguous), so the shims are CUDA-only.

#if !defined(USE_ROCM)

__device__ __forceinline__ void atomicAdd_half(half* address, half val)
{
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do
    {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    }
    while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val)
{
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do
    {
        assumed = old;
        half2 old_val = *((half2*)&old);
        half2 new_val = __hadd2(old_val, val);
        old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
    }
    while (assumed != old);
}

#endif  // !defined(USE_ROCM)

//

#if !defined(USE_ROCM) && defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 700

__device__ __forceinline__ void atomicAdd(half* address, half val) { atomicAdd_half(address, val); }

#if __CUDA_ARCH__ < 600
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) { atomicAdd_half2(address, val); }
#endif

#endif
#endif  // !defined(USE_ROCM) && defined(__CUDA_ARCH__)

#endif
