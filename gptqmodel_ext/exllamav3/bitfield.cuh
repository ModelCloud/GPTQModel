#pragma once

// Bitfield helpers shared by the quant decoders (quant/exl3_dq.cuh) and the
// CUDA tensor-core paths (ptx.cuh).
//
// These are deliberately free of CUDA-only inline PTX so that the HIP/ROCm
// build of the quantization kernels does not have to pull in ptx.cuh (whose
// ldmatrix/mma.sync/cp.async bodies are NVIDIA specific). On NVIDIA the helpers
// still lower to the exact same single instructions as before.

// Extract `length` bits at `offset` from the 64-bit value formed by (hi:lo).
static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    const uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
    uint64_t result64;
    asm ("bfe.u64 %0, %1, %2, %3;"
         : "=l"(result64)
         : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
#else
    return static_cast<uint32_t>((value >> offset) & ((static_cast<uint64_t>(1) << length) - 1));
#endif
}

// Funnel shift right of the 64-bit value (hi:lo) by `imm` bits, truncated to
// 32 bits. On NVIDIA this is a single shf.r.wrap.b32; HIP's __funnelshift_r has
// exactly the same semantics, so both platforms emit the same result.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#else
#define FSHF_IMM(dst, lo, hi, imm) dst = __funnelshift_r((lo), (hi), (imm))
#endif

// Extract the 16 bits at `imm` from `src`.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(USE_ROCM)
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#else
#define BFE16_IMM(dst, src, imm) dst = (((src) >> (imm)) & 0xffffu)
#endif
