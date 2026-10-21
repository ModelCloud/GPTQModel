#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
namespace gptq_cpu
{

    // CPU feature detection
    class CPUFeatures
    {
    public:
        static bool hasAVX2()
        {
            static bool avx2_supported = checkAVX2();
            return avx2_supported;
        }

        static bool hasAVX512BF16()
        {
            static bool bf16_supported = checkAVX512BF16();
            return bf16_supported;
        }

    private:
        static bool checkAVX2()
        {
#ifdef _MSC_VER
            int cpu_info[4];
            __cpuid(cpu_info, 0);
            int n_ids = cpu_info[0];

            if (n_ids >= 7)
            {
                __cpuidex(cpu_info, 7, 0);
                return (cpu_info[1] & (1 << 5)) != 0; // EBX bit 5
            }
            return false;
#else
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_max(0, nullptr) < 7)
            {
                return false;
            }
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            return (ebx & (1 << 5)) != 0; // EBX bit 5
#endif
        }

        static bool checkAVX512()
        {
#ifdef _MSC_VER
            int cpu_info[4];
            __cpuid(cpu_info, 0);
            int n_ids = cpu_info[0];
            if (n_ids < 7)
                return false;

            __cpuidex(cpu_info, 7, 0);
            const bool avx512f = (cpu_info[1] & (1 << 16)) != 0;
            const bool avx512dq = (cpu_info[1] & (1 << 17)) != 0;
            const bool avx512bw = (cpu_info[1] & (1 << 30)) != 0;
            const bool avx512vl = (cpu_info[1] & (1 << 31)) != 0;
            const bool avx512vbmi = (cpu_info[2] & (1 << 1)) != 0;
            if (!(avx512f && avx512dq && avx512bw && avx512vl && avx512vbmi))
                return false;

            __cpuid(cpu_info, 1);
            bool osxsave = (cpu_info[2] & (1 << 27)) != 0; // ECX bit 27: OSXSAVE
            if (!osxsave)
                return false;

            // check XCR0: bits 1,2 (SSE/AVX) and 5,6,7 (AVX-512 state) must be enabled by OS
            unsigned long long xcr0 = _xgetbv(0);
            return ((xcr0 & 0xE6ULL) == 0xE6ULL);
#else
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_max(0, nullptr) < 7)
            {
                return false;
            }

            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            const bool avx512f = (ebx & (1u << 16)) != 0;
            const bool avx512dq = (ebx & (1u << 17)) != 0;
            const bool avx512bw = (ebx & (1u << 30)) != 0;
            const bool avx512vl = (ebx & (1u << 31)) != 0;
            // AVX512VBMI is required by _mm256_permutexvar_epi8.
            const bool avx512vbmi = (ecx & (1u << 1)) != 0;
            if (!(avx512f && avx512dq && avx512bw && avx512vl && avx512vbmi))
                return false;

            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0)
            {
                return false;
            }
            bool osxsave = (ecx & (1 << 27)) != 0; // ECX bit 27: OSXSAVE
            if (!osxsave)
                return false;

            unsigned int xcr0_lo = 0, xcr0_hi = 0;
            __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
            unsigned long long xcr0 = ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
            // require XCR0 bits 1,2,5,6,7 set -> mask 0xE6 (0b11100110)
            return ((xcr0 & 0xE6ULL) == 0xE6ULL);
#endif
        }

        static bool checkAVX512BF16()
        {
            // Require every AVX-512 feature used by the kernel and OS state.
            if (!checkAVX512())
                return false;

#ifdef _MSC_VER
            int cpu_info[4];
            __cpuidex(cpu_info, 7, 1);
            // AVX512_BF16: CPUID.(EAX=7,ECX=1):EAX[bit 5]
            return (cpu_info[0] & (1 << 5)) != 0;
#else
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_max(0, nullptr) < 7)
                return false;
            __cpuid_count(7, 1, eax, ebx, ecx, edx);
            return (eax & (1u << 5)) != 0;
#endif
        }
    };

} // namespace gptq_cpu
