// Based on the Hugging Face kernels-community `quantization-gptq` CPU kernel:
// https://github.com/huggingface/kernels-community/tree/c467920211e144ea621d9e8193ff71ec160ce256/quantization-gptq
// Original author: jiqing-feng <jiqing.feng@intel.com> (Intel).
// Modified for GPT-QModel by ModelCloud.ai.

// AVX512 implementation - compile with -mavx512f -mavx512bf16
#include <algorithm>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <omp.h>
#include <immintrin.h>
#include <cstdint>

namespace int4_qzero_group_scaled_cpu
{
    namespace avx512
    {
// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

        // block size for AMX gemm
        constexpr int block_size_m() { return 2 * TILE_M; }

        constexpr int block_size_n() { return 2 * TILE_N; }

        template <typename T>
        inline int get_cache_blocks(int chunk_size)
        {
            // L2 2MB and ratio of 50%
            const int L2_size = 2048 * 1024 >> 1;
            return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
        }

// forced unroll for perf critical path
#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

        template <int n>
        struct Unroll
        {
            template <typename Func, typename... Args>
            ALWAYS_INLINE void operator()(const Func &f, Args... args) const
            {
                Unroll<n - 1>{}(f, args...);
                f(std::integral_constant<int, n - 1>{}, args...);
            }
        };

        template <>
        struct Unroll<1>
        {
            template <typename Func, typename... Args>
            ALWAYS_INLINE void operator()(const Func &f, Args... args) const
            {
                f(std::integral_constant<int, 0>{}, args...);
            }
        };

        template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
        inline T div_up(T x, T y)
        {
            return (x + y - 1) / y;
        }

        inline int adjust_num_threads(int m)
        {
            // PyTorch's intra-op setting is authoritative. OpenMP's runtime
            // maximum can differ from torch.set_num_threads(), especially when
            // OMP_NUM_THREADS is unset, so sizing work from omp_get_max_threads()
            // oversubscribes the caller's configured thread budget.
            int actual_nth = at::get_num_threads();
            if (m == 1)
                return actual_nth;
            return std::max(1, (actual_nth >> 1) * 2);
        }

        template <typename func_t>
        inline void parallel_2d(int m, int n, const func_t &f)
        {
            int nth = adjust_num_threads(m);
            float r = float(m) / n;
            int nth_m = std::ceil(std::sqrt(r * nth));
            int nth_n = 1;
            for (; nth_m > 0; --nth_m)
            {
                nth_n = nth / nth_m;
                if (nth_m * nth_n == nth)
                {
                    break;
                }
            }
#pragma omp parallel num_threads(nth)
            {
                int ith = omp_get_thread_num();
                int ith_m = ith / nth_n;
                int ith_n = ith % nth_n;

                int thread_block_m = div_up(m, nth_m);
                int thread_block_n = div_up(n, nth_n);

                int begin_m = ith_m * thread_block_m;
                int end_m = std::min(m, begin_m + thread_block_m);
                int begin_n = ith_n * thread_block_n;
                int end_n = std::min(n, begin_n + thread_block_n);

                f(begin_m, end_m, begin_n, end_n);
            }
        }

        template <typename T>
        void gemm_int4_inference(
            int64_t M, int64_t N, int64_t K, const T *__restrict__ x, const unsigned char *__restrict__ w, const uint8_t *__restrict__ zeros,
            const T *__restrict__ absmax, T *__restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride);
    } // namespace avx512
} // namespace int4_qzero_group_scaled_cpu
