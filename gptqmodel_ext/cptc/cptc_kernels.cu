// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kWarp = 32;
constexpr int kOutTile = 16;
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kInputTile = 16;
#else
constexpr int kInputTile = 8;
#endif

struct Geometry {
  int bits;
  int state_mask;
  int shift;
  int block_size;
  int block_bytes;
  int groups;
  int n;
  int half_table;
};

void check_tensor(const at::Tensor& value, const at::Tensor& anchor,
                  at::ScalarType dtype, const char* name) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.device() == anchor.device() &&
              value.scalar_type() == dtype, "CPTC: invalid ", name, " device, dtype or strides");
}

Geometry geometry(const at::Tensor& packed, const at::Tensor& table, const at::Tensor& coeffs,
                  const at::Tensor& scales, int64_t bits, int64_t depth, int64_t shift, int64_t k) {
  check_tensor(packed, packed, at::kByte, "packed bytes");
  check_tensor(table, packed, at::kByte, "E2M5 table");
  check_tensor(coeffs, packed, at::kInt, "coefficients");
  check_tensor(scales, packed, at::kFloat, "scales");
  TORCH_CHECK(bits >= 1 && bits <= 6 && depth >= bits && depth <= 16,
              "CPTC: invalid bit width or depth");
  TORCH_CHECK(packed.dim() == 3 && packed.size(0) > 0 && packed.size(1) > 0 &&
              packed.size(0) <= std::numeric_limits<int>::max() / 2 &&
              scales.dim() == 2 && scales.size(0) == packed.size(0) && scales.size(1) == packed.size(1),
              "CPTC: invalid packed bytes or scales shape");
  const int64_t block_bytes = packed.size(2);
  TORCH_CHECK(block_bytes == 128 * bits / 8 || block_bytes == 256 * bits / 8,
              "CPTC: encoded block size must be 128 or 256 values");
  const int64_t block_size = block_bytes * 8 / bits;
  TORCH_CHECK(k > 0 && k <= std::numeric_limits<int>::max() / 2 &&
              (k + block_size - 1) / block_size == packed.size(1), "CPTC: invalid input size");
  const auto half_table = table.numel();
  TORCH_CHECK(table.dim() == 1 && half_table >= 2 && half_table <= 64 &&
              !(half_table & (half_table - 1)) && (half_table >= (1 << bits)) &&
              shift >= 10 && shift <= 15 && (1 << (16 - shift)) <= half_table && coeffs.numel() == 2,
              "CPTC: invalid coefficient or table geometry");
  return {int(bits), (1 << depth) - 1, int(shift), int(block_size), int(block_bytes),
          int(packed.size(1)), int(packed.size(0)), int(half_table)};
}

__device__ __forceinline__ float e2m5(uint8_t code) {
  const int exponent = (code >> 5) & 3;
  const int mantissa = code & 31;
  const float value = ((code & 127) == 127) ? __int_as_float(0x7fc00000) :
      (exponent ? (32 + mantissa) << (exponent - 1) : mantissa) * (1.0f / 32);
  return code & 128 ? -value : value;
}

__device__ __forceinline__ int cptc_index(const uint8_t* block, int position, const Geometry& g,
                                          uint32_t a0, uint32_t a1) {
  const int bit = position * g.bits;
  const int byte = bit >> 3;
  // At most two bytes wrap. Each window must stay in this block, including
  // the final window of a row or of the entire allocation.
  const int second = byte + 1 < g.block_bytes ? byte + 1 : 0;
  const int third = byte + 2 < g.block_bytes ? byte + 2 : byte + 2 - g.block_bytes;
  const uint32_t window = uint32_t(block[byte]) | (uint32_t(block[second]) << 8) |
                          (uint32_t(block[third]) << 16);
  const uint32_t state = (window >> (bit & 7)) & g.state_mask;
  return (((state * a0) & 0xffff) >> g.shift) + (((state * a1) & 0xffff) >> g.shift);
}

__global__ void decode_kernel(float* output, const uint8_t* packed, const uint8_t* table,
                              const int32_t* coeffs, const float* scales, Geometry g, int k) {
  __shared__ float lookup[127];
  if (threadIdx.x < 2 * g.half_table - 1) {
    int i = threadIdx.x;
    lookup[i] = i < g.half_table ? e2m5(table[i]) : -e2m5(table[2 * (g.half_table - 1) - i]);
  }
  __syncthreads();
  const uint32_t a0 = coeffs[0], a1 = coeffs[1];
  const int64_t count = int64_t(g.n) * k;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    const int col = i % k;
    const int64_t group = (i / k) * g.groups + col / g.block_size;
    const int index = cptc_index(packed + group * g.block_bytes, col % g.block_size, g, a0, a1);
    output[i] = lookup[index] * scales[group];
  }
}

template <bool Bf16>
__device__ __forceinline__ uint16_t operand(float value) {
  if constexpr (Bf16) {
    // All finite E2M5 values are exactly representable in BF16.
    return __float_as_uint(value) >> 16;
  } else {
    return __half_as_ushort(__float2half_rn(value));
  }
}

template <bool Bf16>
__device__ __forceinline__ float add_group(float dot, float scale, float sum,
                                          const uint16_t* x, const uint8_t* packed,
                                          const uint16_t* lookup, Geometry g,
                                          int group, int n, int m, int rows, int k,
                                          uint32_t a0, uint32_t a1) {
  if constexpr (Bf16) {
    // The factored dot can exceed FP32 range before multiplication by a small
    // scale, even if every effective weight and the final output are finite.
    if (m < rows && !isfinite(dot)) {
      const auto* block = packed + (int64_t(n) * g.groups + group) * g.block_bytes;
      const int begin = group * g.block_size;
      const int count = min(g.block_size, k - begin);
      const auto* input = x + int64_t(m) * k + begin;
      double wide = sum;
      for (int i = 0; i < count; ++i) {
        const auto code = lookup[cptc_index(block, i, g, a0, a1)];
        const float weight = __uint_as_float(uint32_t(code) << 16) * scale;
        const float value = __uint_as_float(uint32_t(input[i]) << 16);
        wide += double(weight) * value;
      }
      return float(wide);
    }
  }
  return fmaf(dot, scale, sum);
}

#if defined(__HIP_PLATFORM_AMD__)

template <bool Bf16>
__device__ void mma_tile(float* partial, const uint16_t* x, const uint8_t* packed,
                         const uint16_t* lookup, const float* scales, Geometry g,
                         int lane, int n, int m, int rows, int k, int begin, int end,
                         uint32_t a0, uint32_t a1) {
#if defined(__gfx1200__) || defined(__gfx1201__)
  using Float8 = float __attribute__((ext_vector_type(8)));
  using Half8 = _Float16 __attribute__((ext_vector_type(8)));
  using Short8 = short __attribute__((ext_vector_type(8)));
  const int column = lane & 15;
  const int half = (lane >> 4) * 8;
  const int load_n = n + column;
  const int load_m = m + column;
  Float8 sum{};
  for (int group = begin; group < end; ++group) {
    Float8 dot{};
    for (int step = 0; step < g.block_size; step += 16) {
      Short8 a{}, b{};
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int position = step + half + j;
        const int input_col = group * g.block_size + position;
        if (load_n < g.n) {
          const auto* block = packed + (int64_t(load_n) * g.groups + group) * g.block_bytes;
          a[j] = lookup[cptc_index(block, position, g, a0, a1)];
        }
        if (load_m < rows && input_col < k) b[j] = x[int64_t(load_m) * k + input_col];
      }
      if constexpr (Bf16) {
        dot = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, dot);
      } else {
        dot = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            __builtin_bit_cast(Half8, a), __builtin_bit_cast(Half8, b), dot);
      }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int out = n + half + j;
      if (out < g.n) {
        sum[j] = add_group<Bf16>(dot[j], scales[int64_t(out) * g.groups + group], sum[j],
                                 x, packed, lookup, g, group, out, load_m, rows, k, a0, a1);
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int out = n + half + j;
    if (load_m < rows && out < g.n) partial[int64_t(load_m) * g.n + out] = sum[j];
  }
#else
  __builtin_trap();  // Python selects this implementation only on RDNA4.
#endif
}

#else

__device__ __forceinline__ uint32_t pair(uint16_t low, uint16_t high) {
  return uint32_t(low) | (uint32_t(high) << 16);
}

__device__ __forceinline__ uint32_t weight_pair(const uint8_t* block, int offset,
                                                const uint16_t* lookup, const Geometry& g,
                                                uint32_t a0, uint32_t a1) {
  if (!block) return 0;
  return pair(lookup[cptc_index(block, offset, g, a0, a1)],
              lookup[cptc_index(block, offset + 1, g, a0, a1)]);
}

__device__ __forceinline__ uint32_t input_pair(const uint16_t* x, int offset, int k) {
  if (!x || offset >= k) return 0;
  return pair(x[offset], offset + 1 < k ? x[offset + 1] : 0);
}

template <bool Bf16>
__device__ void mma_tile(float* partial, const uint16_t* x, const uint8_t* packed,
                         const uint16_t* lookup, const float* scales, Geometry g,
                         int lane, int n, int m, int rows, int k, int begin, int end,
                         uint32_t a0, uint32_t a1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int n0 = n + (lane >> 2), n1 = n0 + 8;
  const int load_m = m + (lane >> 2), store_m = m + (lane & 3) * 2;
  const int offset = (lane & 3) * 2;
  const auto* x_row = load_m < rows ? x + int64_t(load_m) * k : nullptr;
  float sum[4]{};
  for (int group = begin; group < end; ++group) {
    const int64_t g0 = int64_t(n0) * g.groups + group, g1 = int64_t(n1) * g.groups + group;
    const auto* block0 = n0 < g.n ? packed + g0 * g.block_bytes : nullptr;
    const auto* block1 = n1 < g.n ? packed + g1 * g.block_bytes : nullptr;
    float dot[4]{};
    for (int step = 0; step < g.block_size; step += 16) {
      const int pos = step + offset;
      const int input_col = group * g.block_size + pos;
      const uint32_t a0r = weight_pair(block0, pos, lookup, g, a0, a1);
      const uint32_t a1r = weight_pair(block1, pos, lookup, g, a0, a1);
      const uint32_t a2r = weight_pair(block0, pos + 8, lookup, g, a0, a1);
      const uint32_t a3r = weight_pair(block1, pos + 8, lookup, g, a0, a1);
      const uint32_t b0 = input_pair(x_row, input_col, k), b1 = input_pair(x_row, input_col + 8, k);
      #define CPTC_MMA(TYPE)                                                                        \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32." TYPE "." TYPE ".f32 "                \
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                        \
                     : "+f"(dot[0]), "+f"(dot[1]), "+f"(dot[2]), "+f"(dot[3])                         \
                     : "r"(a0r), "r"(a1r), "r"(a2r), "r"(a3r), "r"(b0), "r"(b1))
      if constexpr (Bf16) { CPTC_MMA("bf16"); }
      else { CPTC_MMA("f16"); }
      #undef CPTC_MMA
    }
    if (n0 < g.n) {
      sum[0] = add_group<Bf16>(dot[0], scales[g0], sum[0], x, packed, lookup, g,
                               group, n0, store_m, rows, k, a0, a1);
      sum[1] = add_group<Bf16>(dot[1], scales[g0], sum[1], x, packed, lookup, g,
                               group, n0, store_m + 1, rows, k, a0, a1);
    }
    if (n1 < g.n) {
      sum[2] = add_group<Bf16>(dot[2], scales[g1], sum[2], x, packed, lookup, g,
                               group, n1, store_m, rows, k, a0, a1);
      sum[3] = add_group<Bf16>(dot[3], scales[g1], sum[3], x, packed, lookup, g,
                               group, n1, store_m + 1, rows, k, a0, a1);
    }
  }
  if (store_m < rows) {
    if (n0 < g.n) partial[int64_t(store_m) * g.n + n0] = sum[0];
    if (n1 < g.n) partial[int64_t(store_m) * g.n + n1] = sum[2];
  }
  if (store_m + 1 < rows) {
    if (n0 < g.n) partial[int64_t(store_m + 1) * g.n + n0] = sum[1];
    if (n1 < g.n) partial[int64_t(store_m + 1) * g.n + n1] = sum[3];
  }
#elif defined(__CUDA_ARCH__)
  asm("trap;");  // Ampere or newer is required for the fragment instructions.
#endif
}

#endif

template <bool Bf16>
__global__ void mma_kernel(float* partial, const uint16_t* input, const uint8_t* packed,
                           const uint8_t* table, const int32_t* coeffs, const float* scales,
                           Geometry g, int rows, int k, int groups_per_split) {
  __shared__ uint16_t lookup[127];
  if (threadIdx.x < 2 * g.half_table - 1) {
    int i = threadIdx.x;
    lookup[i] = i < g.half_table ? operand<Bf16>(e2m5(table[i])) :
                operand<Bf16>(-e2m5(table[2 * (g.half_table - 1) - i]));
  }
  __syncthreads();
  const int n = (blockIdx.x * (kThreads / kWarp) + threadIdx.x / kWarp) * kOutTile;
  if (n >= g.n) return;
  const int begin = blockIdx.z * groups_per_split;
  const int end = min(begin + groups_per_split, g.groups);
  mma_tile<Bf16>(partial + int64_t(blockIdx.z) * rows * g.n, input, packed, lookup, scales, g,
                 threadIdx.x % kWarp, n, blockIdx.y * kInputTile, rows, k, begin, end,
                 uint32_t(coeffs[0]), uint32_t(coeffs[1]));
}

__device__ __forceinline__ float shuffle_down(float value, int delta, int width) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_down(value, delta, width);
#else
  return __shfl_down_sync(0xffffffffu, value, delta, width);
#endif
}

__device__ __forceinline__ int shuffle_down(int value, int delta, int width) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_down(value, delta, width);
#else
  return __shfl_down_sync(0xffffffffu, value, delta, width);
#endif
}

__device__ __forceinline__ void select_min(float other, int other_index, float& best, int& index) {
  if (other < best || (other == best && other_index < index)) {
    best = other;
    index = other_index;
  }
}

constexpr int kCachedEncodeThreads = 512;

template <int Bits, bool Cached>
__global__ __launch_bounds__(Cached ? kCachedEncodeThreads : kThreads)
void encode_kernel(const float* input, const void* decoded, int32_t* path,
                   uint8_t* back, float* global_cost, int prefixes, int length) {
  // A subgroup evaluates one prefix's incoming branches, then reduces to
  // the minimum cost and its lowest-index predecessor. K=6 uses two values
  // per lane. No warp communication crosses a subgroup boundary.
  constexpr int choices = 1 << Bits;
  constexpr int lanes = choices < kWarp ? choices : kWarp;
  constexpr int teams = kThreads / lanes;
  constexpr int threads = Cached ? kCachedEncodeThreads : kThreads;
  constexpr int cached_prefixes = 1 << (16 - Bits);
  constexpr int prefixes_per_thread = cached_prefixes / kCachedEncodeThreads;
  constexpr int cols = cached_prefixes / choices;
  const int lane = threadIdx.x % lanes;
  const int team = threadIdx.x / lanes;
  const int mask = prefixes - 1;
  input += int64_t(blockIdx.x) * length;
  path += int64_t(blockIdx.x) * length;
  back += int64_t(blockIdx.x) * length * prefixes;
  extern __shared__ float local_cost[];
  const int cost_stride = Cached ? cached_prefixes + choices : prefixes;
  float* costs = global_cost ? global_cost + int64_t(blockIdx.x) * 2 * cost_stride : local_cost;
  __shared__ int seam;
  __shared__ float end_cost[threads];
  __shared__ int end_index[threads];
  auto cost_index = [](int p) {
    // For each high prefix p, evaluating candidate u reads the cost of
    // ((p << Bits) | u) & (prefixes-1). Transpose the cost array to make
    // those reads unit-stride, and pad each row to avoid bank-aligned stores.
    if constexpr (Cached) return (p & (choices - 1)) * (cols + 1) + (p >> Bits);
    else return p;
  };
  // E2M5 values fit exactly in FP16. Cache pairs of 16-bit values once per
  // block so the 16-bit depth does not reload its full codebook each step.
  uint32_t values[Cached ? prefixes_per_thread : 1][Cached ? choices / 2 : 1];
  if constexpr (Cached) {
    const auto* codes = static_cast<const uint32_t*>(decoded);
    #pragma unroll
    for (int j = 0; j < prefixes_per_thread; ++j) {
      const int p = threadIdx.x + j * threads;
      #pragma unroll
      for (int u = 0; u < choices / 2; ++u) values[j][u] = codes[u * cached_prefixes + p];
    }
  }
  for (int pass = 0; pass < 2; ++pass) {
    const int first = pass == 0 ? length / 2 : 0;
    const int pinned = pass ? seam : -1;
    for (int t = 0; t < length; ++t) {
      const float target = input[t ^ first];
      const float* prev = costs + ((t + 1) & 1) * cost_stride;
      float* curr = costs + (t & 1) * cost_stride;
      if constexpr (Cached) {
        #pragma unroll
        for (int j = 0; j < prefixes_per_thread; ++j) {
          const int p = threadIdx.x + j * threads;
          const int previous_col = p & (cols - 1);
          float best = INFINITY;
          int selected = 0;
          #pragma unroll
          for (int u = 0; u < choices; ++u) {
            const uint32_t code = values[j][u / 2] >> (16 * (u & 1));
            const float diff = target - __half2float(__ushort_as_half(uint16_t(code)));
            const float error = diff * diff;
            float score = error;
            if (t != 0) score += prev[u * (cols + 1) + previous_col];
            else if (pass && pinned != ((p << Bits) | u)) score = INFINITY;
            if (score < best) {
              best = score;
              selected = u;
            }
          }
          curr[cost_index(p)] = best;
          if (t >= first) back[int64_t(t) * prefixes + p] = selected;
        }
      } else {
        // Round the team count up so the CUDA shuffles have a full warp mask.
        const int stride_end = ((prefixes + teams - 1) / teams) * teams;
        for (int p = team; p < stride_end; p += teams) {
          float best = INFINITY;
          int selected = lane;
          if (p < prefixes) {
            #pragma unroll
            for (int u = lane; u < choices; u += lanes) {
              const int state = (p << Bits) | u;
              if (t == 0 && pass && state != pinned) continue;
              const float diff = target - static_cast<const float*>(decoded)[state];
              const float score = diff * diff + (t ? prev[state & mask] : 0.0f);
              select_min(score, u, best, selected);
            }
          }
          #pragma unroll
          for (int delta = lanes / 2; delta > 0; delta /= 2) {
            const float other = shuffle_down(best, delta, lanes);
            const int other_index = shuffle_down(selected, delta, lanes);
            select_min(other, other_index, best, selected);
          }
          if (lane == 0 && p < prefixes) {
            curr[p] = best;
            if (t >= first) back[int64_t(t) * prefixes + p] = selected;
          }
        }
      }
      __syncthreads();
    }
    if (pass == 0) {
      // Find the best unrestricted terminal state and trace halfway back to
      // find the original block's seam in the rotated pass.
      const float* final = costs + ((length - 1) & 1) * cost_stride;
      float best = INFINITY;
      int index = prefixes;
      for (int p = threadIdx.x; p < prefixes; p += threads) select_min(final[cost_index(p)], p, best, index);
      end_cost[threadIdx.x] = best;
      end_index[threadIdx.x] = index;
      __syncthreads();
      for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
          select_min(end_cost[threadIdx.x + stride], end_index[threadIdx.x + stride],
                     end_cost[threadIdx.x], end_index[threadIdx.x]);
        }
        __syncthreads();
      }
    }
    if (threadIdx.x == 0) {
      int prefix = pass == 0 ? end_index[0] : pinned & mask;
      int state = 0;
      for (int t = length - 1; t >= first; --t) {
        state = (prefix << Bits) | back[int64_t(t) * prefixes + prefix];
        prefix = state & mask;
        if (pass) path[t] = state;
      }
      if (pass == 0) seam = state;
    }
    __syncthreads();
  }
}

}  // namespace

at::Tensor cptc_decode(const at::Tensor& packed, const at::Tensor& table, const at::Tensor& coeffs,
                       const at::Tensor& scales, int64_t bits, int64_t depth, int64_t shift, int64_t k) {
  const auto g = geometry(packed, table, coeffs, scales, bits, depth, shift, k);
  const c10::cuda::CUDAGuard guard(packed.device());
  auto output = at::empty({g.n, k}, scales.options());
  const auto ctas = std::min<int64_t>((output.numel() + kThreads - 1) / kThreads, 65535);
  decode_kernel<<<ctas, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<float>(), packed.data_ptr<uint8_t>(), table.data_ptr<uint8_t>(),
      coeffs.data_ptr<int32_t>(), scales.data_ptr<float>(), g, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor cptc_mma(const at::Tensor& input, const at::Tensor& packed, const at::Tensor& table,
                    const at::Tensor& coeffs, const at::Tensor& scales,
                    int64_t bits, int64_t depth, int64_t shift) {
  TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(0) <= 16 &&
              (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16),
              "CPTC MMA: expected 1 to 16 rows of FP16 or BF16 input");
  check_tensor(input, packed, input.scalar_type(), "MMA input");
  const auto g = geometry(packed, table, coeffs, scales, bits, depth, shift, input.size(1));
  const c10::cuda::CUDAGuard guard(packed.device());
  const int rows = input.size(0), k = input.size(1);
  const int groups_per_split = (g.groups + 15) / 16;
  const int splits = (g.groups + groups_per_split - 1) / groups_per_split;
  auto output = at::empty({splits, rows, g.n}, scales.options());
  const int n_tile = kOutTile * (kThreads / kWarp);
  const dim3 grid((g.n + n_tile - 1) / n_tile, (rows + kInputTile - 1) / kInputTile, splits);
  const auto kernel = input.scalar_type() == at::kBFloat16 ? mma_kernel<true> : mma_kernel<false>;
  kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<float>(), reinterpret_cast<const uint16_t*>(input.const_data_ptr()),
      packed.data_ptr<uint8_t>(), table.data_ptr<uint8_t>(), coeffs.data_ptr<int32_t>(),
      scales.data_ptr<float>(), g, rows, k, groups_per_split);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return splits == 1 ? output.squeeze(0) : output.sum(0);
}

at::Tensor cptc_encode(const at::Tensor& blocks, const at::Tensor& decoded, int64_t bits) {
  check_tensor(blocks, blocks, at::kFloat, "Viterbi input");
  const bool cached = decoded.scalar_type() == at::kHalf && decoded.numel() == 65536;
  check_tensor(decoded, blocks, cached ? at::kHalf : at::kFloat, "decoded states");
  TORCH_CHECK(bits >= 1 && bits <= 6 && blocks.dim() == 2 && blocks.size(0) > 0 &&
              blocks.size(0) <= 4096 && (blocks.size(1) == 128 || blocks.size(1) == 256) &&
              decoded.dim() == 1 && decoded.numel() >= (1 << bits) && decoded.numel() <= 65536 &&
              !(decoded.numel() & (decoded.numel() - 1)), "CPTC: invalid Viterbi input shape");
  const c10::cuda::CUDAGuard guard(blocks.device());
  const int prefixes = decoded.numel() >> bits;
  const int length = blocks.size(1);
  auto back = at::empty({blocks.size(0), length, prefixes}, blocks.options().dtype(at::kByte));
  auto path = at::empty(blocks.sizes(), blocks.options().dtype(at::kInt));
  const int cost_stride = cached ? prefixes + (1 << bits) : prefixes;
  int shared_bytes = 2 * cost_stride * sizeof(float);
  at::Tensor global;
  if (prefixes > 4096) {
    global = at::empty({blocks.size(0), 2 * cost_stride}, blocks.options());
    shared_bytes = 0;
  }
  using Kernel = decltype(&encode_kernel<1, false>);
  const Kernel kernels[] = {encode_kernel<1, false>, encode_kernel<2, false>, encode_kernel<3, false>,
                            encode_kernel<4, false>, encode_kernel<5, false>, encode_kernel<6, false>};
  const Kernel cached_kernels[] = {encode_kernel<1, true>, encode_kernel<2, true>, encode_kernel<3, true>,
                                   encode_kernel<4, true>, encode_kernel<5, true>, encode_kernel<6, true>};
  const Kernel kernel = (cached ? cached_kernels : kernels)[bits - 1];
  const auto codebook = cached ? decoded.view({prefixes, -1, 2}).transpose(0, 1).contiguous() : decoded;
  kernel<<<blocks.size(0), cached ? kCachedEncodeThreads : kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      blocks.data_ptr<float>(), codebook.const_data_ptr(), path.data_ptr<int32_t>(),
      back.data_ptr<uint8_t>(), global.defined() ? global.data_ptr<float>() : nullptr,
      prefixes, length);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return path;
}
