// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#include <cuda_fp16.h>
#include "frac.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <limits>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include "quantize_tiles_frac_kernel.cuh"

// Fractional-rate trellis tiles. The 256-weight tile is a ring of 16 * K bits: weight i's D(i) new bits,
// D(i) = KA + bit (i mod 16) of MASK, sit at ring position S(i) - D(i) .. S(i) where S(i) is the prefix sum
// of D; its 16-bit window is ring bits [S(i) - 16, S(i)). Bits are stored MSB-first in 32-bit words, the
// same stream convention as the integer pack_trellis output, so a tile is 16 * K uint16 = 8 * K uint32
// words (K half-integer -> whole words). One warp per tile; lane t owns ring positions 8t .. 8t+7 (the
// tensor-core element order the quantizer works in).

// Largest tile in 32-bit words: KA = 7 with every extra bit set, 16 * 8 bits per 16 weights
#define FRAC_MAX_WORDS 64

__device__ __forceinline__ int frac_d(int i, int ka, uint32_t mask) { return ka + ((mask >> (i & 15)) & 1); }

// Ring position where window i ends, S(i)
__device__ __forceinline__ int frac_s(int i, int ka, uint32_t mask, int bpb)
{
    int s = (i >> 4) * bpb;
    for (int j = 0; j <= (i & 15); ++j) s += frac_d(j, ka, mask);
    return s;
}

// 16-bit window starting at ring bit `start` (mod R), from MSB-first 32-bit words
__device__ __forceinline__ uint32_t frac_window(const uint32_t* words, int nw, int start)
{
    const int R = nw * 32;
    start %= R; if (start < 0) start += R;
    const int wi = start >> 5, o = start & 31;
    const uint64_t v = ((uint64_t) words[wi] << 32) | (uint64_t) words[(wi + 1) % nw];
    return (uint32_t) ((v >> (48 - o)) & 0xFFFFu);
}

__global__ __launch_bounds__(128)
void pack_trellis_frac_kernel(uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked,
                              int num_tiles, int ka, uint32_t mask, int bpb)
{
    const int tile = blockIdx.x * 128 + threadIdx.x;
    if (tile >= num_tiles) return;
    const int nw = bpb / 2;
    uint32_t words[FRAC_MAX_WORDS];
    for (int w = 0; w < nw; ++w) words[w] = 0;
    const uint16_t* idx = g_unpacked + (size_t) tile * 256;
    int pos = 0;
    for (int i = 0; i < 256; ++i)
    {
        const int d = frac_d(i, ka, mask);
        const uint32_t v = (uint32_t) idx[i] & ((1u << d) - 1);
        for (int b = d - 1; b >= 0; --b, ++pos)
            if ((v >> b) & 1) words[pos >> 5] |= 1u << (31 - (pos & 31));
    }
    uint32_t* out = (uint32_t*) (g_packed + (size_t) tile * bpb);
    for (int w = 0; w < nw; ++w) out[w] = words[w];
}

__global__ __launch_bounds__(32)
void unpack_trellis_frac_kernel(uint16_t* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
                                int ka, uint32_t mask, int bpb)
{
    __shared__ uint32_t words[FRAC_MAX_WORDS];
    const int tile = blockIdx.x;
    const int t = threadIdx.x;
    const int nw = bpb / 2;
    for (int w = t; w < nw; w += 32) words[w] = ((const uint32_t*) (g_packed + (size_t) tile * bpb))[w];
    __syncwarp();
    for (int j = 0; j < 8; ++j)
    {
        const int i = t * 8 + j;
        g_unpacked[(size_t) tile * 256 + i] = (uint16_t) frac_window(words, nw, frac_s(i, ka, mask, bpb) - 16);
    }
}

static int frac_bpb(int KA, int64_t MASK)
{
    TORCH_CHECK(KA >= 1 && KA <= 3 && MASK == 0xAAAA, "EXL3 half-bit rates require KA in 1..3 and MASK 0xAAAA");
    return 16 * KA + 8;
}

void pack_trellis_frac(at::Tensor packed, at::Tensor unpacked, int KA, int64_t MASK)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int bpb = frac_bpb(KA, MASK);
    TORCH_CHECK_DIM(packed, 3); TORCH_CHECK_DIM(unpacked, 3);
    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1); TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256); TORCH_CHECK_SIZE(packed, 2, bpb);
    TORCH_CHECK(packed.device() == unpacked.device() && packed.is_contiguous() && unpacked.is_contiguous(),
                "frac: packed and unpacked buffers must be contiguous and on the same device");
    TORCH_CHECK_DTYPE(packed, kShort); TORCH_CHECK_DTYPE(unpacked, kShort);
    const int64_t num_tiles = packed.size(0) * packed.size(1);
    if (!num_tiles) return;
    TORCH_CHECK(num_tiles <= std::numeric_limits<int>::max(), "frac: too many trellis tiles");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % alignof(uint32_t) == 0,
                "frac: packed buffer must be 4-byte aligned");
    pack_trellis_frac_kernel<<<CEIL_DIVIDE(num_tiles, 128), 128, 0, stream>>>
        ((uint16_t*) packed.data_ptr(), (const uint16_t*) unpacked.data_ptr(), num_tiles, KA, (uint32_t) MASK, bpb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unpack_trellis_frac(at::Tensor unpacked, at::Tensor packed, int KA, int64_t MASK)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int bpb = frac_bpb(KA, MASK);
    TORCH_CHECK_DIM(packed, 3); TORCH_CHECK_DIM(unpacked, 3);
    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1); TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256); TORCH_CHECK_SIZE(packed, 2, bpb);
    TORCH_CHECK(packed.device() == unpacked.device() && packed.is_contiguous() && unpacked.is_contiguous(),
                "frac: packed and unpacked buffers must be contiguous and on the same device");
    TORCH_CHECK_DTYPE(packed, kShort); TORCH_CHECK_DTYPE(unpacked, kShort);
    const int64_t num_tiles = packed.size(0) * packed.size(1);
    if (!num_tiles) return;
    TORCH_CHECK(num_tiles <= std::numeric_limits<int>::max(), "frac: too many trellis tiles");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % alignof(uint32_t) == 0,
                "frac: packed buffer must be 4-byte aligned");
    unpack_trellis_frac_kernel<<<num_tiles, 32, 0, stream>>>
        ((uint16_t*) unpacked.data_ptr(), (const uint16_t*) packed.data_ptr(), KA, (uint32_t) MASK, bpb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_tiles_frac(
    at::Tensor input_tiles, at::Tensor output_tiles, at::Tensor output_indices,
    at::Tensor temp_costs, at::Tensor temp_edges, int KA, int64_t MASK)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
    TORCH_CHECK(false, "EXL3 half-bit Viterbi quantization requires NVIDIA CUDA");
#else
    frac_bpb(KA, MASK);
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DIM(input_tiles, 2);
    constexpr int L = 256;
    TORCH_CHECK_SIZE(input_tiles, 1, L);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_indices);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_tiles);
    TORCH_CHECK_DTYPE(input_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_indices, kShort);
    TORCH_CHECK_DTYPE(temp_costs, kHalf);
    TORCH_CHECK_DTYPE(temp_edges, kShort);
    const int width = 65536 >> KA;
    TORCH_CHECK_DIM(temp_costs, 3);
    TORCH_CHECK_SIZE(temp_costs, 1, 2);
    TORCH_CHECK_SIZE(temp_costs, 2, width);
    TORCH_CHECK_DIM(temp_edges, 3);
    TORCH_CHECK_SIZE(temp_edges, 1, L);
    TORCH_CHECK_SIZE(temp_edges, 2, width);
    for (const auto& tensor : {input_tiles, output_tiles, output_indices, temp_costs, temp_edges})
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == input_tiles.device(),
                    "quantize_tiles_frac: buffers must be contiguous and on the input device");

    const int64_t count = input_tiles.size(0);
    if (count == 0) return;
    const int64_t capacity = MIN(temp_costs.size(0), temp_edges.size(0));
    TORCH_CHECK(capacity > 0 && count <= std::numeric_limits<int>::max(),
                "quantize_tiles_frac: empty scratch or too many input tiles");
    constexpr auto kernels = std::array {
        quantize_tiles_frac_kernel<1, 0xaaaau>, quantize_tiles_frac_kernel<2, 0xaaaau>,
        quantize_tiles_frac_kernel<3, 0xaaaau>
    };
    auto kernel = kernels[KA - 1];
    constexpr int shmem = L * sizeof(half) + 32 * sizeof(int) + 128;
    for (int64_t start = 0; start < count; start += capacity)
    {
        const int batch = MIN(capacity, count - start);
        kernel<<<batch, QTF_NT, shmem, stream>>>
        (
            input_tiles.const_data_ptr<float>() + L * start,
            output_tiles.data_ptr<float>() + L * start,
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()) + L * start,
            reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
            reinterpret_cast<uint16_t*>(temp_edges.data_ptr<int16_t>()),
            nullptr
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#endif
}
