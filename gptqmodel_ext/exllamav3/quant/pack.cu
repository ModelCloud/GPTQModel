#include <cuda_fp16.h>
#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include "r4nx.cuh"

template <int K>
__global__ __launch_bounds__(128)
void pack_trellis_kernel
(
    uint16_t* __restrict__ g_packed,
    const uint16_t* __restrict__ g_unpacked
)
{
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_unpacked[256];
    __shared__ uint16_t s_packed[packed_size];

    int t = threadIdx.x;
    g_packed += (gridDim.x * blockIdx.y + blockIdx.x) * packed_size;
    g_unpacked += (gridDim.x * blockIdx.y + blockIdx.x) * 256;

    ((uint32_t*) s_unpacked)[t] = ((uint32_t*) g_unpacked)[t];
    __syncthreads();

    // 16 spans of 16 weights to guarantee alignment for any K
    const int spans = 16;
    const int len = 256 / spans;
    if (t < spans)
    {
        int i = len * t;
        int j = K * t;
        int k = 32;
        uint32_t buf = 0;
        for (int n = 0; n < len; ++n)
        {
            uint32_t v = (uint32_t) s_unpacked[i];
            v &= ((1 << K) - 1);
            k -= K;
            buf |= (v << k);
            if (k <= 16)
            {
                s_packed[j] = (uint16_t) (buf >> 16);
                buf <<= 16;
                k += 16;
                j++;
            }
            i++;
        }
    }
    __syncthreads();

    if (t < packed_size / 2)
        ((uint32_t*) g_packed)[t] = SWAP16(((uint32_t*) s_packed)[t]);;
}

#define __(i) pack_trellis_kernel<i>
constexpr auto pack_trellis_kernel_instances = std::array
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

void pack_trellis
(
    at::Tensor packed,
    at::Tensor unpacked,
    int K
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1);
    TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);

    int rows = packed.size(0);
    int cols = packed.size(1);

    dim3 blockDim(128);
    dim3 gridDim(rows, cols);

    // Bind the selected instantiation to a local before launching. Torch's HIPify
    // rewrites `<<<...>>>` launches by scanning backwards for the kernel name and
    // mis-parses an indexed expression such as `instances[K - 1]`, emitting broken
    // `hipLaunchKernelGGL(...)` code. A plain identifier is rewritten correctly and
    // the CUDA build is unchanged.
    auto pack_trellis_kernel = pack_trellis_kernel_instances[K - 1];
    pack_trellis_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) packed.data_ptr(),
        (const uint16_t*) unpacked.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

template <int K>
__global__ __launch_bounds__(128)
void unpack_trellis_kernel
(
    uint16_t* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed
)
{
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_packed[packed_size];

    int t = threadIdx.x;
    g_packed += (gridDim.x * blockIdx.y + blockIdx.x) * packed_size;
    g_unpacked += (gridDim.x * blockIdx.y + blockIdx.x) * 256;

    // Read packed tile
    if (t < packed_size / 2)
        ((uint32_t*) s_packed)[t] = ((uint32_t*) g_packed)[t];
    __syncthreads();

    // Index two words
    int b0 = t * 2 * K + K - 16 + 256 * K;          // start of word0
    int b1 = b0 + K;                                // start of word1
    int b2 = b1 + 16;                               // end of word1
    int i0 = b0 / 32;                               // uint32 containing first bit of word0
    int i1 = (b2 - 1) / 32;                         // uint32 containing last bit of word1, may be == i0
    int s1 = (i1 + 1) * 32 - b2;                    // shift to align word1 to 32-bit boundary

    // Load 32-64 bits containing word0 and word1, overlapping by 16-K bits, correct for endianness
    uint32_t a = ((uint32_t*) s_packed)[i0 % (K * 256 / 32)];
    uint32_t b = ((uint32_t*) s_packed)[i1 % (K * 256 / 32)];
//    a = SWAP16(a);
//    b = SWAP16(b);

    // Shift into place
    uint32_t w1 = __funnelshift_r(b, a, s1);
    uint32_t w0 = w1 >> K;
    w0 &= 0xffff;
    w1 &= 0xffff;

    // Store
    uint32_t word01 = (w1 << 16) | w0;
    ((uint32_t*)g_unpacked)[t] = word01;
}

#define __(i) unpack_trellis_kernel<i>
constexpr auto unpack_trellis_kernel_instances = std::array
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

void unpack_trellis
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1);
    TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);

    int rows = packed.size(0);
    int cols = packed.size(1);

    dim3 blockDim(128);
    dim3 gridDim(cols, rows);

    // See pack_trellis: an indexed instantiation is mis-parsed by HIPify.
    auto unpack_trellis_kernel = unpack_trellis_kernel_instances[K - 1];
    unpack_trellis_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

__global__ __launch_bounds__(32)
void pack_signs_kernel
(
    uint16_t* __restrict__ g_packed,
    const uint16_t* __restrict__ g_unpacked,
    int cols
)
{
    int t = threadIdx.x;
    int idx = 32 * blockIdx.x + t;
    if (idx >= cols) return;
    g_unpacked += 16 * idx;
    g_packed += idx;

    // Not efficient but whatever
    uint16_t out = 0;
    for (int i = 0; i < 16; ++i)
    {
        uint16_t v = *g_unpacked++;
        v &= 0x8000;
        out >>= 1;
        out |= v;
    }

    *g_packed = out;
}

void pack_signs
(
    at::Tensor packed,
    at::Tensor unpacked
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(unpacked, kHalf);
    TORCH_CHECK_DTYPE(packed, kShort);

    int cols = packed.size(0);
    dim3 blockDim(32);
    dim3 gridDim(CEIL_DIVIDE(cols, 32));

    pack_signs_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) packed.data_ptr(),
        (const uint16_t*) unpacked.data_ptr(),
        cols
    );
    cuda_check(cudaPeekAtLastError());
}


// R4NX (and Q4NX) trellis packing: 64 kv-bit states per 256-weight tile,
// concatenated MSB-first and stored as little-endian uint16 pairs. kv = 10, 12,
// 14 cover the fractional 2.5/3.0/3.5 bpw rates and kv = 16 is 4.0 bpw. The
// packer is codebook independent, so the same kernel serves both families; it
// lives here, outside the CUDA-only Q4NX quantizer, so ROCm can pack and unpack
// R4NX checkpoints too.
__global__ __launch_bounds__(128)
void pack_trellis_r4nx_kernel
(
    uint16_t* __restrict__ g_packed,
    const int16_t* __restrict__ g_states,
    int64_t tiles,
    int kv
)
{
    const int64_t tile = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (tile >= tiles) return;

    const uint32_t mask = (1u << kv) - 1u;
    uint16_t* packed = g_packed + tile * 4 * kv;
    const int16_t* states = g_states + tile * 64;

    uint32_t buf = 0;
    int bits = 0;
    int word = 0;
    for (int step = 0; step < 64; ++step)
    {
        buf = (buf << kv) | (static_cast<uint32_t>(static_cast<uint16_t>(states[step])) & mask);
        bits += kv;
        if (bits >= 16)
        {
            bits -= 16;
            // Each 32-bit word is stored little-endian, i.e. its low uint16
            // second in memory.
            packed[word ^ 1] = static_cast<uint16_t>(buf >> bits);
            buf &= (1u << bits) - 1u;
            ++word;
        }
    }
}

void pack_trellis_r4nx(at::Tensor packed, at::Tensor states, int kv)
{
    const at::cuda::OptionalCUDAGuard device_guard(states.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(r4nx_supports_step_bits(kv), "R4NX packing requires kv in {10, 12, 14, 16}");
    // The kernel walks one 64-entry state row per tile, so the row stride must
    // be exactly 64 (a wider row would make it read the wrong tile).
    TORCH_CHECK(states.dim() == 2 && states.size(1) == 64,
                "R4NX packing expects [tiles, 64] int16 state fields");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == states.size(0) && packed.size(1) == 4 * kv,
                "R4NX packing expects [tiles, 4 * kv] int16 output");
    TORCH_CHECK_DTYPE(states, kShort);
    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK(states.device() == packed.device() && states.is_contiguous() && packed.is_contiguous(),
                "R4NX packing requires contiguous buffers on one device");

    const int64_t tiles = states.size(0);
    if (!tiles) return;

    dim3 blockDim(128);
    dim3 gridDim(static_cast<unsigned>(CEIL_DIVIDE(tiles, 128)));
    pack_trellis_r4nx_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) packed.data_ptr(),
        (const int16_t*) states.data_ptr(),
        tiles,
        kv
    );
    cuda_check(cudaPeekAtLastError());
}

__global__ __launch_bounds__(128)
void unpack_trellis_r4nx_kernel
(
    int16_t* __restrict__ g_states,
    const uint16_t* __restrict__ g_packed,
    int64_t tiles,
    int kv
)
{
    const int64_t tile = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (tile >= tiles) return;

    const uint32_t mask = (1u << kv) - 1u;
    const uint16_t* packed = g_packed + tile * 4 * kv;
    int16_t* states = g_states + tile * 64;

    uint32_t buf = 0;
    int bits = 0;
    int word = 0;
    for (int step = 0; step < 64; ++step)
    {
        while (bits < kv)
        {
            buf = (buf << 16) | packed[word ^ 1];
            bits += 16;
            ++word;
        }
        bits -= kv;
        states[step] = static_cast<int16_t>((buf >> bits) & mask);
        buf &= (1u << bits) - 1u;
    }
}

void unpack_trellis_r4nx(at::Tensor states, at::Tensor packed, int kv)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(r4nx_supports_step_bits(kv), "R4NX unpacking requires kv in {10, 12, 14, 16}");
    TORCH_CHECK(states.dim() == 2 && states.size(1) == 64,
                "R4NX unpacking expects [tiles, 64] int16 states");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == states.size(0) && packed.size(1) == 4 * kv,
                "R4NX unpacking expects [tiles, 4 * kv] int16 input");
    TORCH_CHECK_DTYPE(states, kShort);
    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK(states.device() == packed.device() && states.is_contiguous() && packed.is_contiguous(),
                "R4NX unpacking requires contiguous buffers on one device");

    const int64_t tiles = states.size(0);
    if (!tiles) return;

    dim3 blockDim(128);
    dim3 gridDim(static_cast<unsigned>(CEIL_DIVIDE(tiles, 128)));
    unpack_trellis_r4nx_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (int16_t*) states.data_ptr(),
        (const uint16_t*) packed.data_ptr(),
        tiles,
        kv
    );
    cuda_check(cudaPeekAtLastError());
}
