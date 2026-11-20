// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: MIT
// Beam Viterbi encoder for the R4NX codebook.
//
// R4NX reuses the Q4NX packing (same 256-weight tile, same kv-bit states, same
// 64 zero-prefixed states) but its levels are the signed bytes of
// `state * R4NX_MULTIPLIER`, so a candidate's four values are read directly
// instead of being derived from two nibble sums. The search itself is the same
// beam/exact Viterbi as the Q4NX encoder.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <limits>

#include "r4nx.cuh"

namespace {

using Key = unsigned long long;
constexpr int tile_weights = 256;
constexpr int steps = 64;
constexpr int error_levels = 256;  // A byte takes every value from 0 to 255.
constexpr int slot_bits = 7;  // Up to 128 live beam states.
constexpr int slot_mask = (1 << slot_bits) - 1;
constexpr Key empty_key = ~Key{0};

// The level a byte encodes, as a signed 8-bit value.
__device__ __forceinline__ int r4nx_level(uint32_t byte)
{
    return byte < 128 ? int(byte) : int(byte) - 256;
}

template <bool lookup = false>
__device__ __forceinline__ Key candidate_key(uint32_t state, int previous, float prior, const float* target)
{
    const uint32_t bytes = r4nx_bytes(state);
    const int q0 = bytes & 255;
    const int q1 = (bytes >> 8) & 255;
    const int q2 = (bytes >> 16) & 255;
    const int q3 = bytes >> 24;
    const float e0 = lookup ? target[q0] : r4nx_level(q0) - target[0];
    const float e1 = lookup ? target[error_levels + q1] : r4nx_level(q1) - target[1];
    const float e2 = lookup ? target[2 * error_levels + q2] : r4nx_level(q2) - target[2];
    const float e3 = lookup ? target[3 * error_levels + q3] : r4nx_level(q3) - target[3];
    const float cost = prior + (e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3);
    const uint32_t path = (state << slot_bits) | previous;
    return (Key{__float_as_uint(cost)} << 32) | path;
}

// Select a cutoff by cost and state so equal-cost candidates are retained
// deterministically. The state identifies the predecessor class and suffix.
template <int classes, int beam>
__device__ Key beam_cutoff(const Key* candidates)
{
    __shared__ unsigned histogram[256];
    __shared__ unsigned rank;
    __shared__ Key prefix, mask;
    __shared__ bool done;
    if (threadIdx.x == 0)
    {
        rank = beam;
        prefix = mask = 0;
        done = false;
    }
    __syncthreads();

    for (int shift = 56; shift >= 0; shift -= 8)
    {
        histogram[threadIdx.x] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < classes; i += blockDim.x)
        {
            const Key key = candidates[i];
            if (key != empty_key && (key & mask) == prefix)
                atomicAdd(&histogram[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            unsigned bucket = 0;
            unsigned remaining = rank;
            while (remaining > histogram[bucket])
                remaining -= histogram[bucket++];
            rank = remaining;
            prefix |= Key{bucket} << shift;
            mask |= Key{255} << shift;
            done = rank == histogram[bucket];
        }
        __syncthreads();
        if (done) break;
    }
    return prefix | ~mask;
}

template <int KV, int beam>
__global__ void __launch_bounds__(tile_weights) quantize_kernel(
    const float* input, float* output, int16_t* suffixes, uint32_t* history)
{
    constexpr int class_count = 1 << (32 - 2 * KV);
    constexpr int suffix_count = 1 << KV;
    constexpr bool exact = beam == class_count;
    __shared__ float targets[tile_weights];
    __shared__ float errors[exact ? 4 * error_levels : 1];
    __shared__ Key candidates[class_count];
    __shared__ Key kept[beam];
    __shared__ float costs[beam];
    __shared__ int classes[beam];
    __shared__ unsigned count;

    const size_t tile_offset = size_t{blockIdx.x} * tile_weights;
    history += size_t{blockIdx.x} * steps * beam;
    targets[threadIdx.x] = input[tile_offset + threadIdx.x] / EXL3_R4NX_ALPHA;
    if (threadIdx.x >= steps)
        suffixes[tile_offset + threadIdx.x] = 0;
    if (threadIdx.x == 0)
    {
        costs[0] = 0.f;
        classes[0] = 0;  // The prefix preceding this tile is zero.
    }
    __syncthreads();

    for (int step = 0; step < steps; ++step)
    {
        const float* target = targets + 4 * step;
        const int live = step == 0 ? 1 : beam;
        if constexpr (!exact)
        {
            if (threadIdx.x == 0)
                count = 0;
        }
        if constexpr (KV == 16)
        {
            // 4.0 bpw: the state is the whole 16-bit suffix and there is a
            // single overlap class, so the step is a plain 65536-way argmin.
            static_assert(class_count == 1);
            __shared__ Key warp_best[tile_weights / 32];
            Key best = empty_key;
            for (int suffix = threadIdx.x; suffix < suffix_count; suffix += tile_weights)
                best = min(best, candidate_key(uint32_t(suffix), 0, 0.f, target));
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                best = min(best, __shfl_xor_sync(0xffffffffu, best, offset));
            if ((threadIdx.x & 31) == 0)
                warp_best[threadIdx.x / 32] = best;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                Key total = empty_key;
                #pragma unroll
                for (int i = 0; i < tile_weights / 32; ++i)
                    total = min(total, warp_best[i]);
                candidates[0] = total;
            }
        }
        else if constexpr (exact)
        {
            // W3.5 keeps every overlap class. Sixteen threads reduce each
            // class independently, with slots indexed directly by class.
            constexpr int lanes = 16;
            static_assert(class_count * lanes == tile_weights);
            const int successor = threadIdx.x / lanes;
            const int lane = threadIdx.x % lanes;
            // Reuse signed errors while preserving the FP32 sum of squares
            // and its rounding order in candidate_key.
            for (int i = threadIdx.x; i < 4 * error_levels; i += tile_weights)
                errors[i] = r4nx_level(i % error_levels) - target[i / error_levels];
            __syncthreads();
            Key best = empty_key;
            for (int previous = 0; previous < live; ++previous)
            {
                const float prior = costs[previous];
                for (int suffix = lane * class_count + successor; suffix < suffix_count; suffix += lanes * class_count)
                {
                    const uint32_t state = (previous << KV) | suffix;
                    best = min(best, candidate_key<true>(state, previous, prior, errors));
                }
            }
            for (int offset = lanes / 2; offset > 0; offset /= 2)
                best = min(best, __shfl_xor_sync(0xffffffffu, best, offset, lanes));
            if (lane == 0)
                candidates[successor] = best;
        }
        else if constexpr (KV == 12)
        {
            // W3.0's successor class is the low eight suffix bits. Each thread
            // owns one successor class and covers its sixteen suffixes, so the
            // predecessor's prior cost and class are loaded once per
            // predecessor. Four accumulators keep the serial min chain short.
            static_assert(class_count == tile_weights);
            constexpr int suffixes_per_thread = suffix_count / tile_weights;
            Key best[4] = {empty_key, empty_key, empty_key, empty_key};
            for (int previous = 0; previous < live; ++previous)
            {
                const float prior = costs[previous];
                const uint32_t state = (classes[previous] << KV) | threadIdx.x;
                #pragma unroll
                for (int j = 0; j < suffixes_per_thread; ++j)
                {
                    const Key key = candidate_key(state + j * tile_weights, previous, prior, target);
                    best[j & 3] = min(best[j & 3], key);
                }
            }
            candidates[threadIdx.x] = min(min(best[0], best[1]), min(best[2], best[3]));
        }
        else
        {
            // W2.5 shares the low eight bits with the owning thread. The next
            // two suffix bits and low two predecessor bits choose its register.
            static_assert(KV == 10 && suffix_count == 4 * tile_weights && class_count == 16 * tile_weights);
            Key best[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                best[i] = empty_key;
            for (int previous = 0; previous < live; ++previous)
            {
                const int previous_class = classes[previous];
                const float prior = costs[previous];
                const uint32_t state = (previous_class << KV) | threadIdx.x;
                const Key key0 = candidate_key(state, previous, prior, target);
                const Key key1 = candidate_key(state | tile_weights, previous, prior, target);
                const Key key2 = candidate_key(state | (2 * tile_weights), previous, prior, target);
                const Key key3 = candidate_key(state | (3 * tile_weights), previous, prior, target);
                // The switch uses constant register indices and is uniform
                // across all threads visiting this predecessor.
                switch (previous_class & 3)
                {
                    case 0:
                        best[0] = min(best[0], key0);
                        best[1] = min(best[1], key1);
                        best[2] = min(best[2], key2);
                        best[3] = min(best[3], key3);
                        break;
                    case 1:
                        best[4] = min(best[4], key0);
                        best[5] = min(best[5], key1);
                        best[6] = min(best[6], key2);
                        best[7] = min(best[7], key3);
                        break;
                    case 2:
                        best[8] = min(best[8], key0);
                        best[9] = min(best[9], key1);
                        best[10] = min(best[10], key2);
                        best[11] = min(best[11], key3);
                        break;
                    case 3:
                        best[12] = min(best[12], key0);
                        best[13] = min(best[13], key1);
                        best[14] = min(best[14], key2);
                        best[15] = min(best[15], key3);
                        break;
                }
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                candidates[i * tile_weights + threadIdx.x] = best[i];
        }
        __syncthreads();

        // The first step can reach at least as many classes as the beam holds.
        Key cutoff = empty_key;
        if constexpr (!exact)
            cutoff = beam_cutoff<class_count, beam>(candidates);
        if constexpr (exact)
        {
            for (int i = threadIdx.x; i < class_count; i += blockDim.x)
            {
                const Key key = candidates[i];
                if (key == empty_key)
                    continue;
                history[step * beam + i] = uint32_t(key);
                costs[i] = __uint_as_float(key >> 32);
                classes[i] = i;
            }
        }
        else
        {
            // Compact the retained keys and rank them by key so slots follow
            // the same ascending-key order as the reference encoder's stable
            // sort. Compaction order is arbitrary, but ranking makes the slot
            // order canonical, so the `previous` tie-break in the next step's
            // key does not depend on which thread won an atomic.
            for (int i = threadIdx.x; i < class_count; i += blockDim.x)
            {
                const Key key = candidates[i];
                if (key == empty_key || key > cutoff)
                    continue;
                kept[atomicAdd(&count, 1u)] = key;
            }
            __syncthreads();
            for (unsigned j = threadIdx.x; j < count; j += blockDim.x)
            {
                const Key key = kept[j];
                unsigned rank = 0;
                for (unsigned k = 0; k < count; ++k)
                    rank += kept[k] < key;
                history[step * beam + rank] = uint32_t(key);
                costs[rank] = __uint_as_float(key >> 32);
                classes[rank] = int((uint32_t(key) >> slot_bits) & (class_count - 1));
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        int best_slot = 0;
        Key best = empty_key;
        for (int slot = 0; slot < beam; ++slot)
        {
            const uint32_t state = history[(steps - 1) * beam + slot] >> slot_bits;
            const Key key = (Key{__float_as_uint(costs[slot])} << 32) | state;
            if (key < best)
            {
                best = key;
                best_slot = slot;
            }
        }
        for (int step = steps - 1; step >= 0; --step)
        {
            const uint32_t path = history[step * beam + best_slot];
            const uint32_t state = path >> slot_bits;
            suffixes[tile_offset + step] = state & (suffix_count - 1);
            const uint32_t bytes = r4nx_bytes(state);
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                output[tile_offset + 4 * step + c] = r4nx_level((bytes >> (8 * c)) & 255) * EXL3_R4NX_ALPHA;
            best_slot = path & slot_mask;
        }
    }
}

template <int KV>
__global__ void pack_kernel(const int16_t* suffixes, uint16_t* packed, int64_t tile_count)
{
    constexpr int words = KV * 2;
    constexpr int suffixes_per_window = (32 + 2 * KV - 2) / KV;
    const int64_t index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const int64_t tile = index / words;
    if (tile >= tile_count)
        return;
    const int start = (index % words) * 32;
    const int step = start / KV;
    uint64_t window = 0;
    #pragma unroll
    for (int i = 0; i < suffixes_per_window; ++i)
    {
        const uint32_t suffix = step + i < steps ? suffixes[tile * tile_weights + step + i] : 0;
        window = (window << KV) | (suffix & ((1u << KV) - 1));
    }
    const uint32_t word = window >> (suffixes_per_window * KV - 32 - start % KV);
    packed[2 * index] = word;
    packed[2 * index + 1] = word >> 16;
}

void quantize_tiles(const at::Tensor& input, at::Tensor& output, at::Tensor& suffixes, int64_t kv, bool scale_search)
{
    TORCH_CHECK(r4nx_supports_step_bits(kv), "R4NX requires 2.5, 3.0, 3.5, or 4.0 bpw");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == tile_weights
        && output.sizes() == input.sizes() && suffixes.sizes() == input.sizes(),
        "R4NX quantization expects matching [tiles, 256] buffers");
    TORCH_CHECK(input.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat
        && suffixes.scalar_type() == at::kShort, "R4NX quantization expects FP32 input/output and int16 suffixes");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && suffixes.is_contiguous()
        && input.device() == output.device() && input.device() == suffixes.device(),
        "R4NX quantization expects contiguous buffers on one device");
    const int64_t tiles = input.size(0);
    if (!tiles)
        return;
    TORCH_CHECK(tiles <= std::numeric_limits<int>::max(), "Too many R4NX tiles");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto history = at::empty({tiles, steps, kv == 16 ? 1 : (scale_search || kv == 14 ? 16 : 128)}, input.options().dtype(at::kInt));
    const float* in = input.const_data_ptr<float>();
    float* out = output.data_ptr<float>();
    auto* codes = suffixes.data_ptr<int16_t>();
    auto* back = reinterpret_cast<uint32_t*>(history.data_ptr<int32_t>());
    if (kv == 16)
        quantize_kernel<16, 1><<<tiles, tile_weights, 0, stream>>>(in, out, codes, back);
    else if (kv == 10 && scale_search)
        quantize_kernel<10, 16><<<tiles, tile_weights, 0, stream>>>(in, out, codes, back);
    else if (kv == 10)
        quantize_kernel<10, 128><<<tiles, tile_weights, 0, stream>>>(in, out, codes, back);
    else if (kv == 12 && scale_search)
        quantize_kernel<12, 16><<<tiles, tile_weights, 0, stream>>>(in, out, codes, back);
    else if (kv == 12)
        quantize_kernel<12, 128><<<tiles, tile_weights, 0, stream>>>(in, out, codes, back);
    else
        quantize_kernel<14, 16><<<tiles, tile_weights, 0, stream>>>(in, out, codes, back);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_trellis(at::Tensor& packed, const at::Tensor& suffixes, int64_t kv)
{
    TORCH_CHECK(r4nx_supports_step_bits(kv), "R4NX requires 2.5, 3.0, 3.5, or 4.0 bpw");
    TORCH_CHECK(suffixes.dim() == 3 && packed.dim() == 3 && suffixes.size(2) == tile_weights
        && packed.size(0) == suffixes.size(0) && packed.size(1) == suffixes.size(1)
        && packed.size(2) == 4 * kv, "R4NX pack received mismatched tile buffers");
    TORCH_CHECK(packed.scalar_type() == at::kShort && suffixes.scalar_type() == at::kShort
        && packed.is_contiguous() && suffixes.is_contiguous() && packed.device() == suffixes.device(),
        "R4NX pack expects contiguous int16 buffers on one device");
    const int64_t tiles = suffixes.numel() / tile_weights;
    if (!tiles)
        return;
    const int64_t blocks = (tiles * 2 * kv + 127) / 128;
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "Too many R4NX tiles to pack");
    const c10::cuda::CUDAGuard guard(suffixes.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* codes = suffixes.const_data_ptr<int16_t>();
    auto* words = reinterpret_cast<uint16_t*>(packed.data_ptr<int16_t>());
    if (kv == 10)
        pack_kernel<10><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    else if (kv == 12)
        pack_kernel<12><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    else if (kv == 14)
        pack_kernel<14><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    else
        pack_kernel<16><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_exllamav3, m)
{
    m.def("quantize_tiles_r4nx(Tensor input, Tensor(a!) output, Tensor(b!) suffixes, int kv, bool scale_search) -> ()");
    m.def("pack_trellis_r4nx(Tensor(a!) packed, Tensor suffixes, int kv) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_exllamav3, CUDA, m)
{
    m.impl("quantize_tiles_r4nx", &quantize_tiles);
    m.impl("pack_trellis_r4nx", &pack_trellis);
}
