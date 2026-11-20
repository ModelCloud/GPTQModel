// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: MIT
// Beam Viterbi adapted from AlexBrandes (Alex Brandes), AlexBrandes/exllamav3
// at 0674697c45e2450ffd4538eb5ade0cf8e6a78f6d, q4n_quant.cu.
// Copyright (c) 2025 Turboderp. See ../LICENSE.exllamav3.
// ModelCloud extended Q4NX to 3.5 bits per weight with exact Viterbi search.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <limits>

#include "q4nx.cuh"

namespace {

using Key = unsigned long long;
constexpr int tile_weights = 256;
constexpr int steps = 64;
constexpr int slot_bits = 7;  // Up to 128 live beam states.
constexpr int slot_mask = (1 << slot_bits) - 1;
constexpr Key empty_key = ~Key{0};

__device__ __forceinline__ Key candidate_key(uint32_t state, int previous, float prior, const float* target)
{
    const uint32_t sums = q4nx_nibble_sums(state);
    const float e0 = int(sums & 255) - 15 - target[0];
    const float e1 = int((sums >> 8) & 255) - 15 - target[1];
    const float e2 = int((sums >> 16) & 255) - 15 - target[2];
    const float e3 = int(sums >> 24) - 15 - target[3];
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
__global__ void __launch_bounds__(256) quantize_kernel(
    const float* input, float* output, int16_t* suffixes, uint32_t* history)
{
    constexpr int class_count = 1 << (32 - 2 * KV);
    constexpr int suffix_count = 1 << KV;
    constexpr bool exact = beam == class_count;
    __shared__ float targets[tile_weights];
    __shared__ Key candidates[class_count];
    __shared__ float costs[beam];
    __shared__ int classes[beam];
    __shared__ unsigned count;

    const size_t tile_offset = size_t{blockIdx.x} * tile_weights;
    history += size_t{blockIdx.x} * steps * beam;
    targets[threadIdx.x] = input[tile_offset + threadIdx.x] / EXL3_Q4NX_ALPHA;
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
        for (int i = threadIdx.x; i < class_count; i += blockDim.x)
            candidates[i] = empty_key;
        if (threadIdx.x == 0)
            count = 0;
        __syncthreads();

        const float* target = targets + 4 * step;
        if constexpr (exact)
        {
            // W3.5 keeps every overlap class. Sixteen threads reduce each
            // class independently, with slots indexed directly by class.
            static_assert(class_count == 16);
            const int successor = threadIdx.x / 16;
            const int lane = threadIdx.x % 16;
            const int prefixes = step == 0 ? suffix_count / class_count : suffix_count;
            Key best = empty_key;
            for (int prefix = lane; prefix < prefixes; prefix += 16)
            {
                const uint32_t state = prefix * class_count + successor;
                const int previous = state >> KV;
                const Key key = candidate_key(state, previous, costs[previous], target);
                best = min(best, key);
            }
            for (int offset = 8; offset > 0; offset /= 2)
                best = min(best, __shfl_xor_sync(0xffffffffu, best, offset, 16));
            if (lane == 0)
                candidates[successor] = best;
        }
        else
        {
            const int live = step == 0 ? 1 : beam;
            for (int i = threadIdx.x; i < live * suffix_count; i += blockDim.x)
            {
                const int previous = i >> KV;
                const uint32_t state = (classes[previous] << KV) | (i & (suffix_count - 1));
                const Key key = candidate_key(state, previous, costs[previous], target);
                atomicMin(&candidates[state & (class_count - 1)], key);
            }
        }
        __syncthreads();

        // The first step can reach at least as many classes as the beam holds.
        Key cutoff = empty_key;
        if constexpr (!exact)
            cutoff = beam_cutoff<class_count, beam>(candidates);
        for (int i = threadIdx.x; i < class_count; i += blockDim.x)
        {
            const Key key = candidates[i];
            if (key == empty_key || key > cutoff)
                continue;
            const unsigned slot = exact ? i : atomicAdd(&count, 1u);
            history[step * beam + slot] = uint32_t(key);
            costs[slot] = __uint_as_float(key >> 32);
            classes[slot] = i;
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
            const uint32_t sums = q4nx_nibble_sums(state);
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                output[tile_offset + 4 * step + c] = (int((sums >> (8 * c)) & 255) - 15) * EXL3_Q4NX_ALPHA;
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
    TORCH_CHECK(q4nx_supports_step_bits(kv), "Q4NX requires 2.5, 3.0, or 3.5 bpw");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == tile_weights
        && output.sizes() == input.sizes() && suffixes.sizes() == input.sizes(),
        "Q4NX quantization expects matching [tiles, 256] buffers");
    TORCH_CHECK(input.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat
        && suffixes.scalar_type() == at::kShort, "Q4NX quantization expects FP32 input/output and int16 suffixes");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && suffixes.is_contiguous()
        && input.device() == output.device() && input.device() == suffixes.device(),
        "Q4NX quantization expects contiguous buffers on one device");
    const int64_t tiles = input.size(0);
    if (!tiles)
        return;
    TORCH_CHECK(tiles <= std::numeric_limits<int>::max(), "Too many Q4NX tiles");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto history = at::empty({tiles, steps, scale_search || kv == 14 ? 16 : 128}, input.options().dtype(at::kInt));
    const float* in = input.const_data_ptr<float>();
    float* out = output.data_ptr<float>();
    auto* codes = suffixes.data_ptr<int16_t>();
    auto* back = reinterpret_cast<uint32_t*>(history.data_ptr<int32_t>());
    if (kv == 10 && scale_search)
        quantize_kernel<10, 16><<<tiles, 256, 0, stream>>>(in, out, codes, back);
    else if (kv == 10)
        quantize_kernel<10, 128><<<tiles, 256, 0, stream>>>(in, out, codes, back);
    else if (kv == 12 && scale_search)
        quantize_kernel<12, 16><<<tiles, 256, 0, stream>>>(in, out, codes, back);
    else if (kv == 12)
        quantize_kernel<12, 128><<<tiles, 256, 0, stream>>>(in, out, codes, back);
    else
        quantize_kernel<14, 16><<<tiles, 256, 0, stream>>>(in, out, codes, back);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_trellis(at::Tensor& packed, const at::Tensor& suffixes, int64_t kv)
{
    TORCH_CHECK(q4nx_supports_step_bits(kv), "Q4NX requires 2.5, 3.0, or 3.5 bpw");
    TORCH_CHECK(suffixes.dim() == 3 && packed.dim() == 3 && suffixes.size(2) == tile_weights
        && packed.size(0) == suffixes.size(0) && packed.size(1) == suffixes.size(1)
        && packed.size(2) == 4 * kv, "Q4NX pack received mismatched tile buffers");
    TORCH_CHECK(packed.scalar_type() == at::kShort && suffixes.scalar_type() == at::kShort
        && packed.is_contiguous() && suffixes.is_contiguous() && packed.device() == suffixes.device(),
        "Q4NX pack expects contiguous int16 buffers on one device");
    const int64_t tiles = suffixes.numel() / tile_weights;
    if (!tiles)
        return;
    const int64_t blocks = (tiles * 2 * kv + 127) / 128;
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "Too many Q4NX tiles to pack");
    const c10::cuda::CUDAGuard guard(suffixes.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* codes = suffixes.const_data_ptr<int16_t>();
    auto* words = reinterpret_cast<uint16_t*>(packed.data_ptr<int16_t>());
    if (kv == 10)
        pack_kernel<10><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    else if (kv == 12)
        pack_kernel<12><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    else
        pack_kernel<14><<<blocks, 128, 0, stream>>>(codes, words, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_exllamav3, m)
{
    m.def("quantize_tiles_q4nx(Tensor input, Tensor(a!) output, Tensor(b!) suffixes, int kv, bool scale_search) -> ()");
    m.def("pack_trellis_q4nx(Tensor(a!) packed, Tensor suffixes, int kv) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_exllamav3, CUDA, m)
{
    m.impl("quantize_tiles_q4nx", &quantize_tiles);
    m.impl("pack_trellis_q4nx", &pack_trellis);
}
