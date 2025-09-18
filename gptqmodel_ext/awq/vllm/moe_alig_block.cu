#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

const static size_t NUM_MAX_EXPERTS = 64;

#define VLLM_DISPATCH_CASE_INTEGRAL_TYPES(...)             \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define VLLM_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

template <typename scalar_t>
__global__ void moe_alig_block_size_kernel(scalar_t *__restrict__ topk_ids, 
                                int32_t *sorted_token_ids, 
                                int32_t *expert_ids, 
                                int32_t *total_tokens_post_pad,
                                int32_t num_experts, 
                                int32_t block_size, 
                                size_t numel) {
    const size_t tokens_per_thread = ((numel + blockDim.x - 1) / blockDim.x);
    const size_t start_idx = threadIdx.x * tokens_per_thread;
    __shared__ int32_t tokens_cnts[NUM_MAX_EXPERTS + 1][NUM_MAX_EXPERTS];
    __shared__ int32_t cumsum[NUM_MAX_EXPERTS + 1];
    for(int i = 0;i < num_experts;i++){
        tokens_cnts[threadIdx.x + 1][i] = 0;
    }

    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
        ++tokens_cnts[threadIdx.x + 1][topk_ids[i]]; 
    }

    __syncthreads();

    tokens_cnts[0][threadIdx.x] = 0;
    for(int i=1;i<=blockDim.x;++i){
        tokens_cnts[i][threadIdx.x] += tokens_cnts[i-1][threadIdx.x];
    }

    __syncthreads();

    if(threadIdx.x ==0){
        cumsum[0] = 0;
        for(int i=1;i<=num_experts;++i){
            cumsum[i] = cumsum[i-1] + (tokens_cnts[blockDim.x][i - 1] + block_size - 1) / block_size * block_size;
        }
        *total_tokens_post_pad = cumsum[num_experts];
    }

    __syncthreads();

    for(int i= cumsum[threadIdx.x];i<cumsum[threadIdx.x + 1];i += block_size){
        expert_ids[i / block_size] = threadIdx.x;
    }

    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = tokens_cnts[threadIdx.x][expert_id] + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        ++tokens_cnts[threadIdx.x][expert_id];
    }
}

void moe_alig_block_size(
    torch::Tensor topk_ids,
    int num_experts,
    int block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad) {
    const at::cuda::OptionalCUDAGuard device_guard_topk_ids(device_of(topk_ids));
    const at::cuda::OptionalCUDAGuard device_guard_sorted(device_of(sorted_token_ids));
    const at::cuda::OptionalCUDAGuard device_guard_experts(device_of(experts_ids));
    const at::cuda::OptionalCUDAGuard device_guard_num_tokens(device_of(num_tokens_post_pad));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    assert(num_experts <= NUM_MAX_EXPERTS);
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_alig_block_size_kernel", [&] {
        moe_alig_block_size_kernel<scalar_t><<<1, num_experts, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), 
            sorted_token_ids.data_ptr<int32_t>(), 
            experts_ids.data_ptr<int32_t>(), 
            num_tokens_post_pad.data_ptr<int32_t>(), 
            num_experts,
            block_size,
            topk_ids.numel());
    });
}