#include <torch/all.h>

namespace marlin {

template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n);

}  // namespace marlin

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

