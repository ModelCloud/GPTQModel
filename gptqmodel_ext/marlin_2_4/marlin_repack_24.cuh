#include <torch/all.h>

__global__ void gptq_repack_kernel_2_4(
  uint32_t* in,
  uint32_t* out,
  int m,
  int n
);

torch::Tensor gptq_repack_2_4(
    torch::Tensor W
);