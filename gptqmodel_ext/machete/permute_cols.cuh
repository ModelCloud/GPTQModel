#include <torch/all.h>

torch::Tensor permute_cols(torch::Tensor const& A, torch::Tensor const& perm);