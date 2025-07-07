#include <torch/all.h>
#include "core/scalar_type.hpp"

namespace machete {

using namespace vllm;

torch::Tensor mm(torch::Tensor const& A, torch::Tensor const& B,
                 int64_t b_type_id,
                 std::optional<at::ScalarType> const& maybe_out_type,
                 std::optional<torch::Tensor> const& maybe_group_scales,
                 std::optional<torch::Tensor> const& maybe_group_zeros,
                 std::optional<int64_t> maybe_group_size,
                 std::optional<torch::Tensor> const& maybe_channel_scales,
                 std::optional<torch::Tensor> const& maybe_token_scales,
                 std::optional<std::string> maybe_schedule);


torch::Tensor prepack_B(
    torch::Tensor const& B, at::ScalarType const& a_type, int64_t b_type_id,
    std::optional<at::ScalarType> const& maybe_group_scales_type);
};