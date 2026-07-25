// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>

torch::Tensor amplin_gptq_w4_group128_gemv_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales);

torch::Tensor amplin_gptq_w4_group128_gemv_k12288_wide_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales);

torch::Tensor amplin_gptq_w4_group128_gemv_multirow_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales);

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_v0_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v1_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v2_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v2_sync_a128_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v3_cuda(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_tile_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales);

torch::Tensor amplin_mma_lane_tile_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales);

torch::Tensor amplin_mma_lane_m64_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m64_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_n32_global_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n64_shared_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_n64_shared_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n64_tile4_shared_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n64_tile8_shared_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_n64_tile4_shared_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_n64_tile8_shared_a_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n16_padded_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n16_splitk4_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n16_splitk8_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n16_splitk12_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk12_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk16_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk12_pipe2_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk16_pipe2_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk8_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n32_splitk8_pipe2_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

torch::Tensor amplin_mma_lane_m16_n16_splitk16_cuda(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n);

namespace {

torch::Tensor amplin_gptq_w4_group128_gemv_dispatch(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales) {
  return amplin_gptq_w4_group128_gemv_cuda(input, qweight, scales);
}

torch::Tensor amplin_gptq_w4_group128_gemv_k12288_wide_dispatch(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales) {
  return amplin_gptq_w4_group128_gemv_k12288_wide_cuda(input, qweight, scales);
}

torch::Tensor amplin_gptq_w4_group128_gemv_multirow_dispatch(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales) {
  return amplin_gptq_w4_group128_gemv_multirow_cuda(input, qweight, scales);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_dispatch(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_cuda(input, packed_qweight, packed_scales, logical_n);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_v0_dispatch(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_v0_cuda(input, packed_qweight, packed_scales, logical_n);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v1_dispatch(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_m64_v1_cuda(input, packed_qweight, packed_scales, logical_n);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v2_dispatch(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_m64_v2_cuda(input, packed_qweight, packed_scales, logical_n);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v2_sync_a128_dispatch(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_m64_v2_sync_a128_cuda(
      input,
      packed_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_gptq_w4_group128_gemm_hmma_m64_v3_dispatch(
    torch::Tensor input,
    torch::Tensor packed_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_gptq_w4_group128_gemm_hmma_m64_v3_cuda(
      input,
      packed_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_tile_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales) {
  return amplin_mma_lane_tile_cuda(input, packed_lane_qweight, scales);
}

torch::Tensor amplin_mma_lane_tile_global_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor scales) {
  return amplin_mma_lane_tile_global_a_cuda(
      input,
      packed_lane_qweight,
      scales);
}

torch::Tensor amplin_mma_lane_m64_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m64_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m64_global_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m64_global_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_global_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_global_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n32_global_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_n32_global_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n64_shared_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n64_shared_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n64_shared_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_n64_shared_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n64_tile4_shared_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n64_tile4_shared_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n64_tile8_shared_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n64_tile8_shared_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n64_tile4_shared_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_n64_tile4_shared_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n64_tile8_shared_a_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_n64_tile8_shared_a_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_padded_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_padded_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk4_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_splitk4_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk8_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_splitk8_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk12_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_splitk12_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk12_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk12_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk16_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk16_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk12_pipe2_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk12_pipe2_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk16_pipe2_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk16_pipe2_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk8_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk8_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n32_splitk8_pipe2_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n32_splitk8_pipe2_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

torch::Tensor amplin_mma_lane_m16_n16_splitk16_dispatch(
    torch::Tensor input,
    torch::Tensor packed_lane_qweight,
    torch::Tensor packed_scales,
    int64_t logical_n) {
  return amplin_mma_lane_m16_n16_splitk16_cuda(
      input,
      packed_lane_qweight,
      packed_scales,
      logical_n);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_amplin, m) {
  m.def("gemv(Tensor input, Tensor qweight, Tensor scales) -> Tensor");
  m.def("gemv_k12288_wide(Tensor input, Tensor qweight, Tensor scales) -> Tensor");
  m.def("gemv_multirow(Tensor input, Tensor qweight, Tensor scales) -> Tensor");
  m.def("gemm_hmma(Tensor input, Tensor packed_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def("gemm_hmma_v0(Tensor input, Tensor packed_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def("gemm_hmma_m64_v1(Tensor input, Tensor packed_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def("gemm_hmma_m64_v2(Tensor input, Tensor packed_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "gemm_hmma_m64_v2_sync_a128(Tensor input, Tensor packed_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def("gemm_hmma_m64_v3(Tensor input, Tensor packed_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def("mma_lane_tile(Tensor input, Tensor packed_lane_qweight, Tensor scales) -> Tensor");
  m.def("mma_lane_tile_global_a(Tensor input, Tensor packed_lane_qweight, Tensor scales) -> Tensor");
  m.def(
      "mma_lane_m64(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m64_global_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_global_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_n32_global_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n64_shared_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_n64_shared_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n64_tile4_shared_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n64_tile8_shared_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_n64_tile4_shared_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_n64_tile8_shared_a(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_n64_splitk12x2_coop_interleaved(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n16_padded(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n16_splitk4(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n16_splitk8(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n16_splitk12(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk12(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk16(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk12_pipe2(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk12_pipe2_interleaved(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n64_splitk24_pipe2_interleaved(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m32_n64_splitk24_pipe2_interleaved(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n64_splitk12x2_coop_interleaved(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk16_pipe2(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk8(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n32_splitk8_pipe2(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
  m.def(
      "mma_lane_m16_n16_splitk16(Tensor input, Tensor packed_lane_qweight, Tensor packed_scales, int logical_n) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_amplin, CUDA, m) {
  m.impl("gemv", &amplin_gptq_w4_group128_gemv_dispatch);
  m.impl("gemv_k12288_wide", &amplin_gptq_w4_group128_gemv_k12288_wide_dispatch);
  m.impl("gemv_multirow", &amplin_gptq_w4_group128_gemv_multirow_dispatch);
  m.impl("gemm_hmma", &amplin_gptq_w4_group128_gemm_hmma_dispatch);
  m.impl("gemm_hmma_v0", &amplin_gptq_w4_group128_gemm_hmma_v0_dispatch);
  m.impl("gemm_hmma_m64_v1", &amplin_gptq_w4_group128_gemm_hmma_m64_v1_dispatch);
  m.impl("gemm_hmma_m64_v2", &amplin_gptq_w4_group128_gemm_hmma_m64_v2_dispatch);
  m.impl(
      "gemm_hmma_m64_v2_sync_a128",
      &amplin_gptq_w4_group128_gemm_hmma_m64_v2_sync_a128_dispatch);
  m.impl("gemm_hmma_m64_v3", &amplin_gptq_w4_group128_gemm_hmma_m64_v3_dispatch);
  m.impl("mma_lane_tile", &amplin_mma_lane_tile_dispatch);
  m.impl("mma_lane_tile_global_a", &amplin_mma_lane_tile_global_a_dispatch);
  m.impl("mma_lane_m64", &amplin_mma_lane_m64_dispatch);
  m.impl("mma_lane_m64_global_a", &amplin_mma_lane_m64_global_a_dispatch);
  m.impl("mma_lane_m32_global_a", &amplin_mma_lane_m32_global_a_dispatch);
  m.impl("mma_lane_m32_n32_global_a", &amplin_mma_lane_m32_n32_global_a_dispatch);
  m.impl("mma_lane_m16_n64_shared_a", &amplin_mma_lane_m16_n64_shared_a_dispatch);
  m.impl("mma_lane_m32_n64_shared_a", &amplin_mma_lane_m32_n64_shared_a_dispatch);
  m.impl(
      "mma_lane_m16_n64_tile4_shared_a", &amplin_mma_lane_m16_n64_tile4_shared_a_dispatch);
  m.impl(
      "mma_lane_m16_n64_tile8_shared_a", &amplin_mma_lane_m16_n64_tile8_shared_a_dispatch);
  m.impl(
      "mma_lane_m32_n64_tile4_shared_a", &amplin_mma_lane_m32_n64_tile4_shared_a_dispatch);
  m.impl(
      "mma_lane_m32_n64_tile8_shared_a", &amplin_mma_lane_m32_n64_tile8_shared_a_dispatch);
  m.impl(
      "mma_lane_m32_n64_splitk12x2_coop_interleaved",
      &amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_dispatch);
  m.impl("mma_lane_m16_n16_padded", &amplin_mma_lane_m16_n16_padded_dispatch);
  m.impl("mma_lane_m16_n16_splitk4", &amplin_mma_lane_m16_n16_splitk4_dispatch);
  m.impl("mma_lane_m16_n16_splitk8", &amplin_mma_lane_m16_n16_splitk8_dispatch);
  m.impl("mma_lane_m16_n16_splitk12", &amplin_mma_lane_m16_n16_splitk12_dispatch);
  m.impl("mma_lane_m16_n32_splitk12", &amplin_mma_lane_m16_n32_splitk12_dispatch);
  m.impl("mma_lane_m16_n32_splitk16", &amplin_mma_lane_m16_n32_splitk16_dispatch);
  m.impl(
      "mma_lane_m16_n32_splitk12_pipe2",
      &amplin_mma_lane_m16_n32_splitk12_pipe2_dispatch);
  m.impl(
      "mma_lane_m16_n32_splitk12_pipe2_interleaved",
      &amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_dispatch);
  m.impl(
      "mma_lane_m16_n64_splitk24_pipe2_interleaved",
      &amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_dispatch);
  m.impl(
      "mma_lane_m32_n64_splitk24_pipe2_interleaved",
      &amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_dispatch);
  m.impl(
      "mma_lane_m16_n64_splitk12x2_coop_interleaved",
      &amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_dispatch);
  m.impl(
      "mma_lane_m16_n32_splitk16_pipe2",
      &amplin_mma_lane_m16_n32_splitk16_pipe2_dispatch);
  m.impl("mma_lane_m16_n32_splitk8", &amplin_mma_lane_m16_n32_splitk8_dispatch);
  m.impl(
      "mma_lane_m16_n32_splitk8_pipe2",
      &amplin_mma_lane_m16_n32_splitk8_pipe2_dispatch);
  m.impl("mma_lane_m16_n16_splitk16", &amplin_mma_lane_m16_n16_splitk16_dispatch);
}
