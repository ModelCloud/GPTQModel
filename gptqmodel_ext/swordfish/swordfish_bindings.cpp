// SPDX-FileCopyrightText: 2026 AlpinDale and the dphnAI/sonar contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Swordfish: Blackwell (sm100/sm110) weight-quantized GEMM kernels.
// Vendored from https://github.com/dphnAI/sonar and used under the terms of
// the GNU Affero General Public License v3.0 or later. See LICENSE in this
// directory or /licenses/SWORDFISH at the project root for the full license.
//
// Stable ABI torch.op schema definitions for the Swordfish kernel family.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

STABLE_TORCH_LIBRARY_FRAGMENT(gptqmodel_swordfish, ops) {
  // Pack a GPTQ int4/int8 weight into the Swordfish ABI v1 block-linear
  // layout. perm applies the act_order row sort during the repack.
  ops.def(
      "swordfish_prepack_B(Tensor b_q_weight, Tensor? perm, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");

  // w4a16/w8a16 decode GEMM over the Swordfish ABI v1 packed weight.
  // Activations are expected to already be in the group-sorted order that
  // swordfish_prepack_B produced for the weight; group_zps holds prescaled
  // (8 - zp) * scale per group when present.
  ops.def(
      "swordfish_mm(Tensor a, Tensor b_packed, Tensor group_scales, "
      "Tensor? group_zps, int num_bits, int group_size, "
      "SymInt size_k, SymInt size_n) -> Tensor");

  // Dequantize Swordfish-packed weights to dense fp16/bf16, optionally
  // out-major transposed and expert-stacked, for the dense tier.
  ops.def(
      "swordfish_dequant_dense(Tensor b_packed, Tensor group_scales, "
      "Tensor? group_zps, int num_bits, int group_size, SymInt size_k, "
      "SymInt size_n, bool transpose) -> Tensor");

  // Fused-MoE decode GEMM over per-expert Swordfish ABI v1 weights.
  ops.def(
      "swordfish_moe_mm(Tensor a, Tensor b_packed, Tensor group_scales, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, Tensor? topk_weights, "
      "int moe_block_size, int top_k, bool mul_topk_weights, int num_bits, "
      "int group_size, SymInt size_k, SymInt size_n) -> Tensor");

  // w4a16 prefill GEMM over the Swordfish ABI v1 packed weight.
  ops.def(
      "swordfish_prefill_mm(Tensor a, Tensor b_packed, Tensor group_scales, "
      "Tensor? group_zps, int num_bits, int group_size, SymInt size_k, "
      "SymInt size_n) -> Tensor");
}
