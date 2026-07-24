// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include "./tma.h"
#include "./torch_api.h"
#include "./utils.h"
#include <cuda.h>

inline Tensor may_make_tensor_c(std::optional<Tensor> &c, const Tensor &a, KernelData &kernel_data, int64_t top_k) {
  if (c.has_value()) return c.value();

  int64_t shape_m = a.size(0);
  int64_t shape_n = kernel_data.problem_shape_n - kernel_data.pad_shape_n;
  if (kernel_data.gemm_type_id == 1) shape_m = shape_m * top_k;

  auto c_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  return torch_empty({shape_m, shape_n}, c_dtype, a.device());
}

inline Tensor make_tensor_map_buffer(const Tensor &a, KernelData &kernel_data, uint32_t num_ctas) {
  int64_t size = 0;

  if (kernel_data.use_tma_c && (kernel_data.gemm_type_id == 2 || kernel_data.gemm_type_id == 3)) {
    size = 32; // 32 int32 = 128 bytes
  }

  return torch_empty({size * num_ctas}, ScalarType::Int, a.device());
}

inline void check_tensor_common(
    const Tensor &tensor, std::string name,
    int64_t expected_dev,
    ScalarType expected_dtype,
    std::optional<std::vector<int64_t>> expected_shape_ = std::nullopt) {

  ASSERT_CHECK(tensor.is_contiguous(), "error: ", name, ".is_contiguous() != true");
  ASSERT_CHECK(tensor.is_cuda(), "error: ", name, ".is_cuda() != true");
  ASSERT_CHECK(tensor.get_device() == expected_dev, "error: ", name, ".get_device() != a.get_device()");

  if (expected_shape_.has_value()) {
    auto &expected_shape = expected_shape_.value();
    if (expected_shape.size() == 1 && expected_shape[0] == 1) {
      int64_t actual = tensor.dim();
      int64_t expected = 1;
      ASSERT_CHECK(tensor.dim() == 0 || tensor.dim() == 1, name, ".dim() != expected_shape.size() => ",
                   tensor.dim(), " not in [0, 1]");
      if (tensor.dim() == 1) {
        ASSERT_CHECK(tensor.size(0) == 1, name, ".size(0) != expected_shape[0] => ",
                     tensor.size(0), " != 1");
      }
    } else {
      ASSERT_CHECK(tensor.dim() == expected_shape.size(), name, ".dim() != expected_shape.size() => ",
                   tensor.dim(), " != ", expected_shape.size());

      for (int64_t i = 0; i < tensor.dim(); i++) {
        ASSERT_CHECK(tensor.size(i) == expected_shape[i], name, ".size(", i, ") != expected_shape[", i, "] => ",
                     tensor.size(i), " != ", expected_shape[i]);
      }
    }
  }

  ASSERT_CHECK(
      tensor.scalar_type() == expected_dtype, name, ".dtype != expected_dtype => ",
      DTYPE_TO_STRING(tensor.scalar_type()), " != ", DTYPE_TO_STRING(expected_dtype));
}

inline void check_tensor_a(const Tensor &tensor, KernelData &kernel_data, int64_t dev) {
  std::vector<int64_t> expected_shape = {tensor.size(0)};

  int64_t shape_k = kernel_data.problem_shape_k - kernel_data.pad_shape_k;
  if (get_dtype_num_bits(kernel_data.a_dtype_id) == 4) {
    expected_shape.push_back(shape_k / 2);
  } else {
    expected_shape.push_back(shape_k);
  }
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.a_dtype_id);
  check_tensor_common(tensor, "a", dev, expected_dtype, expected_shape);
};

inline void check_tensor_b(Tensor &tensor, KernelData &kernel_data, int64_t dev) {
  uint32_t problem_shape_n = kernel_data.problem_shape_n;
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id);
  uint32_t pack_size_k = 256 / get_dtype_num_bits(kernel_data.a_dtype_id);
  if (kernel_data.use_packed_k_layout) pack_size_k = 64;

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  expected_shape.push_back(problem_shape_k / pack_size_k);
  expected_shape.push_back(problem_shape_n * pack_size_k * num_bits / 32);
  check_tensor_common(tensor, "b", dev, ScalarType::Int, expected_shape);
};

inline void check_tensor_c(Tensor &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m, int64_t top_k) {
  std::vector<int64_t> expected_shape = {shape_m * (kernel_data.gemm_type_id == 1 ? top_k : 1)};
  expected_shape.push_back(kernel_data.problem_shape_n - kernel_data.pad_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  check_tensor_common(tensor, "c", dev, expected_dtype, expected_shape);
};

inline void check_tensor_as(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m, int64_t top_k) {
  if (get_dtype_num_bits(kernel_data.a_dtype_id) == 16) return;
  ASSERT_CHECK(tensor.has_value(), "as must not be none for 4b or 8b activation");

  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.input_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);

  if (kernel_data.mma_type_id == 3) {
    std::vector<int64_t> expected_shape;
    if (kernel_data.use_tma_as || kernel_data.use_m_major_input_scale) {
      int64_t m_pad = (shape_m + 3) / 4 * 4;
      expected_shape = {(int64_t)CEIL_DIV(num_groups, 4), m_pad};
    } else {
      expected_shape = {shape_m, (int64_t)CEIL_DIV(num_groups, 4)};
    }
    check_tensor_common(tensor.value(), "as", dev, ScalarType::Int, expected_shape);
  } else {
    std::vector<int64_t> expected_shape;
    if (kernel_data.use_m_major_input_scale && group_size > 0) {
      int64_t m_pad = (shape_m + 3) / 4 * 4;
      expected_shape = {(int64_t)num_groups, m_pad};
    } else {
      expected_shape = {shape_m, (int64_t)num_groups};
    }
    check_tensor_common(tensor.value(), "as", dev, ScalarType::Float, expected_shape);
  }
};

inline void check_tensor_bs(Tensor &tensor, KernelData &kernel_data, int64_t dev) {
  if (kernel_data.is_tensor_weight_scale) {
    std::vector<int64_t> expected_shape = {kernel_data.gemm_type_id != 0 ? kernel_data.num_experts : 1};
    check_tensor_common(tensor, "bs", dev, ScalarType::Float, expected_shape);
    return;
  }
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t problem_shape_n = kernel_data.problem_shape_n;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t group_size_n = kernel_data.weight_scale_group_size_n;

  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);
  uint32_t num_groups_n = group_size_n == 0 ? 1 : CEIL_DIV(problem_shape_n, group_size_n);

  std::vector<int64_t> expected_shape = {};
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.bs_dtype_id);
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  if (kernel_data.mma_type_id == 3) {
    expected_dtype = ScalarType::Int;
    uint32_t num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);
    uint32_t scale_vec = 256 / num_bits / group_size;
    expected_shape.push_back(num_groups / (scale_vec == 1 ? 2 : 4));
    expected_shape.push_back(kernel_data.problem_shape_n / (scale_vec == 1 ? 2 : 1));
  } else {
    expected_shape.push_back(num_groups);
    if (kernel_data.is_block_weight_scale) {
      expected_shape.push_back(num_groups_n);
    } else {
      expected_shape.push_back(kernel_data.problem_shape_n);
    }
  }
  check_tensor_common(tensor, "bs", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bzp(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (!kernel_data.has_zero_point) return;
  ASSERT_CHECK(tensor.has_value(), "bzp must not be none if has_zero_point");

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id) <= 4 ? 4 : 8;
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  expected_shape.push_back(num_groups);
  ScalarType expected_dtype;
  if (kernel_data.is_fp_zero_point) {
    expected_shape.push_back(kernel_data.problem_shape_n);
    expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  } else {
    expected_shape.push_back(kernel_data.problem_shape_n * num_bits / 32);
    expected_dtype = ScalarType::Int;
  }
  check_tensor_common(tensor.value(), "bzp", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bias(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (!kernel_data.has_bias) return;
  ASSERT_CHECK(tensor.has_value(), "bias must not be none if has_bias");
  std::vector<int64_t> expected_shape = {};
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  expected_shape.push_back(kernel_data.problem_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  check_tensor_common(tensor.value(), "bias", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bs2(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (kernel_data.is_channel_weight_scale_2) {
    ASSERT_CHECK(tensor.has_value(), "bs2 must not be none if is_channel_weight_scale_2");
    std::vector<int64_t> expected_shape = {};
    if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
    expected_shape.push_back(kernel_data.problem_shape_n);
    auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
    check_tensor_common(tensor.value(), "bs2", dev, expected_dtype, expected_shape);
    return;
  } else if (kernel_data.is_tensor_weight_scale_2) {
    ASSERT_CHECK(tensor.has_value(), "bs2 must not be none if is_tensor_weight_scale_2");
    std::vector<int64_t> expected_shape = {kernel_data.gemm_type_id != 0 ? kernel_data.num_experts : 1};
    check_tensor_common(tensor.value(), "bs2", dev, ScalarType::Float, expected_shape);
  }
};

inline void check_tensor_locks(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (!kernel_data.use_stream_k) return;
  ASSERT_CHECK(tensor.has_value(), "locks must not be none if use_stream_k");
  check_tensor_common(tensor.value(), "locks", dev, ScalarType::Int);
};

inline void check_tensor_moe(
    std::optional<Tensor> &sorted_ids,
    std::optional<Tensor> &expert_ids,
    std::optional<Tensor> &num_tokens_padded,
    std::optional<Tensor> &expert_layout,
    KernelData &kernel_data,
    int64_t dev) {

  if (kernel_data.gemm_type_id == 1) {
    ASSERT_CHECK(sorted_ids.has_value(), "sorted_ids must not be none for indexed gemm");
    ASSERT_CHECK(expert_ids.has_value(), "expert_ids must not be none for indexed gemm");
    ASSERT_CHECK(num_tokens_padded.has_value(), "num_tokens_padded must not be none for indexed gemm");
    check_tensor_common(sorted_ids.value(), "sorted_ids", dev, ScalarType::Int);
    check_tensor_common(expert_ids.value(), "expert_ids", dev, ScalarType::Int);
    check_tensor_common(num_tokens_padded.value(), "num_tokens_padded", dev, ScalarType::Int);
  }
  if (kernel_data.gemm_type_id == 2) {
    ASSERT_CHECK(expert_layout.has_value(), "expert_layout must not be none for grouped gemm");
    ASSERT_CHECK(expert_layout.value().scalar_type() == ScalarType::Int || expert_layout.value().scalar_type() == ScalarType::Long,
                 "expert_layout.dtype must be int32 or int64, got ", DTYPE_TO_STRING(expert_layout.value().scalar_type()));
    std::vector<int64_t> expected_shape = {kernel_data.num_experts + 1};
    check_tensor_common(expert_layout.value(), "expert_token_offset", dev, expert_layout.value().scalar_type(), expected_shape);
  }
  if (kernel_data.gemm_type_id == 3) {
    ASSERT_CHECK(expert_layout.has_value(), "expert_layout must not be none for grouped gemm");
    ASSERT_CHECK(expert_layout.value().scalar_type() == ScalarType::Int || expert_layout.value().scalar_type() == ScalarType::Long,
                 "expert_layout.dtype must be int32 or int64, got ", DTYPE_TO_STRING(expert_layout.value().scalar_type()));
    std::vector<int64_t> expected_shape = {kernel_data.num_experts};
    check_tensor_common(expert_layout.value(), "expert_num_tokens", dev, expert_layout.value().scalar_type(), expected_shape);
  }
};

inline CUtensorMap make_tma_desc_a(Tensor tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_a) return CUtensorMap();

  uint32_t tma_block_shape_m = kernel_data.block_shape_m;
  uint32_t tma_block_shape_k = kernel_data.block_shape_k;
  uint32_t swizzle_bytes = 128;
  uint32_t a_dtype_num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);

  if (kernel_data.block_shape_k * a_dtype_num_bits == 512) {
    swizzle_bytes = 64;
  } else {
    tma_block_shape_k = 1024 / a_dtype_num_bits;
  }

  uint32_t tma_block_shape_k_packed =
      a_dtype_num_bits < 8 ? tma_block_shape_k * a_dtype_num_bits / 8 : tma_block_shape_k;

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  return make_tma_desc(tensor, {tma_block_shape_k_packed, tma_block_shape_m}, swizzle_bytes);
}

inline CUtensorMap make_tma_desc_as(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_as) return CUtensorMap();

  uint32_t block_shape_m = kernel_data.block_shape_m;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.input_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  auto tensor = tensor_.value();
  if (kernel_data.mma_type_id == 3) {
    tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
    return make_tma_desc(tensor, {block_shape_m, CEIL_DIV(num_groups, 4)});
  }
  if (group_size == 0) {
    tensor = torch_view_shape(tensor, {1, -1});
  } else {
    tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  }

  return make_tma_desc(tensor, {block_shape_m, num_groups});
}

inline CUtensorMap make_tma_desc_b(Tensor &tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_b) return CUtensorMap();

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id);
  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;

  uint32_t pack_size_k = 256 / get_dtype_num_bits(kernel_data.a_dtype_id);
  if (kernel_data.use_packed_k_layout) pack_size_k = 64;

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  tensor = torch_view_shape(tensor, {tensor.size(0), -1, pack_size_k});

  return make_tma_desc(tensor, {pack_size_k, block_shape_n * num_bits / 32, block_shape_k / pack_size_k});
}

inline CUtensorMap make_tma_desc_c(Tensor tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_c) return CUtensorMap();
  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  return make_tma_desc(tensor, {64, kernel_data.block_shape_m}, 128);
}

inline CUtensorMap make_tma_desc_bs(Tensor tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_bs) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});

  if (kernel_data.mma_type_id == 3) {
    uint32_t num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);
    uint32_t scale_vec = 256 / num_bits / group_size;
    return make_tma_desc(tensor, {block_shape_n / (scale_vec == 1 ? 2 : 1), num_groups / (scale_vec == 1 ? 2 : 4)});
  }

  tensor = torch_view_shape(tensor, {tensor.size(0), -1, 16});

  return make_tma_desc(tensor, {16, block_shape_n / 16, num_groups});
}

inline CUtensorMap make_tma_desc_bzp(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bzp) return CUtensorMap();

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id) <= 4 ? 4 : 8;
  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  auto tensor = tensor_.value();

  return make_tma_desc(tensor, {block_shape_n * num_bits / 32, num_groups});
}

inline CUtensorMap make_tma_desc_bias(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bias) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;

  auto tensor = tensor_.value();
  tensor = torch_view_shape(tensor, {-1, 64});

  return make_tma_desc(tensor, {64, block_shape_n / 64});
}

inline CUtensorMap make_tma_desc_bs2(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bs2) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;

  auto tensor = tensor_.value();
  tensor = torch_view_shape(tensor, {-1, 64});

  return make_tma_desc(tensor, {64, block_shape_n / 64});
}
