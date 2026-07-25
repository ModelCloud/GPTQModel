// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include "./torch_api.h"
#include <cuda.h>

inline CUtensorMapDataType get_tma_dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float: return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case ScalarType::Half: return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case ScalarType::BFloat16: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case ScalarType::Int: return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case ScalarType::Long: return CU_TENSOR_MAP_DATA_TYPE_INT64;
    case ScalarType::Short: return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    case ScalarType::Char: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case ScalarType::Byte: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case ScalarType::Float8_e8m0fnu: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case ScalarType::Float8_e5m2: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case ScalarType::Float8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case ScalarType::Float8_e4m3fnuz: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    default: ASSERT_CHECK(false, "Unsupported torch dtype for TMA");
  }
}

inline CUtensorMapSwizzle get_swizzle_enum(uint32_t swizzle_bytes) {
  if (swizzle_bytes == 0) return CU_TENSOR_MAP_SWIZZLE_NONE;
  if (swizzle_bytes == 32) return CU_TENSOR_MAP_SWIZZLE_32B;
  if (swizzle_bytes == 64) return CU_TENSOR_MAP_SWIZZLE_64B;
  if (swizzle_bytes == 128) return CU_TENSOR_MAP_SWIZZLE_128B;
  ASSERT_CHECK(false, "Unsupported swizzle bytes. Must be 0, 32, 64, or 128.");
}

inline CUtensorMap make_tma_desc(
    std::optional<Tensor> tensor_,
    std::vector<uint32_t> smem_dims,
    uint32_t swizzle_bytes = 0) {

  CUtensorMap tmap = {};
  if (!tensor_.has_value() || smem_dims.size() == 0) return tmap;
  Tensor tensor = tensor_.value();

  uint32_t ndim = tensor.dim();
  size_t elsize = tensor.element_size();

  std::vector<uint64_t> gmem_dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    gmem_dims[i] = static_cast<uint64_t>(tensor.size(ndim - 1 - i));
  }

  std::vector<uint64_t> gmem_strides(ndim - 1);
  for (int i = 0; i < ndim - 1; ++i) {
    gmem_strides[i] = static_cast<uint64_t>(tensor.stride(ndim - 2 - i) * elsize);
  }

  std::vector<uint32_t> element_strides(ndim, 1);

  CUresult res = cuTensorMapEncodeTiled(
      &tmap,
      get_tma_dtype(tensor.scalar_type()),
      ndim,
      tensor.data_ptr(),
      gmem_dims.data(),
      gmem_strides.data(),
      smem_dims.data(),
      element_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      get_swizzle_enum(swizzle_bytes),
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    const char *errStr;
    cuGetErrorString(res, &errStr);
    ASSERT_CHECK(false, "TMA Encode Failed: ", errStr);
  }

  return tmap;
}
