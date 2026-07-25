// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#ifndef USE_TORCH_STABLE_API
#define USE_TORCH_STABLE_API 0
#endif

#if USE_TORCH_STABLE_API

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_inl.h>

using Tensor = torch::stable::Tensor;
using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
using ScalarType = torch::headeronly::ScalarType;
using Device = torch::stable::Device;

#define ASSERT_CHECK STD_TORCH_CHECK
#define COMMON_TORCH_LIBRARY STABLE_TORCH_LIBRARY
#define COMMON_TORCH_LIBRARY_IMPL STABLE_TORCH_LIBRARY_IMPL
#define COMMON_TORCH_BOX TORCH_BOX
#define DTYPE_TO_STRING torch::headeronly::toString

inline Tensor torch_empty(IntArrayRef shape, ScalarType dtype, Device device) {
  return torch::stable::empty(shape, dtype, std::nullopt, device, std::nullopt, std::nullopt);
};
inline Tensor torch_view_shape(const Tensor tensor, IntArrayRef shape) {
  return torch::stable::view(tensor, shape);
};
inline Tensor torch_contiguous(const Tensor tensor) {
  return torch::stable::contiguous(tensor);
};

#else

#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

using Tensor = at::Tensor;
using IntArrayRef = at::IntArrayRef;
using ScalarType = at::ScalarType;
using Device = at::Device;

#define ASSERT_CHECK TORCH_CHECK
#define COMMON_TORCH_LIBRARY TORCH_LIBRARY
#define COMMON_TORCH_LIBRARY_IMPL TORCH_LIBRARY_IMPL
#define COMMON_TORCH_BOX
#define DTYPE_TO_STRING at::toString

inline Tensor torch_empty(IntArrayRef shape, ScalarType dtype, Device device) {
  return at::empty(shape, dtype, std::nullopt, device, std::nullopt, std::nullopt);
};
inline Tensor torch_view_shape(const Tensor tensor, IntArrayRef shape) {
  return tensor.view(shape);
};
inline Tensor torch_contiguous(const Tensor tensor) {
  return tensor.contiguous();
};

#endif
