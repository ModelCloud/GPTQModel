// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#define USE_CUDA 1

#include <cuda.h>
#include <map>
#include <vector>

#include "./elf.h"
#include "./tensor.h"
#include "./torch_api.h"
#include "./utils.h"

static std::unordered_map<int64_t, KernelData> g_kernel_data;

inline int64_t find_kernel_configs_target_index(IntArrayRef &configs, int64_t shape_m) {
  size_t n = configs.size();
  if (n <= 2) return 0;
  if (n > 0 && n % 4 == 0) {
    for (size_t i = 0; i < n; i += 4) {
      int64_t min_val = configs[i];
      int64_t max_val = configs[i + 1];
      max_val = max_val > 0 ? max_val : (1 << 30);
      if (shape_m > min_val && shape_m <= max_val) return i + 2;
    }

    ASSERT_CHECK(false, "shape_m is not within any range defined in configs.");
  }

  ASSERT_CHECK(false, "configs length must be 1-2 or a non-zero multiple of 4.");
};

inline KernelLaunchData find_kernel_launch_data(IntArrayRef &configs, int64_t shape_m) {
  auto n = configs.size();
  int64_t index = find_kernel_configs_target_index(configs, shape_m);
  int64_t kernel_id = configs[index];
  int64_t num_sms = n < 2 ? 0 : configs[index + 1];
  ASSERT_CHECK(g_kernel_data.find(kernel_id) != g_kernel_data.end(), "kernel not existed.");
  KernelData &kernel_data = g_kernel_data[kernel_id];
  KernelLaunchData kernel_launch_data = {kernel_data, num_sms};
  return kernel_launch_data;
};

inline CUstream get_current_cuda_stream(int64_t dev) {
#if USE_TORCH_STABLE_API
  void *stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(dev, &stream_ptr);
  return static_cast<CUstream>(stream_ptr);
#else
  return at::cuda::getCurrentCUDAStream(dev);
#endif
}

inline int64_t get_num_sms(int64_t num_sms, int64_t dev) {
  if (num_sms > 0) return num_sms;
  CUdevice device;
  int32_t dev_sms;
  check_curesult(cuDeviceGet(&device, dev), "cuDeviceGet");
  check_curesult(
      cuDeviceGetAttribute(&dev_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
      "cuDeviceGetAttribute");
  return static_cast<int64_t>(dev_sms);
}

Tensor launch_kernel_impl(
    IntArrayRef configs,
    Tensor a,
    Tensor b,
    Tensor bs,
    std::optional<Tensor> bs2_,
    std::optional<Tensor> as_,
    std::optional<Tensor> bzp_,
    std::optional<Tensor> bias_,
    std::optional<Tensor> c_,
    std::optional<Tensor> sorted_ids_,
    std::optional<Tensor> expert_ids_,
    std::optional<Tensor> num_tokens_padded_,
    std::optional<Tensor> expert_layout_,
    std::optional<Tensor> locks_,
    int64_t top_k,
    int64_t valid_shape_m,
    bool should_check_tensor = true) {

  KernelLaunchData base_kernel_launch_data = find_kernel_launch_data(configs, 1);
  KernelData &base_kernel_data = base_kernel_launch_data.kernel_data;

  int64_t dev = a.get_device();
  int64_t shape_m = a.size(0);
  if (valid_shape_m <= 0) {
    valid_shape_m = shape_m * (base_kernel_data.gemm_type_id == 1 ? top_k : 1);
  }
  KernelLaunchData kernel_launch_data = find_kernel_launch_data(configs, valid_shape_m);
  KernelData &kernel_data = kernel_launch_data.kernel_data;
  int64_t &num_sms = kernel_launch_data.num_sms;
  Tensor c = may_make_tensor_c(c_, a, kernel_data, top_k);
  uint32_t num_ctas = kernel_data.num_ctas_per_sm * get_num_sms(num_sms, dev);
  Tensor tensor_map_buffer = make_tensor_map_buffer(a, kernel_data, num_ctas);
  a = torch_contiguous(a);

  if (should_check_tensor) {
    check_tensor_a(a, kernel_data, dev);
    check_tensor_b(b, kernel_data, dev);
    check_tensor_c(c, kernel_data, dev, shape_m, top_k);
    check_tensor_as(as_, kernel_data, dev, shape_m, top_k);
    check_tensor_bs(bs, kernel_data, dev);
    check_tensor_bzp(bzp_, kernel_data, dev);
    check_tensor_bias(bias_, kernel_data, dev);
    check_tensor_bs2(bs2_, kernel_data, dev);
    check_tensor_locks(locks_, kernel_data, dev);
    check_tensor_moe(sorted_ids_, expert_ids_, num_tokens_padded_, expert_layout_, kernel_data, dev);
  }

  void *a_ptr = a.data_ptr();
  void *b_ptr = b.data_ptr();
  void *c_ptr = c.data_ptr();
  void *as_ptr = as_.has_value() ? as_->data_ptr() : nullptr;
  void *bs_ptr = bs.data_ptr();
  void *bzp_ptr = bzp_.has_value() ? bzp_->data_ptr() : nullptr;
  void *bias_ptr = bias_.has_value() ? bias_->data_ptr() : nullptr;
  void *bs2_ptr = bs2_.has_value() ? bs2_->data_ptr() : nullptr;
  void *sorted_ids_ptr = sorted_ids_.has_value() ? sorted_ids_->data_ptr() : nullptr;
  void *expert_ids_ptr = expert_ids_.has_value() ? expert_ids_->data_ptr() : nullptr;
  void *num_tokens_padded_ptr = num_tokens_padded_.has_value() ? num_tokens_padded_->data_ptr() : nullptr;
  void *expert_layout_ptr = expert_layout_.has_value() ? expert_layout_->data_ptr() : nullptr;
  void *locks_ptr = locks_.has_value() ? locks_->data_ptr() : nullptr;
  void *tensor_map_buffer_ptr = tensor_map_buffer.data_ptr();

  auto tensor_map_a = make_tma_desc_a(a, kernel_data);
  auto tensor_map_as = make_tma_desc_as(as_, kernel_data);
  auto tensor_map_b = make_tma_desc_b(b, kernel_data);
  auto tensor_map_c = make_tma_desc_c(c, kernel_data);
  auto tensor_map_bs = make_tma_desc_bs(bs, kernel_data);
  auto tensor_map_bzp = make_tma_desc_bzp(bzp_, kernel_data);
  auto tensor_map_bias = make_tma_desc_bias(bias_, kernel_data);
  auto tensor_map_bs2 = make_tma_desc_bs2(bs2_, kernel_data);
  auto to_void_ptr = [&](void *ptr) { return ptr; };
  bool use_int64_expert_layout = false;
  if (expert_layout_.has_value()) {
    use_int64_expert_layout = expert_layout_.value().scalar_type() == ScalarType::Long;
  }

  uint32_t shape_m_u32 = static_cast<uint32_t>(shape_m);
  uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  ASSERT_CHECK(static_cast<int64_t>(shape_m_u32) == shape_m, "shape_m overflows uint32_t: ", shape_m);
  ASSERT_CHECK(static_cast<int64_t>(top_k_u32) == top_k, "top_k overflows uint32_t: ", top_k);

  void *kernel_args[] = {
      kernel_data.use_tma_a ? to_void_ptr(&tensor_map_a) : to_void_ptr(&a_ptr),
      kernel_data.use_tma_b ? to_void_ptr(&tensor_map_b) : to_void_ptr(&b_ptr),
      kernel_data.use_tma_c ? to_void_ptr(&tensor_map_c) : to_void_ptr(&c_ptr),
      kernel_data.use_tma_as ? to_void_ptr(&tensor_map_as) : to_void_ptr(&as_ptr),
      kernel_data.use_tma_bs ? to_void_ptr(&tensor_map_bs) : to_void_ptr(&bs_ptr),
      kernel_data.use_tma_bzp ? to_void_ptr(&tensor_map_bzp) : to_void_ptr(&bzp_ptr),
      kernel_data.use_tma_bias ? to_void_ptr(&tensor_map_bias) : to_void_ptr(&bias_ptr),
      kernel_data.use_tma_bs2 ? to_void_ptr(&tensor_map_bs2) : to_void_ptr(&bs2_ptr),
      &sorted_ids_ptr,
      &expert_ids_ptr,
      &num_tokens_padded_ptr,
      &expert_layout_ptr,
      &tensor_map_buffer_ptr,
      &locks_ptr,
      &shape_m_u32,
      &top_k_u32,
      &use_int64_expert_layout};

  CUlaunchConfig config = {};
  config.gridDimX = num_ctas;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = kernel_data.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  CUlaunchAttribute attrs[1];
  if (kernel_data.multi_cast_size_a * kernel_data.multi_cast_size_b > 1) {
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attrs[0].value.clusterDim.x = kernel_data.multi_cast_size_a * kernel_data.multi_cast_size_b;
    attrs[0].value.clusterDim.y = 1;
    attrs[0].value.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
  }

  config.sharedMemBytes = kernel_data.smem_size;
  config.hStream = get_current_cuda_stream(dev);

  CUfunction &func = kernel_data.func;
  constexpr auto SMEM_SIZE_ATTR = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
  check_curesult(cuFuncSetAttribute(func, SMEM_SIZE_ATTR, kernel_data.smem_size), "cuFuncSetAttribute");
  check_curesult(cuLaunchKernelEx(&config, func, kernel_args, nullptr), "cuLaunchKernelEx");

  return c;
};

std::tuple<int64_t, std::string> register_kernel(const std::string &cubin_path) {
  static std::unordered_map<std::string, std::tuple<int64_t, std::string>> g_path_ids;
  auto it = g_path_ids.find(cubin_path);
  if (it != g_path_ids.end()) return it->second;

  CUlibrary library;
  check_curesult(
      cuLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0),
      "cuLibraryLoadFromFile");

  unsigned int num_kernels = 0;
  check_curesult(cuLibraryGetKernelCount(&num_kernels, library), "cuLibraryGetKernelCount");
  std::vector<CUkernel> kernels(num_kernels);
  check_curesult(cuLibraryEnumerateKernels(kernels.data(), num_kernels, library), "cuLibraryEnumerateKernels");

  CUkernel kernel = nullptr;
  std::string kernel_name;
  for (CUkernel candidate : kernels) {
    const char *name = nullptr;
    check_curesult(cuKernelGetName(&name, candidate), "cuKernelGetName");
    if (std::string(name).find("humming") == std::string::npos) continue;
    ASSERT_CHECK(kernel == nullptr, "multiple humming kernels found in ", cubin_path);
    kernel = candidate;
    kernel_name = name;
  }
  ASSERT_CHECK(kernel != nullptr, "no humming kernel found in ", cubin_path);

  CUfunction func;
  check_curesult(cuKernelGetFunction(&func, kernel), "cuKernelGetFunction");

  int64_t hash_id = manual_crc32(cubin_path);
  hash_id = (hash_id << 30) + manual_crc32(kernel_name);

  if (g_kernel_data.find(hash_id) == g_kernel_data.end()) {
    auto reader = CubinReader(cubin_path);

    g_kernel_data[hash_id] = {
        library,
        func,

        reader.getUint32("SMEM_SIZE"),
        reader.getUint32("NUM_THREADS"),
        reader.getUint32("A_DTYPE_ID"),
        reader.getUint32("B_DTYPE_ID"),
        reader.getUint32("C_DTYPE_ID"),
        reader.getUint32("BS_DTYPE_ID"),
        reader.getUint32("PROBLEM_SHAPE_N"),
        reader.getUint32("PROBLEM_SHAPE_K"),
        reader.getUint32("BLOCK_SHAPE_M"),
        reader.getUint32("BLOCK_SHAPE_N"),
        reader.getUint32("BLOCK_SHAPE_K"),
        reader.getUint32("PAD_SHAPE_N"),
        reader.getUint32("PAD_SHAPE_K"),
        reader.getUint32("NUM_EXPERTS"),
        reader.getUint32("INPUT_SCALE_GROUP_SIZE"),
        reader.getUint32("WEIGHT_SCALE_GROUP_SIZE"),
        reader.getUint32("WEIGHT_SCALE_GROUP_SIZE_N"),
        reader.getUint32("NUM_CTAS_PER_SM"),
        reader.getUint32("MULTI_CAST_SIZE_A"),
        reader.getUint32("MULTI_CAST_SIZE_B"),
        reader.getUint32("GEMM_TYPE_ID"),
        reader.getUint32("MMA_TYPE_ID"),

        reader.getBool("USE_STREAM_K"),
        reader.getBool("IS_FP_ZERO_POINT"),
        reader.getBool("IS_CHANNEL_WEIGHT_SCALE"),
        reader.getBool("IS_GROUP_WEIGHT_SCALE"),
        reader.getBool("IS_BLOCK_WEIGHT_SCALE"),
        reader.getBool("IS_TENSOR_WEIGHT_SCALE"),
        reader.getBool("IS_CHANNEL_WEIGHT_SCALE_2"),
        reader.getBool("IS_TENSOR_WEIGHT_SCALE_2"),
        reader.getBool("HAS_ZERO_POINT"),
        reader.getBool("HAS_BIAS"),
        reader.getBool("USE_M_MAJOR_INPUT_SCALE"),
        reader.getBool("USE_TMA_A"),
        reader.getBool("USE_TMA_AS"),
        reader.getBool("USE_TMA_B"),
        reader.getBool("USE_TMA_C"),
        reader.getBool("USE_TMA_BS"),
        reader.getBool("USE_TMA_BS2"),
        reader.getBool("USE_TMA_BZP"),
        reader.getBool("USE_TMA_BIAS"),
        reader.getBool("USE_PACKED_K_LAYOUT")};
  };

  auto result = std::make_tuple(hash_id, kernel_name);
  g_path_ids[cubin_path] = result;
  return result;
}

Tensor launch_kernel(
    Tensor configs_t,
    Tensor a,
    Tensor b,
    Tensor bs,
    std::optional<Tensor> bs2_,
    std::optional<Tensor> as_,
    std::optional<Tensor> bzp_,
    std::optional<Tensor> bias_,
    std::optional<Tensor> c_,
    std::optional<Tensor> sorted_ids_,
    std::optional<Tensor> expert_ids_,
    std::optional<Tensor> num_tokens_padded_,
    std::optional<Tensor> expert_layout_,
    std::optional<Tensor> locks_,
    int64_t top_k,
    int64_t valid_shape_m,
    bool should_check_tensor = true) {
  ASSERT_CHECK(configs_t.scalar_type() == ScalarType::Long, "configs must be int64 tensor.");
  ASSERT_CHECK(configs_t.is_contiguous(), "configs must be contiguous.");
  ASSERT_CHECK(configs_t.get_device() < 0, "configs must be a CPU tensor.");
  IntArrayRef configs(static_cast<int64_t *>(configs_t.data_ptr()),
                      static_cast<size_t>(configs_t.numel()));
  return launch_kernel_impl(configs, a, b, bs, bs2_, as_, bzp_, bias_, c_, sorted_ids_, expert_ids_,
                            num_tokens_padded_, expert_layout_, locks_, top_k, valid_shape_m, should_check_tensor);
}

COMMON_TORCH_LIBRARY(humming, m) {
  m.def(
      "launch_kernel(Tensor configs, Tensor a, Tensor b, Tensor bs, "
      "Tensor? bs2, Tensor? as_, Tensor? bzp, Tensor? bias, Tensor? c, "
      "Tensor? sorted_ids, Tensor? expert_ids, Tensor? num_tokens_padded, Tensor? expert_layout, "
      "Tensor? locks, SymInt top_k, SymInt valid_shape_m, bool should_check_tensor = True) -> Tensor");
  m.def("register_kernel(str cubin_path) -> (int, str)");
};

COMMON_TORCH_LIBRARY_IMPL(humming, CUDA, m) {
  m.impl("launch_kernel", COMMON_TORCH_BOX(&launch_kernel));
};

COMMON_TORCH_LIBRARY_IMPL(humming, Undefined, m) {
  m.impl("register_kernel", COMMON_TORCH_BOX(&register_kernel));
};
