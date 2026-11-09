// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// NF12 / NF12-AD kernel computation.
//
// Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
// Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
// Together AI: https://www.together.ai/
//
// Original implementation: https://github.com/dphnAI/nf12
// Copyright (c) 2026 AlpinDale and the nf12 contributors.
// Licensed under the Apache License, Version 2.0.
//
// Vendored into GPT-QModel (https://github.com/ModelCloud/GPTQModel) and
// modified for the NF12-AD (AlpinDale) method/format. See
// gptqmodel_ext/nf12/README.md for the list of local modifications.
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace nf12::torch_detail {

struct FusedDecodeWeight {
  const void* packed;
  const void* escapes;
  const int64_t* escape_offsets;
  const uint8_t* exp_base;
  int64_t experts, n, k;
  bool bf16t;
};

struct FusedDecodeArgs {
  FusedDecodeWeight w13, w2;
  const void* x;
  const int32_t* ids;
  const float* scales;
  void* activation;
  void* out;
  int64_t topk, ldx, ldids, ldscales, ldactivation, ldout;
};

// The caller guards device before preflight. Output pointers may be null during
// preflight; their intended strides must already be supplied. No device work,
// allocation or synchronization is performed. Capability preparation is cached.
bool fused_decode_supported(int device, const FusedDecodeArgs& args);

// Both outputs must be nonnull. Preparation completes before either launch;
// launch errors propagate, and callers must not fall back after this call.
cudaError_t fused_decode(const FusedDecodeArgs& args, cudaStream_t stream);

}  // namespace nf12::torch_detail
