// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once


enum class WeightScaleType : uint32_t {
  GROUP,
  BLOCK,
  CHANNEL,
  TENSOR,
};


enum class WeightScale2Type : uint32_t {
  NONE,
  CHANNEL,
  TENSOR,
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA,
  MXMMA
};


enum class GemmType : uint32_t {
  DENSE,
  INDEXED,
  GROUPED_CONTIGUOUS,
  GROUPED_MASKED,
};
