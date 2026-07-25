// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <humming/mma/mxmma.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>


template <MmaType kMmaType, class Ctx, class ArithClass>
struct MmaSelector;

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MMA, Ctx, ArithClass> {
  using Type = WMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::WGMMA, Ctx, ArithClass> {
  using Type = WGMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MXMMA, Ctx, ArithClass> {
  using Type = MXMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
using Mma = typename MmaSelector<Ctx::kMmaType, Ctx, ArithClass>::Type;
