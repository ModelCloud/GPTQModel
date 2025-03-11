// Adapted from https://github.com/HandH1998/QQQ

/* 
 * Adapted from https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda.cpp
 * Modified by HandH1998
 * Copyright (C) 2024 HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <torch/extension.h>

void qqq_gemm(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
        torch::Tensor& D,
  const torch::Tensor& s1,
  const torch::Tensor& s2,
  const torch::Tensor& s3,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
);
