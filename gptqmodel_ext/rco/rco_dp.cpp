// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
//
// Stable-ABI CPU kernel for RCO's budget-constrained dynamic program.
// Previously this was a `load_inline` pybind module, which tied the artifact
// to the torch and Python ABIs; it is now a plain torch.ops kernel built
// against torch_stable_compat.h.

#include "../torch_stable_compat.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

torch::Tensor rco_dp(torch::Tensor gained, torch::Tensor steps, int64_t budget) {
  TORCH_CHECK(gained.device().is_cpu() && gained.scalar_type() == torch::kFloat64,
              "gained must be float64 CPU");
  TORCH_CHECK(steps.device().is_cpu() && steps.scalar_type() == torch::kInt64,
              "steps must be int64 CPU");
  TORCH_CHECK(gained.dim() == 2 && steps.sizes() == gained.sizes(),
              "gained and steps must be matching [layers, options]");

  const int64_t n_layers = gained.size(0);
  const int64_t n_options = gained.size(1);
  const int64_t width = budget + 1;
  const double* g = gained.const_data_ptr<double>();
  const int64_t* s = steps.const_data_ptr<int64_t>();
  const double neg_inf = -std::numeric_limits<double>::infinity();

  std::vector<double> table(width, neg_inf);
  std::vector<double> updated(width, neg_inf);
  std::vector<int64_t> choice(n_layers * width, 0);
  table[0] = 0.0;

  for (int64_t layer = 0; layer < n_layers; ++layer) {
    std::fill(updated.begin(), updated.end(), neg_inf);
    int64_t* row = choice.data() + layer * width;
    for (int64_t option = 0; option < n_options; ++option) {
      const int64_t step = s[layer * n_options + option];
      if (step > budget) {
        continue;
      }
      const double gain = g[layer * n_options + option];
      for (int64_t j = step; j < width; ++j) {
        const double best = table[j - step] + gain;
        if (best > updated[j]) {
          updated[j] = best;
          row[j] = option;
        }
      }
    }
    table.swap(updated);
  }

  // Pick the best objective over every feasible budget, matching argmax's
  // first-maximum tie-break.
  double best = neg_inf;
  int64_t position = -1;
  for (int64_t j = 0; j < width; ++j) {
    if (table[j] > best) {
      best = table[j];
      position = j;
    }
  }
  TORCH_CHECK(position >= 0 && std::isfinite(best),
              "RCO rounding found no feasible assignment under the budget");

  auto out = torch::zeros(
      {n_layers}, torch::TensorOptions().dtype(torch::kInt64));
  int64_t* o = out.mutable_data_ptr<int64_t>();
  for (int64_t layer = n_layers - 1; layer >= 0; --layer) {
    const int64_t option = choice[layer * width + position];
    o[layer] = option;
    position -= s[layer * n_options + option];
  }
  return out;
}

}  // namespace

STABLE_TORCH_LIBRARY(gptqmodel_rco, m) {
  m.def("rco_dp(Tensor gained, Tensor steps, int budget) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(gptqmodel_rco, CPU, m) {
  m.impl("rco_dp", TORCH_BOX(&rco_dp));
}
