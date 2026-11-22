// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor cptc_decode(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t);
at::Tensor cptc_mma(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t);
at::Tensor cptc_encode(const at::Tensor&, const at::Tensor&, int64_t);

TORCH_LIBRARY(gptqmodel_cptc, m) {
  m.def("decode(Tensor packed, Tensor table, Tensor coefficients, Tensor scales, "
        "int bits, int depth, int shift, int in_features, int tile_rows=1) -> Tensor");
  m.def("mma(Tensor input, Tensor packed, Tensor table, Tensor coefficients, Tensor scales, "
        "int bits, int depth, int shift, int tile_rows=1) -> Tensor");
  m.def("encode(Tensor blocks, Tensor decoded_states, int bits) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_cptc, CUDA, m) {
  m.impl("decode", cptc_decode);
  m.impl("mma", cptc_mma);
  m.impl("encode", cptc_encode);
}
