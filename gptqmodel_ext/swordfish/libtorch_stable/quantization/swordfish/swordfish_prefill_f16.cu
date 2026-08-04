// SPDX-FileCopyrightText: 2026 AlpinDale and the dphnAI/sonar contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Swordfish: Blackwell (sm100/sm110) weight-quantized GEMM kernels.
// Vendored from https://github.com/dphnAI/sonar and used under the terms of
// the GNU Affero General Public License v3.0 or later. See LICENSE in this
// directory or /licenses/SWORDFISH at the project root for the full license.
//
// fp16-activation instantiations of the Swordfish prefill configurations.

#include "swordfish_prefill_impl.cuh"

namespace swordfish {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
namespace prefill {
template void run_prefill_all<cutlass::half_t>(torch::stable::Tensor&,
                                               torch::stable::Tensor&,
                                               torch::stable::Tensor&,
                                               const void*, bool, bool, int,
                                               torch::stable::Tensor&, int, int,
                                               int, cudaStream_t);
}  // namespace prefill
#endif
}  // namespace swordfish
