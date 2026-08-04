// SPDX-FileCopyrightText: 2026 AlpinDale and the dphnAI/sonar contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Swordfish: Blackwell (sm100/sm110) weight-quantized GEMM kernels.
// Vendored from https://github.com/dphnAI/sonar and used under the terms of
// the GNU Affero General Public License v3.0 or later. See LICENSE in this
// directory or /licenses/SWORDFISH at the project root for the full license.
//
// Compatibility shim for CUTLASS 4.4 architecture-family macros.
// CUTLASS 4.4 gates Blackwell TMA/tcgen05 features behind architecture-family
// macros that are not always enabled for the sm_100/sm_103 targets we build.
// This header is force-included by the Swordfish JIT build and enables the
// family macros only in the device-code pass (__CUDA_ARCH__ is not defined
// during the host pass, so host-side paths keep their stubs).

#ifndef SWORDFISH_ARCH_MACROS_CUH
#define SWORDFISH_ARCH_MACROS_CUH

#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ == 1000)
#    define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#    define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
#    define CUTE_ARCH_TMA_SM100_ENABLED 1
#  elif (__CUDA_ARCH__ == 1030)
#    define CUTLASS_ARCH_MMA_SM103A_ENABLED 1
#    define CUTLASS_ARCH_MMA_SM103F_ENABLED 1
#    define CUTE_ARCH_TMA_SM100_ENABLED 1
#  elif (__CUDA_ARCH__ == 1100)
#    define CUTLASS_ARCH_MMA_SM110A_ENABLED 1
#    define CUTLASS_ARCH_MMA_SM110F_ENABLED 1
#    define CUTE_ARCH_TMA_SM100_ENABLED 1
#  endif
#endif

#endif  // SWORDFISH_ARCH_MACROS_CUH
