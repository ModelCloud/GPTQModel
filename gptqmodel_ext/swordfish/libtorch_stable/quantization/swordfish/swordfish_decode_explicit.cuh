// SPDX-FileCopyrightText: 2026 AlpinDale and the dphnAI/sonar contributors
// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

// Included inside namespace swordfish after the decode kernel definitions.
// The selected kernel and grid are exact. No environment or occupancy policy.
static_assert(kDecodeThreads == 128 && kDecodeWarps == 4 && kStages == 5,
              "update explicit configuration constraints with decode constants");

template <aphrodite::ScalarTypeId D, int T, bool Z, bool W8, bool Q = false>
void decode_explicit_t(const DecodeConfig& cfg, const void* a, const int32_t* b,
                       const void* s, const void* z, void* c,
                       int m, int k, int n, int group_size, cudaStream_t stream) {
  using S = typename marlin::MarlinScalarType<D>::scalar_t;
  auto ap = reinterpret_cast<const S*>(a);
  auto sp = reinterpret_cast<const S*>(s);
  auto zp = reinterpret_cast<const S*>(z);
  auto cp = reinterpret_cast<S*>(c);
  if (cfg.mode == 2) {
    launch_zero_c<S>(c, m, n, stream);
    swordfish_decode_streamk_kernel<D, T, Z, W8, Q>
        <<<int(cfg.ctas), kDecodeThreads, 0, stream>>>(
            ap, b, sp, zp, cp, m, k, n, group_size, ((m + 15) / 16 + T - 1) / T);
  } else if constexpr (T <= 3 && !Q) {
    if (cfg.mode == 1) {
      if (cfg.split > 1) launch_zero_c<S>(c, m, n, stream);
      swordfish_decode_kernel<D, true, T, Z, W8>
          <<<dim3((m + 16 * T - 1) / (16 * T), n / kBlockN, int(cfg.split)), kDecodeThreads, 0, stream>>>(
              ap, b, sp, zp, cp, m, k, n, group_size);
    } else if constexpr (T == 1) {
      swordfish_decode_kernel<D, false, 1, Z, W8>
          <<<dim3((m + 15) / 16, n / kBlockN), kDecodeThreads, 0, stream>>>(
              ap, b, sp, zp, cp, m, k, n, group_size);
    }
  }
  const auto status = cudaGetLastError();
  STD_TORCH_CHECK(status == cudaSuccess, "explicit decode launch: ", cudaGetErrorString(status));
}
template <aphrodite::ScalarTypeId D, bool Z, bool W8 = false>
void decode_explicit(const DecodeConfig& cfg, const void* a, const int32_t* b,
                     const void* s, const void* z, void* c,
                     int m, int k, int n, int group_size, cudaStream_t stream) {
  if (cfg.quad) {
    if (cfg.tiles == 2) decode_explicit_t<D, 2, Z, W8, true>(cfg,a,b,s,z,c,m,k,n,group_size,stream);
    else decode_explicit_t<D, 3, Z, W8, true>(cfg,a,b,s,z,c,m,k,n,group_size,stream);
  } else switch (cfg.tiles) {
    case 1: decode_explicit_t<D, 1, Z, W8>(cfg,a,b,s,z,c,m,k,n,group_size,stream); break;
    case 2: decode_explicit_t<D, 2, Z, W8>(cfg,a,b,s,z,c,m,k,n,group_size,stream); break;
    case 3: decode_explicit_t<D, 3, Z, W8>(cfg,a,b,s,z,c,m,k,n,group_size,stream); break;
    case 4: decode_explicit_t<D, 4, Z, W8>(cfg,a,b,s,z,c,m,k,n,group_size,stream); break;
  }
}
