// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>

namespace nf12_ad::validation {
inline bool aligned(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}
inline bool aligned16(const void* p) { return aligned(p, 16); }

// Bound intermediate byte counts to int64.
inline bool valid_shape(int64_t n, int64_t k) {
  return n > 0 && k > 0 && n % 16 == 0 && k % 64 == 0 &&
         n <= std::numeric_limits<int64_t>::max() / 4 / k;
}
inline bool valid_moe_shape(int64_t experts, int64_t n, int64_t k) {
  return experts >= 1 && valid_shape(n, k) &&
         experts <= std::numeric_limits<int64_t>::max() / 2 / n / k;
}

// Row-major BF16 accessed with 16-byte vectors; callers also bound row counts.
inline bool valid_bf16_matrix(const void* p, int64_t cols, int64_t ld) {
  return p != nullptr && aligned16(p) && ld >= cols && ld % 8 == 0;
}
}  // namespace nf12_ad::validation
