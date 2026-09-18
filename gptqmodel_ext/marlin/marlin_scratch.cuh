#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include <c10/util/Exception.h>

namespace marlin {

// Keep these calculations in int64_t even though the kernel ABI uses int for
// dimensions.  The caller checks the ABI limits separately before launching a
// kernel, while the capacity query must not silently wrap during sizing.
inline int64_t checked_scratch_mul(int64_t lhs, int64_t rhs,
                                   const char* description) {
  TORCH_CHECK(lhs >= 0 && rhs >= 0, description,
              " must be non-negative (got ", lhs, " and ", rhs, ")");
  TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs,
              description, " overflows int64");
  return lhs * rhs;
}

inline int64_t checked_scratch_add(int64_t lhs, int64_t rhs,
                                   const char* description) {
  TORCH_CHECK(lhs >= 0 && rhs >= 0, description,
              " must be non-negative (got ", lhs, " and ", rhs, ")");
  TORCH_CHECK(rhs <= std::numeric_limits<int64_t>::max() - lhs, description,
              " overflows int64");
  return lhs + rhs;
}

inline int64_t checked_scratch_ceil_multiple(int64_t value, int64_t multiple,
                                             const char* description) {
  TORCH_CHECK(value >= 0 && multiple > 0, description,
              " has invalid arguments (got ", value, " and ", multiple,
              ")");
  const int64_t quotient = value / multiple;
  const int64_t remainder = value % multiple;
  const int64_t rounded_quotient =
      checked_scratch_add(quotient, remainder != 0 ? 1 : 0, description);
  return checked_scratch_mul(rounded_quotient, multiple, description);
}

inline std::tuple<int64_t, int64_t> marlin_scratch_sizes_checked(
    int64_t size_m, int64_t size_k, int64_t sms, int64_t max_thread_n,
    bool use_fp32_reduce, bool has_act_order) {
  TORCH_CHECK(size_m >= 0, "size_m must be non-negative (got ", size_m, ")");
  TORCH_CHECK(size_k >= 0, "size_k must be non-negative (got ", size_k, ")");
  TORCH_CHECK(sms >= 0, "sms must be non-negative (got ", sms, ")");
  TORCH_CHECK(max_thread_n >= 0, "max_thread_n must be non-negative (got ",
              max_thread_n, ")");

  int64_t c_tmp_elements = 0;
  if (use_fp32_reduce) {
    const int64_t rounded_m =
        checked_scratch_ceil_multiple(size_m, 16, "Marlin M capacity");
    const int64_t max_m_block_size = std::min<int64_t>(rounded_m, 64);
    c_tmp_elements = checked_scratch_mul(
        checked_scratch_mul(sms, max_m_block_size, "Marlin C scratch capacity"),
        max_thread_n, "Marlin C scratch capacity");
  }

  int64_t a_tmp_elements = 0;
  if (has_act_order) {
    a_tmp_elements =
        checked_scratch_mul(size_m, size_k, "Marlin A scratch capacity");
  }
  return {c_tmp_elements, a_tmp_elements};
}

}  // namespace marlin
