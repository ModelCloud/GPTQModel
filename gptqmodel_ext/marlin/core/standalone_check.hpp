// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace qvq_marlin_detail {
template <typename... Args>
[[noreturn]] inline void fail(const char* condition, Args&&... args) {
  std::ostringstream message;
  message << "Marlin check failed: " << condition << ": ";
  if constexpr (sizeof...(Args) > 0) {
    (message << ... << std::forward<Args>(args));
  }
  throw std::invalid_argument(message.str());
}
}  // namespace qvq_marlin_detail

// Keep the shared dispatch checks identical in the standalone and Torch builds.
// The public C entry points must catch these exceptions.
#define TORCH_CHECK(condition, ...) \
  do { if (!(condition)) qvq_marlin_detail::fail(#condition, ##__VA_ARGS__); } while (false)
#define TORCH_CHECK_NOT_IMPLEMENTED TORCH_CHECK
