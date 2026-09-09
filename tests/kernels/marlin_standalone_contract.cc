// SPDX-License-Identifier: Apache-2.0
// Host-only contract check; intentionally compiled without Torch include paths.
#define QVQ_MARLIN_STANDALONE
#include "core/scalar_type.hpp"
#include <cassert>

int main() {
  static_assert(vllm::ScalarType::from_id(vllm::kU4.id()) == vllm::kU4);
  static_assert(vllm::ScalarType::from_id(vllm::kU4B8.id()) == vllm::kU4B8);
  static_assert(vllm::ScalarType::from_id(vllm::kU8B128.id()) == vllm::kU8B128);
  static_assert(vllm::kU4B8.size_bits() == 4);
  static_assert(vllm::kU8B128.size_bits() == 8);
  assert(std::get<int64_t>(vllm::kU4B8.min()) == -8);
  assert(std::get<int64_t>(vllm::kU4B8.max()) == 7);
  assert(std::get<int64_t>(vllm::kU8B128.min()) == -128);
  assert(std::get<int64_t>(vllm::kU8B128.max()) == 127);
  bool rejected = false;
  try {
    TORCH_CHECK(false, "unsupported tile ", 17);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("unsupported tile 17") != std::string::npos;
  }
  assert(rejected);
  rejected = false;
  try {
    TORCH_CHECK(false);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
