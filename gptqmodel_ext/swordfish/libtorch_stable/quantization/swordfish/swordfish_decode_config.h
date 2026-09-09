// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstdint>
#include <climits>
#include <stdexcept>
namespace swordfish {
struct DecodeConfig {
  static void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
  }
  int64_t mode, tiles, split, ctas;
  bool quad;
  int64_t threads, stages;
  void validate(bool w8, int k, int n) const {
    require(mode >= 0 && mode <= 2 && threads == 128,
                    "explicit decode mode must be 0..2; threads fixed at 128");
    require(tiles >= 1 && tiles <= (mode == 2 ? 4 : 3), "invalid M tile count");
    require(mode != 0 || tiles == 1, "deterministic decode uses one M tile");
    require(!quad || (mode == 2 && (tiles == 2 || tiles == 3) && n % 256 == 0),
                    "CTA quad requires Stream-K T2/T3 and N divisible by 256");
    const int expected_stages = mode == 0 ? 1 : tiles == 1 ? 5 :
        tiles == 2 ? (w8 ? 5 : 4) : tiles == 3 ? 3 : (w8 ? 4 : 2);
    require(stages == expected_stages, "stages differs from compiled kernel constraint");
    require(split >= 1 && split <= 65535 && split <= k / (w8 ? 16 : 32),
                    "split-K out of range");
    require(mode == 1 || split == 1, "split-K only applies to atomic mode");
    require(mode == 2 ? (ctas > 0 && ctas <= INT_MAX / 4) : ctas == 0,
                    "explicit CTA count required only for Stream-K");
  }
};
} // namespace swordfish
