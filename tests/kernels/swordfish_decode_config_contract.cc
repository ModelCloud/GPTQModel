// SPDX-License-Identifier: AGPL-3.0-or-later
#include "gptqmodel_ext/swordfish/libtorch_stable/quantization/swordfish/swordfish_decode_config.h"
#include "gptqmodel_ext/quant_abi/quant_abi.h"
#include <cassert>
#include <cstddef>
using swordfish::DecodeConfig;
static bool accepts(DecodeConfig c, bool w8 = false, int n = 256) {
  try { c.validate(w8, 4096, n); return true; }
  catch (const std::invalid_argument&) { return false; }
}
int main() {
  static_assert(sizeof(QvqQuantTensor) == 56);
  static_assert(sizeof(QvqQuantConfig) == 112);
  static_assert(offsetof(QvqQuantConfig, schedule) == 104);
  for (bool w8 : {false, true}) {
    assert(accepts({0,1,1,0,false,128,1}, w8));
    for (int t = 1; t <= 4; ++t) {
      int stages = t == 1 ? 5 : t == 2 ? (w8 ? 5 : 4) : t == 3 ? 3 : (w8 ? 4 : 2);
      assert(accepts({2,t,1,32,false,128,stages}, w8));
      assert(!accepts({2,t,1,32,false,128,stages+1}, w8));
      if (t <= 3) assert(accepts({1,t,2,0,false,128,stages}, w8));
      if (t == 2 || t == 3) assert(accepts({2,t,1,32,true,128,stages}, w8));
    }
  }
  assert(!accepts({-1,1,1,0,false,128,1}));
  assert(!accepts({3,1,1,0,false,128,1}));
  assert(!accepts({0,1,1,0,false,256,1}));
  assert(!accepts({0,2,1,0,false,128,1}));
  assert(!accepts({0,1,2,0,false,128,1}));
  assert(!accepts({1,1,0,0,false,128,5}));
  assert(!accepts({1,1,65536,0,false,128,5}));
  assert(!accepts({1,4,1,0,false,128,2}));
  assert(!accepts({1,1,1,1,false,128,5}));
  assert(!accepts({2,1,1,0,false,128,5}));
  assert(!accepts({2,1,1,INT_MAX,false,128,5}));
  assert(!accepts({2,1,1,32,true,128,5}));
  assert(!accepts({2,2,1,32,true,128,4}, false, 64));
}
