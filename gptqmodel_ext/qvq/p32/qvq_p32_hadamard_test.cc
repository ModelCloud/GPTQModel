#include "qvq_p32_hadamard.h"

#include <cassert>
#include <initializer_list>

int main() {
  assert(qvq_p32_hadamard_base_order(5120) == 40);
  assert(qvq_p32_hadamard_base_order(6144) == 12);
  assert(qvq_p32_hadamard_base_order(12288) == 12);
  assert(qvq_p32_hadamard_base_order(17408) == 0);
  assert(qvq_p32_hadamard_base_order(8192) == 1);
  for (const int order : {12, 20, 40}) {
    for (int first = 0; first < order; ++first) {
      for (int second = 0; second < order; ++second) {
        int dot = 0;
        for (int column = 0; column < order; ++column) {
          dot += qvq_p32_hadamard_base_value(order, first, column) *
                 qvq_p32_hadamard_base_value(order, second, column);
        }
        assert(dot == (first == second ? order : 0));
      }
    }
  }
  return 0;
}
