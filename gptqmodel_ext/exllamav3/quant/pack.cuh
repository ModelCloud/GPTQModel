#pragma once
#include "../../torch_stable_compat.h"


void pack_trellis
(
    at::Tensor packed,
    at::Tensor unpacked,
    int K
);

void unpack_trellis
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K
);

void pack_signs
(
    at::Tensor packed,
    at::Tensor unpacked
);

// R4NX / Q4NX trellis packing: 64 kv-bit states per tile into 4 * kv uint16
// words, kv in {10, 12, 14, 16}. Codebook independent; available on ROCm.
void pack_trellis_r4nx(at::Tensor packed, at::Tensor states, int kv);

void unpack_trellis_r4nx(at::Tensor states, at::Tensor packed, int kv);
