// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#pragma once
#include "../../torch_stable_compat.h"

// Fractional-bitrate trellis storage (K = KA + popcount(MASK)/16, half-integer in practice): pack /
// unpack the 16-bit windows to the (kb, nb, 16K)-uint16 tile layout and reconstruct fp16 weights
void pack_trellis_frac(at::Tensor packed, at::Tensor unpacked, int KA, int64_t MASK);
void unpack_trellis_frac(at::Tensor unpacked, at::Tensor packed, int KA, int64_t MASK);

void quantize_tiles_frac(
    at::Tensor input_tiles, at::Tensor output_tiles, at::Tensor output_indices,
    at::Tensor temp_costs, at::Tensor temp_edges, int KA, int64_t MASK);
