// Adapted from ExLlamaV3 v1.6.7 (8c593e4), copyright TurboDerp and contributors, MIT.
#pragma once
#include "../../torch_stable_compat.h"
#include <cmath>

// Split the EXL3 bitrate into the integral part and half-bit flag used
// by the kernel tables. Callers validate compatibility with their codebook.
struct BitsK
{
    int bits;
    bool half;
};

inline BitsK bits_from_K(double K)
{
    TORCH_CHECK(std::isfinite(K) && K >= 1.0 && K <= 8.0, "Unsupported EXL3 bitrate ", K);
    const int bits = (int) K;
    const double f = K - bits;
    TORCH_CHECK(bits >= 1 && bits <= 8 && (f == 0.0f || (f == 0.5f && bits <= 3)),
                "Unsupported EXL3 bitrate ", K, " (integer 1..8, or 1.5 / 2.5 / 3.5)");
    return { bits, f == 0.5f };
}

// Half-bit units (2 * bits + half) for runtime switches over mixed bitrates
inline int k2_from_K(float K)
{
    const BitsK b = bits_from_K(K);
    return 2 * b.bits + (b.half ? 1 : 0);
}
