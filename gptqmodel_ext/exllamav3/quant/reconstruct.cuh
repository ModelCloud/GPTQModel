#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);

void reconstruct_frac(at::Tensor unpacked, at::Tensor packed, int K);
