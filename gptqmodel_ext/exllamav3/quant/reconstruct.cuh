#pragma once
#include "../../torch_stable_compat.h"


void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);
