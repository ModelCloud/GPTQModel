#pragma once
#include "../../torch_stable_compat.h"

#include <tuple>

void count_inf_nan
(
    at::Tensor x,
    at::Tensor y
);