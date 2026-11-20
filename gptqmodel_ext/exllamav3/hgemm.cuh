#pragma once
#include "../torch_stable_compat.h"


void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
);
