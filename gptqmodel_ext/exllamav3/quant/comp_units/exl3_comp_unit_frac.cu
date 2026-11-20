#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"

EXL3_FRACTIONAL_KERNEL_INSTANCES(1)
EXL3_FRACTIONAL_KERNEL_INSTANCES(2)
EXL3_FRACTIONAL_KERNEL_INSTANCES(3)
EXL3_Q4NX_KERNEL_INSTANCES(w25, 2, true)
EXL3_Q4NX_KERNEL_INSTANCES(w30, 3, false)
EXL3_Q4NX_KERNEL_INSTANCES(w35, 3, true)
EXL3_R4NX_KERNEL_INSTANCES(w25, 2, true)
EXL3_R4NX_KERNEL_INSTANCES(w30, 3, false)
EXL3_R4NX_KERNEL_INSTANCES(w35, 3, true)
EXL3_R4NX_KERNEL_INSTANCES(w40, 4, false)
