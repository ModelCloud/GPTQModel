#define MARLIN_GEMM_EXPORT_NAME gptq_marlin_gemm_fp16
#define MARLIN_SCRATCH_SIZES_EXPORT_NAME gptq_marlin_scratch_sizes_fp16
#define MARLIN_ENABLE_FP16 1
#define MARLIN_ENABLE_BF16 0

#include "gptq_marlin.cu"
