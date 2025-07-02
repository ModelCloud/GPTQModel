
#include "machete_mm_launcher.cuh"

namespace machete {



extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u4b8f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

torch::Tensor mm_dispatch_f16u4b8f16voidvoidvoidf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u4b8f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.maybe_schedule);
}

extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

torch::Tensor mm_dispatch_bf16u4b8bf16voidvoidvoidbf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u4b8bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.maybe_schedule);
}

extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_f16u8b128f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

torch::Tensor mm_dispatch_f16u8b128f16voidvoidvoidf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_f16u8b128f16voidvoidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.maybe_schedule);
}

extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs);
extern torch::Tensor impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs);

torch::Tensor mm_dispatch_bf16u8b128bf16voidvoidvoidbf16f32(MMArgs args) {
  [[maybe_unused]] auto M = args.A.size(0);
  [[maybe_unused]] auto N = args.B.size(1);
  [[maybe_unused]] auto K = args.A.size(1);
    
  if (!args.maybe_schedule) {
    if (M > 256 && K <= 16384 && N <= 4096)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 256)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 4096 && N <= 4096)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128 && K <= 8192 && N <= 8192)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 128)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 4069)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K <= 4069 && N <= 8192)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64 && K >= 8192 && N >= 12288)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 64)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K <= 6144 && N <= 6144)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32 && K >= 16384 && N >= 12288)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 32)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16 && K <= 12288 && N <= 8192)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (M > 16)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
    if (N >= 26624)
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
    else
        return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  }
  if (*args.maybe_schedule == "128x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x256_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x128_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "128x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x64_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x32_2x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(args);
  if (*args.maybe_schedule == "256x16_1x1x1_TmaMI__TmaCoop_streamK")
    return impl_bf16u8b128bf16voidvoidvoidbf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(args);
  TORCH_CHECK_NOT_IMPLEMENTED(false, "machete_gemm(..) is not implemented for "
                                     "schedule = ", *args.maybe_schedule);
}


static inline std::optional<at::ScalarType> maybe_scalartype(
    std::optional<at::Tensor> const& t) {
    if (!t) {
      return std::nullopt;
    } else {
      return t->scalar_type();
    };
}

torch::Tensor mm_dispatch(MMArgs args) {
  auto out_type = args.maybe_out_type.value_or(args.A.scalar_type());
  auto a_type = args.A.scalar_type();
  auto maybe_g_scales_type = maybe_scalartype(args.maybe_group_scales);
  auto maybe_g_zeros_type = maybe_scalartype(args.maybe_group_zeros);
  auto maybe_ch_scales_type = maybe_scalartype(args.maybe_channel_scales);
  auto maybe_tok_scales_type = maybe_scalartype(args.maybe_token_scales);

  
  if (args.b_type == vllm::kU4B8
      && a_type == at::ScalarType::Half
      && out_type == at::ScalarType::Half
      && maybe_g_scales_type == at::ScalarType::Half
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_f16u4b8f16voidvoidvoidf16f32(args);
  }
  if (args.b_type == vllm::kU4B8
      && a_type == at::ScalarType::BFloat16
      && out_type == at::ScalarType::BFloat16
      && maybe_g_scales_type == at::ScalarType::BFloat16
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_bf16u4b8bf16voidvoidvoidbf16f32(args);
  }
  if (args.b_type == vllm::kU8B128
      && a_type == at::ScalarType::Half
      && out_type == at::ScalarType::Half
      && maybe_g_scales_type == at::ScalarType::Half
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_f16u8b128f16voidvoidvoidf16f32(args);
  }
  if (args.b_type == vllm::kU8B128
      && a_type == at::ScalarType::BFloat16
      && out_type == at::ScalarType::BFloat16
      && maybe_g_scales_type == at::ScalarType::BFloat16
      && !maybe_g_zeros_type
      && !maybe_ch_scales_type
      && !maybe_tok_scales_type
  ) {
      return mm_dispatch_bf16u8b128bf16voidvoidvoidbf16f32(args);
  }
  
  TORCH_CHECK_NOT_IMPLEMENTED(
    false, "machete_mm(..) is not implemented for "
    "a_type=", args.A.scalar_type(),
    ", b_type=", args.b_type.str(),
    ", out_type=", out_type,
    ", with_group_scale_type=", maybe_g_scales_type
        ? toString(*maybe_g_scales_type) : "None",
    ", with_group_zeropoint_type=", maybe_g_zeros_type
        ? toString(*maybe_g_zeros_type) : "None",
    ", with_channel_scale_type=", maybe_ch_scales_type
        ? toString(*maybe_ch_scales_type) : "None",
    ", with_token_scale_type=", maybe_tok_scales_type
        ? toString(*maybe_tok_scales_type) : "None",
    "; implemented types are: \n",
    "\ta_type=f16, b_type=u4b8, with_group_scale_type=f16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=f16, accumulator_type=f32\n",
    "\ta_type=bf16, b_type=u4b8, with_group_scale_type=bf16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=bf16, accumulator_type=f32\n",
    "\ta_type=f16, b_type=u8b128, with_group_scale_type=f16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=f16, accumulator_type=f32\n",
    "\ta_type=bf16, b_type=u8b128, with_group_scale_type=bf16, with_group_zeropoint_type=void, with_channel_scale_type=void, a_token_scale_type=void, out_type=bf16, accumulator_type=f32\n",
    "");
}

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args) {
    auto out_type = args.maybe_out_type.value_or(args.a_type);
    
    
    if (args.b_type == vllm::kU4B8
        && args.a_type == at::ScalarType::Half
        && out_type == at::ScalarType::Half
        && args.maybe_group_scales_type == at::ScalarType::Half
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == vllm::kU4B8
        && args.a_type == at::ScalarType::BFloat16
        && out_type == at::ScalarType::BFloat16
        && args.maybe_group_scales_type == at::ScalarType::BFloat16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == vllm::kU8B128
        && args.a_type == at::ScalarType::Half
        && out_type == at::ScalarType::Half
        && args.maybe_group_scales_type == at::ScalarType::Half
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    if (args.b_type == vllm::kU8B128
        && args.a_type == at::ScalarType::BFloat16
        && out_type == at::ScalarType::BFloat16
        && args.maybe_group_scales_type == at::ScalarType::BFloat16
        && !args.maybe_group_zeros_type
    ) {
        return {
            "128x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x256_2x1x1_TmaMI__TmaCoop_streamK",
            "128x64_2x1x1_TmaMI__TmaCoop_streamK",
            "128x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x128_2x1x1_TmaMI__TmaCoop_streamK",
            "128x16_1x1x1_TmaMI__TmaCoop_streamK",
            "256x64_2x1x1_TmaMI__TmaCoop_streamK",
            "256x32_2x1x1_TmaMI__TmaCoop_streamK",
            "256x16_1x1x1_TmaMI__TmaCoop_streamK"
        };
    }
    
    return {};
};

}; // namespace machete