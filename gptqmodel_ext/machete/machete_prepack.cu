
#include "machete_prepack_launcher.cuh"

namespace machete {

torch::Tensor prepack_B_dispatch(PrepackBArgs args) {
  auto convert_type = args.maybe_group_scales_type.value_or(args.a_type);
  
  if (args.a_type == at::ScalarType::Half
      && args.b_type.size_bits() == 4 
      && convert_type == at::ScalarType::Half) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        cutlass::half_t, // ElementA
        cutlass::uint4b_t, // ElementB
        cutlass::half_t, // ElementConvert
        float, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>
    >(args.B); 
  }
  
  if (args.a_type == at::ScalarType::BFloat16
      && args.b_type.size_bits() == 4 
      && convert_type == at::ScalarType::BFloat16) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        cutlass::bfloat16_t, // ElementA
        cutlass::uint4b_t, // ElementB
        cutlass::bfloat16_t, // ElementConvert
        float, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>
    >(args.B); 
  }
  
  if (args.a_type == at::ScalarType::Half
      && args.b_type.size_bits() == 8 
      && convert_type == at::ScalarType::Half) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        cutlass::half_t, // ElementA
        uint8_t, // ElementB
        cutlass::half_t, // ElementConvert
        float, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>
    >(args.B); 
  }
  
  if (args.a_type == at::ScalarType::BFloat16
      && args.b_type.size_bits() == 8 
      && convert_type == at::ScalarType::BFloat16) {
    return prepack_impl<
      PrepackedLayoutBTemplate<
        cutlass::bfloat16_t, // ElementA
        uint8_t, // ElementB
        cutlass::bfloat16_t, // ElementConvert
        float, // Accumulator
        cutlass::layout::ColumnMajor,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>
    >(args.B); 
  }
  
  TORCH_CHECK_NOT_IMPLEMENTED(false, 
    "prepack_B_dispatch(..) is not implemented for "
    "atype = ", args.a_type,
    ", b_type = ", args.b_type.str(),
    ", with_group_scales_type= ", args.maybe_group_scales_type ? 
        toString(*args.maybe_group_scales_type) : "None");
}

}; // namespace machete