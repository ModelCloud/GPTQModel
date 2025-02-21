# import eora_cuda
#
#
# def gptq_gemm(x, w_q_weight, w_gptq_qzeros, w_gptq_scales, w_g_idx, use_exllama, bit):
#     return eora_cuda.gptq_gemm(x, w_q_weight, w_gptq_qzeros, w_gptq_scales, w_g_idx, use_exllama, bit)
#
#
# def gptq_gemm_eora(x, w_q_weight, w_gptq_qzeros, w_gptq_scales, w_g_idx, use_exllama, bit, Ax, B):
#     return eora_cuda.gptq_gemm_eora(x, w_q_weight, w_gptq_qzeros, w_gptq_scales, w_g_idx, use_exllama, bit, Ax, B)
