import time

import torch
# from eora import fused_concurrent, fused_sequential, cublas_reference, gptq_gemm_eora, gptq_gemm
from gptqmodel_exllama_eora import gptq_gemm, gptq_gemm_lora

m = 1
k = 4096
n = 6144
r = 128

bit = 4
use_exllama = True

x = torch.rand((m, k), device='cuda', dtype=torch.float16)
eora_a = torch.randn((k, r), device='cuda', dtype=torch.float16) / 10.
eora_b = torch.randn((r, n), device='cuda', dtype=torch.float16) / 10.

eora_b[[0,1],:] = 0

# eora_b[:,[0,1]] = 0
# print(eora_b)
# gptq data
gptq_groups = 32
# weight = torch.randint(-2000000, 2000000, (int(k / 2 / bit), n), device='cuda', dtype=torch.int32)
weight = torch.zeros((int(k / 2 / bit), n), device='cuda', dtype=torch.int32)
zeros = torch.zeros((gptq_groups, int(n / 2 / bit)), device='cuda', dtype=torch.int32)
scales = torch.zeros((gptq_groups, n), device='cuda', dtype=torch.float16) / 1000.0
idx = torch.empty((0, ), device='cuda', dtype=torch.int32)

ax = x @ eora_a

def test_eora_kernel():
    gptq_pytorch_out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit) + (ax @ eora_b)
    print("gptq_pytorch_out: ")
    print(gptq_pytorch_out[0][:10])
    gptq_eora_fused_out = gptq_gemm_lora(x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    print("gptq_eora_fused_out: ")
    print(gptq_eora_fused_out[0][:10])
    torch.testing.assert_close(gptq_pytorch_out, gptq_eora_fused_out, rtol=0.05, atol=0.5)  # 5 % relative tolerance, 0.5 absolute tolerance

test_eora_kernel()