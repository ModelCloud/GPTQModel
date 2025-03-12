import torch
from gptqmodel_exllama_eora import gptq_gemm, gptq_gemm_eora

m = 1
k = 4096
n = 6144 // 2
r = 128

bit = 3
use_exllama = True

x = torch.rand((m, k), device='cuda', dtype=torch.float16)
eora_a = torch.randn((k, r), device='cuda', dtype=torch.float16) / 10.
eora_b = torch.randn((n, r), device='cuda', dtype=torch.float16) / 10.
eora_b = eora_b.transpose(0, 1)

# gptq data
gptq_groups = 128
weight = torch.randint(-2000000, 2000000, (int(k / 2 / bit), n), device='cuda', dtype=torch.int32)
zeros = torch.zeros((gptq_groups, int(n / 2 / bit)), device='cuda', dtype=torch.int32)
scales = torch.rand((gptq_groups, n), device='cuda', dtype=torch.float16) / 1000.0
idx = torch.empty((0, ), device='cuda', dtype=torch.int32)

ax = x @ eora_a
residual = ax @ eora_b

def test_eora_kernel():
    gptq_pytorch_out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit) + residual
    gptq_eora_fused_out = gptq_gemm_eora(x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    torch.testing.assert_close(gptq_eora_fused_out, gptq_pytorch_out, rtol=0.05, atol=0.5)  # 5 % relative tolerance, 0.5 absolute tolerance
