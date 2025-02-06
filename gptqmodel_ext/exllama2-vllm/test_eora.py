import torch
import time
# from eora import fused_concurrent, fused_sequential, cublas_reference, gptq_gemm_eora, gptq_gemm
from eora import gptq_gemm_eora, gptq_gemm

m = 1
k = 4096
n = 6144
r = 128

bit = 4
use_exllama = True

x = torch.rand((m, k), device='cuda', dtype=torch.float16)
eora_a = torch.randn((k, r), device='cuda', dtype=torch.float16) / 10.
eora_b = torch.randn((r, n), device='cuda', dtype=torch.float16) / 10.

# gptq data
gptq_groups = 32
weight = torch.randint(-2000000, 2000000, (int(k / 2 / bit), n), device='cuda', dtype=torch.int32)
zeros = torch.zeros((gptq_groups, int(n / 2 / bit)), device='cuda', dtype=torch.int32)
scales = torch.rand((gptq_groups, n), device='cuda', dtype=torch.float16) / 1000.0
idx = torch.empty((0, ), device='cuda', dtype=torch.int32)

ax = x @ eora_a

def test_eora_kernel():
    gptq_pytorch_out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit) + (ax @ eora_b)
    gptq_eora_fused_out = gptq_gemm_eora(x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    torch.testing.assert_close(gptq_pytorch_out, gptq_eora_fused_out, rtol=0.05, atol=2)  # 5 % relative tolerance, 2 absolute tolerance
