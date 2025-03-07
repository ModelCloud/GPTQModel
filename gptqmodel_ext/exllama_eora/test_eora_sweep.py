import pytest
import torch
from gptqmodel_exllama_eora import gptq_gemm, gptq_gemm_eora

m = 1
k = 4096
n = 6144
r = 128

bit_default = 4
use_exllama = True

BLOCK_KN_SIZE=32
r_size = BLOCK_KN_SIZE * r / k


max_k1 = 16384
k_step1 = 128
input1 = [(k, r, bit_default) for k in range(k_step1, max_k1, k_step1) for r in range(k_step1, k, k_step1)]

max_k2 = 4096
k_step2 = 32
input2 = [(k, r, bits) for k in range(k_step2, max_k2, k_step2) for r in range(k_step2, k, k_step2) for bits in [2, 3, 4]]

#same as input 2 but r is not divisible by 32 (35, 67, etc)
input3 = [(k, r, bit_default) for k in range(k_step2, max_k2, k_step2) for r in range(k_step2 + 3, k, k_step2)]

input = input1 + input2 + input3

@pytest.mark.parametrize(
    "k, r, bit",
    input,
)
def test_eora_kernel_sizes(k, r, bit):
    x = torch.rand((m, k), device='cuda', dtype=torch.float16)
    eora_a = torch.randn((k, r), device='cuda', dtype=torch.float16) / 10.
    eora_b = torch.randn((r, n), device='cuda', dtype=torch.float16) / 10.

    ax = x @ eora_a

    gptq_groups = 32
    weight = torch.randint(-2000000, 2000000, (int(k / 2 / bit), n), device='cuda', dtype=torch.int32)
    zeros = torch.zeros((gptq_groups, int(n / 2 / bit)), device='cuda', dtype=torch.int32)
    scales = torch.rand((gptq_groups, n), device='cuda', dtype=torch.float16) / 1000.0
    idx = torch.empty((0,), device='cuda', dtype=torch.int32)

    gptq_pytorch_out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit) + (ax @ eora_b)
    gptq_eora_fused_out = gptq_gemm_eora(x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    torch.testing.assert_close(gptq_pytorch_out, gptq_eora_fused_out, rtol=0.05, atol=1)  # 5 % relative tolerance, 1 absolute tolerance
