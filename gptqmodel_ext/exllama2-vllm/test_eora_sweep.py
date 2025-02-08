import torch
import time
# from eora import fused_concurrent, fused_sequential, cublas_reference, gptq_gemm_eora, gptq_gemm
from eora import gptq_gemm_eora, gptq_gemm
import pytest

m = 1
k = 4096
n = 6144
r = 128

bit = 4
use_exllama = True

BLOCK_KN_SIZE=128
r_size = BLOCK_KN_SIZE * r / k

max_k = 16384
k_step = 32
input = []
for k in range(k_step, max_k, k_step):
    for r in range(k_step, k, k_step):
        if BLOCK_KN_SIZE * r / k == BLOCK_KN_SIZE * r // k:
            print("k:{}, r:{}".format(k, r))
            input = input + [(k, r)]
print(input)

@pytest.mark.parametrize(
    "k, r",
    input,
)
def test_eora_kernel_sizes(k, r):
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
    torch.testing.assert_close(gptq_pytorch_out, gptq_eora_fused_out, rtol=0.05, atol=0.5)  # 5 % relative tolerance, 0.5 absolute tolerance
