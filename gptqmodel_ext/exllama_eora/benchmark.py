import time

import torch
from gptqmodel_exllama_eora import gptq_gemm, gptq_gemm_eora

m = 8
k = 4096
n = 6144
r = 128

bit = 4
use_exllama = True

warmup_iterations = 50
total_iterations = 10000

x = torch.rand((m, k), device='cuda', dtype=torch.float16) * 10.
W = torch.randn((k, n), device='cuda', dtype=torch.float16)
eora_a = torch.randn((k, r), device='cuda', dtype=torch.float16) / 10.
eora_b = torch.randn((r, n), device='cuda', dtype=torch.float16) / 10.


# reference torch version
Y = (x @ W) + ((x @ eora_a) @ eora_b)


# gptq data
gptq_groups = 32
weight = torch.randint(-2000000, 2000000, (int(k / 2 / bit), n), device='cuda', dtype=torch.int32)
zeros = torch.zeros((gptq_groups, int(n / 2 / bit)), device='cuda', dtype=torch.int32)
scales = torch.rand((gptq_groups, n), device='cuda', dtype=torch.float16) / 1000.0
idx = torch.empty((0, ), device='cuda', dtype=torch.int32)

def benchmark_pytorch_reference(W, x, eora_b, eora_a):
    for i in range(warmup_iterations):
        Y = (x @ W) + ((x @ eora_a) @ eora_b)
    torch.cuda.synchronize()
    tick = time.time()
    for i in range(total_iterations):
        Y = (x @ W)
    torch.cuda.synchronize()
    print(f"pytorch baseline: {(time.time() - tick) / total_iterations * 1000} msec")

    torch.cuda.synchronize()
    tick = time.time()
    for i in range(total_iterations):
        Y = (x @ W) + ((x @ eora_a) @ eora_b)
    torch.cuda.synchronize()
    print(f"pytorch LORA baseline: {(time.time() - tick) / total_iterations * 1000} msec")


def benchmark_gptq_kernel(m, weight, zeros, scales, idx, x, eora_b, eora_a) -> float:
    x = torch.rand((m, k), device='cuda', dtype=torch.float16) * 10.

    for i in range(warmup_iterations):
        Y = (x @ W) + ((x @ eora_a) @ eora_b)
    torch.cuda.synchronize()
    tick = time.time()
    for i in range(total_iterations):
        Y = (x @ W)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - tick) / total_iterations * 1000
    print(f"pytorch baseline: {pytorch_time} msec")

    torch.cuda.synchronize()
    tick = time.time()
    for i in range(total_iterations):
        Y = (x @ W) + ((x @ eora_a) @ eora_b)
    torch.cuda.synchronize()
    pytorch_lora_time = (time.time() - tick) / total_iterations * 1000
    print(f"pytorch LORA baseline: {pytorch_lora_time} msec")

    ax = (x @ eora_a)
    out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit)
    for i in range(warmup_iterations):
        out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit)
    torch.cuda.synchronize()
    tick = time.time()
    for i in range(total_iterations):
        out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit)
    torch.cuda.synchronize()
    print(f"gptq: {(time.time() - tick) / total_iterations * 1000} msec")

    tick = time.time()
    for i in range(total_iterations):
        out = gptq_gemm(x, weight, zeros, scales, idx, use_exllama, bit) + (ax @ eora_b)
    torch.cuda.synchronize()
    gptq_lora_pytorch_time = (time.time() - tick) / total_iterations * 1000
    print(f"gptq + pytorch for LORA: {gptq_lora_pytorch_time} msec")

    # gptq+eora kernel
    for i in range(warmup_iterations):
        gptq_eora_out = gptq_gemm_eora(x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    torch.cuda.synchronize()
    tick = time.time()
    for i in range(total_iterations):
        gptq_eora_out = gptq_gemm_eora(x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    torch.cuda.synchronize()
    gptq_fused_kernel_time = (time.time() - tick) / total_iterations * 1000
    print(f"gptq eora kernel: {gptq_fused_kernel_time} msec")
    speedup = gptq_lora_pytorch_time / gptq_fused_kernel_time
    print(f"speedup (gptq+pytorch/fused_kernel ratio) for batch size {m}: {speedup}")
    # print(f"pytorch_lora/fused_kernel ratio for batch size {m}: {pytorch_lora_time / gptq_fused_kernel_time}")
    print("")
    return speedup



benchmark_pytorch_reference(W, x, eora_b, eora_a)
speedups = []
for i in range(1, 50):
    speedups.append(benchmark_gptq_kernel(i, weight, zeros, scales, idx, x, eora_b, eora_a))
print(speedups)
