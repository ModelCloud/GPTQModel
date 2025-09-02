import triton
import triton.language as tl
import torch


@triton.jit()
def _a100_quantized_matmul(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                           stride_am, stride_ak,
                           stride_bk, stride_bn,
                           stride_cm, stride_cn,
                           stride_scales_g, stride_scales_n,
                           stride_zeros_g, stride_zeros_n,
                           groupsize,
                           m, n, k,
                           block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
                           group_size_m: tl.constexpr,
                           ):
    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, block_size_m)
    total_blocks_n = tl.cdiv(n, block_size_n)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = (pid_m * block_size_m + tl.arange(0, block_size_m)) % m
    offs_n = (pid_n * block_size_n + tl.arange(0, block_size_n)) % n

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_size_m), block_size_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)
    offs_k = tl.arange(0, block_size_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    output = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF  # b -> int32
        b = b * scales[None, :] - zeros[None, :]  # b -> fp16

        output += tl.dot(a, b)
        a_ptrs += stride_ak * block_size_k
        b_ptrs += (block_size_k // 8) * stride_bk

    output.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, output)


class a100_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros, group_size, block_size_m, ):
        m, k = a.shape
        _, n = b.shape

        # quant_groupsize = 128
        # block_size_m = 16
        # block_size_n = 32  # [N = 4096 // 32] = 128 blocks
        # block_size_k = 256
        # group_size_m = 8
        # num_warps = 4
        # num_stages = 8
        block_size_n = b.shape[1] // group_size
        block_size_k = b.shape[0] // group_size
        group_size_m = 8
        num_warps = 4
        num_stages = 8

        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)

        c = torch.zeros((m, n), device=b.device, dtype=torch.float16)
        k = _a100_quantized_matmul[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            group_size,
            m, n, k,
            block_size_m, block_size_n, block_size_k, group_size_m,
            num_warps=num_warps, num_stages=num_stages,
        )

        print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared / 1000} kB shared memory\n")

        with open('dequant_simple.txt', 'w') as f:
            print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared / 1000} kB shared memory\n", file=f)
            print("IR", k.asm['ttir'], file=f)
            print("TTGIR", k.asm['ttgir'], file=f)
            print("PTX", k.asm['ptx'], file=f)
            print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared / 1000} kB shared memory\n", file=f)

            print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c
