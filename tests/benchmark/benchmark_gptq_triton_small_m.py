# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Cold-L2 benchmark for the GPTQ Triton decode / small-M paths (#3982).

Two tables:

1. ``TritonV2Linear.forward`` before/after the fused-dispatch reroute.
2. The packed-load ``gptq_gemm`` versus "best today": the pre-rewrite kernel
   inside its old Ada gate (K <= 2048, N <= 2048, M <= 16), otherwise the
   dequantize+matmul fallback.

``triton.testing.do_bench`` flushes L2 before every timed iteration, so the
numbers are cold-cache medians.  Run from the repository root:

    CUDA_VISIBLE_DEVICES=GPU-... PYTHONPATH=$PWD \
        /opt/venvs/cu130t/bin/python tests/benchmark/benchmark_gptq_triton_small_m.py

The pre-rewrite kernel below is a frozen reference, not the shipped code.
"""

import argparse

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.nn_modules.triton_utils.dequant import QuantLinearFunction
from gptqmodel.nn_modules.triton_utils.gptq_gemm import gptq_gemm


SHAPES = [
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
    (4096, 14336),
    (14336, 4096),
    (2048, 2048),
]
MS = [1, 2, 8, 16, 64]


@triton.jit
def _pre_rewrite_kernel(
    x_ptr, qweight_ptr, scales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    m,
    in_features: tl.constexpr, out_features: tl.constexpr,
    num_groups: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    bits: tl.constexpr,
):
    pack_factor: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1
    rows_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    m_mask = rows_m < m
    n_mask = cols < out_features
    accumulator = tl.zeros((block_m, block_n), tl.float32)
    k_offsets = tl.arange(0, block_k)
    for chunk in range(tl.cdiv(in_features, block_k)):
        rows = chunk * block_k + k_offsets
        row_mask = rows < in_features
        groups = tl.load(g_idx_ptr + rows, row_mask, other=0)
        groups = tl.where(groups < 0, groups + num_groups, groups)
        group_mask = row_mask & (groups >= 0) & (groups < num_groups)
        weight_mask = group_mask[:, None] & n_mask[None, :]
        weight_words = tl.load(
            qweight_ptr + (rows[:, None] // pack_factor) * out_features + cols[None, :],
            weight_mask, other=0,
        )
        zero_words = tl.load(
            qzeros_ptr + groups[:, None] * (out_features // pack_factor) + cols[None, :] // pack_factor,
            weight_mask, other=0,
        )
        scales = tl.load(
            scales_ptr + groups[:, None] * out_features + cols[None, :],
            weight_mask, other=0,
        ).to(tl.float32)
        codes = (weight_words >> ((rows[:, None] % pack_factor) * bits)) & maxq
        zeros = (zero_words >> ((cols[None, :] % pack_factor) * bits)) & maxq
        weights = ((codes - zeros).to(tl.float32) * scales).to(x_ptr.dtype.element_ty)
        activations = tl.load(
            x_ptr + rows_m[:, None] * in_features + rows[None, :],
            m_mask[:, None] & row_mask[None, :], other=0,
        )
        accumulator = tl.dot(activations, weights, accumulator)
    tl.store(
        out_ptr + rows_m[:, None] * out_features + cols[None, :],
        accumulator.to(out_ptr.dtype.element_ty),
        m_mask[:, None] & n_mask[None, :],
    )


def pre_rewrite_gptq_gemm(x, qweight, scales, qzeros, g_idx, bits=4):
    m, in_features = x.shape
    out_features = qweight.shape[1]
    output = torch.empty((m, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(out_features, 16))
    with torch.cuda.device(x.device):
        _pre_rewrite_kernel[grid](
            x, qweight, scales, qzeros, g_idx, output,
            m, in_features, out_features, scales.shape[0],
            16, 16, 256, bits, num_warps=4, num_stages=2,
        )
    return output


def _make(module_cls, in_features, out_features, dtype, bits=4, group_size=128):
    module = module_cls(
        bits=bits, group_size=group_size, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=False, register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(-2**31, 2**31, module.qweight.shape, device="cuda", dtype=torch.int32)
        )
        module.qzeros.copy_(
            torch.randint(-2**31, 2**31, module.qzeros.shape, device="cuda", dtype=torch.int32)
        )
        module.scales.uniform_(0.001, 0.02)
    module.post_init()
    return module


def _bench(fn):
    return do_bench(fn, warmup=10, rep=50, return_mode="median") * 1000.0


def _old_tritonv2_forward(module, x):
    x = module._apply_rotation_to_input(x)
    out_shape = x.shape[:-1] + (module.out_features,)
    out = QuantLinearFunction.apply(
        x.reshape(-1, x.shape[-1]),
        module.qweight, module.scales, module.qzeros, module.g_idx,
        module.bits, module.pack_dtype_bits, module.maxq, module.bias,
    ).reshape(out_shape)
    if module.adapter:
        out = module.adapter.apply(x=x, out=out)
    return out.to(dtype=x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--skip-tritonv2", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if not args.skip_tritonv2:
        print("# TritonV2Linear.forward, cold-L2 median us")
        print("K N M old_forward new_forward speedup")
        for k, n in SHAPES:
            module = _make(TritonV2Linear, k, n, dtype, args.bits, args.group_size)
            for m in MS:
                x = torch.randn(m, k, device="cuda", dtype=dtype)
                old = _bench(lambda: _old_tritonv2_forward(module, x))
                new = _bench(lambda: module(x))
                print(f"{k} {n} {m} {old:.2f} {new:.2f} {old / new:.2f}x")

    print("# gptq_gemm, cold-L2 median us")
    print("K N M best_today dequant_mm new_auto speedup")
    for k, n in SHAPES:
        module = _make(TorchLinear, k, n, dtype, args.bits, args.group_size)
        for m in MS:
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            in_old_gate = k <= 2048 and n <= 2048 and m <= 16

            def best_today():
                if in_old_gate:
                    return pre_rewrite_gptq_gemm(
                        x, module.qweight, module.scales, module.qzeros,
                        module.g_idx, bits=args.bits,
                    )
                weight = module.dequantize_weight()
                return torch.mm(x, weight, out_dtype=torch.float32).to(dtype)

            def split_path():
                weight = module.dequantize_weight()
                return torch.mm(x, weight, out_dtype=torch.float32).to(dtype)

            def new_auto():
                return gptq_gemm(
                    x, module.qweight, module.scales, module.qzeros,
                    module.g_idx, bits=args.bits,
                    uniform_group=module._gptq_gemm_uniform_group,
                )

            old = _bench(best_today)
            split = _bench(split_path)
            new = _bench(new_auto)
            print(f"{k} {n} {m} {old:.2f} {split:.2f} {new:.2f} {old / new:.2f}x")


if __name__ == "__main__":
    main()
