# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for issue #3984 (FP8 dequant scales, AWQ GEMM threshold).

Run on a CUDA device:

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=$PWD python tests/benchmark/benchmark_issue_3984.py

Reports microseconds and transient peak MiB. The "old" FP8 path reproduces the
pre-change ``weight.float() / expanded_scale`` formulation; the "native" AWQ
path is the split-K kernel. Shapes match the issue table.
"""

from __future__ import annotations

import statistics
import sys

import torch

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module
import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq
from gptqmodel.utils.awq import awq_dequantize_weights, awq_runtime_available


def _bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
        del out
    # Peak transient allocation is measured separately so resetting the peak
    # counter does not perturb the timing loop.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    del out
    return statistics.median(times), peak


def _fp8_module(out_features, in_features, method, block):
    module = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        format="float8_e4m3fn",
        weight_scale_method=method,
        weight_block_size=block,
        register_buffers=True,
    ).to("cuda")
    module.weight = torch.randn((out_features, in_features), device="cuda").to(torch.float8_e4m3fn)
    module.weight_scale_inv = torch.rand(module._scale_shape(), device="cuda") + 0.5
    return module


def _old_dequant(module, dtype):
    weight = module.weight.to(torch.float32)
    scale = module.weight_scale_inv.to(torch.float32)
    if module.weight_scale_method == "tensor":
        expanded = scale
    elif module.weight_scale_method == "row":
        expanded = scale.view(-1, 1)
    else:
        block_rows, block_cols = module.weight_block_size
        expanded = scale.repeat_interleave(block_rows, 0).repeat_interleave(block_cols, 1)
    return (weight / expanded).to(dtype).transpose(0, 1).contiguous()


def bench_fp8():
    print("FP8 dequant (weight-only), fp16 output")
    print("| scale | shape | old us | old peak MiB | new us | new peak MiB |")
    print("|---|---|---|---|---|---|")
    for method, block in (("row", None), ("block", (128, 128))):
        for out_features, in_features in ((4096, 4096), (4096, 14336)):
            module = _fp8_module(out_features, in_features, method, block)
            old_us, old_peak = _bench(lambda module=module: _old_dequant(module, torch.float16))
            new_us, new_peak = _bench(
                lambda module=module: module._dequantize_native(
                    target_device=torch.device("cuda"), target_dtype=torch.float16
                )
            )
            print(f"| {method} | {out_features}x{in_features} | {old_us:.1f} | {old_peak:.0f} | {new_us:.1f} | {new_peak:.0f} |")
            del module
            torch.cuda.empty_cache()


def _awq_tensors(in_features, out_features, group_size=128):
    qweight = torch.randint(0, 16, (in_features, out_features // 8), device="cuda", dtype=torch.int32)
    scales = torch.rand((in_features // group_size, out_features), device="cuda", dtype=torch.float16) + 0.5
    qzeros = torch.zeros((in_features // group_size, out_features // 8), device="cuda", dtype=torch.int32)
    return qweight, scales, qzeros


def bench_awq(rows_list=(1, 64, 640, 768, 1024, 2048)):
    print("\nAWQ GEMM, fp16 activations, native vs dequant + fp32 mm")
    print("| shape | M | native us | native peak MiB | new us | new peak MiB | speedup |")
    print("|---|---|---|---|---|---|---|")
    for in_features, out_features in ((4096, 4096), (4096, 14336), (14336, 4096)):
        qweight, scales, qzeros = _awq_tensors(in_features, out_features)
        for rows in rows_list:
            x = torch.randn((rows, in_features), device="cuda", dtype=torch.float16)

            def native(x=x, qweight=qweight, scales=scales, qzeros=qzeros):
                return gemm_awq._awq_cuda_gemm_forward(x, qweight, scales, qzeros, 8, fp32_accum=True)

            def new(x=x, qweight=qweight, scales=scales, qzeros=qzeros):
                weight = awq_dequantize_weights(qweight, scales, qzeros, 0, 0, 0, False)
                return torch.mm(x, weight.to(torch.float16), out_dtype=torch.float32).to(torch.float16)

            native_us, native_peak = _bench(native)
            new_us, new_peak = _bench(new)
            print(
                f"| {in_features}x{out_features} | {rows} | {native_us:.1f} | {native_peak:.0f} | "
                f"{new_us:.1f} | {new_peak:.0f} | {native_us / new_us:.2f}x |"
            )
            del x
        del qweight, scales, qzeros
        torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    bench_fp8()
    if awq_runtime_available():
        bench_awq()
    else:
        print("\nnative AWQ extension unavailable; skipping the AWQ GEMM benchmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
