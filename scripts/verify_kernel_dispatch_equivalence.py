# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bitwise-equivalence proof for the #3983 dispatch changes.

The kernels themselves are untouched: only the Python work in front of them
shrinks. For the one change that alters the epilogue expression (the AWQ bias
add), this script captures the kernel result, applies the pre-change expression
as an independent oracle, and requires ``torch.equal`` against the module
output over the issue's shapes, dtypes, batch sizes, and bias settings.

    python scripts/verify_kernel_dispatch_equivalence.py --device cuda:0
"""

from __future__ import annotations

import argparse

import torch

import gptqmodel.nn_modules.qlinear.gemv_awq as gemv_awq
import gptqmodel.nn_modules.qlinear.gemv_fast_awq as gemv_fast_awq


SHAPES = ((4096, 4096), (4096, 14336), (14336, 4096), (2048, 2048))
BATCHES = (1, 16, 300, 640)
DTYPES = (torch.float16, torch.bfloat16)


def _capture(namespace, names):
    captured = {}
    originals = {}

    def make(name, original):
        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            captured["out"] = out
            return out

        return wrapper

    for name in names:
        originals[name] = getattr(namespace, name)
        setattr(namespace, name, make(name, originals[name]))
    return captured, originals


def _restore(namespace, originals):
    for name, original in originals.items():
        setattr(namespace, name, original)


def _build(cls, k, n, bias, device):
    module = cls(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=bias,
        register_buffers=True,
    ).to(device)
    groups = k // 128
    linear = torch.nn.Linear(k, n, bias=False, device=device, dtype=torch.float16)
    scales = (torch.rand(n, groups, device=device) * 0.01 + 1e-3).to(torch.float16)
    zeros = torch.randint(0, 16, (n, groups), device=device)
    module.pack(linear, scales, zeros)
    return module


def _check(module, namespace, kernel_attrs, x, dtype, bias):
    captured, originals = _capture(namespace, kernel_attrs)
    try:
        with torch.no_grad():
            out = module(x)
    finally:
        _restore(namespace, originals)
    kernel_out = captured["out"]
    # Pre-change expression, applied to the same kernel result.
    if bias is not None:
        oracle = (kernel_out.float() + bias.float()).to(dtype)
    else:
        oracle = kernel_out.to(dtype)
    return out, oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        raise SystemExit("AWQ native kernels require CUDA")
    torch.manual_seed(0)

    total = 0
    mismatches = 0
    worst = 0.0

    for label, cls, namespace, kernel_attrs in (
        ("gemv", gemv_awq.AwqGEMVLinear, gemv_awq, ("awq_gemv_forward", "awq_gemmv2_forward")),
        (
            "gemv_fast",
            gemv_fast_awq.AwqGEMVFastLinear,
            gemv_fast_awq,
            ("awq_fast_gemv_forward_decode", "awq_fast_gemm_forward_prefill"),
        ),
    ):
        module_globals = namespace
        for k, n in SHAPES:
            for dtype in DTYPES:
                for bias in (False, True):
                    module = _build(cls, k, n, bias, device)
                    if label == "gemv_fast":
                        setattr(module_globals, "awq_runtime_available", lambda: True)
                    bias_tensor = module.bias
                    for m in BATCHES:
                        x = torch.randn(m, k, device=device, dtype=dtype)
                        out, oracle = _check(module, module_globals, kernel_attrs, x, dtype, bias_tensor)
                        total += 1
                        if not torch.equal(out, oracle):
                            mismatches += 1
                            worst = max(worst, float((out.float() - oracle.float()).abs().max()))
                        elif out.dtype != dtype:
                            mismatches += 1

    print(f"cases={total} mismatches={mismatches} max_abs_diff={worst:.6g}")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
