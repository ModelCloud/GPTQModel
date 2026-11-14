# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-call host overhead in kernel dispatch (#3983).

Measures, over back-to-back eager calls (best of ``--repeat`` runs):

* torch.ops handle resolution for the Marlin/AWQ/ExLlamaV2/QQQ/Machete wrappers;
* ``ExllamaV2Linear._ensure_q_handle`` on an unchanged handle;
* bitsandbytes version validation and QuantState signature rebuild;
* the AWQ bias epilogue (wide FP32 add vs one FP16 add);
* real AWQ GEMM/GEMV/GEMV_FAST module forwards at the issue's shapes.

The same script runs on ``origin/main`` and on the branch, so the before/after
numbers come from one measurement method.

    python scripts/benchmark_kernel_dispatch_overhead.py --device cuda:0
"""

from __future__ import annotations

import argparse
import time
import types

import torch


def _bench(fn, iters: int, repeat: int, sync: bool) -> float:
    best = float("inf")
    for _ in range(repeat):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / iters)
    return best * 1e6


def _resolver(module, fast_name, op_name, extension_name, two_args=False):
    fast = getattr(module, fast_name, None)
    if fast is not None:
        if two_args:
            return lambda: fast(extension_name, op_name)
        return lambda: fast(op_name)
    return lambda: module._extension_api().op(extension_name, op_name)


def bench_ops(iters: int, repeat: int) -> dict[str, float]:
    import gptqmodel.utils.awq as awq
    import gptqmodel.utils.exllamav2 as exllamav2
    import gptqmodel.utils.machete as machete
    import gptqmodel.utils.marlin as marlin
    import gptqmodel.utils.qqq as qqq
    from gptqmodel import extension

    available = set()
    for name in ("marlin_fp16", "awq", "exllamav2", "qqq", "machete"):
        try:
            extension.load(name)
            available.add(name)
        except Exception as exc:  # pragma: no cover - host dependent
            print(f"skip {name}: {str(exc).splitlines()[0]}")

    if not available:
        return {}

    cases = {}
    if "marlin_fp16" in available:
        cases["marlin resolve"] = lambda: marlin._marlin_resolve_op(
            dtype=torch.float16, op_name="marlin_scratch_sizes"
        )
    if "awq" in available:
        cases["awq resolve"] = _resolver(awq, "_awq_op", "gemm_forward", "awq")
    if "exllamav2" in available:
        cases["exllamav2 resolve"] = _resolver(
            exllamav2, "_exllamav2_op", "gemm_half_q_half", "exllamav2", two_args=True
        )
    if "qqq" in available:
        cases["qqq resolve"] = _resolver(qqq, "_qqq_op", "qqq_gemm", "qqq")
    if "machete" in available:
        cases["machete resolve"] = _resolver(machete, "_machete_op", "machete_mm", "machete")
    return {name: _bench(fn, iters, repeat, sync=False) for name, fn in cases.items()}


def bench_exllamav2_handle(iters: int, repeat: int) -> float:
    import gptqmodel.nn_modules.qlinear.exllamav2 as exllamav2

    layer = types.SimpleNamespace(
        qweight=torch.zeros((2, 2), dtype=torch.int32),
        qzeros=torch.zeros((1, 2), dtype=torch.int32),
        scales=torch.ones((1, 2), dtype=torch.float16),
        g_idx=torch.zeros(8, dtype=torch.int32),
        q_handle=None,
        q_tensors=None,
        _q_handle_key=None,
        _q_fast=None,
        ext_make_q_matrix=lambda tensors, **kwargs: 1,
    )
    layer._release_q_handle = lambda: exllamav2.ExllamaV2Linear._release_q_handle(layer)
    ensure = lambda: exllamav2.ExllamaV2Linear._ensure_q_handle(layer)  # noqa: E731
    ensure()
    return _bench(ensure, iters, repeat, sync=False)


def bench_bitsandbytes(iters: int, repeat: int) -> tuple[float, float]:
    import sys

    import gptqmodel.nn_modules.qlinear.bitsandbytes as bnb

    fake_bnb = types.SimpleNamespace(
        __version__="0.49.0",
        functional=types.SimpleNamespace(QuantState=types.SimpleNamespace(from_dict=lambda *a, **k: object())),
    )
    sys.modules["bitsandbytes"] = fake_bnb
    bnb._clear_availability_state()

    state = types.SimpleNamespace(
        is_4bit=True,
        _cpu_packing_active=False,
        weight=torch.zeros(4, 4),
        weight_absmax=torch.zeros(1),
        weight_quant_map=torch.zeros(2),
        weight_quant_state=torch.zeros(3),
        quant_state=None,
        _quant_state_signature=None,
        _quant_state_payload=lambda: {},
    )
    refresh = lambda: bnb.BitsAndBytesLinear._refresh_quant_state(state)  # noqa: E731
    refresh()
    return (
        _bench(lambda: bnb.import_bitsandbytes(), iters, repeat, sync=False),
        _bench(refresh, iters, repeat, sync=False),
    )


def bench_bias_epilogue(iters: int, repeat: int, device: torch.device) -> dict[str, tuple[float, float]]:
    results = {}
    for k, n in ((4096, 4096), (4096, 11008), (14336, 4096), (2048, 2048)):
        out = torch.randn(1, n, device=device, dtype=torch.float16)
        bias = torch.randn(n, device=device, dtype=torch.float16)

        def wide(out=out, bias=bias):
            return (out.float() + bias.float()).to(dtype=torch.float16)

        def single(out=out, bias=bias):
            return torch.add(out, bias)

        results[f"{k}x{n}"] = (
            _bench(wide, iters, repeat, sync=True),
            _bench(single, iters, repeat, sync=True),
        )
    return results


def bench_awq_modules(iters: int, repeat: int, device: torch.device) -> dict[str, float]:
    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
    from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear

    k = n = 4096
    group_size = 128
    groups = k // group_size
    results = {}

    def build(cls, bias: bool):
        module = cls(
            bits=4,
            group_size=group_size,
            sym=False,
            desc_act=False,
            in_features=k,
            out_features=n,
            bias=bias,
            register_buffers=True,
        ).to(device)
        linear = torch.nn.Linear(k, n, bias=False, device=device, dtype=torch.float16)
        scales = (torch.rand(n, groups, device=device) * 0.01 + 1e-3).to(torch.float16)
        zeros = torch.randint(0, 16, (n, groups), device=device)
        module.pack(linear, scales, zeros)
        return module

    x = torch.randn(1, k, device=device, dtype=torch.float16)
    for label, cls, bias in (
        ("awq gemm", AwqGEMMLinear, False),
        ("awq gemm bias", AwqGEMMLinear, True),
        ("awq gemv", AwqGEMVLinear, False),
        ("awq gemv bias", AwqGEMVLinear, True),
        ("awq gemv_fast", AwqGEMVFastLinear, False),
        ("awq gemv_fast bias", AwqGEMVFastLinear, True),
    ):
        try:
            module = build(cls, bias)
            with torch.no_grad():
                module(x)
                results[label] = _bench(lambda m=module: m(x), iters, repeat, sync=True)
        except Exception as exc:  # pragma: no cover - host dependent
            print(f"skip {label}: {type(exc).__name__}: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    ops = bench_ops(args.iters, args.repeat)
    for name, value in ops.items():
        print(f"{name:22s} {value:8.3f} us")

    print(f"{'exllamav2 handle':22s} {bench_exllamav2_handle(args.iters, args.repeat):8.3f} us")
    import_us, refresh_us = bench_bitsandbytes(args.iters, args.repeat)
    print(f"{'bnb version check':22s} {import_us:8.3f} us")
    print(f"{'bnb quant_state sig':22s} {refresh_us:8.3f} us")

    for shape, (wide, single) in bench_bias_epilogue(args.iters, args.repeat, device).items():
        print(f"{'bias epilogue ' + shape:22s} wide={wide:7.3f} us  single={single:7.3f} us")

    for name, value in bench_awq_modules(args.iters, args.repeat, device).items():
        print(f"{name:22s} {value:8.3f} us")


if __name__ == "__main__":
    main()
