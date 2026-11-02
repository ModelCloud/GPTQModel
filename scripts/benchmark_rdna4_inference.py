#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused-vs-split inference benchmark for RDNA4 (gfx1200/gfx1201/gfx1250).

For every (K, N) shape and every M band this compares the path that
``TorchLinear`` actually dispatches against an explicit
``dequantize_weight() + matmul`` reference:

* ``M == 1``        single-token decode (fused GEMV candidate)
* ``M in 2..16``    small-batch prefill (fused small-M GEMM candidate)
* ``M in 64, 128``  larger prefill; the fused gate rejects these by design

The reported ``fused_dispatch`` column comes from the module's own predicates
(``_can_use_rocm_gemv`` / ``_can_use_rocm_gemm``) plus
``gptqmodel.utils.arch.is_rocm_fused_arch``; it is not inferred from timing.
On an unregistered architecture (or with Triton unavailable) both predicates are
False and the row is a plain split-path measurement.

Timing is always bracketed by ``torch.cuda.synchronize``.

``v_wmma`` evidence
-------------------
``tl.dot`` lowering is not guaranteed to use WMMA, so a large speedup alone is
not evidence.  The script reads the compiled kernel's ``asm["amdgcn"]`` from the
Triton JIT cache of the kernel that **actually ran for that row** (the GEMV
kernel for ``M == 1``, the small-M GEMM kernel otherwise).  Reading the wrong
kernel would misattribute evidence: the GEMV kernel has no ``tl.dot`` at all and
legitimately contains no ``v_wmma``, while the GEMM kernel does.  Each row also
records the distinct ``v_wmma*`` mnemonics it saw.  If the JIT cache has no
assembly the script falls back to scanning files under ``TRITON_DUMP_DIR``
(produced with ``AMDGCN_ENABLE_DUMP=1`` set *before* compilation), and if
neither yields text it prints ``v_wmma: unknown（未取得 amdgcn）``.  It never
reports a hit it did not see.

``--dry-run`` prints the plan and exits 0; no GPU is needed.  Without a visible
GPU the same plan is printed with ``"gpu": "unavailable"`` and exit 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


# (K = in_features, N = out_features); a spread of decode and prefill projections.
DEFAULT_SHAPES = ((2048, 2048), (4096, 4096), (5120, 8192), (8192, 2048))
SMALL_M = tuple(range(2, 17))
LARGE_M = (64, 128)
DEFAULT_DTYPE = "float16"
DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 128

V_WMMA_UNKNOWN = "v_wmma: unknown（未取得 amdgcn）"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fused vs dequantize+matmul inference benchmark for RDNA4.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("float16", "bfloat16"))
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, choices=(2, 4, 8))
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument(
        "--shapes",
        default=",".join(f"{k}x{n}" for k, n in DEFAULT_SHAPES),
        help="Comma-separated KxN list, e.g. 2048x2048,4096x4096.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit 0.")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        k_text, _, n_text = item.partition("x")
        if not k_text or not n_text:
            raise ValueError(f"Invalid shape {item!r}; expected KxN, e.g. 4096x4096.")
        shapes.append((int(k_text), int(n_text)))
    if not shapes:
        raise ValueError("No shapes selected.")
    return shapes


def _device_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "is_rocm": torch.version.hip is not None,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "name": None,
        "arch": None,
        "arch_class": None,
        "is_rdna4": None,
        "fused_arch": None,
        "gate_limits": None,
        "error": None,
    }
    if not report["cuda_available"]:
        return report
    try:
        report["name"] = torch.cuda.get_device_name(device)
    except Exception as exc:  # pragma: no cover - runtime dependent
        report["error"] = f"get_device_name failed: {type(exc).__name__}: {exc}"
        return report
    try:
        from gptqmodel.utils import arch

        report["arch"] = arch.rocm_arch(device)
        report["arch_class"] = arch.arch_class(device)
        report["is_rdna4"] = arch.is_rdna4(device)
        report["fused_arch"] = arch.is_rocm_fused_arch(device)
        report["gate_limits"] = list(arch.gate_limits(device))
    except Exception as exc:  # pragma: no cover - runtime dependent
        report["error"] = f"arch probe failed: {type(exc).__name__}: {exc}"
    return report


def _plan(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    device = torch.device(args.device)
    shapes = _parse_shapes(args.shapes)
    return {
        "script": "benchmark_rdna4_inference.py",
        "mode": "dry-run",
        "reason": reason,
        "device": str(device),
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "shapes": [f"{k}x{n}" for k, n in shapes],
        "m_bands": {"decode": [1], "small_prefill": list(SMALL_M), "large_prefill": list(LARGE_M)},
        "warmup": args.warmup,
        "iters": args.iters,
        "output_json": None if args.output_json is None else str(args.output_json),
        "measured_per_row": [
            "fused_dispatch (bool, from module predicates + arch.is_rocm_fused_arch)",
            "fused_ms / split_ms and speedup",
            "output dtype",
            "asm source, the v_wmma* mnemonics found, and hit/unknown",
        ],
        "asm_probe": {
            "primary": "compiled asm['amdgcn'] of the kernel that ran for this row",
            "fallback_env": "set TRITON_DUMP_DIR=<dir> AMDGCN_ENABLE_DUMP=1 before compiling",
            "fallback_scan": "newest *.amdgcn under TRITON_DUMP_DIR",
            "if_unavailable": V_WMMA_UNKNOWN,
        },
        "timing": "torch.cuda.synchronize around every measured band",
        "device_info": _device_report(device),
        "commands": {
            "rocm": (
                "/root/miniconda3/envs/gp_314t_rocm/bin/python scripts/benchmark_rdna4_inference.py "
                f"--device {args.device} --dtype {args.dtype} --bits {args.bits}"
            ),
            "asm_dump": (
                "TRITON_DUMP_DIR=/tmp/triton-dump AMDGCN_ENABLE_DUMP=1 "
                "/root/miniconda3/envs/gp_314t_rocm/bin/python scripts/benchmark_rdna4_inference.py"
            ),
        },
    }


def _print_plan(plan: dict[str, Any]) -> None:
    print(json.dumps(plan, indent=2, sort_keys=True))
    info = plan["device_info"]
    print(
        "plan: "
        f"device={plan['device']} arch={info.get('arch')} class={info.get('arch_class')} "
        f"fused_arch={info.get('fused_arch')} shapes={len(plan['shapes'])} "
        f"m_bands=1|2..16|64,128",
        flush=True,
    )


def _make_module(in_features: int, out_features: int, dtype: torch.dtype, bits: int, group_size: int):
    """Build a packed TorchLinear with deterministic random codes (no checkpoint)."""
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    with torch.no_grad():
        qweight = torch.randint(-2**31, 2**31, module.qweight.shape, generator=generator, dtype=torch.int64)
        qzeros = torch.randint(-2**31, 2**31, module.qzeros.shape, generator=generator, dtype=torch.int64)
        module.qweight.copy_(qweight.to(torch.int32).to(module.qweight.device))
        module.qzeros.copy_(qzeros.to(torch.int32).to(module.qzeros.device))
        module.scales.uniform_(0.001, 0.02)
    module.post_init()
    return module


def _timed_ms(device: torch.device, fn, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1e3 / iters


def _iter_compiled_kernels(jit_fn):
    """Yield cached CompiledKernel objects for a Triton JITFunction."""
    caches = getattr(jit_fn, "device_caches", None)
    if not caches:
        return
    for entry in list(caches.values()):
        kernel_cache = entry[0] if isinstance(entry, (tuple, list)) and entry else None
        if not isinstance(kernel_cache, dict):
            continue
        for kernel in list(kernel_cache.values()):
            yield kernel


def _asm_from_jit(jit_fn) -> tuple[str | None, str | None]:
    """Return ``(amdgcn_text, kernel_name)`` for the compiled instantiations."""
    best: str | None = None
    name: str | None = None
    for kernel in _iter_compiled_kernels(jit_fn):
        asm = getattr(kernel, "asm", None)
        if not isinstance(asm, dict):
            continue
        text = asm.get("amdgcn")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if isinstance(text, str) and text:
            best = text
            name = getattr(kernel, "name", None) or getattr(jit_fn, "fn_name", None)
    return best, name


def _asm_from_dump_dir() -> tuple[str | None, str | None]:
    dump_dir = os.getenv("TRITON_DUMP_DIR")
    if not dump_dir:
        return None, None
    root = Path(dump_dir)
    if not root.is_dir():
        return None, None
    candidates = sorted(root.rglob("*.amdgcn"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None, None
    path = candidates[-1]
    try:
        return path.read_text(errors="replace"), str(path)
    except OSError:
        return None, None


def _wmma_ops(text: str) -> list[str]:
    """Return the distinct ``v_wmma*`` instruction mnemonics present in ``text``."""
    return sorted(set(re.findall(r"v_wmma[0-9a-z_]*", text)))


def _asm_report(source: str, kernel: str | None) -> tuple[bool | None, str, str | None, list[str]]:
    """Return ``(v_wmma hit, source label, asm path, wmma mnemonics)``.

    ``kernel`` names the fused kernel that *actually ran* for this row (``gemv``
    or ``gemm``).  Looking up the wrong JIT function would silently attribute one
    kernel's assembly to another: the GEMV kernel has no ``tl.dot`` at all, so its
    assembly legitimately contains no ``v_wmma``, while the small-M GEMM does.
    """
    if kernel is None:
        return None, "not-dispatched", None, []
    try:
        from gptqmodel.nn_modules.triton_utils.gptq_gemm import _gptq_gemm_small_m_kernel
        from gptqmodel.nn_modules.triton_utils.gptq_gemv import _gptq_gemv_kernel
    except Exception as exc:  # pragma: no cover - optional Triton dependency
        print(f"triton kernel import failed, asm unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None, V_WMMA_UNKNOWN, None, []

    lookup = {"gemv": ("jit:gptq_gemv", _gptq_gemv_kernel), "gemm": ("jit:gptq_gemm", _gptq_gemm_small_m_kernel)}
    label, jit_fn = lookup[kernel]
    text, compiled_name = _asm_from_jit(jit_fn)
    if text:
        return (bool(_wmma_ops(text)), f"{label}[{source}]", None, _wmma_ops(text))

    text, path = _asm_from_dump_dir()
    if text:
        return (bool(_wmma_ops(text)), f"dump:{path}[{source}]", path, _wmma_ops(text))
    return None, V_WMMA_UNKNOWN, None, []


def _bench_module(mod, in_features, out_features, args, device, dtype, fused_arch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band, m_values in (("decode", (1,)), ("small_prefill", SMALL_M), ("large_prefill", LARGE_M)):
        for m in m_values:
            x = torch.randn((m, in_features), device=device, dtype=dtype)
            with torch.inference_mode():
                fused_ok = mod._can_use_rocm_gemv(x) if m == 1 else mod._can_use_rocm_gemm(x)
            dispatch = bool(fused_arch and fused_ok)

            fused_ms = _timed_ms(device, lambda: mod(x), args.warmup, args.iters)

            def split():
                weight = mod.dequantize_weight()
                return torch.matmul(x, weight)

            split_ms = _timed_ms(device, split, args.warmup, args.iters)

            with torch.inference_mode():
                output = mod(x)
                split_output = torch.matmul(x, mod.dequantize_weight())
            max_abs = float((output.float() - split_output.float()).abs().max().item())

            wmma, asm_source, asm_path, wmma_ops = _asm_report(
                "module_forward" if dispatch else "split_path",
                ("gemv" if m == 1 else "gemm") if dispatch else None,
            )
            row = {
                "in_features": in_features,
                "out_features": out_features,
                "band": band,
                "m": m,
                "dtype": str(dtype).replace("torch.", ""),
                "bits": args.bits,
                "fused_arch": fused_arch,
                "fused_dispatch": dispatch,
                "fused_ms": round(fused_ms, 5),
                "split_ms": round(split_ms, 5),
                "speedup": round(split_ms / fused_ms, 3) if fused_ms > 0 else None,
                "output_dtype": str(output.dtype).replace("torch.", ""),
                "dtype_ok": output.dtype == x.dtype,
                "max_abs_vs_split": max_abs,
                "v_wmma": wmma,
                "v_wmma_ops": wmma_ops,
                "asm_source": asm_source,
                "asm_path": asm_path,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            verdict = "hit" if wmma is True else "not found" if wmma is False else V_WMMA_UNKNOWN
            ops = f" ops={','.join(wmma_ops)}" if wmma_ops else ""
            print(
                f"  [{band} M={m} K={in_features} N={out_features}] "
                f"fused_dispatch={dispatch} v_wmma: {verdict} ({asm_source}){ops}",
                flush=True,
            )
    return rows


def run(args: argparse.Namespace) -> int:
    from gptqmodel.utils import arch

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    shapes = _parse_shapes(args.shapes)
    fused_arch = bool(arch.is_rocm_fused_arch(device))
    torch.manual_seed(7)

    rows: list[dict[str, Any]] = []
    for in_features, out_features in shapes:
        torch.cuda.empty_cache()
        mod = _make_module(in_features, out_features, dtype, args.bits, args.group_size)
        rows.extend(_bench_module(mod, in_features, out_features, args, device, dtype, fused_arch))


    payload = {
        "mode": "measured",
        "device": str(device),
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "device_info": _device_report(device),
        "rows": rows,
    }
    print(json.dumps({"summary_rows": len(rows), "fused_rows": sum(r["fused_dispatch"] for r in rows)}, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        _print_plan(_plan(args, "requested via --dry-run"))
        return 0
    if not torch.cuda.is_available():
        _print_plan(_plan(args, "no CUDA/HIP device visible; nothing measured"))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
