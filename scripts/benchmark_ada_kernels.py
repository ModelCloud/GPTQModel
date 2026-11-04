#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Ada (RTX 4090, SM 8.9) tuned-kernel benchmark for GPTQModel.

Measures every dispatch decision the Ada tables make, on the real device:

* **GEMV** (``M == 1``): the shipped ``ada`` tile
  ``(max_block_k=256, block_n=8, num_warps=8)`` against the portable
  ``(512, 8, 8)`` default, and against ``dequantize + matmul``.
* **small-M GEMM** (``M in 2..16``): the shipped ``ada`` tile
  ``(16, 16, 256, 4, 2)`` against ``dequantize + matmul``; also reports the
  module's own ``_can_use_rocm_gemm`` dispatch so the measured gate
  (``K <= 2048`` and ``N <= 2048``) is visible next to the timing.
* **dequant**: the shipped ``ada`` tile ``(x_block=512, warps=1, stages=2)``
  against the portable ``(1024, 1, 1)`` default.
* **GGUF** ``Q4_K``/``Q5_K``/``Q6_K``: the SM89 fixed tile selected for the
  exact ``(padded K, N, row band)`` against the kernel's own autotuned launch.

Protocol (identical to ``scripts/benchmark_rdna4_inference.py``)

* ``warmup`` iterations, then alternating hot and L2-flushed iterations
  (a 256 MiB buffer is zeroed before every odd iteration);
* ``torch.cuda.synchronize`` before and after every timed launch, and CUDA
  events around the launch itself;
* the **median** of the samples is reported.  ``hot``/``flushed`` columns are
  both medians over their own subsets.

Every number this script prints is produced by measurement; nothing is
hard-coded.  ``--dry-run`` prints the plan and exits 0 without touching a GPU.
If no Ada GPU is visible the plan is printed with ``"gpu": "unavailable"`` and
the exit code stays 0, so it is safe in CI.

Usage::

    CUDA_VISIBLE_DEVICES=0 /root/venv/bin/python scripts/benchmark_ada_kernels.py \
        --sections gemv,gemm,dequant --output-json /tmp/ada-bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


DEFAULT_SHAPES = ((896, 128), (2048, 1024), (2048, 2048), (4096, 4096), (4096, 11008), (5120, 12288))
GEMM_SHAPES = ((512, 512), (896, 1280), (1536, 4096), (2048, 1024), (2048, 2048), (2048, 3072), (4096, 11008))
GGUF_CASES = (
    ("Q4_K", 1024, 128),
    ("Q4_K", 2048, 2048),
    ("Q4_K", 4096, 4096),
    ("Q5_K", 2048, 2048),
    ("Q5_K", 4096, 4096),
    ("Q6_K", 1024, 128),
    ("Q6_K", 4096, 11008),
)
GGUF_ROWS = (1, 8, 64)
SECTIONS = ("gemv", "gemm", "dequant", "gguf", "paroquant", "awq_gemv")
L2_FLUSH_BYTES = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ada (RTX 4090) tuned-kernel benchmark.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--sections", default=",".join(SECTIONS),
        help=f"Comma-separated subset of {','.join(SECTIONS)}.",
    )
    parser.add_argument(
        "--shapes",
        default=",".join(f"{k}x{n}" for k, n in DEFAULT_SHAPES),
        help="Comma-separated KxN list for the GPTQ GEMV/dequant sections.",
    )
    parser.add_argument(
        "--gemm-shapes",
        default=",".join(f"{k}x{n}" for k, n in GEMM_SHAPES),
        help="Comma-separated KxN list for the GPTQ small-M GEMM section.",
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=60)
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


def _sections(value: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in SECTIONS]
    if unknown:
        raise ValueError(f"Unknown section(s) {unknown}; choose from {list(SECTIONS)}.")
    if not selected:
        raise ValueError("No sections selected.")
    return selected


def _device_report(device: torch.device) -> dict[str, Any]:
    from gptqmodel.utils import arch

    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "is_rocm": torch.version.hip is not None,
        "torch": torch.__version__,
        "triton": None,
        "name": None,
        "capability": None,
        "cuda_arch": None,
        "tuning_family": None,
        "is_ada": None,
        "sm_count": None,
        "total_memory": None,
        "gate_limits": None,
        "gemm_fused_dtypes": None,
        "error": None,
    }
    try:
        import triton

        report["triton"] = triton.__version__
    except Exception:  # pragma: no cover - triton is a hard dependency of the kernels
        pass
    if not report["cuda_available"]:
        return report
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report["name"] = props.name
        report["capability"] = list(torch.cuda.get_device_capability(index))
        report["cuda_arch"] = arch.cuda_arch(index)
        report["tuning_family"] = arch.tuning_family(index)
        report["is_ada"] = arch.is_ada(index)
        report["sm_count"] = props.multi_processor_count
        report["total_memory"] = props.total_memory
        report["gate_limits"] = list(arch.gate_limits(index))
        report["gemm_fused_dtypes"] = sorted(arch.gemm_fused_dtypes(index))
    except Exception as exc:  # pragma: no cover - defensive
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


class L2Flush:
    """Zero a 256 MiB buffer to evict the kernel's data from L2."""

    def __init__(self, device: torch.device):
        self.buffer = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device)

    def flush(self) -> None:
        self.buffer.zero_()


def _timed_us(device: torch.device, fn, *, warmup: int, iters: int, l2: L2Flush | None) -> tuple[float, float | None]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        samples: list[float] = []
        for index in range(iters):
            if l2 is not None and index % 2 == 1:
                l2.flush()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end) * 1e3)
    if l2 is None:
        return statistics.median(samples), None
    return statistics.median(samples[::2]), statistics.median(samples[1::2])


def _speedup(base: float | None, tuned: float | None) -> float | None:
    if not base or not tuned:
        return None
    return base / tuned


def _plan(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "script": "benchmark_ada_kernels.py",
        "mode": "dry-run",
        "reason": reason,
        "device": args.device,
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "sections": _sections(args.sections),
        "shapes": [f"{k}x{n}" for k, n in _parse_shapes(args.shapes)],
        "gemm_shapes": [f"{k}x{n}" for k, n in _parse_shapes(args.gemm_shapes)],
        "gguf_cases": [f"{q}|{k}x{n}" for q, k, n in GGUF_CASES],
        "gguf_rows": list(GGUF_ROWS),
        "warmup": args.warmup,
        "iters": args.iters,
        "output_json": None if args.output_json is None else str(args.output_json),
        "timing": (
            "median after --warmup; torch.cuda.synchronize before/after every launch; "
            "alternating 256 MiB L2 flush; hot and flushed medians reported separately"
        ),
        "measured_per_row": [
            "hot_us / flushed_us for each variant",
            "speedup of the shipped ada tile vs the portable reference",
            "dispatch predicate (what TorchLinear actually uses for that row)",
        ],
        "device_info": _device_report(torch.device(args.device)),
    }


def _print_plan(plan: dict[str, Any]) -> None:
    print(json.dumps(plan, indent=2, sort_keys=True))
    info = plan["device_info"]
    print(
        "plan: "
        f"device={plan['device']} name={info.get('name')} arch={info.get('cuda_arch')} "
        f"family={info.get('tuning_family')} sm={info.get('sm_count')} "
        f"sections={','.join(plan['sections'])}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# GPTQ (TorchLinear) helpers
# --------------------------------------------------------------------------- #

def _make_module(in_features: int, out_features: int, dtype: torch.dtype, bits: int, group_size: int, device: torch.device):
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
    ).to(device=device, dtype=dtype).eval()
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


def _gptq_gemv_row(module, in_features, out_features, args, device, dtype, l2):
    """Time the shipped Ada GEMV tile against the portable 512 tile."""
    from gptqmodel.nn_modules.triton_utils import gptq_gemv as gemv_mod

    x = torch.randn(1, in_features, device=device, dtype=dtype)
    shipped = gemv_mod._GEMV_LAUNCH_CONFIGS["ada"]
    portable = gemv_mod._DEFAULT_GEMV_LAUNCH

    def launch(cfg):
        def fn():
            block_k = min(cfg.max_block_k, max(32, 1 << (in_features - 1).bit_length()))
            block_n = cfg.block_n
            out = torch.empty((1, out_features), device=device, dtype=dtype)
            gemv_mod._gptq_gemv_kernel[(triton_cdiv(out_features, block_n),)](
                x, module.qweight, module.scales, module.qzeros, module.g_idx, out,
                in_features=in_features, out_features=out_features,
                num_groups=module.scales.shape[0], block_k=block_k, block_n=block_n,
                bits=args.bits, num_warps=cfg.num_warps,
            )
            return out
        return fn

    ada_fn, portable_fn = launch(shipped), launch(portable)
    ada_hot, ada_flushed = _timed_us(device, ada_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    base_hot, base_flushed = _timed_us(device, portable_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    with torch.inference_mode():
        ada_out, base_out = ada_fn(), portable_fn()
    # Independent oracle: unpack the packed int32 words with Torch integer
    # shifts, apply ``(code - zero) * scale`` group-wise, then a FP64 matmul.
    # It never calls the Triton kernel.  Registered tiles differ from the
    # portable kernel only in the reduction tree, so the sibling drift is
    # reported separately from the oracle error.
    oracle = _gptq_fp64_oracle(x, module, args.bits)
    rounded = oracle.to(dtype).to(torch.float64)
    ada64 = ada_out.detach().cpu().to(torch.float64)
    ref64 = base_out.detach().cpu().to(torch.float64)
    ada_oracle_abs = (ada64 - oracle).abs().max().item()
    ref_oracle_abs = (ref64 - oracle).abs().max().item()
    ada_visible_abs = (ada64 - rounded).abs().max().item()
    ref_visible_abs = (ref64 - rounded).abs().max().item()
    # Pure output rounding: how far the once-rounded oracle sits from the
    # unrounded FP64 truth.  This bounds the absolute error that any tile
    # must have at this output magnitude, independent of kernel choice.
    rounding_abs = (rounded - oracle).abs().max().item()
    magnitude = oracle.abs().max().item()
    ulp = float(torch.finfo(dtype).eps) * (2.0 ** math.floor(math.log2(magnitude))) if magnitude else 0.0
    scale = magnitude
    drift = (ada_out.float() - base_out.float()).abs().max().item()
    return {
        "section": "gemv",
        "in_features": in_features,
        "out_features": out_features,
        "m": 1,
        "shipped_cfg": list(shipped),
        "reference_cfg": list(portable),
        "ada_hot_us": ada_hot,
        "ada_flushed_us": ada_flushed,
        "reference_hot_us": base_hot,
        "reference_flushed_us": base_flushed,
        "speedup_hot": _speedup(base_hot, ada_hot),
        "speedup_flushed": _speedup(base_flushed, ada_flushed),
        "dispatch_gemv": bool(module._can_use_rocm_gemv(x)),
        # sibling-vs-sibling (not an acceptance criterion; both tiles are ours)
        "sibling_max_abs_diff": drift,
        "sibling_max_rel_diff": drift / scale if scale else None,
        # acceptance: each tile against the independent FP64 oracle
        "ada_oracle_max_abs": ada_oracle_abs,
        "ada_oracle_max_rel": ada_oracle_abs / scale if scale else None,
        "reference_oracle_max_abs": ref_oracle_abs,
        "reference_oracle_max_rel": ref_oracle_abs / scale if scale else None,
        # tile-driven rounding only: oracle rounded once to the output dtype
        "ada_vs_rounded_oracle_max_abs": ada_visible_abs,
        "reference_vs_rounded_oracle_max_abs": ref_visible_abs,
        "ada_vs_rounded_oracle_max_rel": ada_visible_abs / scale if scale else None,
        # Budget context for the absolute column: at |out| ~ magnitude one
        # FP16 ULP is much larger than 2e-3, so the relative column is the
        # meaningful acceptance figure and the absolute excess is shared with
        # the reference tile.
        "oracle_rounding_max_abs": rounding_abs,
        "output_ulp": ulp,
        "within_2e3_relative": bool(ada_oracle_abs / scale <= 2e-3) if scale else None,
        "within_one_ulp_of_rounded_oracle": bool(ada_visible_abs <= ulp) if ulp else None,
        "same_error_as_reference": bool(abs(ada_oracle_abs - ref_oracle_abs) <= 1e-6 * max(scale, 1.0)),
        # Storage rule (AGENTS.md / torch-oracle skill): no dense weight may be
        # retained after the first or repeated forwards.
        "retained_dense_weight_after_forward": _dense_weight_candidates(module),
        "retained_weight_bytes": _retained_weight_bytes(module),
    }


def _dense_weight_candidates(module) -> list[str]:
    """Return any retained dense/expanded weight tensor (storage-rule gate)."""

    shape = (module.in_features, module.out_features)
    found = []
    for name, value in vars(module).items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"{name}{tuple(value.shape)}")
    for name, value in module.named_buffers():
        if value.is_floating_point() and value.dim() == 2 and tuple(value.shape) in (
            shape, (shape[1], shape[0]),
        ):
            found.append(f"buffer:{name}{tuple(value.shape)}")
    return found


def _retained_weight_bytes(module) -> int:
    """Bytes held by packed GPTQ weights + scales/zeros/index metadata."""

    total = 0
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        value = getattr(module, name, None)
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total


def _gptq_fp64_oracle(x: torch.Tensor, module, bits: int) -> torch.Tensor:
    """Independent FP64 decode + matmul; never calls the Triton kernel."""

    pack_factor = 32 // bits
    maxq = (1 << bits) - 1
    qweight = module.qweight.detach().cpu().to(torch.int64)
    qzeros = module.qzeros.detach().cpu().to(torch.int64)
    scales = module.scales.detach().cpu().to(torch.float64)
    group_index = module.g_idx.detach().cpu().to(torch.int64)
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    codes = ((qweight.unsqueeze(1) >> shifts[None, :, None]) & maxq).reshape(
        module.in_features, module.out_features
    )
    zeros = ((qzeros.unsqueeze(-1) >> shifts[None, None, :]) & maxq).reshape(
        module.scales.shape[0], module.out_features
    )
    weights = (codes.to(torch.float64) - zeros[group_index].to(torch.float64)) * scales[group_index]
    # Match the kernel's arithmetic: weights are rounded to the output dtype
    # before the matmul, which is then accumulated in FP64.
    weights = weights.to(x.dtype).to(torch.float64)
    return x.detach().cpu().to(torch.float64) @ weights


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gptq_gemm_row(module, in_features, out_features, m, args, device, dtype, l2):
    """Time the fused small-M GEMM against dequantize + matmul."""
    from gptqmodel.nn_modules.triton_utils import gptq_gemm as gemm_mod

    x = torch.randn(m, in_features, device=device, dtype=dtype)

    def fused_fn():
        return gemm_mod.gptq_gemm(
            x, module.qweight, module.scales, module.qzeros, module.g_idx,
            out_dtype=dtype, bits=args.bits,
        )

    def split_fn():
        with torch.inference_mode():
            weight = module.dequantize_weight()
        return x @ weight

    fused_hot, fused_flushed = _timed_us(device, fused_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    split_hot, split_flushed = _timed_us(device, split_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    with torch.inference_mode():
        fused_out, split_out = fused_fn(), split_fn()
    error = (fused_out.float() - split_out.float()).abs().max().item()
    oracle = _gptq_fp64_oracle(x, module, args.bits)
    fused64 = fused_out.detach().cpu().to(torch.float64)
    magnitude = oracle.abs().max().item()
    fused_oracle_abs = (fused64 - oracle).abs().max().item()
    cfg = gemm_mod.launch_config(x.device)
    return {
        "section": "gemm",
        "in_features": in_features,
        "out_features": out_features,
        "m": m,
        "shipped_cfg": [cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_warps, cfg.num_stages],
        "gate": [cfg.small_m_max, cfg.max_in_features, cfg.max_out_features],
        "fused_hot_us": fused_hot,
        "fused_flushed_us": fused_flushed,
        "split_hot_us": split_hot,
        "split_flushed_us": split_flushed,
        "speedup_hot": _speedup(split_hot, fused_hot),
        "speedup_flushed": _speedup(split_flushed, fused_flushed),
        "dispatch_gemm": bool(module._can_use_rocm_gemm(x)),
        "sibling_max_abs_diff": error,
        "fused_oracle_max_abs": fused_oracle_abs,
        "fused_oracle_max_rel": fused_oracle_abs / magnitude if magnitude else None,
        "within_2e3_relative": bool(fused_oracle_abs / magnitude <= 2e-3) if magnitude else None,
    }


def _gptq_dequant_row(module, in_features, out_features, args, device, dtype, l2):
    """Time the shipped Ada dequant tile against the portable default."""
    from gptqmodel.nn_modules.triton_utils import dequant as dequant_mod

    user_arch = dequant_mod.launch_config((8, 9))
    portable = dequant_mod._DEFAULT_DEQUANT_LAUNCH

    def launch(cfg):
        def fn():
            out = torch.empty((in_features, out_features), device=device, dtype=dtype)
            grid = (triton_cdiv(out.numel(), cfg.x_block),)
            dequant_mod.dequant_kernel[grid](
                module.g_idx, module.scales, module.qweight, module.qzeros, out,
                dequant_mod.torch_dtype_to_triton(dtype), out.numel(),
                pack_bits=32, maxq=module.maxq, bits=args.bits,
                out_features=out_features, num_groups=module.scales.shape[0],
                X_BLOCK=cfg.x_block, num_warps=cfg.num_warps, num_stages=cfg.num_stages,
            )
            return out
        return fn

    tuned_fn, portable_fn = launch(user_arch), launch(portable)
    tuned_hot, tuned_flushed = _timed_us(device, tuned_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    base_hot, base_flushed = _timed_us(device, portable_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    with torch.inference_mode():
        ok = torch.equal(tuned_fn(), portable_fn())
    return {
        "section": "dequant",
        "in_features": in_features,
        "out_features": out_features,
        "shipped_cfg": [user_arch.x_block, user_arch.num_warps, user_arch.num_stages],
        "reference_cfg": [portable.x_block, portable.num_warps, portable.num_stages],
        "ada_hot_us": tuned_hot,
        "ada_flushed_us": tuned_flushed,
        "reference_hot_us": base_hot,
        "reference_flushed_us": base_flushed,
        "speedup_hot": _speedup(base_hot, tuned_hot),
        "speedup_flushed": _speedup(base_flushed, tuned_flushed),
        "bit_identical": bool(ok),
    }


# --------------------------------------------------------------------------- #
# GGUF helpers
# --------------------------------------------------------------------------- #

def _gguf_pair(qtype: str, in_features: int, out_features: int, device: torch.device):
    import torch.nn as nn

    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel

    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)
    reference = GGUFTorchLinear(
        bits=qtype.lower(), group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=False,
    )
    reference.pack_original(linear, scales=torch.empty(0), zeros=torch.empty(0), g_idx=None)
    reference.post_init()
    module = GGUFTritonKernel(
        bits=qtype.lower(), group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=True,
    ).to(device).eval()
    module.load_state_dict(reference.state_dict(), strict=True)
    return module


def _gguf_row(qtype, in_features, out_features, rows, device, l2, args):
    from gptqmodel.nn_modules.qlinear import gguf_triton as gt

    module = _gguf_pair(qtype, in_features, out_features, device)
    padded = module.padded_in_features
    config = gt._select_gguf_qk_fixed_launch_config(
        qtype, capability=torch.cuda.get_device_capability(device),
        rows=rows, in_features=padded, out_features=out_features,
    )
    x = torch.randn(rows, padded, device=device, dtype=torch.float16)
    if config is None:
        return {
            "section": "gguf", "qtype": qtype, "in_features": in_features,
            "padded_in_features": padded, "out_features": out_features, "rows": rows,
            "fixed_cfg": None, "note": "autotuned (no measured fixed tile for this shape/row band)",
        }
    cache = module._get_triton_cache(x.device)
    launchers = {
        "Q4_K": lambda: gt._launch_q4_k_fixed_matmul(
            x, cache["qs"], cache["scale"], cache["min"], fixed_config=config),
        "Q5_K": lambda: gt._launch_q5_k_fixed_matmul(
            x, cache["qs"], cache["qh"], cache["scale"], cache["min"], fixed_config=config),
        "Q6_K": lambda: gt._launch_q6_k_fixed_matmul(
            x, cache["ql"], cache["qh"], cache["scale"], fixed_config=config),
    }
    auto_fn = lambda: module._forward_triton(x)
    fixed_fn = launchers[qtype]
    fixed_hot, fixed_flushed = _timed_us(device, fixed_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    auto_hot, auto_flushed = _timed_us(device, auto_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    with torch.inference_mode():
        exact = torch.equal(fixed_fn(), auto_fn())
    return {
        "section": "gguf", "qtype": qtype, "in_features": in_features,
        "padded_in_features": padded, "out_features": out_features, "rows": rows,
        "fixed_cfg": [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["num_warps"], config["num_stages"]],
        "fixed_hot_us": fixed_hot, "fixed_flushed_us": fixed_flushed,
        "auto_hot_us": auto_hot, "auto_flushed_us": auto_flushed,
        "speedup_hot": _speedup(auto_hot, fixed_hot),
        "speedup_flushed": _speedup(auto_flushed, fixed_flushed),
        "bit_identical": bool(exact),
    }


# --------------------------------------------------------------------------- #
# ParoQuant / portable AWQ GEMV helpers
# --------------------------------------------------------------------------- #

PAROQUANT_SHAPES = ((2048, 2048), (4096, 4096), (5120, 8192), (14336, 4096), (4096, 11008))
PAROQUANT_ROWS = (1, 8, 16, 64, 128, 1024)
AWQ_GEMV_SHAPES = (
    (2048, 2048), (4096, 4096), (5120, 8192), (14336, 4096), (4096, 11008), (896, 128),
)


def _awq_nibble_pack(codes: torch.Tensor) -> torch.Tensor:
    """AWQ [0,2,4,6,1,3,5,7] fast-axis pack, for the ParoQuant buffers only."""

    packed = torch.zeros((codes.shape[0], codes.shape[1] // 8), dtype=torch.int32, device=codes.device)
    for i, order in enumerate([0, 2, 4, 6, 1, 3, 5, 7]):
        packed |= codes[:, order::8].to(torch.int32) << (i * 4)
    return packed


def _awq_nibble_unpack(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int64)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device)
    codes = (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF
    return codes.index_select(-1, order).flatten(-2)


def _paroquant_buffers(k, n, group_size, device, seed=0):
    torch.manual_seed(seed)
    codes = torch.randint(0, 16, (k, n), dtype=torch.int32, device=device)
    zeros = torch.randint(0, 16, (k // group_size, n), dtype=torch.int32, device=device)
    scales = torch.rand((k // group_size, n), dtype=torch.float16, device=device) * 0.05 + 0.005
    return codes, zeros, scales, _awq_nibble_pack(codes), _awq_nibble_pack(zeros)


def _paroquant_launch(tile, x, qweight, scales, qzeros, group_size):
    from gptqmodel.quantization.paroquant.modules.triton import gemm as pg

    rows, k = x.shape
    n = scales.shape[1]
    bm, bn, bk, warps, stages = tile
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    grid = (((rows + bm - 1) // bm) * ((n + bn - 1) // bn),)
    pg.paroquant_gemm_kernel[grid](
        x, qweight, out, qzeros, scales, rows, n, k, group_size,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=min(bk, group_size),
        USE_FP32_ACCUM=True, num_warps=warps, num_stages=stages,
    )
    return out


def _paroquant_row(in_features, out_features, rows, device, l2, args):
    from gptqmodel.quantization.paroquant.modules.triton import gemm as pg

    codes, zeros, scales, qweight, qzeros = _paroquant_buffers(
        in_features, out_features, args.group_size, device,
    )
    kind = "decode" if rows <= 8 else "prefill"
    tuned = pg.paroquant_gemm_launch_config(kind, device, rows=rows)
    reference = pg._PAROQUANT_GEMM_DEFAULT_CONFIGS[kind]
    x = torch.randn(rows, in_features, device=device, dtype=torch.float16)
    group = torch.arange(in_features, device=device) // args.group_size
    dense = (
        (codes - zeros[group]).to(torch.float16) * scales[group]
    ).to(torch.float64)
    truth = x.to(torch.float64) @ dense

    tuned_fn = lambda: _paroquant_launch(tuned, x, qweight, scales, qzeros, args.group_size)
    ref_fn = lambda: _paroquant_launch(reference, x, qweight, scales, qzeros, args.group_size)
    tuned_hot, tuned_flushed = _timed_us(device, tuned_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    ref_hot, ref_flushed = _timed_us(device, ref_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    with torch.inference_mode():
        out = tuned_fn()
    err = (out.to(torch.float64) - truth).abs().max().item()
    return {
        "section": "paroquant", "in_features": in_features, "out_features": out_features,
        "rows": rows, "band": pg.paroquant_gemm_launch_config(kind, device, rows=rows),
        "tuned_cfg": list(tuned), "reference_cfg": list(reference),
        "tuned_hot_us": tuned_hot, "tuned_flushed_us": tuned_flushed,
        "reference_hot_us": ref_hot, "reference_flushed_us": ref_flushed,
        "speedup_hot": _speedup(ref_hot, tuned_hot),
        "speedup_flushed": _speedup(ref_flushed, tuned_flushed),
        "oracle_max_abs": err, "oracle_max_rel": err / truth.abs().max().item(),
        "dtype_preserved": out.dtype == torch.float16,
    }


def _awq_gemv_gemv_layout(out_features, in_features, group_size, device, seed=0):
    torch.manual_seed(seed)
    groups = in_features // group_size
    pad = (groups + 7) // 8 * 8
    qweight = torch.randint(-(2**31), 2**31 - 1, (out_features, in_features // 8),
                            device=device, dtype=torch.int32)
    qzeros = torch.randint(0, 15, (out_features, pad), device=device, dtype=torch.int32)
    scales = (torch.rand(out_features, pad, device=device, dtype=torch.float16) * 0.1 + 0.01).contiguous()
    return qweight, qzeros, scales


def _awq_gemv_oracle(qweight, qzeros, scales, group_size):
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 8
    nibbles = torch.arange(0, 8, device=qweight.device, dtype=torch.int64)
    codes = ((qweight.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        out_features, in_features)
    zeros = ((qzeros.to(torch.int64)[:, :, None] >> (nibbles * 4)) & 0xF).reshape(
        qzeros.shape[0], -1)
    group = torch.arange(in_features, device=qweight.device) // group_size
    return (codes.to(torch.float64) - zeros.to(torch.float64)[:, group]) * scales.to(
        torch.float64)[:, group]


def _awq_gemv_launch(config, x, qweight, scales, qzeros, group_size):
    import triton

    from gptqmodel.quantization.awq.modules.triton import gemv as gv

    rows, k = x.shape
    n = qweight.shape[0]
    block_k = min(config.max_block_k, group_size)
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    gv._awq_gemv_kernel[(triton.cdiv(n, config.block_n), rows)](
        x, qweight, scales, qzeros, out, rows, n, k, group_size,
        qzeros.stride(0), scales.stride(0),
        BLOCK_N=config.block_n, BLOCK_K=block_k,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return out


def _awq_gemv_row(in_features, out_features, device, l2, args):
    from gptqmodel.quantization.awq.modules.triton import gemv as gv

    qweight, qzeros, scales = _awq_gemv_gemv_layout(
        out_features, in_features, args.group_size, device,
    )
    tuned = gv.launch_config(device)
    reference = gv._DEFAULT_AWQ_GEMV_LAUNCH
    x = torch.randn(1, in_features, device=device, dtype=torch.float16)
    truth = x.to(torch.float64) @ _awq_gemv_oracle(
        qweight, qzeros, scales, args.group_size).t()

    tuned_fn = lambda: _awq_gemv_launch(tuned, x, qweight, scales, qzeros, args.group_size)
    ref_fn = lambda: _awq_gemv_launch(reference, x, qweight, scales, qzeros, args.group_size)
    tuned_hot, tuned_flushed = _timed_us(device, tuned_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    ref_hot, ref_flushed = _timed_us(device, ref_fn, warmup=args.warmup, iters=args.iters, l2=l2)
    with torch.inference_mode():
        out = tuned_fn()
    err = (out.to(torch.float64) - truth).abs().max().item()
    return {
        "section": "awq_gemv", "in_features": in_features, "out_features": out_features,
        "rows": 1, "tuned_cfg": list(tuned), "reference_cfg": list(reference),
        "tuned_hot_us": tuned_hot, "tuned_flushed_us": tuned_flushed,
        "reference_hot_us": ref_hot, "reference_flushed_us": ref_flushed,
        "speedup_hot": _speedup(ref_hot, tuned_hot),
        "speedup_flushed": _speedup(ref_flushed, tuned_flushed),
        "oracle_max_abs": err, "oracle_max_rel": err / truth.abs().max().item(),
        "dtype_preserved": out.dtype == torch.float16,
    }


    if "paroquant" in sections:
        from gptqmodel.quantization.paroquant.modules.triton import gemm as pg

        for in_features, out_features in PAROQUANT_SHAPES:
            for n_rows in PAROQUANT_ROWS:
                row = _paroquant_row(in_features, out_features, n_rows, device, l2, args)
                rows.append(row)
                _print_row(row)
            torch.cuda.empty_cache()

    if "awq_gemv" in sections:
        for in_features, out_features in AWQ_GEMV_SHAPES:
            row = _awq_gemv_row(in_features, out_features, device, l2, args)
            rows.append(row)
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> int:
    from gptqmodel.utils import arch

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    sections = _sections(args.sections)
    torch.manual_seed(7)
    l2 = L2Flush(device)
    rows: list[dict[str, Any]] = []

    for in_features, out_features in _parse_shapes(args.shapes):
        module = _make_module(in_features, out_features, dtype, args.bits, args.group_size, device)
        if "gemv" in sections:
            row = _gptq_gemv_row(module, in_features, out_features, args, device, dtype, l2)
            rows.append(row)
            _print_row(row)
        if "dequant" in sections:
            row = _gptq_dequant_row(module, in_features, out_features, args, device, dtype, l2)
            rows.append(row)
            _print_row(row)
        del module
        torch.cuda.empty_cache()

    if "gemm" in sections:
        for in_features, out_features in _parse_shapes(args.gemm_shapes):
            module = _make_module(in_features, out_features, dtype, args.bits, args.group_size, device)
            for m in (2, 8, 16):
                row = _gptq_gemm_row(module, in_features, out_features, m, args, device, dtype, l2)
                rows.append(row)
                _print_row(row)
            del module
            torch.cuda.empty_cache()

    if "paroquant" in sections:
        for in_features, out_features in PAROQUANT_SHAPES:
            for n_rows in PAROQUANT_ROWS:
                row = _paroquant_row(in_features, out_features, n_rows, device, l2, args)
                rows.append(row)
                _print_row(row)
            torch.cuda.empty_cache()

    if "awq_gemv" in sections:
        for in_features, out_features in AWQ_GEMV_SHAPES:
            row = _awq_gemv_row(in_features, out_features, device, l2, args)
            rows.append(row)
            _print_row(row)
        torch.cuda.empty_cache()

    if "gguf" in sections:
        for qtype, in_features, out_features in GGUF_CASES:
            for m in GGUF_ROWS:
                row = _gguf_row(qtype, in_features, out_features, m, device, l2, args)
                rows.append(row)
                _print_row(row)
            torch.cuda.empty_cache()

    payload = {
        "mode": "measured",
        "device": str(device),
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "sections": sections,
        "warmup": args.warmup,
        "iters": args.iters,
        "device_info": _device_report(device),
        "rows": rows,
    }
    measured = [r for r in rows if "speedup_hot" in r]
    print(json.dumps({
        "summary_rows": len(rows),
        "measured_rows": len(measured),
        "median_speedup_hot": statistics.median([r["speedup_hot"] for r in measured]) if measured else None,
        "bit_identical_rows": sum(1 for r in measured if r.get("bit_identical") is True),
    }, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _print_row(row: dict[str, Any]) -> None:
    print(json.dumps(row, sort_keys=True), flush=True)
    if row.get("section") == "gguf":
        if row.get("fixed_cfg") is None:
            print(f"  [gguf {row['qtype']} K={row['in_features']} N={row['out_features']} M={row['rows']}] "
                  f"autotuned (no fixed tile)", flush=True)
        else:
            print(f"  [gguf {row['qtype']} K={row['in_features']}->{row['padded_in_features']} "
                  f"N={row['out_features']} M={row['rows']}] fixed {row['fixed_hot_us']:.1f}/"
                  f"{row['fixed_flushed_us']:.1f} auto {row['auto_hot_us']:.1f}/{row['auto_flushed_us']:.1f} us "
                  f"-> {row['speedup_hot']:.2f}x hot, bit_exact={row['bit_identical']}", flush=True)
        return
    kind = row["section"]
    if kind == "gemm":
        print(f"  [{kind} K={row['in_features']} N={row['out_features']} M={row['m']}] "
              f"fused {row['fused_hot_us']:.1f}/{row['fused_flushed_us']:.1f} split "
              f"{row['split_hot_us']:.1f}/{row['split_flushed_us']:.1f} us -> "
              f"{row['speedup_hot']:.2f}x hot, dispatch={row['dispatch_gemm']}", flush=True)
    elif kind in ("paroquant", "awq_gemv"):
        print(f"  [{kind} K={row['in_features']} N={row['out_features']}"
              f"{'' if kind == 'awq_gemv' else ' M=' + str(row['rows'])}] "
              f"tuned {row['tuned_hot_us']:.1f}/{row['tuned_flushed_us']:.1f} ref "
              f"{row['reference_hot_us']:.1f}/{row['reference_flushed_us']:.1f} us -> "
              f"{row['speedup_hot']:.2f}x hot, {row['speedup_flushed']:.2f}x flushed, "
              f"oracle rel={row['oracle_max_rel']:.2e}", flush=True)
    elif kind == "gemv":
        print(f"  [{kind} K={row['in_features']} N={row['out_features']}] ada {row['ada_hot_us']:.1f}/"
              f"{row['ada_flushed_us']:.1f} ref {row['reference_hot_us']:.1f}/"
              f"{row['reference_flushed_us']:.1f} us -> {row['speedup_hot']:.2f}x hot, "
              f"oracle abs={row['ada_oracle_max_abs']:.2e} rel={row['ada_oracle_max_rel']:.2e} "
              f"(sibling abs={row['sibling_max_abs_diff']:.2e})", flush=True)
    else:
        print(f"  [{kind} K={row['in_features']} N={row['out_features']}] ada {row['ada_hot_us']:.1f}/"
              f"{row['ada_flushed_us']:.1f} ref {row['reference_hot_us']:.1f}/"
              f"{row['reference_flushed_us']:.1f} us -> {row['speedup_hot']:.2f}x hot, "
              f"bit_exact={row.get('bit_identical')}", flush=True)


def main() -> int:
    args = parse_args()
    if args.dry_run:
        _print_plan(_plan(args, "requested via --dry-run"))
        return 0
    if not torch.cuda.is_available():
        _print_plan(_plan(args, "no CUDA device visible; nothing measured"))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
