#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_idle_preflight import (  # noqa: E402
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)

_GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight() if __name__ == "__main__" else None

import torch  # noqa: E402

from gptqmodel import extension  # noqa: E402
from gptqmodel.utils import amplin  # noqa: E402
from scripts.benchmark_amplin_vs_marlin import (  # noqa: E402
    GROUP_SIZE,
    SIZE_K,
    _build_marlin,
    _dequantized_reference,
    _dtype_name,
    _git_revision,
    _make_case,
    _nvidia_smi_inventory,
    _raw_marlin_call,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit bounded NVTX and CUDA-profiler ranges for raw Amplin GEMV/HMMA and Marlin kernels."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--path",
        choices=(
            "amplin",
            "hmma_v0",
            "hmma_m64_v1",
            "hmma_m64_v2",
            "hmma_m64_v2_sync_a128",
            "hmma_m64_v3",
            "mma_lane_m64",
            "mma_lane_m64_global_a",
            "mma_lane_m32_global_a",
            "mma_lane_m32_n32_global_a",
            "n32_splitk12",
            "n32_splitk16",
            "n32_splitk12_pipe2",
            "n32_splitk16_pipe2",
            "n32_splitk12_pipe2_interleaved",
            "n64_splitk24_pipe2_interleaved",
            "n64_splitk12x2_coop_interleaved",
            "hmma",
            "marlin",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=SIZE_K)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket the measured ranges with cudaProfilerStart/Stop for profiler capture-range control.",
    )
    add_gpu_idle_preflight_args(parser)
    return parser.parse_args()


def _profile_paths(path: str) -> tuple[str, ...]:
    if path == "both":
        return ("amplin", "marlin")
    if path == "all":
        return (
            "amplin",
            "hmma_v0",
            "hmma_m64_v1",
            "hmma_m64_v2",
            "hmma_m64_v2_sync_a128",
            "hmma_m64_v3",
            "mma_lane_m64",
            "mma_lane_m64_global_a",
            "mma_lane_m32_global_a",
            "mma_lane_m32_n32_global_a",
            "n32_splitk12",
            "n32_splitk16",
            "n32_splitk12_pipe2",
            "n32_splitk16_pipe2",
            "n32_splitk12_pipe2_interleaved",
            "n64_splitk24_pipe2_interleaved",
            "n64_splitk12x2_coop_interleaved",
            "hmma",
            "marlin",
        )
    return (path,)


def _resolve_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _cuda_profiler_call(name: str) -> None:
    result = getattr(torch.cuda.cudart(), name)()
    if result not in (None, 0):
        raise RuntimeError(f"{name} failed with CUDA status {result}")


def _run_range(
    *,
    name: str,
    function: Callable[[], torch.Tensor],
    launches: int,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push(name)
    try:
        output = function()
        for _ in range(launches - 1):
            output = function()
    finally:
        torch.cuda.nvtx.range_pop()
    return output


def main() -> None:
    args = _parse_args()
    if args.m <= 0:
        raise ValueError("--m must be positive")
    if args.k <= 0 or args.k % GROUP_SIZE != 0:
        raise ValueError("--k must be positive and divisible by group size 128")
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.warmup < 1 or args.launches < 1:
        raise ValueError("--warmup and --launches must be positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Amplin and Marlin profiling requires a CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (8, 0):
        raise RuntimeError(
            f"Amplin V0 profiling requires compute capability 8.0, got {properties.major}.{properties.minor}"
        )

    dtype = _resolve_dtype(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested BF16 profile but the selected CUDA device does not support BF16")

    selected_paths = _profile_paths(args.path)
    if "marlin" in selected_paths and args.n % 64 != 0:
        raise ValueError("--n must be divisible by 64 when profiling Marlin")
    if (
        (
            "hmma" in selected_paths
            or "hmma_v0" in selected_paths
            or "hmma_m64_v1" in selected_paths
            or "hmma_m64_v2" in selected_paths
            or "hmma_m64_v2_sync_a128" in selected_paths
            or "hmma_m64_v3" in selected_paths
            or "mma_lane_m64" in selected_paths
            or "mma_lane_m64_global_a" in selected_paths
            or "mma_lane_m32_global_a" in selected_paths
        )
        and (args.m % 16 != 0 or args.n % amplin.HMMA_N_TILE != 0)
    ):
        raise ValueError("--m must be divisible by 16 and --n by 64 when profiling Amplin HMMA")
    if (
        "hmma_m64_v1" in selected_paths
        or "hmma_m64_v2" in selected_paths
        or "hmma_m64_v2_sync_a128" in selected_paths
        or "hmma_m64_v3" in selected_paths
        or "mma_lane_m64" in selected_paths
        or "mma_lane_m64_global_a" in selected_paths
    ) and args.m % 64 != 0:
        raise ValueError("--m must be divisible by 64 when profiling an Amplin HMMA M64 control")
    if "mma_lane_m32_global_a" in selected_paths and args.m % 32 != 0:
        raise ValueError("--m must be divisible by 32 when profiling the Amplin direct-A M32 control")
    if "mma_lane_m32_n32_global_a" in selected_paths and (
        args.m % 32 != 0 or args.n % 8 != 0
    ):
        raise ValueError("--m must be divisible by 32 and --n by 8 for the Amplin direct-A M32xN32 control")
    n32_splitk_paths = {
        "n32_splitk12",
        "n32_splitk16",
        "n32_splitk12_pipe2",
        "n32_splitk16_pipe2",
    }
    if n32_splitk_paths.intersection(selected_paths) and (
        args.m not in (2, 4, 8, 16) or args.k != 12288 or args.n % 32 != 0
    ):
        raise ValueError(
            "Amplin N32 split-K profiling requires M=2,4,8,16, K=12288, and N divisible by 32"
        )
    if "n32_splitk12_pipe2_interleaved" in selected_paths and (
        args.m not in (2, 4, 8, 16) or args.k != 12288 or args.n % 32 != 0
    ):
        raise ValueError(
            "Amplin interleaved N32 split-K profiling requires M=2,4,8,16, K=12288, and N divisible by 32"
        )
    n64_interleaved_paths = {
        "n64_splitk24_pipe2_interleaved",
        "n64_splitk12x2_coop_interleaved",
    }
    if n64_interleaved_paths.intersection(selected_paths) and (
        args.m not in (2, 4, 8, 16) or args.k != 12288 or args.n % 64 != 0
    ):
        raise ValueError(
            "Amplin interleaved N64 split-K profiling requires M=2,4,8,16, K=12288, and N divisible by 64"
        )
    hmma_schedules = {}
    if "hmma_v0" in selected_paths:
        hmma_schedules["hmma_v0"] = "m16-v0"
    if "hmma_m64_v1" in selected_paths:
        hmma_schedules["hmma_m64_v1"] = "m64-reuse-v1"
    if "hmma_m64_v2" in selected_paths:
        hmma_schedules["hmma_m64_v2"] = "m64-reuse-v2-wide-b"
    if "hmma_m64_v2_sync_a128" in selected_paths:
        hmma_schedules["hmma_m64_v2_sync_a128"] = "m64-reuse-v2-sync-a128"
    if "hmma_m64_v3" in selected_paths:
        hmma_schedules["hmma_m64_v3"] = "m64-reuse-v3-async-a"
    if "mma_lane_m64" in selected_paths:
        hmma_schedules["mma_lane_m64"] = "m64-register-b-v4"
    if "mma_lane_m64_global_a" in selected_paths:
        hmma_schedules["mma_lane_m64_global_a"] = "m64-register-a-b"
    if "mma_lane_m32_global_a" in selected_paths:
        hmma_schedules["mma_lane_m32_global_a"] = "m32-n64-register-a-b"
    if "mma_lane_m32_n32_global_a" in selected_paths:
        hmma_schedules["mma_lane_m32_n32_global_a"] = "m32-n32-register-a-b-tail"
    if "hmma" in selected_paths:
        m64_grid_ctas = (
            (args.n // amplin.HMMA_N_TILE) * (args.m // 64)
            if args.m % 64 == 0
            else 0
        )
        hmma_schedules["hmma"] = (
            "m64-reuse-v3-async-a"
            if m64_grid_ctas >= properties.multi_processor_count
            else "m16-v0"
        )
    input, canonical_qweight, canonical_scales = _make_case(
        device=device,
        dtype=dtype,
        size_m=args.m,
        size_k=args.k,
        size_n=args.n,
        seed=args.seed,
    )
    reference = _dequantized_reference(input, canonical_qweight, canonical_scales)
    functions: dict[str, Callable[[], torch.Tensor]] = {}

    if "amplin" in selected_paths:
        amplin_raw_op = extension.op("amplin", "gemv")
        functions["amplin"] = lambda: amplin_raw_op(input, canonical_qweight, canonical_scales)

    if (
        "hmma" in selected_paths
        or "hmma_v0" in selected_paths
        or "hmma_m64_v1" in selected_paths
        or "hmma_m64_v2" in selected_paths
        or "hmma_m64_v2_sync_a128" in selected_paths
        or "hmma_m64_v3" in selected_paths
    ):
        packed_hmma_qweight, packed_hmma_scales = amplin.pack_hmma_weights(
            canonical_qweight,
            canonical_scales,
        )

    if (
        "mma_lane_m64" in selected_paths
        or "mma_lane_m64_global_a" in selected_paths
        or "mma_lane_m32_global_a" in selected_paths
        or "mma_lane_m32_n32_global_a" in selected_paths
        or n32_splitk_paths.intersection(selected_paths)
    ):
        packed_mma_lane_qweight = amplin.pack_mma_lane_qweight(canonical_qweight)
        packed_hmma_scales = amplin.pack_hmma_scales(canonical_scales)

    if "n32_splitk12_pipe2_interleaved" in selected_paths:
        packed_mma_lane_n32_qweight = amplin.pack_mma_lane_n32_qweight(canonical_qweight)
        packed_hmma_scales = amplin.pack_hmma_scales(canonical_scales)

    if n64_interleaved_paths.intersection(selected_paths):
        packed_mma_lane_n64_qweight = amplin.pack_mma_lane_n64_qweight(canonical_qweight)
        packed_hmma_scales = amplin.pack_hmma_scales(canonical_scales)

    if "hmma_v0" in selected_paths:
        amplin_hmma_v0_raw_op = extension.op("amplin", "gemm_hmma_v0")
        functions["hmma_v0"] = lambda: amplin_hmma_v0_raw_op(
            input,
            packed_hmma_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "hmma_m64_v1" in selected_paths:
        amplin_hmma_m64_v1_raw_op = extension.op("amplin", "gemm_hmma_m64_v1")
        functions["hmma_m64_v1"] = lambda: amplin_hmma_m64_v1_raw_op(
            input,
            packed_hmma_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "hmma_m64_v2" in selected_paths:
        amplin_hmma_m64_v2_raw_op = extension.op("amplin", "gemm_hmma_m64_v2")
        functions["hmma_m64_v2"] = lambda: amplin_hmma_m64_v2_raw_op(
            input,
            packed_hmma_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "hmma_m64_v2_sync_a128" in selected_paths:
        amplin_hmma_m64_v2_sync_a128_raw_op = extension.op(
            "amplin",
            "gemm_hmma_m64_v2_sync_a128",
        )
        functions["hmma_m64_v2_sync_a128"] = lambda: amplin_hmma_m64_v2_sync_a128_raw_op(
            input,
            packed_hmma_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "hmma_m64_v3" in selected_paths:
        amplin_hmma_m64_v3_raw_op = extension.op("amplin", "gemm_hmma_m64_v3")
        functions["hmma_m64_v3"] = lambda: amplin_hmma_m64_v3_raw_op(
            input,
            packed_hmma_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "mma_lane_m64" in selected_paths:
        amplin_mma_lane_m64_raw_op = extension.op("amplin", "mma_lane_m64")
        functions["mma_lane_m64"] = lambda: amplin_mma_lane_m64_raw_op(
            input,
            packed_mma_lane_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "mma_lane_m64_global_a" in selected_paths:
        amplin_mma_lane_m64_global_a_raw_op = extension.op(
            "amplin",
            "mma_lane_m64_global_a",
        )
        functions["mma_lane_m64_global_a"] = lambda: amplin_mma_lane_m64_global_a_raw_op(
            input,
            packed_mma_lane_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "mma_lane_m32_global_a" in selected_paths:
        amplin_mma_lane_m32_global_a_raw_op = extension.op(
            "amplin",
            "mma_lane_m32_global_a",
        )
        functions["mma_lane_m32_global_a"] = lambda: amplin_mma_lane_m32_global_a_raw_op(
            input,
            packed_mma_lane_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "mma_lane_m32_n32_global_a" in selected_paths:
        amplin_mma_lane_m32_n32_global_a_raw_op = extension.op(
            "amplin",
            "mma_lane_m32_n32_global_a",
        )
        functions["mma_lane_m32_n32_global_a"] = (
            lambda: amplin_mma_lane_m32_n32_global_a_raw_op(
                input,
                packed_mma_lane_qweight,
                packed_hmma_scales,
                args.n,
            )
        )

    n32_splitk_ops = {
        "n32_splitk12": "mma_lane_m16_n32_splitk12",
        "n32_splitk16": "mma_lane_m16_n32_splitk16",
        "n32_splitk12_pipe2": "mma_lane_m16_n32_splitk12_pipe2",
        "n32_splitk16_pipe2": "mma_lane_m16_n32_splitk16_pipe2",
    }
    for path_name, op_name in n32_splitk_ops.items():
        if path_name in selected_paths:
            raw_op = extension.op("amplin", op_name)
            functions[path_name] = lambda raw_op=raw_op: raw_op(
                input,
                packed_mma_lane_qweight,
                packed_hmma_scales,
                args.n,
            )

    if "n32_splitk12_pipe2_interleaved" in selected_paths:
        n32_interleaved_op = extension.op("amplin", "mma_lane_m16_n32_splitk12_pipe2_interleaved")
        functions["n32_splitk12_pipe2_interleaved"] = lambda: n32_interleaved_op(
            input,
            packed_mma_lane_n32_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "n64_splitk24_pipe2_interleaved" in selected_paths:
        n64_interleaved_op = extension.op("amplin", "mma_lane_m16_n64_splitk24_pipe2_interleaved")
        functions["n64_splitk24_pipe2_interleaved"] = lambda: n64_interleaved_op(
            input,
            packed_mma_lane_n64_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "n64_splitk12x2_coop_interleaved" in selected_paths:
        n64_coop_interleaved_op = extension.op("amplin", "mma_lane_m16_n64_splitk12x2_coop_interleaved")
        functions["n64_splitk12x2_coop_interleaved"] = lambda: n64_coop_interleaved_op(
            input,
            packed_mma_lane_n64_qweight,
            packed_hmma_scales,
            args.n,
        )

    if "hmma" in selected_paths:
        amplin_hmma_raw_op = extension.op("amplin", "gemm_hmma")
        functions["hmma"] = lambda: amplin_hmma_raw_op(
            input,
            packed_hmma_qweight,
            packed_hmma_scales,
            args.n,
        )

    marlin_module = None
    if "marlin" in selected_paths:
        marlin_module = _build_marlin(
            device=device,
            dtype=dtype,
            qweight=canonical_qweight,
            scales=canonical_scales,
        )
        marlin_extension = "marlin_fp16" if dtype == torch.float16 else "marlin_bf16"
        marlin_op_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
        marlin_raw_op = extension.op(marlin_extension, marlin_op_name)
        functions["marlin"] = lambda: _raw_marlin_call(
            op=marlin_raw_op,
            input=input,
            module=marlin_module,
        )

    errors = {}
    outputs = {}
    error_limit = 2e-3 if dtype == torch.float16 else 2e-2
    pre_profile_exclusivity = None
    with torch.inference_mode():
        for name, function in functions.items():
            output = function()
            torch.cuda.synchronize(device)
            error = (output.to(torch.float32) - reference).abs()
            max_error = error.max().item()
            mean_error = error.mean().item()
            if not torch.isfinite(output).all() or max_error > error_limit:
                raise AssertionError(
                    f"{name} correctness failed: finite={bool(torch.isfinite(output).all())}, "
                    f"max_abs={max_error}, limit={error_limit}"
                )
            errors[name] = {"max_abs": max_error, "mean_abs": mean_error}

        del reference
        for function in functions.values():
            for _ in range(args.warmup):
                outputs["warmup"] = function()
        torch.cuda.synchronize(device)

        if _GPU_IDLE_PREFLIGHT is not None:
            pre_profile_exclusivity = recheck_gpu_exclusivity(_GPU_IDLE_PREFLIGHT)

        if args.cuda_profiler_api:
            _cuda_profiler_call("cudaProfilerStart")
        try:
            for name in selected_paths:
                range_name = f"{name}_raw_{_dtype_name(dtype)}_m{args.m}_k{args.k}_n{args.n}"
                outputs[name] = _run_range(
                    name=range_name,
                    function=functions[name],
                    launches=args.launches,
                )
                torch.cuda.synchronize(device)
        finally:
            if args.cuda_profiler_api:
                _cuda_profiler_call("cudaProfilerStop")

    hardware = {
        "device_argument": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_revision": _git_revision(),
        "nvidia_smi_inventory": _nvidia_smi_inventory(),
        "gpu_idle_preflight": (
            _GPU_IDLE_PREFLIGHT.as_dict() if _GPU_IDLE_PREFLIGHT is not None else None
        ),
        "pre_profile_exclusivity": pre_profile_exclusivity,
    }
    result = {
        "hardware": hardware,
        "profile": {
            "paths": selected_paths,
            "dtype": _dtype_name(dtype),
            "shape": {"m": args.m, "k": args.k, "n": args.n},
            "quantization": {
                "bits": 4,
                "group_size": GROUP_SIZE,
                "sym": True,
                "desc_act": False,
                "pack_dtype": "torch.int32",
            },
            "warmup": args.warmup,
            "launches_per_range": args.launches,
            "cuda_profiler_api": args.cuda_profiler_api,
            "expected_hmma_schedules": hmma_schedules,
            "errors_vs_fp32_dequant": errors,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
