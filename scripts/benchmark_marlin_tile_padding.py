#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the runtime cost of 4-bit AWQ and GPTQ Marlin tile padding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


# This must be set before importing Torch or GPTQModel.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

BITS = 4
PACK_FACTOR = 32 // BITS
FULL_M_VALUES = (
    1,
    2,
    4,
    8,
    16,
    17,
    32,
    33,
    64,
    128,
    256,
    257,
    512,
    513,
    1023,
    1024,
    2048,
    4096,
    8192,
)
QUICK_M_VALUES = (1, 33, 512)


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    in_features: int
    out_features: int
    group_size: int


@dataclass(frozen=True)
class PhysicalGpu:
    index: int
    pci_bus_id: str
    uuid: str
    name: str
    memory_total_mib: int
    driver_version: str


@dataclass(frozen=True)
class IdleSample:
    sample: int
    utilization_percent: int
    memory_used_mib: int
    allowed_process_memory_mib: int
    residual_memory_mib: int


class BaselineValidationError(RuntimeError):
    """Raised when a fallback cannot reproduce the dense reference."""


class CandidateValidationError(RuntimeError):
    """Raised when padded Marlin cannot reproduce the dense reference."""


QUICK_CASES = (
    BenchCase("n_tail_small", in_features=256, out_features=200, group_size=32),
    BenchCase("k_tail_small", in_features=208, out_features=256, group_size=-1),
    BenchCase(
        "k_tail_native_gemm_small", in_features=160, out_features=256, group_size=32
    ),
    BenchCase("kn_tail_small", in_features=288, out_features=200, group_size=32),
)

DEFAULT_CASES = QUICK_CASES + (
    BenchCase("n_tail_4k", in_features=4096, out_features=4088, group_size=128),
    BenchCase("k_tail_4k", in_features=4128, out_features=4096, group_size=32),
    BenchCase("kn_tail_4k", in_features=4128, out_features=4088, group_size=32),
    BenchCase("n_tail_wide", in_features=4096, out_features=11000, group_size=128),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicit 4-bit Marlin tile padding against a correctness-checked format-matched fallback "
            "and the raw pre-padded Marlin kernel."
        )
    )
    parser.add_argument(
        "--quant-method",
        choices=("awq", "gptq", "both"),
        default="both",
        help="Packed format to benchmark. Both formats are measured independently by default.",
    )
    parser.add_argument(
        "--physical-gpu",
        default=None,
        help="Target physical GPU index, PCI bus id, or UUID. Defaults to a single CUDA_VISIBLE_DEVICES entry.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA ordinal inside the restricted visible set.",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument(
        "--baseline",
        choices=(
            "auto",
            "awq_gemm",
            "awq_gemm_triton",
            "awq_torch",
            "gptq_exllamav2",
            "gptq_triton",
            "gptq_torch_eager",
        ),
        default="auto",
        help="Format-matched fallback to compare. Auto tries production-like kernels before the safe Torch fallback.",
    )
    parser.add_argument(
        "--m-values", default=None, help="Comma-separated logical M values."
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        default=None,
        help="Optional JSON list of K/N/group-size cases.",
    )
    parser.add_argument(
        "--case-pattern",
        default=None,
        help="Optional regular expression applied to case ids.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the four small cases and reduced M matrix.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print selected cases without initializing CUDA.",
    )
    parser.add_argument(
        "--warmup", type=int, default=50, help="Warmup calls per variant and shape."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="CUDA-event samples per variant and round.",
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Independent A/B measurement rounds."
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--accuracy-rows", type=int, default=8)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=30.0,
        help="Maximum wait for the required consecutive idle samples.",
    )
    parser.add_argument(
        "--idle-memory-mib",
        type=int,
        default=32,
        help="Allowed non-process driver memory on the physical GPU.",
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _parse_int(value: str, field: str) -> int:
    cleaned = value.strip().replace(" MiB", "").replace("%", "")
    try:
        return int(cleaned)
    except ValueError as exc:
        raise RuntimeError(f"nvidia-smi returned invalid {field}: {value!r}") from exc


def _nvidia_smi_rows(query: str) -> list[list[str]]:
    command = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    # Some driver builds reject nvidia-smi when CUDA visibility is restricted.
    # Keep the restriction for Torch, but query the physical inventory separately.
    smi_env = os.environ.copy()
    smi_env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, env=smi_env
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"nvidia-smi query failed: {detail.strip()}") from exc
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row
    ]


def _source_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]

    def git_value(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def _query_gpu_inventory() -> list[dict[str, Any]]:
    fields = "gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu,driver_version"
    inventory = []
    for row in _nvidia_smi_rows(fields):
        if len(row) != 8:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {row}")
        inventory.append(
            {
                "index": _parse_int(row[0], "GPU index"),
                "pci_bus_id": row[1],
                "uuid": row[2],
                "name": row[3],
                "memory_total_mib": _parse_int(row[4], "total memory"),
                "memory_used_mib": _parse_int(row[5], "used memory"),
                "utilization_percent": _parse_int(row[6], "GPU utilization"),
                "driver_version": row[7],
            }
        )
    if not inventory:
        raise RuntimeError("nvidia-smi reported no GPUs.")
    return inventory


def _query_compute_processes() -> list[dict[str, Any]]:
    fields = "compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
    processes = []
    for row in _nvidia_smi_rows(fields):
        if len(row) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi compute-process row: {row}")
        used_memory = (
            0 if row[3] in {"N/A", "[N/A]"} else _parse_int(row[3], "process memory")
        )
        processes.append(
            {
                "gpu_uuid": row[0],
                "pid": _parse_int(row[1], "process pid"),
                "process_name": row[2],
                "used_memory_mib": used_memory,
            }
        )
    return processes


def _normalize_pci_bus_id(value: str) -> str:
    return value.strip().upper().removeprefix("00000000:")


def _normalize_uuid(value: str) -> str:
    return value.strip().upper().removeprefix("GPU-")


def _resolve_physical_gpu(selector: str | None) -> PhysicalGpu:
    if selector is None:
        visible = [
            item.strip()
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        ]
        if len(visible) != 1:
            raise RuntimeError(
                "Specify --physical-gpu or restrict CUDA_VISIBLE_DEVICES to exactly one physical GPU identifier."
            )
        selector = visible[0]

    inventory = _query_gpu_inventory()
    normalized_selector = selector.strip().upper()
    matches = [
        item
        for item in inventory
        if normalized_selector == str(item["index"])
        or normalized_selector == item["uuid"].upper()
        or _normalize_pci_bus_id(normalized_selector)
        == _normalize_pci_bus_id(item["pci_bus_id"])
    ]
    if len(matches) != 1:
        choices = ", ".join(f"{item['index']}:{item['uuid']}" for item in inventory)
        raise RuntimeError(
            f"Physical GPU selector {selector!r} matched {len(matches)} devices; available: {choices}"
        )
    item = matches[0]
    return PhysicalGpu(
        index=item["index"],
        pci_bus_id=item["pci_bus_id"],
        uuid=item["uuid"],
        name=item["name"],
        memory_total_mib=item["memory_total_mib"],
        driver_version=item["driver_version"],
    )


def _idle_gate(
    gpu: PhysicalGpu,
    *,
    samples: int,
    interval_seconds: float,
    timeout_seconds: float,
    residual_memory_limit_mib: int,
    allowed_pids: set[int] | None = None,
    phase: str,
) -> list[IdleSample]:
    if samples < 3:
        raise ValueError("--idle-samples must be at least 3.")
    if interval_seconds < 0:
        raise ValueError("--idle-interval-seconds must be non-negative.")
    if timeout_seconds <= 0:
        raise ValueError("--idle-timeout-seconds must be positive.")
    allowed_pids = allowed_pids or set()
    accepted = []
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_busy_reason = None
    while len(accepted) < samples:
        attempts += 1
        inventory = _query_gpu_inventory()
        current = next((item for item in inventory if item["uuid"] == gpu.uuid), None)
        if current is None:
            raise RuntimeError(
                f"Target GPU {gpu.uuid} disappeared during the {phase} idle gate."
            )
        processes = [
            item for item in _query_compute_processes() if item["gpu_uuid"] == gpu.uuid
        ]
        foreign = [item for item in processes if item["pid"] not in allowed_pids]
        if foreign:
            details = ", ".join(
                f"pid={item['pid']} name={item['process_name']}" for item in foreign
            )
            raise RuntimeError(
                f"{phase} idle gate rejected foreign compute processes on GPU {gpu.index}: {details}"
            )
        allowed_memory = sum(
            item["used_memory_mib"] for item in processes if item["pid"] in allowed_pids
        )
        residual_memory = max(0, current["memory_used_mib"] - allowed_memory)
        if (
            current["utilization_percent"] == 0
            and residual_memory <= residual_memory_limit_mib
        ):
            accepted.append(
                IdleSample(
                    sample=len(accepted) + 1,
                    utilization_percent=current["utilization_percent"],
                    memory_used_mib=current["memory_used_mib"],
                    allowed_process_memory_mib=allowed_memory,
                    residual_memory_mib=residual_memory,
                )
            )
            last_busy_reason = None
        else:
            # NVML can report the just-finished warmup briefly; require a fresh
            # run of consecutive idle samples instead of accepting stale state.
            accepted.clear()
            last_busy_reason = f"utilization={current['utilization_percent']}%, residual_memory={residual_memory} MiB"
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{phase} idle gate timed out after {timeout_seconds:.1f}s on GPU {gpu.index}: "
                    f"{last_busy_reason}."
                )
        if len(accepted) != samples:
            time.sleep(interval_seconds)

    print(
        f"idle_gate={phase} accepted physical_gpu={gpu.index} pci={gpu.pci_bus_id} uuid={gpu.uuid} "
        f"samples={samples} attempts={attempts} utilization=0% "
        f"residual_memory_limit_mib={residual_memory_limit_mib}"
    )
    return accepted


def _load_case_file(path: Path) -> list[BenchCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--case-file must contain a JSON list.")
    cases = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Case {index} must be an object.")
        in_features = item.get("in_features", item.get("k"))
        out_features = item.get("out_features", item.get("n"))
        cases.append(
            BenchCase(
                case_id=str(item.get("case_id", f"case_{index}")),
                in_features=int(in_features),
                out_features=int(out_features),
                group_size=int(item.get("group_size", 128)),
            )
        )
    return cases


def _is_tile_aligned(size_n: int, size_k: int) -> bool:
    return (size_n % 64 == 0 and size_k % 128 == 0) or (
        size_n % 128 == 0 and size_k % 64 == 0
    )


def _padded_nk(case: BenchCase) -> tuple[int, int]:
    # Channelwise requests remain one scale group after K padding.
    group = 1 if case.group_size in (-1, case.in_features) else case.group_size
    candidates = (
        (
            math.ceil(case.out_features / 64) * 64,
            math.ceil(case.in_features / math.lcm(128, group)) * math.lcm(128, group),
        ),
        (
            math.ceil(case.out_features / 128) * 128,
            math.ceil(case.in_features / math.lcm(64, group)) * math.lcm(64, group),
        ),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def _tail_kind(case: BenchCase, padded_n: int, padded_k: int) -> str:
    n_tail = padded_n != case.out_features
    k_tail = padded_k != case.in_features
    if n_tail and k_tail:
        return "K+N"
    if n_tail:
        return "N"
    if k_tail:
        return "K"
    return "aligned"


def _validate_case(case: BenchCase) -> None:
    if case.in_features <= 0 or case.out_features <= 0:
        raise ValueError(f"{case.case_id}: K and N must be positive.")
    if case.in_features % PACK_FACTOR or case.out_features % PACK_FACTOR:
        raise ValueError(
            f"{case.case_id}: K and N must be divisible by pack factor {PACK_FACTOR}."
        )
    effective_group_size = (
        case.in_features if case.group_size == -1 else case.group_size
    )
    if effective_group_size <= 0 or case.in_features % effective_group_size:
        raise ValueError(
            f"{case.case_id}: group_size={case.group_size} must divide K={case.in_features}."
        )
    padded_n, padded_k = _padded_nk(case)
    if not _is_tile_aligned(padded_n, padded_k):
        raise AssertionError(
            f"{case.case_id}: computed padded shape {(padded_k, padded_n)} is not tile aligned."
        )
    if (padded_n, padded_k) == (case.out_features, case.in_features):
        raise ValueError(
            f"{case.case_id}: this benchmark requires a tile-misaligned K or N shape."
        )


def _select_cases(args: argparse.Namespace) -> list[BenchCase]:
    cases = (
        _load_case_file(args.case_file)
        if args.case_file is not None
        else list(QUICK_CASES if args.quick else DEFAULT_CASES)
    )
    if args.case_pattern is not None:
        pattern = re.compile(args.case_pattern)
        cases = [case for case in cases if pattern.search(case.case_id)]
    if not cases:
        raise ValueError("No benchmark cases were selected.")
    seen = set()
    for case in cases:
        _validate_case(case)
        if case.case_id in seen:
            raise ValueError(f"Duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
    return cases


def _parse_m_values(args: argparse.Namespace) -> list[int]:
    values = (
        list(QUICK_M_VALUES if args.quick else FULL_M_VALUES)
        if args.m_values is None
        else [int(item.strip()) for item in args.m_values.split(",") if item.strip()]
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("M values must be positive.")
    return list(dict.fromkeys(values))


def _dtype_names(value: str) -> list[str]:
    return ["fp16", "bf16"] if value == "both" else [value]


def _quant_method_names(value: str) -> list[str]:
    return ["awq", "gptq"] if value == "both" else [value]


def _baseline_quant_method(name: str) -> str | None:
    if name == "auto":
        return None
    return name.split("_", 1)[0]


def _validate_baseline_request(
    args: argparse.Namespace, cases: list[BenchCase]
) -> None:
    methods = _quant_method_names(args.quant_method)
    requested_method = _baseline_quant_method(args.baseline)
    if requested_method is not None and methods != [requested_method]:
        raise ValueError(
            f"--baseline={args.baseline} only applies to --quant-method={requested_method}; "
            "use --baseline=auto when benchmarking both formats."
        )

    if args.baseline == "auto":
        return
    invalid = []
    for dtype_name in _dtype_names(args.dtype):
        for case in cases:
            reason = _baseline_rejection_reason(
                args.baseline, case, dtype_name=dtype_name
            )
            if reason is not None:
                invalid.append(f"{dtype_name}:{case.case_id} ({reason})")
    if invalid:
        raise ValueError(
            f"The requested {args.baseline} baseline does not support: "
            + ", ".join(invalid)
        )


def _pack_awq_tensor(unpacked: Any) -> Any:
    import torch

    if unpacked.ndim != 2 or unpacked.shape[1] % PACK_FACTOR:
        raise ValueError(
            f"AWQ tensor shape {tuple(unpacked.shape)} is not packable at {BITS} bits."
        )
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // PACK_FACTOR), dtype=torch.int32
    )
    for lane, source_lane in enumerate(order):
        packed.bitwise_or_(
            unpacked[:, source_lane::PACK_FACTOR].to(torch.int32) << (lane * BITS)
        )
    return packed


def _generate_awq_state(case: BenchCase, dtype: Any, seed: int) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    effective_group_size = (
        case.in_features if case.group_size == -1 else case.group_size
    )
    groups = case.in_features // effective_group_size
    codes = torch.randint(
        0,
        1 << BITS,
        (case.in_features, case.out_features),
        dtype=torch.int32,
        generator=generator,
    )
    zero_points = torch.randint(
        0,
        1 << BITS,
        (groups, case.out_features),
        dtype=torch.int32,
        generator=generator,
    )
    scales = (
        torch.rand(
            (groups, case.out_features), dtype=torch.float32, generator=generator
        )
        * 0.01
        + 0.001
    ).to(dtype)
    return {
        "codes": codes,
        "zero_points": zero_points,
        "qweight": _pack_awq_tensor(codes),
        "qzeros": _pack_awq_tensor(zero_points),
        "scales": scales,
        "effective_group_size": effective_group_size,
    }


def _pack_gptq_rows(unpacked: Any) -> Any:
    """Pack consecutive GPTQ K values into int32 rows."""
    import torch

    if unpacked.ndim != 2 or unpacked.shape[0] % PACK_FACTOR:
        raise ValueError(
            f"GPTQ qweight shape {tuple(unpacked.shape)} is not packable at {BITS} bits."
        )
    packed = torch.zeros(
        (unpacked.shape[0] // PACK_FACTOR, unpacked.shape[1]), dtype=torch.int64
    )
    for lane in range(PACK_FACTOR):
        packed.bitwise_or_(unpacked[lane::PACK_FACTOR].to(torch.int64) << (lane * BITS))
    return (packed & 0xFFFFFFFF).to(torch.int32)


def _pack_gptq_columns(unpacked: Any) -> Any:
    """Pack consecutive GPTQ zero points across output columns."""
    import torch

    if unpacked.ndim != 2 or unpacked.shape[1] % PACK_FACTOR:
        raise ValueError(
            f"GPTQ qzeros shape {tuple(unpacked.shape)} is not packable at {BITS} bits."
        )
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // PACK_FACTOR), dtype=torch.int64
    )
    for lane in range(PACK_FACTOR):
        packed.bitwise_or_(
            unpacked[:, lane::PACK_FACTOR].to(torch.int64) << (lane * BITS)
        )
    return (packed & 0xFFFFFFFF).to(torch.int32)


def _generate_gptq_state(case: BenchCase, dtype: Any, seed: int) -> dict[str, Any]:
    """Build a symmetric GPTQ-v2 state shared by Marlin and its fallback."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    effective_group_size = (
        case.in_features if case.group_size == -1 else case.group_size
    )
    groups = case.in_features // effective_group_size
    codes = torch.randint(
        0,
        1 << BITS,
        (case.in_features, case.out_features),
        dtype=torch.int32,
        generator=generator,
    )
    # uint4b8 is Marlin's symmetric 4-bit type, so logical zero is code 8.
    zero_points = torch.full(
        (groups, case.out_features), 1 << (BITS - 1), dtype=torch.int32
    )
    scales = (
        torch.rand(
            (groups, case.out_features), dtype=torch.float32, generator=generator
        )
        * 0.01
        + 0.001
    ).to(dtype)
    return {
        "codes": codes,
        "zero_points": zero_points,
        "qweight": _pack_gptq_rows(codes),
        "qzeros": _pack_gptq_columns(zero_points),
        "scales": scales,
        "g_idx": torch.arange(case.in_features, dtype=torch.int32)
        // effective_group_size,
        "effective_group_size": effective_group_size,
    }


def _native_awq_gemm_shape_supported(case: BenchCase) -> bool:
    """Mirror the launch checks in the CUDA AWQ GEMM implementation."""

    group_size = case.in_features if case.group_size == -1 else case.group_size
    return (
        case.out_features % 64 == 0
        and group_size % 32 == 0
        and case.out_features % group_size == 0
    )


def _gptq_vector_backend_shape_supported(case: BenchCase) -> bool:
    # ExLlama v2 and Triton require whole 32-value GPTQ packing blocks.
    return case.in_features % 32 == 0 and case.out_features % 32 == 0


def _baseline_rejection_reason(
    name: str, case: BenchCase, *, dtype_name: str
) -> str | None:
    if name == "awq_gemm" and not _native_awq_gemm_shape_supported(case):
        return "native CUDA launch requires N%64=0, group_size%32=0, and N%group_size=0"
    if name == "awq_gemm_triton" and dtype_name != "fp16":
        return "AWQ GEMM Triton declares FP16 support only"
    if name in (
        "gptq_exllamav2",
        "gptq_triton",
    ) and not _gptq_vector_backend_shape_supported(case):
        return "backend requires K and N divisible by 32"
    return None


def _baseline_candidates(
    requested: str, quant_method: str, case: BenchCase, dtype: Any
) -> list[str]:
    import torch

    if requested != "auto":
        return [requested]

    if quant_method == "awq":
        # Keep the native backend in the audit trail even when its launch rejects the shape.
        candidates = ["awq_gemm"]
        if dtype == torch.float16:
            candidates.append("awq_gemm_triton")
        candidates.append("awq_torch")
        return candidates

    # Every Machete-legal shape is already Marlin tile-aligned, so it cannot
    # serve as the fallback for this tail-only matrix.
    return ["gptq_exllamav2", "gptq_triton", "gptq_torch_eager"]


def _module_kwargs(case: BenchCase, dtype: Any, quant_method: str) -> dict[str, Any]:
    return {
        "bits": BITS,
        "group_size": case.group_size,
        "desc_act": False,
        "sym": quant_method == "gptq",
        "in_features": case.in_features,
        "out_features": case.out_features,
        "bias": False,
        "dtype": dtype,
        "register_buffers": True,
    }


def _new_awq_module(
    baseline_name: str,
    case: BenchCase,
    state: dict[str, Any],
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    from gptqmodel.utils.backend import BACKEND

    if baseline_name == "awq_gemm":
        from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear

        module_cls = AwqGEMMLinear
        backend = BACKEND.AWQ_GEMM
    elif baseline_name == "awq_gemm_triton":
        from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear

        module_cls = AwqGEMMTritonLinear
        backend = BACKEND.AWQ_GEMM_TRITON
    elif baseline_name == "awq_torch":
        from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

        module_cls = AwqTorchLinear
        backend = BACKEND.AWQ_TORCH
    else:  # pragma: no cover - argparse restricts this value
        raise ValueError(f"Unknown baseline: {baseline_name}")

    module = module_cls(**_module_kwargs(case, dtype, "awq"), backend=backend).to(
        device
    )
    with torch.no_grad():
        module.qweight.copy_(state["qweight"].to(device=device))
        module.qzeros.copy_(state["qzeros"].to(device=device))
        module.scales.copy_(state["scales"].to(device=device, dtype=dtype))
    module.eval()
    module.post_init()
    return module


def _new_gptq_module(
    baseline_name: str,
    case: BenchCase,
    state: dict[str, Any],
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    from gptqmodel.quantization import FORMAT
    from gptqmodel.utils.backend import BACKEND

    if baseline_name == "gptq_exllamav2":
        from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear

        module_cls = ExllamaV2Linear
        backend = BACKEND.GPTQ_EXLLAMA_V2
    elif baseline_name == "gptq_triton":
        from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

        module_cls = TritonV2Linear
        backend = BACKEND.GPTQ_TRITON
    elif baseline_name == "gptq_torch_eager":
        from gptqmodel.nn_modules.qlinear.torch import TorchLinear

        module_cls = TorchLinear
        backend = BACKEND.GPTQ_TORCH
    else:  # pragma: no cover - argparse restricts this value
        raise ValueError(f"Unknown baseline: {baseline_name}")

    module = module_cls(
        **_module_kwargs(case, dtype, "gptq"),
        backend=backend,
        format=FORMAT.GPTQ_V2,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(state["qweight"].to(device=device))
        module.qzeros.copy_(state["qzeros"].to(device=device))
        module.scales.copy_(
            state["scales"].to(device=device, dtype=module.scales.dtype)
        )
        module.g_idx.copy_(state["g_idx"].to(device=device))
    # The synthetic qzeros store their logical values directly.
    module.qzero_format(format=2)
    module.eval()
    if baseline_name == "gptq_torch_eager":
        # Keep the fallback bounded and free of compilation-worker state.
        module.optimize = lambda *args, **kwargs: None
    if baseline_name == "gptq_exllamav2":
        from gptqmodel.utils.exllamav2 import ScratchSpace

        scratch = ScratchSpace(module.temp_dq_size(), device)
        module.post_init(scratch)
        module._benchmark_scratch = scratch
    else:
        module.post_init()
    return module


def _new_baseline_module(
    quant_method: str,
    baseline_name: str,
    case: BenchCase,
    state: dict[str, Any],
    dtype: Any,
    device: Any,
) -> Any:
    if quant_method == "awq":
        return _new_awq_module(baseline_name, case, state, dtype, device)
    return _new_gptq_module(baseline_name, case, state, dtype, device)


def _build_candidate(
    quant_method: str,
    case: BenchCase,
    state: dict[str, Any],
    dtype: Any,
    device: Any,
) -> Any:
    import torch

    from gptqmodel.utils.backend import BACKEND

    format_kwargs = {}
    if quant_method == "awq":
        from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear

        module_cls = AwqMarlinLinear
        backend = BACKEND.AWQ_MARLIN
    else:
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
        from gptqmodel.quantization import FORMAT

        module_cls = MarlinLinear
        backend = BACKEND.GPTQ_MARLIN
        format_kwargs = {"format": FORMAT.GPTQ_V2}

    candidate = module_cls(
        **_module_kwargs(case, dtype, quant_method),
        backend=backend,
        **format_kwargs,
    ).to(device)
    with torch.no_grad():
        candidate.qweight.copy_(state["qweight"].to(device=device))
        candidate.qzeros.copy_(state["qzeros"].to(device=device))
        candidate.scales.copy_(
            state["scales"].to(device=device, dtype=candidate.scales.dtype)
        )
        if quant_method == "gptq":
            candidate.g_idx.copy_(state["g_idx"].to(device=device))
            candidate.qzero_format(format=2)
    candidate.eval()
    candidate.post_init()
    return candidate


def _variant_functions(
    quant_method: str, candidate: Any, baseline: Any, x: Any, case: BenchCase
) -> dict[str, Callable[[], Any]]:
    from gptqmodel.utils.marlin import (
        apply_awq_marlin_linear,
        apply_gptq_marlin_linear,
        marlin_pad_dim,
    )

    padded_n, padded_k = candidate._marlin_tile_padding
    x_padded = marlin_pad_dim(x, case.in_features, padded_k)

    def raw_padded_marlin() -> Any:
        common = {
            "input": x_padded,
            "weight": candidate.qweight,
            "weight_scale": candidate.scales,
            "weight_zp": candidate.qzeros,
            "g_idx": candidate.g_idx,
            "g_idx_sort_indices": candidate.g_idx_sort_indices,
            "workspace": candidate.workspace,
            "output_size_per_partition": padded_n,
            "input_size_per_partition": padded_k,
            "bias": None,
        }
        if quant_method == "awq":
            return apply_awq_marlin_linear(**common, quant_type=candidate.weight_type)
        return apply_gptq_marlin_linear(
            **common,
            wtype=candidate.weight_type,
            is_k_full=candidate.is_k_full,
            use_fp32_reduce=candidate.fp32,
            use_atomics=False,
        )

    return {
        "marlin_padded": lambda: candidate(x),
        "baseline": lambda: baseline(x),
        "marlin_raw_padded": raw_padded_marlin,
    }


def _dense_weight(
    state: dict[str, Any], case: BenchCase, dtype: Any, device: Any
) -> Any:
    import torch

    group_indices = (
        torch.arange(case.in_features, device=device) // state["effective_group_size"]
    )
    codes = state["codes"].to(device=device, dtype=dtype)
    zero_points = (
        state["zero_points"]
        .to(device=device, dtype=dtype)
        .index_select(0, group_indices)
    )
    scales = (
        state["scales"].to(device=device, dtype=dtype).index_select(0, group_indices)
    )
    return (codes - zero_points) * scales


def _check_accuracy(
    *,
    quant_method: str,
    candidate: Any,
    baseline: Any,
    dense_weight: Any,
    case: BenchCase,
    dtype: Any,
    device: Any,
    rows: int,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(
        (rows, case.in_features), dtype=dtype, device=device, generator=generator
    )
    with torch.inference_mode():
        expected = x @ dense_weight
    try:
        with torch.inference_mode():
            candidate_output = candidate(x)
            raw_output = _variant_functions(quant_method, candidate, baseline, x, case)[
                "marlin_raw_padded"
            ]()
            raw_output = raw_output[:, : case.out_features].contiguous()
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise CandidateValidationError(str(exc)) from exc
    try:
        torch.testing.assert_close(candidate_output, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(raw_output, expected, atol=atol, rtol=rtol)
    except (AssertionError, RuntimeError) as exc:
        raise CandidateValidationError(str(exc)) from exc

    try:
        with torch.inference_mode():
            baseline_output = baseline(x)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(baseline_output, expected, atol=atol, rtol=rtol)
    except (AssertionError, RuntimeError) as exc:
        raise BaselineValidationError(str(exc)) from exc

    def error(output: Any) -> tuple[float, float]:
        difference = (output - expected).abs().float()
        return difference.max().item(), difference.mean().item()

    candidate_max, candidate_mean = error(candidate_output)
    baseline_max, baseline_mean = error(baseline_output)
    raw_max, raw_mean = error(raw_output)
    return {
        "candidate_max_abs": candidate_max,
        "candidate_mean_abs": candidate_mean,
        "baseline_max_abs": baseline_max,
        "baseline_mean_abs": baseline_mean,
        "raw_max_abs": raw_max,
        "raw_mean_abs": raw_mean,
    }


def _check_m_boundary_accuracy(
    *,
    variants: dict[str, Callable[[], Any]],
    x: Any,
    dense_weight: Any,
    case: BenchCase,
    padded_n: int,
    device: Any,
    atol: float,
    rtol: float,
) -> dict[str, float | int]:
    """Validate M-specific dispatch with the first, middle, and last rows."""
    import torch

    row_ids = sorted({0, x.shape[0] // 2, x.shape[0] - 1})
    row_index = torch.tensor(row_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        expected = x.index_select(0, row_index) @ dense_weight

    try:
        with torch.inference_mode():
            candidate_output = variants["marlin_padded"]()
            if tuple(candidate_output.shape) != (x.shape[0], case.out_features):
                raise CandidateValidationError(
                    f"Marlin output shape {tuple(candidate_output.shape)} does not match "
                    f"{(x.shape[0], case.out_features)}."
                )
            candidate_rows = candidate_output.index_select(0, row_index)
            del candidate_output

            raw_output = variants["marlin_raw_padded"]()
            if tuple(raw_output.shape) != (x.shape[0], padded_n):
                raise CandidateValidationError(
                    f"Raw Marlin output shape {tuple(raw_output.shape)} does not match "
                    f"{(x.shape[0], padded_n)}."
                )
            raw_rows = raw_output.index_select(0, row_index)[:, : case.out_features]
            del raw_output
        torch.cuda.synchronize(device)
    except CandidateValidationError:
        raise
    except RuntimeError as exc:
        raise CandidateValidationError(str(exc)) from exc

    try:
        with torch.inference_mode():
            baseline_output = variants["baseline"]()
            if tuple(baseline_output.shape) != (x.shape[0], case.out_features):
                raise BaselineValidationError(
                    f"Baseline output shape {tuple(baseline_output.shape)} does not match "
                    f"{(x.shape[0], case.out_features)}."
                )
            baseline_rows = baseline_output.index_select(0, row_index)
            del baseline_output
        torch.cuda.synchronize(device)
    except BaselineValidationError:
        raise
    except RuntimeError as exc:
        raise BaselineValidationError(str(exc)) from exc

    try:
        torch.testing.assert_close(candidate_rows, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(raw_rows, expected, atol=atol, rtol=rtol)
    except (AssertionError, RuntimeError) as exc:
        raise CandidateValidationError(str(exc)) from exc
    try:
        torch.testing.assert_close(baseline_rows, expected, atol=atol, rtol=rtol)
    except (AssertionError, RuntimeError) as exc:
        raise BaselineValidationError(str(exc)) from exc

    def max_abs(output: Any) -> float:
        return (output - expected).abs().float().max().item()

    return {
        "checked_rows": len(row_ids),
        "candidate_max_abs": max_abs(candidate_rows),
        "baseline_max_abs": max_abs(baseline_rows),
        "raw_max_abs": max_abs(raw_rows),
    }


def _warmup_variants(
    variants: dict[str, Callable[[], Any]], warmup: int, device: Any
) -> None:
    import torch

    with torch.inference_mode():
        for name in sorted(variants):
            for _ in range(warmup):
                variants[name]()
    torch.cuda.synchronize(device)


def _measure_once(fn: Callable[[], Any], iters: int, device: Any) -> list[float]:
    import torch

    # Record on the selected device instead of whichever CUDA index is current.
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        with torch.inference_mode():
            for index in range(iters):
                starts[index].record(stream)
                fn()
                ends[index].record(stream)
    torch.cuda.synchronize(device)
    return [starts[index].elapsed_time(ends[index]) for index in range(iters)]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _bootstrap_lcb(values: list[float], samples: int, seed: int) -> float:
    if samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    if len(values) == 1:
        return values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(_geometric_mean(resampled))
    return _percentile(estimates, 0.05)


def _measure_variants(
    variants: dict[str, Callable[[], Any]],
    *,
    rounds: int,
    iters: int,
    device: Any,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    round_samples = {name: [] for name in variants}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            round_samples[name].append(_measure_once(variants[name], iters, device))

    stats = {}
    for name, samples_by_round in round_samples.items():
        flattened = [
            sample for round_values in samples_by_round for sample in round_values
        ]
        stats[name] = {
            "median_ms": statistics.median(flattened),
            "mean_ms": statistics.mean(flattened),
            "std_ms": statistics.stdev(flattened) if len(flattened) > 1 else 0.0,
            "p95_ms": _percentile(flattened, 0.95),
            "min_ms": min(flattened),
            "max_ms": max(flattened),
            "round_medians_ms": [
                statistics.median(values) for values in samples_by_round
            ],
            "sample_count": len(flattened),
        }

    paired_speedups = [
        baseline / candidate
        for baseline, candidate in zip(
            stats["baseline"]["round_medians_ms"],
            stats["marlin_padded"]["round_medians_ms"],
        )
    ]
    stats["speedup"] = (
        stats["baseline"]["median_ms"] / stats["marlin_padded"]["median_ms"]
    )
    stats["round_speedups"] = paired_speedups
    stats["speedup_geomean"] = _geometric_mean(paired_speedups)
    stats["speedup_lcb95"] = _bootstrap_lcb(
        paired_speedups, bootstrap_samples, seed + 1
    )
    stats["wrapper_overhead"] = (
        stats["marlin_padded"]["median_ms"] / stats["marlin_raw_padded"]["median_ms"]
    )
    return stats


def _print_progress(
    rows: list[dict[str, Any]], gpu: PhysicalGpu, *, final: bool
) -> None:
    from tabulate import tabulate

    table = []
    for row in rows:
        table.append(
            [
                gpu.index,
                row["quant_method"],
                row["case_id"],
                row["dtype"],
                row["m"],
                row["k"],
                row["n"],
                row.get("baseline_name", "pending"),
                "pending"
                if row.get("candidate_median_ms") is None
                else f"{row['candidate_median_ms']:.4f}",
                "pending"
                if row.get("baseline_median_ms") is None
                else f"{row['baseline_median_ms']:.4f}",
                "pending" if row.get("speedup") is None else f"{row['speedup']:.3f}x",
                "pending"
                if row.get("speedup_lcb95") is None
                else f"{row['speedup_lcb95']:.3f}x",
                row["state"],
            ]
        )
    print(f"\n{'Final results' if final else 'Live results'}")
    print(
        tabulate(
            table,
            headers=(
                "GPU",
                "method",
                "case",
                "dtype",
                "M",
                "K",
                "N",
                "baseline",
                "Marlin ms",
                "Baseline ms",
                "speedup",
                "LCB95",
                "state",
            ),
            tablefmt="grid",
        ),
        flush=True,
    )


def _run_benchmark(
    args: argparse.Namespace,
    cases: list[BenchCase],
    m_values: list[int],
    gpu: PhysicalGpu,
    initial_idle_samples: list[IdleSample],
) -> dict[str, Any]:
    import gptqmodel
    import torch

    from gptqmodel.utils.marlin import marlin_padded_nk

    script_repo = Path(__file__).resolve().parents[1]
    package_repo = Path(gptqmodel.__file__).resolve().parents[1]
    if package_repo != script_repo:
        raise RuntimeError(
            "The benchmark imported GPTQModel from a different source tree. "
            f"Set PYTHONPATH={script_repo} before running it."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable after the physical GPU passed the preflight."
        )
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA ordinal {args.device} is outside the visible device set."
        )
    device = torch.device(f"cuda:{args.device}")
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    quant_methods = _quant_method_names(args.quant_method)
    dtype_names = _dtype_names(args.dtype)
    minimum_capability = (8, 0) if "awq" in quant_methods else (7, 5)
    if capability < minimum_capability:
        raise RuntimeError(
            f"Selected Marlin formats require compute capability >= "
            f"{minimum_capability[0]}.{minimum_capability[1]}, got {capability}."
        )
    if capability == (7, 5) and "bf16" in dtype_names:
        raise RuntimeError("GPTQ Marlin on compute capability 7.5 supports FP16 only.")

    runtime_uuid = str(getattr(properties, "uuid", ""))
    if runtime_uuid and _normalize_uuid(runtime_uuid) != _normalize_uuid(gpu.uuid):
        raise RuntimeError(
            f"CUDA ordinal {args.device} resolved to {runtime_uuid}, expected physical GPU {gpu.uuid}."
        )

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    rows = []
    for quant_method in quant_methods:
        for dtype_name in dtype_names:
            for case in cases:
                for m in m_values:
                    rows.append(
                        {
                            "quant_method": quant_method,
                            "case_id": case.case_id,
                            "dtype": dtype_name,
                            "m": m,
                            "k": case.in_features,
                            "n": case.out_features,
                            "baseline_name": None,
                            "candidate_median_ms": None,
                            "baseline_median_ms": None,
                            "speedup": None,
                            "speedup_lcb95": None,
                            "state": "pending",
                        }
                    )

    print(
        f"runtime_device={device} name={properties.name} capability={capability[0]}.{capability[1]} "
        f"sms={properties.multi_processor_count} memory_mib={properties.total_memory // (1024 * 1024)}"
    )
    last_progress = time.monotonic()
    result_rows = []
    row_index = 0
    for method_index, quant_method in enumerate(quant_methods):
        for dtype_index, dtype_name in enumerate(dtype_names):
            dtype = dtype_map[dtype_name]
            for case_index, case in enumerate(cases):
                expected_padded_n, expected_padded_k = _padded_nk(case)
                repo_padded_n, repo_padded_k = marlin_padded_nk(
                    case.out_features, case.in_features, case.group_size
                )
                if (expected_padded_n, expected_padded_k) != (
                    repo_padded_n,
                    repo_padded_k,
                ):
                    raise AssertionError(
                        f"{case.case_id}: benchmark padding disagrees with GPTQModel: "
                        f"{(expected_padded_n, expected_padded_k)} != "
                        f"{(repo_padded_n, repo_padded_k)}"
                    )

                state_seed = (
                    args.seed
                    + method_index * 100_000
                    + dtype_index * 1_000
                    + case_index
                )
                state_builder = (
                    _generate_awq_state
                    if quant_method == "awq"
                    else _generate_gptq_state
                )
                state = state_builder(case, dtype, state_seed)
                candidate = _build_candidate(quant_method, case, state, dtype, device)
                dense_weight = _dense_weight(state, case, dtype, device)
                baseline = None
                baseline_name = None
                baseline_rejections = []
                for candidate_name in _baseline_candidates(
                    args.baseline, quant_method, case, dtype
                ):
                    reason = _baseline_rejection_reason(
                        candidate_name, case, dtype_name=dtype_name
                    )
                    if reason is not None:
                        if args.baseline != "auto":
                            raise RuntimeError(
                                f"{case.case_id}: requested {candidate_name} "
                                f"baseline is invalid: {reason}"
                            )
                        baseline_rejections.append(
                            {"name": candidate_name, "reason": reason}
                        )
                        continue

                    attempted_baseline = None
                    attempted_accuracy = None
                    attempted_m_accuracy = None
                    try:
                        attempted_baseline = _new_baseline_module(
                            quant_method,
                            candidate_name,
                            case,
                            state,
                            dtype,
                            device,
                        )
                        attempted_accuracy = _check_accuracy(
                            quant_method=quant_method,
                            candidate=candidate,
                            baseline=attempted_baseline,
                            dense_weight=dense_weight,
                            case=case,
                            dtype=dtype,
                            device=device,
                            rows=args.accuracy_rows,
                            seed=args.seed
                            + 10_000
                            + method_index * 100_000
                            + dtype_index * 1_000
                            + case_index,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        attempted_m_accuracy = {}
                        for m_index, m in enumerate(m_values):
                            generator = torch.Generator(device=device)
                            generator.manual_seed(
                                args.seed
                                + 20_000
                                + method_index * 100_000
                                + dtype_index * 10_000
                                + case_index * 100
                                + m_index
                            )
                            x = torch.randn(
                                (m, case.in_features),
                                dtype=dtype,
                                device=device,
                                generator=generator,
                            )
                            variants = _variant_functions(
                                quant_method,
                                candidate,
                                attempted_baseline,
                                x,
                                case,
                            )
                            try:
                                attempted_m_accuracy[str(m)] = (
                                    _check_m_boundary_accuracy(
                                        variants=variants,
                                        x=x,
                                        dense_weight=dense_weight,
                                        case=case,
                                        padded_n=repo_padded_n,
                                        device=device,
                                        atol=args.atol,
                                        rtol=args.rtol,
                                    )
                                )
                            finally:
                                del x, variants
                    except CandidateValidationError:
                        # A candidate failure cannot be repaired by changing baselines.
                        raise
                    except BaselineValidationError as exc:
                        if args.baseline != "auto":
                            raise
                        baseline_rejections.append(
                            {
                                "name": candidate_name,
                                "reason": f"dense-reference mismatch: {exc}",
                            }
                        )
                        attempted_baseline = None
                        torch.cuda.empty_cache()
                        continue
                    except Exception as exc:
                        if args.baseline != "auto":
                            raise
                        baseline_rejections.append(
                            {
                                "name": candidate_name,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        torch.cuda.empty_cache()
                        continue

                    assert (
                        attempted_baseline is not None
                        and attempted_accuracy is not None
                        and attempted_m_accuracy is not None
                    )
                    baseline = attempted_baseline
                    baseline_name = candidate_name
                    accuracy = attempted_accuracy
                    m_boundary_accuracy = attempted_m_accuracy
                    break

                if baseline is None or baseline_name is None:
                    reasons = "; ".join(
                        f"{item['name']}: {item['reason']}"
                        for item in baseline_rejections
                    )
                    raise RuntimeError(
                        f"{quant_method}:{case.case_id}: no correctness-checked "
                        f"baseline was available: {reasons}"
                    )
                print(
                    f"baseline_selection method={quant_method} dtype={dtype_name} "
                    f"case={case.case_id} baseline={baseline_name} "
                    f"rejected={len(baseline_rejections)}"
                )
                for pending_row in rows[row_index : row_index + len(m_values)]:
                    pending_row["baseline_name"] = baseline_name

                for m_index, m in enumerate(m_values):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        args.seed
                        + 20_000
                        + method_index * 100_000
                        + dtype_index * 10_000
                        + case_index * 100
                        + m_index
                    )
                    x = torch.randn(
                        (m, case.in_features),
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                    variants = _variant_functions(
                        quant_method, candidate, baseline, x, case
                    )
                    _warmup_variants(variants, args.warmup, device)
                    pre_timing_idle = _idle_gate(
                        gpu,
                        samples=args.idle_samples,
                        interval_seconds=args.idle_interval_seconds,
                        timeout_seconds=args.idle_timeout_seconds,
                        residual_memory_limit_mib=args.idle_memory_mib,
                        allowed_pids={os.getpid()},
                        phase=(
                            f"pre_timing:{quant_method}:{dtype_name}:"
                            f"{case.case_id}:M{m}"
                        ),
                    )
                    measured = _measure_variants(
                        variants,
                        rounds=args.rounds,
                        iters=args.iters,
                        device=device,
                        seed=args.seed
                        + 30_000
                        + method_index * 100_000
                        + dtype_index * 10_000
                        + case_index * 100
                        + m_index,
                        bootstrap_samples=args.bootstrap_samples,
                    )
                    gate_target = 1.0 if m == 1 else 1.05
                    gate_pass = measured["speedup_lcb95"] >= gate_target
                    element_size = torch.tensor([], dtype=dtype).element_size()
                    result = {
                        "quant_method": quant_method,
                        "case": asdict(case),
                        "dtype": dtype_name,
                        "m": m,
                        "baseline": baseline_name,
                        "baseline_rejections": baseline_rejections,
                        "padded_k": repo_padded_k,
                        "padded_n": repo_padded_n,
                        "tail_kind": _tail_kind(case, repo_padded_n, repo_padded_k),
                        "padding_work_ratio": (repo_padded_k * repo_padded_n)
                        / (case.in_features * case.out_features),
                        "input_pad_bytes": m
                        * (repo_padded_k - case.in_features)
                        * element_size,
                        "output_copy_bytes": m * case.out_features * element_size
                        if repo_padded_n != case.out_features
                        else 0,
                        "logical_tflops": 2.0
                        * m
                        * case.in_features
                        * case.out_features
                        / (measured["marlin_padded"]["median_ms"] * 1e9),
                        "accuracy": accuracy,
                        "m_boundary_accuracy": m_boundary_accuracy[str(m)],
                        "measurements": measured,
                        "local_gate_target": gate_target,
                        "local_gate_pass": gate_pass,
                        "pre_timing_idle_samples": [
                            asdict(sample) for sample in pre_timing_idle
                        ],
                    }
                    result_rows.append(result)
                    rows[row_index].update(
                        {
                            "candidate_median_ms": measured["marlin_padded"][
                                "median_ms"
                            ],
                            "baseline_median_ms": measured["baseline"]["median_ms"],
                            "speedup": measured["speedup"],
                            "speedup_lcb95": measured["speedup_lcb95"],
                            "state": "pass" if gate_pass else "hold",
                        }
                    )
                    row_index += 1
                    if (
                        time.monotonic() - last_progress
                        >= args.progress_interval_seconds
                    ):
                        _print_progress(rows, gpu, final=False)
                        last_progress = time.monotonic()
                    del x, variants
                del candidate, baseline, dense_weight, state
                torch.cuda.empty_cache()

    _print_progress(rows, gpu, final=True)
    shape_gates = []
    for quant_method in quant_methods:
        for dtype_name in dtype_names:
            for case in cases:
                matching = [
                    result
                    for result in result_rows
                    if result["quant_method"] == quant_method
                    and result["dtype"] == dtype_name
                    and result["case"]["case_id"] == case.case_id
                ]
                shape_gates.append(
                    {
                        "quant_method": quant_method,
                        "case_id": case.case_id,
                        "dtype": dtype_name,
                        "local_all_m_gate_pass": bool(matching)
                        and all(result["local_gate_pass"] for result in matching),
                        "minimum_lcb95": min(
                            result["measurements"]["speedup_lcb95"]
                            for result in matching
                        ),
                        "promotion_ready": False,
                        "promotion_blocker": "Requires matching evidence from additional GPU configurations.",
                    }
                )

    return {
        "schema_version": 3,
        "command": shlex.join(sys.argv),
        "source": {**_source_provenance(), "package_source_matches_script_repo": True},
        "hardware": {
            **asdict(gpu),
            "runtime_device": str(device),
            "runtime_uuid": runtime_uuid or None,
            "compute_capability": list(capability),
            "sm_count": properties.multi_processor_count,
            "torch_device_name": properties.name,
            "torch_total_memory_mib": properties.total_memory // (1024 * 1024),
        },
        "software": {
            "gptqmodel": gptqmodel.__version__,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "marlin_kernel_rebuild": os.environ.get("GPTQMODEL_KERNEL_REBUILD"),
        },
        "config": {
            "bits": BITS,
            "quant_methods": quant_methods,
            "desc_act": False,
            "sym_by_method": {"awq": False, "gptq": True},
            "requested_baseline": args.baseline,
            "baseline_policy": "first buildable fallback that passes the dense reference",
            "gptq_torch_fallback_mode": (
                "TorchLinear.optimize disabled; internal dequant dispatch unchanged"
            ),
            "warmup": args.warmup,
            "iters": args.iters,
            "rounds": args.rounds,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "m_values": m_values,
            "dtypes": dtype_names,
            "idle_timeout_seconds": args.idle_timeout_seconds,
        },
        "initial_idle_samples": [asdict(sample) for sample in initial_idle_samples],
        "results": result_rows,
        "shape_gates": shape_gates,
        "gate_policy": {
            "decode_m1_minimum_lcb95": 1.0,
            "other_m_minimum_lcb95": 1.05,
            "scope": "local run only; cross-GPU evidence is required before automatic routing",
        },
    }


def main() -> int:
    args = _parse_args()
    cases = _select_cases(args)
    m_values = _parse_m_values(args)
    if args.list_cases:
        for case in cases:
            padded_n, padded_k = _padded_nk(case)
            print(
                json.dumps(
                    {
                        **asdict(case),
                        "padded_k": padded_k,
                        "padded_n": padded_n,
                        "tail_kind": _tail_kind(case, padded_n, padded_k),
                    }
                )
            )
        return 0
    _validate_baseline_request(args, cases)
    if (
        args.warmup < 0
        or args.iters <= 0
        or args.rounds <= 0
        or args.accuracy_rows <= 0
    ):
        raise ValueError(
            "warmup must be non-negative; iters, rounds, and accuracy-rows must be positive."
        )

    gpu = _resolve_physical_gpu(args.physical_gpu)
    initial_idle = _idle_gate(
        gpu,
        samples=args.idle_samples,
        interval_seconds=args.idle_interval_seconds,
        timeout_seconds=args.idle_timeout_seconds,
        residual_memory_limit_mib=args.idle_memory_mib,
        phase="initial",
    )
    payload = _run_benchmark(args, cases, m_values, gpu, initial_idle)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json_out={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
