#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare QVQ LR low-rate kernels with W4 GPTQ kernels on Llama 3.2 1B shapes.

This is a kernel-throughput comparison, not a model-quality comparison. QVQ
V2B2-P32-LR uses its native P32 local-ring layout at W2 through W3.5. Marlin
and Machete use the same synthetic symmetric W4 GPTQ group-128 payload, which
matches the common Llama 3.2 1B GPTQ configuration supported by both kernels.
Each candidate is checked against its own dense dequantized reference before
it is timed.

The default target is physical PCI-ordered GPU 1, the H100 on the development
host used by PR #62. The process performs its strict idle gate before importing
Torch and pins CUDA visibility to the accepted GPU UUID.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import statistics
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MAX_JOBS", "8")
os.environ.setdefault("NINJAFLAGS", "-j8")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "8")
os.environ.setdefault("NVCC_THREADS", "2")

from scripts import benchmark_qvq_cuda_lr as benchmark_utils

DEFAULT_QVQ_BITS = (2.0, 2.5, 3.0, 3.5)
DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32)
GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128


@dataclass(frozen=True)
class Projection:
    name: str
    in_features: int
    out_features: int


@dataclass(frozen=True)
class ShapeCase:
    name: str
    roles: tuple[str, ...]
    in_features: int
    out_features: int
    modules_per_layer: int


LLAMA32_1B_PROJECTIONS = (
    Projection("q_proj", 2048, 2048),
    Projection("k_proj", 2048, 512),
    Projection("v_proj", 2048, 512),
    Projection("o_proj", 2048, 2048),
    Projection("gate_proj", 2048, 8192),
    Projection("up_proj", 2048, 8192),
    Projection("down_proj", 8192, 2048),
)

# Duplicate Q/O, K/V, and gate/up geometries are timed once and retain every
# role in their labels. This measures all distinct M/K/N work in one decoder
# layer without pretending duplicate names are independent samples.
LLAMA32_1B_SHAPES = (
    ShapeCase("attn_qo", ("q_proj", "o_proj"), 2048, 2048, 2),
    ShapeCase("attn_kv", ("k_proj", "v_proj"), 2048, 512, 2),
    ShapeCase("mlp_gate_up", ("gate_proj", "up_proj"), 2048, 8192, 2),
    ShapeCase("mlp_down", ("down_proj",), 8192, 2048, 1),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=1,
        help="Physical PCI-ordered GPU index. PR #62's H100 host target is 1.",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(case.name for case in LLAMA32_1B_SHAPES),
        default=[case.name for case in LLAMA32_1B_SHAPES],
    )
    parser.add_argument("--m", nargs="+", type=_positive_int, default=list(DEFAULT_M_VALUES))
    parser.add_argument("--qvq-bits", nargs="+", type=float, default=list(DEFAULT_QVQ_BITS))
    parser.add_argument(
        "--gptq-group-size",
        type=int,
        default=GPTQ_GROUP_SIZE,
        help="Common W4 GPTQ group size for Marlin and Machete; Machete does not support group 32.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=60)
    parser.add_argument("--idle-samples", type=_positive_int, default=3)
    parser.add_argument("--idle-interval", type=_non_negative_float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=8)
    parser.add_argument(
        "--progress-interval",
        type=_non_negative_float,
        default=60.0,
        help="Seconds between complete live result tables; 0 disables periodic tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qvq_lr_vs_gptq_llama32_1b_h100.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Markdown result table path (defaults to --output with a .md suffix).",
    )
    args = parser.parse_args()
    _validate_contract(
        shapes=[_shape_by_name(name) for name in args.shapes],
        m_values=args.m,
        qvq_bits=args.qvq_bits,
        gptq_group_size=args.gptq_group_size,
    )
    if args.physical_gpu < 0:
        parser.error("--physical-gpu must be non-negative")
    if args.idle_memory_tolerance_mib < 0:
        parser.error("--idle-memory-tolerance-mib must be non-negative")
    return args


def _shape_by_name(name: str) -> ShapeCase:
    return next(case for case in LLAMA32_1B_SHAPES if case.name == name)


def _validate_contract(
    *,
    shapes: list[ShapeCase],
    m_values: list[int] | tuple[int, ...],
    qvq_bits: list[float] | tuple[float, ...],
    gptq_group_size: int,
) -> None:
    if not shapes:
        raise ValueError("at least one Llama 3.2 1B shape is required")
    if not m_values or any(m <= 0 for m in m_values) or len(set(m_values)) != len(m_values):
        raise ValueError("M values must be unique positive integers")
    normalized_bits = tuple(float(bits) for bits in qvq_bits)
    if not normalized_bits or len(set(normalized_bits)) != len(normalized_bits):
        raise ValueError("QVQ rates must be non-empty and unique")
    unsupported_bits = sorted(set(normalized_bits) - set(DEFAULT_QVQ_BITS))
    if unsupported_bits:
        raise ValueError(f"this comparison supports QVQ W2-W3.5 only, got {unsupported_bits}")
    if gptq_group_size not in (-1, 64, 128):
        raise ValueError(
            "the common Marlin/Machete W4 baseline requires group size -1, 64, or 128; "
            f"got {gptq_group_size}"
        )
    for case in shapes:
        k, n = case.in_features, case.out_features
        if k % 32 or n % 8:
            raise ValueError(f"QVQ LR requires K32/N8 geometry, got {case.name}: K={k}, N={n}")
        if k % 64 or n % 128:
            raise ValueError(f"Machete requires K64/N128 geometry, got {case.name}: K={k}, N={n}")
        if gptq_group_size != -1 and k % gptq_group_size:
            raise ValueError(f"GPTQ group size {gptq_group_size} does not divide {case.name} K={k}")


def _args_payload(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _benchmark_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _visible_gpu_matches(torch, hardware: dict[str, str]) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"benchmark requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    torch_uuid = str(getattr(properties, "uuid", "")).removeprefix("GPU-")
    expected_uuid = hardware["uuid"].removeprefix("GPU-")
    if torch_uuid != expected_uuid:
        raise RuntimeError(
            "visible GPU mapping mismatch: "
            f"expected physical={hardware['index']} uuid={hardware['uuid']}, "
            f"got torch_uuid=GPU-{torch_uuid} visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
        )


def _compute_processes_for_uuid(gpu_uuid: str) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "No running processes found" not in result.stderr:
        raise RuntimeError(f"failed to query GPU compute processes: {result.stderr.strip()}")
    processes = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", maxsplit=2)]
        if len(values) == 3 and values[1] == gpu_uuid:
            processes.append({"pid": values[0], "gpu_uuid": values[1], "process_name": values[2]})
    return processes


def _pre_timing_exclusivity_gate(
    *,
    physical_gpu: int,
    gpu_uuid: str,
    samples: int,
    interval: float,
) -> dict[str, str]:
    accepted = None
    idle_samples = 0
    attempts = 0
    max_attempts = samples * 10
    while idle_samples < samples and attempts < max_attempts:
        attempts += 1
        processes = _compute_processes_for_uuid(gpu_uuid)
        foreign = [process for process in processes if int(process["pid"]) != os.getpid()]
        if foreign:
            raise RuntimeError(f"physical GPU {physical_gpu} has foreign compute processes: {foreign}")
        accepted = benchmark_utils._query_gpu(physical_gpu)
        if int(accepted["utilization.gpu"]) == 0:
            idle_samples += 1
        else:
            idle_samples = 0
        if idle_samples < samples and attempts < max_attempts:
            threading.Event().wait(interval)
    if accepted is None or idle_samples < samples:
        utilization = accepted["utilization.gpu"] if accepted is not None else "unknown"
        raise RuntimeError(
            f"physical GPU {physical_gpu} failed pre-timing idle gate after {attempts} samples: "
            f"utilization={utilization}%"
        )
    print(
        "pre-timing gate: "
        f"physical={physical_gpu} pci={accepted['pci.bus_id']} uuid={gpu_uuid} "
        f"memory={accepted['memory.used']}MiB utilization={accepted['utilization.gpu']}% "
        f"consecutive_idle_samples={idle_samples} attempts={attempts} allowed_compute_pid={os.getpid()}",
        flush=True,
    )
    return accepted


def _metrics(actual, reference) -> dict[str, float | bool]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    delta = actual_fp32 - reference_fp32
    return {
        "finite": bool(torch_isfinite_all(actual_fp32)),
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
        "rel_l2": (delta.square().sum() / reference_fp32.square().sum().clamp_min(1e-12)).sqrt().item(),
    }


def torch_isfinite_all(tensor) -> bool:
    # Kept as a tiny seam so host-only unit tests can import this script without
    # importing Torch or initializing CUDA during collection.
    return tensor.isfinite().all().item()


def _assert_correct(
    *,
    kernel: str,
    actual,
    reference,
    expected_shape: tuple[int, int],
    expected_dtype,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    if tuple(actual.shape) != expected_shape:
        raise AssertionError(f"{kernel} returned shape {tuple(actual.shape)}, expected {expected_shape}")
    if actual.dtype != expected_dtype:
        raise AssertionError(f"{kernel} returned dtype {actual.dtype}, expected {expected_dtype}")
    metrics = _metrics(actual, reference)
    if not metrics["finite"]:
        raise AssertionError(f"{kernel} returned non-finite output")
    allowed = atol + rtol * reference.float().abs()
    if bool(((actual.float() - reference.float()).abs() > allowed).any().item()):
        raise AssertionError(
            f"{kernel} failed dense-reference gate atol={atol} rtol={rtol}: {metrics}"
        )
    return metrics


def _tensor_bytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _gptq_source(torch, case: ShapeCase, *, group_size: int, seed: int, dtype):
    groups = 1 if group_size == -1 else case.in_features // group_size
    generator = torch.Generator().manual_seed(seed)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (case.in_features // 8, case.out_features),
        dtype=torch.int32,
        generator=generator,
    )
    scales = (torch.rand((groups, case.out_features), dtype=torch.float32, generator=generator) * 0.005 + 0.005).to(
        dtype
    )
    effective_group_size = case.in_features if group_size == -1 else group_size
    g_idx = torch.arange(case.in_features, dtype=torch.int32) // effective_group_size
    qzeros = torch.zeros((groups, case.out_features // 8), dtype=torch.int32)
    return {"qweight": qweight, "scales": scales, "g_idx": g_idx, "qzeros": qzeros}


def _dense_gptq_weight(torch, source: dict, *, device, weight_type):
    from gptqmodel.utils.machete import unpack_quantized_values_into_int32

    codes = unpack_quantized_values_into_int32(source["qweight"].to(device), weight_type, packed_dim=0)
    logical = codes.float().sub_(float(weight_type.bias))
    scales = source["scales"].to(device=device, dtype=torch.float32)
    g_idx = source["g_idx"].to(device=device, dtype=torch.long)
    return logical.mul_(scales[g_idx])


def _build_gptq_module(torch, kernel: str, case: ShapeCase, *, group_size: int, source: dict, device, dtype):
    if kernel == "gptq_marlin":
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

        cls = MarlinLinear
    elif kernel == "gptq_machete":
        from gptqmodel.nn_modules.qlinear.machete import MacheteLinear

        cls = MacheteLinear
    else:
        raise ValueError(f"unknown GPTQ kernel {kernel!r}")

    module = cls(
        bits=GPTQ_BITS,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=False,
        dtype=dtype,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(source["qweight"].to(device))
        module.scales.copy_(source["scales"].to(device=device, dtype=module.scales.dtype))
        module.g_idx.copy_(source["g_idx"].to(device))
        module.qzeros.copy_(source["qzeros"].to(device))
    module.post_init()
    module.eval()
    payload_bytes = _tensor_bytes(module.qweight) + _tensor_bytes(module.scales)
    if getattr(module, "qzeros", None) is not None:
        payload_bytes += _tensor_bytes(module.qzeros)
    if getattr(module, "g_idx", None) is not None:
        payload_bytes += _tensor_bytes(module.g_idx)
    return module, payload_bytes


def _qvq_payload(torch, case: ShapeCase, *, bits: float, seed: int, device):
    from gptqmodel.quantization.qvq import (
        QVQ_V2B2_P32_LR_RING_STEPS,
        QVQ_V2B2_P32_LR_RINGS_PER_TILE,
        local_ring_states_from_edges,
        pack_local_ring_states,
        pack_qvq_binary_bank_ids,
        reconstruct_local_ring_inner_weight,
    )
    from gptqmodel.quantization.qvq_rates import qvq_transition_bits

    generator = torch.Generator().manual_seed(seed)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    tiles = (case.in_features // 32) * (case.out_features // 8)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    states = local_ring_states_from_edges(edges, bits=bits)
    trellis_cpu = pack_local_ring_states(states, bits=bits)
    selectors = torch.randint(
        0,
        2,
        (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,),
        generator=generator,
        dtype=torch.uint8,
    )
    bank_ids_cpu = pack_qvq_binary_bank_ids(selectors)
    dense = reconstruct_local_ring_inner_weight(
        trellis_cpu,
        bits=bits,
        in_features=case.in_features,
        out_features=case.out_features,
        bank_ids=bank_ids_cpu,
        bank_alt_id=torch.tensor([3], dtype=torch.uint8),
    ).to(device)
    payload_bytes = _tensor_bytes(trellis_cpu) + _tensor_bytes(bank_ids_cpu)
    return trellis_cpu.to(device), bank_ids_cpu.to(device), dense, payload_bytes


def _row_from_timing(
    *,
    case: ShapeCase,
    m: int,
    kernel: str,
    bits: float,
    group_size: int | None,
    packing: str,
    dtype_name: str,
    output_dtype: str,
    payload_bytes: int,
    metrics: dict,
    timing: dict,
) -> dict:
    logical_flops = 2 * m * case.in_features * case.out_features
    median_ms = timing["median_ms"]
    return {
        **asdict(case),
        "roles": list(case.roles),
        "m": m,
        "k": case.in_features,
        "n": case.out_features,
        "kernel": kernel,
        "bits": bits,
        "group_size": group_size,
        "packing": packing,
        "input_dtype": dtype_name,
        "output_dtype": output_dtype,
        "payload_bytes": payload_bytes,
        "payload_mib": payload_bytes / (1024**2),
        "logical_tflops": logical_flops / (median_ms * 1e9),
        "effective_payload_gbs": payload_bytes / (median_ms * 1e6),
        "state": "complete",
        **metrics,
        **timing,
    }


def _cuda_graph_event_timing(torch, fn, *, warmup: int, iterations: int) -> dict[str, float]:
    """Measure device work without host launch gaps in the timed intervals.

    All event/kernel/event triplets are captured into one CUDA Graph.  The
    graph is then submitted with a single host call, so a CPU-starved runner
    cannot delay a kernel after its start timestamp has reached the GPU.
    External events become explicit graph nodes and retain per-launch timing.
    """

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iterations)]
    graph = torch.cuda.CUDAGraph()
    captured_output = None
    with torch.cuda.graph(graph):
        for start, end in zip(starts, ends, strict=True):
            start.record()
            captured_output = fn()
            end.record()

    graph.replay()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    # Keep graph-owned output storage alive through replay and timestamp reads.
    del captured_output
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def _row_key(row: dict) -> tuple:
    return row["name"], row["m"], row["kernel"], float(row["bits"])


def _with_relative_metrics(rows: list[dict]) -> list[dict]:
    baselines: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        if row.get("state") != "complete":
            continue
        if row.get("kernel") in ("gptq_marlin", "gptq_machete"):
            baselines.setdefault((row["name"], row["m"]), {})[row["kernel"]] = row

    annotated = []
    for source in rows:
        row = dict(source)
        matched = baselines.get((row["name"], row["m"]), {})
        if row.get("state") == "complete":
            for baseline_name, output_name in (
                ("gptq_marlin", "speedup_vs_marlin"),
                ("gptq_machete", "speedup_vs_machete"),
            ):
                baseline = matched.get(baseline_name)
                if baseline is not None:
                    row[output_name] = baseline["median_ms"] / row["median_ms"]
        annotated.append(row)
    return annotated


def _expected_rows(shapes: list[ShapeCase], m_values: list[int], qvq_bits: list[float], dtype_name: str) -> list[dict]:
    rows = []
    for case in shapes:
        for m in m_values:
            common = {
                **asdict(case),
                "roles": list(case.roles),
                "m": m,
                "k": case.in_features,
                "n": case.out_features,
                "input_dtype": dtype_name,
                "state": "pending",
            }
            for bits in qvq_bits:
                rows.append({**common, "kernel": "qvq_lr", "bits": bits, "packing": "V2B2-P32-LR"})
            rows.append({**common, "kernel": "gptq_marlin", "bits": 4.0, "packing": "GPTQ"})
            rows.append({**common, "kernel": "gptq_machete", "bits": 4.0, "packing": "GPTQ"})
    return rows


def _print_table(rows: list[dict]) -> None:
    columns = (
        ("Shape", "name"),
        ("Roles", "roles"),
        ("M", "m"),
        ("K", "k"),
        ("N", "n"),
        ("Kernel", "kernel"),
        ("W", "bits"),
        ("Group", "group_size"),
        ("Median ms", "median_ms"),
        ("P95 ms", "p95_ms"),
        ("TFLOP/s", "logical_tflops"),
        ("Payload GB/s", "effective_payload_gbs"),
        ("xMarlin", "speedup_vs_marlin"),
        ("xMachete", "speedup_vs_machete"),
        ("Max abs", "max_abs"),
        ("State", "state"),
    )
    rendered = []
    for row in rows:
        rendered.append({
            "name": str(row.get("name", "?")),
            "roles": "/".join(row.get("roles", ())),
            "m": str(row.get("m", "?")),
            "k": str(row.get("k", "?")),
            "n": str(row.get("n", "?")),
            "kernel": str(row.get("kernel", "?")),
            "bits": f"{float(row['bits']):g}" if "bits" in row else "?",
            "group_size": "P32" if row.get("kernel") == "qvq_lr" else str(row.get("group_size", "?")),
            "median_ms": f"{row['median_ms']:.4f}" if "median_ms" in row else "pending",
            "p95_ms": f"{row['p95_ms']:.4f}" if "p95_ms" in row else "pending",
            "logical_tflops": f"{row['logical_tflops']:.3f}" if "logical_tflops" in row else "pending",
            "effective_payload_gbs": (
                f"{row['effective_payload_gbs']:.2f}" if "effective_payload_gbs" in row else "pending"
            ),
            "speedup_vs_marlin": (
                f"{row['speedup_vs_marlin']:.3f}x" if "speedup_vs_marlin" in row else "pending"
            ),
            "speedup_vs_machete": (
                f"{row['speedup_vs_machete']:.3f}x" if "speedup_vs_machete" in row else "pending"
            ),
            "max_abs": f"{row['max_abs']:.3g}" if "max_abs" in row else "pending",
            "state": str(row.get("state", "pending")),
        })
    widths = {key: max(len(title), *(len(row[key]) for row in rendered)) for title, key in columns}
    border = "+" + "+".join("-" * (widths[key] + 2) for _, key in columns) + "+"
    print(border)
    print("|" + "|".join(f" {title:<{widths[key]}} " for title, key in columns) + "|")
    print(border)
    for row in rendered:
        print("|" + "|".join(f" {row[key]:<{widths[key]}} " for _, key in columns) + "|")
    print(border, flush=True)


def _markdown_report(payload: dict) -> str:
    """Render the complete measured matrix as a durable Markdown report."""

    hardware = payload["hardware"]
    device = payload["device"]
    args = payload["args"]
    qvq = payload["qvq"]
    gptq = payload["gptq"]
    lines = [
        "# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes",
        "",
        (
            "This is a kernel-throughput comparison on the H100 host from PR #62. "
            "W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm."
        ),
        "",
        "## Measurement contract",
        "",
        f"- Commit: `{payload['commit']}`; benchmark SHA256: `{payload['benchmark_sha256']}`",
        (
            f"- GPU: physical `{payload['physical_gpu']}`, `{device['name']}`, "
            f"PCI `{hardware['pci.bus_id']}`, UUID `{hardware['uuid']}`, "
            f"CC `{device['compute_capability']}`, {device['sm_count']} SMs"
        ),
        (
            f"- Torch/CUDA: `{payload['software']['torch']}` / `{payload['software']['cuda']}`; "
            f"input dtype: `{args['dtype']}`"
        ),
        f"- M values: `{args['m']}`; warmup: `{args['warmup']}`; measured launches: `{args['iterations']}`",
        f"- QVQ: `{qvq['format']}`, rates `{qvq['rates']}`, native `{qvq['note']}`",
        (
            f"- GPTQ: symmetric W{gptq['bits']:g}, group `{gptq['group_size']}`, "
            "no activation order; Marlin and Machete use the same W4 source payload"
        ),
        (
            "- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. "
            "CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; "
            "payload GB/s is packed payload bytes divided by median latency."
        ),
        (
            "- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; "
            "GPTQ uses `atol=2e-2, rtol=2e-2`."
        ),
        "",
        "`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.",
        "",
        "| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        group = "P32" if row["kernel"] == "qvq_lr" else str(row["group_size"])
        x_marlin = f"{row.get('speedup_vs_marlin', float('nan')):.3f}x"
        x_machete = f"{row.get('speedup_vs_machete', float('nan')):.3f}x"
        lines.append(
            f"| {row['name']} | {'/'.join(row['roles'])} | {row['m']} | {row['k']} | {row['n']} | "
            f"{row['kernel']} | {float(row['bits']):g} | {group} | {row['median_ms']:.4f} | "
            f"{row['p95_ms']:.4f} | {row['logical_tflops']:.3f} | {row['effective_payload_gbs']:.2f} | "
            f"{x_marlin} | {x_machete} | {row['max_abs']:.3g} |"
        )
    lines.extend([
        "",
        (
            "The four shape rows preserve all seven Llama 3.2 1B projection roles: "
            "`q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), "
            "`gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048)."
        ),
        "",
    ])
    return "\n".join(lines)


def _write_markdown(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_report(payload), encoding="utf-8")


class _LiveResults:
    def __init__(self, expected: list[dict], interval: float):
        self.expected = expected
        self.interval = interval
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self) -> None:
        if self.interval <= 0:
            return
        self.thread = threading.Thread(target=self._run, name="qvq-benchmark-live-results", daemon=True)
        self.thread.start()

    def add(self, row: dict) -> None:
        with self.lock:
            self.rows.append(row)
        print(
            f"complete {row['name']} M{row['m']} {row['kernel']} W{row['bits']:g}: "
            f"median={row['median_ms']:.4f}ms logical={row['logical_tflops']:.3f}TFLOP/s",
            flush=True,
        )

    def snapshot(self) -> list[dict]:
        with self.lock:
            completed = {_row_key(row): dict(row) for row in self.rows}
        merged = [completed.get(_row_key(row), dict(row)) for row in self.expected]
        return _with_relative_metrics(merged)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            snapshot = self.snapshot()
            complete = sum(row.get("state") == "complete" for row in snapshot)
            print(f"live results: complete={complete}/{len(snapshot)}", flush=True)
            _print_table(snapshot)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _prewarm_extensions(torch, dtype) -> None:
    from gptqmodel.utils.machete import (
        _validate_machete_device_support,
        machete_runtime_error,
        prewarm_machete_extension,
    )
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda

    if not prewarm_qvq_cuda():
        raise RuntimeError("QVQ CUDA extension failed to load")
    if not marlin_runtime_available(dtype):
        raise RuntimeError(marlin_runtime_error(dtype))
    if not _validate_machete_device_support():
        raise RuntimeError(machete_runtime_error())
    if not prewarm_machete_extension():
        raise RuntimeError(machete_runtime_error())
    torch.cuda.synchronize()


def _run(args: argparse.Namespace) -> dict:
    commit = benchmark_utils._git_commit()
    fingerprint = benchmark_utils._source_fingerprint()
    shapes = [_shape_by_name(name) for name in args.shapes]
    expected = _expected_rows(shapes, args.m, args.qvq_bits, args.dtype)
    live = _LiveResults(expected, args.progress_interval)
    live.start()
    try:
        hardware = benchmark_utils._idle_preflight(
            args.physical_gpu,
            args.idle_samples,
            args.idle_interval,
            args.idle_memory_tolerance_mib,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = hardware["uuid"]

        import torch

        from gptqmodel.utils.marlin_scalar_type import scalar_types
        from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv

        _visible_gpu_matches(torch, hardware)
        device = torch.device("cuda:0")
        dtype = getattr(torch, args.dtype)
        properties = torch.cuda.get_device_properties(device)
        print(
            "software: "
            f"physical={args.physical_gpu} pci={hardware['pci.bus_id']} uuid={hardware['uuid']} "
            f"name={properties.name} cc={properties.major}.{properties.minor} "
            f"sms={properties.multi_processor_count} memory={properties.total_memory} "
            f"torch={torch.__version__} cuda={torch.version.cuda} input_dtype={args.dtype}",
            flush=True,
        )
        _prewarm_extensions(torch, dtype)
        _pre_timing_exclusivity_gate(
            physical_gpu=args.physical_gpu,
            gpu_uuid=hardware["uuid"],
            samples=args.idle_samples,
            interval=args.idle_interval,
        )

        for shape_index, case in enumerate(shapes):
            input_generator = torch.Generator().manual_seed(20260830 + shape_index * 1000)
            inputs = {
                m: (torch.randn((m, case.in_features), generator=input_generator, dtype=torch.float32) * 0.1).to(
                    device=device, dtype=dtype
                )
                for m in args.m
            }
            gptq_source = _gptq_source(
                torch,
                case,
                group_size=args.gptq_group_size,
                seed=20260830 + shape_index * 1000 + 1,
                dtype=dtype,
            )
            dense_gptq = _dense_gptq_weight(
                torch,
                gptq_source,
                device=device,
                weight_type=scalar_types.uint4b8,
            )
            marlin, marlin_bytes = _build_gptq_module(
                torch,
                "gptq_marlin",
                case,
                group_size=args.gptq_group_size,
                source=gptq_source,
                device=device,
                dtype=dtype,
            )
            machete, machete_bytes = _build_gptq_module(
                torch,
                "gptq_machete",
                case,
                group_size=args.gptq_group_size,
                source=gptq_source,
                device=device,
                dtype=dtype,
            )

            for bits in args.qvq_bits:
                trellis, bank_ids, dense_qvq, qvq_bytes = _qvq_payload(
                    torch,
                    case,
                    bits=bits,
                    seed=20260830 + shape_index * 1000 + int(bits * 10),
                    device=device,
                )
                for m in args.m:
                    x = inputs[m]
                    reference = x.float() @ dense_qvq.float()

                    def qvq_call(x=x, trellis=trellis, bits=bits, n=case.out_features, bank_ids=bank_ids):
                        return qvq_cuda_gemv(
                            x,
                            trellis,
                            bits,
                            out_features=n,
                            output_fp32=True,
                            bank_ids=bank_ids,
                            v2b2_p32_lr=True,
                            bank_alt_id=3,
                        )

                    actual = qvq_call()
                    torch.cuda.synchronize(device)
                    metrics = _assert_correct(
                        kernel=f"QVQ LR W{bits:g}",
                        actual=actual,
                        reference=reference,
                        expected_shape=(m, case.out_features),
                        expected_dtype=torch.float32,
                        atol=2e-3,
                        rtol=0.0,
                    )
                    timing = _cuda_graph_event_timing(
                        torch,
                        qvq_call,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    live.add(_row_from_timing(
                        case=case,
                        m=m,
                        kernel="qvq_lr",
                        bits=bits,
                        group_size=None,
                        packing="V2B2-P32-LR",
                        dtype_name=args.dtype,
                        output_dtype="float32",
                        payload_bytes=qvq_bytes,
                        metrics=metrics,
                        timing=timing,
                    ))
                del trellis, bank_ids, dense_qvq

            for kernel, module, payload_bytes in (
                ("gptq_marlin", marlin, marlin_bytes),
                ("gptq_machete", machete, machete_bytes),
            ):
                for m in args.m:
                    x = inputs[m]
                    reference = x.float() @ dense_gptq.float()

                    def gptq_call(module=module, x=x):
                        return module(x)

                    actual = gptq_call()
                    torch.cuda.synchronize(device)
                    metrics = _assert_correct(
                        kernel=kernel,
                        actual=actual,
                        reference=reference,
                        expected_shape=(m, case.out_features),
                        expected_dtype=dtype,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                    timing = _cuda_graph_event_timing(
                        torch,
                        gptq_call,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    live.add(_row_from_timing(
                        case=case,
                        m=m,
                        kernel=kernel,
                        bits=4.0,
                        group_size=args.gptq_group_size,
                        packing="GPTQ",
                        dtype_name=args.dtype,
                        output_dtype=args.dtype,
                        payload_bytes=payload_bytes,
                        metrics=metrics,
                        timing=timing,
                    ))

            del inputs, gptq_source, dense_gptq, marlin, machete
            torch.cuda.empty_cache()

        rows = _with_relative_metrics(live.rows)
        benchmark_utils._verify_source(commit, fingerprint, phase="after benchmark completed")
        payload = {
            "label": "qvq_lr_vs_w4_gptq_llama32_1b",
            "scope": "kernel throughput only; W4 GPTQ is a figurative performance baseline, not a quality match",
            "commit": commit,
            "source_fingerprint": fingerprint,
            "benchmark_sha256": _benchmark_sha256(),
            "args": _args_payload(args),
            "physical_gpu": args.physical_gpu,
            "hardware": hardware,
            "device": {
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "sm_count": properties.multi_processor_count,
                "total_memory_bytes": properties.total_memory,
            },
            "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
            "timing": {
                "mode": "single_cuda_graph_replay_with_internal_external_events",
                "host_launch_gaps_included": False,
            },
            "qvq": {
                "format": "qvq_v2b2_p32_lr",
                "rates": args.qvq_bits,
                "output_dtype": "float32",
                "group_size": -1,
                "note": "P32 is LR packing geometry, not a GPTQ affine scale group.",
            },
            "gptq": {
                "bits": 4,
                "group_size": args.gptq_group_size,
                "sym": True,
                "desc_act": False,
                "kernels": ["marlin", "machete"],
            },
            "projection_shapes": [asdict(case) for case in shapes],
            "rows": rows,
        }
        benchmark_utils._write_json(args.output, payload)
        markdown_output = args.markdown_output or args.output.with_suffix(".md")
        _write_markdown(markdown_output, payload)
        print(
            f"final results: complete={len(rows)}/{len(expected)} output={args.output} markdown={markdown_output}",
            flush=True,
        )
        _print_table(rows)
        return payload
    finally:
        live.stop()


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
