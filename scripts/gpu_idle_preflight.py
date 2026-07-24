#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only physical-GPU idle and exclusivity checks for performance scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import subprocess
import sys
import time
from typing import Any, Sequence


@dataclass(frozen=True)
class GPUIdlePreflightConfig:
    sample_count: int
    sample_interval_seconds: float
    max_driver_memory_mib: int


@dataclass(frozen=True)
class GPUIdlePreflightState:
    physical_id: int
    pci_bus_id: str
    uuid: str
    name: str
    config: GPUIdlePreflightConfig
    samples: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_gpu_idle_preflight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-gpu-idle-preflight",
        action="store_true",
        help="Explicitly disable the strict physical-GPU idle gate; results are not formal performance data.",
    )
    parser.add_argument(
        "--idle-samples",
        type=int,
        default=3,
        help="Consecutive 0%%-utilization samples required before CUDA import; must be at least 3.",
    )
    parser.add_argument(
        "--idle-interval-seconds",
        type=float,
        default=1.0,
        help="Seconds between pre-CUDA idle samples.",
    )
    parser.add_argument(
        "--idle-max-driver-memory-mib",
        type=int,
        default=16,
        help="Maximum unexplained driver-baseline memory allowed on the requested physical GPU.",
    )


def _config_from_args(args: argparse.Namespace) -> GPUIdlePreflightConfig:
    config = GPUIdlePreflightConfig(
        sample_count=args.idle_samples,
        sample_interval_seconds=args.idle_interval_seconds,
        max_driver_memory_mib=args.idle_max_driver_memory_mib,
    )
    if config.sample_count < 3:
        raise ValueError("--idle-samples must be at least 3")
    if config.sample_interval_seconds < 0:
        raise ValueError("--idle-interval-seconds must be non-negative")
    if config.max_driver_memory_mib < 0:
        raise ValueError("--idle-max-driver-memory-mib must be non-negative")
    return config


def _run_nvidia_smi(arguments: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"GPU idle preflight nvidia-smi query failed: {detail}")
    return result.stdout.strip()


def _parse_process_rows(output: str, expected_uuid: str) -> tuple[dict[str, Any], ...]:
    processes = []
    for row in output.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.split(",", maxsplit=3)]
        if len(fields) != 4 or fields[0] != expected_uuid:
            raise RuntimeError(f"GPU idle preflight received an invalid compute-process row: {row!r}")
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise RuntimeError(f"GPU idle preflight received an invalid process PID: {row!r}") from exc
        memory_text = fields[3]
        try:
            memory_used_mib = int(memory_text)
        except ValueError:
            memory_used_mib = None
        processes.append(
            {
                "gpu_uuid": fields[0],
                "pid": pid,
                "process_name": fields[2],
                "memory_used_mib": memory_used_mib,
            }
        )
    return tuple(processes)


def _query_snapshot(gpu_uuid: str) -> dict[str, Any]:
    inventory = _run_nvidia_smi(
        [
            "-i",
            gpu_uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [row for row in inventory.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"GPU idle preflight expected one inventory row, got: {inventory!r}")
    fields = [field.strip() for field in rows[0].split(",", maxsplit=5)]
    if len(fields) != 6 or fields[2] != gpu_uuid:
        raise RuntimeError(f"GPU idle preflight received an invalid inventory row: {rows[0]!r}")
    try:
        physical_id = int(fields[0])
        memory_used_mib = int(fields[4])
        utilization_pct = int(fields[5])
    except ValueError as exc:
        raise RuntimeError(f"GPU idle preflight received non-integer inventory values: {rows[0]!r}") from exc

    process_output = _run_nvidia_smi(
        [
            "-i",
            gpu_uuid,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "physical_id": physical_id,
        "pci_bus_id": fields[1],
        "uuid": fields[2],
        "name": fields[3],
        "memory_used_mib": memory_used_mib,
        "utilization_pct": utilization_pct,
        "compute_processes": _parse_process_rows(process_output, gpu_uuid),
    }


def _requested_gpu_uuid() -> str:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER=PCI_BUS_ID is required for physical-GPU performance testing.")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly one physical GPU UUID.")
    return gpu_uuid


def bootstrap_gpu_idle_preflight(
    argv: Sequence[str] | None = None,
) -> GPUIdlePreflightState | None:
    """Run the strict idle gate before the caller imports Torch or initializes CUDA."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if any(argument in ("-h", "--help") for argument in arguments):
        return None
    parser = argparse.ArgumentParser(add_help=False)
    add_gpu_idle_preflight_args(parser)
    args, _ = parser.parse_known_args(arguments)
    if args.skip_gpu_idle_preflight:
        print("GPU idle preflight explicitly disabled; timing results are not formal.", flush=True)
        return None

    config = _config_from_args(args)
    gpu_uuid = _requested_gpu_uuid()
    accepted_samples = []
    for sample_index in range(config.sample_count):
        snapshot = _query_snapshot(gpu_uuid)
        if snapshot["compute_processes"]:
            raise RuntimeError(
                "GPU idle preflight rejected "
                f"physical_id={snapshot['physical_id']} pci_bus_id={snapshot['pci_bus_id']} uuid={gpu_uuid} "
                f"foreign_processes={snapshot['compute_processes']}"
            )
        if snapshot["utilization_pct"] != 0 or snapshot["memory_used_mib"] > config.max_driver_memory_mib:
            raise RuntimeError(
                "GPU idle preflight rejected "
                f"physical_id={snapshot['physical_id']} pci_bus_id={snapshot['pci_bus_id']} uuid={gpu_uuid} "
                f"utilization={snapshot['utilization_pct']}% memory={snapshot['memory_used_mib']}MiB "
                f"required_utilization=0% max_driver_memory={config.max_driver_memory_mib}MiB"
            )
        accepted_samples.append(snapshot)
        if sample_index + 1 < config.sample_count:
            time.sleep(config.sample_interval_seconds)

    final = accepted_samples[-1]
    state = GPUIdlePreflightState(
        physical_id=final["physical_id"],
        pci_bus_id=final["pci_bus_id"],
        uuid=final["uuid"],
        name=final["name"],
        config=config,
        samples=tuple(accepted_samples),
    )
    os.environ["GPTQMODEL_IDLE_PREFLIGHT_JSON"] = json.dumps(state.as_dict(), sort_keys=True)
    print(
        "GPU idle preflight accepted: "
        f"physical_id={state.physical_id} pci_bus_id={state.pci_bus_id} uuid={state.uuid} "
        f"utilization={final['utilization_pct']}% memory={final['memory_used_mib']}MiB "
        f"samples={config.sample_count} interval={config.sample_interval_seconds}s "
        f"required_utilization=0% max_driver_memory={config.max_driver_memory_mib}MiB",
        flush=True,
    )
    return state


def recheck_gpu_exclusivity(state: GPUIdlePreflightState) -> dict[str, Any]:
    """Verify target identity and memory ownership immediately before timed launches."""

    snapshot = _query_snapshot(state.uuid)
    identity = (snapshot["physical_id"], snapshot["pci_bus_id"], snapshot["uuid"])
    expected_identity = (state.physical_id, state.pci_bus_id, state.uuid)
    if identity != expected_identity:
        raise RuntimeError(
            f"GPU pre-timing recheck identity mismatch: expected={expected_identity}, observed={identity}"
        )

    current_pid = os.getpid()
    foreign_processes = tuple(
        process for process in snapshot["compute_processes"] if process["pid"] != current_pid
    )
    if foreign_processes:
        raise RuntimeError(
            "GPU pre-timing recheck rejected "
            f"physical_id={state.physical_id} pci_bus_id={state.pci_bus_id} uuid={state.uuid} "
            f"foreign_processes={foreign_processes}"
        )

    own_processes = tuple(
        process for process in snapshot["compute_processes"] if process["pid"] == current_pid
    )
    if any(process["memory_used_mib"] is None for process in own_processes):
        raise RuntimeError(
            "GPU pre-timing recheck could not attribute the current process memory: "
            f"own_processes={own_processes}"
        )
    own_memory_mib = sum(process["memory_used_mib"] for process in own_processes)
    unattributed_memory_mib = max(0, snapshot["memory_used_mib"] - own_memory_mib)
    if unattributed_memory_mib > state.config.max_driver_memory_mib:
        raise RuntimeError(
            "GPU pre-timing recheck rejected unexplained memory residency: "
            f"physical_id={state.physical_id} pci_bus_id={state.pci_bus_id} uuid={state.uuid} "
            f"total_memory={snapshot['memory_used_mib']}MiB own_memory={own_memory_mib}MiB "
            f"unattributed_memory={unattributed_memory_mib}MiB "
            f"max_driver_memory={state.config.max_driver_memory_mib}MiB"
        )

    print(
        "GPU pre-timing exclusivity accepted: "
        f"physical_id={state.physical_id} pci_bus_id={state.pci_bus_id} uuid={state.uuid} "
        f"utilization={snapshot['utilization_pct']}% total_memory={snapshot['memory_used_mib']}MiB "
        f"own_memory={own_memory_mib}MiB unattributed_memory={unattributed_memory_mib}MiB "
        f"foreign_processes=0 max_driver_memory={state.config.max_driver_memory_mib}MiB",
        flush=True,
    )
    return snapshot
