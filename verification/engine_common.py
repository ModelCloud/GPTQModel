#!/usr/bin/env python3
"""Stdlib-only GPU preflight and reporting helpers for engine verification."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

GPU_QUERY_FIELDS = (
    "index",
    "pci.bus_id",
    "uuid",
    "name",
    "driver_version",
    "compute_cap",
    "memory.total",
    "memory.used",
    "utilization.gpu",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exact_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def run_text(command: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        list(command),
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def visible_gpu_targets(expected_count: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    targets = [item.strip() for item in visible.split(",") if item.strip()]
    if len(targets) != expected_count:
        raise RuntimeError(
            f"Expected exactly {expected_count} allocated CUDA_VISIBLE_DEVICES entries; received {visible!r}. "
            "Run this command under a GPU lease and prefer GPU UUIDs."
        )
    if len(set(targets)) != len(targets):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES contains duplicate targets: {targets}"
        )
    return targets


def _parse_csv_row(output: str, expected_columns: int) -> list[str]:
    rows = [
        row
        for row in csv.reader(output.splitlines())
        if any(value.strip() for value in row)
    ]
    if len(rows) != 1 or len(rows[0]) != expected_columns:
        raise RuntimeError(f"Unexpected nvidia-smi output: {output!r}")
    return [value.strip() for value in rows[0]]


def query_gpu(target: str) -> dict[str, Any]:
    output = run_text(
        [
            "nvidia-smi",
            f"--id={target}",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    values = _parse_csv_row(output, len(GPU_QUERY_FIELDS))
    result = dict(zip(GPU_QUERY_FIELDS, values, strict=True))
    for key in ("index", "memory.total", "memory.used", "utilization.gpu"):
        result[key] = int(result[key])
    return result


def query_compute_processes(target: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={target}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.strip()
    if not output or "No running processes found" in output:
        return []

    processes = []
    for values in csv.reader(output.splitlines()):
        if len(values) != 3:
            continue
        try:
            pid = int(values[0].strip())
        except ValueError:
            continue
        try:
            used_memory = int(values[2].strip())
        except ValueError:
            used_memory = None
        processes.append(
            {
                "pid": pid,
                "process_name": values[1].strip(),
                "used_memory_mib": used_memory,
            }
        )
    return processes


def query_all_compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.strip()
    if not output or "No running processes found" in output:
        return []

    processes = []
    for values in csv.reader(output.splitlines()):
        if len(values) != 4:
            continue
        try:
            pid = int(values[1].strip())
        except ValueError:
            continue
        try:
            used_memory = int(values[3].strip())
        except ValueError:
            used_memory = None
        processes.append(
            {
                "gpu_uuid": values[0].strip(),
                "pid": pid,
                "process_name": values[2].strip(),
                "used_memory_mib": used_memory,
            }
        )
    return processes


class StrictIdleGateError(RuntimeError):
    """A strict idle-gate sample observed a busy or not-yet-quiescent GPU."""

    def __init__(self, sample: dict[str, Any]):
        self.sample = sample
        super().__init__(f"GPU failed strict idle gate: {sample}")


def strict_idle_gate(
    targets: Sequence[str],
    *,
    sample_count: int = 3,
    interval_seconds: float = 1.0,
    memory_tolerance_mib: int = 16,
) -> list[dict[str, Any]]:
    if sample_count < 1 or interval_seconds < 0 or memory_tolerance_mib < 0:
        raise ValueError(
            "Idle-gate samples must be positive and thresholds must be non-negative."
        )

    samples = []
    for sample_index in range(sample_count):
        for target in targets:
            gpu = query_gpu(target)
            processes = query_compute_processes(target)
            sample = {
                "sample": sample_index + 1,
                "requested_target": target,
                "physical_id": gpu["index"],
                "pci_bus_id": gpu["pci.bus_id"],
                "uuid": gpu["uuid"],
                "name": gpu["name"],
                "driver_version": gpu["driver_version"],
                "compute_capability": gpu["compute_cap"],
                "memory_total_mib": gpu["memory.total"],
                "memory_used_mib": gpu["memory.used"],
                "utilization_gpu_percent": gpu["utilization.gpu"],
                "compute_processes": processes,
                "memory_tolerance_mib": memory_tolerance_mib,
            }
            samples.append(sample)
            print(
                "PREFLIGHT "
                f"sample={sample_index + 1}/{sample_count} target={target} "
                f"physical_id={gpu['index']} pci_bus_id={gpu['pci.bus_id']} uuid={gpu['uuid']} "
                f"mem={gpu['memory.used']}MiB mem_limit={memory_tolerance_mib}MiB "
                f"util={gpu['utilization.gpu']}% processes={len(processes)}",
                flush=True,
            )
            if (
                gpu["memory.used"] > memory_tolerance_mib
                or gpu["utilization.gpu"] != 0
                or processes
            ):
                raise StrictIdleGateError(sample)
        if sample_index + 1 < sample_count:
            time.sleep(interval_seconds)
    return samples


def wait_for_strict_idle_gate(
    targets: Sequence[str],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    sample_count: int = 3,
    interval_seconds: float = 1.0,
    memory_tolerance_mib: int = 16,
) -> dict[str, Any]:
    """Wait for a complete strict idle gate within a bounded interval.

    This is used at engine transitions and immediately before worker admission,
    where process teardown or driver telemetry can briefly lag an earlier gate.
    Every successful return still requires the requested consecutive strict-idle
    samples; retries never weaken the memory, utilization, or process checks.
    """
    if timeout_seconds < 0 or poll_interval_seconds < 0:
        raise ValueError("Idle transition timeout and polling interval must be non-negative.")

    started_at_utc = utc_now()
    started_monotonic = time.monotonic()
    failed_attempts = []
    attempt = 0
    while True:
        attempt += 1
        try:
            samples = strict_idle_gate(
                targets,
                sample_count=sample_count,
                interval_seconds=interval_seconds,
                memory_tolerance_mib=memory_tolerance_mib,
            )
        except StrictIdleGateError as exc:
            elapsed_seconds = time.monotonic() - started_monotonic
            failed_attempts.append(
                {
                    "attempt": attempt,
                    "observed_at_utc": utc_now(),
                    "elapsed_seconds": elapsed_seconds,
                    "sample": exc.sample,
                }
            )
            if elapsed_seconds >= timeout_seconds:
                raise RuntimeError(
                    "GPU did not reach a strict idle state before the strict-idle timeout; "
                    f"last sample: {exc.sample}"
                ) from exc
            time.sleep(min(poll_interval_seconds, timeout_seconds - elapsed_seconds))
            continue

        return {
            "started_at_utc": started_at_utc,
            "completed_at_utc": utc_now(),
            "wait_seconds": time.monotonic() - started_monotonic,
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "failed_attempts": failed_attempts,
            "strict_idle_preflight": samples,
        }


def collect_torch_runtime_metadata(expected_visible_count: int) -> dict[str, Any]:
    probe = """
import json
import torch

devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append({
        "process_local_index": index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
    })
print(json.dumps({
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "visible_device_count": torch.cuda.device_count(),
    "devices": devices,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Torch runtime metadata probe failed: {detail}")
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Torch runtime metadata probe returned invalid JSON: {result.stdout!r}"
        ) from error
    if metadata.get("visible_device_count") != expected_visible_count:
        raise RuntimeError(
            "Torch metadata probe saw an unexpected number of CUDA devices: "
            f"expected={expected_visible_count}, observed={metadata.get('visible_device_count')}"
        )
    return metadata


def _parent_pid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    close_paren = stat.rfind(")")
    if close_paren < 0:
        return None
    fields = stat[close_paren + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def is_descendant(pid: int, root_pid: int) -> bool:
    visited: set[int] = set()
    current = pid
    while current > 1 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        next_pid = _parent_pid(current)
        if next_pid is None:
            return False
        current = next_pid
    return current == root_pid


def verify_runtime_exclusive(targets: Sequence[str]) -> dict[str, Any]:
    root_pid = os.getpid()
    gpu_rows = [query_gpu(target) for target in targets]
    target_uuids = {str(row["uuid"]) for row in gpu_rows}
    all_processes = query_all_compute_processes()
    target_processes = [
        process for process in all_processes if process["gpu_uuid"] in target_uuids
    ]
    owned = [
        process
        for process in target_processes
        if is_descendant(int(process["pid"]), root_pid)
    ]
    foreign = [
        process
        for process in target_processes
        if not is_descendant(int(process["pid"]), root_pid)
    ]
    owned_uuids = {str(process["gpu_uuid"]) for process in owned}
    print(
        f"EXCLUSIVITY root_pid={root_pid} targets={len(targets)} owned={len(owned)} foreign={len(foreign)}",
        flush=True,
    )
    if foreign:
        raise RuntimeError(f"Foreign GPU processes detected after warmup: {foreign}")
    if owned_uuids != target_uuids:
        raise RuntimeError(
            "Engine process-to-GPU mapping does not cover the requested devices: "
            f"expected={sorted(target_uuids)}, observed={sorted(owned_uuids)}"
        )
    return {
        "root_pid": root_pid,
        "target_uuids": sorted(target_uuids),
        "owned_compute_processes": owned,
        "foreign_compute_processes": foreign,
    }


def collect_gpu_metadata(
    torch_module: Any, targets: Sequence[str]
) -> list[dict[str, Any]]:
    visible_count = int(torch_module.cuda.device_count())
    if visible_count != len(targets):
        raise RuntimeError(
            f"PyTorch sees {visible_count} CUDA devices but the verification command requested {len(targets)}."
        )
    records = []
    for local_index, target in enumerate(targets):
        gpu = query_gpu(target)
        properties = torch_module.cuda.get_device_properties(local_index)
        records.append(
            {
                "process_local_index": local_index,
                "requested_target": target,
                "nvidia_smi": gpu,
                "torch_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "sm_count": properties.multi_processor_count,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return records


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata(torch_module: Any, engine: str) -> dict[str, Any]:
    packages = ("transformers", "safetensors", "flashinfer-python", engine)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
        "packages": {package: package_version(package) for package in packages},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "gpu_allocator_lease_id": os.environ.get("GPU_ALLOCATOR_LEASE_ID"),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def load_json_object(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    if value.startswith("@"):
        payload = json.loads(Path(value[1:]).read_text())
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Engine arguments must decode to a JSON object.")
    return payload


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def comma_separated_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(
            f"Expected comma-separated integers, received {value!r}."
        ) from error
    if not values:
        raise ValueError("At least one integer is required.")
    return values


def render_ascii_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        if len(row) != len(headers):
            raise ValueError("Table rows must have the same width as the header.")
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    return "\n".join(
        (
            separator,
            render(list(headers)),
            separator,
            *(render(row) for row in string_rows),
            separator,
        )
    )
