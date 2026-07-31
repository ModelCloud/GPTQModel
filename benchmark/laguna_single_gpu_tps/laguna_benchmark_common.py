"""Shared stdlib-only support for the Laguna single-GPU TPS benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LAGUNA_BENCH_ROOT", BENCHMARK_DIR.parents[1]))
MODEL_REQUESTED = os.environ.get(
    "LAGUNA_BENCH_MODEL_REQUESTED",
    "/monster/data/model/laguna-s-2.1-4g64",
)
MODEL = Path(
    os.environ.get(
        "LAGUNA_BENCH_MODEL",
        "/monster/data/model/Laguna-S-2.1-GPTQ-4G64",
    )
)
SGLANG_REPO = Path(os.environ.get("LAGUNA_BENCH_SGLANG_REPO", ROOT / "hub/sglang"))
VLLM_REPO = Path(os.environ.get("LAGUNA_BENCH_VLLM_REPO", ROOT / "hub/vllm"))
OUTPUT_DIR = Path(os.environ.get("LAGUNA_BENCH_OUTPUT_DIR", "/tmp"))

BATCH_SIZES = [
    int(item)
    for item in os.environ.get("LAGUNA_BENCH_BATCH_SIZES", "1,2,4,8,16,32,64").split(
        ","
    )
    if item.strip()
]
INPUT_LEN = int(os.environ.get("LAGUNA_BENCH_INPUT_LEN", "256"))
OUTPUT_LEN = int(os.environ.get("LAGUNA_BENCH_OUTPUT_LEN", "128"))
CONTEXT_MARGIN = int(os.environ.get("LAGUNA_BENCH_CONTEXT_MARGIN", "0"))
MAX_MODEL_LEN = INPUT_LEN + OUTPUT_LEN + CONTEXT_MARGIN
REPEATS = int(os.environ.get("LAGUNA_BENCH_REPEATS", "3"))
SEED = 20260730
GPU_IDLE_SAMPLES = 3
GPU_IDLE_INTERVAL_SEC = 1.0
GPU_IDLE_MEMORY_TOLERANCE_MIB = 16
QUANT_EXCLUDE_PATTERN = r"-:^model\.layers\.\d+\.self_attn\.g_proj$"
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()

if (
    not BATCH_SIZES
    or any(batch_size <= 0 for batch_size in BATCH_SIZES)
    or BATCH_SIZES != sorted(set(BATCH_SIZES))
):
    raise ValueError(
        "LAGUNA_BENCH_BATCH_SIZES must be a non-empty, strictly increasing list of positive integers."
    )
if INPUT_LEN <= 0 or OUTPUT_LEN <= 0 or CONTEXT_MARGIN < 0 or REPEATS < 2:
    raise ValueError(
        "LAGUNA_BENCH_INPUT_LEN and LAGUNA_BENCH_OUTPUT_LEN must be positive; "
        "LAGUNA_BENCH_CONTEXT_MARGIN must be non-negative; "
        "LAGUNA_BENCH_REPEATS must be at least 2."
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        command,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def git_revision(repo: Path) -> str:
    return run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def visible_gpu_target() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    targets = [item.strip() for item in visible.split(",") if item.strip()]
    if len(targets) != 1:
        raise RuntimeError(
            "Benchmark requires exactly one allocated CUDA_VISIBLE_DEVICES entry; "
            f"received {visible!r}."
        )
    return targets[0]


def query_gpu(target: str) -> dict[str, Any]:
    fields = [
        "index",
        "pci.bus_id",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "utilization.gpu",
    ]
    output = run_text(
        [
            "nvidia-smi",
            f"--id={target}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one nvidia-smi row for {target}, received: {rows}"
        )
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"Unexpected nvidia-smi row: {rows[0]!r}")
    result = dict(zip(fields, values, strict=True))
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
    for row in output.splitlines():
        values = [value.strip() for value in row.split(",")]
        if len(values) != 3:
            continue
        try:
            pid = int(values[0])
        except ValueError:
            continue
        try:
            used_memory = int(values[2])
        except ValueError:
            used_memory = None
        processes.append(
            {
                "pid": pid,
                "process_name": values[1],
                "used_memory_mib": used_memory,
            }
        )
    return processes


def strict_idle_gate(target: str) -> list[dict[str, Any]]:
    samples = []
    for sample_index in range(GPU_IDLE_SAMPLES):
        gpu = query_gpu(target)
        processes = query_compute_processes(target)
        sample = {
            "sample": sample_index + 1,
            "memory_used_mib": gpu["memory.used"],
            "utilization_gpu_percent": gpu["utilization.gpu"],
            "compute_processes": processes,
        }
        samples.append(sample)
        print(
            "PREFLIGHT "
            f"sample={sample_index + 1}/{GPU_IDLE_SAMPLES} "
            f"physical_id={gpu['index']} pci_bus_id={gpu['pci.bus_id']} "
            f"uuid={gpu['uuid']} mem={gpu['memory.used']}MiB "
            f"mem_limit={GPU_IDLE_MEMORY_TOLERANCE_MIB}MiB "
            f"util={gpu['utilization.gpu']}% processes={len(processes)}",
            flush=True,
        )
        if (
            gpu["memory.used"] > GPU_IDLE_MEMORY_TOLERANCE_MIB
            or gpu["utilization.gpu"] != 0
            or processes
        ):
            raise RuntimeError(f"GPU failed strict idle gate: {sample}")
        if sample_index + 1 < GPU_IDLE_SAMPLES:
            time.sleep(GPU_IDLE_INTERVAL_SEC)
    return samples


def parent_pid(pid: int) -> int | None:
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
        next_pid = parent_pid(current)
        if next_pid is None:
            return False
        current = next_pid
    return current == root_pid


def verify_exclusive_after_warmup(target: str) -> list[dict[str, Any]]:
    processes = query_compute_processes(target)
    root_pid = os.getpid()
    foreign = [
        process
        for process in processes
        if not is_descendant(int(process["pid"]), root_pid)
    ]
    print(
        f"EXCLUSIVITY root_pid={root_pid} gpu_processes={len(processes)} foreign={len(foreign)}",
        flush=True,
    )
    if foreign:
        raise RuntimeError(f"Foreign GPU process detected after warmup: {foreign}")
    if not processes:
        raise RuntimeError("No benchmark-owned compute process found after warmup.")
    return processes


def load_quantization_override() -> dict[str, Any]:
    config = json.loads((MODEL / "config.json").read_text())
    quantization_config = dict(config["quantization_config"])
    dynamic = dict(quantization_config.get("dynamic") or {})
    dynamic[QUANT_EXCLUDE_PATTERN] = {}
    quantization_config["dynamic"] = dynamic
    return {"quantization_config": quantization_config}


def make_prompts() -> list[list[int]]:
    generator = random.Random(SEED)
    prompts = []
    for request_index in range(max(BATCH_SIZES)):
        prompt = [2]
        prompt.extend(generator.randrange(100, 100_000) for _ in range(INPUT_LEN - 1))
        prompt[-1] = 100 + request_index
        prompts.append(prompt)
    return prompts


def prompt_sha256(prompts: list[list[int]]) -> str:
    payload = json.dumps(prompts, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def base_result(
    framework: str,
    target: str,
    preflight: list[dict[str, Any]],
    torch_module: Any,
    framework_version: str | None,
) -> dict[str, Any]:
    gpu = query_gpu(target)
    properties = torch_module.cuda.get_device_properties(0)
    repo = SGLANG_REPO if framework == "sglang" else VLLM_REPO
    return {
        "success": False,
        "framework": framework,
        "framework_version": framework_version,
        "framework_commit": git_revision(repo),
        "worker_pid": os.getpid(),
        "worker_process_started_at_utc": PROCESS_STARTED_AT_UTC,
        "gpu_allocator_lease_id": os.environ.get("GPU_ALLOCATOR_LEASE_ID"),
        "model_requested": MODEL_REQUESTED,
        "model_resolved": str(MODEL),
        "model_architecture": "LagunaForCausalLM",
        "quantization": {
            "weight_bits": 4,
            "activation_dtype": "float16",
            "checkpoint_dense_dtype": "bfloat16",
            "group_size": 64,
            "desc_act": False,
            "sym": True,
            "dense_runtime_exclusion": QUANT_EXCLUDE_PATTERN,
            "dense_runtime_exclusion_reason": (
                "Checkpoint stores self_attn.g_proj as BF16 .weight tensors."
            ),
            "checkpoint_name_compatibility": {
                "from": ".mlp.shared_experts.",
                "to": ".mlp.shared_expert.",
                "reason": (
                    "Checkpoint uses the plural prefix while both main-branch "
                    "runtime models use the singular prefix."
                ),
            },
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch_module.__version__,
            "torch_cuda": torch_module.version.cuda,
            "transformers": package_version("transformers"),
            "flashinfer_python": package_version("flashinfer-python"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        },
        "gpu": {
            "nvidia_smi": gpu,
            "torch_name": properties.name,
            "compute_capability": list(torch_module.cuda.get_device_capability(0)),
            "sm_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "strict_idle_preflight": preflight,
            "peak_sampled_memory_used_mib": gpu["memory.used"],
        },
        "rows": [],
    }
