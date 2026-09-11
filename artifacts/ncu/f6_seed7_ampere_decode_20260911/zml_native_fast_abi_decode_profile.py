#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

C_U32 = ctypes.c_uint32
C_I32 = ctypes.c_int32


class CudaProfiler:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaProfilerStart.argtypes = []
        self.lib.cudaProfilerStart.restype = ctypes.c_int
        self.lib.cudaProfilerStop.argtypes = []
        self.lib.cudaProfilerStop.restype = ctypes.c_int

    def start(self) -> None:
        rc = self.lib.cudaProfilerStart()
        if rc != 0:
            raise RuntimeError(f"cudaProfilerStart failed with status {rc}")

    def stop(self) -> None:
        rc = self.lib.cudaProfilerStop()
        if rc != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {rc}")


@dataclass(frozen=True)
class StepArrays:
    tokens: object
    positions: object
    slots: object
    sample_indices: object
    seq_lens: object
    query_starts: object


class NativeLlama:
    def __init__(self, lib_path: Path) -> None:
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.zml_llama_abi_version.argtypes = []
        self.lib.zml_llama_abi_version.restype = C_U32
        self.lib.zml_llama_last_error.argtypes = []
        self.lib.zml_llama_last_error.restype = ctypes.c_char_p
        self.lib.zml_llama_create.argtypes = [
            ctypes.c_char_p,
            C_U32,
            C_U32,
            C_U32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.zml_llama_create.restype = ctypes.c_int
        self.lib.zml_llama_step.argtypes = [
            ctypes.c_void_p,
            C_U32,
            ctypes.POINTER(C_U32),
            ctypes.POINTER(C_U32),
            ctypes.POINTER(C_U32),
            ctypes.POINTER(C_U32),
            ctypes.POINTER(C_I32),
            ctypes.POINTER(C_I32),
            ctypes.POINTER(C_I32),
            ctypes.POINTER(C_U32),
        ]
        self.lib.zml_llama_step.restype = ctypes.c_int
        self.lib.zml_llama_destroy.argtypes = [ctypes.c_void_p]
        self.lib.zml_llama_destroy.restype = ctypes.c_int
        self.handle = ctypes.c_void_p()

    def abi_version(self) -> int:
        return int(self.lib.zml_llama_abi_version())

    def last_error(self) -> str:
        raw = self.lib.zml_llama_last_error()
        return "" if raw is None else raw.decode("utf-8", errors="replace")

    def create(self, model_path: Path, context: int, batch: int, capacity: int) -> None:
        rc = self.lib.zml_llama_create(
            str(model_path).encode(),
            C_U32(context),
            C_U32(batch),
            C_U32(capacity),
            ctypes.byref(self.handle),
        )
        if rc != 0:
            raise RuntimeError(f"zml_llama_create failed: {self.last_error()}")

    def step(
        self,
        *,
        prefill: bool,
        step: StepArrays,
        block_table: object,
        output: object,
    ) -> None:
        rc = self.lib.zml_llama_step(
            self.handle,
            C_U32(1 if prefill else 0),
            step.tokens,
            step.positions,
            step.slots,
            step.sample_indices,
            block_table,
            step.seq_lens,
            step.query_starts,
            output,
        )
        if rc != 0:
            raise RuntimeError(f"zml_llama_step failed: {self.last_error()}")

    def destroy(self) -> None:
        if self.handle.value:
            self.lib.zml_llama_destroy(self.handle)
            self.handle = ctypes.c_void_p()


def run_text(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, object]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_s": time.time() - start,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"cmd": cmd, "error": repr(exc), "elapsed_s": time.time() - start}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_model_config(model_path: Path) -> dict[str, object]:
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (model_path / "quantize_config.json").open("r", encoding="utf-8") as handle:
        quant = json.load(handle)
    with (model_path / "tokenizer_config.json").open("r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)
    return {"config": config, "quantize_config": quant, "tokenizer_config": tokenizer_config}


def collect_provenance(args: argparse.Namespace) -> dict[str, object]:
    model_path = Path(args.model)
    zml_repo = Path(args.zml_repo)
    qvq_repo = Path(args.qvq_repo)
    files = [
        Path(args.lib),
        Path(args.runner),
        model_path / "config.json",
        model_path / "quantize_config.json",
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
        model_path / "chat_template.jinja",
        model_path / "model.safetensors.index.json",
        model_path / "qvq_quantize_run.json",
        model_path / "rank8_generation_report.json",
    ]
    file_records: dict[str, object] = {}
    for path in files:
        if path.exists():
            file_records[str(path)] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        else:
            file_records[str(path)] = {"missing": True}
    qvq_header = qvq_repo / "gptqmodel_ext/qvq/p32/qvq_p32_abi.h"
    qvq_pin = zml_repo / "third_party/qvq/repo.bzl"
    return {
        "run_id": args.run_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "RUNFILES_DIR": os.environ.get("RUNFILES_DIR"),
            "ZML_LLAMA_TRACE": os.environ.get("ZML_LLAMA_TRACE"),
            "ZML_LLAMA_EAGER": os.environ.get("ZML_LLAMA_EAGER"),
            "ZML_LLAMA_GRAPH_MODE": os.environ.get("ZML_LLAMA_GRAPH_MODE"),
            "GPU_ALLOCATOR_LEASE_ID": os.environ.get("GPU_ALLOCATOR_LEASE_ID"),
        },
        "repos": {
            "zml": {
                "path": str(zml_repo),
                "head": run_text(["git", "rev-parse", "HEAD"], zml_repo),
                "remote": run_text(["git", "remote", "get-url", "origin"], zml_repo),
                "status": run_text(["git", "status", "--short"], zml_repo),
            },
            "qvq": {
                "path": str(qvq_repo),
                "head": run_text(["git", "rev-parse", "HEAD"], qvq_repo),
                "remote": run_text(["git", "remote", "get-url", "origin"], qvq_repo),
                "status": run_text(["git", "status", "--short"], qvq_repo),
            },
            "zml_qvq_pin": qvq_pin.read_text(encoding="utf-8") if qvq_pin.exists() else None,
            "qvq_abi_header": qvq_header.read_text(encoding="utf-8") if qvq_header.exists() else None,
        },
        "model": {
            "path": str(model_path),
            "metadata": read_model_config(model_path),
        },
        "commands": {
            "build": args.build_command,
            "benchmark": " ".join(sys.argv),
        },
        "software": {
            "nvidia_smi": run_text(["nvidia-smi", "-q"], timeout=60),
            "nvcc": run_text(["nvcc", "--version"]),
            "gcc": run_text(["gcc", "--version"]),
            "ldd_lib": run_text(["ldd", str(args.lib)]),
            "pip_freeze": run_text([sys.executable, "-m", "pip", "freeze"], timeout=60),
        },
        "files": file_records,
    }


def gpu_snapshot() -> dict[str, object]:
    return {
        "gpu": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "compute_apps": run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ]
        ),
    }


def parse_gpu_csv(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 8:
            rows.append(
                {
                    "index": parts[0],
                    "pci_bus_id": parts[1],
                    "uuid": parts[2],
                    "name": parts[3],
                    "memory_total": parts[4],
                    "memory_used": parts[5],
                    "memory_free": parts[6],
                    "utilization_gpu": parts[7],
                }
            )
    return rows


def idle_gate(samples: int, max_memory_mib: int, output_dir: Path, label: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    consecutive_idle = 0
    max_attempts = samples * 10
    for attempt in range(max_attempts):
        snap = gpu_snapshot()
        records.append(snap)
        rows = parse_gpu_csv(str(snap["gpu"].get("stdout", "")))
        apps = str(snap["compute_apps"].get("stdout", "")).strip()
        sample_ok = False
        if rows:
            util_text = rows[0]["utilization_gpu"].split()[0]
            used_text = rows[0]["memory_used"].split()[0]
            sample_ok = int(util_text) == 0 and int(used_text) <= max_memory_mib and not apps
        consecutive_idle = consecutive_idle + 1 if sample_ok else 0
        if consecutive_idle == samples:
            break
        if attempt + 1 < max_attempts:
            time.sleep(1)
    ok = consecutive_idle == samples
    result = {
        "label": label,
        "ok": ok,
        "required_consecutive_idle_samples": samples,
        "max_attempts": max_attempts,
        "max_memory_mib": max_memory_mib,
        "samples": records,
    }
    write_json(output_dir / f"idle_gate_{label}.json", result)
    if not ok:
        raise RuntimeError(f"idle gate failed for {label}")
    return result


def capture_idle_gate(expected_uuid: str, samples: int = 3, max_attempts: int = 30) -> dict[str, object]:
    records: list[dict[str, object]] = []
    consecutive_idle = 0
    for attempt in range(max_attempts):
        snap = gpu_snapshot()
        rows = parse_gpu_csv(str(snap["gpu"].get("stdout", "")))
        app_rows = [
            [part.strip() for part in line.split(",")]
            for line in str(snap["compute_apps"].get("stdout", "")).splitlines()
            if line.strip()
        ]
        target_rows = [row for row in rows if row["uuid"] == expected_uuid]
        foreign_apps = [
            {"gpu_uuid": row[0], "pid": row[1]}
            for row in app_rows
            if len(row) >= 2 and row[0] == expected_uuid and row[1] != str(os.getpid())
        ]
        util_percent = None
        sample_ok = False
        if len(target_rows) == 1:
            util_percent = int(target_rows[0]["utilization_gpu"].split()[0])
            sample_ok = util_percent == 0 and not foreign_apps
        records.append(
            {
                "attempt": attempt + 1,
                "gpu": target_rows,
                "compute_apps": app_rows,
                "foreign_apps": foreign_apps,
                "utilization_gpu_percent": util_percent,
                "sample_ok": sample_ok,
            }
        )
        consecutive_idle = consecutive_idle + 1 if sample_ok else 0
        if consecutive_idle == samples:
            break
        if attempt + 1 < max_attempts:
            time.sleep(1)
    result = {
        "ok": consecutive_idle == samples,
        "expected_uuid": expected_uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "profiled_pid": os.getpid(),
        "required_consecutive_idle_samples": samples,
        "max_attempts": max_attempts,
        "samples": records,
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES mapping mismatch: expected {expected_uuid}, "
            f"found {os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    if not result["ok"]:
        raise RuntimeError("pre-capture idle gate failed")
    return result


def u32_array(values: list[int]) -> object:
    return (C_U32 * len(values))(*values)


def i32_array(values: list[int]) -> object:
    return (C_I32 * len(values))(*values)


def token_for(seq: int, pos: int, vocab_size: int, seed: int) -> int:
    usable = min(vocab_size - 2000, 120000)
    return 1000 + ((seq * 1000003 + pos * 9176 + seed * 7919) % usable)


def make_block_table(batch: int, pages_per_seq: int) -> object:
    return i32_array([seq * pages_per_seq + page for seq in range(batch + 1) for page in range(pages_per_seq)])


def make_prefill_plan(
    *,
    batch: int,
    context: int,
    capacity: int,
    vocab_size: int,
    seed: int,
) -> tuple[list[StepArrays], list[str], dict[str, object]]:
    if capacity < batch:
        raise ValueError("capacity must be >= batch")
    positions = [0] * batch
    digests = [hashlib.sha256() for _ in range(batch)]
    steps: list[StepArrays] = []
    base = capacity // batch
    remainder = capacity % batch
    total_calls = batch * ((context + capacity - 1) // capacity)
    for call in range(total_calls):
        counts = [base] * batch
        for idx in range(remainder):
            counts[(call + idx) % batch] += 1
        for seq, pos in enumerate(positions):
            if pos + counts[seq] > context:
                counts[seq] = max(0, context - pos)
        live_rows = sum(counts)
        if live_rows == 0:
            break
        tokens: list[int] = []
        pos_values: list[int] = []
        slot_values: list[int] = []
        starts = [0]
        indices: list[int] = []
        seq_lens: list[int] = []
        for seq, count in enumerate(counts):
            start_row = len(tokens)
            if count == 0:
                raise ValueError(f"zero-row sequence in prefill call {call} seq {seq}")
            for offset in range(count):
                pos = positions[seq] + offset
                token = token_for(seq, pos, vocab_size, seed)
                tokens.append(token)
                pos_values.append(pos)
                slot_values.append(seq * context + pos)
                digests[seq].update(int(token).to_bytes(4, "little"))
            positions[seq] += count
            indices.append(start_row + count - 1)
            seq_lens.append(positions[seq])
            starts.append(len(tokens))
        padding = capacity - len(tokens)
        padding_base = batch * context
        for offset in range(padding):
            tokens.append(0)
            pos_values.append(offset)
            slot_values.append(padding_base + offset)
        starts.append(capacity)
        seq_lens.append(padding)
        steps.append(
            StepArrays(
                tokens=u32_array(tokens),
                positions=u32_array(pos_values),
                slots=u32_array(slot_values),
                sample_indices=u32_array(indices),
                seq_lens=i32_array(seq_lens),
                query_starts=i32_array(starts),
            )
        )
        if all(pos == context for pos in positions):
            break
    if positions != [context] * batch:
        raise RuntimeError(f"prefill plan did not reach context: {positions[:8]}")
    return steps, [digest.hexdigest() for digest in digests], {
        "batch": batch,
        "context": context,
        "capacity": capacity,
        "prefill_calls": len(steps),
        "base_rows_per_sequence": base,
        "rotating_extra_rows": remainder,
        "prompt_tokens": batch * context,
    }


def make_validation_plan(batch: int, context: int, capacity: int, vocab_size: int) -> list[StepArrays]:
    positions = [0] * batch
    steps: list[StepArrays] = []
    while min(positions) < context:
        remaining = [context - pos for pos in positions]
        counts = [min(capacity // batch, rem) for rem in remaining]
        extra = capacity - sum(counts)
        cursor = 0
        while extra > 0 and any(remaining[seq] > counts[seq] for seq in range(batch)):
            seq = cursor % batch
            if remaining[seq] > counts[seq]:
                counts[seq] += 1
                extra -= 1
            cursor += 1
        tokens: list[int] = []
        pos_values: list[int] = []
        slot_values: list[int] = []
        starts = [0]
        indices: list[int] = []
        seq_lens: list[int] = []
        for seq, count in enumerate(counts):
            if count == 0:
                raise RuntimeError("validation plan made an empty live sequence")
            start_row = len(tokens)
            for offset in range(count):
                pos = positions[seq] + offset
                tokens.append(token_for(seq, pos, vocab_size, 17))
                pos_values.append(pos)
                slot_values.append(seq * context + pos)
            positions[seq] += count
            indices.append(start_row + count - 1)
            seq_lens.append(positions[seq])
            starts.append(len(tokens))
        padding = capacity - len(tokens)
        for offset in range(padding):
            tokens.append(0)
            pos_values.append(offset)
            slot_values.append(batch * context + offset)
        starts.append(capacity)
        seq_lens.append(padding)
        steps.append(
            StepArrays(
                tokens=u32_array(tokens),
                positions=u32_array(pos_values),
                slots=u32_array(slot_values),
                sample_indices=u32_array(indices),
                seq_lens=i32_array(seq_lens),
                query_starts=i32_array(starts),
            )
        )
    return steps


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize_seconds(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
        "mean_s": statistics.fmean(values) if values else None,
        "median_s": statistics.median(values) if values else None,
        "p50_s": percentile(values, 0.50),
        "p90_s": percentile(values, 0.90),
        "p95_s": percentile(values, 0.95),
    }


def run_validation_child(args: argparse.Namespace) -> dict[str, object]:
    model_path = Path(args.model)
    context = int(args.context)
    batch = int(args.batch)
    capacity = int(args.capacity)
    vocab_size = int(read_model_config(model_path)["config"]["vocab_size"])
    pages_per_seq = (context + 15) // 16
    native = NativeLlama(Path(args.lib))
    create_start = time.perf_counter()
    native.create(model_path, context, batch, capacity)
    create_s = time.perf_counter() - create_start
    block_table = make_block_table(batch, pages_per_seq)
    output = (C_U32 * batch)()
    prefill_plan = make_validation_plan(batch, context, capacity, vocab_size)
    prefill_outputs: list[list[int]] = []
    for step in prefill_plan:
        native.step(prefill=True, step=step, block_table=block_table, output=output)
        prefill_outputs.append([int(output[idx]) for idx in range(batch)])
    decode_tokens = (C_U32 * batch)(*[int(output[idx]) for idx in range(batch)])
    decode_step = StepArrays(
        tokens=decode_tokens,
        positions=u32_array([context - 1] * batch),
        slots=u32_array([seq * context + context - 1 for seq in range(batch)]),
        sample_indices=u32_array(list(range(batch))),
        seq_lens=i32_array([context] * batch + [0]),
        query_starts=i32_array(list(range(batch + 1)) + [batch]),
    )
    decode_outputs: list[list[int]] = []
    for _ in range(int(args.decode_samples)):
        native.step(prefill=False, step=decode_step, block_table=block_table, output=output)
        row = [int(output[idx]) for idx in range(batch)]
        decode_outputs.append(row)
        for idx, token in enumerate(row):
            decode_tokens[idx] = C_U32(token)
    native.destroy()
    return {
        "status": "ok",
        "mode": args.mode,
        "abi_version": native.abi_version(),
        "context": context,
        "batch": batch,
        "capacity": capacity,
        "create_s": create_s,
        "prefill_outputs": prefill_outputs,
        "decode_outputs": decode_outputs,
    }


def run_batch_child(args: argparse.Namespace) -> dict[str, object]:
    model_path = Path(args.model)
    context = int(args.context)
    batch = int(args.batch)
    capacity = int(args.capacity)
    decode_samples = int(args.decode_samples)
    vocab_size = int(read_model_config(model_path)["config"]["vocab_size"])
    pages_per_seq = (context + 15) // 16
    plan_start = time.perf_counter()
    prefill_steps, prompt_hashes, plan = make_prefill_plan(
        batch=batch,
        context=context,
        capacity=capacity,
        vocab_size=vocab_size,
        seed=int(args.seed),
    )
    plan_s = time.perf_counter() - plan_start
    native = NativeLlama(Path(args.lib))
    create_start = time.perf_counter()
    native.create(model_path, context, batch, capacity)
    create_s = time.perf_counter() - create_start
    block_table = make_block_table(batch, pages_per_seq)
    output = (C_U32 * batch)()
    warmup_step = prefill_steps[0]
    warmup_start = time.perf_counter()
    native.step(prefill=True, step=warmup_step, block_table=block_table, output=output)
    warmup_prefill_s = time.perf_counter() - warmup_start
    progress_at = time.time() + 60.0
    prefill_latencies: list[float] = []
    prefill_start = time.perf_counter()
    for idx, step in enumerate(prefill_steps):
        step_start = time.perf_counter()
        native.step(prefill=True, step=step, block_table=block_table, output=output)
        prefill_latencies.append(time.perf_counter() - step_start)
        if time.time() >= progress_at:
            elapsed = time.perf_counter() - prefill_start
            done_tokens = (idx + 1) * capacity
            print(
                json.dumps(
                    {
                        "progress": True,
                        "batch": batch,
                        "phase": "prefill",
                        "calls_done": idx + 1,
                        "calls_total": len(prefill_steps),
                        "elapsed_s": elapsed,
                        "tokens_per_s": done_tokens / elapsed if elapsed > 0 else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            progress_at = time.time() + 60.0
    prefill_s = time.perf_counter() - prefill_start
    first_generated = [int(output[idx]) for idx in range(batch)]
    decode_tokens = (C_U32 * batch)(*first_generated)
    decode_step = StepArrays(
        tokens=decode_tokens,
        positions=u32_array([context - 1] * batch),
        slots=u32_array([seq * context + context - 1 for seq in range(batch)]),
        sample_indices=u32_array(list(range(batch))),
        seq_lens=i32_array([context] * batch + [0]),
        query_starts=i32_array(list(range(batch + 1)) + [batch]),
    )
    decode_warmup_latencies: list[float] = []
    for _ in range(int(args.decode_warmup)):
        step_start = time.perf_counter()
        native.step(prefill=False, step=decode_step, block_table=block_table, output=output)
        decode_warmup_latencies.append(time.perf_counter() - step_start)
        for idx in range(batch):
            decode_tokens[idx] = C_U32(int(output[idx]))
    decode_latencies: list[float] = []
    decode_outputs: list[list[int]] = []
    profile_decode_samples = int(os.environ.get("ZML_PROFILE_DECODE_SAMPLES", "0"))
    if profile_decode_samples < 0 or profile_decode_samples > decode_samples:
        raise ValueError("ZML_PROFILE_DECODE_SAMPLES must be between 0 and --decode-samples")
    profiler = CudaProfiler() if profile_decode_samples else None
    capture_gate = None
    capture_active = False
    try:
        for sample in range(decode_samples):
            if profiler is not None and sample == 0:
                capture_gate = capture_idle_gate(
                    os.environ.get(
                        "ZML_PROFILE_GPU_UUID",
                        "GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2",
                    )
                )
                print(json.dumps({"capture_idle_gate": capture_gate}, sort_keys=True), flush=True)
                print(
                    json.dumps(
                        {
                            "profile_capture": "start",
                            "batch": batch,
                            "context": context,
                            "decode_samples": profile_decode_samples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                profiler.start()
                capture_active = True
            step_start = time.perf_counter()
            native.step(prefill=False, step=decode_step, block_table=block_table, output=output)
            decode_latencies.append(time.perf_counter() - step_start)
            row = [int(output[idx]) for idx in range(batch)]
            decode_outputs.append(row)
            for idx, token in enumerate(row):
                decode_tokens[idx] = C_U32(token)
            if profiler is not None and sample + 1 == profile_decode_samples:
                profiler.stop()
                capture_active = False
                print(
                    json.dumps(
                        {
                            "profile_capture": "stop",
                            "batch": batch,
                            "context": context,
                            "decode_samples": profile_decode_samples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if time.time() >= progress_at:
                print(
                    json.dumps(
                        {
                            "progress": True,
                            "batch": batch,
                            "phase": "decode",
                            "samples_done": sample + 1,
                            "samples_total": decode_samples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                progress_at = time.time() + 60.0
    finally:
        if profiler is not None and capture_active:
            profiler.stop()
    native.destroy()
    prompt_tokens = int(plan["prompt_tokens"])
    return {
        "status": "ok",
        "abi_version": native.abi_version(),
        "batch": batch,
        "context": context,
        "effective_prompt_tokens_per_sequence": context,
        "capacity": capacity,
        "pages_per_sequence": pages_per_seq,
        "plan": plan,
        "plan_s": plan_s,
        "create_s": create_s,
        "warmup_prefill_s": warmup_prefill_s,
        "prefill": {
            "prompt_tokens": prompt_tokens,
            "calls": len(prefill_steps),
            "wall_s": prefill_s,
            "ttft_s": prefill_s,
            "tokens_per_s": prompt_tokens / prefill_s if prefill_s > 0 else None,
            "chunk_latency_s": summarize_seconds(prefill_latencies),
            "chunk_latencies_s": prefill_latencies,
        },
        "decode": {
            "samples": decode_samples,
            "warmup_samples": int(args.decode_warmup),
            "warmup_latencies_s": decode_warmup_latencies,
            "latency_s": summarize_seconds(decode_latencies),
            "latencies_s": decode_latencies,
            "mean_tokens_per_s": batch / statistics.fmean(decode_latencies) if decode_latencies else None,
            "p50_tokens_per_s": batch / statistics.median(decode_latencies) if decode_latencies else None,
        },
        "prompt_hashes_sha256_u32le_per_sequence": prompt_hashes,
        "first_generated_tokens": first_generated,
        "decode_outputs": decode_outputs,
        "capture_idle_gate": capture_gate,
        "gpu_after": gpu_snapshot(),
        "decode_method_note": "decode writes generated tokens at position context-1 and overwrites that final KV slot on repeated samples so sequence length remains equal to the model context limit",
    }


def child_main(args: argparse.Namespace) -> int:
    output_path = Path(args.output_json)
    started = time.time()
    try:
        if args.child_kind == "batch":
            result = run_batch_child(args)
        else:
            result = run_validation_child(args)
        result["started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
        result["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["elapsed_s"] = time.time() - started
        write_json(output_path, result)
        print(json.dumps({"child_result": str(output_path), "status": result["status"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": time.time() - started,
            "gpu_after_error": gpu_snapshot(),
        }
        write_json(output_path, result)
        print(json.dumps({"child_result": str(output_path), "status": "failed", "error": str(exc)}, sort_keys=True), flush=True)
        return 1


def classify_failure(returncode: int, stderr: str, stdout: str) -> str:
    text = (stderr + "\n" + stdout).lower()
    if returncode in {-signal.SIGKILL, 137}:
        return "oom_or_killed"
    markers = [
        "out of memory",
        "cuda_error_out_of_memory",
        "resource exhausted",
        "allocation failed",
        "std.mem.alloc",
        "cannot allocate memory",
    ]
    if any(marker in text for marker in markers):
        return "oom"
    return "failed"


def run_child(args: argparse.Namespace, name: str, child_args: list[str], env_updates: dict[str, str | None]) -> dict[str, object]:
    stdout_path = Path(args.output_dir) / f"{name}.stdout.log"
    stderr_path = Path(args.output_dir) / f"{name}.stderr.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RUNFILES_DIR"] = str(Path(args.runner).with_suffix("").with_name(Path(args.runner).name + ".runfiles"))
    for key, value in env_updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    cmd = [sys.executable, str(Path(__file__)), "--child", *child_args]
    start = time.time()
    print(json.dumps({"starting_child": name, "cmd": cmd, "env_updates": env_updates}, sort_keys=True), flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle, text=True, env=env)
        progress_at = time.time() + 60.0
        while proc.poll() is None:
            time.sleep(1)
            if time.time() >= progress_at:
                print(
                    json.dumps(
                        {
                            "child_progress": name,
                            "elapsed_s": time.time() - start,
                            "stdout_log": str(stdout_path),
                            "stderr_log": str(stderr_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                progress_at = time.time() + 60.0
        returncode = proc.wait()
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return {
        "name": name,
        "cmd": cmd,
        "returncode": returncode,
        "classification": "ok" if returncode == 0 else classify_failure(returncode, stderr, stdout),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "elapsed_s": time.time() - start,
    }


def compare_validation(graph_path: Path, eager_path: Path) -> dict[str, object]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    eager = json.loads(eager_path.read_text(encoding="utf-8"))
    same = graph.get("prefill_outputs") == eager.get("prefill_outputs") and graph.get("decode_outputs") == eager.get("decode_outputs")
    return {
        "status": "ok" if same else "failed",
        "same_outputs": same,
        "graph_path": str(graph_path),
        "eager_path": str(eager_path),
        "graph_outputs": {
            "prefill_last": graph.get("prefill_outputs", [])[-1:] if graph.get("prefill_outputs") else [],
            "decode": graph.get("decode_outputs"),
        },
        "eager_outputs": {
            "prefill_last": eager.get("prefill_outputs", [])[-1:] if eager.get("prefill_outputs") else [],
            "decode": eager.get("decode_outputs"),
        },
    }


def write_markdown_summary(output_dir: Path, results: dict[str, object]) -> Path:
    lines = [
        "# ZML native Fast ABI F6 Seed7 benchmark",
        "",
        f"Run ID: `{results['run_id']}`",
        f"Model: `{results['model_path']}`",
        f"Context: `{results['context']}`",
        f"Capacity: `{results['capacity']}`",
        "",
        "| Batch | Status | TTFT / prefill s | Prefill tok/s | Decode mean ms | Decode tok/s | Notes |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results["batches"]:
        if item.get("status") == "ok":
            prefill = item["prefill"]
            decode = item["decode"]
            lines.append(
                "| {batch} | ok | {ttft:.6f} | {prefill_tps:.3f} | {decode_ms:.6f} | {decode_tps:.3f} | {notes} |".format(
                    batch=item["batch"],
                    ttft=float(prefill["ttft_s"]),
                    prefill_tps=float(prefill["tokens_per_s"]),
                    decode_ms=float(decode["latency_s"]["mean_s"]) * 1000.0,
                    decode_tps=float(decode["mean_tokens_per_s"]),
                    notes="",
                )
            )
        else:
            lines.append(
                "| {batch} | {status} |  |  |  |  | {notes} |".format(
                    batch=item.get("batch"),
                    status=item.get("status"),
                    notes=item.get("classification") or item.get("error") or "",
                )
            )
    lines.extend(
        [
            "",
            "TTFT is timed as the wall-clock sum of native ABI prefill steps needed to consume the full maximum-context prompt and return the first generated token.",
            "Decode is measured at the model context limit by writing generated tokens at the final valid position and overwriting that final KV slot for repeated samples.",
            "Prompt hashes are SHA-256 over little-endian uint32 token IDs per sequence.",
            "",
            f"Raw JSON: `{output_dir / 'results.json'}`",
        ]
    )
    path = output_dir / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def orchestrator_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    provenance = collect_provenance(args)
    write_json(output_dir / "provenance.json", provenance)
    idle_gate(3, int(args.idle_memory_mib), output_dir, "before_validation")
    validation_graph_path = output_dir / "validation_graph.json"
    validation_eager_path = output_dir / "validation_eager.json"
    common_validation = [
        "--child-kind",
        "validation",
        "--mode",
        "validation",
        "--model",
        args.model,
        "--lib",
        args.lib,
        "--context",
        "256",
        "--batch",
        "3",
        "--capacity",
        "64",
        "--decode-samples",
        "5",
    ]
    graph_run = run_child(args, "validation_graph", [*common_validation, "--output-json", str(validation_graph_path)], {"ZML_LLAMA_EAGER": None, "ZML_LLAMA_TRACE": None})
    if graph_run["returncode"] != 0:
        raise RuntimeError(f"graph validation failed: {graph_run}")
    eager_run = run_child(args, "validation_eager", [*common_validation, "--output-json", str(validation_eager_path)], {"ZML_LLAMA_EAGER": "1", "ZML_LLAMA_TRACE": None})
    if eager_run["returncode"] != 0:
        raise RuntimeError(f"eager validation failed: {eager_run}")
    validation_compare = compare_validation(validation_graph_path, validation_eager_path)
    write_json(output_dir / "validation_compare.json", validation_compare)
    if validation_compare["status"] != "ok":
        raise RuntimeError("eager and command-buffer validation outputs differ")
    trace_path = output_dir / "validation_trace.json"
    trace_run = run_child(args, "validation_trace", [*common_validation, "--output-json", str(trace_path)], {"ZML_LLAMA_EAGER": None, "ZML_LLAMA_TRACE": "1"})
    idle_gate(3, int(args.idle_memory_mib), output_dir, "before_benchmark")
    batches: list[dict[str, object]] = []
    for batch in [int(part) for part in args.batches.split(",") if part.strip()]:
        child_path = output_dir / f"batch_{batch}.json"
        child_args = [
            "--child-kind",
            "batch",
            "--mode",
            "batch",
            "--model",
            args.model,
            "--lib",
            args.lib,
            "--context",
            str(args.context),
            "--batch",
            str(batch),
            "--capacity",
            str(args.capacity),
            "--decode-samples",
            str(args.decode_samples),
            "--decode-warmup",
            str(args.decode_warmup),
            "--seed",
            str(args.seed + batch),
            "--output-json",
            str(child_path),
        ]
        idle_gate(3, int(args.idle_memory_mib), output_dir, f"before_batch_{batch}")
        run = run_child(args, f"batch_{batch}", child_args, {"ZML_LLAMA_EAGER": None, "ZML_LLAMA_TRACE": None})
        if run["returncode"] == 0 and child_path.exists():
            item = json.loads(child_path.read_text(encoding="utf-8"))
            item["run"] = run
        else:
            item = {
                "batch": batch,
                "status": "skipped" if run["classification"] in {"oom", "oom_or_killed"} else "failed",
                "classification": run["classification"],
                "run": run,
                "child_json": str(child_path),
            }
            if child_path.exists():
                item["child_result"] = json.loads(child_path.read_text(encoding="utf-8"))
        batches.append(item)
        write_json(output_dir / "results.partial.json", {"run_id": args.run_id, "batches": batches})
        print(
            json.dumps(
                {
                    "batch_done": batch,
                    "status": item.get("status"),
                    "classification": item.get("classification"),
                    "result_path": str(child_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if item.get("status") == "failed":
            raise RuntimeError(f"batch {batch} failed unexpectedly")
    results = {
        "run_id": args.run_id,
        "model_path": args.model,
        "context": int(args.context),
        "capacity": int(args.capacity),
        "batches_requested": args.batches,
        "validation": {
            "graph_run": graph_run,
            "eager_run": eager_run,
            "trace_run": trace_run,
            "compare": validation_compare,
            "trace_json": str(trace_path),
        },
        "batches": batches,
        "provenance_json": str(output_dir / "provenance.json"),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output_dir / "results.json", results)
    summary = write_markdown_summary(output_dir, results)
    print(json.dumps({"results_json": str(output_dir / "results.json"), "summary_md": str(summary)}, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--child-kind", choices=["validation", "batch"], default="batch")
    parser.add_argument("--mode", default="benchmark")
    parser.add_argument("--run-id", default="zml-native-fast-abi-f6-seed7-sm80-20260910T045313Z")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lib", required=True)
    parser.add_argument("--runner", default="/root/devin-worker/repos/ZML-Ultra/bazel-bin/examples/llm/llama_paged_token_runner")
    parser.add_argument("--zml-repo", default="/root/devin-worker/repos/ZML-Ultra")
    parser.add_argument("--qvq-repo", default="/root/devin-worker/repos/QvQ")
    parser.add_argument("--output-dir", default="/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908/benchmarks/zml_native_fast_abi__zml-native-fast-abi-f6-seed7-sm80-20260910T045313Z")
    parser.add_argument("--output-json", default="child.json")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batches", default="1,2,4,8,12,16,20,24,28,32")
    parser.add_argument("--decode-samples", type=int, default=20)
    parser.add_argument("--decode-warmup", type=int, default=3)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7307)
    parser.add_argument(
        "--build-command",
        default=(
            "PATH=/usr/local/cuda/bin:$PATH ./bazel.sh build --config=release "
            "--jobs=24 --local_resources=cpu=24 --local_resources=memory=192000 "
            "--override_repository=qvq=/root/devin-worker/repos/QvQ "
            "--spawn_strategy=local --@zml//platforms:cuda=true --@zml//platforms:cpu=false "
            "//examples/llm:llama_native //examples/llm:llama_paged_token_runner"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        return child_main(args)
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
