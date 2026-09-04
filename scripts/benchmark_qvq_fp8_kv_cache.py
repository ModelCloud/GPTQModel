# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark QVQ A8 FP8 KV-cache storage, throughput, and peak H200 VRAM."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "w35-a16", "w35-a8"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-row-start", type=int, default=256)
    parser.add_argument("--dataset-rows", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--decode-warmup", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--prefill-warmup", type=int, default=2)
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-memory-mib", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _run_smi(query: str) -> list[list[str]]:
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in output.splitlines()
        if line.strip()
    ]


def _physical_gpu(expected_uuid: str) -> dict[str, str]:
    rows = _run_smi(
        "index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu"
    )
    matches = [row for row in rows if row[2] == expected_uuid]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one physical GPU with UUID {expected_uuid}, found {len(matches)}."
        )
    row = matches[0]
    return {
        "physical_index": row[0],
        "pci_bus_id": row[1],
        "uuid": row[2],
        "name": row[3],
        "memory_total_mib": row[4],
        "memory_used_mib": row[5],
        "utilization_gpu_percent": row[6],
    }


def _compute_processes(expected_uuid: str) -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    processes = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == expected_uuid:
            processes.append(
                {"gpu_uuid": fields[0], "pid": fields[1], "used_memory_mib": fields[2]}
            )
    return processes


def _idle_preflight(args: argparse.Namespace) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_gpu_uuid:
        raise RuntimeError(
            "Benchmark requires an allocator-restricted UUID mapping: "
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.expected_gpu_uuid!r}."
        )
    accepted = []
    for sample in range(args.idle_samples):
        gpu = _physical_gpu(args.expected_gpu_uuid)
        processes = _compute_processes(args.expected_gpu_uuid)
        used = int(gpu["memory_used_mib"])
        utilization = int(gpu["utilization_gpu_percent"])
        accepted.append({"sample": sample + 1, **gpu, "foreign_processes": processes})
        if processes or used > args.idle_memory_mib or utilization != 0:
            raise RuntimeError(
                "H200 idle preflight failed: "
                f"sample={sample + 1}, processes={processes}, used={used} MiB, utilization={utilization}%."
            )
        if sample + 1 < args.idle_samples:
            time.sleep(0.2)
    print(
        "[idle] accepted "
        f"index={accepted[-1]['physical_index']} pci={accepted[-1]['pci_bus_id']} "
        f"uuid={args.expected_gpu_uuid} name={accepted[-1]['name']} "
        f"samples={args.idle_samples} memory_limit={args.idle_memory_mib}MiB",
        flush=True,
    )
    return {"samples": accepted, "memory_limit_mib": args.idle_memory_mib}


def _exclusive_recheck(expected_uuid: str) -> dict:
    """Fail if another process appears after model setup and warmup."""
    gpu = _physical_gpu(expected_uuid)
    processes = _compute_processes(expected_uuid)
    current_pid = str(os.getpid())
    foreign = [process for process in processes if process["pid"] != current_pid]
    if foreign:
        raise RuntimeError(
            f"H200 exclusivity recheck found foreign compute processes: {foreign}."
        )
    if not any(process["pid"] == current_pid for process in processes):
        raise RuntimeError(
            "H200 exclusivity recheck could not find the benchmark CUDA process."
        )
    result = {
        **gpu,
        "benchmark_pid": current_pid,
        "compute_processes": processes,
        "foreign_processes": foreign,
    }
    print(
        "[exclusive] accepted "
        f"pci={gpu['pci_bus_id']} uuid={expected_uuid} pid={current_pid}",
        flush=True,
    )
    return result


class _NvidiaSmiProcessSampler:
    def __init__(self, gpu_uuid: str, interval_seconds: float = 0.05):
        self.gpu_uuid = gpu_uuid
        self.interval_seconds = interval_seconds
        self.pid = str(os.getpid())
        self.peak_process_memory_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            for process in _compute_processes(self.gpu_uuid):
                if process["pid"] == self.pid:
                    self.peak_process_memory_mib = max(
                        self.peak_process_memory_mib,
                        int(process["used_memory_mib"]),
                    )
            self.samples += 1
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join(timeout=2)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _event_statistics(events) -> dict[str, float]:
    milliseconds = [start.elapsed_time(end) for start, end in events]
    return {
        "samples": len(milliseconds),
        "mean_ms": statistics.mean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "p95_ms": _percentile(milliseconds, 0.95),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
    }


def _cache_telemetry(cache) -> dict:
    import torch

    if hasattr(cache, "telemetry"):
        return cache.telemetry()
    layers = getattr(cache, "layers", [])
    tensors = []
    layer_rows = []
    for index, layer in enumerate(layers):
        row = {
            "layer": index,
            "initialized": bool(getattr(layer, "is_initialized", False)),
        }
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                size = tensor.numel() * tensor.element_size()
                tensors.append(tensor)
                row[f"{name}_dtype"] = str(tensor.dtype)
                row[f"{name}_shape"] = list(tensor.shape)
                row[f"{name}_bytes"] = size
        layer_rows.append(row)
    dtype_bytes = {}
    for tensor in tensors:
        name = str(tensor.dtype)
        dtype_bytes[name] = (
            dtype_bytes.get(name, 0) + tensor.numel() * tensor.element_size()
        )
    return {
        "schema": "transformers.kv-cache.v1",
        "class": type(cache).__name__,
        "layers": layer_rows,
        "layer_count": len(layers),
        "initialized_layer_count": sum(row["initialized"] for row in layer_rows),
        "sequence_lengths": sorted(
            {
                int(layer.get_seq_length())
                for layer in layers
                if getattr(layer, "is_initialized", False)
            }
        ),
        "dtype_bytes": dtype_bytes,
        "storage_bytes": sum(dtype_bytes.values()),
    }


def _memory_snapshot(torch, device) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _memory_delta(snapshot: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    return {
        "allocated_delta_bytes": snapshot["allocated_bytes"]
        - baseline["allocated_bytes"],
        "reserved_delta_bytes": snapshot["reserved_bytes"] - baseline["reserved_bytes"],
        "peak_allocated_delta_bytes": snapshot["peak_allocated_bytes"]
        - baseline["allocated_bytes"],
        "peak_reserved_delta_bytes": snapshot["peak_reserved_bytes"]
        - baseline["reserved_bytes"],
    }


def _load_prompt(args, tokenizer) -> tuple[object, object]:
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.qvq_evaluate import _row_text
    from scripts.qvq_quantize import DatasetSlice, load_dataset_slice

    dataset = load_dataset_slice(
        DatasetSlice(
            args.dataset, None, "train", args.dataset_row_start, args.dataset_rows
        )
    )
    token_ids = []
    for row in dataset:
        text = _row_text(dict(row), tokenizer, None)
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
        token_ids.extend(encoded)
        if len(token_ids) >= args.prompt_length:
            break
    if len(token_ids) < args.prompt_length:
        raise RuntimeError(
            f"Dataset slice yielded {len(token_ids)} tokens, fewer than prompt length {args.prompt_length}."
        )
    input_ids = torch.tensor(
        [token_ids[: args.prompt_length]], dtype=torch.long, device="cuda:0"
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.arm != "dense" and not args.checkpoint:
        raise ValueError("Quantized arms require --checkpoint.")
    if (
        min(
            args.prompt_length,
            args.decode_steps,
            args.prefill_repeats,
            args.idle_samples,
        )
        < 1
    ):
        raise ValueError(
            "Prompt length, decode steps, prefill repeats, and idle samples must be positive."
        )
    idle_preflight = _idle_preflight(args)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qvq_fp8_cache import QVQFP8DynamicCache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after the H200 idle preflight.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if props.name != "NVIDIA H200" or props.major != 9:
        raise RuntimeError(
            f"Expected NVIDIA H200 compute capability 9.x, got {props.name} {props.major}.{props.minor}."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids, attention_mask = _load_prompt(args, tokenizer)
    load_started = time.perf_counter()
    if args.arm == "dense":
        loaded = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation=args.attention,
            local_files_only=True,
        )
        model = loaded.eval()
    else:
        loaded = GPTQModel.load(
            args.checkpoint,
            backend=BACKEND.QVQ,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation=args.attention,
            local_files_only=True,
        )
        model = loaded.model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    torch.cuda.empty_cache()
    model_memory = _memory_snapshot(torch, device)

    def run_prefill():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )

    for _ in range(args.prefill_warmup):
        warmup_output = run_prefill()
        del warmup_output
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()

    prefill_events = []
    for _ in range(args.prefill_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_output = run_prefill()
        end.record()
        prefill_events.append((start, end))
        del timed_output
    torch.cuda.synchronize(device)
    prefill_timing = _event_statistics(prefill_events)
    prefill_timing["tokens_per_second"] = args.prompt_length / (
        prefill_timing["median_ms"] / 1000.0
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before_formal = _memory_snapshot(torch, device)
    exclusive_recheck = _exclusive_recheck(args.expected_gpu_uuid)

    with _NvidiaSmiProcessSampler(args.expected_gpu_uuid) as sampler:
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()
        output = run_prefill()
        prefill_end.record()
        torch.cuda.synchronize(device)
        formal_prefill_ms = prefill_start.elapsed_time(prefill_end)
        cache = output.past_key_values
        cache_after_prefill = _cache_telemetry(cache)
        if args.arm == "w35-a8":
            if not isinstance(cache, QVQFP8DynamicCache):
                raise RuntimeError(
                    f"A8 produced {type(cache).__name__}, not QVQFP8DynamicCache."
                )
            cache.assert_fp8_storage()
            if (
                not cache_after_prefill["all_payloads_fp8"]
                or not cache_after_prefill["no_full_precision_residual"]
            ):
                raise RuntimeError(
                    "A8 KV-cache telemetry did not prove exclusive FP8 payload storage."
                )
        elif isinstance(cache, QVQFP8DynamicCache):
            raise RuntimeError(f"{args.arm} unexpectedly enabled the QVQ FP8 cache.")
        after_prefill = _memory_snapshot(torch, device)

        next_token = output.logits[:, -1:].argmax(dim=-1)
        del output
        for _ in range(args.decode_warmup):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
            del output
        torch.cuda.synchronize(device)
        after_decode_warmup = _memory_snapshot(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        before_measured_decode = _memory_snapshot(torch, device)

        decode_events = []
        for _ in range(args.decode_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            end.record()
            decode_events.append((start, end))
            cache = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
            del output
        torch.cuda.synchronize(device)
        decode_timing = _event_statistics(decode_events)
        decode_timing["tokens_per_second"] = 1000.0 / decode_timing["mean_ms"]
        cache_after_decode = _cache_telemetry(cache)
        if args.arm == "w35-a8":
            cache.assert_fp8_storage()
            if cache_after_decode["sequence_lengths"] != [
                args.prompt_length + args.decode_warmup + args.decode_steps
            ]:
                raise RuntimeError(
                    "A8 FP8 cache sequence length did not track the complete generation."
                )
        after_decode = _memory_snapshot(torch, device)
    sampled_peak_process_memory_mib = sampler.peak_process_memory_mib
    sampled_peak_count = sampler.samples

    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()[int(_physical_gpu(args.expected_gpu_uuid)["physical_index"])]
    report = {
        "schema": "qvq.fp8-kv-cache-benchmark.v1",
        "arm": args.arm,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "workload": {
            "batch_size": 1,
            "prompt_tokens": args.prompt_length,
            "decode_warmup_tokens": args.decode_warmup,
            "measured_decode_tokens": args.decode_steps,
            "attention": args.attention,
            "dtype": "torch.bfloat16",
            "dataset": args.dataset,
            "dataset_row_start": args.dataset_row_start,
            "dataset_rows": args.dataset_rows,
        },
        "throughput": {
            "prefill": prefill_timing,
            "formal_prefill_ms": formal_prefill_ms,
            "decode": decode_timing,
        },
        "memory": {
            "model_loaded": model_memory,
            "before_formal": before_formal,
            "after_prefill": after_prefill,
            "prefill_delta_from_baseline": _memory_delta(after_prefill, before_formal),
            "after_decode_warmup": after_decode_warmup,
            "before_measured_decode": before_measured_decode,
            "after_decode": after_decode,
            "measured_decode_delta_from_baseline": _memory_delta(
                after_decode, before_measured_decode
            ),
            "whole_workload_peak_allocated_bytes": max(
                after_prefill["peak_allocated_bytes"],
                after_decode_warmup["peak_allocated_bytes"],
                after_decode["peak_allocated_bytes"],
            ),
            "whole_workload_peak_reserved_bytes": max(
                after_prefill["peak_reserved_bytes"],
                after_decode_warmup["peak_reserved_bytes"],
                after_decode["peak_reserved_bytes"],
            ),
            "sampled_nvml_process_peak_mib": sampled_peak_process_memory_mib,
            "sampled_nvml_interval_ms": sampler.interval_seconds * 1000,
            "sampled_nvml_samples": sampled_peak_count,
        },
        "kv_cache": {
            "after_prefill": cache_after_prefill,
            "after_decode": cache_after_decode,
        },
        "runtime": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
            "git_worktree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=repo_root, text=True
                ).strip()
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "driver": driver.strip(),
            "load_seconds": load_seconds,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu": _physical_gpu(args.expected_gpu_uuid),
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        },
        "idle_preflight": idle_preflight,
        "exclusive_recheck": exclusive_recheck,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
