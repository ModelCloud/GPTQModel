#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Run a serial dense/Classic GPTQ/AdjacentExact-hybrid whole-model A/B."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.quantization.adjacent_model import AdjacentModelConfig  # noqa: E402
from gptqmodel.utils.adjacent_exact import prewarm_adjacent_exact_cuda  # noqa: E402
from gptqmodel.utils.paroquant_benchmark import load_nm_calibration  # noqa: E402
from tests.eval import evaluate, format_eval_result_table, get_eval_task_results  # noqa: E402


MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
CALIBRATION_ROWS = 512
CALIBRATION_CONCAT_SIZE = 2048
QUANT_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 64
SEED = 898
TASKS = ("arc_challenge", "gsm8k_platinum_cot")


def _run_text(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _parse_mib(value: str) -> int | None:
    token = value.strip().split()[0] if value.strip() else ""
    try:
        return int(token)
    except ValueError:
        return None


class ResourceSampler:
    """Sample process RSS and NVML-backed GPU memory without changing the quantizer implementation."""

    def __init__(self, *, gpu_uuid: str, interval_seconds: float = 0.25):
        self.gpu_uuid = gpu_uuid
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.samples: list[dict[str, float | int | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _gpu_memory(self) -> tuple[int | None, int | None]:
        device_value = _run_text(
            [
                "nvidia-smi",
                "-i",
                self.gpu_uuid,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        device_used = _parse_mib(device_value)
        process_rows = _run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        process_used = 0
        process_seen = False
        for row in process_rows.splitlines():
            parts = [part.strip() for part in row.split(",")]
            if len(parts) != 2 or parts[0] != str(self.pid):
                continue
            parsed = _parse_mib(parts[1])
            if parsed is not None:
                process_used += parsed
                process_seen = True
        return device_used, process_used if process_seen else None

    def _sample(self) -> None:
        try:
            device_used, process_used = self._gpu_memory()
            self.samples.append(
                {
                    "monotonic_seconds": time.monotonic(),
                    "rss_bytes": self.process.memory_info().rss,
                    "device_used_mib": device_used,
                    "process_gpu_used_mib": process_used,
                }
            )
        except (OSError, psutil.Error):
            return

    def _loop(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()
        self._sample()

    def start(self) -> None:
        """Start the sampling thread after model and calibration setup."""

        if self._thread is not None:
            raise RuntimeError("ResourceSampler is already running.")
        self._thread = threading.Thread(target=self._loop, name="adjacent-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return baseline, peak, and delta measurements."""

        if self._thread is None:
            raise RuntimeError("ResourceSampler was not started.")
        self._stop.set()
        self._thread.join()
        if not self.samples:
            return {"sample_count": 0}

        first = self.samples[0]

        def maximum(key: str) -> int | None:
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            return int(max(values)) if values else None

        peak_rss = maximum("rss_bytes")
        peak_device = maximum("device_used_mib")
        peak_process = maximum("process_gpu_used_mib")
        start_device = first["device_used_mib"]
        start_process = first["process_gpu_used_mib"]
        return {
            "sample_count": len(self.samples),
            "interval_seconds": self.interval_seconds,
            "start_rss_bytes": first["rss_bytes"],
            "peak_rss_bytes": peak_rss,
            "peak_rss_delta_bytes": peak_rss - int(first["rss_bytes"]) if peak_rss is not None else None,
            "start_device_used_mib": start_device,
            "peak_device_used_mib": peak_device,
            "peak_device_used_delta_mib": (
                peak_device - int(start_device)
                if peak_device is not None and start_device is not None
                else None
            ),
            "start_process_gpu_used_mib": start_process,
            "peak_process_gpu_used_mib": peak_process,
            "peak_process_gpu_used_delta_mib": (
                peak_process - int(start_process)
                if peak_process is not None and start_process is not None
                else None
            ),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_single_gpu() -> tuple[torch.device, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly one GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device, visible


def hardware_metadata(device: torch.device, gpu_uuid: str) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    driver = _run_text(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": gpu_uuid,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def adjacent_summary(module_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not module_stats:
        return {}
    classic = sum(float(item["classic_full_hessian_error"]) for item in module_stats)
    adjacent = sum(float(item["adjacent_full_hessian_error"]) for item in module_stats)
    hybrid = sum(float(item["hybrid_full_hessian_error"]) for item in module_stats)
    adjacent_times = [float(item["adjacent_model_wall_seconds"]) for item in module_stats]
    candidate_times = [float(item["candidate_phase_wall_seconds"]) for item in module_stats]
    executor_counts = {
        executor: sum(item["executor"] == executor for item in module_stats)
        for executor in ("cpu", "cuda")
    }
    return {
        "module_count": len(module_stats),
        "row_group_count": sum(int(item["row_group_count"]) for item in module_stats),
        "classic_full_hessian_error": classic,
        "adjacent_full_hessian_error": adjacent,
        "hybrid_full_hessian_error": hybrid,
        "hybrid_full_hessian_error_reduction": classic - hybrid,
        "hybrid_full_hessian_error_reduction_pct": 100.0 * (classic - hybrid) / classic,
        "hybrid_selected_rows": sum(int(item["hybrid_selected_rows"]) for item in module_stats),
        "hybrid_total_rows": sum(int(item["hybrid_total_rows"]) for item in module_stats),
        "coordinate_converged_row_groups": sum(
            int(item["coordinate_converged_row_groups"]) for item in module_stats
        ),
        "coordinate_capped_row_groups": sum(int(item["coordinate_capped_row_groups"]) for item in module_stats),
        "native_refinements_attempted": sum(int(item["native_refinements_attempted"]) for item in module_stats),
        "native_refinements_certified": sum(int(item["native_refinements_certified"]) for item in module_stats),
        "native_refinements_improved": sum(int(item["native_refinements_improved"]) for item in module_stats),
        "native_nodes_visited": sum(int(item["native_nodes_visited"]) for item in module_stats),
        "adjacent_model_wall_seconds_sum": sum(adjacent_times),
        "adjacent_model_wall_seconds_median": statistics.median(adjacent_times),
        "adjacent_model_wall_seconds_max": max(adjacent_times),
        "candidate_phase_wall_seconds_sum": sum(candidate_times),
        "executor_counts": executor_counts,
    }


def quantize_case(args: argparse.Namespace) -> None:
    device, gpu_uuid = require_single_gpu()
    set_seed(args.seed)
    if args.save_dir.exists() and any(args.save_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty checkpoint directory: {args.save_dir}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_nm_calibration(args.calibration_rows)
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        act_group_aware=True,
        sym=True,
        damp_percent=0.05,
        mse=0.0,
        pack_impl="cpu",
        offload_to_disk=True,
    )
    adjacent_model = None
    if args.method == "adjacent":
        adjacent_model = AdjacentModelConfig(
            coordinate_starts=("nearest", "zero", "one", "linear"),
            max_coordinate_flips=args.max_coordinate_flips,
            coordinate_rebase_interval=8,
            executor=args.executor,
            row_chunk_size=args.row_chunk_size,
            cpu_row_chunk_size=args.cpu_row_chunk_size,
            cpu_workers=args.cpu_workers,
            cpu_min_row_groups=args.cpu_min_row_groups,
            objective_row_chunk_size=args.objective_row_chunk_size,
            native_refinements_per_module=args.native_refinements_per_module,
            native_split_depth=args.native_split_depth,
            native_max_nodes_per_worker=args.native_max_nodes_per_worker,
            certificate_tolerance=1e-12,
            selection_tolerance=1e-7,
        )
        quantize_config.adjacent_model = adjacent_model
        prewarm_adjacent_exact_cuda()

    load_kwargs: dict[str, Any] = {}
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            load_kwargs["attn_implementation"] = "flash_attention_2"
    except Exception:
        pass

    model = GPTQModel.load(
        str(args.model),
        quantize_config=quantize_config,
        trust_remote_code=False,
        dtype="auto",
        **load_kwargs,
    )
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)
    start_reserved = torch.cuda.memory_reserved(device)
    sampler = ResourceSampler(gpu_uuid=gpu_uuid)
    sampler.start()
    quant_start = time.perf_counter()
    try:
        quant_logs = model.quantize(
            calibration,
            calibration_concat_size=args.calibration_concat_size,
            calibration_sort="desc",
            batch_size=args.quant_batch_size,
            backend=BACKEND.GPTQ_TORCH,
        )
        torch.cuda.synchronize(device)
        quant_wall_seconds = time.perf_counter() - quant_start
    finally:
        sampled_memory = sampler.stop()

    torch_memory = {
        "start_allocated_bytes": start_allocated,
        "start_reserved_bytes": start_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated(device) - start_allocated,
        "peak_reserved_delta_bytes": torch.cuda.max_memory_reserved(device) - start_reserved,
    }
    save_start = time.perf_counter()
    model.save(str(args.save_dir))
    save_wall_seconds = time.perf_counter() - save_start
    module_stats = adjacent_model.snapshot() if adjacent_model is not None else []
    payload = {
        "schema": "gptqmodel-adjacent-model-ab-quant-v1",
        "method": args.method,
        "seed": args.seed,
        "model": str(args.model),
        "save_dir": str(args.save_dir),
        "hardware": hardware_metadata(device, gpu_uuid),
        "quantization": {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "act_group_aware": True,
            "damp_percent": 0.05,
            "mse": 0.0,
            "backend": BACKEND.GPTQ_TORCH.value,
            "dtype": "auto (model config bfloat16)",
            "calibration_rows": args.calibration_rows,
            "calibration_concat_size": args.calibration_concat_size,
            "calibration_sort": "desc",
            "quant_batch_size": args.quant_batch_size,
            "quant_wall_seconds": quant_wall_seconds,
            "save_wall_seconds": save_wall_seconds,
            "torch_memory": torch_memory,
            "sampled_memory": sampled_memory,
        },
        "adjacent_model": (
            {
                "coordinate_starts": list(adjacent_model.coordinate_starts),
                "max_coordinate_flips": adjacent_model.max_coordinate_flips,
                "coordinate_rebase_interval": adjacent_model.coordinate_rebase_interval,
                "batch_coordinate_starts_on_cuda": adjacent_model.batch_coordinate_starts_on_cuda,
                "executor": adjacent_model.executor,
                "row_chunk_size": adjacent_model.row_chunk_size,
                "cpu_row_chunk_size": adjacent_model.cpu_row_chunk_size,
                "cpu_workers": adjacent_model.cpu_workers,
                "cpu_min_row_groups": adjacent_model.cpu_min_row_groups,
                "objective_row_chunk_size": adjacent_model.objective_row_chunk_size,
                "native_refinements_per_module": adjacent_model.native_refinements_per_module,
                "native_split_depth": adjacent_model.native_split_depth,
                "native_max_nodes_per_worker": adjacent_model.native_max_nodes_per_worker,
                "certificate_tolerance": adjacent_model.certificate_tolerance,
                "selection_tolerance": adjacent_model.selection_tolerance,
            }
            if adjacent_model is not None
            else None
        ),
        "adjacent_summary": adjacent_summary(module_stats),
        "adjacent_module_stats": module_stats,
        "quant_logs": quant_logs,
        "quant_region_snapshot": model.quant_region_timer.snapshot(),
    }
    write_json(args.result_json, payload)
    print(json.dumps(payload["quantization"], indent=2, sort_keys=True))
    if payload["adjacent_summary"]:
        print(json.dumps(payload["adjacent_summary"], indent=2, sort_keys=True))


def evaluate_case(args: argparse.Namespace) -> None:
    device, gpu_uuid = require_single_gpu()
    set_seed(args.seed)
    backend = BACKEND.AUTO if args.backend == "auto" else BACKEND.MARLIN
    raw_output = args.result_json.with_name(f"{args.result_json.stem}_raw.json")
    sampler = ResourceSampler(gpu_uuid=gpu_uuid)
    torch.cuda.reset_peak_memory_stats(device)
    sampler.start()
    eval_start = time.perf_counter()
    try:
        result = evaluate(
            model_or_id_or_path=str(args.model),
            tasks=list(TASKS),
            batch_size=args.eval_batch_size,
            trust_remote_code=False,
            output_path=str(raw_output),
            backend=backend,
            model_args={
                "dtype": "bfloat16",
                "device": "cuda:0",
                "padding_side": "left",
                "seed": args.seed,
            },
            apply_chat_template=args.apply_chat_template,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        )
        torch.cuda.synchronize(device)
        eval_wall_seconds = time.perf_counter() - eval_start
    finally:
        sampled_memory = sampler.stop()

    payload = {
        "schema": "gptqmodel-adjacent-model-ab-eval-v1",
        "label": args.label,
        "model": str(args.model),
        "backend": backend.value,
        "seed": args.seed,
        "tasks": list(TASKS),
        "batch_size": args.eval_batch_size,
        "apply_chat_template": args.apply_chat_template,
        "eval_wall_seconds": eval_wall_seconds,
        "metrics": get_eval_task_results(result),
        "eval_table": format_eval_result_table(result),
        "sampled_memory": sampled_memory,
        "hardware": hardware_metadata(device, gpu_uuid),
        "raw_result_json": str(raw_output),
    }
    write_json(args.result_json, payload)
    print(payload["eval_table"])


def _metric(payload: dict[str, Any], task: str, *names: str) -> float | None:
    metrics = payload.get("metrics", {}).get(task, {})
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return None


def summarize(args: argparse.Namespace) -> None:
    dense = json.loads(args.dense_eval.read_text())
    classic_quant = json.loads(args.classic_quant.read_text())
    adjacent_quant = json.loads(args.adjacent_quant.read_text())
    classic_eval = json.loads(args.classic_eval.read_text())
    adjacent_eval = json.loads(args.adjacent_eval.read_text())

    rows = []
    for label, evaluation, quantization in (
        ("dense_bf16", dense, None),
        ("classic_gptq", classic_eval, classic_quant),
        ("adjacent_hybrid", adjacent_eval, adjacent_quant),
    ):
        quant = quantization.get("quantization", {}) if quantization else {}
        rows.append(
            {
                "label": label,
                "arc_acc": _metric(evaluation, "arc_challenge", "accuracy,loglikelihood", "acc,ll"),
                "arc_acc_norm": _metric(
                    evaluation,
                    "arc_challenge",
                    "accuracy,loglikelihood_norm",
                    "acc,ll_avg",
                ),
                "gsm8k_platinum_acc": _metric(evaluation, "gsm8k_platinum_cot", "acc,num"),
                "quant_wall_seconds": quant.get("quant_wall_seconds"),
                "peak_process_gpu_used_mib": quant.get("sampled_memory", {}).get("peak_process_gpu_used_mib"),
                "peak_torch_allocated_bytes": quant.get("torch_memory", {}).get("peak_allocated_bytes"),
            }
        )

    adjacent_summary_payload = adjacent_quant.get("adjacent_summary", {})
    comparisons = {
        "arc_acc_adjacent_minus_classic": (
            rows[2]["arc_acc"] - rows[1]["arc_acc"]
            if rows[2]["arc_acc"] is not None and rows[1]["arc_acc"] is not None
            else None
        ),
        "arc_acc_norm_adjacent_minus_classic": (
            rows[2]["arc_acc_norm"] - rows[1]["arc_acc_norm"]
            if rows[2]["arc_acc_norm"] is not None and rows[1]["arc_acc_norm"] is not None
            else None
        ),
        "gsm8k_platinum_adjacent_minus_classic": (
            rows[2]["gsm8k_platinum_acc"] - rows[1]["gsm8k_platinum_acc"]
            if rows[2]["gsm8k_platinum_acc"] is not None and rows[1]["gsm8k_platinum_acc"] is not None
            else None
        ),
        "quant_wall_adjacent_over_classic": (
            rows[2]["quant_wall_seconds"] / rows[1]["quant_wall_seconds"]
            if rows[2]["quant_wall_seconds"] and rows[1]["quant_wall_seconds"]
            else None
        ),
        "classic_hessian_error_from_adjacent_replay": adjacent_summary_payload.get(
            "classic_full_hessian_error"
        ),
        "adjacent_hybrid_hessian_error": adjacent_summary_payload.get("hybrid_full_hessian_error"),
        "adjacent_hybrid_hessian_error_reduction_pct": adjacent_summary_payload.get(
            "hybrid_full_hessian_error_reduction_pct"
        ),
    }
    payload = {
        "schema": "gptqmodel-adjacent-model-ab-summary-v1",
        "seed": args.seed,
        "model": str(args.model),
        "tasks": list(TASKS),
        "rows": rows,
        "comparisons": comparisons,
        "dense_eval": dense,
        "classic_quant": classic_quant,
        "adjacent_quant": adjacent_quant,
        "classic_eval": classic_eval,
        "adjacent_eval": adjacent_eval,
    }
    write_json(args.output, payload)

    headers = ("method", "ARC acc", "ARC norm", "GSM8K plat", "quant s", "peak GPU MiB")
    print(" | ".join(headers))
    print("-|-".join("-" * len(header) for header in headers))
    for row in rows:
        values = (
            row["label"],
            "-" if row["arc_acc"] is None else f"{row['arc_acc']:.6f}",
            "-" if row["arc_acc_norm"] is None else f"{row['arc_acc_norm']:.6f}",
            "-" if row["gsm8k_platinum_acc"] is None else f"{row['gsm8k_platinum_acc']:.6f}",
            "-" if row["quant_wall_seconds"] is None else f"{row['quant_wall_seconds']:.3f}",
            "-" if row["peak_process_gpu_used_mib"] is None else str(row["peak_process_gpu_used_mib"]),
        )
        print(" | ".join(values))
    print(json.dumps(comparisons, indent=2, sort_keys=True))
    print(f"summary_json={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize_parser = subparsers.add_parser("quantize")
    quantize_parser.add_argument("--method", choices=("classic", "adjacent"), required=True)
    quantize_parser.add_argument("--model", type=Path, default=Path(MODEL_ID))
    quantize_parser.add_argument("--save-dir", type=Path, required=True)
    quantize_parser.add_argument("--result-json", type=Path, required=True)
    quantize_parser.add_argument("--seed", type=int, default=SEED)
    quantize_parser.add_argument("--calibration-rows", type=int, default=CALIBRATION_ROWS)
    quantize_parser.add_argument("--calibration-concat-size", type=int, default=CALIBRATION_CONCAT_SIZE)
    quantize_parser.add_argument("--quant-batch-size", type=int, default=QUANT_BATCH_SIZE)
    quantize_parser.add_argument("--max-coordinate-flips", type=int, default=32)
    quantize_parser.add_argument("--executor", choices=("auto", "cpu", "cuda"), default="auto")
    quantize_parser.add_argument("--row-chunk-size", type=int, default=2048)
    quantize_parser.add_argument("--cpu-row-chunk-size", type=int, default=512)
    quantize_parser.add_argument("--cpu-workers", type=int, default=64)
    quantize_parser.add_argument("--cpu-min-row-groups", type=int, default=393_216)
    quantize_parser.add_argument("--objective-row-chunk-size", type=int, default=256)
    quantize_parser.add_argument("--native-refinements-per-module", type=int, default=4)
    quantize_parser.add_argument("--native-split-depth", type=int, default=6)
    quantize_parser.add_argument("--native-max-nodes-per-worker", type=int, default=500)
    quantize_parser.set_defaults(func=quantize_case)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--label", required=True)
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--backend", choices=("auto", "marlin"), required=True)
    evaluate_parser.add_argument("--result-json", type=Path, required=True)
    evaluate_parser.add_argument("--seed", type=int, default=SEED)
    evaluate_parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    evaluate_parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluate_parser.set_defaults(func=evaluate_case)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--dense-eval", type=Path, required=True)
    summary_parser.add_argument("--classic-quant", type=Path, required=True)
    summary_parser.add_argument("--adjacent-quant", type=Path, required=True)
    summary_parser.add_argument("--classic-eval", type=Path, required=True)
    summary_parser.add_argument("--adjacent-eval", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--model", type=Path, default=Path(MODEL_ID))
    summary_parser.add_argument("--seed", type=int, default=SEED)
    summary_parser.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
