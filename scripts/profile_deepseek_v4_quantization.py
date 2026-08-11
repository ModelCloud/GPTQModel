#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Profile bounded DeepSeek V4 GPTQ quantization without changing quantization math.

The default workload quantizes decoder layers 0 and 1 with W4/G64, GAR,
``desc_act=False``, and activation scale search on physical GPUs 6 and 7.
It writes structured wall-time, per-layer, per-subset, GPU, and quality-proxy
telemetry to a separate artifact directory. Saving is intentionally omitted so
iterative performance runs do not measure checkpoint serialization.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = Path("/monster/data/model/DeepSeek-V4-Flash-0731-BF16-Defused")
DEFAULT_CALIBRATION = REPO_ROOT / "dataset/calibration_mix_128k_deepseek_v4_flash_0731/calibration.parquet"
DEFAULT_ARTIFACT_ROOT = Path("/root/gptqmodel-profile-artifacts/deepseek-v4-flash-0731")
DEFAULT_PHYSICAL_GPUS = "6,7"


@dataclass(frozen=True)
class GpuInfo:
    """Describe one physical PCI-bus-ordered GPU accepted by the idle gate."""

    physical_id: int
    pci_bus_id: str
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_pct: int
    compute_capability: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--physical-gpus", default=DEFAULT_PHYSICAL_GPUS)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--scale-search-candidate-chunk-size", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--moe-batch-size", type=int, default=None)
    parser.add_argument("--moe-capture-streams", type=int, default=2)
    parser.add_argument("--moe-parallel-output-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offload-path",
        type=Path,
        default=None,
        help="Optional finalized-module offload directory (for example, a sufficiently large tmpfs).",
    )
    parser.add_argument("--calibration-limit", type=int, default=None)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render messages with the tokenizer chat template instead of using their raw content.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=2.0)
    parser.add_argument("--idle-memory-mb", type=int, default=256)
    parser.add_argument("--idle-max-utilization", type=int, default=0)
    parser.add_argument("--live-interval", type=float, default=60.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=5.0)
    parser.add_argument("--diagnostics", choices=["off", "auto", "channel"], default="auto")
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.layers <= 0:
        parser.error("--layers must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.moe_capture_streams <= 0:
        parser.error("--moe-capture-streams must be positive")
    if args.idle_samples < 3:
        parser.error("--idle-samples must be at least 3")
    if args.calibration_limit is not None and args.calibration_limit <= 0:
        parser.error("--calibration-limit must be positive")
    if args.live_interval <= 0 or args.gpu_sample_interval <= 0:
        parser.error("--live-interval and --gpu-sample-interval must be positive")
    if not 0 <= args.idle_max_utilization <= 100:
        parser.error("--idle-max-utilization must be between 0 and 100")
    if args.torch_profile and args.nvtx:
        parser.error("Use either --torch-profile or --nvtx in one run, not both")
    return args


def _parse_physical_gpus(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("At least one physical GPU is required")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate physical GPU IDs are not allowed: {ids}")
    return ids


def _region_period_snapshots(region_timer: Any) -> list[dict[str, Any]]:
    """Read optional period telemetry while remaining compatible with older revisions."""

    snapshot = getattr(region_timer, "period_snapshots", None)
    return snapshot() if callable(snapshot) else []


def _run_checked(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _query_gpus() -> dict[int, GpuInfo]:
    fields = (
        "index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu,compute_cap"
    )
    output = _run_checked(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    rows: dict[int, GpuInfo] = {}
    for values in csv.reader(output.splitlines()):
        values = [value.strip() for value in values]
        if len(values) != 8:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {values}")
        info = GpuInfo(
            physical_id=int(values[0]),
            pci_bus_id=values[1],
            uuid=values[2],
            name=values[3],
            memory_total_mb=int(values[4]),
            memory_used_mb=int(values[5]),
            utilization_pct=int(values[6]),
            compute_capability=values[7],
        )
        rows[info.physical_id] = info
    return rows


def _query_compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to inspect GPU compute processes: {result.stderr.strip()}")
    processes = []
    for values in csv.reader(result.stdout.splitlines()):
        values = [value.strip() for value in values]
        if not values:
            continue
        if len(values) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi process row: {values}")
        processes.append(
            {
                "gpu_uuid": values[0],
                "pid": int(values[1]),
                "process_name": values[2],
                "used_memory_mb": int(values[3]),
            }
        )
    return processes


def _idle_gate(
    physical_ids: list[int],
    *,
    samples: int,
    interval: float,
    memory_limit_mb: int,
    utilization_limit_pct: int,
) -> list[GpuInfo]:
    accepted: list[GpuInfo] = []
    for sample_index in range(samples):
        inventory = _query_gpus()
        missing = [physical_id for physical_id in physical_ids if physical_id not in inventory]
        if missing:
            raise RuntimeError(f"Physical GPUs are missing from nvidia-smi: {missing}")
        selected = [inventory[physical_id] for physical_id in physical_ids]
        selected_uuids = {info.uuid for info in selected}
        foreign = [process for process in _query_compute_processes() if process["gpu_uuid"] in selected_uuids]
        if foreign:
            raise RuntimeError(f"Requested GPUs have active compute processes: {foreign}")
        for info in selected:
            print(
                f"[idle {sample_index + 1}/{samples}] physical={info.physical_id} pci={info.pci_bus_id} "
                f"uuid={info.uuid} util={info.utilization_pct}% memory={info.memory_used_mb}MiB",
                flush=True,
            )
            if info.utilization_pct > utilization_limit_pct or info.memory_used_mb > memory_limit_mb:
                raise RuntimeError(
                    f"Physical GPU {info.physical_id} failed the idle gate: util={info.utilization_pct}%, "
                    f"memory={info.memory_used_mb}MiB (limit {memory_limit_mb}MiB)"
                )
        accepted = selected
        if sample_index + 1 < samples:
            time.sleep(interval)
    return accepted


def _pre_timing_exclusivity_gate(
    gpu_infos: list[GpuInfo],
    *,
    samples: int,
    interval: float,
    utilization_limit_pct: int,
) -> None:
    requested_uuids = {info.uuid for info in gpu_infos}
    for sample_index in range(samples):
        foreign = [
            process
            for process in _query_compute_processes()
            if process["gpu_uuid"] in requested_uuids and process["pid"] != os.getpid()
        ]
        if foreign:
            raise RuntimeError(f"Foreign process appeared before timing: {foreign}")
        inventory = _query_gpus()
        for original in gpu_infos:
            current = inventory[original.physical_id]
            print(
                f"[pre-timing {sample_index + 1}/{samples}] physical={current.physical_id} "
                f"util={current.utilization_pct}% memory={current.memory_used_mb}MiB foreign_processes=0",
                flush=True,
            )
            if current.utilization_pct > utilization_limit_pct:
                raise RuntimeError(
                    f"Physical GPU {current.physical_id} failed the pre-timing utilization gate: "
                    f"util={current.utilization_pct}% (limit {utilization_limit_pct}%)"
                )
            if current.utilization_pct != 0:
                raise RuntimeError(
                    f"Physical GPU {current.physical_id} was active immediately before timing: "
                    f"util={current.utilization_pct}%"
                )
        if sample_index + 1 < samples:
            time.sleep(interval)


def _normalize_uuid(value: Any) -> str:
    return str(value).lower().removeprefix("gpu-").replace("-", "")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_calibration_rows(records: list[dict[str, Any]], *, apply_chat_template: bool) -> list[dict[str, Any]]:
    """Preserve chat rows or recover raw text according to the dataset scan contract."""

    prepared = []
    for index, record in enumerate(records):
        messages = list(record.get("messages") or [])
        if apply_chat_template:
            prepared.append({"messages": messages})
            continue
        if not messages or any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError(f"Calibration row {index} does not contain string message content")
        prepared.append({"text": "\n".join(message["content"] for message in messages)})
    return prepared


class SubsetProfiler:
    """Collect subset wall times and optional NVTX ranges through public callbacks."""

    def __init__(self, *, enable_nvtx: bool, torch_module=None):
        self.enable_nvtx = enable_nvtx
        self.torch = torch_module
        self.started_at = time.perf_counter()
        self._lock = threading.Lock()
        self._starts: dict[tuple[int, int, str], tuple[float, int]] = {}
        self._events: list[dict[str, Any]] = []
        self._layer_events: list[dict[str, Any]] = []

    def subset_event(
        self,
        *,
        stage: str,
        layer_idx: int,
        subset_index: int,
        subset_total: int,
        module_names: list[str],
        processor: Any,
    ) -> None:
        if callable(processor):
            processor = processor()
        processor = str(processor)
        if stage.endswith("_start"):
            phase = stage.removesuffix("_start")
            key = (layer_idx, subset_index, phase)
            with self._lock:
                self._starts[key] = (time.perf_counter(), threading.get_ident())
            if self.enable_nvtx:
                label = f"gptqmodel/layer_{layer_idx}/{phase}_subset_{subset_index + 1}_of_{subset_total}"
                self.torch.cuda.nvtx.range_push(label)
            return

        if stage not in {"forward_end", "quant_complete"}:
            return
        phase = "forward" if stage == "forward_end" else "quant"
        key = (layer_idx, subset_index, phase)
        ended_at = time.perf_counter()
        with self._lock:
            started = self._starts.pop(key, None)
            if started is None:
                return
            start_time, start_thread = started
            self._events.append(
                {
                    "layer": layer_idx,
                    "subset_index": subset_index,
                    "subset_total": subset_total,
                    "phase": phase,
                    "processor": processor,
                    "module_count": len(module_names),
                    "module_sample": list(module_names[:8]),
                    "wall_s": ended_at - start_time,
                    "elapsed_s": ended_at - self.started_at,
                    "thread_id": start_thread,
                }
            )
        if self.enable_nvtx:
            self.torch.cuda.nvtx.range_pop()

    def layer_complete(self, *, layer_idx: int, submodule_finalized: bool) -> None:
        with self._lock:
            self._layer_events.append(
                {
                    "layer": layer_idx,
                    "submodule_finalized": bool(submodule_finalized),
                    "elapsed_s": time.perf_counter() - self.started_at,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "subsets": [dict(event) for event in self._events],
                "layer_events": [dict(event) for event in self._layer_events],
                "open_event_count": len(self._starts),
            }

    def completed_subset_count(self) -> int:
        with self._lock:
            return sum(event["phase"] == "quant" for event in self._events)


class LiveReporter:
    """Sample GPU state frequently and emit periodic benchmark state tables."""

    def __init__(
        self,
        *,
        physical_ids: list[int],
        subset_profiler: SubsetProfiler,
        report_interval: float,
        sample_interval: float,
    ):
        self.physical_ids = physical_ids
        self.subset_profiler = subset_profiler
        self.report_interval = report_interval
        self.sample_interval = sample_interval
        self.started_at = time.perf_counter()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._run, name="QuantProfileLiveReporter", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self.sample_interval, 1.0) + 5.0)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(sample) for sample in self._samples]

    def _run(self) -> None:
        next_report = self.report_interval
        while not self._stop.wait(self.sample_interval):
            inventory = _query_gpus()
            elapsed = time.perf_counter() - self.started_at
            sample = {
                "elapsed_s": elapsed,
                "quant_subsets": self.subset_profiler.completed_subset_count(),
                "gpus": {
                    str(physical_id): {
                        "utilization_pct": inventory[physical_id].utilization_pct,
                        "memory_used_mb": inventory[physical_id].memory_used_mb,
                    }
                    for physical_id in self.physical_ids
                },
            }
            with self._lock:
                self._samples.append(sample)
            if elapsed < next_report:
                continue
            utilization = ",".join(
                f"{physical_id}:{inventory[physical_id].utilization_pct}%" for physical_id in self.physical_ids
            )
            metrics = (
                f"elapsed={elapsed:.1f}s; quant_subsets={self.subset_profiler.completed_subset_count()}; "
                f"util={utilization}"
            )
            _print_result_table(metrics=metrics, state="running", physical_ids=self.physical_ids)
            while next_report <= elapsed:
                next_report += self.report_interval


def _print_result_table(*, metrics: str, state: str, physical_ids: list[int]) -> None:
    headers = ["GPU ID(s)", "Candidate", "Model/decoder bits", "Endpoint bits", "Size MB", "Metrics", "State"]
    values = [
        ",".join(str(value) for value in physical_ids),
        "current-main",
        "DeepSeek-V4 W4/G64",
        "BF16 (out of scope)",
        "dense=367616",
        metrics,
        state,
    ]
    widths = [max(len(header), len(value)) for header, value in zip(headers, values)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    print("\n" + "\n".join([border, row(headers), border, row(values), border]), flush=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return _json_safe(value.value)
    return str(value)


def _summarize_quant_log(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality_rows = []
    by_layer: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        layer = str(row.get("layer"))
        loss = float(row.get("loss", float("nan")))
        duration = float(row.get("time", 0.0))
        by_layer.setdefault(layer, []).append({"loss": loss, "time": duration})
        quality_rows.append(
            {
                key: _json_safe(row.get(key))
                for key in ("layer", "module", "loss", "samples", "damp")
            }
        )
    quality_rows.sort(key=lambda row: (str(row.get("layer")), str(row.get("module"))))
    canonical = json.dumps(quality_rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    layer_summary = {}
    for layer, layer_rows in sorted(by_layer.items(), key=lambda item: item[0]):
        losses = [row["loss"] for row in layer_rows]
        durations = [row["time"] for row in layer_rows]
        finite_losses = [loss for loss in losses if math.isfinite(loss)]
        layer_summary[layer] = {
            "modules": len(layer_rows),
            "finite_loss_count": len(finite_losses),
            "nonfinite_loss_count": len(losses) - len(finite_losses),
            "loss_mean": statistics.fmean(finite_losses) if finite_losses else None,
            "loss_max": max(finite_losses) if finite_losses else None,
            "module_time_sum_s": sum(durations),
            "module_time_mean_s": statistics.fmean(durations) if durations else None,
            "module_time_median_s": statistics.median(durations) if durations else None,
            "module_time_max_s": max(durations) if durations else None,
        }
    return {
        "quality_fingerprint_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "layers": layer_summary,
    }


@contextlib.contextmanager
def _torch_profile(torch_module, enabled: bool, artifact_dir: Path):
    if not enabled:
        yield None
        return
    activities = [torch_module.profiler.ProfilerActivity.CPU, torch_module.profiler.ProfilerActivity.CUDA]
    with torch_module.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        yield profiler
    profiler.export_chrome_trace(str(artifact_dir / "torch_profile.json"))
    (artifact_dir / "torch_profile_top_ops.txt").write_text(
        profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=200),
        encoding="utf-8",
    )


def _git_revision() -> str:
    return _run_checked(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])


def _artifact_dir(requested: Path | None) -> Path:
    if requested is not None:
        path = requested
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = DEFAULT_ARTIFACT_ROOT / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def main() -> None:
    args = _parse_args()
    physical_ids = _parse_physical_gpus(args.physical_gpus)
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model}")
    if not args.calibration.is_file():
        raise FileNotFoundError(f"Calibration parquet not found: {args.calibration}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_infos = _idle_gate(
        physical_ids,
        samples=args.idle_samples,
        interval=args.idle_interval,
        memory_limit_mb=args.idle_memory_mb,
        utilization_limit_pct=args.idle_max_utilization,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(info.uuid for info in gpu_infos)

    # CUDA-aware imports happen only after the physical-device gate and visibility restriction.
    import pandas as pd
    import torch
    import transformers

    if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        raise RuntimeError("This profile requires a free-threaded Python with the GIL disabled")
    if torch.cuda.device_count() != len(gpu_infos):
        raise RuntimeError(
            f"Expected {len(gpu_infos)} visible GPUs, but Torch reports {torch.cuda.device_count()}"
        )

    visible_hardware = []
    for local_id, expected in enumerate(gpu_infos):
        props = torch.cuda.get_device_properties(local_id)
        if _normalize_uuid(props.uuid) != _normalize_uuid(expected.uuid):
            raise RuntimeError(
                f"cuda:{local_id} UUID {props.uuid} does not match physical GPU "
                f"{expected.physical_id} UUID {expected.uuid}"
            )
        visible_hardware.append(
            {
                "local_cuda_id": local_id,
                "physical_id": expected.physical_id,
                "pci_bus_id": expected.pci_bus_id,
                "uuid": expected.uuid,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "sm_count": props.multi_processor_count,
                "memory_total_bytes": props.total_memory,
            }
        )
        print(
            f"[mapping] cuda:{local_id} -> physical={expected.physical_id} pci={expected.pci_bus_id} "
            f"uuid={expected.uuid} cc={props.major}.{props.minor} sms={props.multi_processor_count}",
            flush=True,
        )

    artifact_dir = _artifact_dir(args.artifacts_dir)
    if args.offload_path is not None:
        args.offload_path.mkdir(parents=True, exist_ok=True)
    calibration_frame = pd.read_parquet(args.calibration)
    if "messages" not in calibration_frame:
        raise ValueError(f"Calibration parquet lacks a messages column: {args.calibration}")
    if args.calibration_limit is not None:
        calibration_frame = calibration_frame.iloc[: args.calibration_limit]
    calibration_records = calibration_frame[["messages"]].to_dict(orient="records")
    calibration_rows = _prepare_calibration_rows(
        calibration_records,
        apply_chat_template=args.apply_chat_template,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT, METHOD, ScaleSearchConfig
    from gptqmodel.quantization.config import (
        ExpertsRoutingBypass,
        MoEConfig,
        MoEExecutionConfig,
        QuantizeConfig,
        VramStrategy,
    )

    quantize_config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        act_group_aware=True,
        scale_search=ScaleSearchConfig.ACTIVATION,
        scale_search_candidate_chunk_size=args.scale_search_candidate_chunk_size,
        moe=MoEConfig(
            routing=ExpertsRoutingBypass(),
            execution=MoEExecutionConfig(
                batch_size=args.moe_batch_size,
                parallel_input_capture=True,
                parallel_input_capture_streams=args.moe_capture_streams,
                parallel_output_replay=args.moe_parallel_output_replay,
            ),
        ),
        auto_forward_data_parallel=True,
        calibration_data_device="balanced",
        dense_vram_strategy=VramStrategy.EXCLUSIVE,
        moe_vram_strategy=VramStrategy.BALANCED,
        offload_to_disk_path=str(args.offload_path) if args.offload_path is not None else None,
        quantization_diagnostics=args.diagnostics,
    )
    print(f"[config] {quantize_config}", flush=True)
    print(f"[artifacts] {artifact_dir}", flush=True)

    load_started = time.perf_counter()
    model = GPTQModel.load(
        str(args.model),
        quantize_config=quantize_config,
        trust_remote_code=args.trust_remote_code,
        dtype="auto",
        device_map="auto",
    )
    model_load_s = time.perf_counter() - load_started
    subset_profiler = SubsetProfiler(enable_nvtx=args.nvtx, torch_module=torch)
    model.subset_callback = subset_profiler
    model.layer_callback = subset_profiler

    _pre_timing_exclusivity_gate(
        gpu_infos,
        samples=args.idle_samples,
        interval=args.idle_interval,
        utilization_limit_pct=args.idle_max_utilization,
    )
    reporter = LiveReporter(
        physical_ids=physical_ids,
        subset_profiler=subset_profiler,
        report_interval=args.live_interval,
        sample_interval=args.gpu_sample_interval,
    )
    reporter.start()
    quantize_started = time.perf_counter()
    try:
        with _torch_profile(torch, args.torch_profile, artifact_dir):
            model.quantize(
                calibration_rows,
                batch_size=args.batch_size,
                backend="auto",
                tokenizer=tokenizer,
                layer_scope=slice(0, args.layers),
            )
        torch.cuda.synchronize()
    finally:
        quantize_wall_s = time.perf_counter() - quantize_started
        reporter.stop()

    quant_log = [_json_safe(row) for row in model.quant_log]
    quant_summary = _summarize_quant_log(quant_log)
    region_timer = model.quant_region_timer
    profile = {
        "schema": "gptqmodel.deepseek_v4_quant_profile.v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "repository": {"root": str(REPO_ROOT), "revision": _git_revision()},
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "gil_enabled": sys._is_gil_enabled(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "cuda_driver": _run_checked(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ).splitlines()[0],
        },
        "hardware": visible_hardware,
        "model": {
            "path": str(args.model.resolve()),
            "source_size_bytes": sum(path.stat().st_size for path in args.model.glob("*") if path.is_file()),
            "dtype": "bf16",
            "layer_scope": {"start": 0, "stop": args.layers, "step": 1},
        },
        "calibration": {
            "path": str(args.calibration.resolve()),
            "sha256": _sha256_file(args.calibration),
            "rows": len(calibration_rows),
            "batch_size": args.batch_size,
            "apply_chat_template": args.apply_chat_template,
        },
        "quantization": _json_safe(quantize_config.to_dict()),
        "timings": {
            "model_load_wall_s": model_load_s,
            "quantize_wall_s": quantize_wall_s,
            "regions": region_timer.snapshot(),
            "periods": _region_period_snapshots(region_timer),
            "callback": subset_profiler.snapshot(),
        },
        "quality_proxy": quant_summary,
        "profiler": {
            "torch_profile": args.torch_profile,
            "nvtx": args.nvtx,
            "gpu_samples": reporter.snapshot(),
            "warmup": "not applicable: one-shot sequential layer quantization",
            "synchronization": "torch.cuda.synchronize after quantize",
        },
    }
    (artifact_dir / "quant_log.json").write_text(
        json.dumps(quant_log, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (artifact_dir / "profile.json").write_text(
        json.dumps(_json_safe(profile), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _print_result_table(
        metrics=(
            f"quant_wall={quantize_wall_s:.1f}s; layers={args.layers}; "
            f"modules={sum(item['modules'] for item in quant_summary['layers'].values())}"
        ),
        state="complete",
        physical_ids=physical_ids,
    )
    print(f"[complete] profile={artifact_dir / 'profile.json'}", flush=True)


if __name__ == "__main__":
    main()
