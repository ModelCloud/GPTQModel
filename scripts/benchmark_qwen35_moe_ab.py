#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_PATH = "/monster/data/model/Qwen3.5-35B-A3B"
DEFAULT_BASELINE_ROOT = "/root/gptqmodel-main"
FAST_LAYER_COUNT_ENV = "GPTQMODEL_FAST_LAYER_COUNT"
FAST_LAYER_POSITION_ENV = "GPTQMODEL_FAST_LAYER_POSITION"
VRAM_STRATEGY_CHOICES = (
    "exclusive",
    "balanced",
    "dense_home_moe_balanced",
)
DENSE_VRAM_STRATEGY_CHOICES = ("exclusive", "balanced")
MOE_VRAM_STRATEGY_CHOICES = ("exclusive", "balanced")


def _csv_arg(value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated CLI device list into a normalized list."""

    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark two MoE layers of Qwen 3.5 on the current repo and a baseline repo, "
            "capturing per-GPU VRAM plus GPTQ timing regions."
        )
    )
    parser.add_argument("--single", action="store_true", help="Run a single repo benchmark inside the current process.")
    parser.add_argument("--repo-root", type=Path, help="Repo root to import in --single mode.")
    parser.add_argument("--json-out", type=Path, help="JSON output path in --single mode.")
    parser.add_argument("--label", default="", help="Friendly label for the current case.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen 3.5 MoE model path.")
    parser.add_argument("--baseline-root", type=Path, default=Path(DEFAULT_BASELINE_ROOT), help="Baseline repo root for A/B mode.")
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1], help="Current repo root for A/B mode.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for logs and JSON outputs.")
    parser.add_argument("--dataset-size", type=int, default=16, help="Calibration rows to use.")
    parser.add_argument("--batch-size", type=int, default=1, help="Quant batch size.")
    parser.add_argument("--quant-layers", type=int, default=2, help="Number of prefix layers to keep in fast-mode quantization.")
    parser.add_argument("--stop-after-layer", type=int, default=1, help="Stop once this zero-based layer index has fully finalized.")
    parser.add_argument("--dtype", default="auto", help="Model dtype passed to GPTQModel.load().")
    parser.add_argument("--attn-implementation", default="eager", choices=("eager", "flash_attention_2"), help="Attention implementation.")
    parser.add_argument("--vram-strategy", default="balanced", choices=VRAM_STRATEGY_CHOICES, help="VRAM strategy to benchmark.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Comma-separated visible device set for one case.")
    parser.add_argument("--dense-vram-strategy", default="balanced", choices=DENSE_VRAM_STRATEGY_CHOICES, help="Dense-pool strategy for repos that support split VRAM config.")
    parser.add_argument("--dense-vram-strategy-devices", default=None, help="Comma-separated dense-pool devices relative to CUDA_VISIBLE_DEVICES, e.g. cuda:0.")
    parser.add_argument("--moe-vram-strategy", default="balanced", choices=MOE_VRAM_STRATEGY_CHOICES, help="MoE-pool strategy for repos that support split VRAM config.")
    parser.add_argument("--moe-vram-strategy-devices", default=None, help="Comma-separated MoE-pool devices relative to CUDA_VISIBLE_DEVICES, e.g. cuda:1,cuda:2.")
    parser.add_argument(
        "--current-vram-strategy",
        default=None,
        choices=VRAM_STRATEGY_CHOICES,
        help="Optional VRAM strategy override for the current repo in A/B mode.",
    )
    parser.add_argument(
        "--baseline-vram-strategy",
        default=None,
        choices=("exclusive", "balanced"),
        help="Optional VRAM strategy override for the baseline repo in A/B mode.",
    )
    parser.add_argument("--current-cuda-visible-devices", default=None, help="Comma-separated CUDA_VISIBLE_DEVICES for the current repo in A/B mode.")
    parser.add_argument("--baseline-cuda-visible-devices", default=None, help="Comma-separated CUDA_VISIBLE_DEVICES for the baseline repo in A/B mode.")
    parser.add_argument("--current-dense-vram-strategy", default=None, choices=DENSE_VRAM_STRATEGY_CHOICES, help="Optional dense-pool strategy override for the current repo in A/B mode.")
    parser.add_argument("--baseline-dense-vram-strategy", default=None, choices=DENSE_VRAM_STRATEGY_CHOICES, help="Optional dense-pool strategy override for the baseline repo in A/B mode.")
    parser.add_argument("--current-dense-vram-strategy-devices", default=None, help="Optional dense-pool device list override for the current repo in A/B mode.")
    parser.add_argument("--baseline-dense-vram-strategy-devices", default=None, help="Optional dense-pool device list override for the baseline repo in A/B mode.")
    parser.add_argument("--current-moe-vram-strategy", default=None, choices=MOE_VRAM_STRATEGY_CHOICES, help="Optional MoE-pool strategy override for the current repo in A/B mode.")
    parser.add_argument("--baseline-moe-vram-strategy", default=None, choices=MOE_VRAM_STRATEGY_CHOICES, help="Optional MoE-pool strategy override for the baseline repo in A/B mode.")
    parser.add_argument("--current-moe-vram-strategy-devices", default=None, help="Optional MoE-pool device list override for the current repo in A/B mode.")
    parser.add_argument("--baseline-moe-vram-strategy-devices", default=None, help="Optional MoE-pool device list override for the baseline repo in A/B mode.")
    return parser.parse_args()


def _git_head(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip()


def _extract_region_total(snapshot: Dict[str, Dict[str, Any]], region: str) -> float:
    stat = snapshot.get(region) or {}
    try:
        return float(stat.get("total", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _spread(values: List[float]) -> float:
    """Measure device imbalance across all visible accelerators for one sample."""

    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _summarize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    layer_records = case.get("layer_records") or []
    final_layer = layer_records[-1] if layer_records else None
    final_devices = (final_layer or {}).get("devices") or []
    reserved_gib = [float(item.get("reserved_gib", 0.0)) for item in final_devices]
    peak_reserved_gib = [float(item.get("max_reserved_gib", 0.0)) for item in final_devices]

    spread_reserved_gib = _spread(reserved_gib)
    peak_spread_reserved_gib = _spread(peak_reserved_gib)

    quant_region_snapshot = case.get("quant_region_snapshot") or {}
    return {
        "label": case.get("label"),
        "repo_root": case.get("repo_root"),
        "git_head": case.get("git_head"),
        "vram_strategy": case.get("vram_strategy"),
        "dense_vram_strategy": case.get("dense_vram_strategy"),
        "dense_vram_strategy_devices": case.get("dense_vram_strategy_devices"),
        "moe_vram_strategy": case.get("moe_vram_strategy"),
        "moe_vram_strategy_devices": case.get("moe_vram_strategy_devices"),
        "split_vram_pools_applied": bool(case.get("split_vram_pools_applied")),
        "cuda_visible_devices": case.get("cuda_visible_devices"),
        "quant_wall_s": float(case.get("quant_wall_s", 0.0)),
        "pre_quant_forward_s": _extract_region_total(quant_region_snapshot, "pre_quant_forward"),
        "process_quant_s": _extract_region_total(quant_region_snapshot, "process_quant"),
        "post_quant_forward_s": _extract_region_total(quant_region_snapshot, "post_quant_forward"),
        "layer_count_observed": len(layer_records),
        "final_layer_idx": final_layer.get("layer_idx") if final_layer else None,
        "final_reserved_gib": reserved_gib,
        "final_peak_reserved_gib": peak_reserved_gib,
        "final_reserved_spread_gib": spread_reserved_gib,
        "final_peak_reserved_spread_gib": peak_spread_reserved_gib,
    }


def _compare_cases(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    current_summary = _summarize_case(current)
    baseline_summary = _summarize_case(baseline)
    fields = [
        "quant_wall_s",
        "pre_quant_forward_s",
        "process_quant_s",
        "post_quant_forward_s",
        "final_reserved_spread_gib",
        "final_peak_reserved_spread_gib",
    ]
    deltas = {}
    for field in fields:
        cur = float(current_summary.get(field, 0.0))
        base = float(baseline_summary.get(field, 0.0))
        pct = None if base == 0.0 else ((cur - base) / base) * 100.0
        deltas[field] = {
            "current": cur,
            "baseline": base,
            "delta": cur - base,
            "delta_pct": pct,
        }

    return {
        "current": current_summary,
        "baseline": baseline_summary,
        "delta": deltas,
    }


def _print_case_summary(case: Dict[str, Any]) -> None:
    summary = _summarize_case(case)
    print(
        f"[{summary['label']}] head={summary['git_head']} "
        f"vram_strategy={summary.get('vram_strategy')} "
        f"dense={summary.get('dense_vram_strategy')}@{summary.get('dense_vram_strategy_devices')} "
        f"moe={summary.get('moe_vram_strategy')}@{summary.get('moe_vram_strategy_devices')} "
        f"split_applied={summary.get('split_vram_pools_applied')} "
        f"visible={summary.get('cuda_visible_devices')}"
    )
    print(
        "  quant_wall_s={quant_wall_s:.3f} pre={pre_quant_forward_s:.3f} "
        "quant={process_quant_s:.3f} post={post_quant_forward_s:.3f}".format(**summary)
    )
    if summary["final_reserved_gib"]:
        rounded_reserved = [round(value, 3) for value in summary["final_reserved_gib"]]
        rounded_peak = [round(value, 3) for value in summary["final_peak_reserved_gib"]]
        print(
            "  final_layer_idx={} reserved_gib={} peak_reserved_gib={} spread_gib={:.3f}".format(
                summary["final_layer_idx"],
                rounded_reserved,
                rounded_peak,
                summary["final_reserved_spread_gib"],
            )
        )


def _print_compare(compare: Dict[str, Any]) -> None:
    print("\n[A/B delta: current - baseline]")
    for field, entry in compare["delta"].items():
        pct_text = "n/a" if entry["delta_pct"] is None else f"{entry['delta_pct']:+.2f}%"
        print(
            f"  {field}: current={entry['current']:.3f} baseline={entry['baseline']:.3f} "
            f"delta={entry['delta']:+.3f} ({pct_text})"
        )


def _run_subprocess_case(
    *,
    script_path: Path,
    repo_root: Path,
    label: str,
    output_dir: Path,
    model_path: str,
    dataset_size: int,
    batch_size: int,
    quant_layers: int,
    stop_after_layer: int,
    dtype: str,
    attn_implementation: str,
    vram_strategy: str,
    cuda_visible_devices: Optional[str],
    dense_vram_strategy: str,
    dense_vram_strategy_devices: Optional[str],
    moe_vram_strategy: str,
    moe_vram_strategy_devices: Optional[str],
) -> Dict[str, Any]:
    json_out = output_dir / f"{label}.json"
    log_out = output_dir / f"{label}.log"
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    else:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["PYTHON_GIL"] = "0"
    env["DEBUG"] = "1"

    cmd = [
        sys.executable,
        str(script_path),
        "--single",
        "--repo-root",
        str(repo_root),
        "--json-out",
        str(json_out),
        "--label",
        label,
        "--model-path",
        model_path,
        "--dataset-size",
        str(dataset_size),
        "--batch-size",
        str(batch_size),
        "--quant-layers",
        str(quant_layers),
        "--stop-after-layer",
        str(stop_after_layer),
        "--dtype",
        dtype,
        "--attn-implementation",
        attn_implementation,
        "--vram-strategy",
        vram_strategy,
        "--dense-vram-strategy",
        dense_vram_strategy,
        "--moe-vram-strategy",
        moe_vram_strategy,
    ]
    if cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", cuda_visible_devices])
    if dense_vram_strategy_devices is not None:
        cmd.extend(["--dense-vram-strategy-devices", dense_vram_strategy_devices])
    if moe_vram_strategy_devices is not None:
        cmd.extend(["--moe-vram-strategy-devices", moe_vram_strategy_devices])

    with log_out.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            cmd,
            check=True,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    with json_out.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _single_case_main(args: argparse.Namespace) -> int:
    if args.repo_root is None or args.json_out is None:
        raise SystemExit("--single requires --repo-root and --json-out")
    if os.environ.get("PYTHON_GIL") != "0":
        raise SystemExit("--single requires PYTHON_GIL=0")
    if args.cuda_visible_devices is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != args.cuda_visible_devices:
        raise SystemExit("--single requires CUDA_VISIBLE_DEVICES to match --cuda-visible-devices")

    repo_root = args.repo_root.resolve()
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "tests" / "models"))

    os.environ[FAST_LAYER_COUNT_ENV] = str(args.quant_layers)
    os.environ[FAST_LAYER_POSITION_ENV] = "first"
    os.environ["DEBUG"] = "1"

    import torch
    from transformers.utils import is_flash_attn_2_available

    from gptqmodel import DEBUG_ON, GPTQModel
    from gptqmodel.looper.module_looper import StopMainLoop
    from gptqmodel.quantization.config import VramStrategy
    from gptqmodel.utils.torch import torch_empty_cache
    from model_test import BACKEND
    from test_qwen3_5_moe import TestQwen3_5Moe

    resolved_vram_strategy = VramStrategy(args.vram_strategy)
    resolved_dense_vram_strategy = VramStrategy(args.dense_vram_strategy)
    dense_vram_strategy_devices = _csv_arg(args.dense_vram_strategy_devices)
    moe_vram_strategy_devices = _csv_arg(args.moe_vram_strategy_devices)
    resolved_moe_vram_strategy = VramStrategy(args.moe_vram_strategy)

    def _safe_sync() -> None:
        if not torch.cuda.is_available():
            return
        for idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(idx)
            except Exception as exc:
                # Snapshot collection is best-effort; keep the benchmark running.
                print(f"Warning: failed to synchronize cuda:{idx} during benchmark snapshot: {exc}", file=sys.stderr)

    def _snapshot_cuda(label: str) -> Dict[str, Any]:
        gc.collect()
        _safe_sync()
        snapshot: Dict[str, Any] = {
            "label": label,
            "monotonic_s": time.perf_counter(),
            "devices": [],
        }
        if not torch.cuda.is_available():
            return snapshot

        for idx in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            snapshot["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "allocated_gib": torch.cuda.memory_allocated(idx) / (1024 ** 3),
                    "reserved_gib": torch.cuda.memory_reserved(idx) / (1024 ** 3),
                    "max_allocated_gib": torch.cuda.max_memory_allocated(idx) / (1024 ** 3),
                    "max_reserved_gib": torch.cuda.max_memory_reserved(idx) / (1024 ** 3),
                    "active_current_gib": stats.get("active_bytes.all.current", 0) / (1024 ** 3),
                    "active_peak_gib": stats.get("active_bytes.all.peak", 0) / (1024 ** 3),
                    "free_gib": free_bytes / (1024 ** 3),
                    "total_gib": total_bytes / (1024 ** 3),
                }
            )
        return snapshot

    class BenchmarkQwen35Moe(TestQwen3_5Moe):
        DATASET_SIZE = args.dataset_size
        QUANT_BATCH_SIZE = args.batch_size
        STOP_AFTER_LAYER = args.stop_after_layer
        EVAL_TASKS = {}
        EVAL_TASKS_FAST = {}
        EVAL_TASKS_SLOW = {}
        USE_FLASH_ATTN = args.attn_implementation == "flash_attention_2"
        VRAM_STRATEGY = resolved_vram_strategy

        def __init__(self, methodName: str = "test_qwen3_5_moe"):
            super().__init__(methodName=methodName)
            self.memory_snapshots: List[Dict[str, Any]] = []
            self.layer_records: List[Dict[str, Any]] = []

        def _record_snapshot(self, label: str) -> Dict[str, Any]:
            snapshot = _snapshot_cuda(label)
            self.memory_snapshots.append(snapshot)
            return snapshot

        def _build_layer_stop_callback(self, layer_idx: int):
            outer = self

            class _Probe:
                def __init__(self, target: int):
                    self._target = target
                    self._triggered = False

                def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
                    if submodule_finalized:
                        outer.layer_records.append(
                            {
                                "layer_idx": layer_idx,
                                "submodule_finalized": True,
                                "devices": _snapshot_cuda(f"layer_{layer_idx}_finalized")["devices"],
                            }
                        )
                    if self._triggered:
                        return None
                    if layer_idx > self._target or (submodule_finalized and layer_idx >= self._target):
                        self._triggered = True
                        raise StopMainLoop
                    return None

            return _Probe(layer_idx)

        def run_benchmark(self) -> Dict[str, Any]:
            torch_empty_cache()
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(idx)

            quantize_config = self._build_quantize_config()
            quantize_config.wait_for_submodule_finalizers = True

            load_kwargs: Dict[str, Any] = {}
            if self.USE_FLASH_ATTN and is_flash_attn_2_available():
                load_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                load_kwargs["attn_implementation"] = "eager"

            torch_fused_backend = self._torch_fused_backend()
            device_map = {"": "cpu"} if self.LOAD_BACKEND == torch_fused_backend else "auto"

            model = None
            dataset = None
            stop_exception = False
            try:
                self._record_snapshot("before_load")
                model = GPTQModel.load(
                    self.NATIVE_MODEL_ID,
                    quantize_config=quantize_config,
                    trust_remote_code=self.TRUST_REMOTE_CODE,
                    dtype=args.dtype,
                    device_map=device_map,
                    **load_kwargs,
                )
                self._record_snapshot("after_load")

                self._layer_stop_callback = None
                if DEBUG_ON and self.STOP_AFTER_LAYER is not None:
                    self._layer_stop_callback = self._build_layer_stop_callback(self.STOP_AFTER_LAYER)
                    model.layer_callback = self._layer_stop_callback

                self._apply_model_compat_quant_overrides(model)
                dataset = self.load_dataset(model.tokenizer, rows=self.DATASET_SIZE)
                self._record_snapshot("after_dataset")

                start = time.perf_counter()
                try:
                    model.quantize(
                        dataset,
                        calibration_concat_size=self.DATASET_CONCAT_SIZE,
                        calibration_concat_separator=self.DATASET_CONCAT_SEPARATOR,
                        calibration_sort=self.DATASET_SORT,
                        backend=self.QUANT_BACKEND,
                        batch_size=self.QUANT_BATCH_SIZE,
                    )
                except StopMainLoop:
                    stop_exception = True
                quant_wall_s = time.perf_counter() - start
                self._record_snapshot("after_quant")

                quant_region_snapshot = model.quant_region_timer.snapshot()
                hf_device_map = getattr(model.model, "hf_device_map", None) or getattr(model, "hf_device_map", None)
                result = {
                    "label": args.label,
                    "repo_root": str(repo_root),
                    "git_head": _git_head(repo_root),
                    "model_path": self.NATIVE_MODEL_ID,
                    "dataset_size": self.DATASET_SIZE,
                    "batch_size": self.QUANT_BATCH_SIZE,
                    "quant_layers": args.quant_layers,
                    "stop_after_layer": self.STOP_AFTER_LAYER,
                    "dtype": str(args.dtype),
                    "attn_implementation": load_kwargs["attn_implementation"],
                    "vram_strategy": self.VRAM_STRATEGY.value,
                    "dense_vram_strategy": getattr(self, "DENSE_VRAM_STRATEGY", None).value if getattr(self, "DENSE_VRAM_STRATEGY", None) is not None else None,
                    "dense_vram_strategy_devices": getattr(self, "DENSE_VRAM_STRATEGY_DEVICES", None),
                    "moe_vram_strategy": getattr(self, "MOE_VRAM_STRATEGY", None).value if getattr(self, "MOE_VRAM_STRATEGY", None) is not None else None,
                    "moe_vram_strategy_devices": getattr(self, "MOE_VRAM_STRATEGY_DEVICES", None),
                    "split_vram_pools_supported": split_vram_pools_supported,
                    "split_vram_pools_applied": split_vram_pools_applied,
                    "python": sys.version,
                    "python_gil_disabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)() is False),
                    "python_gil_env": os.environ.get("PYTHON_GIL"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                    "debug_on": bool(DEBUG_ON),
                    "debug_short_circuit": bool(self._debug_layer_stop_triggered()),
                    "stop_exception_caught": stop_exception,
                    "quant_wall_s": quant_wall_s,
                    "quant_region_snapshot": quant_region_snapshot,
                    "memory_snapshots": self.memory_snapshots,
                    "layer_records": self.layer_records,
                    "visible_devices": [
                        {
                            "index": idx,
                            "name": torch.cuda.get_device_name(idx),
                        }
                        for idx in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else [],
                    "hf_device_map": hf_device_map,
                    "load_backend": self.LOAD_BACKEND.name if isinstance(self.LOAD_BACKEND, BACKEND) else str(self.LOAD_BACKEND),
                    "quant_backend": self.QUANT_BACKEND.name if isinstance(self.QUANT_BACKEND, BACKEND) else str(self.QUANT_BACKEND),
                }
                return result
            finally:
                del dataset
                del model
                torch_empty_cache()

    # Only apply split pools when the imported repo exposes the newer test
    # class knobs; older branches stay on their legacy single-strategy path.
    split_vram_pools_supported = (
        hasattr(BenchmarkQwen35Moe, "DENSE_VRAM_STRATEGY")
        and hasattr(BenchmarkQwen35Moe, "DENSE_VRAM_STRATEGY_DEVICES")
        and hasattr(BenchmarkQwen35Moe, "MOE_VRAM_STRATEGY")
        and hasattr(BenchmarkQwen35Moe, "MOE_VRAM_STRATEGY_DEVICES")
    )
    split_vram_pools_applied = False
    if split_vram_pools_supported:
        BenchmarkQwen35Moe.DENSE_VRAM_STRATEGY = resolved_dense_vram_strategy
        BenchmarkQwen35Moe.DENSE_VRAM_STRATEGY_DEVICES = dense_vram_strategy_devices
        BenchmarkQwen35Moe.MOE_VRAM_STRATEGY = resolved_moe_vram_strategy
        BenchmarkQwen35Moe.MOE_VRAM_STRATEGY_DEVICES = moe_vram_strategy_devices
        split_vram_pools_applied = True

    case = BenchmarkQwen35Moe()
    result = case.run_benchmark()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


def _ab_main(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="qwen35_moe_ab_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    current_vram_strategy = args.current_vram_strategy or args.vram_strategy
    baseline_vram_strategy = args.baseline_vram_strategy or args.vram_strategy
    current_cuda_visible_devices = args.current_cuda_visible_devices or args.cuda_visible_devices
    baseline_cuda_visible_devices = args.baseline_cuda_visible_devices or args.cuda_visible_devices
    current_dense_vram_strategy = args.current_dense_vram_strategy or args.dense_vram_strategy
    baseline_dense_vram_strategy = args.baseline_dense_vram_strategy or args.dense_vram_strategy
    current_dense_vram_strategy_devices = args.current_dense_vram_strategy_devices or args.dense_vram_strategy_devices
    baseline_dense_vram_strategy_devices = args.baseline_dense_vram_strategy_devices or args.dense_vram_strategy_devices
    current_moe_vram_strategy = args.current_moe_vram_strategy or args.moe_vram_strategy
    baseline_moe_vram_strategy = args.baseline_moe_vram_strategy or args.moe_vram_strategy
    current_moe_vram_strategy_devices = args.current_moe_vram_strategy_devices or args.moe_vram_strategy_devices
    baseline_moe_vram_strategy_devices = args.baseline_moe_vram_strategy_devices or args.moe_vram_strategy_devices

    current = baseline = None

    def _start_case(
        *,
        repo_root: Path,
        label: str,
        vram_strategy: str,
        cuda_visible_devices: Optional[str],
        dense_vram_strategy: str,
        dense_vram_strategy_devices: Optional[str],
        moe_vram_strategy: str,
        moe_vram_strategy_devices: Optional[str],
    ):
        json_out = output_dir / f"{label}.json"
        log_out = output_dir / f"{label}.log"
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        else:
            env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
        env["PYTHON_GIL"] = "0"
        env["DEBUG"] = "1"

        cmd = [
            sys.executable,
            str(script_path),
            "--single",
            "--repo-root",
            str(repo_root),
            "--json-out",
            str(json_out),
            "--label",
            label,
            "--model-path",
            args.model_path,
            "--dataset-size",
            str(args.dataset_size),
            "--batch-size",
            str(args.batch_size),
            "--quant-layers",
            str(args.quant_layers),
            "--stop-after-layer",
            str(args.stop_after_layer),
            "--dtype",
            args.dtype,
            "--attn-implementation",
            args.attn_implementation,
            "--vram-strategy",
            vram_strategy,
            "--dense-vram-strategy",
            dense_vram_strategy,
            "--moe-vram-strategy",
            moe_vram_strategy,
        ]
        if cuda_visible_devices is not None:
            cmd.extend(["--cuda-visible-devices", cuda_visible_devices])
        if dense_vram_strategy_devices is not None:
            cmd.extend(["--dense-vram-strategy-devices", dense_vram_strategy_devices])
        if moe_vram_strategy_devices is not None:
            cmd.extend(["--moe-vram-strategy-devices", moe_vram_strategy_devices])

        log_handle = log_out.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc, log_handle, json_out, log_out

    current_proc, current_log_handle, current_json_out, _ = _start_case(
        repo_root=args.current_root.resolve(),
        label="current",
        vram_strategy=current_vram_strategy,
        cuda_visible_devices=current_cuda_visible_devices,
        dense_vram_strategy=current_dense_vram_strategy,
        dense_vram_strategy_devices=current_dense_vram_strategy_devices,
        moe_vram_strategy=current_moe_vram_strategy,
        moe_vram_strategy_devices=current_moe_vram_strategy_devices,
    )
    baseline_proc, baseline_log_handle, baseline_json_out, _ = _start_case(
        repo_root=args.baseline_root.resolve(),
        label="baseline",
        vram_strategy=baseline_vram_strategy,
        cuda_visible_devices=baseline_cuda_visible_devices,
        dense_vram_strategy=baseline_dense_vram_strategy,
        dense_vram_strategy_devices=baseline_dense_vram_strategy_devices,
        moe_vram_strategy=baseline_moe_vram_strategy,
        moe_vram_strategy_devices=baseline_moe_vram_strategy_devices,
    )

    try:
        current_returncode = current_proc.wait()
        baseline_returncode = baseline_proc.wait()
    finally:
        current_log_handle.close()
        baseline_log_handle.close()

    if current_returncode != 0:
        raise subprocess.CalledProcessError(current_returncode, current_proc.args)
    if baseline_returncode != 0:
        raise subprocess.CalledProcessError(baseline_returncode, baseline_proc.args)

    with current_json_out.open("r", encoding="utf-8") as handle:
        current = json.load(handle)
    with baseline_json_out.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)

    compare = _compare_cases(current=current, baseline=baseline)
    compare_path = output_dir / "compare.json"
    with compare_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "current": current,
                "baseline": baseline,
                "compare": compare,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Results written to {output_dir}")
    _print_case_summary(current)
    _print_case_summary(baseline)
    _print_compare(compare)
    return 0


def main() -> int:
    args = _parse_args()
    if args.single:
        return _single_case_main(args)
    return _ab_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
