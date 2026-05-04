#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch


AB_CASE_SETS = (
    "gptq_group_sizes",
    "qwen3_6_27b_gptq",
    "qwen3_6_27b_awq",
    "qwen3_6_35b_a3b_gptq",
    "qwen3_6_35b_a3b_awq",
)
LOOP_MODELS = ("qwen3_6_27b", "qwen3_6_35b_a3b")
LOOP_METHODS = ("gptq", "awq")
LOOP_MODES = (
    "fallback",
    "native",
    "native_drop",
    "lookahead",
    "lookahead_drop",
    "prefetch_all",
    "prefetch_all_drop",
)
QUICK_MODES = (
    ("fallback", ("--no-komodo-native-int4",)),
    ("native_keep", ("--komodo-native-int4", "--no-komodo-drop-source-weights")),
    ("cannoe_keep", ("--komodo-native-int4", "--no-komodo-drop-source-weights", "--cannoe")),
    (
        "cannoe_prefetch_keep",
        ("--komodo-native-int4", "--no-komodo-drop-source-weights", "--cannoe", "--cannoe-prefetch"),
    ),
    ("native_drop", ("--komodo-native-int4", "--komodo-drop-source-weights")),
    ("prefetch_keep", ("--komodo-native-int4", "--komodo-prefetch-native-plan", "--no-komodo-drop-source-weights")),
    ("dequant_cache", ("--no-komodo-native-int4", "--komodo-cache-dequantized")),
)


@dataclass
class MatrixTask:
    label: str
    physical_device: int
    command: list[str]
    env: dict[str, str]


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Device list cannot be empty.")
    return values


def _visible_devices(raw: str) -> list[int]:
    if raw == "all":
        return list(range(torch.npu.device_count()))
    return _parse_csv_ints(raw)


def _add_task(
    tasks: list[MatrixTask],
    *,
    label: str,
    command: list[str],
    devices: list[int],
    env: dict[str, str] | None = None,
) -> None:
    physical_device = devices[len(tasks) % len(devices)]
    task_env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1",
        "ASCEND_GLOBAL_LOG_LEVEL": "3",
        "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
        "ASCEND_RT_VISIBLE_DEVICES": str(physical_device),
        "GPTQMODEL_TEST_NPU_DEVICE": "npu:0",
    }
    if env:
        task_env.update(env)
    tasks.append(MatrixTask(label=label, physical_device=physical_device, command=command, env=task_env))


def _build_tasks(args, output_dir: Path, devices: list[int]) -> list[MatrixTask]:
    tasks: list[MatrixTask] = []
    python = args.python

    if not args.skip_unit:
        for device in devices:
            _add_task(
                tasks,
                label=f"unit_komodo_phys{device}",
                command=[python, "-m", "pytest", "tests/test_npu_support.py", "-q", "-k", "komodo"],
                devices=[device],
            )

    if not args.skip_ab:
        ab_modes = [("native", ())]
        if args.include_cannoe_ab:
            ab_modes.append(("cannoe", ("--cannoe",)))
        for cases in AB_CASE_SETS:
            for tile in args.tiles:
                for mode_name, mode_flags in ab_modes:
                    for drop in (False, True):
                        label = f"ab_{cases}_{mode_name}_tile{tile}_drop{int(drop)}"
                        command = [
                            python,
                            "scripts/benchmark_komodo_npu_ab.py",
                            "--device",
                            "0",
                            "--cases",
                            cases,
                            "--dtype",
                            "fp16",
                            "--warmup",
                            str(args.warmup),
                            "--iters",
                            str(args.iters),
                            "--komodo-native-int4",
                            "--json-output",
                            str(output_dir / f"{label}.json"),
                            *mode_flags,
                        ]
                        command.append("--komodo-drop-source-weights" if drop else "--no-komodo-drop-source-weights")
                        _add_task(
                            tasks,
                            label=label,
                            command=command,
                            devices=devices,
                            env={"GPTQMODEL_KOMODO_PREPACK_TILE_N": str(tile)},
                        )

    if not args.skip_loop:
        for model in LOOP_MODELS:
            for method in LOOP_METHODS:
                for mode in LOOP_MODES:
                    label = f"loop_{model}_{method}_{mode}"
                    command = [
                        python,
                        "scripts/benchmark_komodo_npu_layer_loop.py",
                        "--device",
                        "0",
                        "--model",
                        model,
                        "--method",
                        method,
                        "--mode",
                        mode,
                        "--layers",
                        str(args.layers),
                        "--tokens",
                        str(args.tokens),
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                        "--stabilize-scale",
                        str(args.stabilize_scale),
                        "--json-output",
                        str(output_dir / f"{label}.json"),
                    ]
                    _add_task(
                        tasks,
                        label=label,
                        command=command,
                        devices=devices,
                        env={"GPTQMODEL_KOMODO_PREPACK_TILE_N": str(args.default_tile)},
                    )

    if not args.skip_quick:
        for tile in args.quick_tiles:
            for mode_name, flags in QUICK_MODES:
                label = f"quick_{mode_name}_tile{tile}"
                command = [
                    python,
                    "scripts/benchmark_komodo_npu_ab.py",
                    "--device",
                    "0",
                    "--cases",
                    "quick",
                    "--dtype",
                    "fp16",
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--json-output",
                    str(output_dir / f"{label}.json"),
                    *flags,
                ]
                _add_task(
                    tasks,
                    label=label,
                    command=command,
                    devices=devices,
                    env={"GPTQMODEL_KOMODO_PREPACK_TILE_N": str(tile)},
                )

    if args.include_memory:
        for cases in AB_CASE_SETS:
            for tile in args.memory_tiles:
                label = f"memory_{cases}_tile{tile}_drop1"
                command = [
                    python,
                    "scripts/benchmark_komodo_npu_prepack_memory.py",
                    "--device",
                    "0",
                    "--cases",
                    cases,
                    "--dtype",
                    "fp16",
                    "--tile-n",
                    str(tile),
                    "--drop-source",
                    "--json-output",
                    str(output_dir / f"{label}.json"),
                ]
                _add_task(tasks, label=label, command=command, devices=devices)

    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    return tasks


def _clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()


def _write_manifest(output_dir: Path, tasks: list[MatrixTask]) -> None:
    payload = [asdict(task) for task in tasks]
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _launch_task(index: int, total: int, task: MatrixTask, output_dir: Path) -> dict:
    log_path = output_dir / f"{index:03d}_{task.label}.log"
    log = log_path.open("wb")
    env = os.environ.copy()
    env.update(task.env)
    proc = subprocess.Popen(
        task.command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    print(
        f"START {index:03d}/{total:03d} dev={task.physical_device} pid={proc.pid} {task.label}",
        flush=True,
    )
    return {
        "index": index,
        "task": task,
        "proc": proc,
        "log": log,
        "log_path": log_path,
        "start": time.time(),
    }


def _run_tasks(tasks: list[MatrixTask], *, output_dir: Path, max_active: int) -> dict:
    pending = deque(enumerate(tasks, start=1))
    running: dict[int, dict] = {}
    results = []
    start_all = time.time()
    total = len(tasks)
    physical_devices = sorted({task.physical_device for task in tasks})
    max_active = min(max_active, max(1, len(physical_devices)))

    def pop_next_for_free_device() -> tuple[int, MatrixTask] | None:
        if not pending:
            return None
        busy_devices = {info["task"].physical_device for info in running.values()}
        for _ in range(len(pending)):
            index, task = pending.popleft()
            if task.physical_device not in busy_devices:
                return index, task
            pending.append((index, task))
        return None

    while pending or running:
        while pending and len(running) < max_active:
            next_task = pop_next_for_free_device()
            if next_task is None:
                break
            index, task = next_task
            launched = _launch_task(index, total, task, output_dir)
            running[launched["proc"].pid] = launched

        time.sleep(1.0)
        for pid, info in list(running.items()):
            returncode = info["proc"].poll()
            if returncode is None:
                continue
            info["log"].close()
            elapsed = time.time() - info["start"]
            task = info["task"]
            result = {
                "index": info["index"],
                "label": task.label,
                "physical_device": task.physical_device,
                "returncode": returncode,
                "elapsed_s": elapsed,
                "log": str(info["log_path"]),
            }
            results.append(result)
            status = "PASS" if returncode == 0 else "FAIL"
            print(
                f"{status}  {info['index']:03d}/{total:03d} dev={task.physical_device} "
                f"rc={returncode} elapsed={elapsed:.1f}s {task.label}",
                flush=True,
            )
            del running[pid]

    return {
        "output_dir": str(output_dir),
        "task_count": total,
        "max_active": max_active,
        "elapsed_s": time.time() - start_all,
        "passed": sum(result["returncode"] == 0 for result in results),
        "failed": sum(result["returncode"] != 0 for result in results),
        "results": sorted(results, key=lambda result: result["index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Komodo NPU validation matrix across visible NPUs.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/komodo_npu_matrix"))
    parser.add_argument("--devices", default="all", help="Comma-separated physical NPU ids, or `all`.")
    parser.add_argument("--max-active", type=int, help="Maximum concurrent processes. Defaults to device count.")
    parser.add_argument("--python", default="python")
    parser.add_argument("--tiles", type=int, nargs="+", default=[0, 512, 1024, 2048])
    parser.add_argument("--quick-tiles", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--memory-tiles", type=int, nargs="+", default=[0, 512, 1024, 2048])
    parser.add_argument("--default-tile", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--stabilize-scale", type=float, default=0.001)
    parser.add_argument("--include-memory", action="store_true")
    parser.add_argument("--include-cannoe-ab", action="store_true")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-ab", action="store_true")
    parser.add_argument("--skip-loop", action="store_true")
    parser.add_argument("--skip-quick", action="store_true")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required for the Komodo validation matrix.")

    devices = _visible_devices(args.devices)
    device_count = torch.npu.device_count()
    invalid = [device for device in devices if device < 0 or device >= device_count]
    if invalid:
        raise ValueError(f"Invalid physical NPU ids {invalid}; visible device count is {device_count}.")
    max_active = args.max_active if args.max_active is not None else len(devices)
    if max_active < 1:
        raise ValueError("--max-active must be >= 1.")

    _clean_output_dir(args.output_dir)
    tasks = _build_tasks(args, args.output_dir, devices)
    _write_manifest(args.output_dir, tasks)
    print(f"output_dir={args.output_dir}")
    print(f"tasks={len(tasks)} max_active={max_active} devices={devices}")

    if args.dry_run:
        return

    summary = _run_tasks(tasks, output_dir=args.output_dir, max_active=max_active)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "SUMMARY "
        + json.dumps(
            {
                "task_count": summary["task_count"],
                "max_active": summary["max_active"],
                "elapsed_s": summary["elapsed_s"],
                "passed": summary["passed"],
                "failed": summary["failed"],
            },
            indent=2,
        ),
        flush=True,
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
