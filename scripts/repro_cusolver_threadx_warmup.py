#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress DeviceThreadPool torch.linalg warmups until a cusolver handle init crash reproduces.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to target.")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for the target device.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of isolated child-process trials.")
    parser.add_argument(
        "--scope",
        choices=[ctx.value for ctx in WarmUpCtx],
        default=WarmUpCtx.THREAD.value,
        help="Warmup scope to test.",
    )
    parser.add_argument(
        "--hold-ms",
        type=int,
        default=200,
        help="How long the first N-1 workers stay busy before the last worker warms up.",
    )
    parser.add_argument(
        "--pressure-fraction",
        type=float,
        default=0.0,
        help="Optional fraction of currently free CUDA memory to reserve before the late worker runs.",
    )
    parser.add_argument(
        "--child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _sleep_kernel_ms(ms: int) -> None:
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(ms * 1_000_000))
    else:
        time.sleep(ms / 1000.0)


def _allocate_pressure(device: torch.device, fraction: float) -> list[torch.Tensor]:
    if fraction <= 0.0:
        return []

    free_bytes, _ = torch.cuda.mem_get_info(device)
    target_bytes = max(0, int(free_bytes * fraction))
    if target_bytes <= 0:
        return []

    chunk_bytes = 256 * 1024 * 1024
    reserved: list[torch.Tensor] = []
    reserved_bytes = 0
    while reserved_bytes < target_bytes:
        current_bytes = min(chunk_bytes, target_bytes - reserved_bytes)
        numel = max(1, current_bytes // torch.tensor([], dtype=torch.float16).element_size())
        reserved.append(torch.empty(numel, device=device, dtype=torch.float16))
        reserved_bytes += reserved[-1].numel() * reserved[-1].element_size()
    return reserved


def _format_tids(records: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{label}={tid}" for label, tid in records)


def _run_child_trial(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 2
    if args.device < 0 or args.device >= torch.cuda.device_count():
        print(f"CUDA device index {args.device} is unavailable.", file=sys.stderr)
        return 2

    device = torch.device("cuda", args.device)
    scope = WarmUpCtx(args.scope)
    worker_key = f"cuda:{args.device}"
    thread_records: list[tuple[str, int]] = []
    pressure: list[torch.Tensor] = []
    started = [threading.Event() for _ in range(max(0, args.workers - 1))]
    release = threading.Event()

    def hold_worker(slot: int) -> tuple[int, int]:
        tid = threading.get_ident()
        thread_records.append((f"hold{slot}", tid))
        started[slot].set()
        vec = torch.randn(8192, device=device, dtype=torch.float32)
        vec.mul_(1.01)
        release.wait(timeout=max(1.0, args.hold_ms / 1000.0 * 5))
        _sleep_kernel_ms(args.hold_ms)
        return slot, tid

    def late_worker() -> tuple[int, int]:
        tid = threading.get_ident()
        thread_records.append(("late", tid))
        spd = torch.randn((8, 8), device=device, dtype=torch.float32)
        spd = spd @ spd.transpose(-1, -2) + torch.eye(8, device=device) * 1e-3
        torch.linalg.cholesky(spd)
        return torch.cuda.current_device(), tid

    print(
        f"child start: pid={os.getpid()} device={device} scope={scope.value} "
        f"workers={args.workers} hold_ms={args.hold_ms} pressure_fraction={args.pressure_fraction:.3f}",
        flush=True,
    )
    print(
        f"gil_enabled={getattr(sys, '_is_gil_enabled', lambda: 'n/a')()} "
        f"torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )

    torch.cuda.set_device(device)
    pool = DeviceThreadPool(
        devices=[device],
        inference_mode=False,
        warmups={"cuda": WarmupTask(run_torch_linalg_warmup, scope=scope)},
        workers={worker_key: args.workers},
        empty_cache_every_n=0,
    )

    try:
        futures = [pool.submit(device, hold_worker, index) for index in range(max(0, args.workers - 1))]
        for index, evt in enumerate(started):
            if not evt.wait(timeout=30.0):
                raise TimeoutError(f"worker {index} did not start in time")

        if args.pressure_fraction > 0.0:
            pressure = _allocate_pressure(device, args.pressure_fraction)
            print(f"reserved_pressure_tensors={len(pressure)}", flush=True)

        late_future = pool.submit(device, late_worker)
        late_result = late_future.result(timeout=60.0)
        release.set()
        for future in futures:
            future.result(timeout=60.0)

        print(f"trial ok: late_result={late_result} threads={_format_tids(thread_records)}", flush=True)
        return 0
    finally:
        release.set()
        del pressure
        pool.shutdown(wait=True)


def _run_parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    base_env = os.environ.copy()
    base_env.setdefault("PYTHON_GIL", "0")
    base_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    print(
        f"parent start: python={sys.executable} iterations={args.iterations} "
        f"device=cuda:{args.device} scope={args.scope} workers={args.workers}",
        flush=True,
    )
    for iteration in range(1, args.iterations + 1):
        cmd = [
            sys.executable,
            str(script),
            "--child",
            "--device",
            str(args.device),
            "--workers",
            str(args.workers),
            "--scope",
            args.scope,
            "--hold-ms",
            str(args.hold_ms),
            "--pressure-fraction",
            str(args.pressure_fraction),
        ]
        print(f"[{iteration}/{args.iterations}] launching: {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, env=base_env, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            print(f"[{iteration}/{args.iterations}] child failed with exit code {completed.returncode}", flush=True)
            return completed.returncode
    print(f"completed {args.iterations} trials without reproducing the crash", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.child:
        return _run_child_trial(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
