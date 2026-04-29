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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    in_features: int
    out_features: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress CUDA warmup via generated GPTQ modules. "
            "Earlier layers use fewer pool workers so a late layer triggers first-use warmup on a fresh worker."
        ),
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to target.")
    parser.add_argument("--workers", type=int, default=4, help="DeviceThreadPool workers for the target CUDA device.")
    parser.add_argument("--layers", type=int, default=31, help="Total synthetic layers to quantize.")
    parser.add_argument(
        "--late-layer",
        type=int,
        default=30,
        help="First layer index that uses the full late-module count and triggers the late worker.",
    )
    parser.add_argument(
        "--early-modules",
        type=int,
        default=3,
        help="Modules submitted per layer before --late-layer.",
    )
    parser.add_argument(
        "--late-modules",
        type=int,
        default=4,
        help="Modules submitted per layer from --late-layer onward.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Base hidden size for the generated modules.")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size per synthetic sample.")
    parser.add_argument("--seq-len", type=int, default=64, help="Calibration sequence length per synthetic sample.")
    parser.add_argument("--calib-batches", type=int, default=4, help="Synthetic calibration batches per module.")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bit width.")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size.")
    parser.add_argument(
        "--scope",
        choices=[ctx.value for ctx in WarmUpCtx],
        default=WarmUpCtx.THREAD.value,
        help="Warmup scope for the CUDA device pool.",
    )
    parser.add_argument(
        "--pressure-fraction",
        type=float,
        default=0.0,
        help="Reserve this fraction of currently free CUDA memory just before the late layer starts.",
    )
    parser.add_argument("--iterations", type=int, default=1, help="Number of isolated child-process trials to run.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _module_specs(hidden_dim: int, count: int) -> tuple[ModuleSpec, ...]:
    base_specs = (
        ("self_attention.query_key_value", hidden_dim, hidden_dim + max(64, hidden_dim // 4)),
        ("self_attention.dense", hidden_dim, hidden_dim),
        ("mlp.dense_h_to_4h", hidden_dim, hidden_dim * 4),
        ("mlp.dense_4h_to_h", hidden_dim * 4, hidden_dim),
    )

    specs: list[ModuleSpec] = []
    for index in range(count):
        name, in_features, out_features = base_specs[index % len(base_specs)]
        replica = index // len(base_specs)
        suffix = "" if replica == 0 else f".replica_{replica}"
        specs.append(ModuleSpec(f"{name}{suffix}", in_features, out_features))
    return tuple(specs)


def _allocate_pressure(device: torch.device, fraction: float) -> list[torch.Tensor]:
    if fraction <= 0.0:
        return []

    free_bytes, _ = torch.cuda.mem_get_info(device)
    target_bytes = max(0, int(free_bytes * fraction))
    if target_bytes <= 0:
        return []

    element_size = torch.tensor([], dtype=torch.float16).element_size()
    chunk_bytes = 256 * 1024 * 1024
    reserved: list[torch.Tensor] = []
    reserved_bytes = 0
    while reserved_bytes < target_bytes:
        current_bytes = min(chunk_bytes, target_bytes - reserved_bytes)
        reserved.append(torch.empty(max(1, current_bytes // element_size), device=device, dtype=torch.float16))
        reserved_bytes += reserved[-1].numel() * reserved[-1].element_size()
    return reserved


def _spawn_late_worker(pool: DeviceThreadPool, device: torch.device) -> str:
    key = pool._key(device)
    with pool._dispatch_lock:
        group = list(pool._worker_groups.get(key, []))
        worker_index = len(group)
        worker = pool._spawn_worker(device, key, name=f"DPWorker-{key}#{worker_index}")
        group.append(worker)
        pool._worker_groups[key] = group
        # Force the next submission to hit the freshly spawned worker first.
        pool._dispatch_rr[key] = worker_index
        pool._refresh_serial_worker_locked(key)
    return worker.name


def _prepare_gptq_task(
    *,
    spec: ModuleSpec,
    layer_idx: int,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    calib_batches: int,
    bits: int,
    group_size: int,
) -> GPTQ:
    module = nn.Linear(spec.in_features, spec.out_features, bias=False, dtype=torch.float16).to(device).eval()
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=min(max(1, group_size), spec.in_features),
        desc_act=False,
        sym=True,
    )
    gptq = GPTQ(module, qcfg=qcfg)
    gptq.name = f"transformer.h.{layer_idx}.{spec.name}"
    gptq.quantizer.configure(perchannel=True)

    for batch_idx in range(calib_batches):
        activations = torch.randn(
            batch_size,
            seq_len,
            spec.in_features,
            device=device,
            dtype=torch.float16,
        )
        outputs = module(activations)
        gptq.add_batch(activations, outputs, batch_index=batch_idx)
        del activations, outputs

    return gptq


def _quantize_gptq_task(gptq: GPTQ) -> dict[str, object]:
    thread_id = threading.get_ident()
    current_cuda = torch.cuda.current_device() if torch.cuda.is_available() else None
    started_at = time.perf_counter()
    qweight, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = gptq.quantize()
    elapsed = time.perf_counter() - started_at
    stats = {
        "name": getattr(gptq, "name", "<unnamed>"),
        "thread_id": thread_id,
        "current_cuda": current_cuda,
        "quantize_s": float(duration),
        "elapsed_s": elapsed,
        "avg_loss": avg_loss,
        "damp_percent": float(damp_percent),
        "nsamples": int(nsamples),
        "qweight_shape": tuple(qweight.shape),
    }
    del qweight, q_scales, q_zeros, q_g_idx, gptq
    return stats


def _print_stage_summary(layer_idx: int, result: dict[str, object], first_thread_use: bool) -> None:
    print(
        "StageLayer: layer=%s complete module=%s thread=%s first_thread_use=%s "
        "cuda_ctx=%s quantize_s=%.3f elapsed_s=%.3f nsamples=%s loss=%s shape=%s"
        % (
            layer_idx,
            result["name"],
            result["thread_id"],
            first_thread_use,
            result["current_cuda"],
            result["quantize_s"],
            result["elapsed_s"],
            result["nsamples"],
            result["avg_loss"],
            result["qweight_shape"],
        ),
        flush=True,
    )


def _run_child(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 2
    if args.device < 0 or args.device >= torch.cuda.device_count():
        print(f"CUDA device index {args.device} is unavailable.", file=sys.stderr)
        return 2
    if args.early_modules <= 0 or args.late_modules <= 0:
        print("Module counts must be positive.", file=sys.stderr)
        return 2

    device = torch.device("cuda", args.device)
    scope = WarmUpCtx(args.scope)
    specs = _module_specs(args.hidden_dim, max(args.early_modules, args.late_modules))
    if args.early_modules > args.late_modules:
        raise ValueError("early-modules cannot exceed late-modules")

    torch.cuda.set_device(device)
    initial_workers = args.workers
    spawn_late_worker = (
        args.workers > 1
        and args.early_modules < args.late_modules
        and args.late_modules >= args.workers
    )
    if spawn_late_worker:
        initial_workers = args.workers - 1

    pool = DeviceThreadPool(
        devices=[device],
        inference_mode=False,
        warmups={"cuda": WarmupTask(run_torch_linalg_warmup, scope=scope)},
        workers={f"cuda:{args.device}": initial_workers},
        empty_cache_every_n=0,
    )
    seen_threads: set[int] = set()
    pressure: list[torch.Tensor] = []

    print(
        f"child start: pid={os.getpid()} device={device} scope={scope.value} workers={args.workers} "
        f"initial_workers={initial_workers} late_worker_spawn={spawn_late_worker} "
        f"layers={args.layers} late_layer={args.late_layer} early_modules={args.early_modules} "
        f"late_modules={args.late_modules} hidden_dim={args.hidden_dim} calib_batches={args.calib_batches}",
        flush=True,
    )
    print(
        f"gil_enabled={getattr(sys, '_is_gil_enabled', lambda: 'n/a')()} "
        f"torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )

    try:
        for layer_idx in range(args.layers):
            active_modules = args.early_modules if layer_idx < args.late_layer else args.late_modules
            print(
                f"StageLayer: start layer={layer_idx}/{args.layers - 1} title=`Quantizing layer {layer_idx} of {args.layers - 1}` "
                f"modules={active_modules}",
                flush=True,
            )

            gptqs = []
            for spec in specs[:active_modules]:
                gptqs.append(_prepare_gptq_task(
                    spec=spec,
                    layer_idx=layer_idx,
                    device=device,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    calib_batches=args.calib_batches,
                    bits=args.bits,
                    group_size=args.group_size,
                ))

            if spawn_late_worker and layer_idx == args.late_layer:
                worker_name = _spawn_late_worker(pool, device)
                print(
                    f"StageLayer: spawned late worker name={worker_name} at layer={layer_idx}",
                    flush=True,
                )

            if layer_idx == args.late_layer and args.pressure_fraction > 0.0:
                pressure = _allocate_pressure(device, args.pressure_fraction)
                print(
                    f"StageLayer: late-layer pressure reserved tensors={len(pressure)} fraction={args.pressure_fraction:.3f}",
                    flush=True,
                )

            futures = [pool.submit(device, _quantize_gptq_task, gptq) for gptq in gptqs]

            for future in futures:
                result = future.result(timeout=900.0)
                thread_id = int(result["thread_id"])
                first_thread_use = thread_id not in seen_threads
                seen_threads.add(thread_id)
                _print_stage_summary(layer_idx, result, first_thread_use)

            if pressure:
                del pressure
                pressure = []

            torch.cuda.empty_cache()
            print(f"StageLayer: handoff complete for layer={layer_idx}", flush=True)

    finally:
        del pressure
        pool.shutdown(wait=True)

    print(f"trial complete: distinct_threads={sorted(seen_threads)}", flush=True)
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env.setdefault("PYTHON_GIL", "0")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    print(
        f"parent start: python={sys.executable} iterations={args.iterations} device=cuda:{args.device} "
        f"scope={args.scope} workers={args.workers} late_layer={args.late_layer}",
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
            "--layers",
            str(args.layers),
            "--late-layer",
            str(args.late_layer),
            "--early-modules",
            str(args.early_modules),
            "--late-modules",
            str(args.late_modules),
            "--hidden-dim",
            str(args.hidden_dim),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--calib-batches",
            str(args.calib_batches),
            "--bits",
            str(args.bits),
            "--group-size",
            str(args.group_size),
            "--scope",
            args.scope,
            "--pressure-fraction",
            str(args.pressure_fraction),
        ]
        print(f"[{iteration}/{args.iterations}] launching: {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            print(f"[{iteration}/{args.iterations}] child failed with exit code {completed.returncode}", flush=True)
            return completed.returncode

    print(f"completed {args.iterations} trials without reproducing the crash", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.child:
        return _run_child(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
