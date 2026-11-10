#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Probe PyTorch APIs for free-threaded (``PYTHON_GIL=0``) safety.

A wheel built for CPython ``3.14t`` is not automatically thread-safe.  Some
PyTorch entry points take a global lock, some are unsafe to call from several
threads, and at least one (``torch._C._broadcast_coalesced``) has been observed
both to hang and to publish torn copies on a GIL-disabled build.

Each probe runs in its own subprocess with a hard timeout, so a hang is recorded
as a result instead of wedging the run.  Results feed
``pytorch_gil_0_compat.md``.

Usage::

    PYTHON_GIL=0 python scripts/gil0_api_probe.py --all --json /tmp/gil0.json
    PYTHON_GIL=0 python scripts/gil0_api_probe.py --only torch.cuda.mem_get_info

Run it with ``PYTHON_GIL=0`` on a free-threaded build; on a regular GIL build
the interpreter serializes the calls and every probe trivially passes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List


PROBE_TIMEOUT_SECONDS = 90
THREADS = 8
ITERS = 4
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Probes that must only run once, on the main thread (process-global state).
SINGLE_SHOT = {"torch.cuda.CUDAGraph + torch.cuda.graph"}


def _gpu_count() -> int:
    import torch

    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def build_probes() -> Dict[str, Callable[[], Any]]:
    import torch
    from torch.cuda import comm

    def t(size: int = 256) -> "torch.Tensor":
        return torch.randn(size, size, device="cuda")

    def cross_device_to() -> bool:
        source = torch.randn(2048, 2048, device="cuda:0")
        moved = source.to("cuda:1", non_blocking=True)
        torch.cuda.current_stream("cuda:1").synchronize()
        return torch.equal(moved.cpu(), source.cpu())

    def cross_device_copy() -> bool:
        source = torch.randn(2048, 2048, device="cuda:0")
        target = torch.empty_like(source, device="cuda:1")
        target.copy_(source, non_blocking=True)
        torch.cuda.current_stream("cuda:1").synchronize()
        return torch.equal(target.cpu(), source.cpu())

    def broadcast_coalesced() -> bool:
        first = torch.randn(8192, device="cuda:0")
        second = torch.randn(8192, device="cuda:0")
        replicas = comm.broadcast_coalesced([first, second], [0, 1])[1]
        return torch.equal(replicas[0].cpu(), first.cpu()) and torch.equal(replicas[1].cpu(), second.cpu())

    def comm_broadcast() -> bool:
        source = torch.randn(8192, device="cuda:0")
        replica = comm.broadcast(source, [0, 1])[1]
        return torch.equal(replica.cpu(), source.cpu())

    def comm_reduce_add() -> bool:
        total = comm.reduce_add(
            [torch.ones(8192, device="cuda:0"), torch.ones(8192, device="cuda:1")],
            destination=0,
        )
        return bool(total.sum().item() == 16384)

    def comm_gather() -> bool:
        parts = comm.gather(
            [torch.ones(8192, device="cuda:0"), torch.ones(8192, device="cuda:1")],
            destination=0,
        )
        return bool(sum(part.sum().item() for part in parts) == 16384)

    def comm_scatter() -> bool:
        parts = comm.scatter(torch.ones(8192, device="cuda:0"), [0, 1])
        return all(bool(part.sum().item() == 8192) for part in parts)

    def event_roundtrip() -> bool:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            tensor = torch.ones(1024, device="cuda")
            event = torch.cuda.Event()
            event.record(stream)
        event.synchronize()
        return bool(tensor.sum().item() == 1024)

    def rng_roundtrip() -> bool:
        torch.cuda.set_rng_state_all(torch.cuda.get_rng_state_all())
        return True

    def graph_capture() -> bool:
        static_in = torch.randn(64, device="cuda")
        static_out = torch.empty_like(static_in)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out.copy_(static_in * 2)
        graph.replay()
        torch.cuda.synchronize()
        return torch.equal(static_out.cpu(), (static_in * 2).cpu())

    def save_load() -> bool:
        tensor = torch.randn(1024)
        handle = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        path = handle.name
        handle.close()
        try:
            torch.save({"t": tensor}, path)
            return torch.equal(torch.load(path, weights_only=True)["t"], tensor)
        finally:
            os.unlink(path)

    def generator_roundtrip() -> bool:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1234)
        return bool(torch.randn(64, generator=generator, device="cuda").abs().sum().item() > 0)

    def cpu_autocast_override() -> bool:
        x = torch.full((1, 32), 1000.0)
        weight = torch.full((32, 1), 3.125)
        before = torch.is_autocast_enabled("cpu")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outer = torch.is_autocast_enabled("cpu")
            narrowed = torch.mm(x, weight)
            with torch.autocast("cpu", enabled=False):
                inner = torch.is_autocast_enabled("cpu")
                wide = torch.mm(x, weight)
            resumed_enabled = torch.is_autocast_enabled("cpu")
            resumed = torch.mm(x, weight)
        correct = (
            not before
            and outer
            and not inner
            and resumed_enabled
            and not torch.is_autocast_enabled("cpu")
            and narrowed.dtype == resumed.dtype == torch.bfloat16
            and wide.dtype == torch.float32
            and narrowed.item() == resumed.item() == 99840.0
            and wide.item() == 100000.0
        )
        if not correct:
            raise AssertionError("CPU autocast state or FP32 reduction changed across a nested context")
        return True

    def cuda_autocast_disable() -> bool:
        x = torch.full((1, 32), 1000.0, device="cuda")
        weight = torch.full((32, 1), 3.125, device="cuda")
        before_dtype = torch.get_autocast_dtype("cuda")
        # BF16 keeps 100000 finite; FP16 would overflow it to `inf` and hide
        # the FP32 reduction this probe compares against.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outer = torch.is_autocast_enabled("cuda")
            outer_dtype = torch.get_autocast_dtype("cuda")
            narrowed = torch.mm(x, weight)
            with torch.autocast(device_type="cuda", enabled=False):
                inner = torch.is_autocast_enabled("cuda")
                # A disabled context keeps the enclosing dtype; only the
                # enabled flag changes.
                inner_dtype = torch.get_autocast_dtype("cuda")
                wide = torch.mm(x, weight)
            resumed_enabled = torch.is_autocast_enabled("cuda")
            resumed_dtype = torch.get_autocast_dtype("cuda")
            resumed = torch.mm(x, weight)
        correct = (
            outer
            and not inner
            and resumed_enabled
            and not torch.is_autocast_enabled("cuda")
            and before_dtype == torch.float16
            and outer_dtype == resumed_dtype == inner_dtype == torch.bfloat16
            and torch.get_autocast_dtype("cuda") is before_dtype
            and narrowed.dtype == resumed.dtype == torch.bfloat16
            and wide.dtype == torch.float32
            and narrowed.item() == resumed.item() == 99840.0
            and wide.item() == 100000.0
        )
        if not correct:
            raise AssertionError(
                "CUDA autocast state, context dtype, or FP32 reduction changed across a nested context"
            )
        return True

    def cuda_autocast_dtype() -> bool:
        # `torch.get_autocast_dtype` is read on the NF12-AD fallback path to
        # decide whether an active context narrows BF16 weights. The disabled
        # context keeps the enclosing dtype; only the enabled flag changes.
        before_enabled = torch.is_autocast_enabled("cuda")
        before_dtype = torch.get_autocast_dtype("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outer_dtype = torch.get_autocast_dtype("cuda")
            with torch.autocast(device_type="cuda", enabled=False):
                disabled_enabled = torch.is_autocast_enabled("cuda")
                disabled_dtype = torch.get_autocast_dtype("cuda")
            resumed_dtype = torch.get_autocast_dtype("cuda")
        correct = (
            not before_enabled
            and before_dtype == torch.float16
            and outer_dtype == disabled_dtype == resumed_dtype == torch.bfloat16
            and not disabled_enabled
            and torch.get_autocast_dtype("cuda") is before_dtype
        )
        if not correct:
            raise AssertionError("torch.get_autocast_dtype is not thread-local across nested contexts")
        return True

    def set_device_isolation(index: int = 0) -> bool:
        device = index % max(_gpu_count(), 1)
        torch.cuda.set_device(device)
        return int(torch.cuda.current_device()) == device

    nf12_lock = threading.Lock()
    nf12_thread = threading.local()
    nf12_next_device = [0]
    nf12_ready = [False]

    def nf12_ops() -> bool:
        import gptqmodel_nf12 as nf12

        # Build/compile on the main initializer path, then make the 8 workers
        # exercise actual concurrent operators on two visible GPUs.
        with nf12_lock:
            if not nf12_ready[0]:
                nf12.load()
                for device in range(2):
                    with torch.cuda.device(device):
                        w = torch.full((128, 512), 0.015625, dtype=torch.bfloat16, device="cuda")
                        p = nf12.encode(w)
                        x = torch.full((1, 512), 0.125, dtype=torch.float32, device="cuda")
                        nf12.linear_fp32(x, p)
                        nf12.linear(x.bfloat16(), p)
                        nf12.embedding(torch.tensor([0, 127], device="cuda"), p)
                        short = nf12.encode(torch.full(
                            (1024, 256), 0.015625, dtype=torch.bfloat16, device="cuda"))
                        nf12.linear_bf16_fp32(x[:, :256].bfloat16(), short)
                nf12_ready[0] = True
            if not hasattr(nf12_thread, "device"):
                nf12_thread.device = nf12_next_device[0] % 2
                nf12_next_device[0] += 1
        torch.cuda.set_device(nf12_thread.device)
        w = torch.full((128, 512), 0.015625, dtype=torch.bfloat16, device="cuda")
        p = nf12.encode(w)
        x = torch.full((1, 512), 0.125, dtype=torch.float32, device="cuda")
        ids = torch.tensor([0, 127], device="cuda")
        restored = nf12.decode(p)
        y_bf16 = nf12.linear(x.bfloat16(), p)
        y_fp32 = nf12.linear_fp32(x, p)
        gathered = nf12.embedding(ids, p)
        short = nf12.encode(torch.full(
            (1024, 256), 0.015625, dtype=torch.bfloat16, device="cuda"))
        y_mma_fp32 = nf12.linear_bf16_fp32(x[:, :256].bfloat16(), short)
        torch.cuda.current_stream().synchronize()
        if not (torch.equal(restored, w)
                and bool(torch.all(y_bf16 == 1))
                and bool(torch.all(y_fp32 == 1))
                and bool(torch.all(y_mma_fp32 == 0.5))
                and bool(torch.all(gathered == 0.015625))):
            raise AssertionError("concurrent NF12 encode/decode/linear/embedding returned wrong values")
        return True

    probes: Dict[str, Callable[[], Any]] = {
        "torch.cuda.is_available": torch.cuda.is_available,
        "torch.cuda.device_count": torch.cuda.device_count,
        "torch.cuda.get_device_name": lambda: torch.cuda.get_device_name(0),
        "torch.cuda.get_device_capability": lambda: torch.cuda.get_device_capability(0),
        "torch.cuda.get_device_properties": lambda: torch.cuda.get_device_properties(0).total_memory,
        "torch.cuda.current_device": torch.cuda.current_device,
        "torch.cuda.set_device": set_device_isolation,
        "gptqmodel_nf12.ops": nf12_ops,
        "torch.cuda.current_stream": lambda: str(torch.cuda.current_stream()),
        "torch.cuda.Stream": lambda: str(torch.cuda.Stream()),
        "torch.cuda.stream (context manager)": lambda: str(torch.cuda.stream(torch.cuda.Stream())),
        "torch.cuda.Event": event_roundtrip,
        "torch.cuda.synchronize": torch.cuda.synchronize,
        "torch.cuda.empty_cache": torch.cuda.empty_cache,
        "torch.cuda.mem_get_info": lambda: torch.cuda.mem_get_info(0)[0] >= 0,
        "torch.cuda.memory_allocated": torch.cuda.memory_allocated,
        "torch.cuda.max_memory_allocated": torch.cuda.max_memory_allocated,
        "torch.cuda.reset_peak_memory_stats": torch.cuda.reset_peak_memory_stats,
        "torch.cuda.memory_stats": lambda: len(torch.cuda.memory_stats()),
        "torch.cuda.set_per_process_memory_fraction": lambda: torch.cuda.set_per_process_memory_fraction(1.0),
        "torch.cuda.get_rng_state_all": lambda: len(torch.cuda.get_rng_state_all()),
        "torch.cuda.set_rng_state_all": rng_roundtrip,
        "torch.cuda.can_device_access_peer": lambda: bool(torch.cuda.can_device_access_peer(0, 1)),
        "torch.cuda.is_initialized": lambda: bool(torch.cuda.is_initialized()),
        "torch.cuda.is_current_stream_capturing": lambda: bool(torch.cuda.is_current_stream_capturing()),
        "torch.cuda.get_arch_list": lambda: len(torch.cuda.get_arch_list()),
        "torch.cuda.CUDAGraph + torch.cuda.graph": graph_capture,
        "torch.cuda.comm.broadcast_coalesced": broadcast_coalesced,
        "torch.cuda.comm.broadcast": comm_broadcast,
        "torch.cuda.comm.reduce_add": comm_reduce_add,
        "torch.cuda.comm.gather": comm_gather,
        "torch.cuda.comm.scatter": comm_scatter,
        "torch.empty": lambda: torch.empty(256, 256, device="cuda"),
        "torch.zeros": lambda: torch.zeros(256, 256, device="cuda"),
        "torch.ones": lambda: torch.ones(256, 256, device="cuda"),
        "torch.full": lambda: torch.full((256, 256), 3.0, device="cuda"),
        "torch.randn": t,
        "torch.rand": lambda: torch.rand(256, 256, device="cuda"),
        "torch.arange": lambda: torch.arange(4096, device="cuda"),
        "torch.eye": lambda: torch.eye(256, device="cuda"),
        "torch.tensor": lambda: torch.tensor([1.0, 2.0, 3.0], device="cuda"),
        "torch.as_tensor": lambda: torch.as_tensor([1.0, 2.0], device="cuda"),
        "torch.from_numpy": lambda: torch.from_numpy(__import__("numpy").arange(16)),
        "torch.zeros_like": lambda: torch.zeros_like(t(64)),
        "torch.ones_like": lambda: torch.ones_like(t(64)),
        "torch.empty_like": lambda: torch.empty_like(t(64)),
        "torch.full_like": lambda: torch.full_like(t(64), 2.0),
        "torch.cat": lambda: torch.cat([t(64), t(64)], dim=0),
        "torch.stack": lambda: torch.stack([t(64), t(64)]),
        "torch.split": lambda: torch.split(t(64), 32),
        "torch.unsqueeze": lambda: torch.unsqueeze(t(64), 0),
        "torch.diagonal": lambda: torch.diagonal(t(64)),
        "torch.matmul": lambda: torch.matmul(t(256), t(256)),
        "torch.einsum": lambda: torch.einsum("ij,jk->ik", t(128), t(128)),
        "torch.where": lambda: torch.where(t(128) > 0, t(128), t(128)),
        "torch.maximum": lambda: torch.maximum(t(128), t(128)),
        "torch.clamp": lambda: torch.clamp(t(128), -1.0, 1.0),
        "torch.round": lambda: torch.round(t(128)),
        "torch.rsqrt": lambda: torch.rsqrt(t(128).abs() + 1.0),
        "torch.topk": lambda: torch.topk(t(128).flatten(), 8),
        "torch.argsort": lambda: torch.argsort(t(128).flatten()),
        "torch.sum": lambda: torch.sum(t(128)),
        "torch.mean": lambda: torch.mean(t(128)),
        "torch.all": lambda: torch.all(t(128) > -100.0),
        "torch.equal": lambda: torch.equal(t(64), t(64)),
        "torch.isfinite": lambda: torch.isfinite(t(128)),
        "torch.isnan": lambda: torch.isnan(t(128)),
        "torch.bitwise_and": lambda: torch.bitwise_and(torch.arange(1024, device="cuda"), torch.arange(1024, device="cuda")),
        "torch.bitwise_or": lambda: torch.bitwise_or(torch.arange(1024, device="cuda"), torch.arange(1024, device="cuda")),
        "torch.bitwise_xor": lambda: torch.bitwise_xor(torch.arange(1024, device="cuda"), torch.arange(1024, device="cuda")),
        "torch.floor_divide": lambda: torch.floor_divide(torch.arange(1024, device="cuda:0"), 7),
        "torch.abs": lambda: torch.abs(t(128)),
        "torch.exp": lambda: torch.exp(t(128) * 0.01),
        "torch.log": lambda: torch.log(t(128).abs() + 1.0),
        "torch.pow": lambda: torch.pow(t(128).abs() + 1.0, 2.0),
        "Tensor.to (cuda:0 -> cuda:1)": cross_device_to,
        "Tensor.copy_ (cuda:0 -> cuda:1, non_blocking)": cross_device_copy,
        "torch.save + torch.load": save_load,
        "torch.Generator(device='cuda')": generator_roundtrip,
        "torch.inference_mode": lambda: _no_grad_result(torch, torch.inference_mode),
        "torch.no_grad": lambda: _no_grad_result(torch, torch.no_grad),
        "torch.is_autocast_enabled + torch.autocast (CPU)": cpu_autocast_override,
        "torch.is_autocast_enabled + torch.autocast (CUDA)": cuda_autocast_disable,
        "torch.get_autocast_dtype + torch.autocast (CUDA)": cuda_autocast_dtype,
    }
    return probes


def _no_grad_result(torch: Any, context: Callable[[], Any]) -> bool:
    with context():
        out = torch.ones(64, device="cuda") * 2
    return not out.requires_grad


def _requires_two_gpus(name: str) -> bool:
    return (".comm." in name or "cuda:0 -> cuda:1" in name
            or name.endswith("can_device_access_peer") or name == "gptqmodel_nf12.ops")


def run_child(name: str) -> int:
    import torch

    probes = build_probes()
    fn = probes[name]
    if _requires_two_gpus(name) and _gpu_count() < 2:
        print(json.dumps({"status": "SKIP", "detail": "needs >= 2 GPUs"}))
        return 0
    if name == "gptqmodel_nf12.ops" and torch.version.hip is not None:
        print(json.dumps({"status": "SKIP", "detail": "native NF12 operator is NVIDIA-only"}))
        return 0
    try:
        if name in SINGLE_SHOT:
            fn()
        else:
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                list(pool.map(lambda index: fn(), range(THREADS * ITERS)))
    except Exception as exc:  # noqa: BLE001 - report, never re-raise
        print(json.dumps({"status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}))
        return 0
    print(json.dumps({"status": "PASS"}))
    return 0


def run_probe(name: str) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", name],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "HANG", "detail": f"no result within {PROBE_TIMEOUT_SECONDS}s"}
    if completed.returncode != 0:
        tail = [line for line in (completed.stderr or completed.stdout or "").splitlines() if line][-3:]
        return {"status": "CRASH", "detail": f"exit {completed.returncode}: {' | '.join(tail)}"}
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"status": "ERROR", "detail": "probe produced no JSON result"}


def environment() -> Dict[str, Any]:
    import torch

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "python": sys.version.split()[0],
        "free_threaded": hasattr(sys, "_is_gil_enabled"),
        "gil_enabled": gil_enabled,
        "torch": torch.__version__,
        "backend": "ROCm" if getattr(torch.version, "hip", None) else "CUDA",
        "gpus": gpus,
        "platform": platform.platform(),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--all", action="store_true", help="probe every API (default)")
    parser.add_argument("--only", action="append", default=[], help="probe one API by name")
    parser.add_argument("--json", default="", help="write/read the results file")
    args = parser.parse_args(argv)

    if args.child:
        return run_child(args.child)

    names = args.only or list(build_probes())
    results: Dict[str, Any] = {}
    for index, name in enumerate(names, start=1):
        result = run_probe(name)
        results[name] = result
        print(f"[{index}/{len(names)}] {name}: {result['status']}", file=sys.stderr)

    payload = {"env": environment(), "results": results}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
