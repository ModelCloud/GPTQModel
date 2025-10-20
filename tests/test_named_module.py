# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import os
import queue
import random
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.utils.module_locks import parent_module_lock


def _make_linear(features: int = 8, device: torch.device | None = None) -> torch.nn.Linear:
    layer = torch.nn.Linear(features, features, bias=False)
    if device is not None:
        layer = layer.to(device=device)
    return layer


def test_named_module_register_and_state_locking():
    base = _make_linear()
    named = NamedModule(base, name="proj", full_name="model.layers.0.proj", layer_index=0)

    # register/unregister buffer should route through wrapped module and keep state updates serialized
    buf = torch.ones(1)
    named.register_buffer("unit", buf)
    assert "unit" in dict(named.named_buffers())
    named.unregister_buffer("unit")
    assert "unit" not in dict(named.named_buffers())

    # parameter registration proxies should also touch wrapped module
    param = torch.nn.Parameter(torch.randn_like(base.weight))
    named.register_parameter("alt_weight", param)
    assert dict(named.named_parameters())
    named.unregister_parameter("alt_weight")
    assert "alt_weight" not in dict(named.named_parameters())

    # setattr/getattr should delegate to wrapped module under lock
    named.new_attr = torch.zeros(1)
    assert torch.equal(named.new_attr, torch.zeros(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for streaming")
def test_named_module_streaming_apis():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    layer = _make_linear(device=device)
    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)

    payload = {
        "tensor": torch.randn(8, 8, device=device, dtype=torch.float16),
    }

    named.stream_state_payload_to_cpu(payload)
    assert "tensor" in named.state
    assert named.state["tensor"].is_pinned()

    named.stream_sync()
    torch.testing.assert_close(named.state["tensor"].cpu(), payload["tensor"].cpu())

    params = named.stream_parameters_to_cpu()
    assert params
    named.stream_sync()
    param_lookup = {name: tensor.detach().cpu() for name, tensor in named.module.named_parameters(recurse=False)}
    for name, cpu_tensor in params.items():
        torch.testing.assert_close(cpu_tensor.cpu(), param_lookup[name])

    buffers = named.stream_buffers_to_cpu()
    named.stream_sync()
    buffer_lookup = {name: tensor.detach().cpu() for name, tensor in named.module.named_buffers(recurse=False)}
    for name, cpu_tensor in buffers.items():
        torch.testing.assert_close(cpu_tensor.cpu(), buffer_lookup[name])

    combined = named.stream_all_to_cpu()
    named.stream_sync()
    assert set(combined.keys()) == {"parameters", "buffers"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for subprocess stream test")
def test_named_module_streaming_subprocess_roundtrip():
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "7")

    script = textwrap.dedent(
        """
        import torch
        from gptqmodel.looper.named_module import NamedModule

        layer = torch.nn.Linear(4, 4, bias=False).to(device='cuda', dtype=torch.float16)
        named = NamedModule(layer, name='proj', full_name='model.layers.0.proj', layer_index=0)

        payload = {'x': torch.randn(4, 4, device='cuda', dtype=torch.float16)}

        named.stream_state_payload_to_cpu(payload)
        named.stream_sync()
        torch.testing.assert_close(named.state['x'].cpu(), payload['x'].cpu(), atol=0, rtol=0)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"Subprocess streaming test unavailable: {result.stderr.strip()}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for multi-thread streaming stress test")
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="At least 4 CUDA devices required (0-3)")
def test_named_module_multithreaded_streaming_free_thread_stress():
    if not hasattr(sys, "_is_gil_enabled"):
        pytest.skip("Python runtime does not expose _is_gil_enabled; free-threading build required")
    if sys._is_gil_enabled():
        pytest.skip("GIL is enabled - run with PYTHON_GIL=0 to exercise free-threaded streaming stress")

    thread_count = 12
    duration_s = 30.0
    devices = [torch.device("cuda", idx) for idx in range(4)]
    bf16_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()

    @dataclass(frozen=True)
    class _ModuleContext:
        named: NamedModule
        device: torch.device

    @dataclass(frozen=True)
    class _ExpectedTensor:
        fingerprint: float
        checksum: float

    module_contexts: list[_ModuleContext] = []
    for idx, device in enumerate(devices):
        layer = _make_linear(2048).to(device=device, dtype=torch.bfloat16)
        named = NamedModule(layer, name=f"stress_proj_{idx}", full_name=f"stress.layers.{idx}.proj", layer_index=idx)
        module_contexts.append(_ModuleContext(named=named, device=device))

    pending_jobs: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    error_queue: queue.Queue = queue.Queue()
    stats_lock = threading.Lock()
    stats = {
        "payloads_issued": 0,
        "pending_enqueues": 0,
        "verified_same_thread": 0,
        "verified_cross_thread": 0,
        "empty_cache_calls": 0,
        "gc_collect_calls": 0,
        "largest_tensor_mb": 0,
    }

    def _record_stat(name: str, delta: int = 1) -> None:
        with stats_lock:
            stats[name] = stats.get(name, 0) + delta

    def _update_largest_tensor(val_mb: int) -> None:
        with stats_lock:
            stats["largest_tensor_mb"] = max(stats["largest_tensor_mb"], val_mb)

    def _fingerprint_last_value(tensor: torch.Tensor) -> float:
        flat = tensor.reshape(-1)
        last = flat[-1]
        if last.device.type != "cpu":
            last = last.to(dtype=torch.float32, device="cpu")
        else:
            last = last.to(dtype=torch.float32)
        return float(last.item())

    def _verify_expected(ctx: _ModuleContext, expected_items: tuple[tuple[str, _ExpectedTensor], ...]) -> bool:
        named = ctx.named
        for key, expected in expected_items:
            with parent_module_lock(named.full_name):
                host_tensor = named.state.get(key)
                event_map = named.state.get("streaming_event_map", {})
                pending_event = event_map.get(key)
            if host_tensor is None:
                ctx.named.stream_sync()
                with parent_module_lock(named.full_name):
                    host_tensor = named.state.get(key)
                if host_tensor is None:
                    with parent_module_lock(named.full_name):
                        available = sorted(str(k) for k in named.state.keys())
                    error_queue.put(f"Missing host tensor for key {key}; available={available}")
                    stop_event.set()
                    return False
            if pending_event is not None:
                pending_event.synchronize()
            actual_val = _fingerprint_last_value(host_tensor)
            actual_sum = float(host_tensor.to(dtype=torch.float32, device="cpu").sum().item())
            if (
                not math.isclose(actual_val, expected.fingerprint, rel_tol=0.0, abs_tol=1e-3)
                or not math.isclose(actual_sum, expected.checksum, rel_tol=0.0, abs_tol=1e-2)
            ):
                ctx.named.stream_sync()
                with parent_module_lock(named.full_name):
                    retry_tensor = named.state.get(key)
                if retry_tensor is not None:
                    retry_val = _fingerprint_last_value(retry_tensor)
                    retry_sum = float(retry_tensor.to(dtype=torch.float32, device="cpu").sum().item())
                else:
                    retry_val = None
                    retry_sum = None
                del host_tensor
                with parent_module_lock(named.full_name):
                    named.state.pop(key, None)
                error_queue.put(
                    "Mismatch for "
                    f"{key}: expected(last={expected.fingerprint}, sum={expected.checksum}), "
                    f"got(last={actual_val}, sum={actual_sum}), "
                    f"retry(last={retry_val}, sum={retry_sum})"
                )
                stop_event.set()
                return False
            del host_tensor
            with parent_module_lock(named.full_name):
                named.state.pop(key, None)
                named.state.get("streaming_event_map", {}).pop(key, None)
        return True

    def _maybe_empty_cache(device: torch.device, rng: random.Random, probability: float = 0.25) -> None:
        if rng.random() < probability:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            _record_stat("empty_cache_calls")

    def _try_consume(thread_id: int, rng: random.Random) -> bool:
        try:
            ctx, expected_items = pending_jobs.get_nowait()
        except queue.Empty:
            return False
        try:
            device = ctx.device
            torch.cuda.set_device(device)
            _maybe_empty_cache(device, rng, probability=0.3)
            if rng.random() < 0.3:
                gc.collect()
                _record_stat("gc_collect_calls")
            ctx.named.stream_sync()
            if _verify_expected(ctx, expected_items):
                _record_stat("verified_cross_thread")
            return True
        finally:
            pending_jobs.task_done()

    def _issue_payload(thread_id: int, rng: random.Random, seq_id: int) -> int:
        ctx = rng.choice(module_contexts)
        device = ctx.device
        torch.cuda.set_device(device)
        prefix = f"thread{thread_id}-seq{seq_id}"
        next_seq = seq_id + 1
        tensor_count = rng.randint(1, 3)
        tensor_sizes: list[int] = []
        payload: dict[str, torch.Tensor] = {}
        expected_pairs: list[tuple[str, _ExpectedTensor]] = []
        for idx in range(tensor_count):
            size_mb = rng.randint(3, 32)
            tensor_sizes.append(size_mb)
            numel = max(1, (size_mb * 1024 * 1024) // bf16_bytes)
            if numel >= 1024:
                cols = 256
                rows = max(1, numel // cols)
                shape = (rows, cols)
            else:
                shape = (numel,)
            tensor = torch.randn(shape, device=device, dtype=torch.bfloat16)
            key = f"{prefix}/tensor{idx}"
            payload[key] = tensor
            expected_pairs.append(
                (
                    key,
                    _ExpectedTensor(
                        fingerprint=_fingerprint_last_value(tensor),
                        checksum=float(tensor.to(dtype=torch.float32).sum().item()),
                    ),
                )
            )
        _update_largest_tensor(max(tensor_sizes))
        _record_stat("payloads_issued")
        _maybe_empty_cache(device, rng, probability=0.35)
        ctx.named.stream_state_payload_to_cpu(payload)
        if rng.random() < 0.35:
            gc.collect()
            _record_stat("gc_collect_calls")
        if rng.random() < 0.5:
            ctx.named.stream_sync()
            if _verify_expected(ctx, tuple(expected_pairs)):
                _record_stat("verified_same_thread")
        else:
            pending_jobs.put((ctx, tuple(expected_pairs)))
            _record_stat("pending_enqueues")
        time.sleep(rng.uniform(0.0, 0.003))
        return next_seq

    barrier = threading.Barrier(parties=thread_count + 1)
    deadline = time.monotonic() + duration_s

    def _worker(thread_id: int) -> None:
        rng = random.Random(0x9E3779B97F4A7C15 ^ thread_id)
        seq = 0
        try:
            barrier.wait()
            while time.monotonic() < deadline and not stop_event.is_set():
                processed = False
                if rng.random() < 0.6:
                    processed = _try_consume(thread_id, rng)
                if not processed:
                    seq = _issue_payload(thread_id, rng, seq)
        except Exception as exc:
            stop_event.set()
            error_queue.put(f"Thread {thread_id} error: {exc}")

    threads = [threading.Thread(target=_worker, args=(idx,), name=f"named-module-stress-{idx}") for idx in range(thread_count)]
    for thread in threads:
        thread.start()
    barrier.wait()

    for thread in threads:
        thread.join()

    while not pending_jobs.empty():
        ctx, expected_items = pending_jobs.get()
        torch.cuda.set_device(ctx.device)
        ctx.named.stream_sync()
        if not _verify_expected(ctx, expected_items):
            break
        pending_jobs.task_done()

    for device in devices:
        torch.cuda.synchronize(device=device)
    for ctx in module_contexts:
        with parent_module_lock(ctx.named.full_name):
            keys_to_remove = [key for key in ctx.named.state.keys() if key.startswith("thread")]
            for key in keys_to_remove:
                ctx.named.state.pop(key, None)

    if not error_queue.empty():
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())
        pytest.fail(" ; ".join(errors))

    with stats_lock:
        summary = {
            "payloads_issued": stats["payloads_issued"],
            "pending_enqueues": stats["pending_enqueues"],
            "verified_same_thread": stats["verified_same_thread"],
            "verified_cross_thread": stats["verified_cross_thread"],
            "empty_cache_calls": stats["empty_cache_calls"],
            "gc_collect_calls": stats["gc_collect_calls"],
            "largest_tensor_mb": stats["largest_tensor_mb"],
        }

    print(
        f"NamedModule multi-thread stress summary: "
        f"payloads={summary['payloads_issued']}, pending={summary['pending_enqueues']}, "
        f"verified_same_thread={summary['verified_same_thread']}, "
        f"verified_cross_thread={summary['verified_cross_thread']}, "
        f"empty_cache_calls={summary['empty_cache_calls']}, "
        f"gc_collect_calls={summary['gc_collect_calls']}, "
        f"largest_tensor_mb={summary['largest_tensor_mb']}"
    )
