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

    p2p_guard_verdict = {}

    def p2p_allocator_guard() -> bool:
        """Check the ROCm VMM allocator guard from concurrent workers.

        Every cross-device copy consults this metadata-only guard, so it must be
        safe on worker threads and must return a stable verdict. The probe never
        issues a raw peer copy -- that is exactly the call the guard suppresses.
        """
        from gptqmodel.utils import p2p

        if torch.version.hip is None:
            return True
        with p2p_guard_verdict_lock:
            if "value" not in p2p_guard_verdict:
                p2p_guard_verdict["value"] = (
                    p2p._rocm_raw_peer_copy_unsafe(),
                    p2p._rocm_peer_link_unusable(),
                    p2p._rocm_pytorch_owns_peer_enablement(),
                )
            expected = p2p_guard_verdict["value"]
        actual = (
            p2p._rocm_raw_peer_copy_unsafe(),
            p2p._rocm_peer_link_unusable(),
            p2p._rocm_pytorch_owns_peer_enablement(),
        )
        if actual != expected:
            raise AssertionError("the P2P allocator guard verdict changed across threads")
        return True

    def set_device_isolation(index: int = 0) -> bool:
        device = index % max(_gpu_count(), 1)
        torch.cuda.set_device(device)
        return int(torch.cuda.current_device()) == device

    p2p_guard_verdict_lock = threading.Lock()
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

    nf12_gate_ready = [False]

    def nf12_gate() -> bool:
        import gptqmodel_nf12 as nf12

        with nf12_lock:
            if not nf12_gate_ready[0]:
                nf12.load()
                for device in range(2):
                    with torch.cuda.device(device):
                        weight = torch.full((384, 5120), 0.015625,
                                            dtype=torch.bfloat16, device="cuda")
                        weight[0, 0] = -0.0
                        packed = nf12.encode(weight)
                        nf12.prepare_fp32_gate(packed)
                        x = torch.full((6, 5120), 0.125, device="cuda")
                        nf12.linear_fp32_staged_b1(x, packed)
                        nf12.linear_fp32(x[:1], packed)
                        # The replacement route uses a different Triton
                        # kernel. Compile it before workers race through the
                        # cache test; Triton's cold JIT is not this cache.
                        changed = nf12.encode(-weight)
                        packed.packed = changed.packed
                        packed.escapes = changed.escapes
                        packed.exp_base = changed.exp_base
                        nf12.linear_fp32(x[:1], packed)
                nf12_gate_ready[0] = True
            if not hasattr(nf12_thread, "device"):
                nf12_thread.device = nf12_next_device[0] % 2
                nf12_next_device[0] += 1
        torch.cuda.set_device(nf12_thread.device)
        weight = torch.full((384, 5120), 0.015625,
                            dtype=torch.bfloat16, device="cuda")
        weight[0, 0] = -0.0
        packed = nf12.encode(weight)
        nf12.prepare_fp32_gate(packed)
        x = torch.full((6, 5120), 0.125, device="cuda")
        staged = nf12.linear_fp32_staged_b1(x, packed)
        ordinary = nf12.linear_fp32(x[:1], packed)
        torch.cuda.current_stream().synchronize()
        if not (torch.equal(staged[0:1], ordinary)
                and bool(torch.all((staged - torch.nn.functional.linear(x, weight.float())).abs() < 1e-4))):
            raise AssertionError("concurrent NF12 gate preparation or B1 projection changed values")
        changed = nf12.encode(-weight)
        packed.packed = changed.packed
        packed.escapes = changed.escapes
        packed.exp_base = changed.exp_base
        fallback = nf12.linear_fp32(x[:1], packed)
        nf12.prepare_fp32_gate(packed)
        rebuilt = nf12.linear_fp32_staged_b1(x, packed)
        expected = torch.nn.functional.linear(x, -weight.float())
        torch.cuda.current_stream().synchronize()
        if not (bool(torch.all((fallback - expected[:1]).abs() < 1e-4))
                and bool(torch.all((rebuilt - expected).abs() < 1e-4))):
            raise AssertionError("concurrent NF12 gate cache used a replaced weight")
        return True

    def tensor_cache_revision() -> bool:
        regular = torch.zeros(16, device="cuda")
        before = (regular._version, regular.data_ptr(), regular.device,
                  regular.dtype, tuple(regular.shape), regular.is_inference())
        regular.copy_(torch.ones_like(regular))
        after = (regular._version, regular.data_ptr(), regular.device,
                 regular.dtype, tuple(regular.shape), regular.is_inference())
        if after[0] != before[0] + 1 or after[1:] != before[1:]:
            raise AssertionError("ordinary tensor cache revision did not track an in-place write")
        with torch.inference_mode():
            inferred = torch.zeros(16, device="cuda")
            if not inferred.is_inference():
                raise AssertionError("inference tensor was not marked as unversioned")
            try:
                inferred._version
            except RuntimeError:
                pass
            else:
                raise AssertionError("inference tensor unexpectedly exposed a version")
        return True

    def embedding_row_gather() -> bool:
        # Quantized embeddings decode a lookup with per-token gathers on the
        # packed word row, the intra-word shift, and the scale/zero group. A
        # golden result for one fixed token set is computed once on the main
        # thread; every concurrent worker repeats the gathers and must match it
        # exactly, so a torn or cross-thread read is detected.
        vocab, dim, group_size = 4096, 128, 128
        num_groups = vocab // group_size
        state = {}

        def decode(weight, zeros, scales, groups, ids, device):
            tokens = torch.arange(vocab, device=device, dtype=torch.int32)
            word_of_token = torch.div(tokens, 8, rounding_mode="floor")
            shifts = tokens.remainder(8) * 4
            codes = (weight.index_select(0, word_of_token.index_select(0, ids))
                     >> shifts.index_select(0, ids).view(-1, 1)) & 0xF
            selected = groups.index_select(0, ids)
            return ((codes.to(torch.int16) - zeros.index_select(0, selected))
                    * scales.index_select(0, selected))

        if not hasattr(embedding_row_thread, "device"):
            embedding_row_thread.device = embedding_row_next_device[0] % 2
            embedding_row_next_device[0] += 1
        device = embedding_row_thread.device
        torch.cuda.set_device(device)

        with embedding_row_lock:
            if device not in state:
                weight = torch.randint(
                    0, 1 << 31, (vocab * 4 // 32, dim), dtype=torch.int32, device=device
                )
                zeros = torch.randint(0, 16, (num_groups, dim), dtype=torch.int16, device=device)
                scales = torch.rand(num_groups, dim, device=device, dtype=torch.float16)
                groups = torch.arange(vocab, device=device, dtype=torch.int32) // group_size
                ids = torch.randint(
                    0, vocab, (256,),
                    generator=torch.Generator(device=device).manual_seed(20261112),
                    device=device,
                )
                torch.cuda.synchronize()
                state[device] = (
                    weight, zeros, scales, groups, ids,
                    decode(weight, zeros, scales, groups, ids, device),
                )
            weight, zeros, scales, groups, ids, golden = state[device]

        got = decode(weight, zeros, scales, groups, ids, device)
        torch.cuda.current_stream().synchronize()
        if not bool(torch.equal(got, golden)):
            raise AssertionError("concurrent embedding row gather changed values")
        return True

    embedding_row_thread = threading.local()
    embedding_row_next_device = [0]
    embedding_row_lock = threading.Lock()

    awq_fp32_thread = threading.local()
    awq_fp32_next_device = [0]
    awq_fp32_lock = threading.Lock()
    awq_fp32_state = {}

    def awq_fp32_dequantize() -> bool:
        """Exercise the shared AWQ FP32 dense fallback on two worker devices."""
        from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm

        with awq_fp32_lock:
            if not hasattr(awq_fp32_thread, "device"):
                awq_fp32_thread.device = awq_fp32_next_device[0] % 2
                awq_fp32_next_device[0] += 1
            device = awq_fp32_thread.device
        torch.cuda.set_device(device)

        with awq_fp32_lock:
            if device not in awq_fp32_state:
                # Eight one-codes per packed word; every decoded weight equals
                # its FP32 scale. Both values would be lost through FP16.
                qweight = torch.full((128, 16), 0x11111111, device="cuda", dtype=torch.int32)
                qzeros = torch.zeros((1, 16), device="cuda", dtype=torch.int32)
                scales = torch.empty((1, 128), device="cuda", dtype=torch.float32)
                scales[:, ::2] = 1e-40
                scales[:, 1::2] = 65536.0
                awq_fp32_state[device] = (qweight, qzeros, scales)
            qweight, qzeros, scales = awq_fp32_state[device]

        with torch.inference_mode():
            weight = dequantize_gemm(qweight, qzeros, scales, 4, 128)
        torch.cuda.current_stream().synchronize()
        if weight.dtype != torch.float32 or not torch.equal(weight, scales.expand(128, 128)):
            raise AssertionError("concurrent AWQ FP32 dequantization changed checkpoint scales")
        return True

    gptq_scale_thread = threading.local()
    gptq_scale_next_device = [0]
    gptq_scale_lock = threading.Lock()

    def gptq_scale_storage_guard() -> bool:
        """Check FP16 scale cast decisions from concurrent pack workers."""
        from gptqmodel.nn_modules.qlinear import _scales_for_storage

        with gptq_scale_lock:
            if not hasattr(gptq_scale_thread, "device"):
                gptq_scale_thread.device = gptq_scale_next_device[0] % 2
                gptq_scale_next_device[0] += 1
            device = gptq_scale_thread.device
        torch.cuda.set_device(device)

        valid = torch.tensor([65504.0, 65505.0, 2.0**-24, 0.0], device="cuda")
        stored = _scales_for_storage(valid, name="valid.scales")
        expected = torch.tensor([65504.0, 65504.0, 2.0**-24, 0.0], device="cuda", dtype=torch.float16)
        invalid = torch.tensor([float("nan"), 65520.0, 1e-40], device="cuda")
        try:
            _scales_for_storage(invalid, name="invalid.scales")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("concurrent GPTQ scale guard accepted invalid scales")
        torch.cuda.current_stream().synchronize()
        if not torch.equal(stored, expected):
            raise AssertionError("concurrent GPTQ scale guard changed valid scales")
        if not all(part in message for part in (
            "source nonfinite=1", "cast overflow=1", "cast underflow=1"
        )):
            raise AssertionError(f"concurrent GPTQ scale guard misclassified scales: {message}")
        return True

    nf12_prefill_ready = [False]
    nf12_prefill_weights = {}

    def nf12_prefill_gate() -> bool:
        import gptqmodel_nf12 as nf12

        with nf12_lock:
            if not nf12_prefill_ready[0]:
                nf12.load()
                for device in range(2):
                    with torch.cuda.device(device):
                        weight = torch.full((384, 5120), 0.015625,
                                            dtype=torch.bfloat16, device="cuda")
                        packed = nf12.encode(weight)
                        x = torch.full((256, 5120), 0.125, device="cuda")
                        nf12.linear_fp32(x, packed)
                        nf12_prefill_weights[device] = packed
                nf12_prefill_ready[0] = True
            if not hasattr(nf12_thread, "device"):
                nf12_thread.device = nf12_next_device[0] % 2
                nf12_next_device[0] += 1
        torch.cuda.set_device(nf12_thread.device)
        x = torch.full((256, 5120), 0.125, device="cuda")
        bias = torch.full((384,), 0.25, device="cuda")
        output = nf12.linear_fp32(x, nf12_prefill_weights[nf12_thread.device], bias)
        torch.cuda.current_stream().synchronize()
        if not bool(torch.all(output == 10.25)):
            raise AssertionError("concurrent NF12 prefill gate returned wrong values")
        return True

    probes: Dict[str, Callable[[], Any]] = {
        "torch.cuda.is_available": torch.cuda.is_available,
        "torch.cuda.device_count": torch.cuda.device_count,
        "torch.cuda.get_device_name": lambda: torch.cuda.get_device_name(0),
        "torch.cuda.get_device_capability": lambda: torch.cuda.get_device_capability(0),
        "torch.cuda.get_device_properties": lambda: torch.cuda.get_device_properties(0).total_memory,
        "torch.cuda.current_device": torch.cuda.current_device,
        "gptqmodel.p2p.allocator_guard": p2p_allocator_guard,
        "torch.cuda.set_device": set_device_isolation,
        "gptqmodel_nf12.ops": nf12_ops,
        "gptqmodel_nf12.gate": nf12_gate,
        "Tensor cache revision metadata": tensor_cache_revision,
        "Tensor.index_select row gather": embedding_row_gather,
        "AWQ FP32 dequantize_gemm": awq_fp32_dequantize,
        "GPTQ FP16 scale storage guard": gptq_scale_storage_guard,
        "gptqmodel_nf12.prefill_gate": nf12_prefill_gate,
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
            or name.endswith("can_device_access_peer") or name.startswith("gptqmodel_nf12.")
            or name in ("AWQ FP32 dequantize_gemm", "GPTQ FP16 scale storage guard"))


def run_child(name: str) -> int:
    import torch

    probes = build_probes()
    fn = probes[name]
    if _requires_two_gpus(name) and _gpu_count() < 2:
        print(json.dumps({"status": "SKIP", "detail": "needs >= 2 GPUs"}))
        return 0
    if name.startswith("gptqmodel_nf12.") and torch.version.hip is not None:
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
