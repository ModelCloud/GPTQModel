# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Concurrent GGUF-Triton forwards must not mutate the cache they iterate (#4962).

#4896 retires the cache's build handshake by popping ``ready_event``/
``ready_stream`` from the per-module cache dict, while a consumer that still has
a pending build iterates ``cache.values()`` to ``record_stream()`` the cached
tensors.  Two threads doing that at once raised

    RuntimeError: dictionary changed size during iteration

and failed the request.  The publish step now snapshots the tensors into an
immutable tuple, so the forward path never iterates the shared dict, and the
retirement is a pair of atomic single-key pops.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

IN_FEATURES = 1024
OUT_FEATURES = 512


def _build_module(bits: str = "q4_k", seed: int = 4962):
    torch.manual_seed(seed)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)

    torch_module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=False,
    )
    torch_module.pack_original(linear, scales=None, zeros=None)
    torch_module.post_init()

    triton_module = GGUFTritonKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    triton_module.load_state_dict(torch_module.state_dict(), strict=True)
    return triton_module.to("cuda").eval()


def _cache(module):
    return next(iter(module._gguf_triton_cache.values()))


def _attach_pending_handshake(module, sleep_cycles: int = 50_000_000):
    """Make the cache look like one whose build is still in flight (#4797)."""

    cache = _cache(module)
    builder = torch.cuda.Stream()
    with torch.cuda.stream(builder):
        torch.cuda._sleep(sleep_cycles)
        event = torch.cuda.Event()
        event.record(builder)
    cache["ready_stream"] = builder
    cache["ready_event"] = event
    assert not event.query(), "the handshake must be genuinely unfinished"
    return cache, builder


class _CountingDict(dict):
    """A cache entry that counts every mutation of its keys."""

    # ``mutations`` is test bookkeeping, not part of the mapping's identity:
    # keep dict's value-based equality so the extra attribute cannot make two
    # equal caches compare unequal (or vice versa).
    __eq__ = dict.__eq__
    __ne__ = dict.__ne__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutations = 0

    def __setitem__(self, key, value):
        self.mutations += 1
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self.mutations += 1
        super().__delitem__(key)

    def pop(self, key, *args):
        self.mutations += 1
        return super().pop(key, *args)

    def popitem(self):
        self.mutations += 1
        return super().popitem()

    def clear(self):
        self.mutations += 1
        return super().clear()

    def setdefault(self, key, default=None):
        self.mutations += 1
        return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        self.mutations += 1
        return super().update(*args, **kwargs)


@requires_cuda
@pytest.mark.cuda
def test_retirement_does_not_break_a_concurrent_record_stream_pass(monkeypatch):
    """A peer retiring the handshake must not break another thread's loop."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()
    cache, builder = _attach_pending_handshake(module)

    # Widen the loop so the retirement below lands in the middle of it.
    original = torch.Tensor.record_stream

    def _slow_record_stream(self, stream):
        time.sleep(0.05)
        return original(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", _slow_record_stream)

    errors: list[str] = []
    consumer = torch.cuda.Stream()

    def consume():
        with torch.inference_mode(), torch.cuda.stream(consumer):
            try:
                module(x)
            except Exception as exc:  # noqa: BLE001 - reported through the assert
                errors.append(f"{type(exc).__name__}: {exc}")
            consumer.synchronize()

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        time.sleep(0.15)  # the consumer is now inside the record_stream pass
        builder.synchronize()  # the build completes
        peer = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(peer):
            module(x)  # a peer observes completion and retires the handshake
            peer.synchronize()
    finally:
        thread.join()

    assert not errors, f"concurrent forward failed: {errors[0]}"
    assert "ready_event" not in cache


@requires_cuda
@pytest.mark.cuda
def test_n_thread_stream_burst_matches_the_serial_reference():
    """Every thread on its own stream must get the serial result (#4962)."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        reference = module(x).clone()
        torch.cuda.synchronize()
    cache, builder = _attach_pending_handshake(module)

    threads_count = 8
    barrier = threading.Barrier(threads_count)
    streams = [torch.cuda.Stream() for _ in range(threads_count)]
    outputs: list[torch.Tensor | None] = [None] * threads_count
    errors: list[str] = []

    def work(index: int):
        with torch.inference_mode(), torch.cuda.stream(streams[index]):
            barrier.wait()
            try:
                outputs[index] = module(x).clone()
            except Exception as exc:  # noqa: BLE001 - reported through the assert
                errors.append(f"thread {index}: {type(exc).__name__}: {exc}")
            streams[index].synchronize()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_count)]
    for thread in threads:
        thread.start()
    builder.synchronize()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent burst failed: {errors[0]}"
    for index, output in enumerate(outputs):
        assert output is not None, f"thread {index} produced no output"
        assert torch.equal(output, reference), f"thread {index} produced a different result"


@requires_cuda
@pytest.mark.cuda
def test_steady_state_never_mutates_the_cache():
    """After retirement a forward is a read-only dict lookup (#4956, #4962)."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        first = module(x).clone()
        torch.cuda.synchronize()
        module(x)  # retires the handshake
        torch.cuda.synchronize()

    module._gguf_triton_cache = {
        key: _CountingDict(entry) for key, entry in module._gguf_triton_cache.items()
    }
    for entry in module._gguf_triton_cache.values():
        assert "ready_event" not in entry and "ready_stream" not in entry

    with torch.inference_mode():
        for _ in range(32):
            assert torch.equal(module(x), first)

    for entry in module._gguf_triton_cache.values():
        assert entry.mutations == 0, "the steady-state forward mutated the cache dict"
