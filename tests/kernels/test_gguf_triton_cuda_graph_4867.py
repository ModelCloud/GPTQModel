# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A warm GGUF-Triton cache must not break CUDA graph capture (#4867).

#4841 ordered a cold cache against the consuming stream by recording an event on
the building stream and having every later forward on another stream wait for it.
``torch.cuda.graph`` captures on its own side stream, and a capturing stream may
only wait on events recorded *inside* the capture, so that wait failed with
``cudaErrorStreamCaptureIsolation`` and invalidated the capture as soon as the
module had served one ordinary forward.

``_wait_for_triton_cache_ready`` now retires the handshake once the build has
completed, so a warmed module captures and replays bit-exactly, and refuses
loudly (instead of corrupting) if a capture really needs an unfinished cache.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import (
    GGUFTritonKernel,
    _wait_for_triton_cache_ready,
)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

IN_FEATURES = 1024
OUT_FEATURES = 512
QTYPES = ("q4_k", "q6_k")


def _build_module(bits: str = "q6_k", seed: int = 4867):
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


def _capture_and_replay(module, x):
    """Capture one forward on a side stream and return the replayed output."""

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(x)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = module(x)
    graph.replay()
    torch.cuda.synchronize()
    return captured


class _DelayOnStream:
    """Queue a long device-side sleep in front of every op on ``stream``."""

    def __init__(self, stream: torch.cuda.Stream):
        self.stream = stream

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        outer = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if torch.cuda.current_stream() == outer.stream:
                    torch.cuda._sleep(20_000_000)
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._mode.__exit__(*exc_info)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("bits", QTYPES)
@pytest.mark.parametrize("rows", [1, 16])
def test_capture_after_an_eager_forward_matches_eager(bits, rows):
    module = _build_module(bits)
    x = torch.randn(rows, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        # The ordinary first forward builds the cache on the default stream.
        eager = module(x).clone()
        torch.cuda.synchronize()
        assert "ready_event" in next(iter(module._gguf_triton_cache.values()))

        captured = _capture_and_replay(module, x)

    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@requires_cuda
@pytest.mark.cuda
def test_capture_after_a_cross_stream_build_matches_eager():
    """The #4841 ordering and #4867 capture must coexist.

    The cache is built on one stream while the default stream consumes it, which
    retires the ready event, and only then is a graph captured.
    """

    module = _build_module()
    x = torch.randn(16, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        torch.cuda.synchronize()
        builder = torch.cuda.Stream()
        with torch.cuda.stream(builder), _DelayOnStream(builder):
            built = module(x).clone()
        torch.cuda.synchronize()
        eager = module(x).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(built, eager, rtol=0, atol=0)

        # The cross-stream consumer retired the handshake.
        cache = next(iter(module._gguf_triton_cache.values()))
        assert "ready_event" not in cache and "ready_stream" not in cache

        captured = _capture_and_replay(module, x)

    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@requires_cuda
@pytest.mark.cuda
def test_warm_cache_keeps_no_stream_handshake():
    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            module(x)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

    cache = next(iter(module._gguf_triton_cache.values()))
    assert "ready_event" not in cache
    assert "ready_stream" not in cache


@requires_cuda
@pytest.mark.cuda
def test_capture_neither_queries_nor_waits_on_the_handshake(monkeypatch):
    """Capture must not touch the event at all: both calls are illegal there."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module(x)
    cache = next(iter(module._gguf_triton_cache.values()))

    # Leave a genuinely unfinished build event in the cache.
    builder = torch.cuda.Stream()
    with torch.cuda.stream(builder):
        torch.cuda._sleep(50_000_000)
        event = torch.cuda.Event()
        event.record(builder)
    cache["ready_stream"] = builder
    cache["ready_event"] = event
    assert not event.query()

    def _illegal(*args, **kwargs):
        raise AssertionError("Event.query() is not permitted while capturing")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda.Event, "query", _illegal)
    # Returns without waiting and retires the handshake instead.
    _wait_for_triton_cache_ready(cache, torch.device("cuda"))
    assert "ready_event" not in cache and "ready_stream" not in cache
    torch.cuda.synchronize()
