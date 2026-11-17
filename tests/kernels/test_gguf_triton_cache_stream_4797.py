# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A cold GGUF-Triton cache must be ordered for the stream that consumes it (#4797).

``GGUFTritonKernel`` builds ``_gguf_triton_cache`` on a module's first forward.
The build only *enqueues* the re-layout kernels that fill ``qs``/``scale``/
``min``, and the dict is published as soon as the build returns. A forward of
the same module on another stream therefore found the cached tensors and
launched the fused matmul on its own stream while the building stream had not
written them yet, reading uninitialized memory (garbage or NaN).

The probe reproduces that deterministically without threads: a dispatch mode
queues a long ``torch.cuda._sleep`` in front of every op on the building stream,
so the consumer stream runs first unless the cache carries an ordering event.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

IN_FEATURES = 512
OUT_FEATURES = 256


def _build_module(bits: str = "q4_k", seed: int = 4797):
    """Pack a normal linear into GGUF and mirror it into the Triton kernel."""

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


def _serial_reference(module, x):
    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()
        reference = module(x)
        torch.cuda.synchronize()
    return reference


@requires_cuda
@pytest.mark.cuda
def test_cold_cache_published_on_one_stream_is_ordered_for_another():
    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)
    reference = _serial_reference(module, x)

    # Drop the warm cache so the next forward rebuilds and re-publishes it.
    module.clear_weight_cache()

    stream_build = torch.cuda.Stream()
    stream_consume = torch.cuda.Stream()
    with torch.inference_mode():
        torch.cuda.synchronize()
        with torch.cuda.stream(stream_build), _DelayOnStream(stream_build):
            built = module(x)
        with torch.cuda.stream(stream_consume):
            consumed = module(x)
        torch.cuda.synchronize()

    # The building stream's own result is ordered by construction.
    torch.testing.assert_close(built, reference, rtol=0, atol=0)

    # The consumer stream must not read tensors the builder has not written.
    assert torch.isfinite(consumed).all(), "consumer stream read uninitialized cache"
    torch.testing.assert_close(consumed, reference, rtol=0, atol=0)


@requires_cuda
@pytest.mark.cuda
def test_warm_cache_on_the_same_stream_is_unchanged():
    """The ordering hook must not alter the single-stream path."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        first = module(x)
        second = module(x)
    torch.cuda.synchronize()

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@requires_cuda
@pytest.mark.cuda
def test_ready_event_is_retired_once_the_build_completes():
    """A completed build needs no ordering event on any later stream (#4867)."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()

    cached = list(module._gguf_triton_cache.values())
    assert cached, "the first forward must publish a cache"
    assert all("ready_event" in entry for entry in cached)

    consumer = torch.cuda.Stream()
    with torch.inference_mode():
        with torch.cuda.stream(consumer):
            module(x)
        torch.cuda.synchronize()

    # The build is long finished, so the consumer retired the event instead of
    # waiting on it, and every later forward is a plain kernel launch.
    assert all("ready_event" not in entry for entry in cached)


@requires_cuda
@pytest.mark.cuda
def test_capture_in_a_cuda_graph_after_a_warm_forward_matches_eager():
    """``SUPPORTS_GRAPH_CAPTURE`` must stay true after a normal forward (#4867).

    ``torch.cuda.graph`` captures on its own side stream, so a cross-stream
    ``wait_event`` on the cache builder's event fails the capture with
    ``CUDA_ERROR_STREAM_CAPTURE_ISOLATION``.
    """

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()
        reference = module(x).clone()
        torch.cuda.synchronize()

    # Warm up on a side stream the way the torch docs recommend.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        with torch.inference_mode():
            module(x)
            module(x)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            captured = module(x)

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, reference, rtol=0, atol=0)
