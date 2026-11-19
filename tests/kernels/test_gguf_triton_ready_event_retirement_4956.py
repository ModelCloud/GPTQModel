# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A GGUF-Triton cache must retire its build event on every stream (#4956).

#4841 ordered a cold cache against the consuming stream with an event recorded
on the building stream.  #4896 retired that event once ``Event.query()`` reported
the build complete, but only on the *non-builder* branch, so the normal
single-stream case - the stream that built the cache - kept ``ready_event`` and
``ready_stream`` in the dict forever and paid a ``torch.cuda.current_stream()``
wrapper on every forward (~11 us on this host, ~7% of a decode-sized launch).

The handshake is now retired by the first forward that observes a completed
build, on any stream, and ``_wait_for_triton_cache_ready`` returns before
touching any stream API once it is gone.  These tests pin that, plus the two
behaviours the retirement must not break: the #4797 cold-build ordering and the
#4867 CUDA graph capture.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

IN_FEATURES = 1024
OUT_FEATURES = 512


def _build_module(bits: str = "q4_k", seed: int = 4956):
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


def _caches(module):
    return list(module._gguf_triton_cache.values())


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
@pytest.mark.parametrize("bits", ["q4_k", "q6_k"])
def test_same_stream_warm_forward_retires_the_handshake(bits):
    """The builder stream must retire too, not just a second consumer (#4956)."""

    module = _build_module(bits)
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()
    assert any("ready_event" in entry for entry in _caches(module)), (
        "the first forward must publish the build handshake"
    )

    with torch.inference_mode():
        # Same stream, build known complete: this forward retires the handshake.
        first = module(x).clone()
        torch.cuda.synchronize()
    assert all("ready_event" not in entry for entry in _caches(module))
    assert all("ready_stream" not in entry for entry in _caches(module))

    # ... and it stays retired for every later forward.
    with torch.inference_mode():
        for _ in range(8):
            assert torch.equal(module(x), first)
    assert all("ready_event" not in entry for entry in _caches(module))


@requires_cuda
@pytest.mark.cuda
def test_cold_build_consumed_on_another_stream_is_still_ordered():
    """The #4797 ordering must survive the retirement (#4956)."""

    module = _build_module()
    x = torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        reference = module(x).clone()
        torch.cuda.synchronize()

    # Rebuild the cache on a deliberately slow stream, then consume it from a
    # second stream while the build is still in flight.
    module.clear_weight_cache()
    builder = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.inference_mode():
        torch.cuda.synchronize()
        with torch.cuda.stream(builder), _DelayOnStream(builder):
            built = module(x).clone()
        with torch.cuda.stream(consumer):
            consumed = module(x).clone()
        torch.cuda.synchronize()

    torch.testing.assert_close(built, reference, rtol=0, atol=0)
    assert torch.isfinite(consumed).all(), "consumer stream read uninitialized cache"
    torch.testing.assert_close(consumed, reference, rtol=0, atol=0)

    # A synchronized forward afterwards retires the handshake again.
    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize()
        module(x)
    assert all("ready_event" not in entry for entry in _caches(module))


@requires_cuda
@pytest.mark.cuda
def test_capture_after_a_warm_forward_still_matches_eager():
    """The #4867 capture contract must survive the retirement (#4956)."""

    module = _build_module()
    x = torch.randn(16, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        eager = module(x).clone()
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            module(x)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        graph.replay()
        torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@requires_cuda
@pytest.mark.cuda
def test_steady_state_is_bit_identical_to_the_first_forward():
    module = _build_module()
    x = torch.randn(4, IN_FEATURES, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        first = module(x).clone()
        torch.cuda.synchronize()
        outputs = [module(x).clone() for _ in range(32)]
    for index, output in enumerate(outputs):
        assert torch.equal(output, first), f"forward {index} drifted from the first"
