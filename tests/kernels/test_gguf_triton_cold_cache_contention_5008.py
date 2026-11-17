# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A cold GGUF-Triton burst must publish one cache per module (#5008).

``_get_triton_cache`` built a cache per calling thread and then stored it under
the module's ``(device, qtype, wide_dot)`` key, so the *last* cold builder
replaced the entry every other thread was reading.  A consumer that picked the
entry up after its handshake was retired (``#4956``: no ``record_stream`` on
that path) held the only remaining Python reference: when its forward returned,
the entry's ``scale``/``min`` tensors were freed and the caching allocator could
hand those blocks to the building stream while the consumer's fused matmul was
still pending.  The first concurrent forward of a module then returned NaN or
garbage (11/20 trials on the reporter's host) while every warm forward was
correct.

The cold build is now serialized per module, so a burst builds once and every
thread launches against the entry the module publishes.  That keeps ``#4956``'s
zero-cost steady state (a retired entry is never dropped, so nothing has to
record a stream) and ``#4962``'s rule that the forward path never mutates the
cache it iterates.
"""

from __future__ import annotations

import copy
import threading

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

IN_FEATURES = 1024
OUT_FEATURES = 512


def _build_module(bits: str = "q6_k", seed: int = 5008, in_features: int = IN_FEATURES,
                  out_features: int = OUT_FEATURES):
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)

    source = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()

    module = GGUFTritonKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.load_state_dict(source.state_dict(), strict=True)
    return module.to("cuda").eval()


@requires_cuda
@pytest.mark.cuda
def test_concurrent_cold_build_publishes_one_cache(monkeypatch):
    """Two threads racing the first forward must share the published cache.

    The first builder is parked inside ``_build_triton_cache`` so the second
    thread provably reaches the cold path while the build is in flight.  Before
    the fix both threads built, both stored, and the loser's tensors were the
    ones the peer had already launched against.
    """

    module = _build_module()
    device = torch.device("cuda", torch.cuda.current_device())
    original_build = GGUFTritonKernel._build_triton_cache
    build_started = threading.Event()
    build_release = threading.Event()
    builds: list[int] = []

    def parked_build(self, device, wide_dot=False):
        builds.append(threading.get_ident())
        build_started.set()
        assert build_release.wait(30.0), "the test never released the cold build"
        return original_build(self, device, wide_dot)

    monkeypatch.setattr(GGUFTritonKernel, "_build_triton_cache", parked_build)

    first: dict[str, object] = {}
    second: dict[str, object] = {}
    errors: list[str] = []

    def take(cache_slot):
        try:
            cache_slot["cache"] = module._get_triton_cache(device)
        except Exception as exc:  # noqa: BLE001 - reported through the assert
            errors.append(f"{type(exc).__name__}: {exc}")

    first_thread = threading.Thread(target=take, args=(first,))
    first_thread.start()
    assert build_started.wait(30.0), "the first cold build never started"

    second_thread = threading.Thread(target=take, args=(second,))
    second_thread.start()
    # Let the second thread reach the cold path. It must wait for the first
    # build instead of starting a second one.
    second_thread.join(0.5)
    overlapped_builds = len(builds)

    build_release.set()
    first_thread.join(30.0)
    second_thread.join(30.0)

    assert not errors, f"a concurrent cold lookup failed: {errors[0]}"
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert overlapped_builds == 1, (
        "a second thread entered the cold build while the first was still in "
        f"flight ({overlapped_builds} builders)"
    )
    assert len(builds) == 1, f"the module was built {len(builds)} times"
    assert second["cache"] is first["cache"], (
        "a concurrent cold forward used a cache the module does not publish"
    )
    assert module._gguf_triton_cache[module._triton_cache_key(device)] is first["cache"]


@requires_cuda
@pytest.mark.cuda
def test_cold_burst_matches_the_serial_reference():
    """Every thread's first forward must equal a serially warmed twin.

    This is the end-to-end symptom of #5008: NaN or garbage on the first
    concurrent forward, while warm forwards are correct.  It is a soak:
    the deterministic guard above pins the mechanism, this one pins the result.
    It also asserts the invariant that makes the result correct: every thread
    launched against the entry the module publishes, never against a private
    cache that a peer's store has already dropped.
    """

    modules = [_build_module(seed=5008 + index) for index in range(8)]
    twins = [_build_module(seed=5008 + index) for index in range(8)]
    xs = [torch.randn(1, IN_FEATURES, device="cuda", dtype=torch.float16) for _ in modules]

    with torch.inference_mode():
        reference = [twin(x).clone() for twin, x in zip(twins, xs)]
        torch.cuda.synchronize()
    del twins

    threads_count = 8
    barrier = threading.Barrier(threads_count)
    streams = [torch.cuda.Stream() for _ in range(threads_count)]
    outputs: list[list[torch.Tensor | None]] = [[None] * len(modules) for _ in range(threads_count)]
    used: list[list[object]] = [[None] * len(modules) for _ in range(threads_count)]
    errors: list[str] = []

    def work(index: int):
        with torch.inference_mode(), torch.cuda.stream(streams[index]):
            barrier.wait()
            try:
                for module_index, (module, x) in enumerate(zip(modules, xs)):
                    used[index][module_index] = module._get_triton_cache(x.device)
                    outputs[index][module_index] = module(x).clone()
            except Exception as exc:  # noqa: BLE001 - reported through the assert
                errors.append(f"thread {index}: {type(exc).__name__}: {exc}")
            streams[index].synchronize()

    threads = [threading.Thread(target=work, args=(index,)) for index in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120.0)

    assert not errors, f"a concurrent cold forward failed: {errors[0]}"
    for module_index, module in enumerate(modules):
        published = module._gguf_triton_cache.get(module._triton_cache_key(xs[module_index].device))
        assert published is not None
        for thread_index in range(threads_count):
            assert used[thread_index][module_index] is published, (
                f"thread {thread_index} used a cache module {module_index} does not publish"
            )
    for thread_index in range(threads_count):
        for module_index, expected in enumerate(reference):
            output = outputs[thread_index][module_index]
            assert output is not None, f"thread {thread_index} module {module_index} produced no output"
            assert not torch.isnan(output).any(), (
                f"thread {thread_index} module {module_index} returned NaN"
            )
            assert torch.equal(output, expected), (
                f"thread {thread_index} module {module_index} returned a different result"
            )


@requires_cuda
@pytest.mark.cuda
def test_build_lock_is_not_module_state():
    """The cold-build lock must not make a quantized module un-copyable.

    It is kept in a weak map keyed by the module: a ``threading.Lock``
    attribute would break ``copy.deepcopy`` (and ``torch.save``) of a loaded
    model with ``cannot pickle '_thread.lock' object``.
    """

    module = _build_module(in_features=256, out_features=64)
    assert "_triton_cache_build_lock" not in vars(module)
    with torch.inference_mode():
        x = torch.randn(1, 256, device="cuda", dtype=torch.float16)
        module(x)
        torch.cuda.synchronize()
        module(x)  # retires the handshake, so the cache holds no ``Stream``
        torch.cuda.synchronize()
    entry = next(iter(module._gguf_triton_cache.values()))
    assert "ready_stream" not in entry

    clone = copy.deepcopy(module)
    assert clone is not module
    assert clone.qweight is not module.qweight
    assert clone.qweight.shape == module.qweight.shape
    assert clone._gguf_triton_cache is not module._gguf_triton_cache

    # Each instance owns its own lock, and a deep copy must not share it.
    from gptqmodel.nn_modules.qlinear.gguf_triton import _triton_cache_build_lock

    assert _triton_cache_build_lock(module) is not _triton_cache_build_lock(clone)
    assert _triton_cache_build_lock(module) is _triton_cache_build_lock(module)
