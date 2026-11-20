# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5811: the shared GGUF CUDA scratch arena is released with its last owner.

``_GGMLBridge`` kept one ggml scratch arena per device for the life of the
process.  Plans, weights and layers were freed on unload, but the arena stayed
reachable from the process singleton, and ggml's allocation is invisible to
``torch.cuda.empty_cache()``.  A weight now registers itself with the bridge
and the arena is freed when the last weight on the device goes away; the arena
object survives so its generation stays monotonic and stale plans rebuild.
"""

from __future__ import annotations

import gc
import threading
import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf_cpp import (
    GGUFCudaKernel,
    _get_ggml_bridge,
    _GGMLBridge,
    _GGMLSharedWeight,
)
from tests.kernels.test_gguf_cuda_scratch_arena_4387 import (
    _FakeGGML,
    _FakeTensor,
)


def _bare_bridge(fake: _FakeGGML) -> _GGMLBridge:
    """A bridge with the recording allocator double and no native library."""

    bridge = object.__new__(_GGMLBridge)
    bridge._ggml_base = fake
    bridge._cuda_scratch = {}
    bridge._cuda_weights = {}
    bridge._cuda_backend_lock = threading.RLock()
    return bridge


def _tracked_weight(
    bridge: _GGMLBridge,
    fake: _FakeGGML,
    *,
    device_index: int = 0,
    padded_in_features: int = 128,
    out_features: int = 64,
) -> _GGMLSharedWeight:
    """A real ``_GGMLSharedWeight`` over the recording allocator double.

    The ownership hook is looked up instead of dereferenced so this regression
    also runs against a pre-#5811 tree, where it fails on the arena assertion
    rather than on an attribute error.
    """

    weight_kwargs = {
        "ctx": object(),
        "buffer": fake.ggml_backend_alloc_buffer(1, 4096),
        "tensor": _FakeTensor(0, padded_in_features, out_features),
        "nbytes": 4096,
        "backend_buffer_free": fake.ggml_backend_buffer_free,
        "ctx_free": lambda ctx: None,
        "data_ptr": 0,
        "shape": (out_features, padded_in_features),
        "device": torch.device(f"cuda:{device_index}"),
        "lock": bridge.cuda_backend_lock(device_index),
    }
    # A pre-#5811 bridge has neither the release hook nor the tracker. Omit
    # the hook so the regression still reaches its arena assertion on that
    # tree and fails there, instead of erroring on a new keyword.
    on_release = getattr(bridge, "_on_cuda_weight_released", None)
    if on_release is not None:
        weight_kwargs["on_release"] = on_release
    weight = _GGMLSharedWeight(**weight_kwargs)
    track = getattr(bridge, "_track_cuda_weight", None)
    if track is not None:
        track(weight)
    return weight


def _build_plan(
    bridge: _GGMLBridge,
    weight: _GGMLSharedWeight,
    *,
    rows: int,
    device_index: int = 0,
):
    return bridge.build_quantized_matmul_cuda_plan(
        backend=1,
        weight=weight,
        padded_in_features=128,
        out_features=64,
        rows=rows,
        input_ggml_type=0,
        kernel_name="arena-release",
        device_index=device_index,
    )


def test_arena_is_released_when_the_last_weight_is_unloaded():
    fake = _FakeGGML()
    bridge = _bare_bridge(fake)
    weight = _tracked_weight(bridge, fake)
    weight_buffer = weight.buffer

    plan = _build_plan(bridge, weight, rows=8)
    scratch = bridge._cuda_scratch[0]
    arena_buffer = scratch.buffer
    arena_bytes = scratch.size
    generation = scratch.generation
    assert arena_bytes > 0

    # A second plan of the same device reuses the arena; the live weight keeps
    # it reserved even when the first plan is closed.
    second = _build_plan(bridge, weight, rows=4)
    assert second.scratch is scratch
    assert len(fake.buffers) == 2  # weight buffer + one arena
    plan.close()
    second.close()
    assert fake.freed_buffers == []
    assert scratch.size == arena_bytes

    # Dropping the plan is not enough: it holds a weight, and the test still
    # does too. The arena stays reserved while any owner is alive.
    stale_plan = plan
    del second
    gc.collect()
    assert scratch.size == arena_bytes

    weight_ref = weakref.ref(weight)
    del weight
    gc.collect()
    assert weight_ref() is None
    # The weight's buffer goes first, then the idle arena.
    assert fake.freed_buffers == [weight_buffer, arena_buffer]
    assert scratch.buffer is None
    assert scratch.size == 0
    assert scratch.generation == generation + 1

    # A plan bound to the freed buffer must never run again.
    with pytest.raises(RuntimeError, match="stale ggml CUDA plan"):
        bridge._bind_cuda_plan(stale_plan)

    # The next load re-allocates the arena instead of reusing the freed
    # buffer, and the generation keeps increasing (no generation reuse).
    reloaded = _tracked_weight(bridge, fake)
    reloaded_plan = _build_plan(bridge, reloaded, rows=8)
    assert reloaded_plan.scratch is scratch
    assert scratch.buffer is not None
    assert scratch.size == arena_bytes
    assert scratch.generation > generation + 1
    # One weight buffer and one arena per load: reclamation bounds re-growth.
    assert len(fake.buffers) == 4

    reloaded_plan.close()
    del reloaded_plan, reloaded, stale_plan
    gc.collect()
    assert scratch.size == 0


def test_arena_stays_reserved_until_the_last_of_two_owners_is_gone():
    fake = _FakeGGML()
    bridge = _bare_bridge(fake)
    first = _tracked_weight(bridge, fake)
    second = _tracked_weight(bridge, fake)
    # Each plan holds its weight, exactly as a live layer's cached plan does.
    first_plan = _build_plan(bridge, first, rows=8)
    second_plan = _build_plan(bridge, second, rows=4)
    scratch = bridge._cuda_scratch[0]
    arena_bytes = scratch.size
    assert arena_bytes > 0

    first_plan.close()
    del first_plan, first
    gc.collect()
    # The second layer's plan and weight still own the arena.
    assert scratch.size == arena_bytes
    assert scratch.buffer is not None

    second_plan.close()
    del second_plan, second
    gc.collect()
    assert scratch.size == 0
    assert scratch.buffer is None


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    GGUFCudaKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python CUDA unavailable: {err}")


def _load_layer(state: dict[str, torch.Tensor]) -> GGUFCudaKernel:
    from tests.kernels.test_gguf_cuda_weight_storage_4387 import _cuda_kernel

    return _cuda_kernel(state, bias=False)


def _forward(layer: GGUFCudaKernel, rows: int) -> torch.Tensor:
    from tests.kernels.test_gguf_cuda_weight_storage_4387 import _forward as run
    from tests.kernels.test_gguf_cuda_weight_storage_4387 import _input

    return run(layer, _input(rows=rows).float())


@pytest.mark.cuda
def test_arena_reserved_bytes_return_to_zero_between_two_loads():
    """Load, unload, load, unload in one process; measure the device too."""

    from tests.kernels.test_gguf_cuda_weight_storage_4387 import _packed_state

    _require_cuda()
    bridge = _get_ggml_bridge()
    device_index = torch.cuda.current_device()
    state = _packed_state(0, bias=False)
    rows = 1024

    first = _load_layer(state)
    out = _forward(first, rows)
    assert out.shape == (rows, first.out_features)
    assert torch.isfinite(out).all()
    scratch = bridge._cuda_scratch[device_index]
    reserved = scratch.size
    generation = scratch.generation
    assert reserved > 0
    del out

    # An explicit cache clear on a live model must not drop the arena: the
    # zero-copy qweight still owns the weight, so the next forward reuses the
    # same buffer and generation instead of re-allocating.
    first.clear_weight_cache()
    assert scratch.size == reserved
    assert scratch.generation == generation
    out = _forward(first, rows)
    assert scratch.size == reserved
    assert scratch.generation == generation
    del out

    # Measure with the arena live. The first forward also initializes the
    # ggml CUDA backend, so a pre-forward baseline would include that one-time
    # allocation instead of the arena.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_with_arena, _ = torch.cuda.mem_get_info()

    del first
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_after_unload, _ = torch.cuda.mem_get_info()
    assert scratch.size == 0, "the arena stayed reserved after the last unload"
    assert scratch.buffer is None
    assert scratch.generation == generation + 1
    # ggml's allocation is outside the Torch allocator, so the driver free
    # memory must grow by at least the arena bytes once it is released.
    returned_mib = (free_after_unload - free_with_arena) / 2**20
    reserved_mib = reserved / 2**20
    assert free_after_unload - free_with_arena >= reserved, (
        f"unload returned {returned_mib:.1f} MiB to the driver, less than the "
        f"{reserved_mib:.1f} MiB arena"
    )

    second = _load_layer(state)
    out = _forward(second, rows)
    assert out.shape == (rows, second.out_features)
    assert torch.isfinite(out).all()
    assert bridge._cuda_scratch[device_index] is scratch
    assert scratch.size >= reserved, "the next load did not re-allocate the arena"
    assert scratch.generation > generation + 1
    del out

    del second
    gc.collect()
    torch.cuda.synchronize()
    assert scratch.size == 0
    assert scratch.buffer is None
