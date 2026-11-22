# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #4387 item 2.

``GGUFCudaKernel`` used to keep a ggml backend buffer per module and per batch
size (6.4 GiB for a 211 MiB model, invisible to the Torch allocator).  Plans now
own graph metadata only and bind their tensors into one lazily grown per-device
arena.

The ggml CUDA backend is not installed in this container, so these tests drive
the real allocation/binding code in ``_GGMLBridge`` through a fake ``ggml_base``
that records backend buffers and tensor bindings.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_cpp
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel, _GGMLBridge


class _FakeTensor:
    def __init__(self, nbytes: int, ne0: int, ne1: int) -> None:
        self.nbytes = nbytes
        self.ne0 = ne0
        self.ne1 = ne1
        self.buffer = None
        self.data = None


class _FakeCtx:
    def __init__(self) -> None:
        self.tensors: list[_FakeTensor] = []


class _FakeBuffer:
    def __init__(self, size: int, base: int) -> None:
        self.size = size
        self.base = base


class _FakeGGML:
    """The subset of ggml-base that plan building and binding touch."""

    def __init__(self) -> None:
        self.buffers: list[_FakeBuffer] = []
        self.freed_buffers: list[_FakeBuffer] = []
        self.tensor_allocs: list[tuple[_FakeBuffer, _FakeTensor, int]] = []
        self._next_base = 0x100000

    def ggml_init(self, params):
        return _FakeCtx()

    def ggml_free(self, ctx):
        pass

    def ggml_new_tensor_2d(self, ctx, ggml_type, ne0, ne1):
        tensor = _FakeTensor(ne0 * ne1 * 4, ne0, ne1)
        ctx.tensors.append(tensor)
        return tensor

    def ggml_set_input(self, tensor):
        pass

    def ggml_mul_mat(self, ctx, weight, x):
        # ggml_mul_mat(weight[in, out], x[in, rows]) -> [out, rows]
        tensor = _FakeTensor(weight.ne1 * x.ne1 * 4, weight.ne1, x.ne1)
        ctx.tensors.append(tensor)
        return tensor

    def ggml_new_graph(self, ctx):
        return object()

    def ggml_build_forward_expand(self, graph, tensor):
        pass

    def ggml_nbytes(self, tensor):
        return tensor.nbytes

    def ggml_element_size(self, tensor):
        return 4

    def ggml_backend_get_default_buffer_type(self, backend):
        return "buft"

    def ggml_backend_buft_get_alignment(self, buft):
        return 64

    def ggml_backend_buft_get_alloc_size(self, buft, tensor):
        return tensor.nbytes

    def ggml_get_first_tensor(self, ctx):
        return ctx.tensors[0] if ctx.tensors else None

    def ggml_get_next_tensor(self, ctx, tensor):
        index = ctx.tensors.index(tensor) + 1
        return ctx.tensors[index] if index < len(ctx.tensors) else None

    def ggml_backend_alloc_buffer(self, backend, size):
        buffer = _FakeBuffer(int(size), self._next_base)
        self._next_base += 1 << 24
        self.buffers.append(buffer)
        return buffer

    def ggml_backend_buffer_get_base(self, buffer):
        return buffer.base

    def ggml_backend_buffer_get_size(self, buffer):
        return buffer.size

    def ggml_backend_buffer_free(self, buffer):
        self.freed_buffers.append(buffer)

    def ggml_backend_tensor_alloc(self, buffer, tensor, addr):
        assert tensor.buffer is None
        assert tensor.data is None
        tensor.buffer = buffer
        tensor.data = addr
        self.tensor_allocs.append((buffer, tensor, addr))
        return 0


def _bridge(fake: _FakeGGML) -> _GGMLBridge:
    bridge = object.__new__(_GGMLBridge)
    bridge._ggml_base = fake
    bridge._cuda_scratch = {}
    return bridge


def _build_plan(bridge, *, rows: int, out_features: int = 64, device_index: int = 0):
    weight = SimpleNamespace(tensor=_FakeTensor(0, 0, 0))
    return bridge.build_quantized_matmul_cuda_plan(
        backend=1,
        weight=weight,
        padded_in_features=128,
        out_features=out_features,
        rows=rows,
        input_ggml_type=0,
        kernel_name="test",
        device_index=device_index,
    )


def test_cuda_plans_share_one_scratch_arena_and_do_not_own_buffers():
    fake = _FakeGGML()
    bridge = _bridge(fake)

    plans = [_build_plan(bridge, rows=rows) for rows in (8, 4, 2, 1)]
    assert len(fake.buffers) == 1
    scratch = bridge._cuda_scratch[0]
    assert scratch.buffer is fake.buffers[0]
    for plan in plans:
        assert plan.scratch is scratch
        assert plan.scratch_generation == scratch.generation
        assert not hasattr(plan, "buffer")
        # Input and output are bound at distinct offsets inside the arena.
        assert plan.tensor_offsets[0] == 0
        assert len(set(plan.tensor_offsets)) == len(plan.tensor_offsets)
    # One binding per tensor, all inside the single shared buffer.
    assert len(fake.tensor_allocs) == sum(len(plan.tensors) for plan in plans)
    assert {buffer for buffer, _, _ in fake.tensor_allocs} == {scratch.buffer}

    for plan in plans:
        plan.close()
    assert fake.freed_buffers == []
    shared_buffer = scratch.buffer
    scratch.close()
    assert fake.freed_buffers == [shared_buffer]


def test_cuda_scratch_grows_once_and_invalidates_stale_plans():
    fake = _FakeGGML()
    bridge = _bridge(fake)

    small = _build_plan(bridge, rows=2)
    first_scratch = bridge._cuda_scratch[0]
    first_buffer = first_scratch.buffer
    first_generation = first_scratch.generation
    assert len(fake.buffers) == 1

    large = _build_plan(bridge, rows=8)
    second_scratch = bridge._cuda_scratch[0]
    # The arena object is per device and grows in place (its generation and
    # backend buffer change).
    assert second_scratch is first_scratch
    assert second_scratch.generation == first_generation + 1
    assert second_scratch.buffer is not first_buffer
    assert fake.freed_buffers == [first_buffer]
    assert len(fake.buffers) == 2
    assert large.scratch is second_scratch
    # The old plan's tensors point into the freed buffer and cannot be rebound.
    with pytest.raises(RuntimeError, match="stale ggml CUDA plan"):
        bridge._bind_cuda_plan(small)

    # A later, smaller plan reuses the grown arena without another allocation.
    again = _build_plan(bridge, rows=1)
    assert again.scratch is second_scratch
    assert len(fake.buffers) == 2

    for plan in (small, large, again):
        plan.close()
    second_scratch.close()


class _FakePlan:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeWeight:
    def close(self) -> None:
        pass

    # The layer's qweight stays on the CPU while the input claims CUDA, so the
    # kernel keeps this weight as a cache and never adopts it as storage.
    device = torch.device("cuda:0")

    def backs(self, tensor) -> bool:
        return False


class _CudaTensor:
    """A CPU tensor that advertises a CUDA device to ``GGUFCudaKernel``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self.device = torch.device("cuda:0")
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        # ``forward`` asks the shared ``input_rows()`` helper for the rank, and
        # the empty path asks for ``requires_grad``; keep the fake faithful so
        # it exercises the same guards as a real tensor.
        self.ndim = tensor.ndim
        self.requires_grad = tensor.requires_grad

    def reshape(self, *shape):
        return _CudaTensor(self._tensor.reshape(*shape))


class _KernelBridge:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.generation = 0
        self.created: list[_FakePlan] = []

    def cuda_backend_lock(self, device_index):
        return self.lock

    def cuda_scratch_generation(self, device_index):
        return self.generation

    def build_cuda_weight(self, **kwargs):
        return _FakeWeight()

    def quantized_matmul_cuda(
        self,
        *,
        x,
        gguf_tensor_qtype,
        padded_in_features,
        out_features,
        plan=None,
        weight=None,
    ):
        if plan is None:
            # Simulate the arena growing while the new plan is built.
            self.generation += 1
            plan = _FakePlan()
            self.created.append(plan)
        return torch.zeros((x.shape[0], out_features), dtype=x.dtype), plan


def _make_layer() -> GGUFCudaKernel:
    layer = object.__new__(GGUFCudaKernel)
    object.__setattr__(layer, "qweight", torch.arange(16, dtype=torch.uint8))
    object.__setattr__(layer, "padded_in_features", 8)
    object.__setattr__(layer, "out_features", 4)
    object.__setattr__(layer, "adapter", None)
    object.__setattr__(layer, "gguf_tensor_qtype", "Q4_K")
    object.__setattr__(layer, "_ggml_cuda_plans", OrderedDict())
    object.__setattr__(layer, "_ggml_cuda_weight", None)
    object.__setattr__(layer, "_ggml_cuda_weight_key", None)
    object.__setattr__(layer, "_ggml_cuda_plan_generation", -1)
    return layer


def test_kernel_closes_plans_when_the_arena_generation_changes(monkeypatch):
    bridge = _KernelBridge()
    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", lambda: bridge)
    monkeypatch.setattr(
        gguf_cpp, "_finish_ggml_output", lambda module, x_flat, output: output
    )
    layer = _make_layer()
    x = _CudaTensor(torch.randn(5, 8))

    try:
        layer.forward(x)  # rows=5, generation 1
        layer.forward(x)  # cached plan, generation still 1
        assert len(layer._ggml_cuda_plans) == 1
        first_plan = next(iter(layer._ggml_cuda_plans.values()))
        assert not first_plan.closed

        layer.forward(_CudaTensor(torch.randn(9, 8)))  # rows=9, generation 2
        assert first_plan.closed
        assert len(layer._ggml_cuda_plans) == 1
        second_plan = next(iter(layer._ggml_cuda_plans.values()))
        assert second_plan is not first_plan
        assert not second_plan.closed
        assert layer._ggml_cuda_plan_generation == 2
    finally:
        object.__setattr__(layer, "_ggml_cuda_weight_key", None)
        layer._ggml_cuda_plans.clear()
