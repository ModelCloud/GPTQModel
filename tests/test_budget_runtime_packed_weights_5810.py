# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""``model.budget()`` has to count weights a kernel keeps off the registry (#5810).

``_parameter_devices()`` walked only ``parameters()``/``buffers()``. A QQQ
projection replaces ``B``/``s_group``/``s_channel`` with ``_torch_packed_cache``
on its first native CPU forward, so a 1024 x 1024 layer reported its 1,024-byte
workspace buffer while 543,744 bytes of packed weight stayed resident. GGUF
Triton keeps derived scale planes in a plain cache dict, with the same blind
spot.

``BaseQuantLinear.iter_runtime_weight_tensors()`` lets a module publish those
tensors. The counter merges them with the registered ones by storage, so a view
of a registered buffer, or two views of one cache tensor, is counted once.
"""

from __future__ import annotations

import types

import torch
from torch import nn

from gptqmodel.models.base import BaseQModel, _parameter_devices
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.utils.model import _restore_checkpoint_layout_for_export


WIDTH = 1024
GROUP_SIZE = 128
DTYPE = torch.float16


def _budget_bytes(module) -> int:
    """Measured weight bytes the public budget API reports for this module."""

    return BaseQModel.budget(types.SimpleNamespace(model=module)).weights_bytes


def _owned_bytes(module, *runtime_tensors) -> int:
    """Resident bytes the module owns, counting each storage once.

    The oracle is independent of the hook under test: the caller names the
    runtime tensors, as the issue's reproduction does. A view contributes the
    storage it shares, once.
    """

    owned = {}

    def add(tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            return
        storage = tensor.untyped_storage()
        owned[(tensor.device.index, storage.data_ptr())] = storage.nbytes()

    for tensor in (*module.parameters(), *module.buffers()):
        add(tensor)
    for tensor in runtime_tensors:
        add(tensor)
    return sum(owned.values())


def _build_qqq() -> QQQTorchLinear:
    from tests.test_qqq_kernel_memory_4387 import _build_torch_module

    return _build_torch_module(
        in_features=WIDTH, out_features=WIDTH, group_size=GROUP_SIZE, dtype=DTYPE
    )


def _dropped_qqq() -> QQQTorchLinear:
    """The state the first native CPU forward leaves behind.

    That forward calls ``_packed_weight_for_cpu()``, which builds the cache and
    then drops the checkpoint buffers. It is called directly because this host
    is ARM64 and has no AVX512 packed kernel, so a plain forward takes the
    generic matmul path and never drops anything.
    """

    module = _build_qqq()
    module._packed_weight_for_cpu()
    assert module.B is None, "the fixture did not drop the checkpoint buffers"
    assert module.s_channel is None
    assert module._torch_packed_cache is not None
    return module


def _workspace_bytes(module) -> int:
    workspace = module.workspace
    return workspace.numel() * workspace.element_size()


# ---------------------------------------------------------------------------
# QQQ: the checkpoint layout and the packed cache are one weight in two shapes.
# ---------------------------------------------------------------------------

def test_budget_counts_the_checkpoint_layout_before_the_first_forward():
    module = _build_qqq()

    checkpoint_bytes = sum(
        tensor.untyped_storage().nbytes() for tensor in module.buffers() if tensor is not None
    )

    assert _budget_bytes(module) == checkpoint_bytes == _owned_bytes(module)


def test_budget_keeps_counting_the_weights_after_the_cache_replaces_them():
    module = _build_qqq()
    before = _budget_bytes(module)

    module._packed_weight_for_cpu()

    assert module.B is None
    cache_bytes = sum(
        tensor.untyped_storage().nbytes() for tensor in module._torch_packed_cache
    )
    # The layout swap is byte-neutral: the cache holds the same weight, so the
    # report must not dip to the workspace buffer that survived the drop.
    assert _budget_bytes(module) == before
    assert _budget_bytes(module) == _owned_bytes(module, *module._torch_packed_cache)
    assert _budget_bytes(module) == cache_bytes + _workspace_bytes(module)


def test_budget_after_an_export_counts_one_layout():
    module = _dropped_qqq()
    one_layout = _budget_bytes(module)
    model = nn.Module()
    model.layer = module

    assert _restore_checkpoint_layout_for_export(model) == 1

    assert module._torch_packed_cache is None
    # The export publishes the checkpoint buffers and releases the cache: the
    # same weight, once, not both layouts.
    assert _budget_bytes(module) == one_layout
    assert _budget_bytes(module) == _owned_bytes(module)


def test_budget_after_a_cache_clear_counts_the_restored_layout():
    module = _dropped_qqq()
    one_layout = _budget_bytes(module)

    module.clear_weight_cache()

    assert module._torch_packed_cache is None
    assert module.B is not None
    assert _budget_bytes(module) == one_layout
    assert _budget_bytes(module) == _owned_bytes(module)


def test_budget_finds_a_backend_cache_below_the_module_root():
    module = _dropped_qqq()
    parent = nn.Module()
    parent.block = nn.Module()
    parent.block.layer = module

    assert _budget_bytes(parent) == _owned_bytes(parent, *module._torch_packed_cache)


# ---------------------------------------------------------------------------
# Storage accounting: aliases add nothing, distinct allocations add up.
# ---------------------------------------------------------------------------

class _CacheModule(nn.Module):
    """A backend stand-in that publishes tensors outside the registry."""

    def __init__(self):
        super().__init__()
        self.register_buffer("packed", torch.zeros(64, 32, dtype=torch.int32))
        self.cache = ()

    def iter_runtime_weight_tensors(self):
        yield from self.cache


def _registered_bytes(module) -> int:
    return module.packed.numel() * module.packed.element_size()


def test_a_view_of_a_registered_buffer_adds_nothing():
    module = _CacheModule()
    module.cache = (module.packed.view(torch.int8),)

    assert _budget_bytes(module) == _registered_bytes(module)


def test_two_views_of_one_cache_tensor_count_its_storage_once():
    module = _CacheModule()
    source = torch.zeros(16, 16, dtype=torch.float16)
    module.cache = (source, source.t(), source[:, :8])

    assert _budget_bytes(module) == _registered_bytes(module) + source.numel() * 2


def test_distinct_cache_tensors_are_all_counted():
    module = _CacheModule()
    half = torch.zeros(8, dtype=torch.float16)
    single = torch.zeros(4, dtype=torch.float32)
    module.cache = (half, single)

    assert _budget_bytes(module) == _registered_bytes(module) + 16 + 16


def test_a_module_without_the_hook_reports_only_its_registered_tensors():
    module = nn.Module()
    module.register_buffer("weight", torch.zeros(3, 5, dtype=torch.float16))

    assert _budget_bytes(module) == 30


def test_the_default_hook_publishes_nothing():
    assert tuple(BaseQuantLinear.iter_runtime_weight_tensors(object())) == ()


def test_a_failing_hook_does_not_break_the_budget():
    """A backend that cannot enumerate its cache must not fail the report."""

    class _Unavailable(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.zeros(4, dtype=torch.float16))

        def iter_runtime_weight_tensors(self):
            raise RuntimeError("backend bookkeeping unavailable")

    assert _budget_bytes(_Unavailable()) == 8


class _FakeStorage:
    def __init__(self, pointer: int, nbytes: int):
        self._pointer = pointer
        self._nbytes = nbytes

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._nbytes


class _DeviceTensor:
    """A tensor stand-in that reports a non-host device and its storage."""

    is_meta = False

    def __init__(self, index: int, pointer: int, nbytes: int):
        self.device = types.SimpleNamespace(type="cuda", index=index)
        self._storage = _FakeStorage(pointer, nbytes)

    def untyped_storage(self):
        return self._storage


def test_a_cache_tensor_is_attributed_to_its_own_device():
    """Accelerator storage is reported per device, not folded into the host."""

    class _DeviceCache(nn.Module):
        def iter_runtime_weight_tensors(self):
            yield _DeviceTensor(1, 0x1000, 4096)

    assert _parameter_devices(_DeviceCache()) == {1: 4096}


# ---------------------------------------------------------------------------
# GGUF Triton: the fused kernels read scale planes held in a plain cache dict.
# ---------------------------------------------------------------------------

def _gguf_triton_module() -> GGUFTritonKernel:
    """A Triton kernel instance without constructing one.

    ``GGUFTritonKernel.__init__`` refuses to run without ``triton``, which this
    host does not install, but the cache hook only reads ``_gguf_triton_cache``.
    """

    module = object.__new__(GGUFTritonKernel)
    nn.Module.__init__(module)
    module.register_buffer("qweight", torch.zeros(64, 32, dtype=torch.int32))
    return module


def test_gguf_triton_publishes_its_cached_metadata_tensors():
    module = _gguf_triton_module()
    packed_view = module.qweight.view(torch.int8)
    scale = torch.zeros(8, 64, dtype=torch.float16)
    module._gguf_triton_cache = {
        0: {
            "qs": packed_view,
            "scale": scale,
            "use_u32": True,
            "fixed_decode_config": None,
        },
    }

    published = tuple(module.iter_runtime_weight_tensors())

    assert len(published) == 2
    assert published[0] is packed_view
    assert published[1] is scale


def test_gguf_triton_budget_counts_the_derived_planes_once():
    module = _gguf_triton_module()
    scale = torch.zeros(8, 64, dtype=torch.float16)
    module._gguf_triton_cache = {0: {"qs": module.qweight.view(torch.int8), "scale": scale}}

    packed_bytes = module.qweight.numel() * module.qweight.element_size()
    # The code bytes are a strided view of ``qweight``; only the scale plane is
    # a new allocation, so it is the only thing the hook adds.
    assert _budget_bytes(module) == packed_bytes + scale.numel() * scale.element_size()


def test_gguf_triton_budget_drops_a_cleared_cache():
    module = _gguf_triton_module()
    module._gguf_triton_cache = {0: {"scale": torch.zeros(8, 64, dtype=torch.float16)}}
    assert _budget_bytes(module) > module.qweight.numel() * 4

    module._gguf_triton_cache.clear()

    assert _budget_bytes(module) == module.qweight.numel() * 4
