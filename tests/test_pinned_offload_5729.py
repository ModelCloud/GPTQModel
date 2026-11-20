# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Layer offload must not pin the parameters it installs (#5729).

``move_module_to_cpu_pinned`` used to allocate the *destination* of every large
device-to-host copy with ``pin_memory=True`` and install that tensor as the
module's parameter. A dense module that is never packed (a dynamic exclusion, a
pruned expert, or any other unquantized weight) therefore kept its weights in
non-swappable pinned host memory for the rest of the process, and the pinned
allocator rounded every live block up to a power of two.

The copy now stages each chunk through one reused pinned buffer and writes it
into pageable CPU storage. These tests exercise the real staging branch with
CPU tensors (the branch itself is device-agnostic) and check that the installed
parameter is pageable, bit-exact, and that the pinned footprint stays bounded.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils import model as model_utils
from gptqmodel.utils.model import move_module_to_cpu_pinned


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _stage_cpu_tensors(monkeypatch) -> None:
    """Route CPU tensors through the pinned staging branch.

    Production stages CUDA tensors only, and this host has no CUDA. The branch
    itself does not depend on the device type, so marking CPU as staged
    exercises the real allocation, chunking and copy code.
    """

    monkeypatch.setattr(
        model_utils, "_PINNED_STAGING_DEVICE_TYPES", frozenset({"cpu"})
    )


def test_large_offloaded_parameter_is_pageable(monkeypatch):
    _stage_cpu_tensors(monkeypatch)
    module = nn.Module()
    module.proj = nn.Linear(4096, 4096, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        module.proj.weight.copy_(torch.randn_like(module.proj.weight))
    expected = module.proj.weight.detach().clone()

    assert move_module_to_cpu_pinned(module) is module

    assert module.proj.weight.device.type == "cpu"
    assert not module.proj.weight.is_pinned()
    assert _bitwise_equal(module.proj.weight, expected)


def test_staged_copy_keeps_every_chunk_and_the_tail(monkeypatch):
    _stage_cpu_tensors(monkeypatch)
    values = torch.arange(4099, dtype=torch.bfloat16).reshape(1, 4099)
    module = nn.Module()
    module.register_buffer("weight", values.clone())
    expected = values.clone()

    # 4096 bytes is 2048 bf16 elements, so 4099 elements span three chunks and
    # the last chunk is a three-element tail.
    move_module_to_cpu_pinned(module, chunk_bytes=4096)

    assert not module.weight.is_pinned()
    assert _bitwise_equal(module.weight, expected)


def test_staging_buffer_is_reused_across_tensors(monkeypatch):
    _stage_cpu_tensors(monkeypatch)
    pinned_allocations = []
    real_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            pinned_allocations.append(args[0] if args else None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(model_utils.torch, "empty", tracking_empty)

    module = nn.Module()
    module.left = nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
    module.right = nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)

    move_module_to_cpu_pinned(module)

    assert len(pinned_allocations) == 1
    assert not module.left.weight.is_pinned()
    assert not module.right.weight.is_pinned()


def test_unavailable_pinned_memory_falls_back_to_pageable(monkeypatch):
    _stage_cpu_tensors(monkeypatch)
    real_empty = torch.empty

    def failing_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("pinned memory is unavailable")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(model_utils.torch, "empty", failing_empty)

    module = nn.Module()
    module.proj = nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
    expected = module.proj.weight.detach().clone()

    move_module_to_cpu_pinned(module)

    assert not module.proj.weight.is_pinned()
    assert _bitwise_equal(module.proj.weight, expected)


def test_small_tensors_bypass_pinned_staging(monkeypatch):
    _stage_cpu_tensors(monkeypatch)
    pinned_allocations = []
    real_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            pinned_allocations.append(args[0] if args else None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(model_utils.torch, "empty", tracking_empty)

    module = nn.Module()
    module.small = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)

    move_module_to_cpu_pinned(module)

    assert pinned_allocations == []
    assert not module.small.weight.is_pinned()


def test_empty_tensor_does_not_allocate_staging(monkeypatch):
    _stage_cpu_tensors(monkeypatch)
    pinned_allocations = []
    real_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            pinned_allocations.append(args[0] if args else None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(model_utils.torch, "empty", tracking_empty)

    module = nn.Module()
    module.register_buffer("empty", torch.empty(0, dtype=torch.bfloat16))

    move_module_to_cpu_pinned(module)

    assert pinned_allocations == []
    assert module.empty.numel() == 0


def test_post_quantize_keeps_w4a_chunks_small(monkeypatch):
    calls = []

    def fake_move(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(
        "gptqmodel.models.base.move_module_to_cpu_pinned", fake_move
    )
    model = SimpleNamespace(
        quantize_config=SimpleNamespace(activation_mode="w4afp8")
    )
    module = nn.Module()

    assert BaseQModel.post_quantize(model, module) is module
    assert calls == [{"chunk_bytes": 1024 * 1024}]

    calls.clear()
    model.quantize_config.activation_mode = "gptq"
    assert BaseQModel.post_quantize(model, module) is module
    assert calls == [{}]
