# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4801.

`macaq_quantize_weight(..., compile_flip=True)` runs the flip sweep through
`torch.compile`. Each compiled call leaves a cycle of dead frames (Dynamo
resume frames, the wrapper, `_flip_sweep`) that still holds the sweep's
intermediate tensors, so a run kept 0.4-1.4 GiB resident per Linear after
`quantize()` and peaked higher until Python's cyclic GC happened to run.

The cycle is only collectible once the sweep's frames have returned, so the
collection runs at the call site. `MacaqProcessor.finalize()` collects once
more so the #4747/#4757 "quantize() returns with nothing resident" contract
holds even if something else in the processor keeps a frame alive.
"""

import gc

import torch

from gptqmodel.quantization import macaq
from gptqmodel.quantization.macaq import macaq_quantize_weight


def _weight_and_hessian(out_features: int, in_features: int, device: torch.device, seed: int = 0):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(4 * in_features, in_features, device=device, generator=generator)
    hessian = x.T @ x / x.shape[0] + 1e-2 * torch.eye(in_features, device=device)
    weight = torch.randn(out_features, in_features, device=device, generator=generator) * 0.02
    return weight, hessian, float(x.shape[0])


def _count_collects(monkeypatch):
    collected = []
    monkeypatch.setattr(gc, "collect", lambda: collected.append(1))
    return collected


def test_compiled_flip_collects_once_per_weight(monkeypatch):
    collected = _count_collects(monkeypatch)
    monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda device: True)
    monkeypatch.setattr(macaq, "_compiled_flip_sweep", macaq._flip_sweep)

    weight, hessian, n_tokens = _weight_and_hessian(64, 64, torch.device("cpu"))
    macaq_quantize_weight(
        weight, hessian, n_tokens, n_flip_iters=2, compile_flip=True
    )

    assert collected


def test_eager_flip_does_not_collect(monkeypatch):
    collected = _count_collects(monkeypatch)
    monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda device: True)
    monkeypatch.setattr(macaq, "_compiled_flip_sweep", macaq._flip_sweep)

    weight, hessian, n_tokens = _weight_and_hessian(64, 64, torch.device("cpu"))
    macaq_quantize_weight(
        weight, hessian, n_tokens, n_flip_iters=2, compile_flip=False
    )

    assert collected == []


def test_unsupported_device_does_not_collect(monkeypatch):
    collected = _count_collects(monkeypatch)
    # The compiled sweep is only qualified on NVIDIA CUDA with the GIL and on
    # ROCm gfx1201; elsewhere the eager sweep runs and nothing to collect.
    monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda device: False)

    weight, hessian, n_tokens = _weight_and_hessian(64, 64, torch.device("cpu"))
    macaq_quantize_weight(
        weight, hessian, n_tokens, n_flip_iters=2, compile_flip=True
    )

    assert collected == []
