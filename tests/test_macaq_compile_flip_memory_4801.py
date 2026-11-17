# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4801 / #4900.

#4801: a compiled flip sweep used to leave Dynamo resume frames holding the
sweep's working set, so a run kept 0.4-1.4 GiB resident until a cyclic GC.
#4833 fixed that with a full ``gc.collect()`` per Linear; #4854 then removed
the graph break, so the per-Linear collection only cost time (#4900: MACaQ
quantize 1.3-4x slower on 0.5B-1B models).

The hot path therefore no longer collects. ``MacaqProcessor.finalize()`` keeps
one full collection as the backstop that holds the #4747/#4757 "``quantize()``
returns with nothing resident" contract. ``test_macaq_compile_flip_residency_4801``
pins the no-residue contract itself: the sweep's operands die with no
``gc.collect()``.
"""

import gc
from types import SimpleNamespace

import torch

from gptqmodel.quantization import macaq
from gptqmodel.quantization.macaq import macaq_quantize_weight


def _weight_and_hessian(out_features: int, in_features: int, device: torch.device, seed: int = 0):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(4 * in_features, in_features, device=device, generator=generator)
    hessian = x.T @ x / x.shape[0] + 1e-2 * torch.eye(in_features, device=device)
    weight = torch.randn(out_features, in_features, device=device, generator=generator) * 0.02
    return weight, hessian, float(x.shape[0])


def _record_collects(monkeypatch):
    """Record every ``gc.collect(...)`` call, with its arguments."""

    collected = []
    monkeypatch.setattr(gc, "collect", lambda *args: collected.append(args))
    return collected


def test_compiled_flip_does_not_collect_per_weight(monkeypatch):
    """#4900: no full (or generation-limited) collection in the hot path."""

    collected = _record_collects(monkeypatch)
    monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda device: True)
    monkeypatch.setattr(macaq, "_compiled_flip_sweep", macaq._flip_sweep)

    weight, hessian, n_tokens = _weight_and_hessian(64, 64, torch.device("cpu"))
    macaq_quantize_weight(
        weight, hessian, n_tokens, n_flip_iters=2, compile_flip=True
    )

    assert collected == []


def test_eager_flip_does_not_collect(monkeypatch):
    collected = _record_collects(monkeypatch)
    monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda device: True)
    monkeypatch.setattr(macaq, "_compiled_flip_sweep", macaq._flip_sweep)

    weight, hessian, n_tokens = _weight_and_hessian(64, 64, torch.device("cpu"))
    macaq_quantize_weight(
        weight, hessian, n_tokens, n_flip_iters=2, compile_flip=False
    )

    assert collected == []


def test_unsupported_device_does_not_collect(monkeypatch):
    collected = _record_collects(monkeypatch)
    # The compiled sweep is only qualified on NVIDIA CUDA with the GIL and on
    # ROCm gfx1201; elsewhere the eager sweep runs and nothing to collect.
    monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda device: False)

    weight, hessian, n_tokens = _weight_and_hessian(64, 64, torch.device("cpu"))
    macaq_quantize_weight(
        weight, hessian, n_tokens, n_flip_iters=2, compile_flip=True
    )

    assert collected == []


def test_finalize_keeps_the_collection_backstop(monkeypatch):
    """Removing the finalize collection must fail this test (#4801 contract)."""

    from gptqmodel.looper import macaq_processor
    from gptqmodel.looper.loop_processor import LoopProcessor

    # The base finalize releases process-wide resources; the backstop is the
    # only thing this test is about.
    monkeypatch.setattr(LoopProcessor, "finalize", lambda self, model, **kwargs: None)
    collected = _record_collects(monkeypatch)

    model = SimpleNamespace(
        quantized=False,
        quantize_config=SimpleNamespace(
            method=None,
            format=None,
            checkpoint_format_for_settings=lambda: "mxfp4",
        ),
    )
    processor = object.__new__(macaq_processor.MacaqProcessor)
    processor.finalize(model)

    assert collected, "MacaqProcessor.finalize() lost the gc backstop"
    assert model.quantized is True
