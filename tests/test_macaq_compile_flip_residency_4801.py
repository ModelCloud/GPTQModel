# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4801: a compiled flip sweep must release its working set deterministically.

``flip_refine_levels(compile_flip=True)`` compiles ``_flip_sweep``. A Dynamo
graph break makes the resume frames (which hold ``signed_index``, the Hessian
operands and the masks) part of an unreachable reference cycle, so the whole
working set survives until Python's cyclic GC runs: 0.4-1.4 GiB left on a
Llama-3.1-8B after ``quantize()`` and +1.25 GiB quantize peak. The sweep is
written without data-dependent control flow and compiled with
``fullgraph=True``, so it traces into one graph and pins nothing.

These tests fail if a graph break is reintroduced. They never call
``gc.collect()`` before the assertion: that is the whole contract.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from gptqmodel.quantization import macaq


def _working_set(device: torch.device, out_features: int = 256, in_features: int = 256):
    """Build one Linear's flip operands, mirroring ``macaq_quantize_weight``."""

    generator = torch.Generator(device=device).manual_seed(0)
    activations = torch.randn(
        4 * in_features, in_features, generator=generator, device=device
    )
    hessian = activations.T @ activations
    hessian.diagonal().add_(hessian.diagonal().mean())
    weight = torch.randn(
        out_features, in_features, generator=generator, device=device
    ) * 0.02
    signed_index, _, scale_element = macaq.mxfp4_rtn_signed_index(
        weight, macaq.MXFP4_BLOCK
    )
    return signed_index, scale_element, weight, hessian


def _run_compiled_refinement(device: torch.device):
    """Run one compiled refinement and return weakrefs to its operands.

    The compiled callable is left cached in ``macaq._COMPILED_FLIP_SWEEP``
    (production behavior) while the caller asserts, so a frame cycle rooted in
    the compiled wrapper stays reachable from the module and cannot be freed by
    simply dropping the local inputs.
    """

    signed_index, scale_element, weight, hessian = _working_set(device)
    refs = tuple(
        weakref.ref(tensor)
        for tensor in (signed_index, scale_element, weight, hessian)
    )
    original = macaq._COMPILED_FLIP_SWEEP
    macaq._COMPILED_FLIP_SWEEP = None
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            # Exercise the compiled sweep on every host, including a CPU-only
            # CI box where the GIL/device gate would otherwise pick eager.
            monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda _device: True)
            macaq.flip_refine_levels(
                signed_index,
                scale_element,
                weight,
                hessian,
                4,
                fast=True,
                compile_flip=True,
            )
        assert macaq._COMPILED_FLIP_SWEEP is not None
    except BaseException:
        macaq._COMPILED_FLIP_SWEEP = original
        raise
    # Drop the last direct references; only a leaked frame may keep them alive.
    del signed_index, scale_element, weight, hessian
    return refs, original


def test_compiled_flip_sweep_releases_working_set_without_gc():
    """CPU guard: the operands die with the call, no cyclic GC needed.

    Automatic cyclic collection is disabled so a background GC pass cannot
    mask the leak: with the graph break, these weakrefs stay alive until an
    explicit ``gc.collect()``.
    """

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        refs, original = _run_compiled_refinement(torch.device("cpu"))
        try:
            assert [ref() for ref in refs] == [None] * len(refs)
        finally:
            macaq._COMPILED_FLIP_SWEEP = original
    finally:
        if gc_was_enabled:
            gc.enable()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA residency check")
def test_compiled_flip_sweep_leaves_no_cuda_bytes_without_gc():
    """CUDA guard: ``memory_allocated()`` returns to baseline, no gc.collect()."""

    if not macaq._compiled_flip_supported(torch.device("cuda:0")):
        pytest.skip("compiled flips are not qualified on this GPU/runtime")

    torch.cuda.synchronize()
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    original = macaq._COMPILED_FLIP_SWEEP
    macaq._COMPILED_FLIP_SWEEP = None
    baseline = torch.cuda.memory_allocated()
    try:
        signed_index, scale_element, weight, hessian = _working_set(
            torch.device("cuda:0")
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(macaq, "_compiled_flip_supported", lambda _device: True)
            macaq.flip_refine_levels(
                signed_index,
                scale_element,
                weight,
                hessian,
                4,
                fast=True,
                compile_flip=True,
            )
        assert macaq._COMPILED_FLIP_SWEEP is not None
        del signed_index, scale_element, weight, hessian
        torch.cuda.synchronize()
        # No `gc.collect()`: the graph-break leak leaves ~3 MiB for this
        # working set, so a sub-MiB residue is the "released" contract.
        assert torch.cuda.memory_allocated() - baseline < 2**20
    finally:
        macaq._COMPILED_FLIP_SWEEP = original
        if gc_was_enabled:
            gc.enable()
        gc.collect()
