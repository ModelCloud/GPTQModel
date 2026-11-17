# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""``TorchLinear`` must not keep itself alive through its own attributes.

Two instance attributes used to close a module -> attribute -> module cycle:
``_weight_metadata`` held the module strongly, and ``optimize()`` stored a
``torch.compile`` wrapper around the *bound* ``dequantize_weight``.  A cyclic
garbage collector eventually frees them, but until then a dropped GPU model
keeps its VRAM (#4847).
"""

import gc
import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear


def _module() -> TorchLinear:
    module = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
    )
    module.optimize = lambda *args, **kwargs: None
    module.post_init()
    return module


def _dies_without_gc(build) -> bool:
    gc.collect()
    gc.disable()
    try:
        ref = build()
        return ref() is None
    finally:
        gc.enable()


def test_weight_metadata_does_not_pin_the_module():
    def build() -> weakref.ref:
        module = _module()
        assert module.weight.shape == torch.Size((64, 64))
        return weakref.ref(module)

    assert _dies_without_gc(build)


def test_weight_metadata_survives_transpose_and_reports_the_module():
    module = _module()
    metadata = module.weight
    assert metadata.T.shape == torch.Size((64, 64))
    assert metadata.device == module.qweight.device
    del module
    gc.collect()
    with pytest.raises(RuntimeError, match="released"):
        metadata.device


def test_compiled_dequantize_does_not_pin_the_module(monkeypatch):
    import gptqmodel.nn_modules.qlinear.torch as torch_linear

    monkeypatch.setattr(torch_linear, "torch_compile", lambda fn, **kwargs: fn)

    def build() -> weakref.ref:
        module = _module()
        TorchLinear.optimize(module)
        assert type(module.dequantize_weight).__name__ == "_CompiledDequantizeWeight"
        # The compiled wrapper still resolves the live module.
        assert torch.equal(module.dequantize_weight(), TorchLinear.dequantize_weight(module))
        return weakref.ref(module)

    assert _dies_without_gc(build)


def test_pickle_rebuilds_the_weight_metadata():
    import pickle

    module = _module()
    del module.optimize  # drop the stub so the module is picklable again
    assert module.weight.shape == torch.Size((64, 64))
    clone = pickle.loads(pickle.dumps(module))
    assert clone.weight.shape == torch.Size((64, 64))
    assert clone.weight.device == clone.qweight.device
