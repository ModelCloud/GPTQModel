# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must stage and pack a module at the width `dynamic` assigns to it.

`AWQConfig` accepts a per-module width (`{"bits": 8}` on one layer of a 4-bit
run). The width was honoured when the quant module was created, but the staging
pass (`pseudo_quantize_tensor`) and the clip search read `qcfg.bits` directly,
so a W8 module staged 4-bit codes and packed them into an 8-bit container.
"""

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import AWQConfig


def _stub(bits=4, dynamic=None):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=True, dynamic=dynamic)
        _module_bits = AWQProcessor._module_bits
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


def test_module_bits_follows_the_dynamic_override():
    stub = _stub(dynamic={"model.layers.0.mlp.down_proj": {"bits": 8}})
    assert stub._module_bits("model.layers.0.mlp.down_proj") == 8
    assert stub._module_bits("model.layers.1.mlp.down_proj") == 4
    assert stub._module_bits(None) == 4


def test_a_non_numeric_override_is_not_width_zero():
    """An excluded module (`bits: false`) or a `format`-only rule is not 2^0."""
    stub = _stub(
        dynamic={
            "-model.layers.0.mlp.up_proj": {"bits": 4},
            "model.layers.1.mlp.up_proj": {"group_size": 64},
        }
    )
    assert stub._module_bits("model.layers.0.mlp.up_proj") == 4
    assert stub._module_bits("model.layers.1.mlp.up_proj") == 4


def test_staging_uses_the_requested_width():
    """W8 staging must reach the 8-bit code range, not the 4-bit one."""
    stub = _stub()
    torch.manual_seed(0)
    weight = torch.randn(16, 64, dtype=torch.float32)

    wq4, scales4, _ = stub.pseudo_quantize_tensor(weight, bits=4)
    wq8, scales8, _ = stub.pseudo_quantize_tensor(weight, bits=8)

    # Codes are recoverable as wq / scale; the widest code has to fill each
    # signed range (7 for 4-bit, 127 for 8-bit).
    codes4 = (wq4 / scales4).round().abs().max().item()
    codes8 = (wq8 / scales8).round().abs().max().item()
    assert codes4 <= 8
    assert codes8 > 8
    assert codes8 <= 128

    # An 8-bit round trip must be far more accurate than a 4-bit one.
    err4 = (wq4 - weight).pow(2).mean().item()
    err8 = (wq8 - weight).pow(2).mean().item()
    assert err8 < err4 / 10


def test_tensor_into_matches_the_allocating_path():
    """The clip search writes into a preallocated buffer; it must agree."""
    stub = _stub()
    torch.manual_seed(1)
    weight = torch.randn(8, 64, dtype=torch.float32)
    for bits in (4, 8):
        expected, _, _ = stub.pseudo_quantize_tensor(weight, bits=bits)
        into = torch.empty_like(weight)
        stub._pseudo_quantize_tensor_into(weight, into, bits=bits)
        assert torch.allclose(into, expected, rtol=1e-6, atol=1e-6), f"bits={bits}"


@pytest.mark.parametrize("bits", [4, 8])
def test_packing_does_not_silently_use_the_global_width(bits):
    """A W8 module in a W4 run must not be packed with W4 codes."""
    stub = _stub(dynamic={"model.layers.0.mlp.down_proj": {"bits": bits}})
    assert stub._module_bits("model.layers.0.mlp.down_proj") == bits
