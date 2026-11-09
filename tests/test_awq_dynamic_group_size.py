# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import AWQConfig


def _stub(bits=4, dynamic=None):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=True, dynamic=dynamic)
        _module_bits = AWQProcessor._module_bits
        _module_group_size = AWQProcessor._module_group_size
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


def test_module_group_size_follows_the_dynamic_override():
    stub = _stub(dynamic={"model.layers.0.mlp.down_proj": {"group_size": 16}})
    stub.qcfg.group_size = 128

    group_size = stub._module_group_size(
        "model.layers.0.mlp.down_proj"
    )
    weight = torch.randn(8, 128, dtype=torch.float16)
    _, scales, zeros = stub.pseudo_quantize_tensor(
        weight,
        bits=4,
        group_size=group_size,
    )

    assert group_size == 16
    assert scales.shape == (8, 8)
    assert zeros.shape == (8, 8)
