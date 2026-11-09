# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import AWQConfig


def _stub(bits=4, dynamic=None):
    class Stub:
        qcfg = AWQConfig(bits=bits, group_size=-1, sym=True, dynamic=dynamic)
        pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
        _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into

    return Stub()


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_asymmetric_one_sided_group_preserves_constant(value):
    stub = _stub()
    stub.qcfg.group_size = 16
    stub.qcfg.sym = False

    actual, _, _ = stub.pseudo_quantize_tensor(
        torch.full((8, 16), value, dtype=torch.float16)
    )

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, value),
        rtol=0,
        atol=torch.finfo(torch.float16).eps,
    )

def test_asymmetric_finite_fp16_extrema_do_not_overflow():
    stub = _stub()
    stub.qcfg.group_size = 16
    stub.qcfg.sym = False
    weight = torch.tensor(
        [-65504.0, 65504.0] * 8,
        dtype=torch.float16,
    ).repeat(8, 1)

    actual, scales, _ = stub.pseudo_quantize_tensor(weight)

    assert actual.isfinite().all()
    assert scales.isfinite().all()
    assert actual.min().item() >= torch.finfo(torch.float16).min
    assert actual.max().item() <= torch.finfo(torch.float16).max
