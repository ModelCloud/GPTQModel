# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hadamard transforms must accept a zero-element operand.

Every quant-linear backend applies the online rotation before it checks for an
empty batch. The native extension used to launch a zero-sized grid
(``CUDA error: invalid argument``) and the Torch butterfly used to infer a
``-1`` extent on a zero-element tensor (``RuntimeError``), so a decode/prefill
step with no rows crashed instead of returning an empty tensor.
"""

import pytest
import torch

from gptqmodel.quantization.rotation import hadamard_utils as had


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(
    "op_name",
    [
        "fast_hadamard_transform",
        "fast_hadamard_transform_12N",
        "fast_hadamard_transform_20N",
        "fast_hadamard_transform_28N",
        "fast_hadamard_transform_40N",
    ],
)
def test_native_hadamard_accepts_an_empty_batch(op_name):
    from gptqmodel.utils.hadamard import hadamard_op

    op = hadamard_op(op_name)
    x = torch.empty((0, 128), device="cuda", dtype=torch.float16)

    y = op(x, 1.0)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


@requires_cuda
@pytest.mark.cuda
def test_native_hadamard_keeps_the_leading_empty_extent():
    from gptqmodel.utils.hadamard import hadamard_transform

    x = torch.empty((3, 0, 256), device="cuda", dtype=torch.float32)

    y = hadamard_transform(x, 1.0)

    assert y.shape == x.shape


@pytest.mark.parametrize("shape", [(0, 256), (2, 0, 256)])
def test_torch_fallback_accepts_a_zero_element_operand(shape):
    x = torch.empty(shape, dtype=torch.float16)

    y = had._TorchHadamardTransform.hadamard_transform(x, 1.0)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_apply_online_hadamard_short_circuits_an_empty_operand():
    x = torch.empty((2, 0, 256), dtype=torch.float16)

    y = had.apply_online_hadamard(x, True, False, None, 1)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
