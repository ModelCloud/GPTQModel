# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Triton activation kernels must address rows past the 2^31 element boundary.

``row * K`` was computed in 32-bit, so once ``rows * K`` exceeded 2^31 the
address wrapped and rows past the boundary read or wrote the wrong memory.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qqq_triton import dynamic_quant


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


@requires_cuda
@pytest.mark.cuda
def test_qqq_dynamic_quant_addresses_rows_past_two_to_the_31_elements():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.total_memory < 12 * 1024**3:
        pytest.skip("the overflow regression needs at least 12 GiB of device memory")

    width = 2048
    rows = (2**31 // width) + 1
    x = torch.randn(rows, width, device="cuda", dtype=torch.float16)

    codes, scales = dynamic_quant(x)
    torch.cuda.synchronize()

    assert codes.shape == x.shape
    assert scales.shape == (rows, 1)

    # The last row sits past the 32-bit element boundary. An independent Torch
    # computation of the documented QQQ quantization pins its value.
    for row in (0, rows - 1):
        values = x[row].float()
        scale = values.abs().max() * (1.0 / 127.0)
        expected = torch.clamp(torch.round(values / scale), -128, 127).to(torch.int8)
        assert torch.equal(codes[row], expected)
        assert torch.equal(scales[row, 0], scale)
