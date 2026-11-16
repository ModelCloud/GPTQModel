# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Triton activation quantizers must not fold NaN into a finite code (#4220).

``tl.max`` ignores NaN, so ``amax`` stays finite, the clamp turns the NaN
quotient into a saturated code, and the NaN disappears.  Both quantizers now
poison the row scale so the NaN reaches the caller (the Torch reference
``fp8_token_qdq`` rejects such a row outright).
"""

import pytest
import torch

from gptqmodel.utils.rocm import IS_ROCM


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton quantizers are CUDA-only")


def _nan_row_input(rows=2, width=128):
    x = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    x[0, 5] = float("nan")
    return x


def test_qqq_triton_quantizer_propagates_nan():
    from gptqmodel.nn_modules.qlinear.qqq_triton import dynamic_quant

    x = _nan_row_input()
    codes, scales = dynamic_quant(x)

    assert torch.isnan(scales[0, 0])
    assert torch.isnan((codes.float() * scales)[0, 5])
    # The finite row must keep its exact reference scale.
    expected = x[1].float().abs().amax() / 127.0
    torch.testing.assert_close(scales[1, 0], expected, rtol=0, atol=0)


@pytest.mark.skipif(
    IS_ROCM or (torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9)),
    reason="W4A Triton E4M3 kernels are CUDA-only (SM 8.9+) and raise on ROCm",
)
def test_fp8_triton_quantizer_propagates_nan():
    from gptqmodel.nn_modules.qlinear.w4a_triton import fp8_pack

    x = _nan_row_input()
    codes, scales = fp8_pack(x)

    assert torch.isnan(scales[0])
    assert torch.isnan((codes.float() * scales[:, None])[0, 5])
    expected = x[1].float().abs().amax() / 448.0
    torch.testing.assert_close(scales[1], expected, rtol=0, atol=0)
