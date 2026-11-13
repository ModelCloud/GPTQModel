# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from gptqmodel.quantization.rotation import hadamard_utils as had


def _walsh_oracle(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Independent FP64 unnormalized Walsh transform scaled by ``scale``."""
    width = x.shape[-1]
    columns = torch.arange(width, dtype=torch.int64)
    products = columns[:, None] & columns[None, :]
    parity = torch.zeros_like(products)
    for bit in range(width.bit_length() - 1):
        parity ^= (products >> bit) & 1
    matrix = (1 - 2 * parity).double()
    return (x.detach().cpu().double() @ matrix) * float(scale)


@pytest.mark.parametrize("width", [8192, 16384])
@pytest.mark.parametrize("amplitude", [8.0, -8.0])
def test_portable_hadamard_half_precision_constant_row_stays_finite(width, amplitude):
    """The pre-normalization DC sum can exceed FP16 even when the result cannot."""
    x = torch.full((1, width), amplitude, dtype=torch.float16)
    scale = 1.0 / math.sqrt(width)

    actual = had._TorchHadamardTransform.hadamard_transform(x, scale)

    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    expected_dc = torch.tensor(amplitude * math.sqrt(width), dtype=torch.float16)
    assert actual[0, 0] == expected_dc
    assert torch.count_nonzero(actual[0, 1:]) == 0


@pytest.mark.parametrize("backend_name", ["_TorchHadamardTransform", "_TritonHadamardTransform"])
def test_online_hadamard_half_precision_does_not_overflow(backend_name, monkeypatch):
    """Both fallback wrappers route the CPU half-precision case through the fix."""
    backend = getattr(had, backend_name)()
    monkeypatch.setattr(had, "fast_hadamard_transform", backend)
    x = torch.full((2, 8192), 8.0, dtype=torch.float16)

    actual = had.apply_online_hadamard(x, True, False, None, 1)

    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    expected_dc = torch.tensor(8.0 * math.sqrt(8192), dtype=torch.float16)
    torch.testing.assert_close(actual[:, 0], expected_dc.expand(2), rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 1:]) == 0


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
        # FP32 control: butterfly summation order differs from the matmul oracle.
        (torch.float32, 2e-4, 2e-4),
    ],
)
def test_portable_hadamard_matches_fp64_walsh_oracle(dtype, rtol, atol):
    torch.manual_seed(3827)
    x = (torch.rand(4, 256, dtype=torch.float64) * 500 - 250).to(dtype)
    scale = 1.0 / math.sqrt(x.shape[-1])

    actual = had._TorchHadamardTransform.hadamard_transform(x, scale)
    expected = _walsh_oracle(x, scale).to(dtype)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
