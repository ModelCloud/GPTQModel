# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.models.w4a_token_global import token_global_scale


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_token_global_scale_matches_fp64_row_oracle(dtype):
    source = torch.tensor([[0., 0., 0.], [-3., 2., 1.], [0.01, -.001, .005]], dtype=dtype)
    actual = token_global_scale(source)
    expected = torch.tensor([1., max(abs(float(v)) for v in source[1]) / 1792.,
                             max(abs(float(v)) for v in source[2]) / 1792.], dtype=torch.float64)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(token_global_scale(source[1:2]), actual[1:2], rtol=0, atol=0)


@pytest.mark.parametrize("source", [torch.ones(3), torch.ones(2, 0), torch.ones(2, 3, dtype=torch.int32),
                                    torch.full((2, 3), float("nan")), torch.full((2, 3), float("inf")),
                                    torch.full((2, 3), 1e-44)])
def test_token_global_scale_rejects_invalid_or_unrepresentable_input(source):
    with pytest.raises(ValueError):
        token_global_scale(source)


def test_empty_token_global_scale():
    assert token_global_scale(torch.empty(0, 128)).shape == (0,)
