# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 32, 33, 64])
def test_exllamav3_wide_inputs_use_torch_reconstruction(dtype, rows):
    module = object.__new__(ExllamaV3Linear)
    module.out_features = 4
    calls = []

    class Native:
        def forward(self, x, params, out_dtype):
            calls.append(("native", x.dtype, out_dtype))
            return x

    class TorchPath:
        def forward(self, x):
            calls.append(("torch", x.dtype))
            return torch.ones((*x.shape[:-1], 4), dtype=x.dtype)

    module._ensure_inner = lambda: Native()
    module._ensure_torch_inner = lambda: TorchPath()
    output = module.forward(torch.ones((rows, 3), dtype=dtype))

    assert output.dtype == dtype
    assert output.shape == (rows, 4)
    assert calls == [("torch", dtype)]


@pytest.mark.parametrize("rows", [32, 33])
def test_exllamav3_half_inputs_keep_native_dispatch(rows):
    module = object.__new__(ExllamaV3Linear)
    module.out_features = 4
    calls = []

    class Native:
        def forward(self, x, params, out_dtype):
            calls.append((x.dtype, out_dtype, x.shape[0]))
            return torch.ones((rows, 4), dtype=out_dtype)

    module._ensure_inner = lambda: Native()
    module._ensure_torch_inner = lambda: pytest.fail("FP16 must keep the native path")

    output = module.forward(torch.ones((rows, 3), dtype=torch.float16))

    assert output.shape == (rows, 4)
    assert output.dtype == torch.float16
    assert calls == [(torch.float16, torch.float16, rows)]


@pytest.mark.parametrize("rows, branch", [(32, "bc"), (33, "hgemm")])
def test_native_exl3_row_boundary_uses_expected_half_kernel(monkeypatch, rows, branch):
    from gptqmodel.exllamav3.modules.quant import exl3 as native_exl3

    calls = []

    class FakeBC:
        def run(self, x, y):
            calls.append("bc")
            y.copy_(x)

    class FakeExt:
        @staticmethod
        def had_r_128(x, y, suh, svh, scale):
            y.copy_(x)

        @staticmethod
        def hgemm(x, weight, y):
            calls.append("hgemm")
            y.copy_(x @ weight)

    monkeypatch.setattr(native_exl3, "ext", FakeExt)
    inner = object.__new__(native_exl3.LinearEXL3)
    inner.in_features = inner.out_features = 128
    inner.out_dtype = torch.float16
    inner.suh = inner.svh = inner.bias = None
    inner.bc = FakeBC()
    inner.get_inner_weight_tensor = lambda: torch.eye(128, dtype=torch.float16)

    x = torch.ones((rows, 128), dtype=torch.float16)
    actual = inner.forward(x, {})

    assert calls == [branch]
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
