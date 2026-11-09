# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import linear_fp32_reduce
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from tests.test_torch_weight_storage import state_tensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["matrix", "batched", "transposed", "sliced"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_rounds_after_bias(dtype, layout, with_bias):
    value = 300.0 if with_bias else 3.0
    x = torch.full((2, 3, 8 if layout == "sliced" else 4), value, device="cuda", dtype=dtype)
    if layout == "matrix":
        x = x.reshape(-1, 4)
    elif layout == "transposed":
        x = x.transpose(0, 1)
    elif layout == "sliced":
        x = x[..., ::2]
    weight = torch.full((8, 4), 100.0, device="cuda", dtype=dtype)
    bias = torch.full((8,), -60000.0, device="cuda", dtype=dtype) if with_bias else None
    oracle = x.double() @ weight.double().t()
    if bias is not None:
        oracle += bias.double()

    for _ in range(2):
        actual = linear_fp32_reduce(x, weight, bias)
        assert actual.dtype == dtype
        assert actual.shape == x.shape[:-1] + (8,)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_noncontiguous_bias_and_weight_storage(dtype, monkeypatch):
    module = (
        VPTQLinear(
            bits=1,
            in_features=128,
            out_features=64,
            bias=True,
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            group_size=128,
            group_num=-1,
            enable_norm=False,
            enable_perm=False,
            dtype=dtype,
        )
        .to("cuda")
        .eval()
    )
    module.indices.zero_()
    module.bias.data.fill_(-60000.0)
    x = torch.full((2, 3, 128), 30.0, device="cuda", dtype=dtype).transpose(0, 1)
    references = []
    original = module.dequantize

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "dequantize", tracked)
    allowed = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}
    for centroid in (32.0, 16.0):
        module.centroids.weight.data.fill_(centroid)
        oracle = x.double().sum(-1, keepdim=True) * centroid + module.bias.double()
        for _ in range(2):
            with torch.inference_mode():
                actual = module(x)
            assert actual.dtype == dtype
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, oracle.to(dtype), rtol=0, atol=0)
            assert references and all(reference() is None for reference in references)
            assert all(tensor.untyped_storage().data_ptr() in allowed for tensor in state_tensors(module))
