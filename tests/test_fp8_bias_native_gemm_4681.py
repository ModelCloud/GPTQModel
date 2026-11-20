# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A wide FP8 bias must not force the slow FP32 dequantize path (#4681).

FP8 checkpoints store the bias in FP32. ``matmul_fp32_reduce`` keeps such a
bias in its FP32 epilogue, so the native low-precision GEMM with FP32
reduction is still valid; the old guard treated ``bias.dtype != x.dtype`` as a
reason to dequantize the whole weight and run an FP32 SGEMM on every biased
layer (2-5x slower prefill on Qwen1.5, GPT-2/OPT-style models).
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from tests.fp8_test_utils import pack_fp8_module


pytestmark = [pytest.mark.cpu]


def _module(rows=1, columns=4, bias=True):
    return TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=columns,
        out_features=rows,
        bias=bias,
        register_buffers=False,
        format="float8_e4m3fn",
    )


def _biased_module(bias_value=70000.0):
    linear = torch.nn.Linear(4, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(bias_value)
    module = _module()
    pack_fp8_module(module, linear)
    return module


def test_wide_bias_keeps_the_native_gemm_decision():
    module = _biased_module()
    x = torch.zeros((2, 4), dtype=torch.bfloat16)

    assert module.bias.dtype == torch.float32
    assert module._native_gemm_eligible(x, device_type="cuda") is True

    x = torch.zeros((2, 4), dtype=torch.float16)
    assert module._native_gemm_eligible(x, device_type="cuda") is True


def test_cpu_never_takes_the_native_gemm():
    module = _biased_module()
    x = torch.zeros((2, 4), dtype=torch.bfloat16)

    assert module._native_gemm_eligible(x, device_type="cpu") is False
    assert module._native_gemm_eligible(x) is False


def test_fp32_activations_never_take_the_native_gemm():
    module = _biased_module()
    x = torch.zeros((2, 4), dtype=torch.float32)

    assert module._native_gemm_eligible(x, device_type="cuda") is False


def test_out_of_range_weight_still_falls_back_on_cuda():
    """The range guard is separate from the bias rule and must stay."""
    linear = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(65536.0)
    module = _module(rows=1, columns=2, bias=False)
    pack_fp8_module(module, linear)

    x = torch.zeros((1, 2), dtype=torch.float16)

    assert module._native_gemm_eligible(x, device_type="cuda") is False
    assert module._native_gemm_eligible(x.to(torch.bfloat16), device_type="cuda") is True


def test_eligible_forward_forwards_the_wide_bias_to_the_fp32_reducer(monkeypatch):
    """The non-lean fast path must hand the FP32 bias to the FP32 epilogue.

    Row/tensor scales now add the bias in the lean path's own FP32 epilogue
    (#5146); this exercises the block/fallback branch that still delegates to
    ``matmul_fp32_reduce``.
    """

    import gptqmodel.nn_modules.qlinear.fp8 as fp8_module

    module = _biased_module()
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    seen = {}

    def fake_reduce(activations, weight, bias=None):
        seen["bias"] = bias
        return torch.zeros((activations.shape[0], weight.shape[1]), dtype=activations.dtype)

    monkeypatch.setattr(fp8_module, "matmul_fp32_reduce", fake_reduce)
    monkeypatch.setattr(module, "_native_gemm_eligible", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "_lean_epilogue_eligible", lambda *args, **kwargs: False)

    output = module._forward_dequant_matmul(x, bias=module.bias)

    assert seen["bias"] is module.bias
    assert output.shape == (2, 1)


def test_wide_bias_forward_matches_the_fp32_oracle_on_cpu():
    module = _biased_module()
    x = torch.ones((2, 4), dtype=torch.bfloat16)

    output = module(x)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, torch.full((2, 1), 70000.0, dtype=torch.bfloat16))
