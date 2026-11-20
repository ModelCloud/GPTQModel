# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The TritonV2 eval decode must stay finite for finite inputs (#5738).

The packed fixture decodes to ``(15 - 0) * 8192 = 122880`` on every element.
That weight is finite in FP32 and exactly representable in BF16, but not in
FP16. ``QuantLinearFunction`` asked the Triton dequant kernel for the
activation dtype, so the FP16 forward rounded the weight to ``inf`` and the
matmul returned ``NaN``. The module must take the same range-safe decode as
``BACKEND.TORCH`` (#5606) and agree with an independent FP64 oracle.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT
from tests.test_torch_weight_storage import state_tensors


BITS = 4
GROUP_SIZE = 32
FEATURES = 32
SCALE = 8192.0
DECODED = ((1 << BITS) - 1) * SCALE  # 122880, outside FP16's 65504
X_VALUE = 0.25


def _fixture(cls):
    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        in_features=FEATURES,
        out_features=FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
        register_buffers=True,
    )
    module.qweight.fill_(-1)  # every nibble is 15
    module.qzeros.zero_()     # zero point 0
    module.scales.fill_(SCALE)
    module.post_init()
    return module.cuda().eval()


def _input(dtype):
    x = torch.zeros(1, FEATURES, dtype=dtype, device="cuda")
    x[0, 0] = X_VALUE
    return x


def _oracle(dtype):
    """Independent FP64 product of the decoded weight and the input."""

    weight = torch.full((FEATURES, FEATURES), DECODED, dtype=torch.float64, device="cuda")
    return (_input(torch.float64) @ weight).to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tritonv2_eval_decode_matches_the_range_safe_torch_backend(dtype):
    pytest.importorskip("triton")
    torch_module = _fixture(TorchLinear)
    triton_module = _fixture(TritonV2Linear)
    # 122880 is outside FP16's finite range, so the whole-weight Triton decode
    # cannot stage it and the module has to keep the parent's wide decode.
    assert triton_module._decoded_weight_fits(torch.float16) is False
    assert triton_module._decoded_weight_fits(torch.bfloat16) is True

    x = _input(dtype)
    oracle = _oracle(dtype)
    with torch.inference_mode():
        torch_out = torch_module(x)
        triton_out = triton_module(x)

    for name, out in (("BACKEND.TORCH", torch_out), ("TritonV2", triton_out)):
        assert out.dtype is dtype, f"{name} returned {out.dtype}"
        assert torch.isfinite(out).all(), f"{name} returned {out.flatten()[:4]}"
        torch.testing.assert_close(out, oracle, rtol=0, atol=0)
    torch.testing.assert_close(triton_out, torch_out, rtol=0, atol=0)
    # The wide weight is call-local; no dense or dequantized copy may persist.
    assert not any(
        tensor.shape == (FEATURES, FEATURES) for tensor in state_tensors(triton_module)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tritonv2_range_safe_scales_keep_the_native_triton_decode(monkeypatch):
    """The gate must not reroute checkpoints whose decode already fits."""

    pytest.importorskip("triton")
    from gptqmodel.nn_modules.triton_utils import dequant as dequant_module

    module = _fixture(TritonV2Linear)
    module.scales.fill_(1.0)  # decoded weight 15 fits FP16
    module.clear_weight_cache()
    assert module._decoded_weight_fits(torch.float16) is True
    # Keep the whole-weight path under test instead of the fused GEMV/GEMM.
    module._rocm_gemv_enabled = False
    module._rocm_gemm_enabled = False

    calls = []
    original_apply = dequant_module.QuantLinearFunction.apply

    def spy(*args, **kwargs):
        calls.append(1)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(dequant_module.QuantLinearFunction, "apply", staticmethod(spy))

    x = _input(torch.float16)
    with torch.inference_mode():
        out = module(x)

    assert calls, "range-safe scales bypassed the native Triton decode"
    oracle = (_input(torch.float64) @ torch.full(
        (FEATURES, FEATURES), (1 << BITS) - 1, dtype=torch.float64, device="cuda"
    )).to(torch.float16)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, oracle, rtol=0, atol=0)
