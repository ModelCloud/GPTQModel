# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MacaqTritonLinear must follow the activation dtype (#4315).

#4217 fixed only the Torch fallback.  The Triton path still passed the
build-time ``self.compute_dtype`` to ``mxfp4_w4a16_linear`` and built the
W4A4 identity ``BH`` in that dtype, so a module built for bf16 and moved with
``.to(fp16)``/``.float()`` silently computed in bf16.  These tests run the
module's real ``forward`` on CPU with the Triton entry points replaced by
recorders that mirror the kernels' dtype contract.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import macaq as macaq_module
from gptqmodel.nn_modules.qlinear._numerics import linear_fp32_reduce
from gptqmodel.nn_modules.qlinear.macaq import MacaqTritonLinear
from gptqmodel.quantization import macaq as macaq_core


IN_FEATURES = 128
OUT_FEATURES = 64


@pytest.fixture(autouse=True)
def _allow_cpu_construction(monkeypatch):
    """Let the CUDA-only module class be built on CPU for a dtype test."""

    monkeypatch.setattr(MacaqTritonLinear, "cached_validate_once", classmethod(lambda cls: (True, None)))
    monkeypatch.setattr(MacaqTritonLinear, "_can_use_triton", lambda self, x: True)


def _make_layer(*, precision="w4a16", rotation="none", pre_mode="none", bias=True,
                dtype=torch.bfloat16):
    layer = MacaqTritonLinear(
        bits=4,
        group_size=32,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        activation_precision=precision,
        rotation=rotation,
        pre_mode=pre_mode,
        dtype=dtype,
    )
    torch.manual_seed(0)
    layer.W_packed.random_(0, 255)
    layer.W_scales.fill_(122)
    layer.post_init()
    layer.eval()
    return layer


def _reference_w4a16(layer, x, dtype):
    """The kernel's contract, evaluated with the Torch dequantizer."""

    xb = x.to(dtype)
    if layer.pre_scale is not None:
        xb = xb * layer.pre_scale.to(dtype)
    if layer.BH is not None:
        xb = macaq_core.rotate_blocks(xb, layer.BH)
    weight = macaq_core.dequantize_mxfp4(
        layer.W_packed, layer.W_scales, IN_FEATURES, dtype=dtype
    )
    bias = None if layer.bias is None else layer.bias.to(dtype)
    return linear_fp32_reduce(xb, weight, bias)


def _install_w4a16_recorder(monkeypatch):
    seen = []

    def fake_w4a16(x, w_packed, w_scales, *, pre_scale=None, bh=None, bias=None,
                   compute_dtype=None):
        dtype = compute_dtype or x.dtype
        seen.append(dtype)
        xb = x.to(dtype)
        if pre_scale is not None:
            xb = xb * pre_scale.to(dtype)
        if bh is not None:
            xb = macaq_core.rotate_blocks(xb, bh)
        weight = macaq_core.dequantize_mxfp4(w_packed, w_scales, x.shape[-1], dtype=dtype)
        return linear_fp32_reduce(
            xb, weight, None if bias is None else bias.to(dtype)
        )

    monkeypatch.setattr(macaq_module, "mxfp4_w4a16_linear", fake_w4a16)
    return seen


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.float32])
def test_macaq_triton_w4a16_follows_input_dtype_after_to(monkeypatch, target_dtype):
    """`.to(fp16)`/`.float()` must move the Triton compute dtype too."""

    seen = _install_w4a16_recorder(monkeypatch)
    layer = _make_layer(precision="w4a16").to(target_dtype)
    x = torch.randn(3, IN_FEATURES, dtype=target_dtype)

    out = layer(x)

    assert seen == [target_dtype]
    assert out.dtype == target_dtype
    torch.testing.assert_close(out, _reference_w4a16(layer, x, target_dtype), rtol=0, atol=0)


def test_macaq_triton_w4a16_keeps_bf16_for_bf16_input(monkeypatch):
    """The construction-time bf16 default is unchanged for bf16 activations."""

    seen = _install_w4a16_recorder(monkeypatch)
    layer = _make_layer(precision="w4a16")
    x = torch.randn(3, IN_FEATURES, dtype=torch.bfloat16)

    out = layer(x)

    assert seen == [torch.bfloat16]
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, _reference_w4a16(layer, x, torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.float32])
def test_macaq_triton_w4a4_identity_bh_uses_forward_dtype(monkeypatch, target_dtype):
    """The W4A4 identity BH must be built in the dtype the kernel will use."""

    seen = []

    def fake_w4a4(x, w_packed, w_scales, *, pre_scale=None, bh=None, bias=None, **kwargs):
        assert bh is not None
        seen.append((bh.dtype, bh.device))
        return torch.zeros(x.shape[:-1] + (w_packed.shape[0],), dtype=x.dtype)

    monkeypatch.setattr(macaq_module, "mxfp4_w4a4_linear", fake_w4a4)
    layer = _make_layer(precision="w4a4", rotation="none", pre_mode="none").to(target_dtype)
    x = torch.randn(3, IN_FEATURES, dtype=target_dtype)

    out = layer(x)

    assert seen == [(target_dtype, x.device)]
    assert out.dtype == target_dtype

    # Weight dtypes are frozen after the first forward
    # (contracts/kernel/forward-dtype.md).
    other = torch.float32 if target_dtype is torch.float16 else torch.float16
    with pytest.raises(RuntimeError, match="frozen after the first forward"):
        layer.to(other)


def test_macaq_triton_w4a4_identity_bh_stays_bf16(monkeypatch):
    """A bf16 module that is never moved keeps the bf16 identity BH."""

    seen = []

    def fake_w4a4(x, w_packed, w_scales, *, pre_scale=None, bh=None, bias=None, **kwargs):
        seen.append(bh.dtype)
        return torch.zeros(x.shape[:-1] + (w_packed.shape[0],), dtype=x.dtype)

    monkeypatch.setattr(macaq_module, "mxfp4_w4a4_linear", fake_w4a4)
    layer = _make_layer(precision="w4a4", rotation="none", pre_mode="none")

    layer(torch.randn(2, IN_FEATURES, dtype=torch.bfloat16))

    assert seen == [torch.bfloat16]
