# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5695: the INT8 pack must keep the source-precision bias.

``BitsAndBytesLinear.pack_original`` stored the bias in the compute dtype, so
an FP32 source bias of ``-257`` became ``-256`` in the packed state before any
forward ran. Both values are exactly representable in BF16, so the loss was
pure narrowing, and the forward cast the already-rounded buffer again. The
INT8 epilogues add the bias in FP32 after the exact INT32 product; keep the
packed bias in FP32 and cast only at that single output rounding.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitsandbytes import (
    BITSANDBYTES_AVAILABLE,
    BitsAndBytesLinear,
)


pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes backend unavailable"),
]


def _issue_fixture(dtype: torch.dtype) -> tuple[torch.nn.Linear, torch.Tensor]:
    """Build the issue's 16x4 unit-weight fixture and its BF16/FP32 input."""

    linear = torch.nn.Linear(16, 4)
    linear.weight.data.zero_()
    linear.weight.data[:, 0] = 1.0
    linear.bias.data.fill_(-257.0)
    x = torch.zeros(1, 16, dtype=dtype)
    x[0, 0] = 256.0
    return linear, x


def _kernel(dtype: torch.dtype, *, register_buffers: bool = False) -> BitsAndBytesLinear:
    return BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=4,
        bias=True,
        dtype=dtype,
        register_buffers=register_buffers,
    )


def _packed(dtype: torch.dtype) -> tuple[BitsAndBytesLinear, torch.Tensor]:
    linear, x = _issue_fixture(dtype)
    kernel = _kernel(dtype)
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()
    kernel.eval()
    return kernel, x


def test_int8_pack_keeps_fp32_bias_for_bfloat16_compute():
    kernel, x = _packed(torch.bfloat16)

    assert kernel.bias.dtype == torch.float32
    assert kernel.bias[0].item() == -257.0
    assert kernel.state_dict()["bias"].dtype == torch.float32

    with torch.inference_mode():
        out = kernel(x)
    assert out[0, 0].item() == -1.0


def test_int8_fp32_path_is_unchanged():
    kernel, x = _packed(torch.float32)

    assert kernel.bias.dtype == torch.float32
    assert kernel.bias[0].item() == -257.0

    with torch.inference_mode():
        out = kernel(x)
    assert out[0, 0].item() == -1.0


@pytest.mark.parametrize("bias", [-256.0, 1.5])
def test_int8_bf16_representable_bias_is_unchanged(bias: float):
    linear, _ = _issue_fixture(torch.bfloat16)
    linear.bias.data.fill_(bias)
    kernel = _kernel(torch.bfloat16)
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()

    assert kernel.bias.dtype == torch.float32
    assert kernel.bias[0].item() == bias
    assert torch.equal(kernel.bias, linear.bias.detach().float())


def test_int8_bf16_source_bias_is_unchanged():
    linear = torch.nn.Linear(16, 4, bias=True).to(torch.bfloat16)
    linear.weight.data.zero_()
    linear.weight.data[:, 0] = 1.0
    linear.bias.data.fill_(-257.0)
    kernel = _kernel(torch.bfloat16)
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel.post_init()

    assert kernel.bias.dtype == torch.float32
    assert torch.equal(kernel.bias, linear.bias.detach().float())


def test_int8_state_dict_reload_keeps_fp32_bias():
    kernel, x = _packed(torch.bfloat16)
    reloaded = _kernel(torch.bfloat16, register_buffers=True)

    assert reloaded.bias.dtype == torch.float32
    reloaded.load_state_dict(kernel.state_dict(), strict=True)
    reloaded.post_init()
    reloaded.eval()

    assert reloaded.bias[0].item() == -257.0
    with torch.inference_mode():
        out = reloaded(x)
    assert out[0, 0].item() == -1.0


def test_int8_to_dtype_and_device_moves_keep_fp32_bias():
    kernel, x = _packed(torch.bfloat16)

    moved = kernel.to(torch.float16)
    assert moved is kernel
    assert moved.bias.dtype == torch.float32
    assert moved.bias[0].item() == -257.0

    moved = moved.to(torch.bfloat16).to("cpu")
    assert moved.bias.dtype == torch.float32
    assert moved.bias[0].item() == -257.0
    with torch.inference_mode():
        out = moved(x)
    assert out[0, 0].item() == -1.0


def test_int8_to_empty_round_trip_keeps_fp32_bias():
    kernel, x = _packed(torch.bfloat16)
    state = kernel.state_dict()

    meta = kernel.to_empty(device="meta")
    assert meta.bias.device.type == "meta"
    assert meta.bias.dtype == torch.float32

    restored = meta.to_empty(device="cpu")
    assert restored.bias.device.type == "cpu"
    assert restored.bias.dtype == torch.float32
    restored.load_state_dict(state, strict=True)
    restored.post_init()
    restored.eval()

    assert restored.bias[0].item() == -257.0
    with torch.inference_mode():
        out = restored(x)
    assert out[0, 0].item() == -1.0


def test_int8_pack_original_reentry_refreshes_fp32_bias():
    kernel = _kernel(torch.bfloat16)
    linear, _ = _issue_fixture(torch.bfloat16)

    kernel.pack_original(linear=linear, scales=None, zeros=None)
    assert kernel.bias.dtype == torch.float32
    assert kernel.bias[0].item() == -257.0

    linear.bias.data.fill_(-256.0)
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    assert kernel.bias.dtype == torch.float32
    assert kernel.bias[0].item() == -256.0
