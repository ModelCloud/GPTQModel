# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import linear_fp32_reduce, linear_wide_bias
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def test_cpu_fp16_bias_cancels_before_output_rounding():
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=1,
        bias=True,
        register_buffers=True,
    )
    module.qweight.zero_()
    blocks = module.qweight.view(1, -1, 34)
    blocks[:, :, :2] = torch.tensor(
        [2.0], dtype=torch.float16
    ).view(torch.uint8)
    blocks[:, :, 2:] = 64
    module.bias.fill_(-65504.0)
    x = torch.ones((1, 512), dtype=torch.float16)

    output = module._forward_dequant_matmul(x, bias=module.bias)

    assert output.dtype == torch.float16
    assert output[0, 0].item() == 32.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_wide_bias_broadcasts_after_output_reshape(dtype):
    """A wide bias that only broadcasts after the reshape must still apply.

    The wide epilogue flattens the activation for its 2-D ``torch.mm``. Adding a
    per-batch bias such as ``(2, 1, 48)`` to that flattened ``(6, 48)`` product
    raises a shape error, so the bias has to be added once the output is
    reshaped back to ``(2, 3, 48)``.
    """

    generator = torch.Generator().manual_seed(11)
    x = ((torch.rand((2, 3, 32), generator=generator) - 0.5)).to(dtype)
    weight = ((torch.rand((48, 32), generator=generator) - 0.5) / 8).to(dtype)
    # FP32 keeps the stored width, so the wide epilogue is selected.
    bias = torch.rand((2, 1, 48), generator=generator) - 0.5
    oracle = (x.double() @ weight.double().t() + bias.double()).to(dtype)

    for actual in (linear_wide_bias(x, weight, bias), linear_fp32_reduce(x, weight, bias)):
        assert actual.shape == (2, 3, 48)
        assert actual.dtype is dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


def test_cpu_wide_bias_keeps_matmul_broadcast_shape():
    """A bias matching the full output shape broadcasts like the plain expression."""

    generator = torch.Generator().manual_seed(5)
    x = (torch.rand((2, 3, 32), generator=generator) - 0.5).to(torch.float16)
    weight = ((torch.rand((48, 32), generator=generator) - 0.5) / 8).to(torch.float16)
    bias = torch.rand((2, 3, 48), generator=generator, dtype=torch.float32) - 0.5
    expected = (x.double() @ weight.double().t() + bias.double()).to(torch.float16)

    actual = linear_wide_bias(x, weight, bias)

    assert actual.shape == (2, 3, 48)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
