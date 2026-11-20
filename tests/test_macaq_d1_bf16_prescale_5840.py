# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5840: MACaQ must quantize against the stored BF16 D1 pre-scale.

The FP32 balance pre-scale normalized the weight and Hessian, and only the
finished result was rounded to BF16. Inference applies the stored BF16 value,
so a coordinate that sits between two E2M1 codes could be packed for a
transform the layer never runs.
"""

import pytest
import torch

from gptqmodel.quantization import macaq


pytestmark = [pytest.mark.cpu]

_WEIGHT = 0.250244140625  # between E2M1 codes 0 and 0.5 under the stored scale
_PEAK = 5.0
_D1 = 1.0021832  # rounds to 1.0 in BF16


def _fixture():
    in_features = 32
    out_features = 4
    weight = torch.zeros(out_features, in_features, dtype=torch.float32)
    weight[:, 0] = _WEIGHT
    weight[:, 1] = _PEAK
    n_tokens = 32
    # ``balance_pre_scale_from_diag`` with alpha=0 returns 1/act_rms; make it D1.
    sum_x2 = torch.full((in_features,), n_tokens / _D1**2, dtype=torch.float64)
    hessian = torch.diag(sum_x2.to(torch.float32))
    return weight, hessian, n_tokens


def test_stored_pre_scale_drives_the_quantized_codes():
    weight, hessian, n_tokens = _fixture()

    result = macaq.macaq_quantize_weight(
        weight,
        hessian,
        n_tokens=n_tokens,
        block_size=32,
        had_block=32,
        rotation="none",
        n_flip_iters=0,
        ridge=0.0,
        do_flip=False,
        pre_mode="balance",
        pre_alpha=0.0,
        exp_screen=False,
        exp_offsets=(0,),
        fast_flip=True,
        compile_flip=False,
        return_dequantized=True,
    )

    # The stored BF16 transform is what inference applies.
    assert result.pre_scale is not None
    assert result.pre_scale.dtype == torch.bfloat16
    assert torch.equal(result.pre_scale, torch.ones_like(result.pre_scale))

    # FP64 nearest-code oracle in the stored frame: amax 5.0 -> E8M0 exponent 0,
    # so the grid is [0, 0.5, 1, ...] and 0.2502... is closest to 0.5.
    normalized = weight.double()[0] / result.pre_scale.double()
    exponent = torch.floor(torch.log2(normalized.abs().amax())) - 2
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64) * torch.exp2(
        exponent
    )
    expected = (grid - normalized[0]).abs().argmin().item()
    assert expected == 1, "the fixture no longer sits between codes 0 and 1"

    code = result.W_packed[0, 0].item() & 15
    assert code == expected

    # Reconstructed weight in the original frame: 0.5, not the pre-fix 0.0.
    restored = (result.dequantized_weight.double() * result.pre_scale.double())[0, 0].item()
    assert restored == 0.5


def test_pre_scale_keeps_the_fp32_definition_rounded_once():
    """Only the rounding point moves; the D1 definition is unchanged."""

    weight, hessian, n_tokens = _fixture()
    fp32_scale = macaq.balance_pre_scale(weight, hessian, n_tokens, alpha=0.0)
    assert fp32_scale[0].item() != 1.0, "the fixture must exercise BF16 rounding"

    result = macaq.macaq_quantize_weight(
        weight,
        hessian,
        n_tokens=n_tokens,
        block_size=32,
        had_block=32,
        rotation="none",
        n_flip_iters=0,
        ridge=0.0,
        do_flip=False,
        pre_mode="balance",
        pre_alpha=0.0,
        exp_screen=False,
        exp_offsets=(0,),
        fast_flip=True,
        compile_flip=False,
        return_dequantized=True,
    )

    assert torch.equal(result.pre_scale, fp32_scale.to(torch.bfloat16))
    assert result.pre_scale.float()[0].item() == 1.0
