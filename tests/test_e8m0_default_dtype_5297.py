# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""E8M0/MXFP4 decoders must not follow the process default dtype.

#5274 replaced ``torch.ldexp(torch.ones_like(x, dtype=torch.float32), e)`` with
``torch.pow(2.0, e)``. A Python float base with an *integer* exponent tensor
follows the process default dtype, so under ``torch.float16`` (which a
concurrent ``GPTQModel.load(..., dtype=torch.float16)`` installs for the whole
process during model construction) every E8M0 scale outside ~[2**-24, 2**15]
decoded to 0 or inf, silently zeroing MXFP4/MXFP8 source weights (#5297).

The exponents are widened to float32 before ``pow``; these tests pin the result
to the float32 baseline for every default dtype.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_bf16_kernels import FP16_SAFE_AMAX
from gptqmodel.quantization.dtype import decode_e8m0_scale
from gptqmodel.utils.kimi_k3 import decode_mxfp4


DEFAULT_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]
HAS_CUDA = torch.cuda.is_available()


@pytest.fixture
def default_dtype():
    """Set the process default dtype and restore it afterwards."""

    original = torch.get_default_dtype()

    def _set(dtype):
        torch.set_default_dtype(dtype)
        return dtype

    try:
        yield _set
    finally:
        torch.set_default_dtype(original)


def _e8m0_bytes(device="cpu") -> torch.Tensor:
    # 0 and 254 are the extreme finite scales; 255 decodes to NaN.
    return torch.arange(256, dtype=torch.uint8, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", DEFAULT_DTYPES)
def test_e8m0_decode_ignores_the_default_dtype(default_dtype, dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA device required")

    encoded = _e8m0_bytes(device)
    torch.set_default_dtype(torch.float32)
    baseline = decode_e8m0_scale(encoded)

    default_dtype(dtype)
    decoded = decode_e8m0_scale(encoded)

    assert decoded.dtype == torch.float32
    assert decoded.device == encoded.device
    finite = encoded != 255
    assert torch.equal(decoded[finite], baseline[finite])
    assert torch.isnan(decoded[255]).all()
    # The two ends the fp16 default used to collapse to 0 / inf.
    assert decoded[0].item() == 2.0**-127
    assert decoded[254].item() == 2.0**127


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", DEFAULT_DTYPES)
def test_mxfp4_decode_ignores_the_default_dtype(default_dtype, dtype, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA device required")

    scales = torch.tensor([[0], [100], [127], [150], [254]], dtype=torch.uint8, device=device)
    # 0x22 is two 1.0 codes, so every decoded value is the block scale itself.
    packed = torch.full((len(scales), 16), 0x22, dtype=torch.uint8, device=device)

    torch.set_default_dtype(torch.float32)
    baseline = decode_mxfp4(packed, scales, dtype=torch.float32)

    default_dtype(dtype)
    decoded = decode_mxfp4(packed, scales, dtype=torch.float32)

    assert decoded.dtype == torch.float32
    assert torch.equal(decoded, baseline)
    # Scale 150 is 2**23: representable, but inf under an fp16 default before.
    assert decoded[3, 0].item() == 2.0**23


@pytest.mark.parametrize("dtype", DEFAULT_DTYPES)
def test_exl3_prescale_keeps_a_wide_row_finite(default_dtype, dtype):
    """A bf16 row at the top of the range needs an exponent fp16 cannot hold."""
    module = object.__new__(ExllamaV3Linear)
    module.out_features = 3
    module.bias = None
    x = torch.tensor(
        [[3.0e38, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.bfloat16
    )

    torch.set_default_dtype(torch.float32)
    baseline_input, baseline_scale = module._prescale_from_row_amax(x, module._row_amax(x))
    assert torch.isfinite(baseline_scale).all()

    default_dtype(dtype)
    native_input, output_scale = module._prescale_from_row_amax(x, module._row_amax(x))

    assert output_scale.dtype == baseline_scale.dtype == torch.float32
    assert torch.equal(output_scale, baseline_scale)
    assert torch.equal(native_input, baseline_input)
    # frexp(3e38 / FP16_SAFE_AMAX) is binade 116, so the scale is 2**116: a
    # finite float32 that an fp16 default would have turned into inf.
    assert output_scale[0, 0].item() == 2.0**116
    assert float(native_input.float().abs().max()) <= FP16_SAFE_AMAX
