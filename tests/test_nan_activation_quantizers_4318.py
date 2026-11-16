# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MACaQ W4A4 and NVFP4 activation quantizers must not fold NaN into finite codes (#4318).

Same failure class as #4220.  ``tl.max`` drops NaN, so the block scale stays
finite and the NaN quotient falls through the threshold comparisons to a
plausible finite code.  Both formats have a NaN scale code - E8M0 ``0xFF`` for
MXFP4 and E4M3 ``0x7F`` for NVFP4 - so the quantizer writes that instead and
every consumer (hardware ``tl.dot_scaled``, Torch dequant, NVFP4 decode) sees a
non-finite output.

Inf is folded onto the same path: neither format can encode an infinity, and
#4220's FP8 quantizers already turn an Inf row into NaN, so the Torch and Triton
paths stay consistent with each other.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.macaq import (
    dequantize_mxfp4,
    macaq_quantize_activations,
    pack_mxfp4_from_signed_index,
)
from gptqmodel.utils.macaq_triton import (
    dequantize_mxfp4 as triton_dequantize_mxfp4,
)
from gptqmodel.utils.macaq_triton import (
    make_hadamard,
    quantize_mxfp4,
    rot_scale_quantize_mxfp4,
    triton_mxfp4_available,
)
from gptqmodel.utils.rocm import IS_ROCM


MXFP4_BLOCK = 32
WIDTH = 256

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="activation quantizers run on a GPU"
)
requires_triton = pytest.mark.skipif(
    not triton_mxfp4_available(), reason="Triton MXFP4 needs CUDA or gfx1201"
)


def _nonfinite_input(rows=4, width=WIDTH):
    """Rows 1 and 2 hold NaN/Inf; rows 0 and 3 stay finite."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(rows, width, device="cuda", dtype=torch.float32, generator=generator)
    x[1, 7] = float("nan")
    x[2, 9] = float("inf")
    return x


# ---------------------------------------------------------------------------
# Torch reference path (CPU-safe)


def test_pack_maps_nonfinite_exponent_to_e8m0_nan():
    """A NaN/Inf block exponent packs as E8M0 0xFF, not a plausible exponent."""
    signed_index = torch.zeros(2, 64, dtype=torch.int64)
    log2_scale = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
    log2_scale[0, 0] = float("nan")
    log2_scale[1, 1] = float("inf")

    _, scales = pack_mxfp4_from_signed_index(signed_index, log2_scale, MXFP4_BLOCK)
    packed_bytes = scales.view(torch.uint8)
    assert packed_bytes[0, 0].item() == 255
    assert packed_bytes[1, 1].item() == 255
    # Finite blocks keep their exact exponent.
    assert packed_bytes[0, 1].item() == 126
    assert packed_bytes[1, 0].item() == 127


def test_e8m0_nan_byte_decodes_to_nan():
    """E8M0 0xFF is the format's NaN, not ``2**128``.

    A nonzero code is used so the old ``exp2(255 - 127) == inf`` path would
    decode to ``inf`` rather than to ``0 * inf == NaN``.
    """
    packed = torch.full((1, 32), 0x11, dtype=torch.uint8)  # magnitude 0.5
    scales = torch.tensor([[255, 126]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    out = dequantize_mxfp4(packed, scales, 64, dtype=torch.float32)
    assert torch.isnan(out[0, :32]).all()
    assert torch.isfinite(out[0, 32:]).all()
    # 0.5 * 2 ** (126 - 127) for the finite block.
    assert out[0, 32].item() == 0.25


# ---------------------------------------------------------------------------
# MACaQ W4A4 activation quantizers


@requires_cuda
def test_macaq_torch_activations_propagate_nonfinite():
    """The Torch reference path must not return a finite row for NaN/Inf input."""
    x = _nonfinite_input()
    for bh in (None, make_hadamard(MXFP4_BLOCK, "cuda")):
        out = macaq_quantize_activations(x, bh=bh)
        assert out.isfinite().all(dim=1).tolist() == [True, False, False, True]
        assert torch.isnan(out).any(dim=1).tolist() == [False, True, True, False]


@requires_triton
def test_macaq_triton_quantizer_writes_e8m0_nan():
    """The hardware quantizer stores E8M0 0xFF for a NaN/Inf block."""
    x = _nonfinite_input()
    packed, e8m0 = quantize_mxfp4(x.to(torch.bfloat16).float().contiguous())
    packed_bytes = e8m0.view(torch.uint8)
    assert packed_bytes[1, 0].item() == 255
    assert packed_bytes[2, 0].item() == 255

    decoded = triton_dequantize_mxfp4(packed, e8m0, WIDTH)
    assert decoded.isfinite().all(dim=1).tolist() == [True, False, False, True]


@requires_triton
def test_macaq_triton_rot_quantizer_writes_e8m0_nan():
    """The fused rotate+quantize kernel poisons its scale as well."""
    x = _nonfinite_input()
    _, e8m0 = rot_scale_quantize_mxfp4(
        x.contiguous(), None, make_hadamard(MXFP4_BLOCK, "cuda")
    )
    packed_bytes = e8m0.view(torch.uint8)
    assert packed_bytes[1, 0].item() == 255
    assert packed_bytes[2, 0].item() == 255


@requires_triton
def test_macaq_module_paths_agree_on_nonfinite():
    """Both module paths must agree that the NaN/Inf rows are non-finite."""
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.macaq import MacaqTritonLinear

    module = MacaqTritonLinear(
        bits=4, group_size=32, in_features=WIDTH, out_features=128, bias=False,
        activation_precision="w4a4", rotation="hadamard", pre_mode="none",
        dtype=torch.bfloat16, backend=BACKEND.MACAQ_TRITON,
    )
    module._allocate_buffers(bias=False)
    module.W_packed.random_(0, 255)
    module.W_scales.fill_(122)
    module.post_init()
    module.cuda()

    x = _nonfinite_input()
    for out in (module._forward_triton(x), module._forward_torch(x)):
        assert out.isfinite().all(dim=1).tolist() == [True, False, False, True]


# ---------------------------------------------------------------------------
# NVFP4 activation packing (CUDA only)


@pytest.mark.skipif(IS_ROCM, reason="NVFP4 activation packing is CUDA-only")
@requires_cuda
@pytest.mark.parametrize(
    "recipe", ["least_squares", "nvidia", "four_six", "least_squares_headroom"]
)
def test_nvfp4_recipes_propagate_nonfinite(recipe):
    """Every NVFP4 recipe keeps the NaN/Inf rows non-finite after decode."""
    from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation
    from gptqmodel.quantization.activation_floatx import normalize_nvfp4_recipe

    x = _nonfinite_input()
    global_scale = torch.tensor(0.01, device="cuda")
    decoded = pack_activation(
        x, "w4a_nvfp4", global_scale=global_scale,
        recipe=normalize_nvfp4_recipe(recipe),
    ).decode(torch.float32)
    assert decoded.isfinite().all(dim=1).tolist() == [True, False, False, True]
