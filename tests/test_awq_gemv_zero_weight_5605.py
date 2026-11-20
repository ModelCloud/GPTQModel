# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact zero weights must stay zero when the AWQ FP16 scale rounds (#5605).

``1.4 * 2**-24`` is a finite FP32 scale that rounds to the FP16 subnormal
``2**-24``.  The GEMV packers used to build the affine offset from the FP32
source scale while dividing by the stored FP16 scale, so a zero weight encoded
as ``round(8 * 1.4) = 11`` instead of the zero point 8.  Both portable readers
then added three spurious steps for every zero weight.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTorchLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import unpack_awq
from gptqmodel.utils.packing_math import awq_packed_codes


IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128
BITS = 4
ZERO = 8
SOURCE_SCALE = 1.4 * 2.0**-24
STORED_SCALE = 2.0**-24

CASES = (
    (AwqGEMVTorchLinear, torch.int32, "gemv"),
    (AwqGEMVFastTorchLinear, torch.int16, "gemv_fast"),
)


def _unpack_gemv_fast(module: torch.nn.Module) -> torch.Tensor:
    """Read every GEMV_FAST code from its interleaved int16 words."""
    words = module.qweight.long()
    out = torch.arange(OUT_FEATURES, device=words.device)[:, None]
    col = torch.arange(IN_FEATURES, device=words.device)[None, :]
    within = col % 32
    transposed = ((within // 2) % 4) * 8 + (within // 8) * 2 + within % 2
    permuted = (transposed // 8) * 8 + (transposed % 8) // 2 + (transposed % 2) * 4
    index = (col % 64) // 32 * 32 + permuted
    word_col = col // 64 * 64 + out % 4 * 16 + index // 4
    return (words[out // 4, word_col] >> (4 * (index % 4))) & 15


def _make_case(cls, pack_dtype):
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[:, 0] = 2.0**-21
    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=pack_dtype,
    )
    module.pack(
        linear,
        torch.full((OUT_FEATURES, 1), SOURCE_SCALE),
        torch.full((OUT_FEATURES, 1), float(ZERO)),
    )
    return linear, module


def _packed_codes(module, cls) -> torch.Tensor:
    if cls is AwqGEMVTorchLinear:
        codes, _ = unpack_awq(module.qweight, None, BITS, count=IN_FEATURES)
        return codes
    return _unpack_gemv_fast(module)


@pytest.mark.parametrize("cls,pack_dtype,name", CASES, ids=[case[2] for case in CASES])
def test_exact_zero_weight_keeps_the_stored_zero_code(cls, pack_dtype, name):
    _, module = _make_case(cls, pack_dtype)

    # The source scale rounds to the FP16 subnormal that the kernel stores.
    stored = module.scales if cls is AwqGEMVTorchLinear else module.scales.t()
    assert torch.unique(stored[:, :1]).tolist() == [STORED_SCALE]

    codes = _packed_codes(module, cls)
    assert codes.shape == (OUT_FEATURES, IN_FEATURES)
    # Column 0 saturates (2**-21 is above this group's grid); every exact zero
    # weight must keep the checkpoint zero point instead.
    assert torch.equal(codes[:, 1:], torch.full_like(codes[:, 1:], ZERO))


@pytest.mark.parametrize("cls,pack_dtype,name", CASES, ids=[case[2] for case in CASES])
def test_masked_forward_matches_the_dense_zero(cls, pack_dtype, name):
    linear, module = _make_case(cls, pack_dtype)

    x = torch.full((1, IN_FEATURES), 1024.0, dtype=torch.float16)
    x[:, 0] = 0.0
    with torch.no_grad():
        dense = linear(x)[0, 0]
        packed = module(x)[0, 0]

    assert dense.item() == 0.0
    assert packed.item() == 0.0


def test_predictor_uses_the_stored_scale_for_the_zero_point():
    weight = torch.zeros((OUT_FEATURES, IN_FEATURES), dtype=torch.float16)
    scales = torch.full((OUT_FEATURES, 1), SOURCE_SCALE, dtype=torch.float32)
    zeros = torch.full((OUT_FEATURES, 1), ZERO, dtype=torch.int32)
    groups = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE

    codes = awq_packed_codes(
        weight, scales, zeros, groups, bits=BITS, scale_dtype=torch.float16
    )

    assert torch.equal(codes, torch.full_like(codes, ZERO))
