# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ packers must not derive codes from a scale the storage dtype lost.

The GEMM, GEMV, and GEMV_FAST layouts store their group scales as FP16.  A
source FP32 scale that overflows to infinity or underflows to zero changes every
code computed from it, and clamping the resulting code afterwards only hides the
corruption.  Packing must reject the storage cast instead, before replacing any
module state.
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.quantization.awq.utils.packing_utils import unpack_awq
from gptqmodel.utils.packing import hf_pack_layer


pytestmark = [pytest.mark.cpu]

AWQ_KERNELS = (AwqGEMMLinear, AwqGEMVLinear, AwqGEMVFastLinear)

IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128


def _make_module(cls) -> nn.Module:
    """Build just enough of a native GPU kernel to exercise CPU packing."""
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.bits = 4
    module.group_size = GROUP_SIZE
    module.in_features = IN_FEATURES
    module.out_features = OUT_FEATURES
    module.bias = None
    module.pack_dtype = torch.int32
    module.adapter = None
    module.zeros_name = "qzeros"
    return module


class _PackingOnlyAllSpark(AllSparkAwqLinear):
    """AllSpark AWQ with its derived-kernel cache release stubbed out."""

    def clear_weight_cache(self) -> None:
        pass


def _make_allspark_module() -> nn.Module:
    module = _PackingOnlyAllSpark.__new__(_PackingOnlyAllSpark)
    nn.Module.__init__(module)
    module.bits = 8
    module.group_size = IN_FEATURES
    module.in_features = IN_FEATURES
    module.out_features = OUT_FEATURES
    module.requested_group_size = IN_FEATURES
    module.maxq = (1 << 8) - 1
    module.compute_dtype = torch.float16
    module.bias = None
    module.adapter = None
    module.pack_dtype = torch.int32
    return module


def _make_linear(weight: float) -> nn.Linear:
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.fill_(weight)
    return linear


def _scales(scale: float) -> torch.Tensor:
    return torch.full(
        (OUT_FEATURES, IN_FEATURES // GROUP_SIZE), scale, dtype=torch.float32
    )


def _zeros(zero: int = 8) -> torch.Tensor:
    return torch.full(
        (OUT_FEATURES, IN_FEATURES // GROUP_SIZE), zero, dtype=torch.int32
    )


def _first_code(module: nn.Module) -> int:
    codes, _ = unpack_awq(module.qweight, None, module.bits, count=IN_FEATURES)
    return int(codes[0, 0])


@pytest.mark.parametrize("cls", AWQ_KERNELS)
@pytest.mark.parametrize(
    "scale",
    [
        2.0**-26,
        2.0**-25,
        float(torch.nextafter(torch.tensor(2.0**-25), torch.tensor(0.0))),
    ],
    ids=["below-midpoint", "midpoint-ties-to-even", "just-below-midpoint"],
)
def test_rejects_scale_that_underflows_fp16(cls, scale):
    module = _make_module(cls)

    with pytest.raises(ValueError, match="cannot be stored in torch.float16"):
        hf_pack_layer(module, _make_linear(2.0**-24), _scales(scale), _zeros())


@pytest.mark.parametrize("cls", AWQ_KERNELS)
@pytest.mark.parametrize(
    "scale",
    [65520.0, 65536.0, 1e5],
    ids=["overflow-midpoint", "next-power-of-two", "far-overflow"],
)
def test_rejects_scale_that_overflows_fp16(cls, scale):
    module = _make_module(cls)

    with pytest.raises(ValueError, match="cannot be stored in torch.float16"):
        hf_pack_layer(module, _make_linear(0.0), _scales(scale), _zeros())


@pytest.mark.parametrize("cls", AWQ_KERNELS)
@pytest.mark.parametrize(
    "scale,weight,expected",
    [
        (65504.0, 0.0, 1),  # largest finite FP16 scale
        (float(torch.nextafter(torch.tensor(65520.0), torch.tensor(0.0))), 0.0, 1),
        (2.0**-24, 2.0**-24, 2),  # smallest positive FP16 subnormal
        (
            float(torch.nextafter(torch.tensor(2.0**-25), torch.tensor(torch.inf))),
            2.0**-24,
            2,
        ),
    ],
    ids=["max-finite", "just-below-overflow-midpoint", "min-subnormal", "just-above-underflow-midpoint"],
)
def test_representable_scale_boundaries_keep_their_codes(cls, scale, weight, expected):
    module = _make_module(cls)

    # Use a unit zero point so GEMV_FAST's FP16 offset stays representable and
    # this case measures the scale boundary alone.
    hf_pack_layer(module, _make_linear(weight), _scales(scale), _zeros(1))

    assert _first_code(module) == expected


@pytest.mark.parametrize("cls", AWQ_KERNELS)
def test_failed_pack_leaves_module_state_intact(cls):
    module = _make_module(cls)
    module.register_buffer("scales", torch.tensor([7.0], dtype=torch.float16))
    module.register_buffer("qweight", torch.tensor([11], dtype=torch.int32))
    module.register_buffer("qzeros", torch.tensor([13], dtype=torch.int32))
    before = {name: value.clone() for name, value in module.named_buffers()}

    with pytest.raises(ValueError, match="cannot be stored in torch.float16"):
        hf_pack_layer(module, _make_linear(0.0), _scales(65520.0), _zeros())

    after = dict(module.named_buffers())
    assert set(after) == set(before)
    for name, value in before.items():
        assert torch.equal(after[name], value)


@pytest.mark.parametrize(
    "scale",
    [
        2.0**-26,
        2.0**-25,
        65520.0,
        65536.0,
    ],
    ids=["underflow", "underflow-tie", "overflow-midpoint", "overflow"],
)
def test_allspark_rejects_unrepresentable_scale(scale):
    module = _make_allspark_module()

    with pytest.raises(ValueError, match="cannot be stored in torch.float16"):
        hf_pack_layer(module, _make_linear(0.0), _scales(scale), _zeros(128))


@pytest.mark.parametrize(
    "scale,weight,expected",
    [
        (65504.0, 0.0, 128),
        (float(torch.nextafter(torch.tensor(65520.0), torch.tensor(0.0))), 0.0, 128),
        (2.0**-24, 2.0**-24, 129),
        (
            float(torch.nextafter(torch.tensor(2.0**-25), torch.tensor(torch.inf))),
            2.0**-24,
            129,
        ),
    ],
    ids=["max-finite", "just-below-overflow-midpoint", "min-subnormal", "just-above-underflow-midpoint"],
)
def test_allspark_representable_scale_boundaries_keep_their_codes(scale, weight, expected):
    module = _make_allspark_module()

    hf_pack_layer(module, _make_linear(weight), _scales(scale), _zeros(128))

    assert _first_code(module) == expected


def test_allspark_failed_pack_leaves_module_state_intact():
    module = _make_allspark_module()
    module.register_buffer("scales", torch.tensor([7.0], dtype=torch.float16))
    module.register_buffer("qweight", torch.tensor([11], dtype=torch.int32))
    module.register_buffer("qzeros", torch.tensor([13], dtype=torch.int32))
    before = {name: value.clone() for name, value in module.named_buffers()}

    with pytest.raises(ValueError, match="cannot be stored in torch.float16"):
        hf_pack_layer(module, _make_linear(0.0), _scales(65520.0), _zeros(128))

    after = dict(module.named_buffers())
    assert set(after) == set(before)
    for name, value in before.items():
        assert torch.equal(after[name], value)
