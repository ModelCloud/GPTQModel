# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for asymmetric Machete zero-point handling (#5800).

Machete's generated kernels are Hopper-only (SM90), so these tests run on CPU.
They pin the Python-side launch contract and the runtime representation the
native ``transform_A_kblock`` consumes. Native SM90 execution is not measured
here; the PR records that gap.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from gptqmodel.nn_modules.qlinear import machete, machete_awq
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.utils.machete import scalar_types, unpack_quantized_values_into_int32
from gptqmodel.utils.packing_math import zero_points_in_scale_dtype


def _packed_word(value: int) -> int:
    word = sum((value & 0xF) << (4 * lane) for lane in range(8))
    return word - 2**32 if word >= 2**31 else word


def _bare_linear(cls):
    """Build an uninitialized module so the test can observe the native kwargs."""

    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.group_size = module.in_features = module.out_features = 128
    module.scales = torch.full((1, 128), 1 / 16, dtype=torch.float16)
    module.qzeros = torch.full((1, 128), 3.0, dtype=torch.float16)
    module.qweight = torch.empty(16, 128, dtype=torch.int32)
    module.weight_type = scalar_types.uint4
    module.has_zero_points = True
    module.input_perm = torch.empty(0, dtype=torch.int32)
    module.bias = None
    module.adapter = None
    return module


@pytest.mark.parametrize(
    ("module_module", "cls"),
    ((machete, MacheteLinear), (machete_awq, AwqMacheteLinear)),
)
@pytest.mark.parametrize("stored_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_forward_forwards_one_operand_dtype(
    module_module, cls, stored_dtype, activation_dtype
):
    """Activation, group scale and group zero point must share one dtype.

    All generated asymmetric U4/U8 kernels declare ``b_group_scale`` and
    ``b_group_zeropoint`` equal to the activation type, and ``mm_dispatch``
    rejects the unmatched signature.
    """

    module = _bare_linear(cls)
    module.scales = module.scales.to(dtype=stored_dtype)
    module.qzeros = module.qzeros.to(dtype=stored_dtype)
    activation = torch.ones(1, 128, dtype=activation_dtype)

    with patch.object(module_module, "machete_mm", return_value=activation) as native:
        module(activation)

    operands = native.call_args.kwargs
    forwarded = tuple(
        operands[name].dtype for name in ("a", "b_group_scales", "b_group_zeros")
    )
    assert forwarded == (activation_dtype,) * 3
    # The launch derives temporaries; checkpoint storage keeps its dtype.
    assert module.scales.dtype == stored_dtype
    assert module.qzeros.dtype == stored_dtype


def _asymmetric_module(dtype, groups, cls=MacheteLinear):
    """Build a loaded asymmetric module from per-group ``(scale, code, zero)``."""

    in_features = 128 * len(groups)
    with patch.object(
        cls, "validate", classmethod(lambda klass, **kwargs: (True, None))
    ):
        module = cls(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=False,
            in_features=in_features,
            out_features=128,
            register_buffers=True,
        )

    module.scales = torch.nn.Parameter(
        torch.tensor(
            [[scale] * module.out_features for scale, _, _ in groups], dtype=dtype
        ),
        requires_grad=False,
    )
    # GPTQ packs eight codes per word along in_features; AWQ packs them along
    # out_features, so a group spans one qweight row per input channel.
    rows_per_group = 16 if cls is MacheteLinear else 128
    for index, (_, code, zero) in enumerate(groups):
        module.qweight.data[index * rows_per_group:(index + 1) * rows_per_group, :] = (
            _packed_word(code)
        )
        module.qzeros.data[index, :] = _packed_word(zero)

    # The native prepack is SM90-only; keep the packed codes so the test can
    # recover them and exercise the post_init representation on CPU.
    module_module = machete if cls is MacheteLinear else machete_awq
    with patch.object(
        module_module,
        "machete_prepack_B",
        side_effect=lambda weight, **kwargs: weight.clone(),
    ):
        module.post_init()
    return module


@pytest.mark.parametrize("cls", [MacheteLinear, AwqMacheteLinear])
@pytest.mark.parametrize(
    ("dtype", "groups", "legacy_differs"),
    (
        # Narrow BF16 scale where a pre-rounded `-zero * scale` offset moved the
        # weight by one BF16 step (1.0625 instead of 1.015625).
        (torch.bfloat16, [(1.015625, 14, 13)], True),
        # FP16 `code * scale` overflows before the offset is added; the
        # difference-then-scale order stays finite.
        (torch.float16, [(5000.0, 14, 10)], True),
        # Equal code and zero point decode to exactly zero.
        (torch.bfloat16, [(1.015625, 13, 13)], False),
        # Zero point above the code gives a negative weight.
        (torch.bfloat16, [(0.5, 3, 10)], False),
        # Per-group scales and zero points stay on their own group boundary.
        (torch.bfloat16, [(1.015625, 14, 13), (0.5, 6, 5)], True),
    ),
)
def test_same_dtype_dequantized_weights_match_checkpoint_oracle(
    dtype, groups, legacy_differs, cls
):
    """The runtime zero table must decode to the checkpoint's affine weights.

    ``post_init`` keeps the logical zero point, and the native
    ``transform_A_kblock`` subtracts it from the upcast code before the scale
    multiply, so the weight is ``round((code - zero) * scale)``. The CPU mirror
    below applies that operand order; the oracle is computed independently in
    float64 from the checkpoint values.
    """

    module = _asymmetric_module(dtype, groups, cls)

    expected_zeros = torch.tensor(
        [[zero] * module.out_features for _, _, zero in groups], dtype=dtype
    )
    torch.testing.assert_close(module.qzeros, expected_zeros, rtol=0, atol=0)
    assert module.qzeros.dtype == dtype
    assert module.has_zero_points

    codes = unpack_quantized_values_into_int32(
        module.qweight.data, module.weight_type, packed_dim=0
    )
    zeros = module.qzeros.repeat_interleave(module.group_size, dim=0)
    scales = module.scales.repeat_interleave(module.group_size, dim=0)

    # CPU mirror of `transform_A_kblock` (ConvertAndScaleWithZero): upcast the
    # code to the scale dtype, subtract the zero point, then scale.
    dequantized = (codes.to(dtype) - zeros) * scales

    oracle = torch.cat(
        [
            torch.full(
                (module.group_size, module.out_features),
                (code - zero) * scale,
                dtype=dtype,
            )
            for scale, code, zero in groups
        ],
        dim=0,
    )
    assert bool(torch.isfinite(dequantized).all())
    torch.testing.assert_close(dequantized, oracle, rtol=0, atol=0)

    # The fixture discriminates: the old pre-rounded offset arithmetic, which
    # rounded `-zero * scale` into the storage dtype before adding it, misses.
    legacy_offset = torch.tensor(
        [-zero * scale for scale, _, zero in groups], dtype=dtype
    ).repeat_interleave(module.group_size, dim=0).unsqueeze(1)
    legacy = codes.to(dtype) * scales + legacy_offset
    if legacy_differs:
        assert not torch.equal(legacy, oracle)
    else:
        assert torch.equal(legacy, oracle)


def _mainloop_transform_a_kblock() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext"
        / "machete"
        / "machete_mainloop.cuh"
    )
    text = path.read_text(encoding="utf-8")
    start = text.index("CUTLASS_DEVICE void transform_A_kblock(")
    end = text.index("// Modified from upstream", start)
    return text[start:end]


def test_native_mainloop_subtracts_zero_before_scaling():
    """The native kernel must round once, from `(code - zero) * scale`.

    The generated Machete kernels are SM90-only and cannot be built on the
    hosts that run this test, so this pins the operand order the compiler
    would otherwise be the only check on.
    """

    body = _mainloop_transform_a_kblock()
    assert "ConvertAndScaleWithZero" in body
    assert "cute::minus{}" in body
    assert "cute::plus{}" not in body
    assert body.index("cute::minus{}") < body.index("cute::multiplies{}")


def test_zero_point_must_be_exactly_representable():
    """A storage dtype that cannot hold the integer zero point fails closed."""

    scales = torch.ones((1, 3)).to(torch.float8_e4m3fn)
    zeros = torch.tensor([[0, 15, 255]], dtype=torch.int32)
    with pytest.raises(ValueError, match="Zero points cannot be stored exactly"):
        zero_points_in_scale_dtype(scales, zeros)

    exact = zero_points_in_scale_dtype(scales[:, :2], zeros[:, :2])
    assert exact.dtype == torch.float8_e4m3fn
    assert exact.tolist() == [[0.0, 15.0]]
