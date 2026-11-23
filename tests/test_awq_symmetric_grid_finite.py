# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The exported symmetric AWQ grid must be finite in the decode dtype.

A symmetric group decodes code ``c`` as ``c * stored_scale``.  ``amax/max_code``
rounds to the *nearest* representable scale, so a finite maximum weight can land
on a top code whose product is not: FP16 ``65504 / 7`` rounds up to 9360 and
``7 * 9360 == 65520`` is the value FP16 rounds to ``inf``.  The staging clamp
then rewrote the staged value to 65504, but the packer kept code 15 and the
checkpoint decoded ``7 * 9360``.  Staging and storage must name one grid point.
"""

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm, unpack_awq
from gptqmodel.quantization.config import FORMAT, AWQConfig
from gptqmodel.utils.packing import hf_pack_layer


pytestmark = [pytest.mark.cpu]

BITS = 4
GROUP_SIZE = 32
MAX_INT = 2 ** (BITS - 1) - 1
MIN_INT = -(2 ** (BITS - 1))
IN_FEATURES = 32
OUT_FEATURES = 32

FP16_MAX = float(torch.finfo(torch.float16).max)
# The largest FP16 scale whose top code stays finite: `nextafter(65504/7)`.
FP16_TOP_SCALE = float(
    torch.nextafter(torch.tensor(FP16_MAX / MAX_INT, dtype=torch.float16), torch.tensor(0.0, dtype=torch.float16))
)
# What the kernel reads back for the top code. `7 * 9352` is 65464 exactly, but
# the FP16 product rounds to the adjacent representable value.
FP16_TOP_DECODE = float(torch.tensor(FP16_TOP_SCALE * MAX_INT, dtype=torch.float16))

GEMM_KERNELS = (AwqGEMMLinear, AwqGEMVLinear)


class _Stub:
    """Call the processor's public staging path without a full quantizer."""

    def __init__(self, fmt: FORMAT, sym: bool = True) -> None:
        self.qcfg = AWQConfig(bits=BITS, group_size=GROUP_SIZE, sym=sym, format=fmt)

    pseudo_quantize_tensor = AWQProcessor.pseudo_quantize_tensor
    _pseudo_quantize_tensor_into = AWQProcessor._pseudo_quantize_tensor_into


def _make_module(cls) -> nn.Module:
    """Build just enough of a native kernel to exercise CPU packing."""

    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.bits = BITS
    module.group_size = GROUP_SIZE
    module.in_features = IN_FEATURES
    module.out_features = OUT_FEATURES
    module.bias = None
    module.pack_dtype = torch.int32
    module.pack_factor = 32 // BITS
    module.adapter = None
    module.zeros_name = "qzeros"
    return module


def _constant_weight(value: float) -> torch.Tensor:
    return torch.full((OUT_FEATURES, IN_FEATURES), value, dtype=torch.float16)


def _gemv_decoded(module: nn.Module) -> torch.Tensor:
    """Decode the GEMV layout the way its portable kernel does."""

    codes = _gemv_codes(module)
    zeros = (module.qzeros.to(torch.int32) >> 0) & ((1 << BITS) - 1)
    group_index = torch.arange(IN_FEATURES) // GROUP_SIZE
    groups = IN_FEATURES // GROUP_SIZE
    zero = zeros[:, :groups].index_select(1, group_index)
    scale = module.scales[:, :groups].index_select(1, group_index)
    weight = (codes.to(torch.float32) - zero.to(torch.float32)) * scale.to(torch.float32)
    return weight.t().contiguous()


def _gemv_codes(module: nn.Module) -> torch.Tensor:
    """The GEMV layout stores natural-order nibbles: column ``c`` is code ``c``."""

    qweight = module.qweight.to(torch.int32)
    shifts = torch.arange(module.pack_factor, dtype=torch.int32) * BITS
    mask = (1 << BITS) - 1
    return ((qweight[:, :, None] >> shifts) & mask).reshape(OUT_FEATURES, IN_FEATURES)


def _pack(fmt: FORMAT, cls, value: float):
    """Stage, pack, then read the checkpoint back through the CPU decoder."""

    source = _constant_weight(value)
    stub = _Stub(fmt)
    staged, scales, zeros = stub.pseudo_quantize_tensor(source.clone())

    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(staged)
    module = _make_module(cls)
    hf_pack_layer(module, linear, scales.contiguous(), zeros.contiguous())

    if cls is AwqGEMVLinear:
        return staged, module, _gemv_decoded(module), _gemv_codes(module)
    decoded = dequantize_gemm(module.qweight, module.qzeros, module.scales, BITS, GROUP_SIZE)
    codes, _ = unpack_awq(module.qweight, None, BITS, count=IN_FEATURES)
    return staged, module, decoded, codes


@pytest.mark.parametrize("cls", GEMM_KERNELS)
@pytest.mark.parametrize(
    "value",
    [65504.0, -65504.0, 65472.0, 60000.0],
    ids=["fp16-max", "fp16-max-negative", "just-below-midpoint", "ordinary-large"],
)
def test_symmetric_export_grid_decodes_finitely(cls, value):
    """Pseudo-quantize, pack and unpack must agree on a finite grid point."""

    staged, module, decoded, codes = _pack(FORMAT.GEMM, cls, value)

    stored_scale = module.scales.reshape(-1)[0]
    assert torch.isfinite(stored_scale)
    # The grid point the checkpoint names is `scale * signed_code`; it must be
    # finite in both the staging width and the decoder's width, and the staged
    # value must be that same point rounded into the staging dtype.
    assert torch.isfinite(staged).all()
    assert torch.isfinite(decoded).all()
    # The fixture is uniform, so every lane is the same signed code either way.
    signed_codes = codes.to(torch.int32) - 2 ** (BITS - 1)
    assert signed_codes.unique().numel() == 1
    grid_point = stored_scale.item() * int(signed_codes[0, 0])
    torch.testing.assert_close(
        staged, torch.full_like(staged, torch.tensor(grid_point, dtype=torch.float16))
    )
    torch.testing.assert_close(
        decoded, torch.full_like(decoded, torch.tensor(grid_point, dtype=decoded.dtype))
    )


@pytest.mark.parametrize("cls", GEMM_KERNELS)
def test_fp16_max_keeps_the_representable_top_code(cls):
    """65504 must export as `9352 * 7`, not the overflowing `9360 * 7`."""

    _, module, decoded, codes = _pack(FORMAT.GEMM, cls, FP16_MAX)

    assert module.scales.reshape(-1)[0].item() == FP16_TOP_SCALE
    assert int(codes.to(torch.int32).max()) == 2 ** (BITS - 1) + MAX_INT
    if cls is AwqGEMMLinear:
        # 0x88888888 is the symmetric zero point (8) in every 4-bit lane.
        assert int(module.qzeros[0, 0]) & 0xFFFFFFFF == 0x88888888
    else:
        # The GEMV layout pads the group axis to whole words.
        assert int(module.qzeros[0, 0]) & 0xF == 2 ** (BITS - 1)
    # The grid point is 65464. GEMM multiplies in FP16 and rounds to 65472;
    # the portable GEMV decoder multiplies in FP32 and keeps 65464 exact. Both
    # are finite, which is the invariant.
    expected = torch.tensor(FP16_TOP_SCALE * MAX_INT, dtype=decoded.dtype)
    assert decoded[0, 0].item() == expected.item()
    assert decoded.isfinite().all()


@pytest.mark.parametrize("value", [65504.0, -65504.0], ids=["positive", "negative"])
def test_fp16_max_negative_endpoint_is_finite(value):
    """The negative endpoint uses code 1 (=-7) and must stay finite too."""

    _, module, decoded, codes = _pack(FORMAT.GEMM, AwqGEMMLinear, value)

    assert module.scales.reshape(-1)[0].item() == FP16_TOP_SCALE
    expected_code = 1 if value < 0 else 2 ** (BITS - 1) + MAX_INT
    if value < 0:
        assert int(codes.to(torch.int32).min()) == expected_code
    else:
        assert int(codes.to(torch.int32).max()) == expected_code
    # The grid fits `amax` to the top code, so a full-scale negative weight
    # lands on signed code -MAX_INT (checkpoint code 1), not on -8.
    grid_point = FP16_TOP_SCALE * (-MAX_INT if value < 0 else MAX_INT)
    torch.testing.assert_close(
        decoded,
        torch.full_like(decoded, torch.tensor(grid_point, dtype=decoded.dtype)),
    )


def test_gemv_fast_rejects_the_offset_rather_than_writing_a_corrupt_grid():
    """The offset layouts must fail closed instead of storing a broken offset."""

    source = _constant_weight(FP16_MAX)
    stub = _Stub(FORMAT.GEMV_FAST)
    staged, scales, zeros = stub.pseudo_quantize_tensor(source.clone())

    assert torch.isfinite(staged).all()
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(staged)
    module = _make_module(AwqGEMVFastLinear)

    with pytest.raises(ValueError, match="Scaled zero-point offset is not finite"):
        hf_pack_layer(module, linear, scales.contiguous(), zeros.contiguous())


@pytest.mark.parametrize("cls", (AwqGEMMLinear, AwqGEMVLinear))
def test_in_place_candidate_search_uses_the_exported_grid(cls):
    """The clip search must score the grid the packer will store."""

    source = _constant_weight(FP16_MAX)
    stub = _Stub(FORMAT.GEMM)
    dst = torch.empty_like(source)
    stub._pseudo_quantize_tensor_into(source.clone(), dst)

    staged, _, _ = stub.pseudo_quantize_tensor(source.clone())
    torch.testing.assert_close(dst, staged, rtol=0, atol=0)
    assert torch.isfinite(dst).all()
    assert dst[0, 0].item() == FP16_TOP_DECODE


def test_ordinary_range_grid_is_bit_identical_to_the_previous_scale():
    """A representable group keeps its exact `amax / max_code` grid."""

    generator = torch.Generator().manual_seed(2026)
    weights = (torch.randn(OUT_FEATURES, IN_FEATURES, generator=generator) * 0.2).half()
    staged, scales, _ = _Stub(FORMAT.GEMM).pseudo_quantize_tensor(weights.clone())

    # The control arithmetic is the pre-fix formula, computed independently.
    amplitude = weights.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    expected_scales = (amplitude / MAX_INT).to(weights.dtype)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    assert torch.isfinite(staged).all()
    assert (staged.float() - weights.float()).abs().max() <= scales.max().item()


def test_fp32_source_uses_the_narrower_storage_width():
    """The repair follows the checkpoint width, not the source weight width.

    A FP32 source weight keeps its amplitude, but the GEMM packer stores FP16
    scales and the kernel decodes in FP16, so the bound is the FP16 one.
    """

    source = torch.full((OUT_FEATURES, IN_FEATURES), FP16_MAX, dtype=torch.float32)
    staged, scales, _ = _Stub(FORMAT.GEMM).pseudo_quantize_tensor(source.clone())

    assert scales.dtype == torch.float32
    assert scales.reshape(-1)[0].item() == FP16_TOP_SCALE
    assert torch.isfinite(staged).all()
    # `7 * 9352` is finite in FP32 as well, so the staged grid survives the
    # source-width tensor; the point is that the scale came from the FP16 bound.
    assert staged.max().item() == FP16_TOP_SCALE * MAX_INT


@pytest.mark.parametrize("value", [2.0**-22, 2.0**-24, 2.0**-26], ids=["normal", "min-subnormal", "below-subnormal"])
def test_subnormal_positive_scales_do_not_trip_the_staging_guard(value):
    """A positive amplitude that narrows to a subnormal scale stays finite."""

    staged, scales, _ = _Stub(FORMAT.GEMM).pseudo_quantize_tensor(_constant_weight(value))

    assert torch.isfinite(staged).all()
    assert torch.isfinite(scales).all()
    assert staged.abs().max() <= torch.finfo(torch.float16).max


# --------------------------------------------------------------------------
# Native kernels
# --------------------------------------------------------------------------
# The staged grid is only useful if the kernel that reads the checkpoint also
# stays finite.  These nodes are skipped on hosts without the corresponding
# native runtime; the CPU nodes above already pin staging, packing and the
# portable decoders on every host.

_NATIVE_IN = 64
_NATIVE_OUT = 64
_NATIVE_GROUP = 64


def _native_constant(value: float) -> torch.Tensor:
    return torch.full((_NATIVE_OUT, _NATIVE_IN), value, dtype=torch.float16)


def _native_staged(value: float, fmt: FORMAT):
    stub = _Stub(fmt)
    return stub.pseudo_quantize_tensor(_native_constant(value))


def _packing_stub(cls, fmt: FORMAT) -> nn.Module:
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.bits = BITS
    module.group_size = _NATIVE_GROUP
    module.sym = True
    module.in_features = _NATIVE_IN
    module.out_features = _NATIVE_OUT
    module.bias = None
    module.pack_dtype = torch.int32
    module.pack_factor = 32 // BITS
    module.adapter = None
    module.zeros_name = "qzeros"
    module.device = torch.device("cpu")
    return module


def _native_case_output(module, staged):
    """Feed a single unit input so the output is exactly one decoded weight."""

    x = torch.zeros(1, 1, _NATIVE_IN, dtype=torch.float16, device="cuda")
    x[0, 0, 0] = 1.0
    with torch.inference_mode():
        out = module(x)
    return out.reshape(-1)[0].item(), bool(out.isfinite().all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native AWQ kernel requires an accelerator")
@pytest.mark.parametrize("value", [65504.0, -65504.0])
def test_native_gemm_decodes_the_finite_grid(value):
    """The CUDA GEMM dequantizer must read back the staged grid, not `inf`."""

    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
    from gptqmodel.utils.awq import awq_dequantize_weights, awq_runtime_available

    if not awq_runtime_available():
        pytest.skip("native AWQ GEMM extension unavailable on this host")

    staged, scales, zeros = _native_staged(value, FORMAT.GEMM)
    source = nn.Linear(_NATIVE_IN, _NATIVE_OUT, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(staged)
    module = _packing_stub(AwqGEMMLinear, FORMAT.GEMM)
    hf_pack_layer(module, source, scales.contiguous(), zeros.contiguous())
    module = module.cuda().half()

    with torch.inference_mode():
        decoded = awq_dequantize_weights(
            module.qweight, module.scales.half(), module.qzeros, 0, 0, 0, False
        ).to(torch.float32)

    assert decoded.isfinite().all()
    assert decoded[0, 0].item() == float(staged[0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native AWQ kernel requires an accelerator")
@pytest.mark.parametrize("value", [65504.0, -65504.0])
def test_native_gemv_decodes_the_finite_grid(value):
    """The GEMV decode (native on CUDA, Triton on ROCm) must stay finite."""

    from gptqmodel.nn_modules.qlinear.gemv_awq import (
        AwqGEMVLinear,
        AwqGEMVTritonLinear,
    )
    from gptqmodel.utils.rocm import IS_ROCM

    staged, scales, zeros = _native_staged(value, FORMAT.GEMV)
    source = nn.Linear(_NATIVE_IN, _NATIVE_OUT, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(staged)
    module = _packing_stub(AwqGEMVLinear, FORMAT.GEMV)
    hf_pack_layer(module, source, scales.contiguous(), zeros.contiguous())
    if IS_ROCM:
        # The native binary is CUDA-only; the Triton kernel reads the same layout.
        triton = _packing_stub(AwqGEMVTritonLinear, FORMAT.GEMV)
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            if hasattr(module, name):
                triton.register_buffer(name, getattr(module, name).clone())
        module = triton
    module = module.cuda().half()

    got, finite = _native_case_output(module, staged)
    assert finite
    assert got == float(staged[0, 0])
