# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch
from compile_state_utils import under_compile
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_matches_manual_dequant(dtype):
    if dtype not in AwqTorchLinear.SUPPORTS_DTYPES:
        pytest.skip(f"dtype {dtype} not supported by AwqTorchLinear")
    torch.manual_seed(0)

    bits = 4
    in_features = 32
    out_features = 64
    group_size = 16

    assert out_features % (32 // bits) == 0
    assert in_features % group_size == 0

    groups = in_features // group_size
    pack_cols = out_features

    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, pack_cols), dtype=torch.int32)
    scales = ((torch.rand(groups, pack_cols, dtype=torch.float32) * 2.0) + 0.25).to(dtype)
    bias = torch.randn(out_features, dtype=dtype)

    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=dtype,
        register_buffers=True,
    )

    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales.to(module.scales.dtype))
    module.bias.copy_(bias.to(module.bias.dtype))
    module.post_init()
    module.eval()

    batch = 4
    x = torch.randn(batch, in_features, dtype=dtype)

    output_first = module(x)
    output_second = module(x)

    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
    ).to(dtype=dtype)
    # The biased path keeps the product and the bias in FP32 and rounds once
    # (issue #3977). A low-precision matmul followed by a low-precision add
    # rounds twice and can differ by one output ulp, so use an FP64 oracle.
    expected = (x.double() @ dequant_weight.double() + module.bias.double()).to(dtype)

    assert output_first.dtype == dtype
    assert output_second.dtype == dtype

    atol = 1e-4 if dtype == torch.float32 else 5e-3
    rtol = 1e-4 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(output_first, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(output_second, expected, atol=atol, rtol=rtol)


def test_awq_torch_backend_selection():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.TORCH_AWQ,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is AwqTorchLinear


_AWQ_TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_unit_awq_module(*, scales, bias, device=None):
    """Pack an AWQ layer whose dequantized weight is exactly the scale value.

    ``(code - zero)`` is one for every element, so the dequantized weight
    equals ``scales`` and the independent FP64 oracle is easy to state.
    """

    bits = 4
    group_size = 16
    in_features = 16
    out_features = 8
    device = _AWQ_TEST_DEVICE if device is None else device

    int_weight = torch.ones((in_features, out_features), dtype=torch.int32)
    zero_points = torch.zeros((1, out_features), dtype=torch.int32)
    module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=False,
    )
    module.register_buffer("qweight", _pack_awq_tensor(int_weight, bits).to(device))
    module.register_buffer("qzeros", _pack_awq_tensor(zero_points, bits).to(device))
    module.register_buffer("scales", scales.to(device))
    module.register_buffer("bias", bias.to(device))
    return module


def _awq_oracle(*, module, x, scales, bias):
    """Independent FP64 oracle over the original checkpoint-precision values.

    The packed INT4 words are decoded here rather than by the module's
    dequantizer. ``scales`` and ``bias`` are the values handed to the module,
    not the live buffers: the bug under test corrupts those buffers, and an
    oracle that read them would reproduce the corruption.
    """

    assert module.bits == 4
    order = (0, 2, 4, 6, 1, 3, 5, 7)

    def unpack(packed):
        packed = packed.cpu()
        codes = torch.empty((packed.shape[0], packed.shape[1] * 8), dtype=torch.int32)
        for column in range(packed.shape[1]):
            for lane, source_column in enumerate(order):
                codes[:, column * 8 + source_column] = (packed[:, column] >> (4 * lane)) & 15
        return codes

    codes = unpack(module.qweight).to(torch.float64)
    zeros = unpack(module.qzeros).to(torch.float64)
    group_scales = scales.cpu().to(torch.float64)
    groups = torch.arange(codes.shape[0]) // module.group_size
    weight = (codes - zeros[groups]) * group_scales[groups]
    output = x.cpu().to(torch.float64) @ weight
    return output if bias is None else output + bias.cpu().to(torch.float64)


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_wide_checkpoint_bias_rounds_after_wide_accumulation(activation_dtype):
    """A stored FP32 bias must cancel the reduction before the output is rounded.

    The reduction is ``16 * 4096 = 65536``; the stored bias is ``-65535``, so
    the FP64 oracle is ``1``. Narrowing the bias to the activation dtype turns
    it into ``-inf`` (FP16) or ``-65536`` (BF16) and destroys the result.
    """

    scales = torch.ones((1, 8), dtype=torch.float32)
    bias = torch.full((8,), -65535.0, dtype=torch.float32)
    module = _make_unit_awq_module(scales=scales, bias=bias)
    module.post_init()
    x = torch.full((1, 16), 4096.0, dtype=activation_dtype, device=_AWQ_TEST_DEVICE)
    expected = _awq_oracle(module=module, x=x, scales=scales, bias=bias)

    actual = module(x)

    assert actual.dtype is activation_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected.to(device=actual.device, dtype=activation_dtype), rtol=0, atol=0
    )
    assert module.scales.dtype is torch.float32
    assert module.bias.dtype is torch.float32


@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_pack_keeps_fp32_source_bias(activation_dtype):
    source = torch.nn.Linear(16, 8, bias=True, dtype=torch.float32)
    source.weight.data.fill_(1)
    source.bias.data.fill_(-65535)
    module = AwqTorchLinear(
        bits=4, group_size=16, sym=True, desc_act=False,
        in_features=16, out_features=8, bias=True,
    )
    module.pack(source, torch.ones((8, 1), dtype=torch.float16), torch.zeros((8, 1)))
    module.post_init()
    x = torch.full((1, 16), 4096.0, dtype=activation_dtype)

    actual = module(x)
    expected = _awq_oracle(
        module=module, x=x, scales=torch.ones((1, 8), dtype=torch.float16), bias=source.bias.data
    )
    torch.testing.assert_close(actual, expected.to(activation_dtype), rtol=0, atol=0)
    assert module.bias.dtype is torch.float32


def test_awq_torch_pack_keeps_fp32_source_scales():
    tiny = torch.tensor(1e-40, dtype=torch.bfloat16).float().item()
    source = torch.nn.Linear(16, 8, bias=False, dtype=torch.float32)
    source.weight.data.fill_(tiny)
    scales = torch.full((8, 1), tiny, dtype=torch.float32)
    module = AwqTorchLinear(
        bits=4, group_size=16, sym=True, desc_act=False,
        in_features=16, out_features=8, bias=False,
    )
    module.pack(source, scales, torch.zeros((8, 1)))
    module.post_init()
    stored_scales = module.scales.clone()

    module(torch.ones((1, 16), dtype=torch.float16))
    x_bf16 = torch.ones((1, 16), dtype=torch.bfloat16)
    actual = module(x_bf16)
    expected = _awq_oracle(module=module, x=x_bf16, scales=scales.t(), bias=None)
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
    assert module.scales.dtype is torch.float32
    assert torch.equal(module.scales, scales.t())
    assert torch.equal(module.scales, stored_scales)


def test_awq_torch_forward_preserves_checkpoint_scales_and_bias():
    """Forward must not rewrite stored BF16 precision with the activation dtype.

    ``1e-40`` is representable in BF16 but underflows to zero in FP16, so an
    in-place conversion permanently zeroes the scale and every later BF16
    forward returns zero as well.
    """

    tiny = torch.tensor([[1e-40] * 8], dtype=torch.bfloat16)
    bias = torch.zeros(8, dtype=torch.bfloat16)
    module = _make_unit_awq_module(scales=tiny, bias=bias)
    module.post_init()
    stored_scales = module.scales.clone()
    stored_bias = module.bias.clone()
    x_fp16 = torch.ones((1, 16), dtype=torch.float16, device=_AWQ_TEST_DEVICE)
    x_bf16 = torch.ones((1, 16), dtype=torch.bfloat16, device=_AWQ_TEST_DEVICE)
    expected = _awq_oracle(module=module, x=x_bf16, scales=tiny, bias=bias)

    module(x_fp16)

    actual = module(x_bf16)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(
        actual, expected.to(device=actual.device, dtype=torch.bfloat16), rtol=0, atol=0
    )
    assert module.scales.dtype is torch.bfloat16
    assert torch.equal(module.scales, stored_scales)
    assert torch.equal(module.bias, stored_bias)
    # The dequantized [in_features, out_features] weight is call-local.
    retained = list(vars(module).values()) + list(module.buffers()) + list(module.parameters())
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (16, 8)
        for value in retained
    )


@pytest.mark.parametrize("checkpoint_dtype", [torch.float32, torch.bfloat16])
def test_awq_torch_loader_preserves_checkpoint_operand_dtypes(tmp_path, checkpoint_dtype):
    """The model loader must not narrow AWQ buffers before forward sees them."""

    if checkpoint_dtype is torch.float32:
        scales = torch.ones((1, 8), dtype=checkpoint_dtype)
        bias = torch.full((8,), -65535.0, dtype=checkpoint_dtype)
        x = torch.full((1, 16), 4096.0, dtype=torch.float16)
    else:
        scales = torch.full((1, 8), 1e-40, dtype=checkpoint_dtype)
        bias = torch.zeros(8, dtype=checkpoint_dtype)
        x = torch.ones((1, 16), dtype=torch.bfloat16)

    source = _make_unit_awq_module(scales=scales, bias=bias, device=torch.device("cpu"))
    expected = _awq_oracle(module=source, x=x, scales=scales, bias=bias).to(
        device=_AWQ_TEST_DEVICE, dtype=x.dtype
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = AwqTorchLinear(
                bits=4,
                group_size=16,
                sym=True,
                desc_act=False,
                in_features=16,
                out_features=8,
                bias=True,
                register_buffers=True,
            )

        def tie_weights(self, **_kwargs):
            pass

    model = Model()
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {f"layer.{name}": tensor.contiguous() for name, tensor in source.state_dict().items()},
        str(checkpoint),
        metadata={"format": "pt"},
    )
    load_checkpoint_in_model_then_tie_weights(
        model, checkpoint=str(checkpoint), device_map={"": str(_AWQ_TEST_DEVICE)}, dtype=torch.float16
    )

    assert model.layer.scales.dtype is checkpoint_dtype
    assert model.layer.bias.dtype is checkpoint_dtype
    assert torch.equal(model.layer.scales.cpu(), scales)
    assert torch.equal(model.layer.bias.cpu(), bias)
    torch.testing.assert_close(model.layer(x.to(_AWQ_TEST_DEVICE)), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1.0, 1e-40, 65536.0])
def test_awq_torch_dequantize_supports_non_half_compute_dtype(scale):
    """Paro's FP32 dense fallback must retain FP32 checkpoint scales."""

    scales = torch.full((1, 8), scale, dtype=torch.float32)
    bias = torch.zeros(8, dtype=torch.float32)
    module = _make_unit_awq_module(scales=scales, bias=bias)
    module.post_init()

    weight = module._dequantize_weight_for_inference(
        torch.ones((1, 16), dtype=torch.float32, device=_AWQ_TEST_DEVICE), dtype=torch.float32
    )

    assert weight.dtype is torch.float32
    assert weight.shape == (16, 8)
    assert torch.equal(
        weight, torch.full((16, 8), scale, dtype=torch.float32, device=_AWQ_TEST_DEVICE)
    )


# --- AWQ decode-range contract ------------------------------------------------
#
# A portable AWQ checkpoint stores one scale per group and integer codes, so a
# decoded weight is ``(code - zero) * scale``. Decoding that grid into the
# activation dtype before the product turns a large scale into ``inf`` and a
# tiny scale into ``0``; no later FP32 accumulation can recover either. The
# tests below pack known codes and bytes, then compare with an independent FP64
# oracle that decodes the words itself from the source scale value.

_AWQ_DEVICES = [torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)
_AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_layer(*, scale, code, zero, bits=4, in_features=128, out_features=128,
                group_size=128, scale_dtype=torch.float32, device=None):
    """Build an ``AwqTorchLinear`` whose codes and zero points are constant."""

    device = torch.device("cpu") if device is None else device
    codes = torch.full((in_features, out_features), code, dtype=torch.int32)
    zeros = torch.full((in_features // group_size, out_features), zero, dtype=torch.int32)
    module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.float16,
    )
    module.register_buffer("qweight", _pack_awq_tensor(codes, bits).to(device))
    module.register_buffer("qzeros", _pack_awq_tensor(zeros, bits).to(device))
    module.register_buffer(
        "scales",
        torch.full((in_features // group_size, out_features), scale, dtype=scale_dtype).to(device),
    )
    module.bias = None
    module.post_init()
    return module.to(device).eval()


def _decode_oracle(*, module, x, scale, code, zero):
    """Decode the packed words and accumulate in FP64, independently.

    ``scale`` is the checkpoint value, never ``module.scales``: the oracle must
    not inherit a conversion the implementation applies.
    """

    order = _AWQ_PACK_ORDER

    def unpack(packed):
        packed = packed.detach().cpu().to(torch.int64) & 0xFFFFFFFF
        codes = torch.empty((packed.shape[0], packed.shape[1] * 8), dtype=torch.float64)
        for column in range(packed.shape[1]):
            word = packed[:, column]
            for lane, source_column in enumerate(order):
                codes[:, column * 8 + source_column] = (word >> (4 * lane)) & 15
        return codes

    codes = unpack(module.qweight)
    zeros = unpack(module.qzeros)
    assert torch.equal(codes, torch.full_like(codes, float(code)))
    assert torch.equal(zeros, torch.full_like(zeros, float(zero)))

    groups = torch.arange(module.in_features) // module.group_size
    group_scales = torch.full(
        (module.in_features // module.group_size, module.out_features),
        float(scale),
        dtype=torch.float64,
    )
    weight = (codes - zeros[groups]) * group_scales[groups]
    x64 = x.detach().cpu().to(torch.float64)
    return x64 @ weight


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
@pytest.mark.parametrize(
    ("label", "scale", "code", "zero", "value", "expected"),
    [
        # 7 * 10000 = 70000 exceeds the FP16 maximum; the FP16 decode is `inf`
        # for every one of the 16384 weights.
        ("weight-overflow", 10000.0, 7, 0, 0.001, 8963.623046875),
        # 1e-8 is below the FP16 subnormal range; the FP16 decode is `0` for
        # every weight, so 60000 activations produce zero.
        ("scale-underflow", 1e-8, 1, 0, 60000.0, 0.07680000000000024),
    ],
)
def test_awq_torch_decode_range_matches_fp64_oracle(
    device, label, scale, code, zero, value, expected
):
    module = _pack_layer(scale=scale, code=code, zero=zero, device=device)
    x = torch.full((1, 128), value, dtype=torch.float16, device=device)

    actual = module(x)
    reference = _decode_oracle(module=module, x=x, scale=scale, code=code, zero=zero)

    assert actual.dtype is torch.float16
    assert actual.device == x.device
    assert torch.isfinite(actual).all(), label
    assert actual[0, 0].item() != 0.0, label
    # Only the final FP16 output rounding may differ from the FP64 reference.
    torch.testing.assert_close(
        actual, reference.to(device=actual.device, dtype=actual.dtype), rtol=0, atol=0
    )
    assert reference[0, 0].item() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_decode_range_keeps_checkpoint_scale_precision(device):
    """Each supported checkpoint scale dtype decodes against the same oracle."""

    in_features = 128
    code, zero = 3, 2
    # Codes (3 - 2) = 1, so the weight equals the stored scale exactly.
    scale = 0.5
    x = torch.linspace(-1.0, 1.0, in_features, dtype=torch.float16).reshape(1, -1).to(device)

    for scale_dtype in (torch.float16, torch.bfloat16, torch.float32):
        module = _pack_layer(
            scale=scale, code=code, zero=zero, scale_dtype=scale_dtype, device=device
        )
        stored = module.scales.clone()
        qweight = module.qweight.clone()
        qzeros = module.qzeros.clone()

        actual = module(x)
        reference = _decode_oracle(
            module=module, x=x, scale=stored.float().max().item(), code=code, zero=zero
        )

        assert actual.dtype is torch.float16
        torch.testing.assert_close(
            actual, reference.to(device=device, dtype=torch.float16), rtol=0, atol=0
        )
        assert module.scales.dtype is scale_dtype
        assert torch.equal(module.scales, stored)
        assert torch.equal(module.qweight, qweight)
        assert torch.equal(module.qzeros, qzeros)


def _fp16_boundary_scales():
    """FP16 decode boundary and one representable FP16 step on each side.

    ``span`` is the largest INT4 code. A scale is range-safe when its FP16 cast
    is nonzero and ``span * scale`` is still finite. The neighbours are exact
    FP32 source values so a source that underflows to zero stays
    distinguishable from a checkpoint that stores zero.
    """

    span = (1 << 4) - 1
    info = torch.finfo(torch.float16)
    # Largest FP16 scale whose product with the largest code stays finite.
    upper_fit = float(torch.tensor(info.max / span, dtype=torch.float16))
    while upper_fit * span > info.max:
        upper_fit = float(torch.nextafter(
            torch.tensor(upper_fit, dtype=torch.float16), torch.tensor(0.0, dtype=torch.float16)
        ))
    upper_below = float(torch.nextafter(
        torch.tensor(upper_fit, dtype=torch.float16), torch.tensor(0.0, dtype=torch.float16)
    ))
    upper_above = float(torch.nextafter(
        torch.tensor(upper_fit, dtype=torch.float16), torch.tensor(torch.inf, dtype=torch.float16)
    ))
    lower_fit = info.smallest_normal * info.eps  # 2**-24, smallest subnormal
    return [
        ("below-upper", upper_below, True),
        ("at-upper", upper_fit, True),
        ("above-upper", upper_above, False),
        ("above-lower", lower_fit * 2, True),
        ("at-lower", lower_fit, True),
        ("below-lower", lower_fit / 2, False),
    ]


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
@pytest.mark.parametrize(("label", "scale", "fits_fp16"), _fp16_boundary_scales())
def test_awq_torch_decode_range_threshold_neighbors(device, label, scale, fits_fp16):
    """The decode dtype switches exactly at the FP16 range boundary."""

    assert scale != 0.0
    # Code 15 with zero 0 is the largest representable magnitude.
    module = _pack_layer(
        scale=scale, code=15, zero=0, scale_dtype=torch.float32, device=device
    )
    assert module._decode_dtype_for_activation(torch.float16) is (
        torch.float16 if fits_fp16 else torch.float32
    )
    # BF16 keeps the FP32 exponent range, so every FP16 boundary case stays in
    # the activation dtype there.
    assert module._decode_dtype_for_activation(torch.bfloat16) is torch.bfloat16

    # The input keeps the reduction inside the FP16 output range; the test
    # is about decoded weights, not about an overflowing visible sum.
    x = torch.full((1, 128), 0.001, dtype=torch.float16, device=device)
    actual = module(x)
    reference = _decode_oracle(module=module, x=x, scale=scale, code=15, zero=0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, reference.to(device=device, dtype=torch.float16), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_decode_range_both_activation_dtypes(device, activation_dtype):
    """The underflow case must keep a nonzero weight for both dtypes."""

    scale, code, zero, value = 1e-8, 1, 0, 60000.0
    module = _pack_layer(scale=scale, code=code, zero=zero, device=device)
    x = torch.full((1, 128), value, dtype=activation_dtype, device=device)

    actual = module(x)
    reference = _decode_oracle(module=module, x=x, scale=scale, code=code, zero=zero)

    assert actual.dtype is activation_dtype
    assert actual[0, 0].item() != 0.0
    torch.testing.assert_close(
        actual, reference.to(device=device, dtype=activation_dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_decode_range_is_call_local(device):
    """The wide decode must not leave a dense weight on the module."""

    in_features = out_features = 128
    module = _pack_layer(scale=10000.0, code=7, zero=0, device=device)
    x = torch.full((2, in_features), 0.001, dtype=torch.float16, device=device)

    with torch.inference_mode():
        first = module(x)
        second = module(x)

    assert torch.equal(first, second)
    assert first[0, 0] == 8960.0
    retained = list(vars(module).values()) + list(module.buffers()) + list(module.parameters())
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (in_features, out_features)
        for value in retained
    )
    # The decision follows the live scales: replacing them and clearing the
    # derived cache selects the next decode without a stale result.
    assert module._decode_dtype_for_activation(torch.float16) is torch.float32
    module.scales = torch.full_like(module.scales, 0.5)
    module.clear_weight_cache()
    assert module._decode_dtype_for_activation(torch.float16) is torch.float16
    # A stale decision must not select the narrow decode for an overflowing
    # checkpoint: the key covers identity, version, device, dtype, and shape.
    module.scales.data.copy_(torch.full_like(module.scales, 1e4))
    module.clear_weight_cache()
    assert module._decode_dtype_for_activation(torch.float16) is torch.float32


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_decode_range_bound_is_conservative(device):
    """An in-range code grid may still take the wide decode."""

    # The bound is 15 * 10000, so FP16 is rejected even though every code here
    # is 1 and the actual weight, 10000, is inside the FP16 finite range.
    module = _pack_layer(scale=10000.0, code=1, zero=0, device=device)
    assert module._decode_dtype_for_activation(torch.float16) is torch.float32

    x = torch.ones((1, 128), dtype=torch.float16, device=device)
    reference = _decode_oracle(module=module, x=x, scale=10000.0, code=1, zero=0)
    torch.testing.assert_close(
        module(x), reference.to(device=device, dtype=torch.float16), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_decode_range_widens_past_fp32(device):
    """A BF16 scale beyond the FP32 weight range decodes in FP64."""

    # 15 * 1e38 overflows FP32 but is representable in FP64; a BF16 activation
    # would narrow the weight to `inf` and an FP32 decode would do the same.
    scale = 1e38
    module = _pack_layer(
        scale=scale, code=1, zero=0, scale_dtype=torch.bfloat16, device=device
    )
    assert module._decode_dtype_for_activation(torch.bfloat16) is torch.float64

    # A single 1.0 activation keeps the visible sum at 1e38, which BF16 still
    # holds; the check is that the decoded weight was not narrowed to `inf`.
    x = torch.zeros((1, 128), dtype=torch.bfloat16, device=device)
    x[0, 0] = 1.0
    actual = module(x)
    assert torch.isfinite(actual).all()
    assert actual[0, 0].item() == float(torch.tensor(scale, dtype=torch.bfloat16))


def _under_compile(fn):
    """Run ``fn`` while ``dynamo_is_compiling()`` reports ``True``.

    The compiled region cannot read the scale tensor on the host, so this is
    the branch a dynamo trace takes.
    """

    with under_compile():
        return fn()


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
@pytest.mark.parametrize(
    ("label", "scale", "code", "zero"),
    [
        ("fits", 0.001, 3, 2),
        ("weight-overflow", 10000.0, 7, 0),
        ("scale-underflow", 1e-8, 1, 0),
    ],
)
def test_awq_torch_compiled_decode_dtype_matches_eager(device, label, scale, code, zero):
    """A compiled graph must choose the same decode dtype as eager (#4794).

    #4737 moved the compile guard in front of the cache key and returned
    ``False`` for every dtype, so ``_decode_dtype_for_activation`` fell through
    to FP64 inside every compiled graph. ``linear_wide_bias`` then promoted the
    product too, running the whole GEMM in FP64.
    """

    module = _pack_layer(scale=scale, code=code, zero=zero, device=device)
    for activation_dtype in (torch.float16, torch.bfloat16):
        eager = module._decode_dtype_for_activation(activation_dtype)
        compiled = _under_compile(
            lambda dtype=activation_dtype: module._decode_dtype_for_activation(dtype)
        )
        assert compiled is eager, label
        assert compiled is not torch.float64, label


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_compiled_decode_keeps_fp64_past_fp32(device):
    """A BF16 scale beyond the FP32 range must still decode in FP64 (#4794).

    The narrow decode is now decided outside the compiled region, so the
    pathological widening that `_decode_dtype_for_activation` already
    documents must survive the change.
    """

    module = _pack_layer(
        scale=1e38, code=1, zero=0, scale_dtype=torch.bfloat16, device=device
    )
    compiled = _under_compile(
        lambda: module._decode_dtype_for_activation(torch.bfloat16)
    )
    assert compiled is torch.float64


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_compiled_decode_follows_the_scales(device):
    """The compiled answer is recomputed when the scales change (#4794)."""

    module = _pack_layer(scale=10000.0, code=7, zero=0, device=device)
    assert (
        _under_compile(lambda: module._decode_dtype_for_activation(torch.float16))
        is torch.float32
    )

    module.scales = torch.full_like(module.scales, 0.5)
    module.clear_weight_cache()
    assert (
        _under_compile(lambda: module._decode_dtype_for_activation(torch.float16))
        is torch.float16
    )


@pytest.mark.parametrize("device", _AWQ_DEVICES, ids=str)
def test_awq_torch_compiled_forward_is_bitwise_eager(device):
    """The traced forward must not silently widen the decode (#4794).

    A real ``torch.compile`` trace takes the compiled branch, so a decode that
    escalated to FP64 changed the visible result (up to 0.03 on the reporter's
    checkpoint) and cost the FP64 GEMM.
    """

    module = _pack_layer(scale=0.001, code=3, zero=2, device=device)
    x = torch.randn(1, 8, 128, dtype=torch.float16, device=device)

    with torch.inference_mode():
        eager = module(x)

    compiled = torch.compile(module, backend="eager", fullgraph=True)
    with torch.inference_mode():
        actual = compiled(x)

    assert actual.dtype is torch.float16
    assert torch.equal(actual, eager)
