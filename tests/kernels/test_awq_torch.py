# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch
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

    bias_expected = module.bias.to(dtype=dtype)
    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
    ).to(dtype=dtype)
    expected = torch.matmul(x.to(dtype), dequant_weight)
    expected = expected + bias_expected

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


def test_awq_torch_dequantize_supports_non_half_compute_dtype():
    """The shared dequantize helper keeps its half-weight contract.

    ``ParoLinear`` calls it with the activation dtype, which can be FP32 on the
    dense fallback. The helper must dequantize at a half precision and then
    widen the finished weight instead of raising.
    """

    scales = torch.ones((1, 8), dtype=torch.float32)
    bias = torch.zeros(8, dtype=torch.float32)
    module = _make_unit_awq_module(scales=scales, bias=bias)
    module.post_init()

    weight = module._dequantize_weight_for_inference(
        torch.ones((1, 16), dtype=torch.float32, device=_AWQ_TEST_DEVICE), dtype=torch.float32
    )

    assert weight.dtype is torch.float32
    assert weight.shape == (16, 8)
    assert torch.equal(
        weight, torch.ones((16, 8), dtype=torch.float32, device=_AWQ_TEST_DEVICE)
    )
