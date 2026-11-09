# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU fused GPTQ must preserve the checkpoint's asymmetric zero codes."""

import pytest
import torch
from test_torch_weight_storage import state_tensors

from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization import FORMAT


pytestmark = pytest.mark.cpu


def _layer(group_size: int, desc_act: bool) -> TorchFusedLinear:
    inputs, outputs = 2 * group_size, 32
    layer = TorchFusedLinear(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=desc_act,
        in_features=inputs,
        out_features=outputs,
        bias=False,
        format=FORMAT.GPTQ_V2,
        register_buffers=True,
    )
    rows = torch.arange(inputs)[:, None]
    columns = torch.arange(outputs)[None, :]
    codes = ((3 * rows + 5 * columns + 2) % 16).to(torch.int32)
    zeros = ((5 * torch.arange(2)[:, None] + 3 * columns + 1) % 16).to(torch.int32)
    scales = torch.tensor([.25, .125, .0625])[
        (torch.arange(2)[:, None] + columns) % 3
    ].half()
    for lane in range(8):
        layer.qweight |= (codes[lane::8] & 15) << (4 * lane)
        layer.qzeros |= (zeros[:, lane::8] & 15) << (4 * lane)
    layer.scales.copy_(scales)
    groups = torch.arange(inputs, dtype=torch.int32) // group_size
    if desc_act:
        groups = groups[torch.randperm(inputs, generator=torch.Generator().manual_seed(3740))]
    layer.g_idx.copy_(groups)
    layer.qzero_format(2)
    layer.post_init()
    return layer


def _checkpoint_oracle(state: dict[str, torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack signed int32 checkpoint words in Python, then matmul in FP32."""
    packed_weights = state["qweight"].tolist()
    packed_zeros = state["qzeros"].tolist()
    inputs = len(packed_weights) * 8
    outputs = len(packed_weights[0])
    codes = torch.empty((inputs, outputs), dtype=torch.float32)
    zeros = torch.empty((len(packed_zeros), outputs), dtype=torch.float32)
    for block, words in enumerate(packed_weights):
        for output, signed_word in enumerate(words):
            word = signed_word & 0xFFFFFFFF
            for lane in range(8):
                codes[block * 8 + lane, output] = (word >> (4 * lane)) & 15
    for group, words in enumerate(packed_zeros):
        for block, signed_word in enumerate(words):
            word = signed_word & 0xFFFFFFFF
            for lane in range(8):
                zeros[group, block * 8 + lane] = (word >> (4 * lane)) & 15
    groups = state["g_idx"].long()
    scales = state["scales"].float()
    weight = (codes - zeros.index_select(0, groups)) * scales.index_select(0, groups)
    return x.float() @ weight, zeros


def _assert_packed_only(layer: TorchFusedLinear) -> None:
    inputs, outputs = layer.in_features, layer.out_features
    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.numel() == inputs * outputs // 2
    assert layer.torch_fused_op.qweight_uint8 is layer.qweight
    assert layer.torch_fused_op.scales_and_zeros is layer.scales_and_zeros
    assert not any(
        tensor.ndim == 2 and tensor.shape in ((inputs, outputs), (outputs, inputs))
        for tensor in state_tensors(layer)
    )


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("desc_act", [False, True])
def test_fused_cpu_asymmetric_zero_codes_match_packed_oracle(group_size, dtype, desc_act):
    layer = _layer(group_size, desc_act)
    checkpoint = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    inputs = layer.in_features
    x = torch.stack((
        torch.ones(inputs),
        (torch.arange(inputs).remainder(9).float() - 4) / 16,
    )).to(dtype)
    expected, zero_codes = _checkpoint_oracle(checkpoint, x)

    layer.eval()
    first = layer(x)
    assert str(getattr(layer.linear_mode, "name", layer.linear_mode)).upper() == "INFERENCE"
    assert first.dtype == dtype
    torch.testing.assert_close(first, expected.to(dtype), rtol=2e-3, atol=2e-3)
    expected_offsets = (8 - zero_codes) * checkpoint["scales"].float()
    torch.testing.assert_close(layer.qzeros.float(), expected_offsets, rtol=0, atol=0)
    _assert_packed_only(layer)

    second = layer(x)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    _assert_packed_only(layer)


def test_fused_cpu_asymmetric_qzeros_survive_eval_train_and_cache_clear():
    layer = _layer(32, desc_act=True)
    checkpoint = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    x = torch.ones((2, layer.in_features), dtype=torch.bfloat16)
    expected, _ = _checkpoint_oracle(checkpoint, x)
    layer.eval()
    first = layer(x)
    torch.testing.assert_close(first, expected.to(x.dtype), rtol=2e-3, atol=2e-3)

    layer.clear_weight_cache()
    for name, tensor in checkpoint.items():
        assert torch.equal(layer.state_dict()[name], tensor)
    again = layer(x)
    torch.testing.assert_close(again, first, rtol=0, atol=0)
    _assert_packed_only(layer)

    layer.train()
    for name, tensor in checkpoint.items():
        assert torch.equal(layer.state_dict()[name], tensor)
    assert layer.qzeros.dtype == torch.int32
    x_train = x.clone().requires_grad_(True)
    train_output = layer(x_train)
    torch.testing.assert_close(train_output, expected.to(x.dtype), rtol=2e-3, atol=2e-3)
    train_output.sum().backward()
    assert x_train.grad is not None and torch.isfinite(x_train.grad).all()

    layer.to(dtype=torch.float32)
    assert torch.equal(layer.qzeros, checkpoint["qzeros"])
    assert layer.scales.dtype == torch.float32
    layer.load_state_dict(checkpoint, strict=True)
    assert torch.equal(layer.qzeros, checkpoint["qzeros"])


def test_fused_cpu_promotes_large_affine_offset_before_fp16_rounding():
    layer = _layer(128, desc_act=False)
    layer.qweight.zero_()
    layer.qzeros.zero_()
    layer.scales.fill_(16384)
    checkpoint = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    x = torch.full((1, layer.in_features), 1 / 256, dtype=torch.float16)
    expected, _ = _checkpoint_oracle(checkpoint, x)

    layer.eval()
    actual = layer(x)
    assert layer.scales.dtype == torch.float32
    assert layer.qzeros.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0, atol=0)
    _assert_packed_only(layer)
