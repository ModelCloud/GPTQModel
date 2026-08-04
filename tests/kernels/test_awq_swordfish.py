# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.swordfish import AwqSwordfishLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.swordfish import (
    _validate_swordfish_device_support,
    prewarm_swordfish_extension,
    swordfish_runtime_error,
)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for lane, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (lane * bits)
    return packed


def _mock_awq_module_tensors(
    *,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = in_features // group_size
    maxq = (1 << bits) - 1

    float_weight = torch.randn(in_features, out_features, dtype=torch.float32) * 0.2
    weight_groups = float_weight.view(groups, group_size, out_features)

    w_min = weight_groups.amin(dim=1)
    w_max = weight_groups.amax(dim=1)
    scales = ((w_max - w_min).clamp_min(1e-6) / maxq).to(dtype)
    zero_points = torch.round((-w_min / scales.to(torch.float32))).clamp_(0, maxq).to(torch.int32)
    quantized = torch.round(
        weight_groups / scales.to(torch.float32).unsqueeze(1) + zero_points.unsqueeze(1)
    ).clamp_(0, maxq).to(torch.int32)

    qweight = _pack_awq_tensor(quantized.view(in_features, out_features), bits)
    qzeros = _pack_awq_tensor(zero_points, bits)
    bias = torch.randn(out_features, dtype=dtype)

    return qweight, qzeros, scales, bias


def _build_awq_module(
    module_cls,
    *,
    device: torch.device,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    dtype: torch.dtype,
):
    module = module_cls(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        dtype=dtype,
    ).to(device)

    with torch.no_grad():
        module.qweight.copy_(qweight.to(device))
        module.qzeros.copy_(qzeros.to(device))
        module.scales.copy_(scales.to(dtype).to(device))
        module.bias.copy_(bias.to(dtype).to(device))

    module.post_init()
    module.eval()
    return module


def _assert_awq_matches_torch(
    *,
    device: torch.device,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    torch.manual_seed(11)
    qweight, qzeros, scales, bias = _mock_awq_module_tensors(
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
    )

    baseline = _build_awq_module(
        AwqTorchLinear,
        device=device,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bias=bias,
        dtype=dtype,
    )
    candidate = _build_awq_module(
        AwqSwordfishLinear,
        device=device,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bias=bias,
        dtype=dtype,
    )

    for m in (1, 8, 64):
        x = torch.randn((m, in_features), device=device, dtype=dtype)
        with torch.inference_mode():
            expected = baseline(x)
            actual = candidate(x)
            repeat = candidate(x)
        torch.cuda.synchronize(device)

        assert candidate.qzeros.numel() > 0
        diff = (actual - expected).abs()
        max_err = diff.max().item()
        mae = diff.mean().item()
        assert max_err < atol, f"m={m}: max error {max_err} exceeds {atol}"
        assert mae < rtol, f"m={m}: MAE {mae} exceeds {rtol}"
    torch.testing.assert_close(repeat, expected, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_swordfish_cuda_matches_torch_awq_128():
    if not _validate_swordfish_device_support():
        pytest.skip(swordfish_runtime_error())
    prewarm_swordfish_extension()

    _assert_awq_matches_torch(
        device=torch.device("cuda:0"),
        bits=4,
        group_size=128,
        in_features=256,
        out_features=128,
        dtype=torch.bfloat16,
        atol=0.15,
        rtol=0.05,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_swordfish_cuda_matches_torch_awq_64():
    if not _validate_swordfish_device_support():
        pytest.skip(swordfish_runtime_error())
    prewarm_swordfish_extension()

    _assert_awq_matches_torch(
        device=torch.device("cuda:0"),
        bits=4,
        group_size=64,
        in_features=256,
        out_features=128,
        dtype=torch.bfloat16,
        atol=0.15,
        rtol=0.05,
    )
