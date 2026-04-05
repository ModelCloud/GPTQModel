# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.

from __future__ import annotations

import os

import pytest
import torch


# Keep this CPU test isolated from optional BitBLAS/TVM import side effects.
os.environ.setdefault("GPTQMODEL_DISABLE_BITBLAS", "1")

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_int8_awq import TorchInt8AwqLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


def _has_int8_weight_mm() -> bool:
    return hasattr(torch.ops.aten, "_weight_int8pack_mm")


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


def _copy_awq_buffers(src: AwqTorchLinear, dst: TorchInt8AwqLinear) -> None:
    dst.qweight.copy_(src.qweight)
    dst.qzeros.copy_(src.qzeros)
    dst.scales.copy_(src.scales)
    if src.bias is not None and dst.bias is not None:
        dst.bias.copy_(src.bias)


def _mock_awq_module_tensors(
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = in_features // group_size
    maxq = (1 << bits) - 1

    float_weight = torch.randn(in_features, out_features, dtype=torch.float32) * 0.2
    weight_groups = float_weight.view(groups, group_size, out_features)

    w_min = weight_groups.amin(dim=1)
    w_max = weight_groups.amax(dim=1)
    scales = ((w_max - w_min).clamp_min(1e-6) / maxq).to(torch.float16)
    zero_points = torch.round((-w_min / scales.to(torch.float32))).clamp_(0, maxq).to(torch.int32)
    quantized = torch.round(
        weight_groups / scales.to(torch.float32).unsqueeze(1) + zero_points.unsqueeze(1)
    ).clamp_(0, maxq).to(torch.int32)

    qweight = _pack_awq_tensor(quantized.view(in_features, out_features), bits)
    qzeros = _pack_awq_tensor(zero_points, bits)
    bias = torch.randn(out_features, dtype=torch.float16)

    return qweight, qzeros, scales, bias


@pytest.mark.skipif(
    not TORCH_HAS_FUSED_OPS or not _has_int8_weight_mm(),
    reason="Torch int8 fused op requires a recent PyTorch CPU build with aten::_weight_int8pack_mm.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_int8_awq_cpu_kernel_deviation_against_torch_awq(dtype: torch.dtype):
    torch.manual_seed(11)

    bits = 4
    group_size = 32
    in_features = 256
    out_features = 192

    qweight, qzeros, scales, bias = _mock_awq_module_tensors(
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
    )

    baseline = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    candidate = TorchInt8AwqLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )

    baseline.qweight.copy_(qweight)
    baseline.qzeros.copy_(qzeros)
    baseline.scales.copy_(scales)
    baseline.bias.copy_(bias)

    _copy_awq_buffers(src=baseline, dst=candidate)
    baseline.post_init()
    candidate.post_init()
    assert getattr(candidate, "qweight", None) is None
    assert getattr(candidate, "qzeros", None) is None
    assert getattr(candidate, "scales", None) is None
    assert candidate.int8_weight_nk is not None
    assert candidate.int8_channel_scale is not None
    baseline.eval()
    candidate.eval()

    x = torch.randn((48, in_features), dtype=dtype, device="cpu")
    with torch.inference_mode():
        ref = baseline(x)
        out = candidate(x)

    diff = (out - ref).abs().to(torch.float32)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    assert max_abs <= 0.5
    assert mean_abs <= 0.08
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), rtol=0.08, atol=0.5)
    assert candidate.int8_module is not None


def test_torch_int8_awq_kernel_is_cpu_only():
    with pytest.raises(NotImplementedError):
        TorchInt8AwqLinear.validate_device(DEVICE.XPU)


def test_torch_int8_awq_backend_selection():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.TORCH_INT8_AWQ,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is TorchInt8AwqLinear
