# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
from logbar import LogBar


# Keep this CPU test isolated from optional BitBLAS/TVM import side effects.
os.environ.setdefault("GPTQMODEL_DISABLE_BITBLAS", "1")

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


log = LogBar.shared()
deviation_cols = log.columns(
    cols=[
        {"label": "DType", "width": "fit"},
        {"label": "DescAct", "width": "fit"},
        {"label": "MaxAbsDiff", "width": "fit"},
        {"label": "MeanAbsDiff", "width": "fit"},
        {"label": "MaxRelDiff", "width": "fit"},
    ],
    padding=1,
)


def _log_deviation_header_once() -> None:
    if getattr(_log_deviation_header_once, "_printed", False):
        return
    log.info("\nTorchInt8 CPU Deviation vs Torch Baseline")
    deviation_cols.info.header()
    _log_deviation_header_once._printed = True


def _has_int8_weight_mm() -> bool:
    return hasattr(torch.ops.aten, "_weight_int8pack_mm")


def _mock_gptq_linear(
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor]:
    maxq = (1 << (bits - 1)) - 1
    weight = torch.randn((in_features, out_features), dtype=torch.float32)

    reshaped = weight.view(in_features // group_size, group_size, out_features)
    w_g = reshaped.permute(1, 0, 2).reshape(group_size, -1)

    scales = torch.maximum(
        w_g.abs().max(dim=0, keepdim=True).values,
        torch.full((1, w_g.shape[1]), 1e-6, device=w_g.device),
    )
    scales = scales / maxq

    q = torch.round(w_g / scales).clamp_(-maxq, maxq)
    ref = (q * scales).to(dtype=torch.float16)

    ref = ref.reshape(group_size, -1, out_features)
    ref = ref.permute(1, 0, 2).reshape(in_features, out_features)

    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = ref.t().contiguous()

    scales = scales.reshape(-1, out_features).contiguous()
    zeros = torch.zeros_like(scales, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    return linear, scales, zeros, g_idx


def _copy_gptq_buffers(src: TorchLinear, dst: TorchInt8Linear) -> None:
    dst.qweight.copy_(src.qweight)
    dst.qzeros.copy_(src.qzeros)
    dst.scales.copy_(src.scales)
    dst.g_idx.copy_(src.g_idx)
    dst.qzero_format(format=src.qzero_format())
    if src.bias is not None and dst.bias is not None:
        dst.bias.copy_(src.bias)


@pytest.mark.skipif(
    not TORCH_HAS_FUSED_OPS or not _has_int8_weight_mm(),
    reason="Torch int8 fused op requires a recent PyTorch CPU build with aten::_weight_int8pack_mm.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("desc_act", [False, True])
def test_torch_int8_cpu_kernel_deviation_against_torch(dtype: torch.dtype, desc_act: bool):
    torch.manual_seed(7)

    bits = 4
    group_size = 32
    in_features = 256
    out_features = 192

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)
    if desc_act:
        groups = in_features // group_size
        g_idx = (torch.arange(in_features, dtype=torch.int32) * 3) % groups

    baseline = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    candidate = TorchInt8Linear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )

    baseline.optimize = lambda *args, **kwargs: None
    candidate.optimize = lambda *args, **kwargs: None
    baseline.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    _copy_gptq_buffers(src=baseline, dst=candidate)
    baseline.post_init()
    candidate.post_init()
    assert getattr(candidate, "qweight", None) is None
    assert getattr(candidate, "qzeros", None) is None
    assert getattr(candidate, "scales", None) is None
    assert getattr(candidate, "g_idx", None) is None
    assert candidate.int8_weight_nk is not None
    assert candidate.int8_channel_scale is not None
    baseline.eval()
    candidate.eval()

    x = torch.randn((48, in_features), dtype=dtype, device="cpu")
    with torch.inference_mode():
        ref = baseline(x)
        out = candidate(x)

    diff = (out - ref).abs().to(torch.float32)
    rel = diff / ref.abs().clamp_min(1e-6)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    max_rel = float(rel.max().item())

    _log_deviation_header_once()
    deviation_cols.info(
        str(dtype),
        str(desc_act),
        f"{max_abs:.6f}",
        f"{mean_abs:.6f}",
        f"{max_rel:.6f}",
    )

    # vLLM-style int4->float->int8 re-quantization introduces expected approximation noise.
    assert max_abs <= 0.5
    assert mean_abs <= 0.08
    torch.testing.assert_close(out, ref, rtol=0.08, atol=0.5)
    assert candidate.int8_module is not None


def test_torch_int8_kernel_is_cpu_only():
    with pytest.raises(NotImplementedError):
        TorchInt8Linear.validate_device(DEVICE.XPU)


def test_torch_int8_supports_expected_bits():
    assert TorchInt8Linear.SUPPORTS_BITS == [2, 4, 8]
