# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel.nn_modules.qlinear.swordfish import SwordfishLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.swordfish import (
    prewarm_swordfish_extension,
)


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA not available"
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return f"Swordfish requires Blackwell (sm100+); found sm{major}{minor}"
    return None


_SKIP_REASON = _skip_reason()


def _quantize_sym(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    g_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    half_range = 1 << (bits - 1)
    if g_idx is None:
        g_idx = (torch.arange(in_features, device=weight.device, dtype=torch.int32) // group_size)
    num_groups = int(g_idx.max().item()) + 1

    q = torch.empty_like(weight)
    scales = torch.zeros((out_features, num_groups), dtype=weight.dtype, device=weight.device)
    for g in range(num_groups):
        mask = g_idx == g
        block = weight[:, mask]
        max_abs = block.abs().max(dim=1, keepdim=True).values
        max_abs[max_abs == 0] = 1.0
        scale = max_abs / (half_range - 1)
        q_block = torch.round(block / scale).clamp(-(half_range), half_range - 1) + half_range
        q[:, mask] = q_block
        scales[:, g : g + 1] = scale

    zeros = torch.full((out_features, num_groups), half_range, dtype=weight.dtype, device=weight.device)
    return q, scales, zeros, g_idx


def _pack_torch_reference(
    bits: int,
    group_size: int,
    sym: bool,
    desc_act: bool,
    in_features: int,
    out_features: int,
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    bias: bool = False,
) -> TorchLinear:
    linear = nn.Linear(in_features, out_features, bias=bias, dtype=weight.dtype, device="cpu")
    with torch.no_grad():
        linear.weight.copy_(weight)

    torch_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        register_buffers=True,
    )
    torch_linear.pack(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    torch_linear = torch_linear.to(device=weight.device)
    torch_linear.post_init()
    return torch_linear


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestSwordfishKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prewarm_swordfish_extension()

    def _copy_to_swordfish(self, torch_linear: TorchLinear) -> SwordfishLinear:
        sf = SwordfishLinear(
            bits=torch_linear.bits,
            group_size=torch_linear.requested_group_size,
            desc_act=torch_linear.desc_act,
            sym=torch_linear.sym,
            in_features=torch_linear.in_features,
            out_features=torch_linear.out_features,
            bias=False,
            dtype=torch_linear.scales.dtype,
        )
        sf.qweight = nn.Parameter(torch_linear.qweight.data.detach().clone().contiguous(), requires_grad=False)
        sf.scales = nn.Parameter(torch_linear.scales.data.detach().clone().contiguous(), requires_grad=False)
        if torch_linear.g_idx is not None and torch_linear.g_idx.numel() > 0:
            sf.g_idx = nn.Parameter(torch_linear.g_idx.data.detach().clone().contiguous(), requires_grad=False)
        if torch_linear.qzeros is not None and torch_linear.qzeros.numel() > 0:
            sf.qzeros = nn.Parameter(torch_linear.qzeros.data.detach().clone().contiguous(), requires_grad=False)
        if torch_linear.bias is not None:
            sf.bias = nn.Parameter(torch_linear.bias.data.detach().clone().contiguous(), requires_grad=False)
        sf.post_init()
        return sf

    @parameterized.expand([
        (4, 128, True, False, torch.bfloat16),
        (4, -1, True, False, torch.bfloat16),
        (4, 64, True, True, torch.bfloat16),
        (4, 128, True, True, torch.bfloat16),
        (4, 128, True, False, torch.bfloat16),
    ])
    def test_swordfish_matches_torch(self, bits, group_size, sym, desc_act, dtype):
        in_features = 256
        out_features = 512
        device = torch.device("cuda:0")

        torch.manual_seed(42)
        weight = torch.randn((out_features, in_features), dtype=dtype, device="cpu") * 0.5
        g_idx = None
        if desc_act:
            perm = torch.randperm(in_features)
            weight = weight[:, perm]
            g_idx = (torch.arange(in_features, dtype=torch.int32) // group_size)[perm]

        effective_group_size = group_size if group_size > 0 else in_features
        _, scales, zeros, g_idx = _quantize_sym(weight, bits, effective_group_size, g_idx)

        torch_linear = _pack_torch_reference(
            bits, group_size, sym, desc_act, in_features, out_features, weight, scales, zeros, g_idx
        )
        torch_linear = torch_linear.to(device)
        sf = self._copy_to_swordfish(torch_linear)
        sf = sf.to(device)

        for m in (1, 8, 64):
            x = torch.randn((m, in_features), dtype=dtype, device=device) * 0.5
            with torch.no_grad():
                ref = x @ torch_linear.dequantize_weight().to(dtype)
                if torch_linear.bias is not None:
                    ref = ref + torch_linear.bias.to(dtype)
                out = sf(x)
            diff = (out - ref).abs()
            max_err = diff.max().item()
            mae = diff.mean().item()
            self.assertLess(max_err, 0.15, f"m={m}: max error too high ({max_err})")
            self.assertLess(mae, 0.05, f"m={m}: MAE too high ({mae})")
