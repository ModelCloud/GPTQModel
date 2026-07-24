# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel.nn_modules.qlinear.humming import HummingAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear


try:
    from gptqmodel.humming.layer import HummingLayer  # noqa: F401

    _HUMMING_AVAILABLE = True
except Exception as _humming_exc:  # pragma: no cover - vendored code may not be on PYTHONPATH
    _HUMMING_AVAILABLE = False
    _HUMMING_SKIP_REASON = f"Humming kernel not available: {_humming_exc}"


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _pack_awq_reference(
    bits: int,
    group_size: int,
    in_f: int,
    out_f: int,
    sym: bool,
    device: torch.device,
) -> AwqTorchLinear:
    """Pack a dense linear into the standard AWQ layout using the Torch AWQ packer."""
    linear = nn.Linear(in_f, out_f, bias=False).to(torch.float16)
    weight = linear.weight.data.detach()

    w = weight.view(out_f, in_f // group_size, group_size)
    if sym:
        scales = w.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
        q = torch.round(w / scales).clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1).to(torch.int32)
        q = q + (2 ** (bits - 1))
        zeros = torch.full((out_f, in_f // group_size), 2 ** (bits - 1), dtype=torch.float16)
    else:
        wmin = w.amin(dim=-1, keepdim=True)
        wmax = w.amax(dim=-1, keepdim=True)
        scales = (wmax - wmin) / (2 ** bits - 1)
        q = torch.round((w - wmin) / scales).clamp(0, 2 ** bits - 1).to(torch.int32)
        zeros = -wmin / scales
        zeros = zeros.squeeze(-1).to(torch.float16)

    scales = scales.squeeze(-1).to(torch.float16)
    zeros = zeros.to(torch.float16)

    mod = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=sym,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        pack_dtype=torch.int32,
    )
    mod.pack(linear, scales, zeros)
    mod = mod.to(device)
    mod.eval()
    mod.post_init()
    return mod


class TestHummingAwqKernelOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _HUMMING_AVAILABLE:
            raise unittest.SkipTest(_HUMMING_SKIP_REASON)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Humming kernel tests require CUDA.")

    @parameterized.expand([
        (4, 128, False, torch.float16),
        (4, 128, True, torch.float16),
        (4, 64, False, torch.bfloat16),
    ])
    def test_humming_awq_output_matches_torch(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        dtype: torch.dtype,
    ):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this device")

        in_f = 256
        out_f = 256
        device = torch.device("cuda:0")

        awq_mod = _pack_awq_reference(
            bits=bits,
            group_size=group_size,
            in_f=in_f,
            out_f=out_f,
            sym=sym,
            device=device,
        )

        x = torch.randn(2, 8, in_f, device=device, dtype=dtype)
        with torch.inference_mode():
            ref_out = awq_mod(x)

        hum = HummingAwqLinear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=sym,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            pack_dtype=torch.int32,
            dtype=dtype,
        ).to(device)
        hum.qweight.copy_(awq_mod.qweight)
        hum.qzeros.copy_(awq_mod.qzeros)
        hum.scales.copy_(awq_mod.scales)
        hum.eval()
        hum.post_init()

        with torch.inference_mode():
            hum_out = hum(x)

        self.assertEqual(hum_out.shape, ref_out.shape)
        self.assertTrue(
            torch.allclose(hum_out.to(torch.float32), ref_out.to(torch.float32), atol=0.01, rtol=0.05),
            "Humming AWQ output deviates from the Torch AWQ reference",
        )


if __name__ == "__main__":
    unittest.main()
