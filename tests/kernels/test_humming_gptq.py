# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import itertools
import os
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from gptqmodel.nn_modules.qlinear.humming import HummingGptqLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


try:
    from gptqmodel.humming.layer import HummingLayer  # noqa: F401

    _HUMMING_AVAILABLE = True
except Exception as _humming_exc:  # pragma: no cover - vendored code may not be on PYTHONPATH
    _HUMMING_AVAILABLE = False
    _HUMMING_SKIP_REASON = f"Humming kernel not available: {_humming_exc}"

try:
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

    _MARLIN_AVAILABLE = True
except Exception as _marlin_exc:  # pragma: no cover - marlin kernels may not be installed
    _MARLIN_AVAILABLE = False
    _MARLIN_SKIP_REASON = f"Marlin kernel not available: {_marlin_exc}"


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _pack_gptq_reference(
    bits: int,
    group_size: int,
    in_f: int,
    out_f: int,
    sym: bool,
    device: torch.device,
) -> tuple[TorchLinear, torch.Tensor]:
    """Quantize a dense linear with the Torch GPTQ packer and return the packed module + dense weight."""
    linear = nn.Linear(in_f, out_f, bias=False).to(torch.float16)
    weight = linear.weight.data.detach()

    # group_size == -1 means per-channel / one group.
    gs = in_f if group_size == -1 else group_size
    num_groups = in_f // gs
    w = weight.view(out_f, num_groups, gs)

    if sym:
        scales = w.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
        q = torch.round(w / scales).clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1).to(torch.int32)
        q = q + (2 ** (bits - 1))
        zeros = torch.full((out_f, num_groups), 2 ** (bits - 1), dtype=torch.float16)
    else:
        wmin = w.amin(dim=-1, keepdim=True)
        wmax = w.amax(dim=-1, keepdim=True)
        scales = (wmax - wmin) / (2 ** bits - 1)
        q = torch.round((w - wmin) / scales).clamp(0, 2 ** bits - 1).to(torch.int32)
        zeros = -wmin / scales
        zeros = zeros.squeeze(-1).contiguous().to(torch.float16)

    scales = scales.squeeze(-1).contiguous().to(torch.float16)
    zeros = zeros.contiguous().to(torch.float16)
    g_idx = torch.tensor([i // gs for i in range(in_f)], dtype=torch.int32)

    mod = TorchLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=sym,
        in_features=in_f,
        out_features=out_f,
        bias=False,
        pack_dtype=torch.int32,
    )
    mod.pack(linear, scales, zeros, g_idx)
    mod = mod.to(device)
    mod.eval()
    mod.post_init()
    return mod, weight.to(device)


def _build_marlin_reference(
    torch_mod: TorchLinear,
    bits: int,
    group_size: int,
    sym: bool,
    in_f: int,
    out_f: int,
    dtype: torch.dtype,
    device: torch.device,
) -> MarlinLinear | None:
    """Build a Marlin reference from the same packed GPTQ buffers, if supported."""
    if not _MARLIN_AVAILABLE:
        return None
    if bits not in MarlinLinear.SUPPORTS_BITS:
        return None
    if sym not in MarlinLinear.SUPPORTS_SYM:
        return None
    if group_size not in MarlinLinear.SUPPORTS_GROUP_SIZE:
        return None
    if dtype not in MarlinLinear.SUPPORTS_DTYPES:
        return None

    try:
        mar = MarlinLinear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=sym,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            dtype=dtype,
        ).to(device)
        mar.qweight.data.copy_(torch_mod.qweight.data)
        mar.scales.data.copy_(torch_mod.scales.data)
        mar.qzeros.data.copy_(torch_mod.qzeros.data)
        mar.g_idx.data.copy_(torch_mod.g_idx.data)
        mar.post_init()
        return mar
    except Exception:
        return None


# Full bit / group-size coverage in fp16, plus a small bf16 sweep for 4/8-bit.
_TEST_CASES = [
    (bits, group_size, sym, torch.float16)
    for bits, group_size, sym in itertools.product(
        (2, 3, 4, 8),
        (128, 64, 32, -1),
        (True, False),
    )
] + [
    (bits, 64, sym, torch.bfloat16)
    for bits, sym in itertools.product((4, 8), (True, False))
]


class TestHummingGptqKernelOutput(unittest.TestCase):
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    @classmethod
    def setUpClass(cls):
        if not _HUMMING_AVAILABLE:
            raise unittest.SkipTest(_HUMMING_SKIP_REASON)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Humming kernel tests require CUDA.")

    @parameterized.expand(_TEST_CASES)
    def test_humming_gptq_output_matches_torch(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        dtype: torch.dtype,
    ):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this device")

        in_f = 512
        out_f = 256
        device = torch.device("cuda:0")

        torch_mod, dense_weight = _pack_gptq_reference(
            bits=bits,
            group_size=group_size,
            in_f=in_f,
            out_f=out_f,
            sym=sym,
            device=device,
        )

        x = torch.randn(2, 8, in_f, device=device, dtype=dtype)
        with torch.inference_mode():
            ref_out = torch_mod(x.to(torch.float16))
            dense_out = x.to(torch.float16) @ dense_weight.t()

        hum = HummingGptqLinear(
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
        hum.qweight.copy_(torch_mod.qweight)
        hum.qzeros.copy_(torch_mod.qzeros)
        hum.scales.copy_(torch_mod.scales)
        hum.g_idx.copy_(torch_mod.g_idx)
        hum.eval()
        hum.post_init()

        with torch.inference_mode():
            hum_out = hum(x)

        self.assertEqual(hum_out.shape, ref_out.shape)
        self.assertTrue(
            torch.allclose(hum_out.to(torch.float32), ref_out.to(torch.float32), atol=0.15, rtol=0.15),
            f"Humming GPTQ output deviates from the Torch GPTQ reference (bits={bits}, group_size={group_size}, sym={sym}, dtype={dtype})",
        )
        # Low-bit quantization has large intrinsic error vs the dense float weights;
        # Humming is still expected to match the quantized Torch reference tightly.
        dense_atol = {2: 2.0, 3: 1.5, 4: 0.25, 8: 0.25}.get(bits, 0.25)
        self.assertTrue(
            torch.allclose(hum_out.to(torch.float32), dense_out.to(torch.float32), atol=dense_atol, rtol=0.25),
            f"Humming GPTQ output deviates from the dense reference (bits={bits}, group_size={group_size}, sym={sym}, dtype={dtype})",
        )

        marlin_mod = _build_marlin_reference(
            torch_mod=torch_mod,
            bits=bits,
            group_size=group_size,
            sym=sym,
            in_f=in_f,
            out_f=out_f,
            dtype=dtype,
            device=device,
        )
        if marlin_mod is not None:
            with torch.inference_mode():
                marlin_out = marlin_mod(x)
            self.assertTrue(
                torch.allclose(hum_out.to(torch.float32), marlin_out.to(torch.float32), atol=0.15, rtol=0.15),
                f"Humming GPTQ output deviates from the Marlin reference (bits={bits}, group_size={group_size}, sym={sym}, dtype={dtype})",
            )


if __name__ == "__main__":
    unittest.main()
