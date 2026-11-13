# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare Marlin with the Torch backend at extreme packed code values."""

import sys

import pytest
import torch


MARLIN_SUPPORTED = (
    sys.platform == "linux"
    and torch.cuda.is_available()
    and not torch.version.hip
    and torch.cuda.get_device_capability(0) >= (7, 5)
)
pytestmark = pytest.mark.skipif(
    not MARLIN_SUPPORTED, reason="GPTQ Marlin requires Linux and NVIDIA CUDA compute capability >= 7.5"
)


@pytest.mark.parametrize("pattern", (0, -1, -65536, 65535))
def test_marlin_repack_matches_torch_at_extreme_codes(pattern):
    """0x00000000, 0xFFFFFFFF, 0xFFFF0000 and 0x0000FFFF as signed int32."""
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    k, n, group, dtype = 256, 128, 128, torch.float16

    def build(cls):
        module = cls(
            bits=4, group_size=group, sym=True, desc_act=False, in_features=k,
            out_features=n, bias=False, dtype=dtype,
        ).to("cuda:0", dtype)
        return module

    reference, candidate = build(TorchLinear), build(MarlinLinear)
    # Integer inputs and a binary scale keep the expected sums exact in FP16.
    lane = torch.arange(k, device="cuda:0").remainder(8).to(dtype) - 3
    x = torch.stack((lane, -lane, lane.roll(1), lane.roll(4)))
    with torch.no_grad():
        qweight = torch.full((k // 8, n), pattern, dtype=torch.int32, device="cuda:0")
        qzeros = torch.full(reference.qzeros.shape, -2004318072, dtype=torch.int32, device="cuda:0")
        scales = torch.full(reference.scales.shape, 1 / 128, dtype=dtype, device="cuda:0")
        for module in (reference, candidate):
            module.qweight.copy_(qweight)
            module.qzeros.copy_(qzeros)
            module.scales.copy_(scales)
            if hasattr(module, "qzero_format"):
                module.qzero_format(format=2)
            module.post_init()
        left = reference(x).float()
        right = candidate(x).float()
    assert torch.equal(left, right), (left - right).abs().max()
