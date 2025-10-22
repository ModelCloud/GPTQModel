# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import pytest
import torch

from gptqmodel.nn_modules.qlinear.machete import MacheteQuantLinear
from gptqmodel.utils.machete import (
    _validate_machete_device_support,
    machete_import_exception,
)


@pytest.mark.cuda
def test_machete_forward_smoke():
    if machete_import_exception is not None:
        pytest.skip(f"Machete extension unavailable: {machete_import_exception}")
    if not torch.cuda.is_available() or not _validate_machete_device_support():
        pytest.skip("Machete smoke test requires a Hopper (SM90+) CUDA device")

    layer = MacheteQuantLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=256,
        out_features=512,
        bias=True,
    ).cuda()

    layer.qweight.data.zero_()
    layer.scales.data.fill_(1.0)
    layer.qzeros.data.zero_()
    g_idx = (
        torch.arange(layer.in_features, dtype=torch.int32, device=layer.g_idx.device)
        // layer.group_size
    )
    layer.g_idx.data.copy_(g_idx)
    layer.bias.data.zero_()
    layer.post_init()

    x = torch.randn(4, layer.in_features, dtype=torch.float16, device="cuda")
    out = layer(x)

    assert out.shape == (4, layer.out_features)
    assert out.dtype == x.dtype
    assert torch.all(torch.isfinite(out))
