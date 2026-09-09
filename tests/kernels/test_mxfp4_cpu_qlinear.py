# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.mxfp4_cpu import Mxfp4CpuLinear


@pytest.mark.parametrize("use_vnni", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_mxfp4_cpu_linear_forward(dtype, use_vnni):
    torch.manual_seed(42)
    in_features = 64
    out_features = 32
    linear = nn.Linear(in_features, out_features, bias=False).eval()

    module = Mxfp4CpuLinear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        use_vnni=use_vnni,
    )
    module.pack_original(linear, scales=None, zeros=None)

    x = torch.randn(1, in_features, dtype=torch.float32)
    out = module(x.to(dtype))
    assert out.shape == (1, out_features)

    ref = x @ linear.weight.data.T
    if dtype == torch.float8_e4m3fn:
        out = out.to(torch.float32)
    max_err = (out - ref).abs().max()
    assert max_err < 1.0


def test_mxfp4_cpu_linear_dequantize_weight():
    torch.manual_seed(42)
    in_features = 64
    out_features = 32
    linear = nn.Linear(in_features, out_features, bias=False).eval()

    module = Mxfp4CpuLinear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
    )
    module.pack_original(linear, scales=None, zeros=None)

    dequant = module.dequantize_weight()
    assert dequant.shape == (in_features, out_features)


@pytest.mark.parametrize('use_vnni', [False, True])
def test_gsq_native_payload_save_reload(tmp_path, use_vnni):
    generator = torch.Generator().manual_seed(7)
    layer = nn.Linear(32, 8, bias=False)
    layer.weight.data.copy_(torch.randn(8, 32, generator=generator) * .05)
    module = Mxfp4CpuLinear(4, -1, False, True, 32, 8, use_vnni=use_vnni)
    module.pack_original(layer, None, None, gsq={'enabled': True, 'steps': 20})
    # Exactly representable activation values isolate payload/kernel agreement.
    inputs = torch.ones(2, 32, dtype=torch.bfloat16)
    before = module(inputs)
    reference = inputs.float() @ module.dequantize_weight()
    error = (before.float()-reference).abs()
    assert torch.isfinite(before).all()
    assert error.mean() <= .002 and error.max() <= .046875
    path = tmp_path / 'payload.pt'
    # VNNI packing is derived from canonical bytes on reload.
    torch.save({'qweight': module.qweight, 'scales': module.scales}, path)
    restored = Mxfp4CpuLinear(4, -1, False, True, 32, 8, use_vnni=use_vnni)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    after = restored(inputs)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
