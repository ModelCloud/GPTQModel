# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ grouped export must take its INT8 channel scale from the quantized Q.

GPTQ error propagation can push ``max|Q|`` above ``max|W|``. The exporter used
to fit the per-channel INT8 scale to ``W`` (the weight still resident in the
layer at that point), so packed codes reached ~140 and were clamped to 127.
The exported weights then differed from the in-memory ``wq`` that loss logging
and next-layer replay consume, and from upstream QQQ, which assigns
``weight.data = Q`` before fitting the scale.
"""

import torch

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization.config import QQQConfig
from gptqmodel.quantization.qqq import QQQ, Quantizer


ROWS = 128
COLUMNS = 256
GROUP_SIZE = 128


def _eight_bit_channel_scale(weight):
    """8-bit symmetric per-output-channel scale, independent of the exporter."""
    quantizer = Quantizer(qcfg=QQQConfig(group_size=GROUP_SIZE, desc_act=False))
    quantizer.configure(bits=8, perchannel=True, groupsize=-1, sym=True, mse=False)
    quantizer.find_params(weight.clone(), weight=True)
    return quantizer.scale.reshape(-1)


def _quantize(group_size=GROUP_SIZE):
    torch.manual_seed(4008)
    layer = torch.nn.Linear(COLUMNS, ROWS, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.normal_(0, 0.05)
    original = layer.weight.detach().clone()

    config = QQQConfig(
        group_size=group_size, desc_act=False, damp_percent=0.01, offload_to_disk=False
    )
    task = QQQ(layer, config)
    task.quantizer.configure(
        4, perchannel=True, sym=True, mse=False, groupsize=group_size
    )
    generator = torch.Generator().manual_seed(4008)
    for _ in range(8):
        task.add_batch(
            torch.randn(1, 16, COLUMNS, dtype=torch.float16, generator=generator), None
        )
    wq, scales, zeros, g_idx, _duration, _loss, _damp, scale_extra, _nsamples = task.quantize()
    return original, wq, scales, zeros, g_idx, scale_extra


def _pack_and_dequantize(wq, scales, scale_extra):
    packed = QQQTorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=COLUMNS,
        out_features=ROWS,
        bias=False,
    )
    linear = torch.nn.Linear(COLUMNS, ROWS, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(wq)
    packed.pack(linear, scales, scale_extra)
    plane, s_channel = packed._dequantize_weight_for_torch()
    # `_dequantize_weight_for_torch` works in [in_features, out_features].
    return plane.t(), s_channel.reshape(-1)


def test_grouped_export_scale_derives_from_the_quantized_weight():
    original, wq, _scales, _zeros, _g_idx, scale_extra = _quantize()

    # The case is only meaningful if quantization overshoots the source range.
    assert wq.abs().max() > original.abs().max()
    assert scale_extra is not None

    # Independent recomputation: an 8-bit symmetric per-channel scale of Q.
    reference = _eight_bit_channel_scale(wq)
    torch.testing.assert_close(scale_extra.reshape(-1), reference, rtol=0, atol=0)
    # ... which is exactly max|Q| / 127, not max|W| / 127.
    torch.testing.assert_close(reference, wq.float().abs().amax(dim=1) / 127, rtol=0, atol=0)
    assert not torch.allclose(
        reference, original.float().abs().amax(dim=1) / 127, rtol=0, atol=0
    )


def test_grouped_export_does_not_saturate_and_reproduces_wq():
    original, wq, scales, _zeros, _g_idx, scale_extra = _quantize()
    plane, s_channel = _pack_and_dequantize(wq, scales, scale_extra)

    # Counterfactual: the W-fitted scale the exporter used before this fix
    # clamps real codes, so the comparison cannot go vacuous.
    pre_fix_scale = original.float().abs().amax(dim=1) / 127
    pre_fix_codes = (wq.float() / pre_fix_scale.reshape(-1, 1)).round()
    assert int((pre_fix_codes.abs() > 127.5).sum()) > 0

    # Packed codes stay inside the INT8 range: nothing is clamped.
    assert plane.abs().max().item() <= 127.0
    post_fix_codes = (wq.float() / s_channel.reshape(-1, 1)).round()
    assert int((post_fix_codes.abs() > 127.5).sum()) == 0

    # The export is exact to the format's own INT8 step.
    dequantized = plane * s_channel.reshape(-1, 1)
    max_error = (dequantized - wq.float()).abs().max().item()
    half_step = 0.5 * s_channel.max().item()
    assert max_error <= half_step, (max_error, half_step)


def test_channelwise_export_has_no_extra_scale():
    _original, _wq, _scales, _zeros, _g_idx, scale_extra = _quantize(group_size=-1)
    assert scale_extra is None
