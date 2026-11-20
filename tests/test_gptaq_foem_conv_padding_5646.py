# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM must calibrate the convolution that will actually run (#5646).

Both overrides built their im2col view with ``nn.Unfold(padding=module.padding)``,
which only expresses zero padding: a reflect-padded layer was calibrated as if
it were zero-padded, and ``padding="same"`` was not resolved at all.  Compare
the full Hessian and cross-moment against explicit patches for reflect,
asymmetric same, and a zero-padding control.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


pytestmark = [pytest.mark.cpu]


def _patches(x, pads, mode, kernel, stride=(1, 1), dilation=(1, 1)):
    """Explicit im2col view of ``x`` on the layer's effective padding."""
    (top, bottom), (left, right) = pads
    padded = F.pad(
        x, (left, right, top, bottom), mode="constant" if mode == "zeros" else mode
    )
    unfold = nn.Unfold(kernel, dilation=dilation, padding=0, stride=stride)
    return unfold(padded).permute([1, 0, 2]).flatten(1)


def _make_task(kind, layer, native):
    wrapped = NamedModule(layer, name="p", full_name="model.layers.0.p", layer_index=0)
    wrapped.state[NATIVE_INPUTS_STATE_KEY] = [native]
    if kind is GPTAQ:
        kwargs = {"gptaq": GPTAQConfig(alpha=1.0)}
    else:
        kwargs = {"foem": FOEMConfig(alpha=1.0, beta=0.0)}
    qcfg = QuantizeConfig(
        bits=4, group_size=-1, desc_act=False, act_group_aware=False, **kwargs
    )
    task = kind(wrapped, qcfg)
    task.quantizer.configure(perchannel=True)
    return task


CASES = (
    ("reflect", 3, 1, "reflect", ((1, 1), (1, 1))),
    ("same-asymmetric", 2, "same", "reflect", ((0, 1), (0, 1))),
    ("zeros-control", 3, 1, "zeros", ((1, 1), (1, 1))),
)


@pytest.mark.parametrize("kind", (GPTAQ, FOEM), ids=("gptaq", "foem"))
@pytest.mark.parametrize(
    "name,kernel,padding,mode,pads", CASES, ids=[case[0] for case in CASES]
)
def test_conv_padding_hessian_and_cross_moment(kind, name, kernel, padding, mode, pads):
    torch.manual_seed(0)
    x = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    layer = nn.Conv2d(
        1, 1, kernel, padding=padding, padding_mode=mode, bias=False
    )
    native = x + 0.5
    task = _make_task(kind, layer, native)
    task.add_batch(x, None)

    quantized_patches = _patches(x, pads, mode, kernel)
    native_patches = _patches(native, pads, mode, kernel)
    tokens = quantized_patches.shape[1]

    # The explicit patches must reproduce the layer output before they can be
    # trusted as the calibration geometry.
    flattened = quantized_patches.t()
    expected_dense = (flattened @ layer.weight.detach().reshape(-1)).reshape_as(layer(x))
    torch.testing.assert_close(expected_dense, layer(x))

    normalizer = 1.0 if kind is GPTAQ else float(tokens)
    expected_hessian = (
        2.0 / normalizer * quantized_patches.double() @ quantized_patches.double().t()
    )
    expected_cross = (
        2.0
        / normalizer
        * (native_patches.double() - quantized_patches.double())
        @ quantized_patches.double().t()
    )

    torch.testing.assert_close(task.H.double(), expected_hessian, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(task.dXXT.double(), expected_cross, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", (GPTAQ, FOEM), ids=("gptaq", "foem"))
def test_reflect_padding_matches_the_reported_hessian(kind):
    """The issue's 3x3 reflect repro: 141 -> 2 * 141 (GPTAQ) or 2 * 141 / 9 (FOEM)."""
    x = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    layer = nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect", bias=False)
    task = _make_task(kind, layer, x)
    task.add_batch(x, None)

    expected = 282.0 if kind is GPTAQ else 2.0 * 141.0 / 9.0
    assert task.H[0, 0].item() == pytest.approx(expected, rel=1e-6)
