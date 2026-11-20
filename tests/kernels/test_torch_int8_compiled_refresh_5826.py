# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A compiled Torch INT8 projection must follow checkpoint edits (#5826).

``TorchInt8Linear`` declares ``COMPILE_LEVEL.INDUCTOR``. The eager
``_refresh_int8_runtime()`` validates the checkpoint version/storage signature
and repacks the derived runtime module when it changes, but a traced forward
cannot run that check. The compiled branch therefore kept the projection bound
at trace time: an in-place scale edit was silently ignored, and a strict
``load_state_dict()`` (which clears the binding) made the next compiled call
raise.

The compiled branch now derives the small affine terms from the live checkpoint
tensors, and a strict reload republishes the derived runtime at the eager load
boundary. The fixture uses the issue's exactly representable oracle
``in_features * (code - zero) * scale``.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.quantization import FORMAT


IN_FEATURES = 256
OUT_FEATURES = 64
GROUP_SIZE = 32
WEIGHT_CODE = 129
SCALE = 0.25


def _packed_zero(zero: int) -> int:
    """Pack one 8-bit zero point into every lane of a GPTQ V2 int32 word."""

    word = sum(zero << (8 * lane) for lane in range(4))
    return word - (1 << 32) if word >= (1 << 31) else word


def _build(*, sym: bool = True, zero: int = 128, out_features: int = OUT_FEATURES) -> TorchInt8Linear:
    layer = TorchInt8Linear(
        bits=8,
        group_size=GROUP_SIZE,
        sym=sym,
        desc_act=True,
        in_features=IN_FEATURES,
        out_features=out_features,
        format=FORMAT.GPTQ_V2,
    ).eval()
    # 0x81818181: weight code 129 in every lane.
    layer.qweight.fill_(sum(WEIGHT_CODE << (8 * lane) for lane in range(4)) - (1 << 32))
    layer.qzeros.fill_(_packed_zero(zero))
    layer.scales.fill_(SCALE)
    layer.g_idx.copy_(torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE)
    layer.qzero_format(2)
    layer.post_init()
    return layer


def _activation() -> torch.Tensor:
    return torch.ones((1, IN_FEATURES), dtype=torch.float32)


def _oracle(zero: int, scale: float) -> float:
    return IN_FEATURES * (WEIGHT_CODE - zero) * scale


def _compiled(layer: TorchInt8Linear):
    return torch.compile(layer, backend="inductor", fullgraph=True)


def test_compiled_forward_observes_an_in_place_scale_edit():
    layer = _build()
    x = _activation()
    compiled = _compiled(layer)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(128, SCALE))

    layer.scales.mul_(2)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(128, SCALE * 2))
    assert layer(x)[0, 0].item() == pytest.approx(_oracle(128, SCALE * 2))


def test_compiled_forward_after_strict_reload_does_not_raise():
    layer = _build()
    x = _activation()
    compiled = _compiled(layer)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(128, SCALE))

    state = {name: value.clone() for name, value in layer.state_dict().items()}
    state["scales"].fill_(1)
    layer.load_state_dict(state, strict=True)

    # Call compiled first: an eager forward would repair the binding and hide
    # the load transition this regression covers.
    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(128, 1.0))
    assert layer(x)[0, 0].item() == pytest.approx(_oracle(128, 1.0))


def test_compiled_asymmetric_forward_observes_a_scale_edit():
    layer = _build(sym=False, zero=127)
    x = _activation()
    compiled = _compiled(layer)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(127, SCALE))

    layer.scales.mul_(2)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(127, SCALE * 2))
    assert layer(x)[0, 0].item() == pytest.approx(_oracle(127, SCALE * 2))


def test_compiled_asymmetric_partial_output_width_observes_a_scale_edit():
    # ``out_features=5`` packs two zero words (eight lanes); the three padding
    # lanes must be cropped before the offset is rebuilt, exactly like
    # ``transform_cpu`` does (#5851).
    layer = _build(sym=False, zero=127, out_features=5)
    x = _activation()
    compiled = _compiled(layer)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(127, SCALE))

    layer.scales.mul_(2)

    assert compiled(x)[0, 0].item() == pytest.approx(_oracle(127, SCALE * 2))
    assert layer(x)[0, 0].item() == pytest.approx(_oracle(127, SCALE * 2))
