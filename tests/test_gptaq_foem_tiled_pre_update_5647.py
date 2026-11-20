# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression tests for #5647: native-delta corrections across tile boundaries.

GPTAQ and FOEM defer the upper-triangular ``P`` product for the columns past a
solver tile to the end of that tile. The in-tile updates mutate the working
block in place, so reading that block after the column loop applied the
correction to the post-update weight instead of each column's pre-update
working value. A single-column tile (the scalar schedule) was therefore wrong
while one big tile was right. These tests pin the pre-update transition and
compare every tile size against the single-tile schedule.
"""

import pytest
import torch
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


def _quantize(kind, weight, calibration, native, blocksize, beta=0.0):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    wrapped = NamedModule(
        layer, name="p", full_name="model.layers.0.p", layer_index=0
    )
    wrapped.state[NATIVE_INPUTS_STATE_KEY] = [native]
    if kind is GPTAQ:
        extra = {"gptaq": GPTAQConfig(alpha=1.0)}
    else:
        extra = {"foem": FOEMConfig(alpha=1.0, beta=beta)}
    qcfg = QuantizeConfig(
        bits=4,
        group_size=-1,
        desc_act=False,
        act_group_aware=False,
        **extra,
    )
    task = kind(wrapped, qcfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(calibration, None)
    return task.quantize(blocksize=blocksize)[0]


@pytest.mark.cpu
@pytest.mark.parametrize("kind", [GPTAQ, FOEM], ids=["gptaq", "foem"])
@pytest.mark.parametrize("blocksize", [1, 2, 4])
def test_native_delta_transition_uses_pre_update_weight(kind, blocksize):
    """The second column must see the first column's value before it was updated.

    The identity calibration makes the damped Hessian diagonal and leaves only
    ``P[0, 1] = 1 / 1.05``. Column 1 therefore sees ``0.20 + 0.45 / 1.05 =
    0.628571...``, i.e. 5 steps of the ``2 / 15`` grid, regardless of where the
    tile boundary falls. Using the first column's post-update value ``0.4``
    gives 4 steps instead.
    """

    weight = torch.tensor([[0.45, 0.20, 0.55, 1.0]])
    calibration = torch.eye(4)
    native = calibration.clone()
    native[1, 0] = 1.0

    q = _quantize(kind, weight, calibration, native, blocksize)

    assert q[0, 1].item() == pytest.approx(10.0 / 15.0, abs=1e-6)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "kind,beta",
    [(GPTAQ, 0.0), (FOEM, 0.0), (FOEM, 0.1)],
    ids=["gptaq", "foem", "foem-beta"],
)
@pytest.mark.parametrize("blocksize", [1, 2, 3])
def test_tiled_native_delta_matches_single_tile(kind, beta, blocksize):
    """Every tile schedule must agree with the one-tile reference."""

    columns = 8
    weight = torch.linspace(-0.9, 0.9, 2 * columns).reshape(2, columns)
    calibration = torch.eye(columns)
    native = calibration.clone()
    # Upper-triangular native deltas (token row > input column) put several
    # nonzero entries in ``P``, including ones that cross tile boundaries.
    for row, column in ((2, 0), (5, 3), (6, 4)):
        native[row, column] = 1.0

    reference = _quantize(
        kind, weight, calibration, native, blocksize=columns, beta=beta
    )
    tiled = _quantize(kind, weight, calibration, native, blocksize=blocksize, beta=beta)

    assert torch.equal(tiled, reference)
