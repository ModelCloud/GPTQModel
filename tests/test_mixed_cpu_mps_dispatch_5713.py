# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""A mixed CPU/MPS device map must keep every entry on its requested device.

``simple_dispatch_model`` used to treat any map whose values were all in
``{"cpu", "mps"}`` as a single-device map and move the whole model to the first
entry's device (#5713). That ignored the explicit placement of every other
entry: a CPU-pinned module could land on MPS, or an MPS-mapped model could run
entirely on CPU depending on dict order.
"""

import pytest
import torch

from gptqmodel.utils.model import simple_dispatch_model


requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


def _model() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    )


def _weight_devices(model: torch.nn.Module) -> dict:
    return {name: parameter.device.type for name, parameter in model.named_parameters() if name.endswith("weight")}


def test_cpu_or_mps_placement_canonicalises_spellings():
    from gptqmodel.utils.model import _cpu_or_mps_placement

    assert _cpu_or_mps_placement("cpu") == ("cpu", 0)
    assert _cpu_or_mps_placement(torch.device("cpu:0")) == ("cpu", 0)
    assert _cpu_or_mps_placement("mps") == ("mps", 0)
    assert _cpu_or_mps_placement("mps:0") == ("mps", 0)
    assert _cpu_or_mps_placement("disk") is None
    assert _cpu_or_mps_placement("cuda:1") is None


@requires_mps
@pytest.mark.parametrize(
    "device_map, expected",
    [
        ({"0": "cpu", "2": "mps"}, {"0.weight": "cpu", "2.weight": "mps"}),
        ({"0": "mps", "2": "cpu"}, {"0.weight": "mps", "2.weight": "cpu"}),
    ],
)
def test_mixed_cpu_mps_map_keeps_each_entry_on_its_device(device_map, expected):
    model = _model()

    simple_dispatch_model(model, device_map)

    assert _weight_devices(model) == expected
    assert model.hf_device_map == device_map


@requires_mps
def test_mixed_map_does_not_collapse_onto_the_first_entry():
    # The first entry used to decide the whole model in either order.
    model = _model()

    simple_dispatch_model(model, {"0": "mps", "2": "cpu"})

    assert model[0].weight.device.type == "mps"
    assert model[2].weight.device.type == "cpu"


@requires_mps
def test_mps_spellings_still_take_the_single_device_shortcut():
    model = _model()

    simple_dispatch_model(model, {"0": "mps", "2": "mps:0"})

    assert {parameter.device.type for parameter in model.parameters()} == {"mps"}
    assert not any(hasattr(module, "_hf_hook") for module in model.modules())


@requires_mps
def test_all_mps_map_still_takes_the_single_device_shortcut():
    model = _model()

    simple_dispatch_model(model, {"0": "mps", "2": "mps"})

    assert {parameter.device.type for parameter in model.parameters()} == {"mps"}
    assert not any(hasattr(module, "_hf_hook") for module in model.modules())
