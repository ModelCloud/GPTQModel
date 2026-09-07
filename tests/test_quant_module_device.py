# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.utils.model import create_quant_module


@pytest.mark.parametrize(
    "device", ["cuda:0", "cuda:1", torch.device("cuda:1"), "cpu", torch.device("cpu")]
)
def test_quant_module_validation_normalizes_indexed_devices(device):
    seen = []

    class Dummy(torch.nn.Module):
        @classmethod
        def validate(cls, **kwargs):
            seen.append(kwargs["device"])
            return True, None

        def __init__(self, **kwargs):
            super().__init__()
            self.bias = None

    root = torch.nn.Module()
    root.proj = torch.nn.Linear(32, 32, bias=False)
    create_quant_module(
        name="proj",
        linear_cls=Dummy,
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=32,
        module=root,
        submodule=root.proj,
        sym=True,
        device=device,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
    )
    assert seen == [DEVICE.CUDA if "cuda" in str(device) else DEVICE.CPU]
