# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.models import BaseQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import QuantizeConfig, QVQConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import make_quant


@pytest.mark.parametrize(("bits", "words_per_tile"), ((2.5, 20), (3.5, 28)))
def test_qvq_scoped_fractional_rate_metadata_allocates_and_loads_exact_trellis(bits, words_per_tile):
    """Exercise half-step corruption from installed module through reload allocation."""

    class ScopedModel(BaseQModel):
        def _get_layer_names(self):
            return ["model.layers.0"]

    class TinyModel(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].proj = module

    installed = QVQLinear(bits=bits, in_features=16, out_features=16, dtype=torch.float32)
    scoped = ScopedModel.__new__(ScopedModel)
    torch.nn.Module.__init__(scoped)
    scoped.model = TinyModel(installed)
    scoped.quantize_config = QVQConfig(bits=2, device="cpu", offload_to_disk=False)

    dynamic = scoped._capture_quantized_layer_dynamic()
    assert dynamic == {"+:.*model\\.layers\\.0\\.proj": {"bits": bits}}

    scoped.quantize_config.dynamic = dynamic
    serialized = scoped.quantize_config.to_dict()
    assert next(iter(serialized["dynamic"].values()))["bits"] == bits
    reloaded_config = QuantizeConfig.from_quant_config(serialized)

    shell = TinyModel(torch.nn.Linear(16, 16, bias=False, dtype=torch.float32))
    make_quant(
        shell,
        reloaded_config,
        {"model.layers.0.proj": {}},
        BACKEND.QVQ,
        "lm_head",
        device=DEVICE.CPU,
        dtype=torch.float32,
    )
    reloaded = shell.model.layers[0].proj
    assert isinstance(reloaded, QVQLinear)
    assert reloaded.bits == bits
    assert reloaded.trellis.shape == installed.trellis.shape == (1, words_per_tile)

    reloaded.load_state_dict(installed.state_dict())
    source = torch.randn(3, 16)
    torch.testing.assert_close(reloaded(source), installed(source), rtol=0, atol=0)


def test_qvq_fractional_reproducer_would_fail_with_integer_truncation():
    """Pin the pre-fix geometry mismatch independently of the fixed recorder."""

    installed = QVQLinear(bits=2.5, in_features=16, out_features=16, dtype=torch.float32)
    truncated_reload = QVQLinear(bits=int(installed.bits), in_features=16, out_features=16, dtype=torch.float32)

    assert installed.trellis.shape == (1, 20)
    assert truncated_reload.trellis.shape == (1, 16)
    with pytest.raises(RuntimeError, match="size mismatch"):
        truncated_reload.load_state_dict(installed.state_dict())
