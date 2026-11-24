# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The derived AllSpark kernel bias is a cache, never model state.

Both AllSpark kernels keep ``self.bias`` in FP32 and derive a narrowed copy for
the kernel. That copy is assigned to an instance attribute, so an
``nn.Parameter`` input used to be re-registered under ``_bias_value`` and
appeared in ``state_dict``. A saved checkpoint then carried an unexpected key
and could not be loaded back into the module.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear


def _bare_module(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.bias = torch.nn.Parameter(torch.zeros(8, dtype=torch.float16))
    module.compute_dtype = torch.float16
    module._bias_key = None
    module._bias_value = None
    return module


@pytest.mark.parametrize("cls", [AllSparkLinear, AllSparkAwqLinear])
def test_a_parameter_bias_is_not_re_registered_under_the_cache_name(cls):
    module = _bare_module(cls)

    value = module._kernel_bias()

    assert isinstance(value, torch.Tensor)
    assert not isinstance(value, torch.nn.Parameter)
    assert "_bias_value" not in dict(module.named_parameters())
    assert "_bias_value" not in dict(module.named_buffers())
    assert "_bias_value" not in module.state_dict()


@pytest.mark.parametrize("cls", [AllSparkLinear, AllSparkAwqLinear])
def test_the_cache_still_tracks_in_place_bias_writes(cls):
    module = _bare_module(cls)
    first = module._kernel_bias()

    with torch.no_grad():
        module.bias.add_(1.0)
    second = module._kernel_bias()

    assert first is not second
    assert torch.equal(second, module.bias)
