# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
import pytest
import torch
import torch.nn as nn
import transformers

from gptqmodel.utils.hf import suspend_hf_weight_init


def test_suspend_hf_weight_init_restores_globals_after_exception():
    modeling_utils = transformers.modeling_utils
    had_init_flag = hasattr(modeling_utils, "_init_weights")
    original_init_flag = getattr(modeling_utils, "_init_weights", None)
    original_kaiming_uniform = torch.nn.init.kaiming_uniform_
    original_uniform = torch.nn.init.uniform_
    original_normal = torch.nn.init.normal_

    with pytest.raises(RuntimeError, match="boom"):
        with suspend_hf_weight_init():
            assert torch.nn.init.kaiming_uniform_ is not original_kaiming_uniform
            assert torch.nn.init.uniform_ is not original_uniform
            assert torch.nn.init.normal_ is not original_normal
            assert getattr(modeling_utils, "_init_weights") is False
            raise RuntimeError("boom")

    assert torch.nn.init.kaiming_uniform_ is original_kaiming_uniform
    assert torch.nn.init.uniform_ is original_uniform
    assert torch.nn.init.normal_ is original_normal

    if had_init_flag:
        assert getattr(modeling_utils, "_init_weights") == original_init_flag
    else:
        assert not hasattr(modeling_utils, "_init_weights")

    linear = nn.Linear(32, 32, bias=False)
    assert torch.isfinite(linear.weight).all()
