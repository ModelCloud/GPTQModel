# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Marlin must honor a non-contiguous checkpoint ``g_idx``.

``Qwen2-1.5B-Instruct-GPTQ-4bits-gp32`` declares ``desc_act: false`` but stores
a block-permuted ``g_idx``. Dropping that index silently decodes contiguous
groups and returns wrong logits.
"""

from pathlib import Path

import pytest
import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


_MODEL = "/monster/data/model/Qwen2-1.5B-Instruct-GPTQ-4bits-gp32"
_LAYER = "model.layers.0.self_attn.q_proj"


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(_MODEL).is_dir(),
    reason="requires CUDA and the local Qwen2 non-contiguous g_idx checkpoint",
)


def test_marlin_decodes_noncontiguous_g_idx():
    marlin_model = GPTQModel.load(_MODEL, device="cuda:0", backend=BACKEND.MARLIN, dtype=torch.bfloat16)
    torch_model = GPTQModel.load(_MODEL, device="cuda:0", backend=BACKEND.GPTQ_TORCH, dtype=torch.bfloat16)

    marlin_modules = {
        name: module for name, module in marlin_model.model.named_modules() if isinstance(module, MarlinLinear)
    }
    torch_modules = {
        name: module for name, module in torch_model.model.named_modules() if isinstance(module, TorchLinear)
    }

    torch_linear = torch_modules[_LAYER]
    assert torch_linear.g_idx.numel() == torch_linear.in_features
    assert not torch.all(torch_linear.g_idx[1:] >= torch_linear.g_idx[:-1]), "checkpoint g_idx must be non-monotonic"

    marlin_linear = marlin_modules[_LAYER]
    # The runtime must switch to activation-order decoding instead of emptying g_idx.
    assert marlin_linear.desc_act is True
    assert marlin_linear.is_k_full is True
    assert marlin_linear.g_idx.numel() == marlin_linear.in_features

    x = torch.randn(4, torch_linear.in_features, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = torch_linear(x)
        actual = marlin_linear(x)

    relative_error = (
        (actual.float() - expected.float()).norm() / expected.float().norm()
    ).item()
    assert relative_error < 2e-3, relative_error
