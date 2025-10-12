# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
import torch.nn as nn

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _build_gptq(damp_percent: float, damp_auto_increment: float) -> GPTQ:
    module = nn.Linear(2, 2, bias=False)
    qcfg = QuantizeConfig(damp_percent=damp_percent, damp_auto_increment=damp_auto_increment)
    return GPTQ(module, qcfg=qcfg)


def test_hessian_inverse_handles_rank_deficiency():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.05)
    device = gptq.module.target_device
    hessian = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device)

    hessian_inv, damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert hessian_inv.shape == hessian.shape
    assert 0 < damp < 1
    assert torch.allclose(hessian_inv, torch.triu(hessian_inv))


def test_hessian_inverse_returns_none_for_indefinite_matrix():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.25)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32, device=device)

    hessian_inv, damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is None
    assert damp == 1.0
