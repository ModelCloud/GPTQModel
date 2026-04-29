# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
)
from gptqmodel.quantization.gptq import GPTQ


def _build_gar_perm(columns: int, group_size: int) -> torch.Tensor:
    diag_h = torch.arange(columns, dtype=torch.float32)
    local_perms, local_values = compute_local_perms(
        diag_h,
        group_size,
        return_values=True,
    )
    global_perm = compute_global_perm(
        diag_h,
        group_size,
        precomputed_values=local_values,
    )
    perm = compose_final_perm(local_perms, global_perm, group_size)
    return extend_perm_with_tail(perm, columns)


def test_gar_perm_keeps_remainder_columns():
    perm = _build_gar_perm(columns=2880, group_size=128)

    assert perm.numel() == 2880
    assert set(perm.tolist()) == set(range(2880))


def test_gar_perm_roundtrip_preserves_weight_layout():
    columns = 17
    weight = torch.randn(6, columns)
    perm = _build_gar_perm(columns=columns, group_size=5)

    restored = weight[:, perm][:, torch.argsort(perm)]
    assert torch.allclose(restored, weight)


@torch.inference_mode()
def test_gptq_quantize_keeps_tail_group_statistics():
    torch.manual_seed(0)

    layer = nn.Linear(10, 6, bias=False, dtype=torch.float32).eval()
    qcfg = QuantizeConfig(bits=4, group_size=4, act_group_aware=True)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)

    gptq.add_batch(torch.randn(3, 10, dtype=torch.float32), None)
    qweight, scales, zeros, g_idx, *_ = gptq.quantize(blocksize=4)

    assert qweight.shape == layer.weight.shape
    assert scales.shape[1] == 3
    assert zeros.shape[1] == 3
    assert g_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
