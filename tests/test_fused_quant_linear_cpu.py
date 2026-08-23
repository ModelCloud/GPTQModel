# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.fused_quant_linear import _FusedMarlinKernel, _FusedQuantGroup


def test_base_qmodel_to_moves_unregistered_fused_kernel_tensors():
    """Fusion-owned tensors must follow the wrapped model across device moves."""

    model = nn.Module()
    model.proj = nn.Linear(2, 2)

    group = object.__new__(_FusedQuantGroup)
    group.kernel = SimpleNamespace(
        qweight=torch.ones((2, 2), dtype=torch.int32),
        scales=torch.ones((1, 2)),
        qzeros=torch.zeros((1, 1), dtype=torch.int32),
        g_idx=torch.arange(2, dtype=torch.int32),
        bias=None,
    )
    model.proj._gptqmodel_fused_group = group

    qmodel = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.to("meta")

    assert qmodel.model.proj.weight.device.type == "meta"
    assert group.kernel.qweight.device.type == "meta"
    assert group.kernel.scales.device.type == "meta"


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
@pytest.mark.parametrize("migration_api", ["to", "cuda"])
def test_base_qmodel_to_moves_fused_marlin_state_between_visible_gpus(migration_api):
    source = torch.device("cuda:0")
    target = torch.device("cuda:1")

    model = nn.Module()
    model.proj = nn.Linear(2, 2, device=source)

    kernel = object.__new__(_FusedMarlinKernel)
    kernel.qweight = torch.ones((2, 2), dtype=torch.int32, device=source)
    kernel.scales = torch.ones((1, 2), device=source)
    kernel.qzeros = torch.zeros((1, 1), dtype=torch.int32, device=source)
    kernel.g_idx = torch.arange(2, dtype=torch.int32, device=source)
    kernel.g_idx_sort_indices = torch.arange(2, dtype=torch.int32, device=source)
    kernel.bias = torch.zeros(2, device=source)
    # A deliberately invalid source buffer distinguishes recreation from `.to(target)`.
    kernel.workspace = torch.full((1,), 7, dtype=torch.int32, device=source)

    group = object.__new__(_FusedQuantGroup)
    group.kernel = kernel
    model.proj._gptqmodel_fused_group = group

    qmodel = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    if migration_api == "to":
        qmodel.to(target)
    else:
        qmodel.cuda(target.index)

    assert qmodel.model.proj.weight.device == target
    for name in ("qweight", "scales", "qzeros", "g_idx", "g_idx_sort_indices", "bias", "workspace"):
        assert getattr(kernel, name).device == target
    expected_workspace_blocks = max(
        torch.cuda.get_device_properties(target).multi_processor_count,
        128,
    )
    assert kernel.workspace.numel() == expected_workspace_blocks
    assert torch.count_nonzero(kernel.workspace).item() == 0


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_fused_marlin_cross_cuda_apply_rebuilds_workspace_without_transforming_source():
    source = torch.device("cuda:0")
    target = torch.device("cuda:1")

    kernel = object.__new__(_FusedMarlinKernel)
    kernel.qweight = torch.ones((2, 2), dtype=torch.int32, device=source)
    kernel.scales = torch.ones((1, 2), device=source)
    kernel.qzeros = torch.zeros((1, 1), dtype=torch.int32, device=source)
    kernel.g_idx = torch.arange(2, dtype=torch.int32, device=source)
    kernel.g_idx_sort_indices = torch.arange(2, dtype=torch.int32, device=source)
    kernel.bias = torch.zeros(2, device=source)
    source_workspace = torch.full((128,), 7, dtype=torch.int32, device=source)
    kernel.workspace = source_workspace

    group = object.__new__(_FusedQuantGroup)
    group.kernel = kernel

    def transform(tensor):
        assert tensor is not source_workspace, "cross-device Marlin workspace must be rebuilt, not copied"
        return tensor.to(target)

    group._apply(transform)

    expected_workspace_blocks = max(
        torch.cuda.get_device_properties(target).multi_processor_count,
        128,
    )
    assert kernel.workspace.device == target
    assert kernel.workspace.numel() == expected_workspace_blocks
    assert torch.count_nonzero(kernel.workspace).item() == 0
