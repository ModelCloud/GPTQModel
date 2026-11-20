# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip search must size its scratch from the projection's MSE snapshot.

A ``dynamic`` rule may set ``mse.clip_grid_budget`` for one projection. The
clip search resolved bits, group size, format, and symmetry per projection but
always read the top-level ``qcfg.mse.clip_grid_budget``, so a 1 MiB override
under an 8 MiB default still requested a 7.5 MiB workspace (#5791).
"""

from math import prod
from unittest.mock import patch

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import AWQConfig, MSEConfig


MIB = 1 << 20
MODULE_NAME = "layers.0.proj"
MATCHED_PATTERN = r"^layers\.0\.proj$"


def _run(*, top_budget, dynamic=None):
    torch.manual_seed(555)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].proj = nn.Linear(256, 256, bias=False).eval()
    x = torch.randn(16, 256)

    config = AWQConfig(
        bits=4,
        group_size=128,
        offload_to_disk=False,
        device="cpu",
        mse=MSEConfig(clip_grid_budget=top_budget),
        dynamic=dynamic,
    )
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.model = model
    processor.gptq_model = None
    processor.qcfg = config
    layer = model.layers[0]

    real_empty = torch.empty
    workspaces = []

    def trace_empty(*args, **kwargs):
        if args and isinstance(args[0], (tuple, torch.Size)) and len(args[0]) == 5:
            assert kwargs.get("dtype") == torch.float32
            workspaces.append(prod(args[0]) * 4)
        return real_empty(*args, **kwargs)

    with patch.object(torch, "empty", trace_empty):
        [(name, bounds)] = processor._search_best_clip(
            layer, {"proj": layer.proj}, {"proj": x}
        )

    return name, max(workspaces), bounds


def _rule(budget, *, pattern=MATCHED_PATTERN):
    return {pattern: {"mse": MSEConfig(clip_grid_budget=budget)}}


def test_dynamic_budget_sizes_the_clip_workspace():
    base_name, base_bytes, base_bounds = _run(top_budget=8 * MIB)
    dynamic_name, dynamic_bytes, dynamic_bounds = _run(
        top_budget=8 * MIB, dynamic=_rule(MIB)
    )
    small_name, small_bytes, small_bounds = _run(top_budget=MIB)

    assert base_name == dynamic_name == small_name == "proj"
    assert base_bytes > dynamic_bytes
    assert dynamic_bytes == small_bytes <= MIB
    torch.testing.assert_close(dynamic_bounds, base_bounds, rtol=0, atol=0)
    torch.testing.assert_close(small_bounds, base_bounds, rtol=0, atol=0)


def test_unmatched_dynamic_rule_keeps_the_top_level_budget():
    _, base_bytes, base_bounds = _run(top_budget=8 * MIB)
    _, unmatched_bytes, unmatched_bounds = _run(
        top_budget=8 * MIB,
        dynamic=_rule(MIB, pattern=r"^layers\.0\.other$"),
    )

    assert unmatched_bytes == base_bytes
    torch.testing.assert_close(unmatched_bounds, base_bounds, rtol=0, atol=0)


def test_dynamic_zero_budget_is_preserved():
    _, base_bytes, base_bounds = _run(top_budget=8 * MIB)
    _, top_zero_bytes, top_zero_bounds = _run(top_budget=0)
    _, dynamic_zero_bytes, dynamic_zero_bounds = _run(
        top_budget=8 * MIB, dynamic=_rule(0)
    )

    assert dynamic_zero_bytes == top_zero_bytes < base_bytes
    torch.testing.assert_close(dynamic_zero_bounds, base_bounds, rtol=0, atol=0)
    torch.testing.assert_close(top_zero_bounds, base_bounds, rtol=0, atol=0)
