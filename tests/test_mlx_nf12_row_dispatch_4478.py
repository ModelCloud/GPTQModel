# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ensure every 16-column NF12 block has a threadgroup, including tails."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_SOURCE = Path(__file__).resolve().parents[1] / "gptqmodel/nn_modules/qlinear/mlx_nf12.py"
_ROWS = (1, 4, 8, 12, 13, 16)
_OUT_FEATURES = (16, 48, 512, 528, 1024, 1040, 2048, 2064, 2080, 2096)


def _load_module():
    core = ModuleType("mlx.core")
    core.float16 = object()
    core.bfloat16 = object()
    core.float32 = object()
    nn = ModuleType("mlx.nn")
    nn.Module = type("Module", (), {})
    mlx = ModuleType("mlx")
    mlx.core = core
    mlx.nn = nn

    spec = importlib.util.spec_from_file_location("mlx_nf12_row_dispatch_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    previous = {name: sys.modules.get(name) for name in ("mlx", "mlx.core", "mlx.nn")}
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    sys.modules["mlx.nn"] = nn
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module, core


def test_nf12_short_row_dispatch_covers_partial_output_threads():
    module, _ = _load_module()
    for rows in _ROWS:
        for out_features in _OUT_FEATURES:
            for in_features in (64, 192, 512, 2048, 8192):
                use_mma, tile_rows, splits, blocks, grid_n, grid_m = module._linear_launch(
                    rows, out_features, in_features,
                )
                assert use_mma == (rows > 4)
                assert tile_rows * (grid_m - 1) < rows <= tile_rows * grid_m
                assert 1 <= blocks * splits <= 8
                assert (grid_n - 1) * blocks < out_features // 16 <= grid_n * blocks
