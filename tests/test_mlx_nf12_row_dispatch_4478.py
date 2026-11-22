# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ensure every 16-column NF12 block has a threadgroup, including tails."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_SOURCE = Path(__file__).resolve().parents[1] / "gptqmodel/nn_modules/qlinear/mlx_nf12.py"
_ROWS = (1, 4, 8, 12, 13, 16, 17, 31, 33, 65, 128)
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


def test_nf12_prefill_routes_respect_hardware_and_activation_precision(monkeypatch):
    module, core = _load_module()
    for accelerated in (False, True):
        monkeypatch.setattr(module, "_has_accelerated_tensorops", lambda: accelerated)
        for dtype in (core.float16, core.bfloat16):
            assert module._linear_route(32, 128, 2048, dtype) == ("tensor" if accelerated else "decode")
            assert module._linear_route(64, 4096, 4160, dtype) == ("tensor" if accelerated else "decode")
            assert module._linear_route(129, 4096, 4160, dtype) == "decode"
        assert module._linear_route(32, 128, 2048, core.float32) == "simd"
        assert module._linear_route(32, 4096, 2048, core.float32) == "decode"


def test_nf12_tensor_tiles_fit_threadgroup_memory(monkeypatch):
    module, _ = _load_module()
    for ultra in (False, True):
        monkeypatch.setattr(module, "_is_m5_ultra", lambda: ultra)
        for m, n, k in ((16, 2048, 4096), (32, 2064, 4096), (64, 2064, 4096),
                        (32, 128256, 2048), (64, 128256, 2048), (127, 4096, 14336)):
            bm, bn, groups, splits = module._tensor_tile(m, n, k)
            assert bm <= m and bn % 16 == 0 and groups % splits == 0
            assert 4 * groups * bn * (max(32, bm) if splits > 1 else 32) <= 32 * 1024
            if not ultra:
                assert (bn, groups) == (16, 4)
