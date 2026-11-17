# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MLX NF12 short-row dispatch must write every accepted output column (#4478).

The dot-product path launches ``out_features // (16 * blocks)`` threadgroups,
each owning ``blocks`` adjacent 16-column blocks. A block count that does not
divide the 16-column count floors the launch and leaves the trailing columns
unwritten. No Apple hardware is needed to pin the host dispatch: load the
shipped module against stand-in MLX primitives and record the launch.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_SOURCE = Path(__file__).resolve().parents[1] / "gptqmodel/nn_modules/qlinear/mlx_nf12.py"
_ROWS = (1, 4, 8, 12, 13, 16)
_OUT_FEATURES = (512, 528, 1024, 1040, 2048, 2064, 2080, 2096)


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


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.size = 1
        for dim in self.shape:
            self.size *= dim

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _Array(shape, self.dtype)


def _record_rows_launch(module, mx, rows, out_features):
    launches = []

    def factory(launch_rows, blocks, dtype):
        def launch(**kwargs):
            launches.append(
                {
                    "rows": launch_rows,
                    "blocks": blocks,
                    "grid": tuple(kwargs["grid"]),
                    "out_features": dict(kwargs["template"])["N"],
                }
            )
            return [_Array(kwargs["output_shapes"][0], dtype)]

        return launch

    module._rows_kernel = factory
    layer = module.MlxNF12Linear.__new__(module.MlxNF12Linear)
    layer.in_features = 64
    layer.out_features = out_features
    layer.packed = layer.escapes = layer.exp_base = layer.bias = None
    module.MlxNF12Linear.__call__(layer, _Array((rows, 64), mx.bfloat16))
    assert len(launches) == 1
    return launches[0]


def test_nf12_short_row_dispatch_writes_each_accepted_column_once():
    module, mx = _load_module()
    for rows in _ROWS:
        for out_features in _OUT_FEATURES:
            launch = _record_rows_launch(module, mx, rows, out_features)
            assert launch["rows"] == rows
            assert launch["out_features"] == out_features

            blocks = launch["blocks"]
            groups = launch["grid"][1]
            writes = [0] * out_features
            for group in range(groups):
                for block in range(blocks):
                    base = (group * blocks + block) * 16
                    for column in range(base, min(base + 16, out_features)):
                        writes[column] += 1

            assert writes == [1] * out_features, (rows, out_features, blocks, groups)


def test_nf12_row_blocks_divide_the_output_block_count():
    module, _ = _load_module()

    # Divisible widths keep the wide block choices.
    assert module._row_blocks(1, 1024) == 2
    assert module._row_blocks(1, 2048) == 4
    assert module._row_blocks(8, 2048) == 2

    # Widths whose 16-column count is not divisible by the preferred block
    # count step down instead of flooring the launch.
    assert module._row_blocks(1, 1040) == 1
    assert module._row_blocks(1, 2064) == 1
    assert module._row_blocks(1, 2080) == 2
    assert module._row_blocks(1, 2096) == 1

    for rows in _ROWS:
        for out_features in _OUT_FEATURES:
            blocks = module._row_blocks(rows, out_features)
            assert (out_features // 16) % blocks == 0
