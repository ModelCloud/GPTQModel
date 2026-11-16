# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MlxQQQLinear must not narrow BF16 activations to FP16 (#4221).

Narrowing BF16 turns ``|x| > 65504`` into inf, the row scale becomes inf, and
the output becomes NaN.  No Apple hardware is needed to pin the dtype
decision: load the module against stand-in MLX primitives and check it.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_SOURCE = Path(__file__).resolve().parents[1] / "gptqmodel/nn_modules/qlinear/mlx_qqq.py"


def _load_module():
    core = ModuleType("mlx.core")
    core.float16 = object()
    core.bfloat16 = object()
    core.float32 = object()
    mlx = ModuleType("mlx")
    mlx.core = core
    nn = ModuleType("mlx.nn")
    nn.Module = type("Module", (), {})
    mlx.nn = nn

    spec = importlib.util.spec_from_file_location("mlx_qqq_dtype_test", _SOURCE)
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


def test_mlx_qqq_keeps_bf16_activations_wide():
    module, mx = _load_module()

    assert module._quantization_dtype(mx.bfloat16) is mx.bfloat16
    # FP16 and FP32 keep the historical FP16 quantizer operand.
    assert module._quantization_dtype(mx.float16) is mx.float16
    assert module._quantization_dtype(mx.float32) is mx.float16
