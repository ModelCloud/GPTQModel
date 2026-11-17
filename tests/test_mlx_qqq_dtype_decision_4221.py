# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MlxQQQLinear must not narrow wide activations before dynamic quantization.

Narrowing BF16 or FP32 turns ``|x| > 65504`` into inf, the row scale becomes
inf, and the output becomes NaN (#4221, #4537). No Apple hardware is needed to
pin the dtype decision and the quantizer operand: load the module against
stand-in MLX primitives and check the shipped call path.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "gptqmodel/nn_modules/qlinear/mlx_qqq.py"


class _DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"mx.{self.name}"


_MX = SimpleNamespace(
    float16=_DType("float16"),
    bfloat16=_DType("bfloat16"),
    float32=_DType("float32"),
)


class _Module:
    def __contains__(self, name):
        return hasattr(self, name)


class _Kernel:
    def __init__(self, spec):
        self.name = spec["name"]
        self.source = spec["source"]


class _Array:
    def __init__(self, dtype, shape=(1, 1), ops=None):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.size = 1
        for dim in self.shape:
            self.size *= dim
        self.ops = [] if ops is None else ops

    def astype(self, dtype):
        self.ops.append(("astype", self.dtype, dtype))
        return _Array(dtype, self.shape, self.ops)

    def __mul__(self, other):
        dtype = _promote(self.dtype, other.dtype)
        self.ops.append(("mul", self.dtype, other.dtype, dtype))
        return _Array(dtype, self.shape, self.ops)

    def __add__(self, other):
        dtype = _promote(self.dtype, other.dtype)
        self.ops.append(("add", self.dtype, other.dtype, dtype))
        return _Array(dtype, self.shape, self.ops)


def _promote(left, right):
    if left is _MX.float32 or right is _MX.float32:
        return _MX.float32
    if left is _MX.bfloat16 or right is _MX.bfloat16:
        return _MX.bfloat16
    return _MX.float16


class _Linear:
    def __init__(self, raw):
        self._raw = raw
        self.weight = _Array(_MX.float16, (64, 64))

    def __call__(self, _x):
        return self._raw


def _load_module():
    core = ModuleType("mlx.core")
    core.float16 = _MX.float16
    core.bfloat16 = _MX.bfloat16
    core.float32 = _MX.float32
    core.fast = SimpleNamespace(metal_kernel=lambda **spec: _Kernel(spec))
    mlx = ModuleType("mlx")
    mlx.core = core
    nn = ModuleType("mlx.nn")
    nn.Module = _Module
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


def _install_dynamic_quant(module, mx):
    seen = []

    def dynamic_quant(value):
        seen.append(value.dtype)
        return (
            _Array(value.dtype, value.shape),
            _Array(mx.float32, (value.shape[0], 1)),
        )

    module._dynamic_quant = dynamic_quant
    return seen


def test_mlx_qqq_keeps_wide_activations_wide():
    module, mx = _load_module()

    assert module._quantization_dtype(mx.float16) is mx.float16
    assert module._quantization_dtype(mx.bfloat16) is mx.bfloat16
    # FP32 is a supported activation dtype (the finalizer has an fp32 variant)
    # and must not lose range before the dynamic scale is computed.
    assert module._quantization_dtype(mx.float32) is mx.float32
    # Unknown dtypes keep the historical FP16 fallback.
    assert module._quantization_dtype(object()) is mx.float16


@pytest.mark.parametrize(
    "dtype", [_MX.float16, _MX.bfloat16, _MX.float32], ids=["fp16", "bf16", "fp32"]
)
def test_mlx_qqq_quantizer_operand_keeps_activation_dtype(dtype):
    module, mx = _load_module()
    seen = _install_dynamic_quant(module, mx)
    layer = object.__new__(module.MlxQQQLinear)
    layer.linear = _Linear(_Array(mx.float32, (2, 64)))
    layer.channel_scale = _Array(mx.float32, (1, 64))

    output = layer(_Array(dtype, (2, 64)))

    assert seen == [dtype]
    assert output.dtype is dtype


def test_mlx_qqq_dynamic_quant_range_is_fp32():
    module, mx = _load_module()
    module._dynamic_quant_kernel.cache_clear()

    source = module._dynamic_quant_kernel(256).source

    assert "float local_max = 0.0f;" in source
    assert "metal::abs(float(x[row * K + col]))" in source
    assert "float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] / 127.0f;" in source
    assert "float(x[row * K + col]) / scale" in source
