# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MLX QQQ must keep the scaled reduction and bias wide (#4534).

No MLX/Metal runtime is available on the Linux CI host, so these tests pin the
shipped epilogue contract with stand-in primitives: the fused finalizer's
generated source and the Python epilogue's dtype/cast trace. Device numerics
remain an Apple-host qualification.
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


class _Module:
    def __contains__(self, name):
        return hasattr(self, name)

    def freeze(self):
        pass


class _Array:
    def __init__(self, dtype, label, shape=(1, 1), ops=None):
        self.dtype = dtype
        self.label = label
        self.shape = tuple(shape)
        self.size = 1
        for dim in self.shape:
            self.size *= dim
        self.ops = [] if ops is None else ops

    def astype(self, dtype):
        self.ops.append(("astype", self.dtype, dtype))
        return _Array(dtype, f"{self.label}.astype", self.shape, self.ops)

    def __mul__(self, other):
        dtype = _promote(self.dtype, other.dtype)
        self.ops.append(("mul", self.dtype, other.dtype, dtype))
        return _Array(dtype, f"({self.label}*{other.label})", self.shape, self.ops)

    def __add__(self, other):
        dtype = _promote(self.dtype, other.dtype)
        self.ops.append(("add", self.dtype, other.dtype, dtype))
        return _Array(dtype, f"({self.label}+{other.label})", self.shape, self.ops)


def _promote(left, right):
    if left is _MX.float32 or right is _MX.float32:
        return _MX.float32
    if left is _MX.bfloat16 or right is _MX.bfloat16:
        return _MX.bfloat16
    return _MX.float16


class _Kernel:
    def __init__(self, spec, launches):
        self.name = spec["name"]
        self.source = spec["source"]
        self._launches = launches

    def __call__(self, **kwargs):
        self._launches.append(kwargs)
        return [_Array(kwargs["output_dtypes"][0], "kernel_output")]


class _Recorder:
    def __init__(self):
        self.kernels = []
        self.launches = []

    def metal_kernel(self, **spec):
        kernel = _Kernel(spec, self.launches)
        self.kernels.append(kernel)
        return kernel


class _Linear:
    def __init__(self, raw):
        self._raw = raw
        self.weight = _Array(_MX.float16, "weight", shape=(64, 64))

    def __call__(self, _x):
        return self._raw


_MX = SimpleNamespace(
    float16=_DType("float16"),
    bfloat16=_DType("bfloat16"),
    float32=_DType("float32"),
)


def _load_module():
    core = ModuleType("mlx.core")
    core.float16 = _MX.float16
    core.bfloat16 = _MX.bfloat16
    core.float32 = _MX.float32
    recorder = _Recorder()
    core.fast = SimpleNamespace(metal_kernel=recorder.metal_kernel)
    nn = ModuleType("mlx.nn")
    nn.Module = _Module
    mlx = ModuleType("mlx")
    mlx.core = core
    mlx.nn = nn

    spec = importlib.util.spec_from_file_location("mlx_qqq_wide_epilogue_test", _SOURCE)
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
    return module, core, recorder


def _make_layer(module, mx, *, rows, bias, raw_dtype=None):
    layer = object.__new__(module.MlxQQQLinear)
    layer.linear = _Linear(_Array(raw_dtype or mx.float32, "raw", shape=(rows, 64)))
    layer.channel_scale = _Array(mx.float32, "channel_scale", shape=(64,))
    if bias:
        layer.bias = _Array(mx.float16, "bias", shape=(64,))
    else:
        layer.zero_bias = (_Array(mx.float16, "zero_bias", shape=(64,)),)
    return layer


def _install_dynamic_quant(module, mx):
    module._dynamic_quant = lambda value: (
        _Array(value.dtype, "quantized", value.shape),
        _Array(mx.float32, "input_scale", (value.shape[0], 1)),
    )


def test_mlx_qqq_fused_finalizer_keeps_bias_wide_and_casts_once():
    module, mx, recorder = _load_module()
    module._finalize_kernel.cache_clear()

    kernel = module._finalize_kernel(mx.bfloat16)
    assert "float value = float(raw[index]) * input_scale[row] * channel_scale[column];" in kernel.source
    assert "if (HAS_BIAS) value += float(bias[column]);" in kernel.source
    assert "output[index] = bfloat16_t(value);" in kernel.source
    assert kernel.source.count("bfloat16_t(") == 1
    assert "half value" not in kernel.source
    assert "half(value +" not in kernel.source

    module._finalize_kernel.cache_clear()
    _install_dynamic_quant(module, mx)
    layer = _make_layer(module, mx, rows=1, bias=True)
    output = layer(_Array(mx.bfloat16, "x", shape=(1, 64)))

    launch = recorder.launches[-1]
    assert launch["inputs"][0].dtype is mx.float32
    assert launch["output_dtypes"] == [mx.bfloat16]
    assert dict(launch["template"])["HAS_BIAS"] is True
    assert output.dtype is mx.bfloat16


@pytest.mark.parametrize("dtype", [_MX.bfloat16, _MX.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_mlx_qqq_python_epilogue_keeps_bias_wide_and_casts_once(dtype, bias):
    module, mx, _ = _load_module()
    _install_dynamic_quant(module, mx)
    layer = _make_layer(module, mx, rows=2, bias=bias)

    output = layer(_Array(dtype, "x", shape=(2, 64)))

    astypes = [op for op in output.ops if op[0] == "astype"]
    assert len(astypes) == 1
    assert astypes[0][1] is mx.float32
    assert astypes[0][2] is dtype
    assert output.dtype is dtype
    if bias:
        add_index = next(i for i, op in enumerate(output.ops) if op[0] == "add")
        astype_index = next(i for i, op in enumerate(output.ops) if op[0] == "astype")
        assert add_index < astype_index
        assert output.ops[add_index][3] is mx.float32
