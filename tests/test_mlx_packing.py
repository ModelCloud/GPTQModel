# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from types import SimpleNamespace

import numpy as np
import pytest


mlx = pytest.importorskip("mlx.core")

from gptqmodel.utils.mlx_packing import repack_awq_4bit, repack_gptq_4bit  # noqa: E402


@pytest.mark.parametrize("format", ["gptq", "awq"])
def test_packed_4bit_matches_source_codes(format):
    rng = np.random.default_rng(17)
    in_features, out_features, group_size = 128, 64, 64
    codes = rng.integers(0, 16, (in_features, out_features), dtype=np.uint32)
    zeros = rng.integers(0, 16, (in_features // group_size, out_features), dtype=np.uint32)
    scales = rng.uniform(0.01, 0.1, zeros.shape).astype(np.float16)
    shifts = np.arange(8, dtype=np.uint32) * 4

    if format == "gptq":
        qweight = np.bitwise_or.reduce(codes.reshape(-1, 8, out_features) << shifts[None, :, None], axis=1)
        qzeros = np.bitwise_or.reduce(zeros.reshape(-1, out_features // 8, 8) << shifts, axis=-1)
        repack = repack_gptq_4bit
    else:
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        qweight = np.bitwise_or.reduce(codes.reshape(in_features, -1, 8)[:, :, order] << shifts, axis=-1)
        qzeros = np.bitwise_or.reduce(zeros.reshape(-1, out_features // 8, 8)[:, :, order] << shifts, axis=-1)
        repack = repack_awq_4bit

    weight, mlx_scales, biases = repack(qweight, qzeros, scales, in_features, out_features)
    assert weight.dtype == np.uint32
    assert weight.shape == (out_features, in_features // 8)

    actual = np.array(mlx.dequantize(
        mlx.array(weight), mlx.array(mlx_scales), mlx.array(biases), group_size, 4
    ))
    expected = (codes.astype(np.float32) - np.repeat(zeros.astype(np.float32), group_size, axis=0)) * np.repeat(scales.astype(np.float32), group_size, axis=0)
    np.testing.assert_allclose(actual, expected.T, rtol=0.01, atol=0.002)

    x = mlx.array(rng.normal(size=(3, in_features)).astype(np.float16))
    y = mlx.quantized_matmul(x, mlx.array(weight), scales=mlx.array(mlx_scales),
                             biases=mlx.array(biases), group_size=group_size, bits=4)
    np.testing.assert_allclose(np.array(y), np.array(x).astype(np.float32) @ expected, rtol=0.02, atol=0.04)


@pytest.mark.parametrize("format", ["gptq", "awq"])
@pytest.mark.parametrize("holder", ["torch", "mlx"])
def test_packed_layers_load_into_mlx_quantized_linear(monkeypatch, format, holder):
    import mlx.nn as mlx_nn
    import torch

    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear, MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
    from gptqmodel.utils import mlx as mlx_utils

    source = torch.nn.Module()
    linear_class = (
        MlxQuantLinear if format == "gptq" else AwqMlxQuantLinear
    ) if holder == "mlx" else (
        TorchLinear if format == "gptq" else AwqTorchLinear
    )
    source.linear = linear_class(
        bits=4, group_size=64, sym=False, desc_act=False,
        in_features=128, out_features=64, bias=False,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
    )
    if format == "gptq":
        source.linear.qzero_format(2)
    source.linear.qweight.fill_(0x76543210)
    source.linear.qzeros.fill_(0x11111111)
    source.linear.scales.fill_(0.125)

    class ModelArgs:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class TinyModel(mlx_nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = mlx_nn.Linear(128, 64, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (TinyModel, ModelArgs))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, mlx_nn.QuantizedLinear)
    assert config["quantization"]["bits"] == 4
    x = mlx.ones((1, 128), dtype=mlx.float16)
    if format == "gptq":
        expected = source.linear.dequantize_weight().float().sum(dim=0).numpy()
    else:
        expected = dequantize_gemm(source.linear.qweight, source.linear.qzeros,
                                   source.linear.scales, 4, 64).float().sum(dim=0).numpy()
    np.testing.assert_allclose(np.array(model(x))[0], expected, atol=0.05)


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal requires macOS")
@pytest.mark.parametrize("format", ["gptq", "awq"])
def test_mlx_quant_linear_registry_validates_capabilities(format):
    import torch

    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear, MlxQuantLinear
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear, validate_quant_linear

    linear_class, method, checkpoint_format = (
        (MlxQuantLinear, METHOD.GPTQ, FORMAT.GPTQ_V2) if format == "gptq"
        else (AwqMlxQuantLinear, METHOD.AWQ, FORMAT.GEMM)
    )
    contract = dict(
        bits=4, group_size=64, desc_act=False, sym=False,
        pack_dtype=torch.int32, dtype=torch.float16,
        in_features=128, out_features=64, device=DEVICE.MPS,
    )
    assert select_quant_linear(
        **{key: contract[key] for key in ("bits", "group_size", "desc_act", "sym", "pack_dtype", "dtype", "device")},
        backend=BACKEND.MLX, format=checkpoint_format, quant_method=method,
    ) is linear_class
    assert validate_quant_linear(linear_class, **contract)[0]
    for change in (
        {"bits": 3}, {"group_size": 256}, {"desc_act": True},
        {"pack_dtype": torch.int16}, {"dtype": torch.float32},
        {"in_features": 120}, {"out_features": 63}, {"device": DEVICE.CPU},
    ):
        assert not validate_quant_linear(linear_class, **(contract | change))[0]


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal requires macOS")
def test_auto_selects_mlx_only_for_compatible_models():
    import torch

    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.loader import _auto_select_mlx_backend
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND

    class Config:
        def to_dict(self):
            return {"model_type": "qwen2"}

    qcfg = SimpleNamespace(
        bits=4, pack_dtype=torch.int32, group_size=128, sym=False,
        desc_act=False, dynamic=None, rotation=None,
    )

    def select(backend=BACKEND.AUTO, device=DEVICE.MPS, method=METHOD.GPTQ,
               format_code=FORMAT.GPTQ_V2):
        return _auto_select_mlx_backend(
            backend, device, Config(), qcfg, method, format_code, None
        )

    assert select() == BACKEND.MLX
    assert select(method=METHOD.AWQ, format_code=FORMAT.GEMM) == BACKEND.MLX
    assert select(backend=BACKEND.GPTQ_TORCH) == BACKEND.GPTQ_TORCH
    assert select(device=DEVICE.CPU) == BACKEND.AUTO
    assert select(method=METHOD.AWQ, format_code=FORMAT.GEMV) == BACKEND.AUTO
    qcfg.desc_act = True
    assert select() == BACKEND.AUTO
    qcfg.desc_act = False
    qcfg.group_size = 256
    assert select() == BACKEND.AUTO
