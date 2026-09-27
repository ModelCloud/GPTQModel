# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Marlin format: IST-DASLab contributors, MIT, https://github.com/IST-DASLab/marlin
# BitBLAS format: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""MLX inference for GPTQ Marlin and BitBLAS checkpoint layouts."""

import gc

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("fmt,bits", (("marlin", 4), ("bitblas", 2), ("bitblas", 8)))
def test_gptq_checkpoint_formats_auto_select_mlx(fmt, bits):
    from types import SimpleNamespace

    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.loader import _auto_select_mlx_backend
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND

    class Config:
        def to_dict(self):
            return {"model_type": "qwen3_5"}

    qcfg = SimpleNamespace(
        bits=bits,
        group_size=128,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        dynamic=None,
        rotation=None,
    )
    assert _auto_select_mlx_backend(
        BACKEND.AUTO,
        DEVICE.MPS,
        Config(),
        qcfg,
        METHOD.GPTQ,
        FORMAT(fmt),
        None,
    ) == BACKEND.MLX


def test_gptq_format_registry_enforces_layout_capabilities():
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.mlx import (GPTQBitBLASMlxQuantLinear,
                                                  GPTQMarlinMlxQuantLinear)
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    common = dict(
        group_size=128, desc_act=False, pack_dtype=torch.int32,
        dtype=torch.float16, device=DEVICE.MPS, backend=BACKEND.MLX,
        quant_method=METHOD.GPTQ,
    )
    for bits in (4, 8):
        assert select_quant_linear(
            **common, bits=bits, sym=True, format=FORMAT.MARLIN,
        ) is GPTQMarlinMlxQuantLinear
    for bits in (2, 4, 8):
        for sym in (True, False):
            assert select_quant_linear(
                **common, bits=bits, sym=sym, format=FORMAT.BITBLAS,
            ) is GPTQBitBLASMlxQuantLinear
    for bits, sym in ((2, True), (3, True), (4, False), (5, True), (6, True), (7, True), (8, False)):
        with pytest.raises(ValueError):
            select_quant_linear(
                **common, bits=bits, sym=sym, format=FORMAT.MARLIN,
            )


def _pack_bytes(codes, bits):
    values = 8 // bits
    shifts = np.arange(values, dtype=np.uint8) * bits
    return np.bitwise_or.reduce(
        codes.reshape(*codes.shape[:-1], -1, values).astype(np.uint8) << shifts,
        axis=-1,
    ).view(np.int8)


def _pack_gptq_rows(codes, bits):
    values = 32 // bits
    shifts = np.arange(values, dtype=np.uint32) * bits
    return np.bitwise_or.reduce(
        codes.reshape(-1, values, codes.shape[1]).astype(np.uint32)
        << shifts[None, :, None],
        axis=1,
    ).astype(np.int32)


def _pack_gptq_columns(codes, bits):
    values = 32 // bits
    shifts = np.arange(values, dtype=np.uint32) * bits
    return np.bitwise_or.reduce(
        codes.reshape(codes.shape[0], -1, values).astype(np.uint32) << shifts,
        axis=-1,
    ).astype(np.int32)


def _load(monkeypatch, source):
    from gptqmodel.utils import mlx as mlx_utils

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
            )

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    root = torch.nn.Module()
    root.linear = source
    model, config = mlx_utils._packed_mlx_weights(root, {}, "lm_head")
    assert config["_gptqmodel_custom_mlx_runtime"]
    return model


def _rounded_oracle(inputs, weight, bias, dtype):
    with np.errstate(all="ignore"):
        output = torch.from_numpy(np.asarray(inputs.astype(mx.float32))).double() @ weight.T
    if bias is not None:
        output += bias.double()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return output.to(target).float().numpy()


@pytest.mark.parametrize("bits", (2, 4, 8))
@pytest.mark.parametrize("group_size", (-1, 32, 64, 128))
@pytest.mark.parametrize("sym", (True, False), ids=("sym", "asym"))
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
def test_bitblas_gptq_all_bits_groups_and_output_dtypes(
    monkeypatch, bits, group_size, sym, dtype,
):
    from gptqmodel.nn_modules.qlinear.mlx import GPTQBitBLASMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_gptq import MlxGPTQLinear

    in_features, out_features = 128, 64
    effective_group = in_features if group_size == -1 else group_size
    groups = in_features // effective_group
    rng = np.random.default_rng(7100 + bits * 10 + effective_group + int(sym))
    if sym:
        signed = rng.integers(
            -(1 << (bits - 1)),
            1 << (bits - 1),
            (out_features, in_features),
            dtype=np.int16,
        )
        storage_codes = (signed & ((1 << bits) - 1)).astype(np.uint8)
        logical_codes = storage_codes ^ (1 << (bits - 1))
        zeros = np.full((out_features, groups), 1 << (bits - 1), dtype=np.uint8)
    else:
        storage_codes = rng.integers(
            0, 1 << bits, (out_features, in_features), dtype=np.uint8,
        )
        logical_codes = storage_codes
        zeros = rng.integers(
            0, 1 << bits, (out_features, groups), dtype=np.uint8,
        )
    scales = rng.uniform(0.0002, 0.001, (out_features, groups)).astype(np.float16)
    source = GPTQBitBLASMlxQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
        dtype=torch.float16,
    )
    source.qweight.copy_(torch.from_numpy(_pack_bytes(storage_codes, bits)))
    source.scales.copy_(torch.from_numpy(scales))
    if not sym:
        source.qzeros.copy_(torch.from_numpy(_pack_bytes(zeros.T, bits)))
    source.bias.copy_(torch.linspace(-0.01, 0.01, out_features, dtype=torch.float16))

    model = _load(monkeypatch, source)
    assert isinstance(model.linear, MlxGPTQLinear)
    assert model.linear.linear.weight.nbytes == source.qweight.numel()
    inputs = mx.array(
        rng.normal(0, 0.1, (2, 3, in_features)).astype(np.float32),
    ).astype(dtype)
    actual = model(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    expanded_zeros = torch.from_numpy(zeros).repeat_interleave(effective_group, dim=1)
    expanded_scales = torch.from_numpy(scales).double().repeat_interleave(
        effective_group, dim=1,
    )
    weight = (
        torch.from_numpy(logical_codes.astype(np.int16)).double()
        - expanded_zeros.double()
    ) * expanded_scales
    expected = _rounded_oracle(inputs, weight, source.bias, dtype)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )


@pytest.mark.parametrize("bits", (4, 8))
@pytest.mark.parametrize("group_size", (-1, 32, 64, 128))
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
def test_marlin_gptq_all_bits_groups_and_output_dtypes(
    monkeypatch, bits, group_size, dtype,
):
    from gptqmodel.nn_modules.qlinear.mlx import GPTQMarlinMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_gptq import MlxGPTQLinear
    from gptqmodel.quantization import FORMAT

    in_features, out_features = 128, 64
    effective_group = in_features if group_size == -1 else group_size
    groups = in_features // effective_group
    rng = np.random.default_rng(7200 + bits * 10 + effective_group)
    codes = rng.integers(
        0, 1 << bits, (in_features, out_features), dtype=np.uint32,
    )
    zeros = np.full((groups, out_features), 1 << (bits - 1), dtype=np.uint32)
    scales = rng.uniform(0.0002, 0.001, zeros.shape).astype(np.float16)
    source = GPTQMarlinMlxQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
        dtype=torch.float16,
        format=FORMAT.MARLIN,
    )
    source.qweight.copy_(torch.from_numpy(_pack_gptq_rows(codes, bits)))
    source.qzeros.copy_(torch.from_numpy(
        _pack_gptq_columns((zeros - 1) & ((1 << bits) - 1), bits),
    ))
    source.scales.copy_(torch.from_numpy(scales))
    source.bias.copy_(torch.linspace(-0.01, 0.01, out_features, dtype=torch.float16))

    model = _load(monkeypatch, source)
    assert isinstance(model.linear, MlxGPTQLinear)
    inputs = mx.array(
        rng.normal(0, 0.1, (2, 3, in_features)).astype(np.float32),
    ).astype(dtype)
    actual = model(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    weight = (
        torch.from_numpy(codes.astype(np.int64)).double()
        - torch.from_numpy(zeros).double().repeat_interleave(effective_group, dim=0)
    ) * torch.from_numpy(scales).double().repeat_interleave(effective_group, dim=0)
    expected = _rounded_oracle(inputs, weight.T, source.bias, dtype)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("rows", (1, 16))
@pytest.mark.parametrize("bits", (2, 4, 8))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_gptq_format_runtime_qwen38_27b_shapes(
    name, out_features, in_features, bits, rows, dtype,
):
    del name
    from gptqmodel.nn_modules.qlinear.mlx_gptq import MlxGPTQLinear

    group_size = 128
    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=group_size, bits=bits,
    )
    codes_per_word = 32 // bits
    shifts = np.arange(codes_per_word, dtype=np.uint32) * bits
    codes = np.resize(np.array([0, 2], dtype=np.uint32), in_features)
    packed_row = np.bitwise_or.reduce(
        codes.reshape(-1, codes_per_word) << shifts,
        axis=-1,
    )
    linear.weight = mx.array(
        np.broadcast_to(packed_row, (out_features, packed_row.size)).copy(),
    )
    linear.scales = mx.full(
        (out_features, in_features // group_size), 0.002, dtype=mx.float32,
    )
    linear.biases = mx.full(
        (out_features, in_features // group_size), -0.002, dtype=mx.float32,
    )
    layer = MlxGPTQLinear(linear)
    rng = np.random.default_rng(380027 + in_features + out_features + rows)
    inputs = mx.array(
        rng.normal(0, 0.01, (rows, in_features)).astype(np.float32),
    ).astype(dtype)
    actual = layer(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    values = np.tile(np.array([-0.002, 0.002], dtype=np.float64), in_features // 2)
    host_inputs = np.asarray(inputs.astype(mx.float32)).astype(np.float64)
    with np.errstate(all="ignore"):
        expected_column = host_inputs @ values
    expected = np.broadcast_to(expected_column[:, None], (rows, out_features)).copy()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    expected = torch.from_numpy(expected).to(target).float().numpy()
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )
    del layer, actual
    mx.clear_cache()
    gc.collect()


def test_gptq_and_awq_reject_one_bit_kernel_selection():
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear
    from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
    from gptqmodel.nn_modules.qlinear.mlx import GPTQBitBLASMlxQuantLinear
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    assert 1 not in BitblasLinear.SUPPORTS_BITS
    assert 1 not in AWQBitBlasKernel.SUPPORTS_BITS
    assert 1 not in GPTQBitBLASMlxQuantLinear.SUPPORTS_BITS
    with pytest.raises(ValueError):
        select_quant_linear(
            bits=1,
            group_size=128,
            desc_act=False,
            sym=True,
            device=DEVICE.MPS,
            backend=BACKEND.MLX,
            format=FORMAT.BITBLAS,
            quant_method=METHOD.GPTQ,
            pack_dtype=torch.int32,
            dtype=torch.float16,
        )
