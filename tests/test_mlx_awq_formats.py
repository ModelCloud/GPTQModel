# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq
# Marlin format: IST-DASLab contributors, MIT, https://github.com/IST-DASLab/marlin
# BitBLAS format: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""MLX inference for AWQ Marlin and BitBLAS checkpoint layouts."""

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


FORMAT_CASES = (
    ("marlin", 4, "AWQMarlinMlxQuantLinear"),
    ("marlin", 8, "AWQMarlinMlxQuantLinear"),
    *(("bitblas", bits, "AWQBitBLASMlxQuantLinear") for bits in range(2, 9)),
)


def _pack_bytes(codes, bits):
    code_bits = (
        codes.astype(np.uint16)[..., None]
        >> np.arange(bits, dtype=np.uint16)
    ) & 1
    return np.packbits(
        code_bits.reshape(*codes.shape[:-1], -1), axis=-1, bitorder="little",
    ).view(np.int8)


def _repeated_word(code, bits):
    value = sum(int(code) << shift for shift in range(0, 32, bits))
    return np.asarray(value, dtype=np.uint32).view(np.int32).item()


def _source(fmt, bits, group_size, sym, desc_act, in_features, out_features, bias):
    from gptqmodel.nn_modules.qlinear.mlx import (
        AWQBitBLASMlxQuantLinear,
        AWQMarlinMlxQuantLinear,
    )

    cls = AWQMarlinMlxQuantLinear if fmt == "marlin" else AWQBitBLASMlxQuantLinear
    return cls(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        register_buffers=True,
    )


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
        output = torch.from_numpy(np.asarray(inputs.astype(mx.float32))).double() @ weight
    if bias is not None:
        output += bias.double()
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return output.to(target).float().numpy()


@pytest.mark.parametrize("fmt,bits,expected", FORMAT_CASES)
def test_awq_checkpoint_formats_auto_select_mlx(fmt, bits, expected):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.loader import _auto_select_mlx_backend
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    class Config:
        def to_dict(self):
            return {"model_type": "qwen3_5"}

    qcfg = SimpleNamespace(
        bits=bits,
        group_size=128,
        desc_act=False,
        sym=False,
        pack_dtype=torch.int32,
        dynamic=None,
        rotation=None,
    )
    selected = select_quant_linear(
        bits=bits,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.MPS,
        backend=BACKEND.MLX,
        format=FORMAT(fmt),
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert selected.__name__ == expected
    assert _auto_select_mlx_backend(
        BACKEND.AUTO,
        DEVICE.MPS,
        Config(),
        qcfg,
        METHOD.AWQ,
        FORMAT(fmt),
        None,
    ) == BACKEND.MLX


@pytest.mark.parametrize(
    "fmt,bits",
    (("marlin", 2), ("marlin", 3), ("marlin", 5), ("marlin", 6),
     ("marlin", 7), ("bitblas", 1), ("bitblas", 9)),
)
def test_awq_checkpoint_formats_reject_unsupported_bits(fmt, bits):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    with pytest.raises(ValueError):
        select_quant_linear(
            bits=bits,
            group_size=128,
            desc_act=False,
            sym=False,
            device=DEVICE.MPS,
            backend=BACKEND.MLX,
            format=FORMAT(fmt),
            quant_method=METHOD.AWQ,
            pack_dtype=torch.int32,
            dtype=torch.float16,
        )


@pytest.mark.parametrize("fmt,bits,_expected", FORMAT_CASES)
@pytest.mark.parametrize("group_size", (-1, 32, 64, 128))
@pytest.mark.parametrize("sym", (True, False), ids=("sym", "asym"))
@pytest.mark.parametrize("desc_act", (True, False), ids=("desc_act", "no_desc_act"))
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "bias"))
def test_awq_formats_all_capabilities_match_torch(
    monkeypatch,
    fmt,
    bits,
    _expected,
    group_size,
    sym,
    desc_act,
    dtype,
    use_bias,
):
    from gptqmodel.quantization.awq.utils.packing_utils import pack_awq

    in_features, out_features = 128, 64
    effective_group = in_features if group_size == -1 else group_size
    groups = in_features // effective_group
    rng = np.random.default_rng(
        3234 + bits * 100 + effective_group + int(sym) * 7 + int(desc_act) * 13,
    )
    codes = rng.integers(
        0, 1 << bits, (in_features, out_features), dtype=np.uint8,
    )
    zeros = rng.integers(
        0, 1 << bits, (groups, out_features), dtype=np.uint8,
    )
    scales = rng.uniform(0.0002, 0.001, (groups, out_features)).astype(np.float16)
    source = _source(
        fmt, bits, group_size, sym, desc_act, in_features, out_features, use_bias,
    )
    if fmt == "marlin":
        qweight, qzeros = pack_awq(
            torch.from_numpy(codes.astype(np.int32)),
            torch.from_numpy(zeros.astype(np.int32)),
            bits,
        )
        source.qweight.copy_(qweight)
        source.qzeros.copy_(qzeros)
        source.scales.copy_(torch.from_numpy(scales))
    else:
        source.qweight.copy_(torch.from_numpy(_pack_bytes(codes.T, bits)))
        source.qzeros.copy_(torch.from_numpy(_pack_bytes(zeros, bits)))
        source.scales.copy_(torch.from_numpy(scales.T))
    if use_bias:
        source.bias.copy_(
            torch.linspace(-0.01, 0.01, out_features, dtype=torch.float16),
        )

    model = _load(monkeypatch, source)
    from gptqmodel.nn_modules.qlinear.mlx_awq import MlxAWQLinear

    assert isinstance(model.linear, MlxAWQLinear)
    target_bits = 8 if bits == 7 else bits
    assert model.linear.linear.weight.nbytes == (
        out_features * in_features * target_bits // 8
    )
    inputs = mx.array(
        rng.normal(0, 0.1, (2, 3, in_features)).astype(np.float32),
    ).astype(dtype)
    actual = model(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    weight = (
        torch.from_numpy(codes.astype(np.int16)).double()
        - torch.from_numpy(zeros).double().repeat_interleave(effective_group, dim=0)
    ) * torch.from_numpy(scales).double().repeat_interleave(effective_group, dim=0)
    expected = _rounded_oracle(inputs, weight, source.bias, dtype)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )


@pytest.mark.parametrize("fmt,bits,_expected", FORMAT_CASES)
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("rows", (1, 16), ids=("decode", "prefill16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_awq_formats_qwen38_projection_accuracy(
    monkeypatch,
    fmt,
    bits,
    _expected,
    dtype,
    rows,
    name,
    out_features,
    in_features,
):
    del name
    group_size = 128
    source = _source(
        fmt, bits, group_size, False, False, in_features, out_features, True,
    )
    low = (1 << (bits - 1)) - 1
    high = low + 2
    zero = 1 << (bits - 1)
    if fmt == "marlin":
        source.qweight[0::2].fill_(_repeated_word(low, bits))
        source.qweight[1::2].fill_(_repeated_word(high, bits))
        source.qzeros.fill_(_repeated_word(zero, bits))
        source.scales.fill_(0.002)
    else:
        weight_row = np.resize(
            np.asarray((low, high), dtype=np.uint8), in_features,
        )[None, :]
        zero_row = np.full((1, out_features), zero, dtype=np.uint8)
        source.qweight.copy_(
            torch.from_numpy(np.broadcast_to(
                _pack_bytes(weight_row, bits), source.qweight.shape,
            ).copy()),
        )
        source.qzeros.copy_(
            torch.from_numpy(np.broadcast_to(
                _pack_bytes(zero_row, bits), source.qzeros.shape,
            ).copy()),
        )
        source.scales.fill_(0.002)
    source.bias.copy_(
        torch.linspace(-0.01, 0.01, out_features, dtype=torch.float16),
    )

    model = _load(monkeypatch, source)
    rng = np.random.default_rng(380027 + in_features + out_features + bits + rows)
    inputs = mx.array(
        rng.normal(0, 0.01, (rows, in_features)).astype(np.float32),
    ).astype(dtype)
    actual = model(inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    values = torch.tensor([-0.002, 0.002], dtype=torch.float64).repeat(
        in_features // 2,
    )
    weight = values[:, None].expand(-1, out_features)
    expected = _rounded_oracle(inputs, weight, source.bias, dtype)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002,
    )
    del model, source, actual
    mx.clear_cache()
    gc.collect()
