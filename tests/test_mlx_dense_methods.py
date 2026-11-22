# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes reference: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
"""Packed bitsandbytes MLX transfer checked against Torch."""

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("holder,bits,invalid_bits", [
    ("BitsAndBytesMlxQuantLinear", 4, 3),
])
def test_dense_holder_validates_without_affine_group_limits(holder, bits, invalid_bits):
    if holder == "BitsAndBytesMlxQuantLinear":
        # The MLX BNB holder reuses BitsAndBytesLinear's dependency probe.
        pytest.importorskip("bitsandbytes")
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear import mlx as mlx_holders

    cls = getattr(mlx_holders, holder)
    params = {
        "group_size": -1, "desc_act": False, "sym": True,
        "in_features": 96, "out_features": 64, "pack_dtype": torch.int32,
        "dtype": torch.float16, "device": DEVICE.MPS,
    }
    assert cls.validate(bits=bits, **params)[0]
    assert not cls.validate(bits=invalid_bits, **params)[0]


def _load_weight_only(monkeypatch, source, input_dims, output_dims):
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
    from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import MlxBitsAndBytesLinear
    from gptqmodel.utils import mlx as mlx_utils

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(input_dims, output_dims, bias=True)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    holder = torch.nn.Module()
    holder.linear = source
    model, config = mlx_utils._packed_mlx_weights(holder, {}, "lm_head")
    assert "quantization" not in config
    if isinstance(source, BitsAndBytesLinear):
        assert isinstance(model.linear, MlxBitsAndBytesLinear)
        assert config["_gptqmodel_custom_mlx_runtime"]
    else:
        assert isinstance(model.linear, nn.Linear)
    return model


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bits,quant_type,compress", [
    (4, "nf4", False), (4, "nf4", True), (4, "fp4", False), (4, "fp4", True),
    (8, "int8", False),
])
def test_bitsandbytes_mlx_dense_matches_torch(monkeypatch, bits, quant_type, compress, dtype):
    pytest.importorskip("bitsandbytes")
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
    from gptqmodel.nn_modules.qlinear.mlx import BitsAndBytesMlxQuantLinear

    torch.manual_seed(43)
    linear = torch.nn.Linear(128, 64, bias=True).half()
    source = BitsAndBytesLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=128, out_features=64, bias=True,
        dtype=torch.float16, format=quant_type,
        block_size=64, compress_statistics=compress,
    )
    source.pack_original(linear, None, None)
    assert BitsAndBytesMlxQuantLinear.source_compatible(source)
    model = _load_weight_only(monkeypatch, source, 128, 64)
    reference_weight = source.dequantize_weight().detach().to("cpu", torch.float16)
    payload = BitsAndBytesMlxQuantLinear.native_payload(source)
    np.testing.assert_array_equal(np.asarray(model.linear.weight), np.asarray(payload["weight"]).reshape(model.linear.weight.shape))
    assert model.linear.weight.nbytes == source.weight.numel() * source.weight.element_size()
    rng = np.random.default_rng(117)
    x = rng.normal(0, 0.2, (2, 3, 128)).astype(np.float32)
    mlx_input = mx.array(x).astype(dtype)
    actual = model(mlx_input)
    mx.eval(actual)
    assert actual.dtype == dtype
    torch_input = torch.from_numpy(np.asarray(mlx_input.astype(mx.float32))).double()
    expected = (torch_input @ reference_weight.double().T + source.bias.double()).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("block_size", (32, 64, 128, 256, 512, 1024, 2048, 4096))
@pytest.mark.parametrize("quant_type", ("nf4", "fp4"))
@pytest.mark.parametrize("compress", (False, True), ids=("plain-scales", "nested-scales"))
def test_bitsandbytes_mlx_all_4bit_block_sizes(
    monkeypatch, dtype, block_size, quant_type, compress,
):
    pytest.importorskip("bitsandbytes")
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

    torch.manual_seed(97 + block_size)
    linear = torch.nn.Linear(block_size, 8, bias=False).half()
    source = BitsAndBytesLinear(
        bits=4, group_size=-1, sym=True, desc_act=False,
        in_features=block_size, out_features=8, bias=False,
        dtype=torch.float16, format=quant_type,
        block_size=block_size, compress_statistics=compress,
    )
    source.pack_original(linear, None, None)
    model = _load_weight_only(monkeypatch, source, block_size, 8)
    x = mx.array(np.random.default_rng(block_size).normal(0, 0.2, (2, block_size)).astype(np.float32)).astype(dtype)
    actual = model(x)
    mx.eval(actual)
    assert actual.dtype == dtype
    reference_weight = source.dequantize_weight().detach().to("cpu", torch.float64)
    expected = (torch.from_numpy(np.asarray(x.astype(mx.float32))).double() @ reference_weight.T).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bits,quant_type,compress", (
    (4, "nf4", False), (4, "fp4", True), (8, "int8", False),
))
def test_bitsandbytes_mlx_irregular_shape_fallback(
    monkeypatch, dtype, bits, quant_type, compress,
):
    pytest.importorskip("bitsandbytes")
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

    torch.manual_seed(211)
    linear = torch.nn.Linear(65, 3, bias=True).half()
    source = BitsAndBytesLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=65, out_features=3, bias=True,
        dtype=torch.float16, format=quant_type,
        block_size=32, compress_statistics=compress,
    )
    source.pack_original(linear, None, None)
    model = _load_weight_only(monkeypatch, source, 65, 3)
    assert "affine_biases" not in model.linear
    x = mx.array(np.random.default_rng(211).normal(0, 0.2, (2, 65)).astype(np.float32)).astype(dtype)
    actual = model(x)
    mx.eval(actual)
    reference_weight = source.dequantize_weight().detach().to("cpu", torch.float64)
    expected = (torch.from_numpy(np.asarray(x.astype(mx.float32))).double() @ reference_weight.T
                + source.bias.double()).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=2e-3, atol=2e-3)
