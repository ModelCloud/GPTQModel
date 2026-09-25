# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# MLX dense matmul reference: Apple Inc., MIT, https://github.com/ml-explore/mlx
# bitsandbytes reference: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# MXFP8 matmul reference: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Weight-only FP8 and bitsandbytes MLX transfer checked against Torch."""

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("holder,bits,invalid_bits", [
    ("FP8MlxQuantLinear", 8, 4),
    ("BitsAndBytesMlxQuantLinear", 4, 3),
])
def test_dense_holder_validates_without_affine_group_limits(holder, bits, invalid_bits):
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear import mlx as mlx_holders

    cls = getattr(mlx_holders, holder)
    params = dict(
        group_size=-1, desc_act=False, sym=True,
        in_features=96, out_features=64, pack_dtype=torch.int32,
        dtype=torch.float16, device=DEVICE.MPS,
    )
    assert cls.validate(bits=bits, **params)[0]
    assert not cls.validate(bits=invalid_bits, **params)[0]


def _load_dense(monkeypatch, source, input_dims, output_dims):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8DenseLinear, MlxFP8Linear
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
    if isinstance(source, TorchFP8Linear) and source.fp8_format == "float8_e4m3fn" and source.weight_scale_method in {"row", "tensor"}:
        assert isinstance(model.linear, MlxFP8Linear)
        assert config["_gptqmodel_custom_mlx_runtime"]
    else:
        assert "quantization" not in config
        if isinstance(source, TorchFP8Linear):
            assert isinstance(model.linear, MlxFP8DenseLinear)
            assert config["_gptqmodel_custom_mlx_runtime"]
        else:
            assert isinstance(model.linear, nn.Linear)
    return model


@pytest.mark.parametrize("fp8_format", ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"])
@pytest.mark.parametrize("scale_method,block_size", [("tensor", None), ("row", None), ("block", (16, 32))])
def test_fp8_mlx_dense_matches_torch(monkeypatch, fp8_format, scale_method, block_size):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear

    torch.manual_seed(91)
    linear = torch.nn.Linear(128, 64, bias=True).half()
    source = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=128, out_features=64, bias=True,
        dtype=torch.float16, format=fp8_format,
        weight_scale_method=scale_method, weight_block_size=block_size,
    )
    source.pack_original(linear, None, None)
    assert FP8MlxQuantLinear.source_compatible(source)
    model = _load_dense(monkeypatch, source, 128, 64)
    reference_weight = source.dequantize_weight(device="cpu", dtype=torch.float16).T
    if scale_method in {"row", "tensor"} and fp8_format == "float8_e4m3fn":
        packed, scales, _, _ = FP8MlxQuantLinear.pack_source(source)
        np.testing.assert_array_equal(np.array(model.linear.linear.weight), packed)
        np.testing.assert_array_equal(np.array(model.linear.linear.scales), scales)
    else:
        np.testing.assert_array_equal(np.array(model.linear.linear.weight), reference_weight.numpy())
    rng = np.random.default_rng(210)
    x = rng.normal(0, 0.2, (2, 3, 128)).astype(np.float16)
    actual = model(mx.array(x))
    mx.eval(actual)
    expected = torch.from_numpy(x).double() @ reference_weight.double().T + source.bias.double()
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=0.002, atol=0.002)


@pytest.mark.parametrize("fp8_format", ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"])
@pytest.mark.parametrize("scale_method,block_size", [("tensor", None), ("row", None), ("block", (16, 32))])
def test_fp8_mlx_preserves_bfloat16_activation(monkeypatch, fp8_format, scale_method, block_size):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    source = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=128, out_features=64, bias=True,
        dtype=torch.float16, format=fp8_format,
        weight_scale_method=scale_method, weight_block_size=block_size,
    )
    source.pack_original(torch.nn.Linear(128, 64, bias=True).half(), None, None)
    model = _load_dense(monkeypatch, source, 128, 64)
    actual = model(mx.ones((2, 128), dtype=mx.bfloat16))
    mx.eval(actual)
    assert actual.dtype == mx.bfloat16


def test_fp8_e8m0_checkpoint_decodes_for_mlx(monkeypatch):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("This Torch build has no E8M0 storage dtype")
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    source = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=128, out_features=64, bias=True,
        dtype=torch.float16, format="float8_e8m0fnu", weight_scale_method="row",
    )
    source.weight.copy_(torch.full((64, 128), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu))
    source.weight_scale_inv.fill_(2)
    source.bias.fill_(0.01)
    model = _load_dense(monkeypatch, source, 128, 64)
    reference_weight = source.dequantize_weight(device="cpu", dtype=torch.float16).T
    np.testing.assert_array_equal(np.array(model.linear.linear.weight), reference_weight.numpy())
    output = model(mx.ones((1, 128), dtype=mx.float16))
    mx.eval(output)
    expected = torch.ones((1, 128), dtype=torch.float64) @ reference_weight.double().T + source.bias.double()
    np.testing.assert_allclose(np.array(output), expected.numpy(), rtol=0.002, atol=0.002)


@pytest.mark.parametrize("bits,quant_type,compress", [
    (4, "nf4", False), (4, "nf4", True), (4, "fp4", False), (4, "fp4", True),
    (8, "int8", False),
])
def test_bitsandbytes_mlx_dense_matches_torch(monkeypatch, bits, quant_type, compress):
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
    model = _load_dense(monkeypatch, source, 128, 64)
    reference_weight = source.dequantize_weight().detach().to("cpu", torch.float16)
    np.testing.assert_array_equal(np.array(model.linear.weight), reference_weight.numpy())
    rng = np.random.default_rng(117)
    x = rng.normal(0, 0.2, (2, 3, 128)).astype(np.float16)
    actual = model(mx.array(x))
    mx.eval(actual)
    expected = torch.from_numpy(x).double() @ reference_weight.double().T + source.bias.double()
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=0.002, atol=0.002)
