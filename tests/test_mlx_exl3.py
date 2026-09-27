# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
"""EXL3 packed MLX conversion and inference checks."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
@pytest.mark.parametrize("bits", range(1, 9), ids=lambda bits: f"{bits}bit")
def test_exl3_mlx_packed_matches_torch_decoder(monkeypatch, bits, codebook, dtype, record_property):
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear
    from gptqmodel.utils import mlx as mlx_utils

    generator = torch.Generator(device="cpu").manual_seed(90 + bits)
    dims = 128
    tensors = {
        "trellis": torch.randint(-32768, 32767, (dims // 16, dims // 16, bits * 16),
                                 dtype=torch.int16, generator=generator),
        "suh": torch.randint(0, 2, (dims,), dtype=torch.int8, generator=generator).half() * 2 - 1,
        "svh": torch.randint(0, 2, (dims,), dtype=torch.int8, generator=generator).half() * 2 - 1,
        "bias": torch.randn(dims, dtype=torch.float16, generator=generator) * 0.01,
    }
    if codebook != "3inst":
        tensors[codebook] = torch.tensor([1], dtype=torch.uint32)
    source = torch.nn.Module()
    source.linear = ExllamaV3TorchLinear.from_tensors(
        in_features=dims, out_features=dims, name="linear", tensors=tensors,
    ).eval()

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(dims, dims, bias=True)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, MlxEXL3Linear)
    assert model.linear.bits == bits
    assert model.linear.codebook == codebook
    assert config["_gptqmodel_custom_mlx_runtime"]
    assert "quantization" not in config
    reference_weight = source.linear.get_weight_tensor(dtype=torch.float16)
    np.testing.assert_array_equal(np.array(model.linear.trellis), tensors["trellis"].numpy())
    np.testing.assert_array_equal(np.array(model.linear.suh), tensors["suh"].numpy())
    np.testing.assert_array_equal(np.array(model.linear.svh), tensors["svh"].numpy())
    assert not hasattr(model.linear, "linear")
    rng = np.random.default_rng(73)
    x = rng.normal(0, 0.15, (2, 3, dims)).astype(np.float32)
    mlx_input = mx.array(x).astype(dtype)
    actual = model(mlx_input)
    mx.eval(actual)
    assert actual.dtype == dtype
    torch_input = torch.from_numpy(np.asarray(mlx_input.astype(mx.float32))).double()
    expected = (torch_input @ reference_weight.double() + tensors["bias"].double()).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    visible = np.asarray(actual.astype(mx.float32))
    max_abs = float(np.max(np.abs(visible - expected)))
    record_property("max_abs_vs_torch_dense", max_abs)
    np.testing.assert_allclose(visible, expected, rtol=0.01, atol=0.01)


def test_exl3_mlx_unpacks_legacy_sign_bitfields(monkeypatch):
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.utils import mlx as mlx_utils

    dims = 128
    su = torch.tensor([0xA55A] * (dims // 16), dtype=torch.uint16).view(torch.int16)
    sv = torch.tensor([0x5AA5] * (dims // 16), dtype=torch.uint16).view(torch.int16)
    tensors = {
        "trellis": torch.zeros((dims // 16, dims // 16, 64), dtype=torch.int16),
        "su": su,
        "sv": sv,
    }
    source = torch.nn.Module()
    source.linear = ExllamaV3TorchLinear.from_tensors(
        in_features=dims, out_features=dims, name="linear", tensors=tensors,
    ).eval()

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(dims, dims, bias=False)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    model, _ = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    masks = 1 << torch.arange(16, dtype=torch.int32)
    expected_su = torch.where((su.view(torch.uint16).int()[:, None] & masks) != 0, -1.0, 1.0)
    expected_sv = torch.where((sv.view(torch.uint16).int()[:, None] & masks) != 0, -1.0, 1.0)
    np.testing.assert_array_equal(np.array(model.linear.suh), expected_su.flatten().half().numpy())
    np.testing.assert_array_equal(np.array(model.linear.svh), expected_sv.flatten().half().numpy())


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
@pytest.mark.parametrize("bits", range(1, 9), ids=lambda bits: f"{bits}bit")
def test_exl3_mlx_fused_row_matches_torch_decoder(bits, codebook, dtype, record_property):
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear

    generator = torch.Generator(device="cpu").manual_seed(313 + bits)
    in_features, out_features = 128, 256
    tensors = {
        "trellis": torch.randint(
            -32768, 32767,
            (in_features // 16, out_features // 16, bits * 16),
            dtype=torch.int16, generator=generator,
        ),
        "suh": torch.randint(
            0, 2, (in_features,), dtype=torch.int8, generator=generator,
        ).half() * 2 - 1,
        "svh": torch.randint(
            0, 2, (out_features,), dtype=torch.int8, generator=generator,
        ).half() * 2 - 1,
        "bias": torch.randn(out_features, dtype=torch.float16, generator=generator) * 0.01,
    }
    if codebook != "3inst":
        tensors[codebook] = torch.tensor([1], dtype=torch.uint32)
    reference = ExllamaV3TorchLinear.from_tensors(
        in_features=in_features, out_features=out_features,
        name="reference", tensors=tensors,
    )
    layer = MlxEXL3Linear(in_features, out_features, bits, codebook, bias=True)
    layer.trellis = mx.array(tensors["trellis"].numpy())
    layer.suh = mx.array(tensors["suh"].numpy())
    layer.svh = mx.array(tensors["svh"].numpy())
    layer.bias = mx.array(tensors["bias"].numpy())
    x = mx.array(
        np.random.default_rng(313 + bits).normal(0, 0.15, (1, in_features)).astype(np.float32),
    ).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype
    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    expected = (torch_input @ reference.get_weight_tensor(dtype=torch.float32).double()
                + tensors["bias"].double()).to(torch_dtype).float().numpy()
    visible = np.asarray(actual.astype(mx.float32))
    max_abs = float(np.max(np.abs(visible - expected)))
    record_property("max_abs_vs_torch_dense", max_abs)
    np.testing.assert_allclose(visible, expected, rtol=0.01, atol=0.01)
