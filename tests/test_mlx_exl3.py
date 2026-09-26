# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
# MLX dense runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""EXL3 Torch decode into MLX dense inference with numerical checks."""

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_exl3_mlx_dense_matches_torch_decoder(monkeypatch, bits, dtype):
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
    assert config["_gptqmodel_custom_mlx_runtime"]
    assert "quantization" not in config
    reference_weight = source.linear.get_weight_tensor(dtype=torch.float16)
    np.testing.assert_array_equal(np.array(model.linear.linear.weight), reference_weight.T.numpy())
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
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002)
