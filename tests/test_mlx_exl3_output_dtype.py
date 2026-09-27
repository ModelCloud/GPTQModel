# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Check packed EXL3 MLX outputs on every Qwen3.8-27B projection shape."""

import gc
import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear  # noqa: E402


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("rows", (1, 16), ids=("decode", "prefill16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_qwen38_packed_outputs_preserve_dtype_and_match_torch(
    name, out_features, in_features, rows, dtype, record_property,
):
    bits = 4
    rng = np.random.default_rng(sum(name.encode()) + out_features + in_features + rows + 173)
    sign_input = np.where(np.arange(128) % 3, 1, -1).astype(np.float16)
    sign_output = np.where(np.arange(128) % 5, 1, -1).astype(np.float16)
    bias_tile = (np.sin(np.arange(128) * 0.17) * 0.002).astype(np.float16)
    tiny_tensors = {
        "trellis": torch.zeros((8, 8, bits * 16), dtype=torch.int16),
        "suh": torch.from_numpy(sign_input),
        "svh": torch.from_numpy(sign_output),
        "bias": torch.from_numpy(bias_tile),
        "mcg": torch.tensor([1], dtype=torch.uint32),
    }
    reference_layer = ExllamaV3TorchLinear.from_tensors(
        in_features=128, out_features=128, name="reference", tensors=tiny_tensors,
    )
    reference_weight = reference_layer.get_weight_tensor(dtype=torch.float32)

    layer = MlxEXL3Linear(in_features, out_features, bits, "mcg", bias=True)
    layer.trellis = mx.zeros(
        (in_features // 16, out_features // 16, bits * 16), dtype=mx.int16,
    )
    layer.suh = mx.array(np.tile(sign_input, in_features // 128))
    layer.svh = mx.array(np.tile(sign_output, out_features // 128))
    layer.bias = mx.array(np.tile(bias_tile, out_features // 128))

    x = mx.array(rng.normal(0, 0.015, (rows, in_features)).astype(np.float32)).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype

    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).float()
    folded_input = torch_input.view(rows, -1, 128).sum(dim=1)
    output_tile = folded_input @ reference_weight + torch.from_numpy(bias_tile).float()
    torch_output = output_tile.repeat(1, out_features // 128)
    rounded_oracle = torch_output.to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    visible = np.asarray(actual.astype(mx.float32))
    record_property("projection", name)
    record_property("rows", rows)
    record_property("max_abs_preserved_vs_rounded_torch", float(np.max(np.abs(visible - rounded_oracle))))
    record_property("max_abs_preserved_vs_fp32_torch", float(np.max(np.abs(visible - torch_output.numpy()))))
    np.testing.assert_allclose(visible, rounded_oracle, rtol=0.01, atol=0.01)
    del layer, actual
    gc.collect()
    mx.clear_cache()
