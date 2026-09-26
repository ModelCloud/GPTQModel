# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Check decoded EXL3 MLX outputs against independent Torch math."""

import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear  # noqa: E402


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_qwen38_outputs_preserve_dtype_and_match_torch(
    name, out_features, in_features, dtype, record_property,
):
    rng = np.random.default_rng(sum(name.encode()) + out_features + in_features + 173)
    weight = rng.normal(0, 0.015, (out_features, in_features)).astype(np.float16)
    bias = rng.normal(0, 0.002, out_features).astype(np.float16)
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight = mx.array(weight)
    linear.bias = mx.array(bias)
    layer = MlxEXL3Linear(linear)

    x = mx.array(rng.normal(0, 0.15, (3, in_features)).astype(np.float32)).astype(dtype)
    base = linear(x)
    actual = layer(x)
    mx.eval(base, actual)
    assert actual.dtype == dtype

    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    torch_weight = torch.from_numpy(weight).double()
    torch_bias = torch.from_numpy(bias).double()
    torch_output = torch_input @ torch_weight.T + torch_bias
    raw_oracle = torch_output.numpy()
    rounded_oracle = torch_output.to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    base_values = np.asarray(base.astype(mx.float32))
    visible = np.asarray(actual.astype(mx.float32))
    record_property("base_output_dtype", str(base.dtype))
    record_property("max_abs_base_vs_fp64_torch", float(np.max(np.abs(base_values - raw_oracle))))
    record_property("max_abs_preserved_vs_rounded_torch", float(np.max(np.abs(visible - rounded_oracle))))
    record_property("max_abs_preserved_vs_fp64_torch", float(np.max(np.abs(visible - raw_oracle))))
    np.testing.assert_allclose(base_values, raw_oracle, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded_oracle, rtol=2e-3, atol=2e-3)
