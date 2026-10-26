# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Check GGUF affine output dtype and drift against independent Torch math."""

import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.gguf import _dequantize_gguf_tensor_numpy  # noqa: E402
from gptqmodel.utils.mlx_gguf_packing import repack_gguf_affine  # noqa: E402


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_gguf_q4_0_qwen38_outputs_preserve_dtype_and_match_torch(
    name, out_features, in_features, dtype, record_property,
):
    group_size = 32
    rng = np.random.default_rng(sum(name.encode()) + out_features + in_features + 74)
    blocks = rng.integers(0, 256, (out_features, in_features // group_size, 18), dtype=np.uint8)
    source_scales = rng.uniform(0.0005, 0.002, blocks.shape[:2]).astype(np.float16)
    blocks[..., :2] = source_scales.view(np.uint8).reshape(*blocks.shape[:2], 2)
    source_weight = blocks.reshape(out_features, -1)
    packed, scales, biases, params = repack_gguf_affine(source_weight, "Q4_0", in_features)
    torch_weight = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(source_weight, "Q4_0").copy(),
    ).double()

    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=params["group_size"],
        bits=params["bits"], mode=params["mode"],
    )
    linear.weight = mx.array(packed)
    linear.scales = mx.array(scales)
    linear.biases = mx.array(biases)
    layer = MlxGGUFLinear(linear)

    x_source = rng.normal(0, 0.15, (3, in_features)).astype(np.float32)
    x = mx.array(x_source).astype(dtype)
    internal = linear(x)
    mx.eval(internal)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype

    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    raw_oracle = (torch_input @ torch_weight.T).numpy()
    rounded_oracle = (torch_input @ torch_weight.T).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    internal_values = np.asarray(internal.astype(mx.float32))
    visible = np.asarray(actual.astype(mx.float32))
    record_property("max_abs_internal_fp32", float(np.max(np.abs(internal_values - raw_oracle))))
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - rounded_oracle))))
    record_property("max_abs_vs_fp64_torch", float(np.max(np.abs(visible - raw_oracle))))
    np.testing.assert_allclose(internal_values, raw_oracle, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded_oracle, rtol=2e-3, atol=2e-3)
