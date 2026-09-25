# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full Qwen3.8-27B projection checks for every merged GGUF Metal packer."""

import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
pytest.importorskip("torch")

from tests import test_mlx_gguf_quantization as gguf_tests  # noqa: E402


ORACLES = {
    "Q1_0": gguf_tests._torch_gguf_q1_0_oracle,
    "Q2_0": gguf_tests._torch_gguf_q2_0_oracle,
    "Q4_0": gguf_tests._torch_gguf_q4_0_oracle,
    "Q4_K": gguf_tests._torch_gguf_q4_k_oracle,
    "Q5_K": gguf_tests._torch_gguf_q5_k_oracle,
    "Q6_K": gguf_tests._torch_gguf_q6_k_oracle,
    "TQ1_0": gguf_tests._torch_gguf_tq1_0_oracle,
    "TQ2_0": gguf_tests._torch_gguf_tq2_0_oracle,
    "MXFP4": gguf_tests._torch_gguf_mxfp4_oracle,
    "Q8_0": gguf_tests._torch_gguf_q8_0_oracle,
}


@pytest.mark.parametrize("qtype,oracle", ORACLES.items())
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_gguf_qwen38_27b_complete_packed_bytes(
    qtype, oracle, name, out_features, in_features,
):
    """Compare all bytes while bounding Torch oracle peak memory by row chunks."""
    del name
    seed = 3800 + out_features + in_features
    generator = np.random.default_rng(seed)
    source = generator.normal(0, 0.5, (out_features, in_features)).astype(np.float32)
    weight = mx.array(source).astype(mx.bfloat16)
    mx.eval(weight)
    quantized_source = np.asarray(weight.astype(mx.float32))
    del source
    actual = np.asarray(gguf_tests.native.gguf_quantize_weight_mlx(weight, qtype))
    for start in range(0, out_features, 128):
        stop = min(start + 128, out_features)
        expected = oracle(quantized_source[start:stop])
        np.testing.assert_array_equal(actual[start:stop], expected)
