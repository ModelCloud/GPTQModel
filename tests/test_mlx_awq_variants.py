# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ packing reference: ModelCloud.ai, Apache-2.0, gptqmodel/nn_modules/qlinear/gemv_fast_awq.py
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Independent code and output checks for both AWQ GEMV checkpoint layouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("variant,group_size", [
    ("gemv", 64), ("gemv", 128), ("gemv", -1),
    ("fast", 16), ("fast", 32), ("fast", 64), ("fast", 128), ("fast", -1),
    ("llm", 16), ("llm", 32), ("llm", 64), ("llm", 128), ("llm", -1),
])
def test_awq_gemv_packed_transfer_matches_codes_and_torch(variant, group_size, monkeypatch):
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
    from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear, pack_intweight
    from gptqmodel.nn_modules.qlinear.mlx import (AwqGemvFastMlxQuantLinear,
                                                  AwqGemvMlxQuantLinear, LLMAwqMlxQuantLinear)
    from gptqmodel.utils import mlx as mlx_utils

    input_dims, output_dims = 256, 64
    group = input_dims if group_size == -1 else group_size
    groups = input_dims // group
    rng = np.random.default_rng(2100 + group_size + {"gemv": 0, "fast": 10, "llm": 20}[variant])
    source_class = ({"gemv": AwqGEMVLinear, "fast": AwqGEMVFastLinear,
                     "llm": LLMAwqLinear}[variant] if not (variant == "gemv" and group_size == -1)
                    else AwqGemvMlxQuantLinear)
    holder_class = {"gemv": AwqGemvMlxQuantLinear, "fast": AwqGemvFastMlxQuantLinear,
                    "llm": LLMAwqMlxQuantLinear}[variant]
    source = torch.nn.Module()
    source.linear = source_class(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True, dtype=torch.float16,
    )
    codes = rng.integers(0, 16, (output_dims, input_dims), dtype=np.uint32)
    zeros = rng.integers(0, 16, (output_dims, groups), dtype=np.uint32)
    scales = rng.uniform(0.002, 0.015, (output_dims, groups)).astype(np.float16)
    codes[0, 0], codes[-1, -1] = 0, 15
    zeros[0, 0], zeros[-1, -1] = 15, 0
    if variant == "gemv":
        words = np.bitwise_or.reduce(
            codes.reshape(output_dims, -1, 8) << (np.arange(8, dtype=np.uint32) * 4), axis=-1,
        ).astype(np.int32)
        zero_words = np.bitwise_or.reduce(
            np.pad(zeros, ((0, 0), (0, (-groups) % 8))).reshape(output_dims, -1, 8)
            << (np.arange(8, dtype=np.uint32) * 4), axis=-1,
        ).astype(np.int32)
        source.linear.qweight.copy_(torch.from_numpy(words))
        source.linear.qzeros[:, :zero_words.shape[1]].copy_(torch.from_numpy(zero_words))
        source.linear.scales[:, :groups].copy_(torch.from_numpy(scales))
    else:
        source.linear.qweight.copy_(pack_intweight(torch.from_numpy(codes.astype(np.int32)), 4, 64))
        source.linear.scales[:groups].copy_(torch.from_numpy(scales.T))
        scaled_zeros = -(zeros.astype(np.float32) * scales.astype(np.float32)).astype(np.float16)
        source.linear._runtime_zeros()[:groups].copy_(torch.from_numpy(scaled_zeros.T))
    assert holder_class.source_compatible(source.linear)

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(input_dims, output_dims, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, SimpleNamespace(from_dict=lambda _: None)))
    model, _ = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    packed = np.asarray(model.linear.weight).astype(np.uint32)
    unpacked = ((packed[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(output_dims, input_dims)
    np.testing.assert_array_equal(unpacked, codes)

    inputs = rng.normal(0, 0.2, (2, 3, input_dims)).astype(np.float16)
    actual = model(mx.array(inputs))
    mx.eval(actual)
    decoded = ((codes.astype(np.float32) - np.repeat(zeros, group, axis=1))
               * np.repeat(scales, group, axis=1)).astype(np.float32)
    expected = (torch.from_numpy(inputs).float() @ torch.from_numpy(decoded).T).half().numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002)
