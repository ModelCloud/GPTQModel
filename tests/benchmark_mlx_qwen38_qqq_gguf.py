# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ reference: vLLM, Apache-2.0, https://github.com/vllm-project/vllm
# GGUF reference: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Compare main Torch MPS and packed MLX QQQ/GGUF at Qwen3.8-27B shapes."""

import argparse
import gc
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from benchmark_mlx_qwen38_27b import milliseconds
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.mlx import GGUFMlxQuantLinear, QQQMlxQuantLinear
from gptqmodel.nn_modules.qlinear.mlx_qqq import MlxQQQLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def build_qqq(input_dims, output_dims):
    source = QQQTorchLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True, dtype=torch.float16,
    )
    source.B.fill_(0x11111111)
    source.s_channel.fill_(0.001)
    source.s_group.fill_(1)
    assert QQQMlxQuantLinear.source_compatible(source)
    packed, scales, biases, params = QQQMlxQuantLinear.pack_source(source)
    native = nn.QuantizedLinear(input_dims, output_dims, bias=False, **params)
    native.load_weights([
        ("weight", mx.array(packed)), ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    _, channel_scale = source._dequantize_weight_for_torch()
    return source, MlxQQQLinear(native, channel_scale.numpy())


def build_gguf(input_dims, output_dims):
    source = GGUFTorchLinear(
        bits="q4_0", group_size=-1, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True,
    )
    block = np.zeros((1, 1, 18), dtype=np.uint8)
    block[..., :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    block[..., 2:] = 0x98
    source.qweight.copy_(torch.from_numpy(np.tile(block, (output_dims, input_dims // 32, 1))
                                          .reshape(tuple(source.qweight.shape))))
    assert GGUFMlxQuantLinear.source_compatible(source)
    packed, scales, biases, params = GGUFMlxQuantLinear.pack_source(source)
    native = nn.QuantizedLinear(input_dims, output_dims, bias=False, **params)
    native.load_weights([
        ("weight", mx.array(packed)), ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    return source, native


def benchmark(method, name, output_dims, input_dims, rows, repeats):
    source, native = (build_qqq if method == "qqq" else build_gguf)(input_dims, output_dims)
    source = source.to("mps")
    rng = np.random.default_rng(380027 + rows)
    inputs = rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16)
    torch_inputs = torch.from_numpy(inputs).to("mps")
    mlx_inputs = mx.array(inputs)
    torch_output = source(torch_inputs)
    torch.mps.synchronize()
    mlx_output = native(mlx_inputs)
    mx.eval(mlx_output)
    torch_values = torch_output.cpu().numpy()
    mlx_values = np.asarray(mlx_output)
    max_error = float(np.max(np.abs(torch_values.astype(np.float32) - mlx_values.astype(np.float32))))
    if not np.allclose(torch_values, mlx_values, rtol=0.002, atol=0.002):
        raise AssertionError(f"{method} {name} rows={rows}: output error {max_error}")
    torch_ms = milliseconds(lambda: source(torch_inputs), lambda _: torch.mps.synchronize(), repeats)
    mlx_ms = milliseconds(lambda: native(mlx_inputs), mx.eval, repeats)
    result = {"method": method, "projection": name, "out": output_dims, "in": input_dims,
              "rows": rows, "torch_main_ms": round(torch_ms, 3),
              "mlx_ms": round(mlx_ms, 3), "speedup": round(torch_ms / mlx_ms, 2),
              "max_abs_error": round(max_error, 6)}
    torch.mps.empty_cache()
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qqq", "gguf"), required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for rows in args.rows:
            print(json.dumps(benchmark(args.method, name, output_dims, input_dims, rows,
                                       args.repeats)), flush=True)


if __name__ == "__main__":
    main()
