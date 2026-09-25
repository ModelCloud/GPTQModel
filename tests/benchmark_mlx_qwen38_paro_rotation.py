# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant reference: Z Lab, MIT, https://github.com/z-lab/paroquant
# MLX Metal runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
"""Compare merged and fused ParoQuant MLX rotations on Qwen projection shapes."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_paro import MlxParoLinear
from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _paired_milliseconds(main_fn, new_fn, repeats):
    for _ in range(3):
        mx.eval(main_fn(), new_fn())
    durations = ([], [])
    for iteration in range(repeats):
        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            mx.eval((main_fn if index == 0 else new_fn)())
            durations[index].append((time.perf_counter() - start) * 1000)
    return statistics.median(durations[0]), statistics.median(durations[1])


def _make_layer(output_dims, input_dims, krot):
    phase = np.arange(output_dims, dtype=np.uint8)[:, None] % 16
    index = np.arange(input_dims, dtype=np.uint8)[None, :] % 16
    codes = ((index + phase) % 16).astype(np.uint8)
    lanes = codes.reshape(output_dims, input_dims // 8, 8)
    packed = np.zeros((output_dims, input_dims // 8), dtype=np.uint32)
    for lane in range(8):
        packed |= lanes[:, :, lane].astype(np.uint32) << (4 * lane)
    linear = nn.QuantizedLinear(input_dims, output_dims, bias=False, group_size=128, bits=4)
    linear.load_weights([
        ("weight", mx.array(packed)),
        ("scales", mx.full((output_dims, input_dims // 128), 0.01, dtype=mx.float16)),
        ("biases", mx.full((output_dims, input_dims // 128), -0.08, dtype=mx.float32)),
    ])
    rng = np.random.default_rng(380027)
    pairs = np.empty((krot, input_dims), dtype=np.int16)
    for stage in range(krot):
        for group in range(input_dims // 128):
            pairs[stage, group * 128:(group + 1) * 128] = rng.permutation(128)
    theta = rng.uniform(-0.025, 0.025, (krot, input_dims // 2)).astype(np.float16)
    channel_scales = rng.uniform(0.95, 1.05, (1, input_dims)).astype(np.float16)
    return MlxParoLinear(linear, pairs, theta, channel_scales, 128)


def benchmark(name, output_dims, input_dims, rows, krot, repeats):
    layer = _make_layer(output_dims, input_dims, krot)
    rng = np.random.default_rng(380027 + rows)
    x = mx.array(rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16))
    # Reconstruct the exact merged-main runtime coefficients once, outside
    # the timed calls. The fused path keeps float32 trigonometric values.
    main_cosine = tuple(value.astype(mx.float16) for value in layer.cosine)
    main_sine = tuple(value.astype(mx.float16) for value in layer.sine)
    mx.eval(*main_cosine, *main_sine)

    def main_path():
        rotated = x * layer.channel_scales
        for partner, cosine, sine in zip(layer.partner, main_cosine, main_sine):
            old = rotated
            rotated = old * cosine + mx.take(old, partner, axis=-1) * sine
        return layer.linear(rotated)

    def new_path():
        return layer(x)

    old_result, new_result = main_path(), new_path()
    mx.eval(old_result, new_result)
    old_values, new_values = np.asarray(old_result), np.asarray(new_result)
    np.testing.assert_allclose(new_values, old_values, rtol=0.002, atol=0.002, err_msg=name)
    main_ms, new_ms = _paired_milliseconds(main_path, new_path, repeats)
    result = {
        "projection": name, "out": output_dims, "in": input_dims,
        "rows": rows, "krot": krot,
        "main_mlx_ms": round(main_ms, 3), "fused_mlx_ms": round(new_ms, 3),
        "speedup": round(main_ms / new_ms, 2),
        "max_abs_error": float(np.max(np.abs(old_values.astype(np.float32)
                                             - new_values.astype(np.float32)))),
    }
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--krot", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for krot in args.krot:
            for rows in args.rows:
                print(json.dumps(benchmark(name, output_dims, input_dims, rows,
                                           krot, args.repeats)), flush=True)


if __name__ == "__main__":
    main()
