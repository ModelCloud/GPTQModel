# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant reference: Z Lab, MIT, https://github.com/z-lab/paroquant
"""Compare merged and fused ParoQuant MLX rotations on Qwen projection shapes."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_paro import MlxParoLinear, _rotate_stage
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


def _torch_oracle(output_dims, input_dims, krot, inputs):
    """Independently rotate and multiply the patterned packed checkpoint."""
    rng = np.random.default_rng(380027)
    pairs = np.empty((krot, input_dims), dtype=np.int16)
    for stage in range(krot):
        for group in range(input_dims // 128):
            pairs[stage, group * 128:(group + 1) * 128] = rng.permutation(128)
    theta = rng.uniform(-0.025, 0.025, (krot, input_dims // 2)).astype(np.float16)
    channels = rng.uniform(0.95, 1.05, (1, input_dims)).astype(np.float16)
    rotated = torch.from_numpy(inputs).double() * torch.from_numpy(channels).double()
    offsets = torch.arange(input_dims // 128).repeat_interleave(64) * 128
    for stage in range(krot):
        pair_row = torch.from_numpy(pairs[stage].reshape(-1, 2).astype(np.int64))
        first, second = pair_row[:, 0] + offsets, pair_row[:, 1] + offsets
        angle = torch.from_numpy(theta[stage]).double()
        cosine, sine = torch.cos(angle), torch.sin(angle)
        left, right = rotated[:, first], rotated[:, second]
        next_rotated = torch.empty_like(rotated)
        next_rotated[:, first] = left * cosine + right * sine
        next_rotated[:, second] = -left * sine + right * cosine
        rotated = next_rotated
    codes = (torch.arange(input_dims)[None, :] + torch.arange(16)[:, None]).remainder(16).double()
    weight = codes * float(np.float16(0.01)) + float(np.float32(-0.08))
    phases = rotated @ weight.T
    return phases[:, np.arange(output_dims) % 16].numpy()


def benchmark(name, output_dims, input_dims, rows, krot, repeats, dtype):
    layer = _make_layer(output_dims, input_dims, krot)
    rng = np.random.default_rng(380027 + rows)
    x = mx.array(rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16)).astype(dtype)

    def main_path():
        rotated = x
        if dtype == mx.float16:
            for stage, (partner, cosine, sine) in enumerate(zip(layer.partner, layer.cosine, layer.sine)):
                rotated = _rotate_stage(rotated, partner, cosine, sine, layer.channel_scales, stage == 0)
        else:
            rotated = rotated * layer.channel_scales
            for partner, cosine, sine in zip(layer.partner, layer.cosine, layer.sine):
                old = rotated
                rotated = old * cosine + mx.take(old, partner, axis=-1) * sine
        return layer.linear(rotated).astype(dtype)

    def new_path():
        return layer(x)

    old_result, new_result = main_path(), new_path()
    mx.eval(old_result, new_result)
    old_values = np.asarray(old_result.astype(mx.float32)).astype(np.float64)
    new_values = np.asarray(new_result.astype(mx.float32)).astype(np.float64)
    oracle = _torch_oracle(output_dims, input_dims, krot, np.asarray(x.astype(mx.float32)))
    np.testing.assert_allclose(new_values, oracle, rtol=0.002, atol=0.002, err_msg=name)
    main_ms, new_ms = _paired_milliseconds(main_path, new_path, repeats)
    result = {
        "projection": name, "out": output_dims, "in": input_dims,
        "rows": rows, "krot": krot, "dtype": str(dtype).split(".")[-1],
        "main_mlx_ms": round(main_ms, 3), "fused_mlx_ms": round(new_ms, 3),
        "speedup": round(main_ms / new_ms, 2),
        "main_max_abs": float(np.max(np.abs(old_values - oracle))),
        "fused_max_abs": float(np.max(np.abs(new_values - oracle))),
        "fused_vs_main": float(np.max(np.abs(new_values - old_values))),
    }
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--krot", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], nargs="+",
                        default=["float16", "bfloat16"])
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for krot in args.krot:
            for rows in args.rows:
                for dtype_name in args.dtype:
                    dtype = mx.float16 if dtype_name == "float16" else mx.bfloat16
                    print(json.dumps(benchmark(name, output_dims, input_dims, rows,
                                               krot, args.repeats, dtype)), flush=True)


if __name__ == "__main__":
    main()
