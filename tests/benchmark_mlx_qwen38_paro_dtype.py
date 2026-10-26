# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant reference: Z Lab, MIT, https://github.com/z-lab/paroquant
"""Benchmark ParoQuant's output dtype correction against merged main."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from benchmark_mlx_qwen38_paro_rotation import _make_layer
from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def torch_oracle(output_dims, input_dims, krot, inputs):
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


def timed(fn):
    started = time.perf_counter()
    mx.eval(fn())
    return (time.perf_counter() - started) * 1000


def benchmark(name, output_dims, input_dims, rows, krot, repeats, dtype):
    layer = _make_layer(output_dims, input_dims, krot)
    rng = np.random.default_rng(380027 + rows)
    x = mx.array(rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16)).astype(
        mx.float16 if dtype == "float16" else mx.bfloat16
    )
    def main_fn():
        return layer._forward_unrounded(x)

    def new_fn():
        return layer(x)
    old, new = main_fn(), new_fn()
    mx.eval(old, new)
    oracle = torch_oracle(output_dims, input_dims, krot, np.asarray(x.astype(mx.float32)))
    actual = np.asarray(new.astype(mx.float32)).astype(np.float64)
    baseline = np.asarray(old.astype(mx.float32)).astype(np.float64)
    error = actual - oracle
    for _ in range(5):
        mx.eval(main_fn(), new_fn())
    main_samples, new_samples = [], []
    for index in range(repeats):
        if index % 2:
            new_samples.append(timed(new_fn))
            main_samples.append(timed(main_fn))
        else:
            main_samples.append(timed(main_fn))
            new_samples.append(timed(new_fn))
    main_ms, new_ms = statistics.median(main_samples), statistics.median(new_samples)
    result = {
        "projection": name, "out": output_dims, "in": input_dims,
        "rows": rows, "krot": krot, "dtype": dtype,
        "main_output_dtype": str(old.dtype).split(".")[-1],
        "new_output_dtype": str(new.dtype).split(".")[-1],
        "main_ms": round(main_ms, 4), "new_ms": round(new_ms, 4),
        "speedup": round(main_ms / new_ms, 3),
        "main_max_abs": float(np.max(np.abs(baseline - oracle))),
        "new_max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(oracle)),
        "max_tolerance_ratio": float(np.max(np.abs(error) / (0.002 + 0.002 * np.abs(oracle)))),
    }
    if dtype == "float16" and result["max_tolerance_ratio"] > 1:
        raise AssertionError(f"{name} krot={krot} rows={rows}: tolerance exceeded")
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for krot in (1, 8):
            for rows in (1, 16):
                print(json.dumps(benchmark(name, output_dims, input_dims, rows, krot,
                                           args.repeats, args.dtype)), flush=True)


if __name__ == "__main__":
    main()
