# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant reference: z-lab, https://github.com/z-lab/paroquant
"""Synchronized ParoQuant MPS fallback versus packed MLX at Qwen3.8-27B shapes.

Run with the repository environment's Python. The Torch fallback is the same
ParoLinear implementation in origin/main; only the packed MLX path is new.
"""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx import ParoMlxQuantLinear
from gptqmodel.nn_modules.qlinear.mlx_paro import MlxParoLinear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.quantization import FORMAT
from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def milliseconds(fn, synchronize, repeats):
    synchronize(fn())
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        synchronize(fn())
        durations.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations)


def benchmark(name, output_dims, input_dims, rows, repeats, krot, rotate):
    source = ParoLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True, dtype=torch.float16, format=FORMAT.PAROQUANT,
        krot=krot,
    )
    source.qweight.fill_(np.array(0x98765432, dtype=np.uint32).view(np.int32).item())
    source.qzeros.fill_(np.array(0x88888888, dtype=np.uint32).view(np.int32).item())
    source.scales.fill_(0.01)
    if rotate:
        source.theta.fill_(0.01)
        source.channel_scales.fill_(1.01)
    assert ParoMlxQuantLinear.source_compatible(source)
    packed, scales, biases, params = ParoMlxQuantLinear.pack_source(source)
    native = nn.QuantizedLinear(input_dims, output_dims, bias=False, **params)
    native.load_weights([
        ("weight", mx.array(packed)), ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    native = MlxParoLinear(native, np.asarray(source.pairs), np.asarray(source.theta),
                           np.asarray(source.channel_scales), source.group_size)
    source = source.to("mps")
    source.post_init()
    rng = np.random.default_rng(380027 + rows)
    inputs = rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16)
    torch_inputs = torch.from_numpy(inputs).to("mps")
    mlx_inputs = mx.array(inputs)
    torch_output = source(torch_inputs)
    torch.mps.synchronize()
    mlx_output = native(mlx_inputs)
    mx.eval(mlx_output)
    max_error = float(np.max(np.abs(torch_output.cpu().numpy().astype(np.float32)
                                    - np.asarray(mlx_output).astype(np.float32))))
    if not np.allclose(torch_output.cpu().numpy(), np.asarray(mlx_output), rtol=0.002, atol=0.002):
        raise AssertionError(f"{name} rows={rows}: Torch/MLX error {max_error}")
    torch_ms = milliseconds(lambda: source(torch_inputs), lambda _: torch.mps.synchronize(), repeats)
    mlx_ms = milliseconds(lambda: native(mlx_inputs), mx.eval, repeats)
    result = {"projection": name, "out": output_dims, "in": input_dims,
              "rows": rows, "krot": krot, "rotated": rotate,
              "torch_main_ms": round(torch_ms, 3),
              "mlx_ms": round(mlx_ms, 3), "speedup": round(torch_ms / mlx_ms, 2),
              "max_abs_error": round(max_error, 6)}
    torch.mps.empty_cache()
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--krot", type=int, default=1)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--projection", action="append", help="Limit to named projections")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for rows in args.rows:
            print(json.dumps(benchmark(name, output_dims, input_dims, rows,
                                       args.repeats, args.krot, args.rotate)), flush=True)


if __name__ == "__main__":
    main()
