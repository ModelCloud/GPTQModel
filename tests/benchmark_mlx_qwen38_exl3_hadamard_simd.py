# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare SIMD-local EXL3 Hadamard stages with the current MLX kernel."""

import argparse
import gc
import statistics
import time
from functools import lru_cache, partial

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_exl3_hadamard import exl3_hadamard_128_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_hadamard import (
    _normalized_rms_drift,
    _torch_hadamard_oracle,
)


@lru_cache(maxsize=32)
def _main_kernel(columns, axis):
    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_hadamard_128_main_benchmark",
        input_names=["input"],
        output_names=["output"],
        source="""
            threadgroup float current[128];
            threadgroup float next_values[128];

            uint lane = thread_position_in_threadgroup.x;
            uint vector = threadgroup_position_in_grid.x;
            uint element;
            if (AXIS == 1) {
                uint row = vector / COLUMN_BLOCKS;
                uint column = (vector % COLUMN_BLOCKS) * 128 + lane;
                element = row * COLUMNS + column;
            } else {
                uint row = (vector / COLUMNS) * 128 + lane;
                uint column = vector % COLUMNS;
                element = row * COLUMNS + column;
            }

            current[lane] = input[element];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 1; stride < 128; stride <<= 1) {
                uint partner = lane ^ stride;
                float own = current[lane];
                float other = current[partner];
                next_values[lane] = (lane & stride) == 0
                    ? own + other
                    : other - own;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                current[lane] = next_values[lane];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            output[element] = current[lane] * 0.08838834764831845f;
        """,
    )


def _main_hadamard(matrix, *, axis):
    matrix = mx.array(matrix)
    finite = mx.all(mx.isfinite(matrix))
    mx.eval(finite)
    if not bool(finite.item()):
        raise ValueError("matrix must contain only finite values")

    rows, columns = matrix.shape
    column_blocks = max(1, columns // 128)
    vector_count = rows * column_blocks if axis == 1 else rows // 128 * columns
    output = _main_kernel(columns, axis)(
        inputs=[mx.contiguous(matrix)],
        template=[
            ("AXIS", axis),
            ("COLUMNS", columns),
            ("COLUMN_BLOCKS", column_blocks),
        ],
        grid=(vector_count * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[matrix.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    return output


def _interleaved_ms(main_call, candidate_call, *, warmups, repeats):
    for _ in range(warmups):
        main_call()
        candidate_call()

    main_samples = []
    candidate_samples = []
    for repeat in range(repeats):
        calls = (
            ((main_call, main_samples), (candidate_call, candidate_samples))
            if repeat % 2 == 0
            else ((candidate_call, candidate_samples), (main_call, main_samples))
        )
        for function, samples in calls:
            start = time.perf_counter()
            function()
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(main_samples), statistics.median(candidate_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=101)
    args = parser.parse_args()

    print(
        "projection | axis | shape | candidate/main unequal | "
        "Torch max abs | Torch normalized RMS | main ms | candidate ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(10421 + rows + columns)
        source = rng.normal(0.0, 0.2, (rows, columns)).astype(np.float32)
        mlx_source = mx.array(source)
        mx.eval(mlx_source)

        for axis in (0, 1):
            expected = _torch_hadamard_oracle(source, axis)
            main_output = np.asarray(_main_hadamard(mlx_source, axis=axis))
            candidate_output = np.asarray(exl3_hadamard_128_mlx(mlx_source, axis=axis))
            unequal = int(np.count_nonzero(candidate_output != main_output))
            error = np.abs(candidate_output - expected)
            max_abs = float(error.max())
            normalized_rms = _normalized_rms_drift(candidate_output, expected)
            allowed = 1e-6 + 1e-6 * np.abs(expected)
            outside = int(np.count_nonzero(error > allowed))
            if outside:
                raise AssertionError(
                    f"{name} axis={axis}: {outside} outputs exceed tolerance"
                )

            main_call = partial(_main_hadamard, mlx_source, axis=axis)
            candidate_call = partial(exl3_hadamard_128_mlx, mlx_source, axis=axis)
            main_ms, candidate_ms = _interleaved_ms(
                main_call,
                candidate_call,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            print(
                f"{name} | {axis} | {rows}x{columns} | {unequal} | "
                f"{max_abs:.9g} | {normalized_rms:.9g} | {main_ms:.3f} | "
                f"{candidate_ms:.3f} | {main_ms / candidate_ms:.3f}x",
                flush=True,
            )
            del expected, main_output, candidate_output, error
            gc.collect()

        del source, mlx_source
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
