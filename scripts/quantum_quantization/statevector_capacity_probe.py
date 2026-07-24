#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Allocate a CUDA-Q state vector to verify a single-GPU qubit boundary."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the state alive until Enter is pressed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.qubits <= 40:
        raise ValueError("--qubits must be in [1, 40].")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "," in visible_devices:
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly one GPU before importing CUDA-Q."
        )

    import cudaq

    cudaq.set_target("nvidia", option=args.precision)
    if cudaq.num_available_gpus() != 1:
        raise RuntimeError(
            f"Expected one visible GPU, found {cudaq.num_available_gpus()}."
        )

    bytes_per_amplitude = 8 if args.precision == "fp32" else 16
    raw_state_bytes = (1 << args.qubits) * bytes_per_amplitude
    raw_state_gib = raw_state_bytes / float(1 << 30)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(args.qubits)
    kernel.h(qubits)

    if args.hold:
        state = cudaq.get_state(kernel)
        print(
            f"READY precision={args.precision} qubits={args.qubits} "
            f"raw_state_gib={raw_state_gib:.3f} target={cudaq.get_target().name}",
            flush=True,
        )
        input("Press Enter to release the state vector: ")
        del state
    else:
        counts = cudaq.sample(kernel, shots_count=1)
        print(
            f"SUCCESS precision={args.precision} qubits={args.qubits} "
            f"raw_state_gib={raw_state_gib:.3f} counts={counts}"
        )


if __name__ == "__main__":
    main()
