#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Dequantize an FP8 E4M3 model into BF16 using gptqmodel.utils.model_dequant."""

from __future__ import annotations

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7" #"expandable_segments:True"


import argparse
from pathlib import Path

import torch

from gptqmodel.utils.model_dequant import dequantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dequantize FP8 model shards to BF16")
    parser.add_argument("model_path", type=Path, nargs="?", help="Path to the FP8 model directory")
    parser.add_argument("output_path", type=Path, nargs="?", help="Destination directory for the BF16 model")
    parser.add_argument("--model_path", dest="model_path_opt", type=Path, help="Path to the FP8 model directory")
    parser.add_argument("--output_path", dest="output_path_opt", type=Path, help="Destination directory for the BF16 model")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Output dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for intermediate dequantization (e.g. cpu, cuda, cuda:0)",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path is not None else args.model_path_opt
    output_path = args.output_path if args.output_path is not None else args.output_path_opt
    if model_path is None or output_path is None:
        raise SystemExit("model_path and output_path must be provided either positionally or via flags")

    dtype = resolve_dtype(args.dtype)
    device = None
    if args.device is not None and args.device.lower() != "cpu":
        device = args.device
    print(
        "[dequantize_fp8_model] args",
        {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "dtype": dtype,
            "device": device or "cpu",
        },
    )
    dequantize(model_path, output_path, target_dtype=dtype, device=device)


if __name__ == "__main__":
    main()
