#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for dequantizing GPTQModel safetensor shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch

from gptqmodel.utils.model_dequant import dequantize_model


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - argparse ensures this
        raise ValueError(f"Unsupported dtype {name}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        nargs="?",
        help="Path to the quantized model directory",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        help="Path where the dequantized model will be written",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path_flag",
        type=Path,
        help="Explicit model path (overrides positional argument)",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path_flag",
        type=Path,
        help="Explicit output path (overrides positional argument)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Target floating point dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to stage tensors during dequantization (cpu, cuda, cuda:7, ...)",
    )
    parser.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="Optional environment variable overrides applied before execution",
    )
    return parser.parse_args()


def _apply_env(overrides: Optional[list[str]]) -> None:
    if not overrides:
        return
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Invalid --env entry {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        os.environ[key] = value


def main() -> None:
    args = _parse_args()
    _apply_env(args.env)

    model_path = args.model_path_flag or args.model_path
    output_path = args.output_path_flag or args.output_path

    if model_path is None or output_path is None:
        raise SystemExit("model_path and output_path must be provided (positionally or via flags)")

    dtype = _resolve_dtype(args.dtype)
    device = args.device if args.device is not None else "cpu"

    debug_payload = {
        "model_path": str(Path(model_path)),
        "output_path": str(Path(output_path)),
        "dtype": dtype,
        "device": device,
    }
    print(f"[dequantize_model] parsed args: {debug_payload}")

    dequantize_model(model_path, output_path, target_dtype=dtype, device=device)


if __name__ == "__main__":
    main()
