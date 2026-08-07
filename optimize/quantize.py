#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dense-to-GPTQ quantization CLI.

Reproduces the DeepSeek-V4-Flash-0731 and Laguna-S-2.1 W4G64/W4G128 runs:
    python -m optimize.quantize \
        --model-path /monster/data/model/DeepSeek-V4-Flash-0731-BF16-Defused \
        --output /monster/data/model/DeepSeek-V4-Flash-0731-W4G64-GAR-activation \
        --gpus 0,1 \
        --calibration-parquet /path/to/calibration.parquet \
        --bits 4 --group-size 64 \
        --moe-routing-bypass --moe-vram-strategy balanced
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time
from pathlib import Path

# Set thread limits before importing torch/numpy.
from optimize._common import (
    add_common_args,
    build_quantize_config,
    idle_gate,
    load_calibration_data,
    parse_gpus,
    set_env,
    set_torch_threads,
    verify_visible_uuids,
)

set_env()

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    return parser.parse_args()


def quantize(args: argparse.Namespace) -> None:
    """Run dense-to-GPTQ quantization; caller is responsible for GPU visibility."""
    set_torch_threads()
    faulthandler.enable(file=sys.stderr, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except (AttributeError, ValueError):
        pass

    # act_group_aware is on by default; --no-act-group-aware disables it.
    args.act_group_aware = not args.no_act_group_aware

    quantize_config = build_quantize_config(args)
    print(f"[quant] QuantizeConfig: {quantize_config}", flush=True)

    print(f"[load] Loading dense model from {args.model_path} ...", flush=True)
    model = GPTQModel.load(
        args.model_path,
        quantize_config=quantize_config,
        trust_remote_code=args.trust_remote_code,
        dtype="auto",
        device_map="auto",
        backend=BACKEND.AUTO,
    )

    calibration = load_calibration_data(
        parquet_path=args.calibration_parquet,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_size=args.dataset_size,
    )

    print(
        f"[quant] Calibrating on {len(calibration)} samples, "
        f"concat_size={args.calibration_concat_size}, batch_size={args.batch_size}",
        flush=True,
    )
    start = time.time()
    model.quantize(
        calibration,
        calibration_concat_size=args.calibration_concat_size,
        calibration_concat_separator=args.calibration_concat_separator,
        calibration_sort=args.calibration_sort,
        batch_size=args.batch_size,
        backend=BACKEND.AUTO,
    )
    print(f"[quant] Quantization finished in {time.time() - start:.1f}s", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[save] Saving quantized model to {output} ...", flush=True)
    save_start = time.time()
    model.save(str(output))
    print(f"[save] Save took {time.time() - save_start:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    print("[done] quantize + save complete", flush=True)


def _main() -> None:
    args = _parse_args()
    if not args.output:
        raise SystemExit("error: --output is required for quantize.py")

    physical_gpus = parse_gpus(args.gpus)
    gpu_infos = idle_gate(physical_gpus, timeout=args.idle_timeout)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical_gpus)
    verify_visible_uuids(physical_gpus, gpu_infos)

    quantize(args)


if __name__ == "__main__":
    _main()
