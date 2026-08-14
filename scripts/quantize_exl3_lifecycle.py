#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Quantize and save a full EXL3 checkpoint with exact lifecycle telemetry."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.quantization import EXL3Config


EXL3_INFERENCE_DTYPE = torch.float16


def parse_args() -> argparse.Namespace:
    """Parse an exact full-model EXL3 lifecycle run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default="/monster/data/model/dataset/nm-calibration")
    parser.add_argument("--dataset-config", default="LLM")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--bits", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run production EXL3 quantization and persist a reloadable checkpoint."""

    args = parse_args()
    if args.rows < 1 or args.row_start < 0:
        raise ValueError("EXL3 calibration requires positive --rows and nonnegative --row-start.")
    if args.layers < 1 or args.batch_size < 1:
        raise ValueError("EXL3 lifecycle requires positive --layers and --batch-size.")

    output = Path(args.output).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    config = EXL3Config(
        bits=args.bits,
        device=args.device,
        codebook="mcg",
        out_scales="auto",
        offload_to_disk=False,
    )
    load_started = time.perf_counter()
    model = GPTQModel.load(
        args.model,
        quantize_config=config,
        dtype=EXL3_INFERENCE_DTYPE,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    load_seconds = time.perf_counter() - load_started

    dataset = load_dataset(args.dataset, name=args.dataset_config, split="train")
    row_stop = args.row_start + args.rows
    if row_stop > len(dataset):
        raise ValueError(
            f"EXL3 calibration requires rows [{args.row_start}, {row_stop}), "
            f"but dataset contains only {len(dataset)} rows."
        )
    calibration = dataset.select(range(args.row_start, row_stop))

    quant_started = time.perf_counter()
    quant_log = model.quantize(
        calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=args.batch_size,
        backend=BACKEND.EXL3_EXLLAMA_V3,
        layer_scope=slice(0, args.layers),
    )
    quant_seconds = time.perf_counter() - quant_started

    exl3_modules = [name for name, module in model.model.named_modules() if isinstance(module, ExllamaV3Linear)]
    if not exl3_modules:
        raise AssertionError("The EXL3 lifecycle did not install any ExllamaV3Linear modules.")

    save_started = time.perf_counter()
    model.save(str(output))
    save_seconds = time.perf_counter() - save_started
    del model
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "model": str(args.model),
        "output": str(output),
        "bits": args.bits,
        "backend": BACKEND.EXL3_EXLLAMA_V3.value,
        "dtype": str(EXL3_INFERENCE_DTYPE),
        "device": args.device,
        "layers": args.layers,
        "calibration": {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "row_start": args.row_start,
            "rows": len(calibration),
            "concat_size": None,
            "batch_size": args.batch_size,
        },
        "exl3": {
            "codebook": "mcg",
            "out_scales": "auto",
            "sigma_reg": 0.025,
            "seed": 787,
        },
        "exl3_module_count": len(exl3_modules),
        "quant_log_rows": sum(len(rows) for rows in quant_log.values()),
        "seconds": {"load_dense": load_seconds, "quantize": quant_seconds, "save": save_seconds},
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
