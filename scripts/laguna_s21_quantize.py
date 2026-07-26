#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import QuantizeConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/monster/data/model/Laguna-S-2.1",
        help="Path to the dense Laguna-S-2.1 checkpoint.",
    )
    parser.add_argument(
        "--quant-config",
        type=str,
        default="laguna_s21/quant_config_quantize.json",
        help="Path to the QuantizeConfig JSON to use for quantization.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/monster/data/model/dataset/nm-calibration",
        help="Path to the calibration dataset directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="LLM",
        help="Dataset config name.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=512,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--calibration-concat-size",
        type=int,
        default=2048,
        help="Calibration concatenation size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/monster/data/model/Laguna-S-2.1-GPTQ",
        help="Where to save the quantized checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype for loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code for the Laguna model.",
    )
    args = parser.parse_args()

    quant_config_dict = json.loads(Path(args.quant_config).read_text())
    quant_config = QuantizeConfig.from_quant_config(quant_config_dict)

    print(f"[quant] Loading dense model from {args.model_path}")
    print(f"[quant] QuantizeConfig: bits={quant_config.bits}, group_size={quant_config.group_size}, dynamic_rules={len(quant_config.dynamic or {})}")

    model = GPTQModel.load(
        args.model_path,
        quantize_config=quant_config,
        device_map="auto",
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        backend=BACKEND.AUTO,
    )

    print(f"[quant] Loading calibration dataset: {args.dataset_path} ({args.dataset_name}/{args.dataset_split})")
    dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
    if args.dataset_size > 0:
        dataset = dataset.select(range(min(args.dataset_size, len(dataset))))

    print(f"[quant] Calibrating on {len(dataset)} samples, concat_size={args.calibration_concat_size}, batch_size={args.batch_size}")
    model.quantize(
        dataset,
        calibration_concat_size=args.calibration_concat_size,
        calibration_concat_separator="\n",
        calibration_sort="desc",
        batch_size=args.batch_size,
        backend=BACKEND.AUTO,
    )

    os.makedirs(args.save_path, exist_ok=True)
    print(f"[quant] Saving quantized model to {args.save_path}")
    model.save(args.save_path)
    print("[quant] Done")


if __name__ == "__main__":
    main()
