#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Post-quantization embed+lm_head requantization CLI.

Loads an existing GPTQ checkpoint, resolves its actual input/output embedding
paths, applies per-module bits/group_size overrides, and requantizes only the
selected embedding tensors.

Example:
    python -m optimize.requant_embed_lm_head \
        --model-path /path/to/quantized-model \
        --gpus 0,1 \
        --calibration-parquet /path/to/calibration.parquet \
        --bits 4 --group-size 64 --embed-quant-mode both
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time
from pathlib import Path

from optimize._common import (
    add_common_args,
    idle_gate,
    load_calibration_data,
    parse_gpus,
    set_env,
    set_torch_threads,
    verify_visible_uuids,
)

set_env()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument(
        "--embed-quant-mode",
        choices=["input", "output", "both"],
        default="both",
        help="Which embedding tensors to requantize (input=embed_tokens, output=lm_head).",
    )
    return parser.parse_args()


def requant_embed_lm_head(args: argparse.Namespace) -> Path:
    """Run embed+lm_head requantization; caller is responsible for GPU visibility."""
    import torch
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import QuantizeEmbed

    set_torch_threads()
    faulthandler.enable(file=sys.stderr, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except (AttributeError, ValueError):
        pass

    mode = QuantizeEmbed(args.embed_quant_mode)
    print(f"[requant] Loading quantized checkpoint {args.model_path} ...", flush=True)
    model = GPTQModel.load(
        args.model_path,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        backend=BACKEND.AUTO,
    )

    target_names: list[str] = []
    if mode in (QuantizeEmbed.INPUT, QuantizeEmbed.BOTH):
        input_name = model.get_input_embeddings_name()
        if not isinstance(input_name, str) or not input_name:
            raise ValueError("Could not resolve the model input-embedding path for requantization.")
        target_names.append(input_name)
    if mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH):
        output_name = model.get_output_embeddings_name() or getattr(model, "lm_head", None)
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("Could not resolve the model output-embedding path for requantization.")
        if output_name not in target_names:
            target_names.append(output_name)

    # GPTQModel.load ignores a caller quantize_config for already-quantized
    # checkpoints, so apply the embed/lm_head overrides after loading.
    embed_bits = args.bits
    embed_group_size = args.group_size
    act_group_aware = not args.no_act_group_aware
    embed_cfg = {
        "bits": embed_bits,
        "group_size": embed_group_size,
        "sym": True,
        "desc_act": args.desc_act,
        "act_group_aware": act_group_aware,
        "scale_search": args.scale_search,
    }
    if model.quantize_config.dynamic is None:
        model.quantize_config.dynamic = {}
    for name in target_names:
        model.quantize_config.dynamic[name] = dict(embed_cfg)
    for name in target_names:
        effective = model.quantize_config.dynamic_get(name, default=None)
        assert effective == embed_cfg, f"dynamic override for {name} not effective: {effective}"
    print(f"[requant] effective dynamic overrides: {model.quantize_config.dynamic}", flush=True)

    calibration = load_calibration_data(
        parquet_path=args.calibration_parquet,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_size=args.dataset_size,
    )

    print(f"[requant] Requantizing embed/lm_head with mode={mode} ...", flush=True)
    start = time.time()
    model.requantize(
        calibration=calibration,
        calibration_concat_size=args.calibration_concat_size,
        calibration_concat_separator=args.calibration_concat_separator,
        calibration_sort=args.calibration_sort,
        batch_size=args.batch_size,
        backend=BACKEND.AUTO,
        embed_quant_mode=mode,
    )
    print(f"[requant] embed+lm_head requantization finished in {time.time() - start:.1f}s", flush=True)

    suffix = f"_embed_lmhead_w{embed_bits}g{embed_group_size}"
    output = Path(args.output) if args.output else Path(str(args.model_path) + suffix)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[save] Saving requantized model to {output} ...", flush=True)
    save_start = time.time()
    model.save(str(output))
    print(f"[save] Save took {time.time() - save_start:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    print("[done] requant embed+lm_head complete", flush=True)
    return output


def _main() -> None:
    args = _parse_args()
    physical_gpus = parse_gpus(args.gpus)
    gpu_infos = idle_gate(physical_gpus, timeout=args.idle_timeout)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical_gpus)
    verify_visible_uuids(physical_gpus, gpu_infos)

    requant_embed_lm_head(args)


if __name__ == "__main__":
    _main()
