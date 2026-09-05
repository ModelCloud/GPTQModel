#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Requantize an existing checkpoint's input embedding, output embedding, or both.

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
    add_requant_args,
    idle_gate,
    load_calibration_data,
    parse_gpus,
    set_env,
    set_torch_threads,
    verify_visible_uuids,
)

set_env()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_requant_args(parser)
    parser.add_argument(
        "--embed-quant-mode",
        choices=["input", "output", "both"],
        default="both",
        help="Embedding endpoints to requantize: input, output, or both.",
    )
    return parser.parse_args()


def requant_embed_lm_head(args: argparse.Namespace) -> Path:
    """Run embedding requantization after GPU visibility has been configured."""
    import torch

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import QuantizeEmbed

    set_torch_threads()
    faulthandler.enable(file=sys.stderr, all_threads=True)
    try:
        faulthandler.register(
            signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False
        )
    except (AttributeError, ValueError):
        # Best-effort only: some platforms/runtime contexts do not support SIGUSR1
        # registration for faulthandler; continue without this diagnostic hook.
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
            raise ValueError(
                "Could not resolve the model input-embedding path for requantization."
            )
        target_names.append(input_name)
    if mode in (QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH):
        output_name = model.get_output_embeddings_name() or getattr(
            model, "lm_head", None
        )
        if not isinstance(output_name, str) or not output_name:
            raise ValueError(
                "Could not resolve the model output-embedding path for requantization."
            )
        if output_name not in target_names:
            target_names.append(output_name)

    embed_config = {
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": True,
        "desc_act": args.desc_act,
        "act_group_aware": not args.no_act_group_aware,
        "scale_search": args.scale_search,
    }
    # Dynamic lookups are cached because configs are otherwise immutable after
    # construction. Build a new mapping so lookups performed during checkpoint
    # loading cannot mask the newly selected endpoints. Put the explicit target
    # paths first so a pre-existing broad pattern cannot win first-match lookup.
    existing_dynamic = model.quantize_config.dynamic or {}
    target_overrides = {name: dict(embed_config) for name in target_names}
    model.quantize_config.dynamic = {
        **target_overrides,
        **{
            pattern: override
            for pattern, override in existing_dynamic.items()
            if pattern not in target_overrides
        },
    }
    for name in target_names:
        effective = model.quantize_config.dynamic_get(name, default=None)
        if effective != embed_config:
            raise RuntimeError(
                f"Dynamic override for {name} is not effective: {effective}"
            )
    print(
        f"[requant] Effective dynamic overrides: {model.quantize_config.dynamic}",
        flush=True,
    )

    calibration = load_calibration_data(
        parquet_path=args.calibration_parquet,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_size=args.dataset_size,
    )

    print(f"[requant] Requantizing embeddings with mode={mode.value} ...", flush=True)
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
    print(
        f"[requant] Requantization finished in {time.time() - start:.1f}s", flush=True
    )

    suffix = f"_embed_lmhead_w{args.bits}g{args.group_size}"
    output = Path(args.output) if args.output else Path(f"{args.model_path}{suffix}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"[save] Saving requantized model to {output} ...", flush=True)
    save_start = time.time()
    model.save(str(output))
    print(f"[save] Save took {time.time() - save_start:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    print("[done] Embedding requantization complete", flush=True)
    return output


def _main() -> None:
    args = _parse_args()
    physical_gpus = parse_gpus(args.gpus)
    gpu_infos = idle_gate(physical_gpus, timeout=args.idle_timeout)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in physical_gpus)
    verify_visible_uuids(physical_gpus, gpu_infos)
    requant_embed_lm_head(args)


if __name__ == "__main__":
    _main()
