# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Smoke test one calibration quantizer with fused same-input forward enabled."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def preflight_gpu(target_index: int, samples: int = 3, interval: float = 0.5, max_wait: float = 60.0, memory_mib: int = 2000):
    """Wait for target GPU to be idle (0%% util for `samples` consecutive reads) and below memory threshold."""
    start = time.time()
    idle_streak = 0
    last_util = None
    while time.time() - start < max_wait:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        row = None
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == target_index:
                row = parts
                break
        if row is None:
            raise RuntimeError(f"GPU index {target_index} not found by nvidia-smi")
        mem_mib = int(row[1])
        util = int(row[2])
        if util == 0 and mem_mib <= memory_mib:
            idle_streak += 1
            if idle_streak >= samples:
                print(f"Preflight passed for GPU {target_index}: util={util}%, memory={mem_mib} MiB")
                return
        else:
            idle_streak = 0
        if util != last_util or time.time() - start < 2.0:
            print(f"Preflight GPU {target_index}: util={util}%, memory={mem_mib} MiB (streak {idle_streak}/{samples})")
            last_util = util
        time.sleep(interval)
    raise RuntimeError(f"GPU {target_index} did not reach idle state within {max_wait}s")


def build_config(method: str, backend: str):
    from gptqmodel import FusedForwardConfig, QuantizeConfig
    from gptqmodel.quantization.config import FORMAT, METHOD

    fused = FusedForwardConfig(splice="view")

    if method == "gptq":
        cfg = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            act_group_aware=True,
            scale_search="activation",
            damp_percent=0.05,
            fused_forward=fused,
        )
    elif method == "awq":
        cfg = QuantizeConfig(
            bits=4,
            group_size=128,
            method=METHOD.AWQ,
            format=FORMAT.GEMM,
            fused_forward=fused,
        )
    elif method == "qqq":
        cfg = QuantizeConfig(
            bits=4,
            group_size=128,
            method=METHOD.QQQ,
            format=FORMAT.QQQ,
            desc_act=True,
            fused_forward=fused,
        )
    elif method == "paro":
        cfg = QuantizeConfig(
            bits=4,
            group_size=128,
            method=METHOD.PARO,
            format=FORMAT.PAROQUANT,
            opt_scope="module",
            opt_rotation_epochs=2,
            opt_finetune_epochs=2,
            opt_train_samples=512,
            opt_validation_samples=128,
            opt_batch_size=64,
            opt_stage_impl="fast",
            opt_pair_impl="fast",
            opt_quantizer_impl="reference",
            fused_forward=fused,
        )
    elif method == "exl3":
        cfg = QuantizeConfig(
            bits=3.0,
            group_size=-1,
            method=METHOD.EXL3,
            format=FORMAT.EXL3,
            desc_act=False,
            sym=True,
            fused_forward=fused,
        )
    else:
        raise ValueError(f"Unknown method {method}")

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["gptq", "awq", "qqq", "paro", "exl3"])
    parser.add_argument("--backend", required=True)
    parser.add_argument("--gpu-index", type=int, default=6)
    parser.add_argument("--no-preflight", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # Reduce Triton autotune overhead for Paro smoke tests.
    if args.method == "paro":
        os.environ["GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE"] = "0"

    if not args.no_preflight:
        preflight_gpu(args.gpu_index, memory_mib=2000)

    cal = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. " * 10,
        "Quantization reduces the precision of model weights to lower memory and inference costs. " * 10,
        "Large language models have demonstrated remarkable capabilities in natural language understanding. " * 10,
        "The transformer architecture relies on self-attention mechanisms to process sequential data. " * 10,
    ] * 2

    from gptqmodel import BACKEND, GPTQModel

    backend = getattr(BACKEND, args.backend)
    cfg = build_config(args.method, args.backend)

    model = GPTQModel.load(
        "/monster/data/model/Llama-3.2-1B-Instruct",
        quantize_config=cfg,
        trust_remote_code=False,
        dtype="auto",
        device_map="cuda:0",
        attn_implementation="eager",
    )

    t0 = time.perf_counter()
    model.quantize(
        cal,
        calibration_concat_size=256,
        calibration_sort="desc",
        batch_size=1,
        backend=backend,
    )
    elapsed = time.perf_counter() - t0
    print(f"METHOD={args.method} BACKEND={args.backend} OK elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    main()
