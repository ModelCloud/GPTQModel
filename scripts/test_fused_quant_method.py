# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Smoke test one calibration quantizer with fused same-input forward enabled."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def preflight_gpu(target_indices: List[int], samples: int = 3, interval: float = 0.5, max_wait: float = 60.0, memory_mib: int = 2000):
    """Wait for target GPUs to be idle (0%% util for `samples` consecutive reads) and below memory threshold."""
    start = time.time()
    idle_streaks = dict.fromkeys(target_indices, 0)
    last_utils = dict.fromkeys(target_indices)
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
        rows = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx = int(parts[0])
            if idx in idle_streaks:
                rows[idx] = parts
        if len(rows) != len(target_indices):
            missing = set(target_indices) - set(rows)
            raise RuntimeError(f"GPU indices {missing} not found by nvidia-smi")
        all_idle = True
        for idx, parts in rows.items():
            mem_mib = int(parts[1])
            util = int(parts[2])
            if util == 0 and mem_mib <= memory_mib:
                idle_streaks[idx] += 1
            else:
                idle_streaks[idx] = 0
                all_idle = False
            if util != last_utils[idx] or time.time() - start < 2.0:
                print(f"Preflight GPU {idx}: util={util}%, memory={mem_mib} MiB (streak {idle_streaks[idx]}/{samples})")
                last_utils[idx] = util
        if all_idle and all(s >= samples for s in idle_streaks.values()):
            print(f"Preflight passed for GPUs {target_indices}: all idle")
            return
        time.sleep(interval)
    raise RuntimeError(f"GPUs {target_indices} did not reach idle state within {max_wait}s")


def build_config(method: str, backend: str, moe_bypass: bool = False):
    from gptqmodel import FusedForwardConfig, QuantizeConfig
    from gptqmodel.quantization.config import FORMAT, METHOD, ExpertsRoutingBypass, MoEConfig

    fused = FusedForwardConfig(splice="view")
    moe = MoEConfig(routing=ExpertsRoutingBypass()) if moe_bypass else None

    if method == "gptq":
        cfg = QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            act_group_aware=True,
            scale_search="activation",
            damp_percent=0.05,
            fused_forward=fused,
            moe=moe,
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
    parser.add_argument("--gpu-index", type=str, default="6", help="Comma-separated CUDA indices in PCI bus order")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--model", default="/monster/data/model/Llama-3.2-1B-Instruct")
    parser.add_argument("--moe-bypass", action="store_true")
    parser.add_argument("--true-sequential", action="store_true")
    args = parser.parse_args()

    gpu_indices = [int(x.strip()) for x in args.gpu_index.split(",")]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)

    # Reduce Triton autotune overhead for Paro smoke tests.
    if args.method == "paro":
        os.environ["GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE"] = "0"

    if not args.no_preflight:
        preflight_gpu(gpu_indices, memory_mib=2000)

    cal = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. " * 10,
        "Quantization reduces the precision of model weights to lower memory and inference costs. " * 10,
        "Large language models have demonstrated remarkable capabilities in natural language understanding. " * 10,
        "The transformer architecture relies on self-attention mechanisms to process sequential data. " * 10,
    ] * 2

    from gptqmodel import BACKEND, GPTQModel

    backend = getattr(BACKEND, args.backend)
    cfg = build_config(args.method, args.backend, moe_bypass=args.moe_bypass)
    if args.true_sequential:
        cfg.true_sequential = True

    model = GPTQModel.load(
        args.model,
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
