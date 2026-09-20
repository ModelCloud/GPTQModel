#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Replay one model-facing Flash-Next expert CUDA graph for Nsight Systems."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_qvq_p32_qwen38_flash_next_runtime_h100 import _build_mlp


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=float, default=3.0)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    import torch

    from gptqmodel.nn_modules.qvq_grouped_runtime import install_qvq_hopper_groups
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda

    if not prewarm_qvq_cuda():
        raise RuntimeError("failed to prewarm QVQ CUDA")
    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = _build_mlp(torch, bits=args.bits, device=device)
    if install_qvq_hopper_groups(model, qkv=False) != {"gate_up": 1}:
        raise RuntimeError("failed to install grouped Flash-Next MLP")
    if not hasattr(model, "_gptqmodel_qvq_fused_mlp_runtime"):
        raise RuntimeError("failed to install Flash-Next fused MLP runtime")
    value = torch.randn((args.m, 2560), device=device, dtype=dtype) * 0.02
    with torch.inference_mode():
        for _ in range(3):
            model(value)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(value)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print(float(output.float().sum().item()))


if __name__ == "__main__":
    _main()
