#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Nsight Systems friendly Laguna-S-2.1-GPTQ-4G64 decode step."""
from __future__ import annotations

import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/monster/data/model/Laguna-S-2.1-GPTQ-4G64")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--moe-backend", type=str, default="per_expert")
    p.add_argument("--fuse", action="store_true",
                   help="Call model.fuse(qkv, gate_up, gate_up_activation) before profiling.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["GPTQMODEL_MARLIN_MOE_BACKEND"] = args.moe_backend

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.utils.moe_dispatch import enable_grouped_dispatch_for_model, register_linear_loop_experts

    model = GPTQModel.load(
        args.model_path,
        backend=BACKEND.GPTQ_MARLIN,
        trust_remote_code=True,
        device="cuda:0",
        attn_implementation="flash_attention_2",
    )

    gen_cfg = getattr(model.model, "generation_config", None)
    if gen_cfg is not None:
        if getattr(gen_cfg, "speculative_config", None) is not None:
            gen_cfg.speculative_config = None
        gen_cfg.do_sample = False

    if args.fuse:
        counts = model.fuse(qkv=True, gate_up=True, gate_up_activation=True, free_original_weights=True)
        print(f"Fused: {counts}")

    register_linear_loop_experts()
    enable_grouped_dispatch_for_model(model.model)

    vocab_size = getattr(model.config, "vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device="cuda:0")

    # Warmup
    with torch.inference_mode():
        _ = model.generate(input_ids, max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("generate_step")
    with torch.inference_mode():
        _ = model.generate(input_ids, max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
