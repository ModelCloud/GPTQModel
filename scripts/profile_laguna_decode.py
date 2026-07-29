#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile one decode step of Laguna-S-2.1-GPTQ-FIXED with torch.profiler."""

from __future__ import annotations

import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, record_function, profile  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/monster/data/model/Laguna-S-2.1-GPTQ-FIXED")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--fuse", action="store_true")
    p.add_argument("--disable-speculative", action="store_true")
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    p.add_argument("--trace", type=str, default="laguna_decode_trace.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    from gptqmodel import BACKEND, GPTQModel

    backend = BACKEND.GPTQ_MARLIN
    print("Loading model...")
    model = GPTQModel.load(
        args.model_path,
        backend=backend,
        trust_remote_code=True,
        device="cuda:0",
        attn_implementation=args.attn_implementation,
    )
    print("Model loaded.")

    if args.disable_speculative:
        gen_cfg = getattr(model.model, "generation_config", None) or getattr(model, "generation_config", None)
        if gen_cfg is not None:
            if getattr(gen_cfg, "speculative_config", None) is not None:
                gen_cfg.speculative_config = None
            gen_cfg.do_sample = False
            print("Disabled speculative decoding / do_sample.")

    if args.fuse:
        counts = model.fuse(
            qkv=True,
            gate_up=True,
            gate_up_activation=True,
            free_original_weights=True,
        )
        print(f"Fused: {counts}")

        from gptqmodel.utils.moe_dispatch import enable_grouped_dispatch_for_model, register_linear_loop_experts
        registered = register_linear_loop_experts()
        moe_enabled = enable_grouped_dispatch_for_model(model.model)
        print(f"Registered GPT-QModel linear_loop experts: {registered}")
        print(f"Enabled grouped MoE dispatch for {moe_enabled} module(s).")

    vocab_size = getattr(model.config, "vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device="cuda:0")

    # Warmup
    with torch.inference_mode():
        _ = model.generate(input_ids, max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    print("Profiling one generate step...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with record_function("generate_step"):
            with torch.inference_mode():
                _ = model.generate(input_ids, max_new_tokens=1, do_sample=False, use_cache=True)
            torch.cuda.synchronize()

    print("\nTop CUDA kernels by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    trace_path = args.trace
    print(f"\nExporting Chrome trace to {trace_path}")
    prof.export_chrome_trace(trace_path)


if __name__ == "__main__":
    main()
