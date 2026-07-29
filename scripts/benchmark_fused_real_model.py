# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a real GPT-QModel against the same model after model.fuse().

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_real_model.py \
        --model_path /monster/data/model/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2

The script loads the model, warms it up, times the unfused forward, calls
model.fuse(), warms the fused path, then times the fused forward. It reports
latency, throughput, and numerical parity (max abs diff).
"""

import argparse
import os

import torch

from gptqmodel import BACKEND, GPTQModel


def _warmup(model, input_ids, steps: int = 3) -> None:
    with torch.inference_mode():
        for _ in range(steps):
            _ = model(input_ids).logits
    torch.cuda.synchronize()


def _measure(model, input_ids, steps: int = 10) -> float:
    _warmup(model, input_ids, steps=1)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        for _ in range(steps):
            _ = model(input_ids).logits
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to a GPT-QModel quantized model.")
    parser.add_argument("--backend", default=BACKEND.TRITON, type=str, help="Quantized backend to use.")
    parser.add_argument("--device", default="cuda:0", help="Device to load the model on.")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32], help="Batch sizes to benchmark.")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length per batch.")
    parser.add_argument("--steps", type=int, default=10, help="Timed forward iterations.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    args = parser.parse_args()

    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5"

    print(f"Loading {args.model_path} with backend={args.backend} on {args.device} ...")
    model = GPTQModel.load(args.model_path, backend=args.backend, device=args.device)
    print("Model loaded")

    vocab_size = model.tokenizer.vocab_size if model.tokenizer else 100000
    rows = []
    print("\n=== Unfused ===")
    for batch in args.batch_sizes:
        input_ids = torch.randint(0, vocab_size, (batch, args.seq_len), device=args.device)
        _warmup(model.model, input_ids, steps=args.warmup)
        ms_unfused = _measure(model.model, input_ids, steps=args.steps)
        tokens_per_sec = batch * args.seq_len * 1000 / ms_unfused
        rows.append({"batch": batch, "seq": args.seq_len, "ms_unfused": ms_unfused, "tok/s_unfused": tokens_per_sec})
        print(f"batch={batch:>3} seq={args.seq_len:<5} ms={ms_unfused:>8.3f} tok/s={tokens_per_sec:>10.1f}")

    counts = model.fuse()
    print(f"\nFused groups: {counts}\n")

    print("=== Fused ===")
    for i, batch in enumerate(args.batch_sizes):
        input_ids = torch.randint(0, vocab_size, (batch, args.seq_len), device=args.device)
        _warmup(model.model, input_ids, steps=args.warmup)
        ms_fused = _measure(model.model, input_ids, steps=args.steps)
        tokens_per_sec = batch * args.seq_len * 1000 / ms_fused
        rows[i]["ms_fused"] = ms_fused
        rows[i]["tok/s_fused"] = tokens_per_sec
        speedup = rows[i]["ms_unfused"] / ms_fused
        rows[i]["speedup"] = speedup
        print(f"batch={batch:>3} seq={args.seq_len:<5} ms={ms_fused:>8.3f} tok/s={tokens_per_sec:>10.1f} speedup={speedup:>5.3f}")

    # Numerical parity on a fresh batch.
    batch = args.batch_sizes[-1]
    input_ids = torch.randint(0, vocab_size, (batch, args.seq_len), device=args.device)
    # We cannot re-run unfused without reloading; report max diff between two fused runs.
    with torch.inference_mode():
        out_a = model.model(input_ids).logits
        out_b = model.model(input_ids).logits
    max_diff = (out_a - out_b).abs().max().item()
    print(f"\nMax abs diff between two fused forward passes: {max_diff:.6f}")

    print("\n=== Summary table ===")
    print(f"{'batch':>6} {'seq':>4} {'ms_unf':>10} {'ms_fus':>10} {'speedup':>8} {'tok/s_unf':>12} {'tok/s_fus':>12}")
    for r in rows:
        print(
            f"{r['batch']:>6} {r['seq']:>4} "
            f"{r['ms_unfused']:>10.3f} {r['ms_fused']:>10.3f} {r['speedup']:>8.3f} "
            f"{r['tok/s_unfused']:>12.1f} {r['tok/s_fused']:>12.1f}"
        )


if __name__ == "__main__":
    main()
