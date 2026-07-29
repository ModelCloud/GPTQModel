#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end `model.fuse()` validation on Laguna-S-2.1-GPTQ-FIXED.

Loads the real quantized checkpoint, optionally fuses QKV and gate/up groups,
runs forward/generate on the requested batch/sequence sizes, and reports
latency, throughput, and numerical parity.

The `--gpu` flag must be parsed before `import torch` so `CUDA_VISIBLE_DEVICES`
takes effect.
"""

from __future__ import annotations

import argparse
import os
import sys

# Parse --gpu before any CUDA context is created.
_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
        break
    if arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/monster/data/model/Laguna-S-2.1-GPTQ-FIXED")
    p.add_argument("--backend", default="GPTQ_MARLIN")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--fuse", action="store_true", help="Apply model.fuse() before benchmark")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _run_forward(
    model,
    input_ids: torch.Tensor,
    repeats: int,
    warmup: int,
    max_new_tokens: int | None,
) -> list[float]:
    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            if max_new_tokens and max_new_tokens > 0:
                _ = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            else:
                _ = model(input_ids)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start.record()
            if max_new_tokens and max_new_tokens > 0:
                _ = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            else:
                _ = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return times


def _vocab_size(model) -> int:
    return getattr(model, "config", None) and getattr(model.config, "vocab_size", None) or 32000


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)

    from gptqmodel import BACKEND, GPTQModel

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"GPU: {torch.cuda.current_device()} ({device_name})")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")
    print(f"Fuse: {args.fuse}")
    print(f"Batch sizes: {args.batch_sizes}, seq_len: {args.seq_len}, max_new_tokens: {args.max_new_tokens}")

    backend = getattr(BACKEND, args.backend.upper(), BACKEND.GPTQ_MARLIN)
    print("Loading model...")
    model = GPTQModel.load(
        args.model_path,
        backend=backend,
        trust_remote_code=True,
        device="cuda:0",
    )
    print("Model loaded.")

    vocab_size = _vocab_size(model)
    cases = [(b, args.seq_len) for b in args.batch_sizes]
    # Reuse the same input_ids for unfused and fused parity comparison.
    input_ids_by_case: dict[tuple[int, int], torch.Tensor] = {}
    for batch, seq in cases:
        input_ids_by_case[(batch, seq)] = torch.randint(
            0, vocab_size, (batch, seq), device="cuda:0"
        )

    unfused_logits: dict[tuple[int, int], torch.Tensor] = {}
    fused_logits: dict[tuple[int, int], torch.Tensor] = {}
    results = []

    for batch, seq in cases:
        input_ids = input_ids_by_case[(batch, seq)]
        print(f"\nBenchmarking unfused batch={batch} seq={seq}")
        with torch.inference_mode():
            out = model(input_ids)
            logits_unfused = out.logits if hasattr(out, "logits") else out
        unfused_logits[(batch, seq)] = logits_unfused.detach().cpu()

        times = _run_forward(
            model,
            input_ids,
            repeats=args.repeats,
            warmup=args.warmup,
            max_new_tokens=args.max_new_tokens,
        )
        mean_ms = _mean(times)
        total_tokens = batch * (seq + (args.max_new_tokens or 0))
        tok_per_sec = total_tokens / (mean_ms / 1000.0)
        results.append({
            "batch": batch,
            "seq": seq,
            "ms": mean_ms,
            "tok/s": tok_per_sec,
            "state": "unfused",
        })
        print(f"  unfused: {mean_ms:.3f} ms, {tok_per_sec:.1f} tok/s")

    if args.fuse:
        print("\nFusing model...")
        counts = model.fuse(qkv=True, gate_up=True, free_original_weights=True)
        print(f"  Fused counts: {counts}")

        for batch, seq in cases:
            input_ids = input_ids_by_case[(batch, seq)]
            print(f"\nBenchmarking fused batch={batch} seq={seq}")
            with torch.inference_mode():
                out = model(input_ids)
                logits_fused = out.logits if hasattr(out, "logits") else out
            fused_logits[(batch, seq)] = logits_fused.detach().cpu()

            times = _run_forward(
                model,
                input_ids,
                repeats=args.repeats,
                warmup=args.warmup,
                max_new_tokens=args.max_new_tokens,
            )
            mean_ms = _mean(times)
            total_tokens = batch * (seq + (args.max_new_tokens or 0))
            tok_per_sec = total_tokens / (mean_ms / 1000.0)
            results.append({
                "batch": batch,
                "seq": seq,
                "ms": mean_ms,
                "tok/s": tok_per_sec,
                "state": "fused",
            })
            print(f"  fused: {mean_ms:.3f} ms, {tok_per_sec:.1f} tok/s")

    print("\n=== Summary ===")
    print(f"{'state':<10} {'batch':>6} {'seq':>5} {'ms':>10} {'tok/s':>12}")
    for r in results:
        print(f"{r['state']:<10} {r['batch']:>6} {r['seq']:>5} {r['ms']:>10.3f} {r['tok/s']:>12.1f}")

    if args.fuse:
        print("\n=== Accuracy ===")
        for batch, seq in cases:
            lu = unfused_logits[(batch, seq)]
            lf = fused_logits[(batch, seq)]
            max_diff = (lu - lf).abs().max().item()
            mean_diff = (lu - lf).abs().mean().item()
            print(f"batch={batch} seq={seq}: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")


if __name__ == "__main__":
    main()
