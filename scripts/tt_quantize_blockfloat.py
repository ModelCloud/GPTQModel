# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Quantize a decoder model onto the Tenstorrent block-float grid and save it.

GPTQ error feedback is solved per column directly on the hardware grid, so the
saved checkpoint holds codes plus one exponent per 16-value block -- the weights
load as native ``bfloat4_b``/``bfloat8_b`` with no host-side re-quantisation.

Usage::

    python scripts/tt_quantize_blockfloat.py <model-id-or-path> --out DIR [--bits 4]

The checkpoint records the source model id and whether its tokenizer has a chat
template, because a native checkpoint scored with the wrong tokenizer (or an
untemplated prompt for an instruct model) produces a meaningless number.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="model id or local path to quantize")
    parser.add_argument("--out", required=True, help="checkpoint output directory")
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument(
        "--calibration",
        default="The capital of France is Paris. " * 40,
        help="calibration text used to build the per-layer Hessians",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("USE_TORCH_XLA", "0")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.utils.tt_blockfloat_quant import (
        load_native_checkpoint_into_dense_model,
        quantize_model_blockfloat,
        save_blockfloat_checkpoint,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    probe = "The capital of France is"
    ids = tokenizer(probe, return_tensors="pt").input_ids
    with torch.no_grad():
        dense = model(ids).logits[0, -1].float().clone()

    started = time.time()
    report = quantize_model_blockfloat(
        model,
        tokenizer,
        bits=args.bits,
        nsamples=args.nsamples,
        text=args.calibration,
    )
    print(
        f"[Q] quantized {len(report)} layers in {time.time() - started:.0f}s",
        flush=True,
    )
    errors = [v["norm_l1"] for v in report.values()]
    on_grid = all(v["on_grid"] for v in report.values())
    print(
        f"[Q] mean norm_l1={sum(errors) / len(errors):.4f} on_grid={on_grid}",
        flush=True,
    )
    if not on_grid:
        print("[Q] ERROR: some layers are not on the block-float grid", flush=True)
        return 1

    info = save_blockfloat_checkpoint(
        model,
        args.out,
        bits=args.bits,
        metadata={
            "source_model": args.model,
            "tokenizer_has_chat_template": bool(
                getattr(tokenizer, "chat_template", None)
            ),
        },
    )
    print(
        f"[Q] saved {info['layers']} layers, "
        f"{info['payload_bytes'] / 1024 / 1024:.1f} MiB -> {args.out}",
        flush=True,
    )

    with torch.no_grad():
        quantized = model(ids).logits[0, -1].float()
    cosine = float(
        torch.nn.functional.cosine_similarity(dense, quantized, dim=0)
    )
    print(
        f"[Q] logits cosine={cosine:.5f} "
        f"argmax dense={int(dense.argmax())} quant={int(quantized.argmax())}",
        flush=True,
    )

    restored = load_native_checkpoint_into_dense_model(model, args.out)
    with torch.no_grad():
        reloaded = model(ids).logits[0, -1].float()
    drift = float((reloaded - quantized).abs().max())
    print(
        f"[Q] reloaded {len(restored)} layers, logits maxdiff={drift:.2e}",
        flush=True,
    )
    if drift != 0.0:
        print("[Q] ERROR: checkpoint did not reload bit-exactly", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
