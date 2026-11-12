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
    parser.add_argument(
        "--embed-bits",
        type=int,
        default=8,
        choices=(4, 8),
        help=(
            "width for the (tied) input embedding and output head. 8 is the "
            "default because a vocabulary-sized matrix tolerates far less "
            "relative error than a decoder projection."
        ),
    )
    parser.add_argument(
        "--bits-map",
        default=None,
        help=(
            "per-layer storage widths, e.g. "
            "'q_proj=8,k_proj=8,v_proj=8,o_proj=8,down_proj=8'. Layers that do "
            "not match keep --bits. bfloat8_b costs one byte per weight and is "
            "close to lossless; bfloat4_b is roughly a 2.8-bit format, so the "
            "extra byte is best spent on the layers that carry the most output "
            "error rather than on all of them."
        ),
    )
    parser.add_argument("--nsamples", type=int, default=8)
    calibration_source = parser.add_mutually_exclusive_group(required=True)
    calibration_source.add_argument(
        "--calibration",
        help=(
            "calibration text for the per-layer Hessians. Required in some form: "
            "a narrow passage tunes GPTQ's error feedback to that passage, which "
            "is catastrophic on other text (OPT-125M measured perplexity 296 from "
            "a 409-token synthetic passage against 20.3 from real text)."
        ),
    )
    calibration_source.add_argument(
        "--calibration-dataset",
        metavar="PATH_OR_REPO",
        help=(
            "calibration records from the same source the GPTQ test harness uses: "
            "a .json/.json.gz/.jsonl/.parquet file or a Hub dataset repo id "
            "(defaults to $GPTQMODEL_CALIBRATION_PARQUET, then the local "
            "nm-calibration parquet)"
        ),
    )
    parser.add_argument(
        "--calibration-rows",
        type=int,
        default=0,
        help=(
            "use only the first N calibration records (0 = all). The GPTQ test "
            "harness calibrates on a fixed record count, so this is the knob that "
            "aligns the two paths without changing the source file."
        ),
    )
    args = parser.parse_args(argv)

    bits = args.bits
    if args.bits_map:
        bits = {"*": args.bits}
        for entry in args.bits_map.split(","):
            entry = entry.strip()
            if not entry:
                continue
            pattern, _, width = entry.partition("=")
            if not pattern or width not in ("4", "8"):
                parser.error(
                    f"--bits-map entry {entry!r} must look like 'q_proj=8' with a width of 4 or 8"
                )
            bits[pattern.strip()] = int(width)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("USE_TORCH_XLA", "0")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.utils.tt_blockfloat_quant import (
        load_native_checkpoint_into_dense_model,
        quantize_model_blockfloat,
        save_blockfloat_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    if args.calibration is not None:
        calibration = args.calibration
        calibration_tokens = tokenizer(
            calibration, return_tensors="pt"
        ).input_ids.shape[1]
        calibration_records = 1
    else:
        from gptqmodel.utils.calibration import load_calibration_records

        calibration = load_calibration_records(
            args.calibration_dataset, rows=args.calibration_rows
        )
        calibration_tokens = sum(
            tokenizer(row, return_tensors="pt").input_ids.shape[1]
            for row in calibration
        )
        calibration_records = len(calibration)
    # Report both numbers. They differ, and the difference is the point: the
    # record count is what the source holds, the window budget is what the
    # Hessian actually reads.
    print(
        f"[Q] calibration: {calibration_records} record(s), {calibration_tokens} "
        f"tokens; Hessian reads {args.nsamples} windows of 256 = "
        f"{args.nsamples * 256} tokens spread across the stream",
        flush=True,
    )

    probe = "The capital of France is"
    ids = tokenizer(probe, return_tensors="pt").input_ids
    with torch.no_grad():
        dense = model(ids).logits[0, -1].float().clone()

    started = time.time()
    report = quantize_model_blockfloat(
        model,
        tokenizer,
        bits=bits,
        nsamples=args.nsamples,
        text=calibration,
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

    info = save_blockfloat_model(
        model,
        args.out,
        bits=bits,
        embed_bits=args.embed_bits,
        tokenizer=tokenizer,
        metadata={
            "source_model": args.model,
            "tokenizer_has_chat_template": bool(
                getattr(tokenizer, "chat_template", None)
            ),
        },
    )
    print(
        f"[Q] saved {info['layers']} layers, "
        f"{info['payload_bytes'] / 1024 / 1024:.1f} MiB "
        f"({'+'.join(str(w) for w in info['bits_per_layer'])}-bit linears, "
        f"{args.embed_bits}-bit embed) -> {args.out}",
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
