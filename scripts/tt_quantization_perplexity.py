# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure the quality cost of native block-float quantization as perplexity.

A 60-question GSM8K run cannot resolve a few points of damage: one question is
1.7 points, so any delta inside ~±6 points is noise. Perplexity over a fixed
held-out text is the same measurement with far less variance, and it costs
seconds per window instead of ~40 s per question.

Run it twice -- once dense, once with ``--native`` -- and compare. The text must
not be the calibration text, or the number is optimistic.

Usage::

    python scripts/tt_quantization_perplexity.py <model> --text FILE [--native DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def load_text(path: str, *, field: str | None = None) -> str:
    """Read plain text, or concatenate a JSONL field such as ``question``."""

    with open(path) as handle:
        raw = handle.read()
    if path.endswith(".jsonl"):
        parts = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            parts.append(row[field] if field else " ".join(str(v) for v in row.values()))
        return "\n\n".join(parts)
    return raw


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--text", required=True, help="held-out text or JSONL file")
    parser.add_argument("--field", default=None, help="JSONL field to concatenate")
    parser.add_argument("--native", default=None, help="block-float checkpoint dir")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--windows", type=int, default=40)
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("USE_TORCH_XLA", "0")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    restored = 0
    if args.native:
        from gptqmodel.utils.tt_blockfloat_quant import (
            load_native_checkpoint_into_dense_model,
        )

        restored = len(load_native_checkpoint_into_dense_model(model, args.native))
    model.eval()

    text = load_text(args.text, field=args.field)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    window = args.window
    usable = min(args.windows, ids.numel() // window)
    if usable == 0:
        print(f"[P] ERROR: only {ids.numel()} tokens, need {window}", flush=True)
        return 2

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for index in range(usable):
            chunk = ids[index * window : (index + 1) * window].unsqueeze(0)
            out = model(chunk, labels=chunk)
            # HF averages over window-1 predicted tokens.
            total_nll += float(out.loss) * (window - 1)
            total_tokens += window - 1
            if (index + 1) % 10 == 0:
                print(f"[P] {index + 1}/{usable} windows", flush=True)

    ppl = math.exp(total_nll / total_tokens)
    print(
        f"[P] RESULT model={args.model} native={args.native} restored={restored} "
        f"tokens={total_tokens} ppl={ppl:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
