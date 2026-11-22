#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare Inkling 2 Mini's native model forwards with and without device NF12."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from bench_nf12_tenstorrent import _errors, _runtime, _same
from PIL import Image
from transformers import AutoTokenizer, Inkling2ImageProcessor

from gptqmodel.models.inkling2_loop import Inkling2LoopForConditionalGeneration, install_inkling2_nf12
from gptqmodel.models.inkling2_loop.vision import _bilinear_taps


def _vision_inputs(checkpoint):
    return {
        "square": {
            "pixel_values": torch.randn(4, 40, 40, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        "rectangle": {
            "pixel_values": torch.randn(6, 40, 40, 3),
            "image_grid_thw": torch.tensor([[1, 2, 3]]),
        },
        "processor_pair": Inkling2ImageProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )([Image.new("RGB", (79, 38), (17, 79, 244)), Image.new("RGB", (140, 70), (140, 10, 90))],
          return_tensors="pt"),
    }


def _vision(tower, inputs):
    parts = {}
    hooks = [
        module.register_forward_hook(lambda _m, _a, out, name=name: parts.update({name: out}))
        for name, module in (
            ("patch", tower.embeddings.patch_embedding),
            ("embeddings", tower.embeddings),
            ("block_0", tower.layers[0]),
        )
    ]
    try:
        out = tower(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    pos_sums = []
    merge = tower.config.spatial_merge_size
    for _, h, w in inputs["image_grid_thw"].tolist():
        indices, weights = _bilinear_taps(h * merge, w * merge, tower.embeddings.grid_size, merge)
        pos_sums.append((tower.embeddings.position_embedding(indices) * weights[:, :, None]).sum(1))
    parts["position"] = torch.cat(pos_sums).bfloat16()
    parts["hidden"] = out.last_hidden_state
    parts["tokens"] = out.pooler_output
    return parts


def _logits(model, ids, inputs):
    return model(input_ids=ids, use_cache=False, **inputs).logits


def _logit_errors(output, oracle, ids, image_token):
    expected = oracle.log_softmax(-1)
    actual = output.log_softmax(-1)
    targets = ids[:, 1:].flatten().masked_fill(ids[:, 1:].flatten() == image_token, -100)
    want_ppl = F.cross_entropy(oracle[:, :-1].flatten(0, 1), targets).exp()
    got_ppl = F.cross_entropy(output[:, :-1].flatten(0, 1), targets).exp()
    return {
        **_errors(output, oracle),
        "kld": float((expected.exp() * (expected - actual)).sum(-1).mean()),
        "bf16_ppl": float(want_ppl),
        "nf12_ppl": float(got_ppl),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backend", choices=("ttnn", "ttxla"), required=True)
    p.add_argument("--chip", type=int, default=0)
    p.add_argument("--vision-only", action="store_true")
    p.add_argument("--descriptors", type=Path, help="Write weight provenance into a local JSON file")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(12)
    with torch.inference_mode():
        model = Inkling2LoopForConditionalGeneration.from_pretrained(
            args.checkpoint, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager",
        ).eval()
        tower = model.model.vision_tower
        cases = _vision_inputs(args.checkpoint)
        vision_reference = {name: _vision(tower, inputs) for name, inputs in cases.items()}
        paired = {}
        if not args.vision_only:
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
            text_ids = tokenizer("Describe the colors.", return_tensors="pt").input_ids
            image_inputs = cases["processor_pair"]
            image_tokens = int(image_inputs["image_grid_thw"].prod(-1).sum())
            image_ids = torch.cat((torch.full((1, image_tokens), model.config.image_token_id), text_ids), dim=1)
            paired = {
                "text": (text_ids, {}, _logits(model, text_ids, {})),
                "image": (image_ids, image_inputs, _logits(model, image_ids, image_inputs)),
            }

        with _runtime(args.backend, args.chip) as (_, device, _, sync):
            descriptors = install_inkling2_nf12(
                model, backend=args.backend, device=device, text=not args.vision_only,
            )
            sync()
            if args.descriptors:
                args.descriptors.write_text(json.dumps({key: asdict(value) for key, value in descriptors.items()},
                                                       indent=2) + "\n")
            print(json.dumps({"backend": args.backend, "matrices": len(descriptors),
                              "canonical_bytes": sum(d.packed_bytes for d in descriptors.values())}), flush=True)
            for name, inputs in cases.items():
                actual = _vision(tower, inputs)
                expected = vision_reference[name]
                _same(actual["position"], expected["position"])
                print(json.dumps({
                    "vision": name, "patch_input_tiles": len(inputs["pixel_values"]),
                    **{stage: _errors(value, expected[stage]) for stage, value in actual.items()},
                }), flush=True)
            for name, (ids, inputs, expected) in paired.items():
                actual = _logits(model, ids, inputs)
                print(json.dumps({"logits": name, **_logit_errors(actual, expected, ids, model.config.image_token_id)}), flush=True)


if __name__ == "__main__":
    main()
