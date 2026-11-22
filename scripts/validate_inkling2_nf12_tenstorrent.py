#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare Inkling 2 Mini's native model forwards with and without device NF12."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from bench_nf12_tenstorrent import _errors, _runtime, _same
from PIL import Image

from gptqmodel.models.ports.inkling2_loop import Inkling2LoopForConditionalGeneration, install_inkling2_nf12
from gptqmodel.models.ports.inkling2_loop.image_processing import Inkling2ImageProcessor
from gptqmodel.models.ports.inkling2_loop.tenstorrent_nf12 import _targets
from gptqmodel.models.ports.inkling2_loop.vision import _bilinear_taps
from gptqmodel.models.ports.native.auto import NativeTokenizer


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


def _logits_with_image_features(model, ids, features):
    inputs_embeds = model.model.get_input_embeddings()(ids)
    features = features.to(inputs_embeds)
    mask = model.model.get_placeholder_mask(ids, inputs_embeds, features, model.config.image_token_id)
    return model(inputs_embeds=inputs_embeds.masked_scatter(mask, features), use_cache=False).logits


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


def _matched_hook(source, backend, copy, counts, name):
    def compare(module, arguments, actual):
        x = arguments[0]
        if not hasattr(module, "in_features"):
            expected = module.embed_norm(source[x])
        else:
            layer = module.layer
            baseline = layer.upload_baseline(source)
            bias = module.bias
            if backend == "ttnn":
                rows = x.reshape(-1, x.shape[-1])
                expected = layer.linear_device(
                    layer.upload_activation(rows),
                    None if bias is None else layer.upload_bias(bias),
                    baseline=baseline,
                )
                expected = copy(expected)[: len(rows), :module.out_features].reshape_as(actual)
            else:
                bias = None if bias is None else bias.to(layer.device)
                expected = copy(layer.linear(x.to(layer.device), bias, baseline=baseline))
        try:
            _same(actual, expected)
        except AssertionError as exc:
            raise AssertionError(f"Matched device BF16 forward differs for {name}") from exc
        counts[name] += 1
    return compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backend", choices=("ttnn", "ttxla"), required=True)
    p.add_argument("--chip", type=int, default=0)
    p.add_argument("--vision-only", action="store_true")
    p.add_argument("--embedding-layout", choices=("canonical", "row_major"), default="canonical")
    p.add_argument("--verify-weights", action="store_true", help="Check every selected weight and forward against a BF16 device reference")
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
        print(json.dumps({"phase": "baseline_vision", "cases": list(cases)}), flush=True)
        paired = {}
        if not args.vision_only:
            tokenizer = NativeTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
            text_ids = tokenizer("Describe the colors.", return_tensors="pt").input_ids
            image_inputs = cases["processor_pair"]
            image_tokens = int(image_inputs["image_grid_thw"].prod(-1).sum())
            image_ids = torch.cat((torch.full((1, image_tokens), model.config.image_token_id), text_ids), dim=1)
            for case, ids, inputs in (("text", text_ids, {}), ("image", image_ids, image_inputs)):
                paired[case] = (ids, inputs, _logits(model, ids, inputs))
                print(json.dumps({"phase": "baseline_logits", "case": case, "tokens": ids.numel()}), flush=True)

        sources = {}
        if args.verify_weights:
            for name in _targets(model, vision=True, text=not args.vision_only):
                sources[name] = model.get_submodule(name).weight.detach()

        with _runtime(args.backend, args.chip) as (_, device, copy, sync):
            descriptors = install_inkling2_nf12(
                model, backend=args.backend, device=device, text=not args.vision_only,
                embedding_layout=args.embedding_layout,
            )
            sync()
            if args.descriptors:
                entries = {}
                for key, value in descriptors.items():
                    layer = model.get_submodule(key.removesuffix(".weight")).layer
                    entry = entries[key] = {
                        **asdict(value),
                        "packed_bytes": value.packed_bytes,
                        "resident_bytes": layer.resident_bytes,
                    }
                    if hasattr(layer, "storage_descriptor"):
                        entry["device_storage"] = asdict(layer.storage_descriptor)
                args.descriptors.write_text(json.dumps(entries, indent=2) + "\n")
            print(json.dumps({"backend": args.backend, "matrices": len(descriptors),
                              "canonical_bytes": sum(d.packed_bytes for d in descriptors.values())}), flush=True)
            for name, source in sources.items():
                layer = model.get_submodule(name).layer
                for start in range(0, len(source), 2048):
                    rows = torch.arange(start, min(start + 2048, len(source)))
                    _same(layer.embedding(rows).cpu(), source[rows])
            counts = Counter()
            hooks = []
            if sources:
                print(json.dumps({"verified_logical_matrices": len(sources)}), flush=True)
                hooks = [
                    model.get_submodule(name).register_forward_hook(
                        _matched_hook(source, args.backend, copy, counts, name)
                    )
                    for name, source in sources.items()
                ]
                sources.clear()
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
                if name == "image" and args.verify_weights:
                    fixed_vision = _logits_with_image_features(model, ids, vision_reference["processor_pair"]["tokens"])
                    print(json.dumps({
                        "logits": "image_cpu_vision", **_logit_errors(fixed_vision, expected, ids, model.config.image_token_id),
                    }), flush=True)
            for hook in hooks:
                hook.remove()
            if counts:
                print(json.dumps({"matched_forward_calls": counts.total(), "matched_modules": len(counts)}), flush=True)


if __name__ == "__main__":
    main()
