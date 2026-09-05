#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Run layer -> module/subset propagation sweeps; default to CPU.

Examples:
  python optimize/sweep_sensitivity.py --tiny --output /tmp/tiny-sensitivity
  python optimize/sweep_sensitivity.py --model /models/checkpoint --data held-out.jsonl \
      --layers-path model.layers --probe shared-input --output /tmp/sensitivity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimize.calibration_coverage import _load_tokenizer, _preflight_physical_gpus, _tokenize_sample


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Checkpoint path or Hugging Face model ID")
    source.add_argument("--tiny", action="store_true", help="Random tiny Llama; harness correctness smoke test only")
    parser.add_argument("--revision")
    parser.add_argument("--data", type=Path, help="Disjoint JSONL rows with text, messages, or input_ids")
    parser.add_argument("--layers-path", default="model.layers", help="Actual decoder ModuleList path")
    parser.add_argument("--subsets", type=Path, help="JSON object: label -> full module paths (definition groups)")
    parser.add_argument(
        "--modules", type=Path, help="JSON list of exact target paths; e.g. only optimized QuantLinear modules"
    )
    parser.add_argument("--probe", choices=("output", "shared-input"), default="output")
    parser.add_argument("--amplitudes", default="0.001,0.002")
    parser.add_argument("--top-layers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-group-size", type=int, default=8)
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--physical-gpu", type=int, help="Optional NVIDIA GPU; otherwise CPU. Preflight runs before Torch"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def render_report(report):
    lines = [
        "# Linear sensitivity sweep",
        "",
        "Diagnostic measurements; no model-quality or kernel acceptance claim.",
        "",
        f"Selected layers: {', '.join(report['selected_layers'])}",
        "",
        "| Stage | Target | Amplitude | Final rel. L2 | g effective | Within 2e-3 max-abs |",
        "|---|---|---:|---:|---:|---|",
    ]

    def number(value):
        return "undefined" if value is None else f"{value:.6g}"

    for row in report["rows"]:
        target = row["target"].replace("|", "\\|")
        lines.append(
            f"| {row['stage']} | {target} | {row['amplitude']:.6g} | "
            f"{number(row['final']['relative_l2'])} | {number(row['g_effective'])} | "
            f"{row['within_kernel_output_tolerance']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.max_batches, args.max_length, args.threads) <= 0:
        raise ValueError("Batch, length and thread limits must be positive")
    if args.model and not args.data:
        raise ValueError("A real checkpoint requires held-out --data")
    hardware = _preflight_physical_gpus([args.physical_gpu], allow_busy=False) if args.physical_gpu is not None else []

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from optimize.sensitivity import SensitivitySweep, output_noise, shared_input_noise

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda:0" if hardware else "cpu"
    dtype = getattr(torch, args.dtype)
    if args.tiny:
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=dtype, trust_remote_code=args.trust_remote_code
        ).to(device)
    batches = []
    if args.data:
        tokenizer = None
        with args.data.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "input_ids" in row:
                    chunks = [
                        {
                            key: row[key][: args.max_length]
                            for key in ("input_ids", "attention_mask", "sensitivity_mask")
                            if key in row
                        }
                    ]
                else:
                    if args.tiny:
                        raise ValueError("Tiny mode data must supply input_ids")
                    if tokenizer is None:
                        tokenizer = (
                            AutoTokenizer.from_pretrained(
                                args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
                            )
                            if (args.revision is not None)
                            else _load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
                        )
                    sample = row.get("messages", row.get("text"))
                    if sample is None:
                        raise ValueError("Data row requires input_ids, text or messages")
                    chunks = _tokenize_sample(tokenizer, sample, args.max_length, min_length=1)
                for chunk in chunks:
                    ids = torch.tensor([chunk["input_ids"]], dtype=torch.long, device=device)
                    if ids.numel() == 0:
                        raise ValueError("Empty token sequence")
                    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                    for key in ("attention_mask", "sensitivity_mask"):
                        if key in chunk:
                            batch[key] = torch.tensor([chunk[key]], device=device)
                            if batch[key].shape != ids.shape:
                                raise ValueError(f"{key} must align with input_ids")
                    batches.append(batch)
                    if len(batches) >= args.max_batches:
                        break
                if len(batches) >= args.max_batches:
                    break
    else:
        batches = [
            {"input_ids": torch.randint(1, 64, (1, min(args.max_length, 12)), device=device)}
            for _ in range(args.max_batches)
        ]
    layer_list = model.get_submodule(args.layers_path)
    layers = [f"{args.layers_path}.{name}" for name, _ in layer_list.named_children()]
    subsets = json.loads(args.subsets.read_text()) if args.subsets else None
    modules = json.loads(args.modules.read_text()) if args.modules else None
    sweep = SensitivitySweep(
        model,
        layers,
        subsets=subsets,
        seed=args.seed,
        module_names=modules,
        candidate=output_noise if args.probe == "output" else shared_input_noise,
    )

    def progress(row):
        print(
            f"[{row['stage']}] {row['target']} amplitude={row['amplitude']:g} "
            f"E={row['final']['relative_l2']} g={row['g_effective']}",
            flush=True,
        )

    report = sweep.sweep(
        batches,
        amplitudes=[float(a) for a in args.amplitudes.split(",")],
        top_k=args.top_layers,
        pairwise=not args.no_pairs,
        max_group_size=args.max_group_size,
        progress=progress,
    )
    report["run"] = {
        "model": args.model,
        "revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "model_class": type(model).__name__,
        "config": model.config.to_dict(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "dtype": args.dtype,
        "hardware": hardware,
        "tiny_random_fixture": args.tiny,
        "layers_path": args.layers_path,
        "data": str(args.data) if args.data else None,
        "token_sha256": hashlib.sha256(json.dumps([b["input_ids"].tolist() for b in batches]).encode()).hexdigest(),
        "input_sha256": hashlib.sha256(
            json.dumps([{k: v.tolist() for k, v in b.items()} for b in batches], sort_keys=True).encode()
        ).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "sensitivity.md").write_text(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
