#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""De-fuse a DeepSeek-V4 BF16 checkpoint and save it per transformer layer."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from defuser import convert_model
from safetensors.torch import save_file
from tokenicer import Tokenicer
from transformers import AutoConfig, AutoModelForCausalLM


def _group_state_dict_by_layer(state_dict: dict[str, torch.Tensor]) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    other: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        m = re.match(r"^(model\.layers\.(\d+)\.)(.*)$", name)
        if m:
            idx = int(m.group(2))
            layers.setdefault(idx, {})[name] = tensor
        else:
            other.setdefault("misc", {})[name] = tensor
    return layers, other


def save_per_layer(
    model_dir: Path | str,
    output_dir: Path | str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"[load] Loading model from {model_dir} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[load] Loaded {param_count / 1e9:.1f}B parameters")

    print("[defuse] Running Defuser convert_model ...")
    converted = convert_model(model)
    print(f"[defuse] convert_model returned {converted}")

    print("[save] Grouping state dict per layer ...")
    state_dict = model.state_dict()
    layers, other = _group_state_dict_by_layer(state_dict)

    weight_map: dict[str, str] = {}
    total_size = 0
    file_index = 1
    num_files = len(layers) + (1 if other else 0)

    def _write_shard(name: str, shard: dict[str, torch.Tensor]) -> None:
        nonlocal file_index, total_size
        filename = f"model-{file_index:05d}-of-{num_files:05d}.safetensors"
        save_file(shard, str(output_dir / filename))
        for key in shard:
            weight_map[key] = filename
        total_size += sum(t.element_size() * t.numel() for t in shard.values())
        print(f"[save] Wrote {filename} ({len(shard)} tensors)")
        file_index += 1

    if other and other.get("misc"):
        _write_shard("misc", other["misc"])

    for idx in sorted(layers):
        _write_shard(f"layer_{idx}", layers[idx])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("[save] Saving config / tokenizer ...")
    model.config.save_pretrained(output_dir)
    tok = Tokenicer.load(str(model_dir), trust_remote_code=True)
    if tok.tokenizer is not None:
        tok.tokenizer.save_pretrained(str(output_dir))
    else:
        # Fallback: copy any tokenizer-style files from source.
        for item in model_dir.iterdir():
            if item.name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
            ) or "merges" in item.name or "vocab" in item.name:
                dest = output_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    print(f"[done] Per-layer checkpoint saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    save_per_layer(args.model_dir, args.output_dir, args.device, dtype)


if __name__ == "__main__":
    main()
