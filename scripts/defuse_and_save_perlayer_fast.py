#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""De-fuse a DeepSeek-V4 BF16 checkpoint one layer at a time and save per-layer shards.

Loads the full fused model into CPU RAM, defuses only one decoder layer at a time,
writes that layer to a separate safetensors file, then discards the layer before
moving on. This keeps peak CPU memory close to the original fused checkpoint size
rather than creating the full defused copy in RAM all at once.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from defuser.modeling.moe_experts_interface import _unfuse_experts_weights_inplace
from safetensors.torch import save_file
from tokenicer import Tokenicer
from transformers import AutoConfig, AutoModelForCausalLM


def _apply_expert_gate(module: Any, gate_out: Any, up_out: Any) -> Any:
    if gate_out is None:
        return module.act_fn(up_out)
    if hasattr(module, "_apply_gate"):
        return module._apply_gate(torch.cat([gate_out, up_out], dim=-1))
    return module.act_fn(gate_out) * up_out


def _fast_experts_forward(
    self: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    batch_format = hidden_states.dim() == 3
    if batch_format:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        top_k_index = top_k_index.view(-1, top_k_index.size(-1))
        top_k_weights = top_k_weights.view(-1, top_k_weights.size(-1))
    else:
        hidden_dim = hidden_states.size(-1)

    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    dtype = hidden_states.dtype

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1).to(dtype)
    expert_ids = top_k_index.reshape(-1)
    selected_hidden = hidden_states[token_idx]

    out = torch.zeros(token_idx.size(0), hidden_dim, device=device, dtype=dtype)
    active_experts = torch.unique(expert_ids).cpu().tolist()
    for expert_idx in active_experts:
        mask = expert_ids == expert_idx
        expert = getattr(self, str(expert_idx))
        ei = selected_hidden[mask]
        if hasattr(expert, "gate_proj"):
            gate_out = expert.gate_proj(ei)
            up_out = expert.up_proj(ei)
        else:
            gate_out = None
            up_out = expert.up_proj(ei)
        gated = _apply_expert_gate(self, gate_out, up_out)
        out[mask] = expert.down_proj(gated).to(dtype)

    out = out * sample_weights.unsqueeze(-1)
    final = out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    if batch_format:
        final = final.view(batch_size, seq_len, hidden_dim)
    return final


def _defuse_moe_experts(layer: torch.nn.Module) -> None:
    """In-place defuse of a single decoder layer's mlp.experts."""

    ok = _unfuse_experts_weights_inplace(layer.mlp.experts)
    if ok:
        layer.mlp.experts.forward = MethodType(_fast_experts_forward, layer.mlp.experts)
    else:
        print(f"[warn] Could not defuse layer.mlp.experts: {layer}")


def _add_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def save_per_layer_fast(
    model_dir: Path | str,
    output_dir: Path | str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"

    print(f"[load] Loading fused model from {model_dir} on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print("[dtype] Casting all floating point parameters to bfloat16 ...")
    model = model.to(dtype)

    num_layers = len(model.model.layers)
    print(f"[defuse] Defusing and saving {num_layers} layers one at a time ...")

    weight_map: dict[str, str] = {}
    total_size = 0
    file_index = 1

    def _write_shard(name: str, shard: dict[str, torch.Tensor]) -> None:
        nonlocal file_index, total_size
        filename = f"model-{file_index:05d}-of-{num_files:05d}.safetensors"
        save_file(shard, str(output_dir / filename))
        for key in shard:
            weight_map[key] = filename
        total_size += sum(t.element_size() * t.numel() for t in shard.values())
        print(f"[save] {name}: {filename} ({len(shard)} tensors, {sum(t.element_size()*t.numel() for t in shard.values())/1024**3:.1f} GB)")
        file_index += 1

    num_files = num_layers + 1  # + misc

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        _defuse_moe_experts(layer)

        sd = layer.state_dict()
        prefixed = _add_prefix(sd, f"model.layers.{layer_idx}.")
        _write_shard(f"layer_{layer_idx}", prefixed)

        # Drop the defused layer to keep CPU memory bounded.
        model.model.layers[layer_idx] = None
        del layer, sd, prefixed
        gc.collect()

    # Misc shard: everything that is not a layer parameter.
    print("[save] Collecting non-layer parameters ...")
    full_sd = model.state_dict()
    misc: dict[str, torch.Tensor] = {}
    for name, tensor in full_sd.items():
        if not re.match(r"^model\.layers\.\d+\.", name):
            misc[name] = tensor
    _write_shard("misc", misc)
    del full_sd, misc
    gc.collect()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("[save] Saving config / tokenizer files ...")
    model.config.save_pretrained(output_dir)
    tok = Tokenicer.load(str(model_dir), trust_remote_code=True)
    if tok.tokenizer is not None:
        tok.tokenizer.save_pretrained(str(output_dir))
    else:
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

    print(f"[done] Per-layer defused checkpoint saved to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    save_per_layer_fast(args.model_dir, args.output_dir, dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
