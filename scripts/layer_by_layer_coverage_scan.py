#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Layer-by-layer calibration coverage scanner for DeepSeek-V4 BF16 defused checkpoints.

Keeps the full model in CPU RAM but only one decoder layer on GPU 6/7 at a time,
so the scan fits on two ~96 GB GPUs while the BF16 weights stay on host memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch


# Make optimize/calibration_coverage importable without turning it into a package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "optimize"))

import calibration_coverage as cov  # noqa: E402


def _apply_expert_gate(module: Any, gate_out: Any, up_out: Any) -> Any:
    """Gating helper matching Defuser's moe_experts_interface logic."""

    import torch

    if gate_out is None:
        return module.act_fn(up_out)
    if hasattr(module, "_apply_gate"):
        return module._apply_gate(torch.cat([gate_out, up_out], dim=-1))
    return module.act_fn(gate_out) * up_out


def _fast_experts_forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
    """Vectorized per-expert loop that touches only active experts."""

    import torch

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
        if hasattr(expert, "gate_proj"):
            gate_out = expert.gate_proj(selected_hidden[mask])
            up_out = expert.up_proj(selected_hidden[mask])
        else:
            gate_out = None
            up_out = expert.up_proj(selected_hidden[mask])
        gated = _apply_expert_gate(self, gate_out, up_out)
        out[mask] = expert.down_proj(gated).to(dtype)

    out = out * sample_weights.unsqueeze(-1)
    final = out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    if batch_format:
        final = final.view(batch_size, seq_len, hidden_dim)
    return final


def _fast_grouped_experts_forward(
    self: Any,
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
) -> Any:
    """Single grouped GEMM over all token assignments; avoids per-expert host sync."""

    import torch
    import torch.nn.functional as F

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

    # Expand each token for every chosen expert.
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int64).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    expert_ids = top_k_index.reshape(-1).long()
    gathered_hidden = hidden_states[token_idx]
    gathered_weights = top_k_weights.reshape(-1).to(dtype)

    # Sort assignments by expert so group sizes are contiguous.
    sorted_expert_ids, order = torch.sort(expert_ids)
    sorted_hidden = gathered_hidden[order]
    sorted_weights = gathered_weights[order]

    # Determine active experts and remap sorted ids to active indices.
    group_sizes = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
    active_experts = torch.nonzero(group_sizes > 0, as_tuple=False).flatten().long()
    num_active = active_experts.shape[0]

    active_index = torch.full((self.num_experts,), -1, dtype=torch.int64, device=device)
    active_index[active_experts] = torch.arange(num_active, device=device)
    sorted_active_idx = active_index[sorted_expert_ids]
    sort_order2 = torch.sort(sorted_active_idx).indices
    sorted_hidden = sorted_hidden[sort_order2]
    sorted_weights = sorted_weights[sort_order2]
    sorted_token_idx = token_idx[order][sort_order2]

    group_sizes_active = torch.bincount(sorted_active_idx, minlength=num_active)
    offsets = torch.cumsum(group_sizes_active, dim=0).to(torch.int32)

    # Gather active weights and transpose to grouped_mm layout [groups, K, M].
    gate_up = self.gate_up_proj[active_experts].transpose(1, 2).contiguous()
    down = self.down_proj[active_experts].transpose(1, 2).contiguous()

    gate_up_out = F.grouped_mm(sorted_hidden, gate_up, offs=offsets)
    gate, up = gate_up_out.chunk(2, dim=-1)
    gate = gate.clamp(max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    gated = self.act_fn(gate) * up

    down_out = F.grouped_mm(gated, down, offs=offsets)
    weighted = down_out * sorted_weights.unsqueeze(-1)

    final = torch.zeros_like(hidden_states)
    final.index_add_(0, sorted_token_idx, weighted.to(dtype))

    if batch_format:
        final = final.view(batch_size, seq_len, hidden_dim)
    return final


def _patch_fast_grouped_experts_forward(model: Any) -> None:
    """Replace fused DeepseekV4Experts forward with a grouped-GEMM version."""

    count = 0
    for module in model.modules():
        if module.__class__.__name__ == "DeepseekV4Experts" and hasattr(module, "gate_up_proj"):
            module.forward = MethodType(_fast_grouped_experts_forward, module)
            count += 1
    print(f"[patch] Grouped-expert fast-forward patched {count} DeepseekV4Experts module(s)")


def _patch_fast_experts_forward(model: Any) -> None:
    """Replace Defuser's all-experts loop with an active-expert loop for speed."""

    count = 0
    for module in model.modules():
        if module.__class__.__name__ == "DeepseekV4Experts":
            module.forward = MethodType(_fast_experts_forward, module)
            count += 1
    print(f"[patch] Per-expert fast-forward patched {count} DeepseekV4Experts module(s)")


def _parse_physical_gpu(spec: str | None) -> list[int]:
    if not spec:
        return []
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _preflight_gpus(physical_indices: list[int], allow_busy: bool) -> list[torch.device]:
    """Resolve physical GPUs, set CUDA_VISIBLE_DEVICES, and return torch devices."""

    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,compute_mode,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    by_index: dict[int, dict[str, Any]] = {}
    for line in out.strip().splitlines():
        idx, uuid, name, mem_used, compute_mode, util = [x.strip() for x in line.split(",", 5)]
        by_index[int(idx)] = {
            "index": int(idx), "uuid": uuid, "name": name,
            "mem_used": int(mem_used), "util": int(util),
        }

    targets = []
    for i in physical_indices:
        if i not in by_index:
            raise RuntimeError(f"Physical GPU {i} not found")
        info = by_index[i]
        if not allow_busy and info["util"] > 0:
            raise RuntimeError(f"Physical GPU {i} is busy (util={info['util']}%)")
        targets.append(info)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(t["uuid"] for t in targets)

    import torch
    return [torch.device(f"cuda:{i}") for i in range(len(targets))]


def _measure_layer_weight_size(model: Any, device: torch.device) -> int:
    """Move the first decoder layer to a GPU and measure the delta in allocated memory."""

    import torch

    layer = model.model.layers[0]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    layer.to(device)
    after = torch.cuda.memory_allocated(device)
    layer.to("cpu")
    torch.cuda.empty_cache()
    size = after - before
    print(f"[mem] Measured layer-0 weight footprint on {device}: {size / (1 << 30):.2f} GiB")
    return size


def _estimate_max_layers_per_gpu(model: Any, devices: list[torch.device], requested: int | None) -> int:
    """Return the number of contiguous layers that can be kept on each GPU.

    If ``requested`` is a positive number, use it directly.  Otherwise measure the
    first layer and compute the maximum that fits while leaving a few GiB for
    activations, attention temporaries, and allocator fragmentation.
    """

    import torch

    if requested is not None and requested > 0:
        print(f"[mem] Using requested {requested} layer(s) per GPU")
        return requested

    if not devices:
        return 1

    per_layer = _measure_layer_weight_size(model, devices[0])
    reserve = 8 * (1 << 30)  # 8 GiB headroom for activations / caches
    max_layers = []
    for device in devices:
        free, _ = torch.cuda.mem_get_info(device)
        capacity = free - reserve
        n = max(1, int(capacity // per_layer)) if capacity > 0 else 1
        max_layers.append(n)
        print(
            f"[mem] {device}: free={free / (1 << 30):.1f} GiB, "
            f"per-layer={per_layer / (1 << 30):.2f} GiB -> max {n} layer(s)/GPU"
        )
    layers_per_device = min(max_layers)
    print(f"[mem] Running with {layers_per_device} layer(s) per GPU ({layers_per_device * len(devices)}-layer blocks)")
    return layers_per_device


def _is_already_defused(model: Any) -> bool:
    """Detect a checkpoint that already has per-expert nn.Linear children."""

    for name, module in model.named_modules():
        if module.__class__.__name__ == "DeepseekV4Experts":
            children = list(module.named_children())
            return any(name.isdigit() for name, _ in children[:10])
    return False


def _load_model(model_dir: Path, dtype_str: str, defuse: bool = True) -> Any:
    """Load a DeepSeek-V4 BF16 checkpoint on CPU and ensure consistent dtype."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"
    dtype = getattr(torch, dtype_str)
    print(f"[load] Loading model from {model_dir} on CPU ({dtype_str}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    already_defused = _is_already_defused(model)
    if defuse and not already_defused:
        from defuser import convert_model

        print("[defuse] Ensuring model is defused ...")
        convert_model(model)
        already_defused = True
    elif defuse and already_defused:
        print("[defuse] Checkpoint already defused; skipping convert_model ...")
    print("[dtype] Casting model to target dtype ...")
    model = model.to(dtype)
    if already_defused:
        _patch_fast_experts_forward(model)
    model.eval()
    return model


def _tokenize_chunks(
    tokenizer,
    samples: list[str] | list[list[dict[str, str]]],
    concat_size: int,
    min_length: int,
    apply_chat_template: bool,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for sample in samples:
        for chunk in cov._tokenize_sample(tokenizer, sample, concat_size, min_length, apply_chat_template):
            chunks.append(chunk)
    return chunks


def _prepare_embeddings(
    model,
    chunks: list[dict[str, Any]],
    hc_mult: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pre-compute CPU embeddings and input_ids for every chunk."""

    import torch

    input_ids: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    embed = model.model.embed_tokens
    for chunk in chunks:
        ids = torch.tensor([chunk["input_ids"]], dtype=torch.long)
        input_ids.append(ids)
        with torch.no_grad():
            inputs_embeds = embed(ids)
        h = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
        hidden_states.append(h)
    return input_ids, hidden_states


def _get_cached_mask_and_rope(
    model,
    config,
    length: int,
    cache: dict[int, tuple],
) -> tuple[Any, Any]:
    """Cache causal mask and rotary embeddings for a given sequence length."""

    import torch
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import create_sliding_window_causal_mask

    if length in cache:
        return cache[length]

    position_ids = torch.arange(length).unsqueeze(0)
    sample_embeds = model.model.embed_tokens(position_ids)  # [1, L, hidden]
    causal_mask = create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=sample_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = {
        "main": model.model.rotary_emb(sample_embeds, position_ids=position_ids, layer_type="main"),
        "compress": model.model.rotary_emb(sample_embeds, position_ids=position_ids, layer_type="compress"),
    }
    cache[length] = (causal_mask, position_embeddings)
    return causal_mask, position_embeddings


def _load_block_layers(
    model: Any,
    devices: list[torch.device],
    block_start: int,
    layers_per_device: int,
) -> list[list[int]]:
    """Move a contiguous slice of decoder layers onto the target GPUs and return the assignment."""

    num_layers = len(model.model.layers)
    num_devices = len(devices)
    block_map: list[list[int]] = []
    for d_idx in range(num_devices):
        lo = block_start + d_idx * layers_per_device
        hi = min(num_layers, lo + layers_per_device)
        layer_idxs = list(range(lo, hi))
        for li in layer_idxs:
            model.model.layers[li].to(devices[d_idx])
        block_map.append(layer_idxs)
    return block_map


def _offload_block_layers(model: Any, block_map: list[list[int]]) -> None:
    """Move the layers of the current block back to CPU."""

    for layer_idxs in block_map:
        for li in layer_idxs:
            model.model.layers[li].to("cpu")


def scan_layer_by_layer(
    model,
    profile: cov.DatasetProfile,
    input_ids: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    devices: list[torch.device],
    batch_chunks: int,
    layers_per_device: int = 1,
) -> None:
    """Run one dataset through the model, optionally keeping >1 contiguous layer per GPU."""

    import torch

    config = model.config
    num_layers = len(model.model.layers)
    cache: dict[int, tuple] = {}
    total_tokens = sum(ids.numel() for ids in input_ids)

    # Group chunks by sequence length so batched tensors are rectangular.
    by_length: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for chunk_idx, (ids, h) in enumerate(zip(input_ids, hidden_states)):
        by_length.setdefault(ids.shape[1], []).append((chunk_idx, ids, h))

    start = time.perf_counter()

    if layers_per_device == 1:
        # Original one-layer-at-a-time path (kept as fallback / single-GPU path).
        for layer_idx in range(num_layers):
            device = devices[layer_idx % len(devices)]
            layer = model.model.layers[layer_idx]
            layer.to(device)

            for length, group in by_length.items():
                causal_mask, position_embeddings = _get_cached_mask_and_rope(model, config, length, cache)
                cm_base = causal_mask.to(device)
                pos_emb_gpu = {
                    "main": (position_embeddings["main"][0].to(device), position_embeddings["main"][1].to(device)),
                    "compress": (position_embeddings["compress"][0].to(device), position_embeddings["compress"][1].to(device)),
                }

                for batch_start in range(0, len(group), batch_chunks):
                    batch = group[batch_start : batch_start + batch_chunks]
                    batch_size = len(batch)

                    cm_gpu = cm_base.expand(batch_size, -1, -1, -1)
                    batch_ids = torch.cat([ids for _, ids, _ in batch], dim=0).to(device)
                    batch_h = torch.cat([h for _, _, h in batch], dim=0).to(device)
                    batch_position_ids = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)

                    with torch.no_grad():
                        out = layer(
                            batch_h,
                            input_ids=batch_ids,
                            position_embeddings=pos_emb_gpu,
                            position_ids=batch_position_ids,
                            attention_mask=cm_gpu,
                            past_key_values=None,
                            use_cache=False,
                        )

                    split_sizes = [h.size(0) for _, _, h in batch]
                    for (chunk_idx, _, _), out_chunk in zip(batch, torch.split(out, split_sizes, dim=0)):
                        hidden_states[chunk_idx] = out_chunk.to("cpu", non_blocking=False)

                    del out, batch_h, batch_ids, batch_position_ids

            layer.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            if layer_idx == 0 or (layer_idx + 1) % 5 == 0 or layer_idx == num_layers - 1:
                elapsed = time.perf_counter() - start
                print(
                    f"[scan] layer {layer_idx + 1}/{num_layers} on {device} done; "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
    else:
        # Block path: keep ``layers_per_device`` contiguous layers on each GPU and
        # pass hidden states GPU-to-GPU inside each batch.  This avoids the
        # per-layer CPU<->GPU round trips and ``gc.collect``/``empty_cache`` calls.
        num_devices = len(devices)
        block_size = layers_per_device * num_devices
        num_blocks = (num_layers + block_size - 1) // block_size

        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_map = _load_block_layers(model, devices, block_start, layers_per_device)
            active_layers = [idxs for idxs in block_map if idxs]
            if not active_layers:
                continue
            first_active = next(i for i, idxs in enumerate(block_map) if idxs)

            print(
                f"[block] Block {block_idx + 1}/{num_blocks} loaded on {devices}: "
                f"{[[layer_i + 1 for layer_i in idxs] for idxs in block_map]}",
                flush=True,
            )

            # Free cached/reserved blocks that are not backing active tensors before
            # the forward pass, so the allocator has room for activations.
            torch.cuda.empty_cache()

            for length, group in by_length.items():
                causal_mask, position_embeddings = _get_cached_mask_and_rope(model, config, length, cache)

                # Move reusable mask and RoPE buffers to each GPU once per block/length.
                cm_base: dict[torch.device, torch.Tensor] = {}
                pos_emb: dict[torch.device, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
                for d_idx, device in enumerate(devices):
                    if block_map[d_idx]:
                        cm_base[device] = causal_mask.to(device)
                        pos_emb[device] = {
                            "main": (position_embeddings["main"][0].to(device), position_embeddings["main"][1].to(device)),
                            "compress": (position_embeddings["compress"][0].to(device), position_embeddings["compress"][1].to(device)),
                        }

                for batch_start in range(0, len(group), batch_chunks):
                    batch = group[batch_start : batch_start + batch_chunks]
                    batch_size = len(batch)
                    split_sizes = [h.size(0) for _, _, h in batch]

                    # Begin the block on the first active GPU.
                    device = devices[first_active]
                    batch_h = torch.cat([h for _, _, h in batch], dim=0).to(device)
                    batch_ids = torch.cat([ids for _, ids, _ in batch], dim=0).to(device)
                    batch_position_ids = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)

                    for d_idx in range(first_active, num_devices):
                        if not block_map[d_idx]:
                            continue
                        device = devices[d_idx]
                        if batch_h.device != device:
                            batch_h = batch_h.to(device)
                            batch_ids = batch_ids.to(device)
                            batch_position_ids = batch_position_ids.to(device)

                        cm_gpu = cm_base[device].expand(batch_size, -1, -1, -1)
                        pos_emb_gpu = pos_emb[device]

                        for li in block_map[d_idx]:
                            layer = model.model.layers[li]
                            with torch.no_grad():
                                batch_h = layer(
                                    batch_h,
                                    input_ids=batch_ids,
                                    position_embeddings=pos_emb_gpu,
                                    position_ids=batch_position_ids,
                                    attention_mask=cm_gpu,
                                    past_key_values=None,
                                    use_cache=False,
                                )

                    for (chunk_idx, _, _), out_chunk in zip(batch, torch.split(batch_h, split_sizes, dim=0)):
                        hidden_states[chunk_idx] = out_chunk.to("cpu", non_blocking=False)

                    del batch_h, batch_ids, batch_position_ids

            _offload_block_layers(model, block_map)
            torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.perf_counter() - start
            last_layer = max(max(idxs) for idxs in active_layers)
            print(
                f"[scan] block {block_idx + 1}/{num_blocks} (layers {block_start + 1}-{last_layer + 1}) done; "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    profile.total_tokens = int(total_tokens)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Defused BF16 checkpoint directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, action="append", help="Dataset spec (path or path:name)")
    parser.add_argument("--reference", required=True, help="Reference dataset spec")
    parser.add_argument("--physical-gpu", default="6,7", help="Physical nvidia-smi GPU indices, comma-separated")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--concat-size", type=int, default=1024)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--sketch-samples", type=int, default=256)
    parser.add_argument("--min-conditional-gain", type=float, default=0.0)
    parser.add_argument("--target-gain", type=float, default=None)
    parser.add_argument("--target-tokens", type=int, default=131072)
    parser.add_argument("--target-tokens-mode", choices=("gain", "gain_per_token"), default="gain_per_token")
    parser.add_argument("--target-moe-expert-tokens", type=int, default=None)
    parser.add_argument("--fallback-threshold", default="0.5%")
    parser.add_argument("--text-separator", default="===========")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--greedy-threads", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Max rows per dataset (0=full)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--batch-chunks", type=int, default=16, help="Number of independent chunks to batch per layer forward")
    parser.add_argument("--cuda-alloc-conf", default="expandable_segments:True,max_split_size_mb:512", help="PYTORCH_CUDA_ALLOC_CONF value")
    parser.add_argument(
        "--layers-per-gpu",
        type=int,
        default=None,
        help="Contiguous layers to keep on each GPU (default: auto from free memory)",
    )
    parser.add_argument("--fused", action="store_true", help="Use fused mlp.experts (no Defuser); needed for MoE expert coverage")
    parser.add_argument("--moe-expert-coverage", action="store_true", help="Track per-expert routed-token coverage for fused MoE modules")
    parser.add_argument("--moe-routing-bypass", action="store_true", help="Route every token to every expert (slow; diagnostic)")
    parser.add_argument("--moe-expert-min-tokens", type=int, default=16, help="Min routed tokens for an expert to count as covered")
    parser.add_argument("--moe-router-coverage-weight", type=float, default=1.0, help="Weight for uncovered expert mass in score")
    parser.add_argument("--moe-expert-diag", action="store_true", help="Accumulate per-expert Hessian diagonals (expensive)")
    args = parser.parse_args()

    devices = _preflight_gpus(_parse_physical_gpu(args.physical_gpu), args.allow_busy_gpu)
    if not devices:
        raise RuntimeError("At least one physical GPU must be specified")
    print(f"[gpu] Using devices: {devices}")

    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    import torch

    torch.set_num_threads(min(32, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)

    tokenizer = cov._load_tokenizer(str(args.model), trust_remote_code=args.trust_remote_code)
    model = _load_model(args.model, args.torch_dtype, defuse=not args.fused)

    layers_per_device = _estimate_max_layers_per_gpu(model, devices, args.layers_per_gpu)

    target_groups = cov.find_target_groups(model)
    module_count = sum(len(g.members) for g in target_groups)
    print(f"[model] Found {len(target_groups)} target groups ({module_count} modules)")

    context = cov.ScanContext()
    handles = cov.register_hooks(target_groups, context)

    moe_expert_targets: list[cov.MoEExpertTarget] = []
    moe_routers: list[tuple[str, Any]] = []
    if args.moe_expert_coverage or args.moe_routing_bypass:
        moe_expert_targets = cov.find_moe_expert_modules(model)
        moe_routers = cov.find_moe_routers(model)
        print(f"[moe] Found {len(moe_expert_targets)} fused expert modules and {len(moe_routers)} routers")
    if args.moe_routing_bypass:
        if moe_routers:
            cov.apply_moe_routing_bypass(moe_routers)
        else:
            print("[warn] --moe-routing-bypass set but no routers found")
    if args.moe_expert_coverage:
        if moe_expert_targets:
            handles.extend(cov.register_moe_hooks(moe_expert_targets, context))
            cov.MOE_ROUTER_COVERAGE_WEIGHT = args.moe_router_coverage_weight
            cov.MOE_EXPERT_MIN_TOKENS = args.moe_expert_min_tokens
            cov.MOE_EXPERT_DIAG_ENABLED = args.moe_expert_diag
            print(f"[moe] Expert coverage enabled; min_tokens={args.moe_expert_min_tokens}, weight={args.moe_router_coverage_weight}")
        else:
            print("[warn] --moe-expert-coverage set but no fused MoE expert modules found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and tokenize datasets.
    dataset_chunks: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    profile_paths: list[tuple[str, Path]] = []
    profile_names: set[str] = set()

    for spec in args.dataset:
        dataset_path, dataset_name = cov._parse_dataset_spec(spec)
        profile_name = dataset_name or Path(dataset_path).name
        base_name = profile_name
        counter = 1
        while profile_name in profile_names:
            profile_name = f"{base_name}_{counter}"
            counter += 1
        profile_names.add(profile_name)

        print(f"[data] Loading calibration `{profile_name}` from {dataset_path} ...")
        raw = cov._load_raw_samples(dataset_path, dataset_name, args.text_separator, output_dir)
        if args.max_samples > 0:
            raw = raw[: args.max_samples]
        chunks = _tokenize_chunks(tokenizer, raw, args.concat_size, args.min_length, args.apply_chat_template)
        print(f"[data]   {len(chunks)} chunks, {sum(len(c['input_ids']) for c in chunks)} tokens")
        input_ids, hidden = _prepare_embeddings(model, chunks, model.config.hc_mult)
        dataset_chunks[profile_name] = (input_ids, hidden)

    print(f"[data] Loading reference from {args.reference} ...")
    ref_path, ref_name = cov._parse_dataset_spec(args.reference)
    ref_profile_name = ref_name or f"ref:{Path(ref_path).name}"
    ref_raw = cov._load_raw_samples(ref_path, ref_name, args.text_separator, output_dir)
    if args.max_samples > 0:
        ref_raw = ref_raw[: args.max_samples]
    ref_chunks = _tokenize_chunks(tokenizer, ref_raw, args.concat_size, args.min_length, args.apply_chat_template)
    print(f"[data]   {len(ref_chunks)} chunks, {sum(len(c['input_ids']) for c in ref_chunks)} tokens")
    ref_input_ids, ref_hidden = _prepare_embeddings(model, ref_chunks, model.config.hc_mult)

    # Scan calibration datasets.
    scan_start = time.perf_counter()
    profiles: dict[str, cov.DatasetProfile] = {}
    for profile_name, (input_ids, hidden) in dataset_chunks.items():
        profile = cov.DatasetProfile.from_groups(profile_name, target_groups, args.sketch_samples)
        context.active_profile = profile
        try:
            scan_layer_by_layer(model, profile, input_ids, hidden, devices, args.batch_chunks, layers_per_device)
        finally:
            context.active_profile = None
        profile_path = output_dir / f"{profile_name}.profile.pt"
        cov._save_profile(profile, profile_path)
        profiles[profile_name] = profile
        profile_paths.append((profile_name, profile_path))
        print(f"[scan] {profile_name}: {profile.total_tokens} tokens", flush=True)

    # Scan reference.
    ref = cov.DatasetProfile.from_groups(ref_profile_name, target_groups, args.sketch_samples)
    context.active_profile = ref
    try:
        scan_layer_by_layer(model, ref, ref_input_ids, ref_hidden, devices, args.batch_chunks, layers_per_device)
    finally:
        context.active_profile = None
    print(f"[scan] reference: {ref.total_tokens} tokens", flush=True)
    scan_time = time.perf_counter() - scan_start

    for handle in handles:
        handle.remove()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Scoring.
    t = time.perf_counter()
    print("[score] Computing standalone scores ...")
    standalone_scores = {name: cov.score(p, ref) for name, p in profiles.items()}
    standalone_time = time.perf_counter() - t

    t = time.perf_counter()
    print("[score] Running greedy complementarity selection ...")
    selected, greedy_order, final_score, cumulative_gain, warnings = cov.greedy_select(
        profiles,
        ref,
        args.min_conditional_gain,
        args.sketch_samples,
        args.greedy_threads,
        target_gain=args.target_gain,
        target_tokens=args.target_tokens,
        target_tokens_mode=args.target_tokens_mode,
        target_moe_expert_tokens=args.target_moe_expert_tokens,
    )
    for w in warnings:
        print(f"[warn] {w}")
    greedy_time = time.perf_counter() - t

    t = time.perf_counter()
    fallback_by_dataset = {
        name: cov.find_fallback_modules(p, args.fallback_threshold) for name, p in profiles.items()
    }
    fallback_selected = cov.find_fallback_modules(selected, args.fallback_threshold)
    fallback_time = time.perf_counter() - t

    t = time.perf_counter()
    greedy_gains = {item["name"]: float(item["conditional_gain"]) for item in greedy_order}
    selected_order = {name: idx for idx, name in enumerate(greedy_gains)}
    complementarity = []
    for name, p in profiles.items():
        if name in selected_order:
            gain = greedy_gains[name]
            verdict = "selected"
        else:
            merged = selected.merge(p)
            s = cov.score(merged, ref)
            gain = final_score - s
            verdict = "complementary" if gain > 0 else "redundant"
        complementarity.append({"name": name, "conditional_gain": gain, "verdict": verdict})

    def _sort_complementarity(x: dict[str, Any]):
        if x["name"] in selected_order:
            return (0, selected_order[x["name"]], str(x["name"]))
        return (1, -float(x["conditional_gain"]), str(x["name"]))

    complementarity.sort(key=_sort_complementarity)
    comp_time = time.perf_counter() - t

    report_config = {
        "model": str(args.model),
        "datasets": args.dataset,
        "reference": args.reference,
        "max_samples": args.max_samples,
        "concat_size": args.concat_size,
        "min_length": args.min_length,
        "sketch_samples": args.sketch_samples,
        "min_conditional_gain": args.min_conditional_gain,
        "fallback_threshold": args.fallback_threshold,
        "torch_dtype": args.torch_dtype,
        "device": [str(d) for d in devices],
        "apply_chat_template": args.apply_chat_template,
        "target_gain": args.target_gain,
        "target_tokens": args.target_tokens,
        "target_tokens_mode": args.target_tokens_mode,
        "greedy_threads": args.greedy_threads,
        "target_moe_expert_tokens": args.target_moe_expert_tokens,
        "moe_expert_coverage": args.moe_expert_coverage,
        "moe_routing_bypass": args.moe_routing_bypass,
        "moe_router_coverage_weight": args.moe_router_coverage_weight,
        "moe_expert_min_tokens": args.moe_expert_min_tokens,
        "moe_expert_diag": args.moe_expert_diag,
        "fused": args.fused,
        "layers_per_gpu": layers_per_device,
    }
    selected_mix = {
        "datasets": [item["name"] for item in greedy_order],
        "score_start": round(float(final_score + cumulative_gain), 6) if greedy_order else None,
        "score": float(final_score),
        "cumulative_gain": float(cumulative_gain),
        "total_tokens": int(selected.total_tokens),
        "fallback_count": len(fallback_selected),
    }
    report = cov._build_report(
        report_config,
        ref,
        profiles,
        standalone_scores,
        selected,
        greedy_order,
        final_score,
        cumulative_gain,
        fallback_by_dataset,
        fallback_selected,
        complementarity,
        {
            "scan": scan_time,
            "standalone_scores": standalone_time,
            "greedy": greedy_time,
            "fallback": fallback_time,
            "complementarity": comp_time,
            "total": scan_time + standalone_time + greedy_time + fallback_time + comp_time,
        },
        warnings,
    )

    if args.moe_expert_coverage and moe_expert_targets:
        ref_active = {
            name: int((counts > 0).sum())
            for name, counts in (getattr(ref, "expert_counts", {}) or {}).items()
        }
        report["moe"] = {
            "expert_modules": len(moe_expert_targets),
            "num_experts": {t.name: t.num_experts for t in moe_expert_targets},
            "routing_bypass": args.moe_routing_bypass,
            "router_coverage_weight": args.moe_router_coverage_weight,
            "expert_min_tokens": args.moe_expert_min_tokens,
            "target_moe_expert_tokens": args.target_moe_expert_tokens,
            "reference_active_experts": ref_active,
            "per_dataset_uncovered_routed_mass": {
                name: cov.moe_uncovered_mass(p, ref) for name, p in profiles.items()
            },
            "per_dataset_min_expert_tokens": {
                name: cov.moe_min_expert_tokens(p, ref) for name, p in profiles.items()
            },
            "selected_uncovered_routed_mass": cov.moe_uncovered_mass(selected, ref),
            "selected_min_expert_tokens": cov.moe_min_expert_tokens(selected, ref),
        }

    with open(output_dir / "coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (output_dir / "coverage_report.md").write_text(report["markdown"], encoding="utf-8")

    # Save selected mix information (placeholder: full implementation can materialize the dataset).
    (output_dir / "selected_mix.json").write_text(json.dumps(selected_mix, indent=2), encoding="utf-8")

    print(f"[done] Report written to {output_dir / 'coverage_report.json'}")
    print(f"[done] Selected mix: {selected_mix['datasets']}")
    print(f"[done] Total tokens: {selected_mix['total_tokens']}, final score: {selected_mix['score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
