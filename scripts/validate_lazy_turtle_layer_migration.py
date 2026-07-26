#!/usr/bin/env python3
"""Validate LazyTurtle parallel grouped loading across layer-to-layer expert swaps on OLMoE."""

import json
import os
import sys
import time

try:
    _gil_disabled = not sys._is_gil_enabled()
except AttributeError:
    _gil_disabled = False

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.model import move_to


def main():
    model_dir = os.environ.get("MODEL_DIR", "/monster/data/model/OLMoE-1B-7B-0924")
    device = torch.device("cpu")

    print(f"Loading OLMoE model from {model_dir} ...")
    model = GPTQModel.from_pretrained(
        model_dir,
        quantize_config=QuantizeConfig(bits=4, group_size=128, offload_to_disk=True),
    )

    with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    layers = model.model.model.layers
    layer_limit = int(os.environ.get("LAYER_LIMIT", 0)) or len(layers)
    total_layers = min(layer_limit, len(layers))
    total_elapsed = 0.0
    total_errors = 0

    print(f"Validating {total_layers} MoE layers (GIL disabled={_gil_disabled})")

    for layer_idx, layer in enumerate(layers[:total_layers]):
        experts = layer.mlp.experts
        module_path = f"model.layers.{layer_idx}.mlp.experts"

        start = time.perf_counter()
        model.shell_module_materialize(experts, device=device)
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed

        # Validate every expert weight against the original checkpoint in one shard open.
        params_by_name = {f"{module_path}.{name}": param for name, param in experts.named_parameters()}
        layer_keys = set(params_by_name)
        layer_shards = {weight_map[name] for name in layer_keys}
        for shard in layer_shards:
            shard_keys = {name for name in layer_keys if weight_map[name] == shard}
            with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
                for full_name in sorted(shard_keys):
                    param = params_by_name[full_name]
                    expected = f.get_tensor(full_name)
                    if param.shape != expected.shape:
                        print(f"  Layer {layer_idx} shape mismatch {full_name}: {param.shape} vs {expected.shape}")
                        total_errors += 1
                        continue
                    if not torch.equal(param, expected):
                        max_diff = (param - expected).abs().max().item()
                        print(f"  Layer {layer_idx} value mismatch {full_name}: max_diff={max_diff}")
                        total_errors += 1

        print(f"Layer {layer_idx:2d} materialized in {elapsed:.3f}s")

        # Simulate migration: move this expert group back to meta so the next swap reloads it.
        move_to(experts, device=torch.device("meta"))

    print(f"\nTotal migration time: {total_elapsed:.3f}s for {total_layers} layers")
    if total_errors:
        print(f"FAILED with {total_errors} mismatches")
        raise SystemExit(1)
    print("All layers validated successfully.")


if __name__ == "__main__":
    main()
