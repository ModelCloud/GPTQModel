#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Time Inkling 2 Mini's FP32 router and check TTNN expert selection."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from gptqmodel.models.ports.inkling2_loop import Inkling2LoopConfig
from gptqmodel.models.ports.inkling2_loop.experts import _route
from gptqmodel.models.ports.inkling2_loop.modeling import Inkling2LoopTopKRouter
from gptqmodel.utils.ttnn_runtime import import_ttnn


def _measure(op, repeats, sync=lambda: None):
    op()
    sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        op()
        sync()
        samples.append((time.perf_counter_ns() - start) / 1_000)
    return [statistics.median(samples), sorted(samples)[math.ceil(0.95 * repeats) - 1]]


def _checkpoint_router(path, layer):
    config = Inkling2LoopConfig.from_pretrained(path, local_files_only=True).text_config
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    weights = []
    for suffix in ("weight", "bias"):
        name = f"model.llm.layers.{layer}.mlp.gate.{suffix}"
        with safe_open(path / index[name], framework="pt", device="cpu") as shard:
            weights.append(shard.get_tensor(name))
    return config, *weights


def _reference(x, w, bias, k, scale):
    logits = torch.nn.functional.linear(x.float(), w.float())
    scores = torch.sigmoid(logits)
    adjusted = scores + bias
    ids = torch.topk(adjusted, k, dim=-1, sorted=False).indices
    raw = scores.gather(-1, ids)
    return logits, adjusted, ids, raw / (raw.sum(-1, keepdim=True) + 1e-20) * scale


def _compare_routes(actual, expected, weights=None, expected_weights=None):
    actual, expected = actual.long(), expected.long()
    sorted_actual, actual_order = actual.sort(-1)
    sorted_expected, expected_order = expected.sort(-1)
    matches = (sorted_actual == sorted_expected).all(-1)
    output = {
        "rows": len(expected),
        "rows_with_same_ids": int(matches.sum()),
        "rows_with_same_id_order": int((actual == expected).all(-1).sum()),
    }
    if weights is not None and matches.any():
        actual_aligned = weights.gather(-1, actual_order)[matches]
        expected_aligned = expected_weights.gather(-1, expected_order)[matches]
        output["weight_error_max"] = float((actual_aligned - expected_aligned).abs().max())
    return output


def _dense_dispatch(ids, experts):
    mask = torch.nn.functional.one_hot(ids, num_classes=experts).permute(2, 1, 0)
    groups = []
    for expert in range(experts):
        slots, tokens = torch.where(mask[expert])
        if tokens.numel():
            groups.append((expert, tokens, slots))
    return tuple(groups)


def _measure_router(ttnn, device, path, layer, sizes, repeats):
    config, weight, bias = _checkpoint_router(path, layer)
    router = Inkling2LoopTopKRouter(config).eval()
    router.weight.copy_(weight.float())
    router.e_score_correction_bias.copy_(bias)
    k, scale, experts = router.top_k, router.routed_scaling_factor, router.num_experts

    def sync():
        ttnn.synchronize_device(device)

    def upload(value, dtype=ttnn.float32):
        return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    compute = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=True)
    gate_weight = upload(weight.T.contiguous().float())
    correction = upload(bias.reshape(1, -1))

    def linear(x):
        return ttnn.matmul(x, gate_weight, dtype=ttnn.float32, compute_kernel_config=compute)

    def choose(scores):
        return ttnn.topk(scores, k, dim=-1, sorted=False)

    def route_on_device(x):
        logits = linear(x)
        scores = ttnn.sigmoid(logits)
        adjusted = ttnn.add(scores, correction)
        _, ids = choose(adjusted)
        raw = ttnn.gather(scores, -1, ids)
        total = ttnn.add(ttnn.sum(raw, dim=-1, keepdim=True), 1e-20)
        return logits, ids, ttnn.mul(ttnn.div(raw, total), scale)

    print(json.dumps({"layer": layer, "weight": list(weight.shape), "checkpoint_weight_dtype": str(weight.dtype),
                      "correction_dtype": str(bias.dtype), "top_k": k, "scale": scale,
                      "device": "Blackhole", "cpu_threads": torch.get_num_threads(), "repeats": repeats}), flush=True)
    for m in sizes:
        x = torch.randn((m, router.hidden_dim), generator=torch.Generator().manual_seed(37 + layer)).bfloat16()
        logits, adjusted, expected_ids, expected_weights = _reference(x, weight, bias, k, scale)
        actual_logits, actual_weights, actual_ids = router(x)
        torch.testing.assert_close(actual_logits, logits, atol=0, rtol=0)
        torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)

        device_x, device_scores = upload(x.float()), upload(adjusted)

        def mixed():
            return tuple(ttnn.to_torch(value) for value in route_on_device(upload(x.float())))

        tcpu = {
            "gate": _measure(lambda: torch.nn.functional.linear(x.float(), router.weight), repeats),
            "topk": _measure(lambda: torch.topk(adjusted, k, dim=-1, sorted=False), repeats),
            "gate_through_weights": _measure(lambda: router(x), repeats),
            "dense_expert_dispatch": _measure(lambda: _dense_dispatch(expected_ids, experts), repeats),
            "grouped_expert_dispatch": _measure(lambda: tuple(_route(x, expected_ids, expected_weights, experts)), repeats),
        }
        direct = tuple(_route(x, expected_ids, expected_weights, experts))
        dense = _dense_dispatch(expected_ids, experts)
        assert [group[0] for group in direct] == [group[0] for group in dense]
        for (_, tokens, slots), (_, ref_tokens, ref_slots) in zip(direct, dense, strict=True):
            torch.testing.assert_close(tokens, ref_tokens, atol=0, rtol=0)
            torch.testing.assert_close(slots, ref_slots, atol=0, rtol=0)
        tdevice = {
            "gate": _measure(lambda: linear(device_x), repeats, sync),
            "topk": _measure(lambda: choose(device_scores), repeats, sync),
            "gate_through_weights": _measure(lambda: route_on_device(device_x), repeats, sync),
            "mixed_gate_through_weights": _measure(mixed, repeats),
        }
        _, device_only_ids = choose(device_scores)
        device_logits, device_ids, device_weights = mixed()
        print(json.dumps({"m": m, "cpu_us_median_p95": tcpu, "device_us_median_p95": tdevice,
                          "topk_with_cpu_scores": _compare_routes(ttnn.to_torch(device_only_ids), expected_ids),
                          "full_device_route": _compare_routes(device_ids, expected_ids, device_weights, expected_weights),
                          "logit_error_max": float((device_logits - logits).abs().max())}), flush=True)

    tied_scores = torch.zeros((2, experts), dtype=torch.float32)
    tied_expected = torch.topk(tied_scores, k, sorted=False).indices
    _, device_tied = choose(upload(tied_scores))
    print(json.dumps({"equal_score_cutoff": _compare_routes(ttnn.to_torch(device_tied), tied_expected)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chip", required=True, type=int)
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 12, 23))
    parser.add_argument("--m", type=int, nargs="+", default=(1, 4, 32, 128))
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.m) < 1 or args.repeats < 1 or args.cpu_threads < 1:
        parser.error("M, repeats and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    ttnn = import_ttnn()
    device = ttnn.open_device(device_id=args.chip)
    try:
        if device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError("Router benchmark requires Blackhole")
        with torch.inference_mode():
            for layer in args.layers:
                _measure_router(ttnn, device, args.checkpoint, layer, args.m, args.repeats)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
