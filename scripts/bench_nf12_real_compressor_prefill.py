# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint BF16 NF12 compressor prefill gate on one visible GPU.

Run one process per physical GPU with CUDA_VISIBLE_DEVICES set to that GPU.
The packed weight is the only retained inference weight; dense BF16 is an
independent benchmark oracle. CUDA graphs measure the full current NF12 call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import gptqmodel_nf12 as nf12


SHARD = Path("/fast/model/DeepSeek-V4.1-Flash-1017/model-00023-of-00048.safetensors")
KEY = "layers.20.attn.compressor.wkv.weight"


def capture(fn, repeat: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            fn()
    return graph


def replay_us(graph: torch.cuda.CUDAGraph, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--target", type=float, default=1.05)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.rounds < 1 or args.target <= 0:
        parser.error("repeat, rounds, and target must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        parser.error("set CUDA_VISIBLE_DEVICES to exactly --physical-gpu")
    if sys._is_gil_enabled():
        parser.error("use the PyTorch 2.15 free-threaded runtime with PYTHON_GIL=0")
    device = torch.device(args.device)
    if device != torch.device("cuda:0") or torch.cuda.device_count() != 1:
        parser.error("expose one physical GPU as cuda:0")
    if torch.cuda.get_device_capability(device)[0] < 8:
        parser.error("SM80 or newer is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    with safe_open(SHARD, framework="pt", device="cpu") as checkpoint:
        source_weight = checkpoint.get_tensor(KEY)
    if source_weight.dtype != torch.bfloat16 or tuple(source_weight.shape) != (512, 5120):
        raise AssertionError("unexpected checkpoint compressor weight")
    weight_sha256 = hashlib.sha256(source_weight.view(torch.uint8).numpy().tobytes()).hexdigest()
    weight = source_weight.to(device)
    generator = torch.Generator(device=device).manual_seed(18201)
    x = (torch.randn((512, 5120), device=device, generator=generator) * 0.05).bfloat16()
    packed = nf12.encode(weight)
    restored = nf12.decode(packed)
    if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
        raise AssertionError("NF12 did not restore all BF16 weight bits")
    if set(vars(packed)) != {"packed", "escapes", "exp_base", "shape"}:
        raise AssertionError("NF12Weight retained an unexpected tensor")

    def dense_fn():
        return F.linear(x, weight)

    def nf12_fn():
        return nf12.linear(x, packed)
    expected = dense_fn()
    actual = nf12_fn()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    bit_mismatches = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
    topk_mismatch_rows = int((actual.topk(8, dim=-1).indices !=
                              expected.topk(8, dim=-1).indices).any(dim=-1).sum())
    dense_graph = capture(dense_fn, args.repeat)
    nf12_graph = capture(nf12_fn, args.repeat)
    torch.cuda.synchronize()
    for _ in range(4):
        dense_graph.replay()
        nf12_graph.replay()
    torch.cuda.synchronize()
    dense_samples, nf12_samples = [], []
    for _ in range(args.rounds):
        dense_samples.append(replay_us(dense_graph, args.repeat))
        nf12_samples.append(replay_us(nf12_graph, args.repeat))
        nf12_samples.append(replay_us(nf12_graph, args.repeat))
        dense_samples.append(replay_us(dense_graph, args.repeat))
    dense_us = statistics.median(dense_samples)
    nf12_us = statistics.median(nf12_samples)
    result = {
        "gpu": args.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "gil_enabled": sys._is_gil_enabled(),
        "checkpoint_shard": str(SHARD),
        "weight_key": KEY,
        "weight_sha256": weight_sha256,
        "shape_nkm": [512, 5120, 512],
        "activation_seed": 18201,
        "math_policy": "TF32 off; highest FP32 matmul precision",
        "method": "CUDA graph; ABBA paired order",
        "repeat": args.repeat,
        "rounds": args.rounds,
        "target": args.target,
        "dense_bf16_us": round(dense_us, 3),
        "nf12_bf16_us": round(nf12_us, 3),
        "speedup": round(dense_us / nf12_us, 4),
        "passes_speed": dense_us >= args.target * nf12_us,
        "passes_decisions": topk_mismatch_rows == 0,
        "weight_bit_mismatches": 0,
        "output_bit_mismatches": bit_mismatches,
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        "topk_mismatch_rows": topk_mismatch_rows,
        "dense_weight_bytes": weight.nbytes,
        "packed_weight_bytes": packed.nbytes,
        "call_local_decode_bytes": weight.nbytes,
    }
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("gpu", "dense_bf16_us", "nf12_bf16_us",
                                              "speedup", "target", "passes_speed",
                                              "output_bit_mismatches", "topk_mismatch_rows")}))
    return int(args.strict and not (result["passes_speed"] and result["passes_decisions"]))


if __name__ == "__main__":
    raise SystemExit(main())
