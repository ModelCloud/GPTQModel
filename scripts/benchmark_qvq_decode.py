#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark QVQ W4 decode (eager path vs CUDA-graph replay) on one checkpoint.

Two model instances are loaded so the CUDA-graph replay and the eager forward
are compared at an identical cache state. The eager attention mask is made
capture-safe (no CPU->CUDA scalar tensor during capture).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_qvq_decode.py \
        --checkpoint /monster/data/model/Llama-3.2-1B-Instruct-QVQ-W4-block-cal512-3b2052a6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.backend import BACKEND  # noqa: E402


def _make_eager_mask_capture_safe() -> None:
    """Replace the eager attention mask's CPU->CUDA scalar tensor with a Python
    scalar so the decode step can be captured into a CUDA graph."""

    import transformers.masking_utils as masking_utils

    def eager_mask_capture_safe(**kwargs):
        kwargs.pop("allow_is_causal_skip", None)
        mask = masking_utils.sdpa_mask(
            batch_size=kwargs["batch_size"],
            q_length=kwargs["q_length"],
            kv_length=kwargs["kv_length"],
            q_offset=kwargs.get("q_offset", 0),
            kv_offset=kwargs.get("kv_offset", 0),
            mask_function=kwargs.get("mask_function", masking_utils.causal_mask_function),
            attention_mask=kwargs.get("attention_mask"),
            allow_is_causal_skip=False,
            allow_is_bidirectional_skip=kwargs.get("allow_is_bidirectional_skip", False),
            use_vmap=kwargs.get("use_vmap", False),
            device=kwargs["device"],
        )
        if mask is not None:
            mask = torch.where(mask, 0.0, torch.finfo(kwargs["dtype"]).min)
        return mask

    masking_utils.eager_mask = eager_mask_capture_safe
    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask_capture_safe


def _patch_rmsnorm() -> None:
    """Use torch's fused RMSNorm kernel for fp16 CUDA Llama norms.

    The transformers LlamaRMSNorm forward chains 7 kernels per norm
    (fp32 copy, pow, mean, add, rsqrt, two multiplies, fp16 copy). The fused
    op is bitwise identical on the W4 model path (max abs diff 0.0 on
    [1, 1, 2048] fp16 inputs) and ~4x faster; it is also a single capture-safe
    graph node."""

    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    original_forward = LlamaRMSNorm.forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.float16
            and hidden_states.shape[-1] == self.weight.shape[0]
        ):
            return F.rms_norm(hidden_states, self.weight.shape, self.weight, self.variance_epsilon)
        return original_forward(self, hidden_states)

    LlamaRMSNorm.forward = forward


def _load(checkpoint: str) -> GPTQModel:
    model = GPTQModel.load(
        checkpoint,
        device="cuda:0",
        dtype="float16",
        backend=BACKEND.QVQ,
        attn_implementation="eager",
    )
    return model.to("cuda:0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--graph-replays", type=int, default=60)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _make_eager_mask_capture_safe()
    _patch_rmsnorm()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    model_eager = _load(str(args.checkpoint))
    model_graph = _load(str(args.checkpoint))

    # Identical warmup on both models so the graph capture and the eager call
    # below compute the same decode step from the same cache state.
    ids = torch.randint(0, 32000, (1, 1)).cuda()
    with torch.inference_mode():
        for _ in range(5):
            model_eager(ids)
            model_graph(ids)
    torch.cuda.synchronize()

    # ---- CUDA graph capture of one decode step ----
    graph = torch.cuda.CUDAGraph()
    static_ids = ids.clone()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        with torch.inference_mode():
            model_graph(static_ids)
    torch.cuda.current_stream().wait_stream(side_stream)
    with torch.cuda.graph(graph):
        with torch.inference_mode():
            graph_logits = model_graph(static_ids).logits
    torch.cuda.synchronize()

    # Equality: eager model (same cache state) vs graph replay.
    with torch.inference_mode():
        eager_logits = model_eager(ids).logits
        graph.replay()
    torch.cuda.synchronize()
    logits_equal = torch.equal(graph_logits.view(torch.int16), eager_logits.view(torch.int16))

    # ---- eager (no graph) decode timing ----
    eager_times: list[float] = []
    with torch.inference_mode():
        for _ in range(args.steps):
            start = time.perf_counter()
            model_eager(ids)
            torch.cuda.synchronize()
            eager_times.append(time.perf_counter() - start)

    # ---- graph decode timing ----
    graph_times: list[float] = []
    for _ in range(args.graph_replays):
        start = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        graph_times.append(time.perf_counter() - start)
    graph_times.sort()
    eager_times.sort()

    result = {
        "checkpoint": str(args.checkpoint),
        "eager_decode_min_ms": eager_times[0] * 1000,
        "eager_decode_median_ms": statistics.median(eager_times) * 1000,
        "graph_decode_min_ms": graph_times[0] * 1000,
        "graph_decode_median_ms": statistics.median(graph_times) * 1000,
        "graph_vs_eager_speedup": (statistics.median(eager_times) * 1000) / (statistics.median(graph_times) * 1000),
        "graph_logits_bitwise_equal_eager": bool(logits_equal),
    }
    print(
        "eager decode/step: min %.1f ms median %.1f ms\n"
        "graph decode/step: min %.1f ms median %.1f ms\n"
        "graph vs eager: %.1fx   logits bitwise equal: %s"
        % (
            result["eager_decode_min_ms"],
            result["eager_decode_median_ms"],
            result["graph_decode_min_ms"],
            result["graph_decode_median_ms"],
            result["graph_vs_eager_speedup"],
            logits_equal,
        )
    )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
