#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Audit the dispatched YAQA collector on synthetic operator inputs with NCU.

This isolates collection scheduling and validation; it is not model-quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_idle_preflight import (
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=5120)
    parser.add_argument("--out-features", type=int, default=17408)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--collector-source", type=Path)
    add_gpu_idle_preflight_args(parser)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.in_features, args.out_features, args.rank) < 1:
        raise ValueError("all profiling dimensions must be positive")
    preflight = bootstrap_gpu_idle_preflight()
    if preflight is None:
        raise RuntimeError("collection profiling requires the GPU idle preflight")

    import torch

    from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b
    from scripts.benchmark_qvq_yaqa_qwen38 import (
        _git_revision,
        _hardware,
        _load_collector_source,
    )

    if args.collector_source is not None:
        capture_yaqa_sketch_b = _load_collector_source(args.collector_source, "profile").capture_yaqa_sketch_b
    device = torch.device("cuda:0")
    torch.manual_seed(20260906)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(args.in_features, args.out_features, bias=False, device=device)
            self.register_buffer("activation", torch.randn(
                args.batch_size, args.sequence_length, args.in_features, device=device,
            ))

        def forward(self, input_ids, attention_mask, use_cache=False):
            del input_ids, attention_mask, use_cache
            return SimpleNamespace(logits=self.proj(self.activation))

    model = Probe().eval()
    batches = [{
        "input_ids": torch.zeros((args.batch_size, args.sequence_length), dtype=torch.long),
        "attention_mask": torch.ones((args.batch_size, args.sequence_length), dtype=torch.long),
    }]

    def run():
        return capture_yaqa_sketch_b(
            model, batches, {"proj": model.proj}, device=device, seed=20260906,
            first_decoder_layer=model.proj, accumulator_device=device,
            gram_strategy="streaming_projected", gram_projection_rank=args.rank,
        )

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    recheck_gpu_exclusivity(preflight)
    torch.cuda.cudart().cudaProfilerStart()
    result = run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({
        "scope": "synthetic operator instruction audit; not model-quality evidence",
        "revision": _git_revision(),
        "projection_kernel_source_sha256": hashlib.sha256(
            (REPO_ROOT / "gptqmodel/quantization/qvq_yaqa_cuda.py").read_bytes()
        ).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(inspect.getfile(capture_yaqa_sketch_b)).read_bytes()).hexdigest(),
        "hardware": _hardware(torch, preflight),
        "shape": {key: getattr(args, key) for key in (
            "batch_size", "sequence_length", "in_features", "out_features", "rank",
        )},
        "capture": result[2],
    }), flush=True)


if __name__ == "__main__":
    main()
