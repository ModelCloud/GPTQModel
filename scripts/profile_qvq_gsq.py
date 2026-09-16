#!/usr/bin/env python3
"""Profile QVQ-GSQ on one real serialized Llama projection."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.qvq import (
    repack_p32_planar_to_window,
    rht_preprocess_weight,
)
from gptqmodel.quantization.qvq_gsq import refine_trellis_fisher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="model.layers.0.mlp.down_proj")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gumbel-samples", type=int, default=4)
    parser.add_argument("--hard-eval-interval", type=int, default=10)
    parser.add_argument("--coordinate-sweeps", type=int, default=0,
                        help="hard coordinate sweeps before the relaxation; use 0 to isolate GSQ")
    parser.add_argument("--soft-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat the complete projection fit in one process")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    qdir = Path("/root/qvq-results/w3-p32-gsq-ab-20260915/gsq")
    index = json.loads((qdir / "model.safetensors.index.json").read_text())["weight_map"]

    def quantized(suffix):
        key = args.module + "." + suffix
        with safe_open(str(qdir / index[key]), framework="pt", device="cuda:0") as handle:
            return handle.get_tensor(key)

    with safe_open("/monster/data/model/Llama-3.2-1B-Instruct/model.safetensors",
                   framework="pt", device="cuda:0") as handle:
        weight = handle.get_tensor(args.module + ".weight").float()
    trellis, su, sv, banks, alt = [
        quantized(name) for name in ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")]
    baseline = repack_p32_planar_to_window(trellis, bits=3)
    target = rht_preprocess_weight(weight, su.reciprocal(), sv.reciprocal()).float()
    input_hessian = torch.eye(target.shape[0], device="cuda")
    output_hessian = torch.eye(target.shape[1], device="cuda")
    config = GSQConfig(
        enabled=True, steps=args.steps, candidates=33, seed=7,
        max_candidate_bytes=4 * 1024**3,
        qvq_gumbel_samples=args.gumbel_samples, qvq_coordinate_sweeps=args.coordinate_sweeps,
        qvq_hard_eval_interval=args.hard_eval_interval, qvq_soft_dtype=args.soft_dtype)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    repeat_seconds = []
    result = None
    for repeat in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"profile_qvq_gsq.repeat_{repeat}")
        started = time.perf_counter()
        result = refine_trellis_fisher(
            baseline, target=target, input_hessian=input_hessian, output_hessian=output_hessian,
            config=config, bits=3, bank_ids=banks, bank_alt_id=alt, layout="p32_window")
        torch.cuda.synchronize()
        repeat_seconds.append(time.perf_counter() - started)
        torch.cuda.nvtx.range_pop()
    payload = {
        "module": args.module, "shape": list(target.shape), "steps": args.steps,
        "gumbel_samples": args.gumbel_samples, "coordinate_sweeps": args.coordinate_sweeps,
        "seconds": repeat_seconds[-1],
        "repeat_seconds": repeat_seconds,
        "before": result.calibration_before, "after": result.calibration_after,
        "changed_tiles": int((result.choices != 0).sum()), "diagnostics": result.diagnostics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
