#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""One-shot Qwen3-8B GPTQ quantization driven by the SLQ weight analyzer.

Runs on a single requested physical GPU (default 5). The script:
1. Waits for the target GPU to be idle.
2. Loads the dense Qwen3-8B checkpoint from /monster/data/model/Qwen3-8B.
3. Uses gptqmodel.quantization.slq to compute per-layer linear sensitivity and
   allocate non-uniform bitwidths around a 3-bit average budget.
4. Quantizes with 3-bit/group32, activation scale search, GAR enabled,
   desc_act=False, and the SLQ dynamic bitwidth map.
5. Saves the quantized model and runs a short generation smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure GPU testing skill defaults are respected before any torch CUDA init.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gptqmodel import GPTQModel
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.slq import build_dynamic_bits, linear_sensitivity, allocate_bitwidth_ilp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/monster/data/model/Qwen3-8B")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--budget", type=float, default=3.2, help="Average bits target for SLQ allocation.")
    parser.add_argument("--candidate-bits", default="2,3,4", help="Comma-separated candidate bitwidths.")
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--act-group-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-search", default="activation")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-concat-size", type=int, default=2048)
    parser.add_argument("--sample-size", type=int, default=500_000, help="Per-weight subsample for SLQ analysis.")
    parser.add_argument("--output", default="/tmp/qwen3_8b_slq_gptq")
    return parser.parse_args()


def _idle_gate(physical_gpu: int, samples: int = 3, interval: float = 2.0, memory_slack_mb: int = 256) -> dict:
    """Return physical GPU metadata once the device is idle for N samples."""

    accepted = None
    consecutive = 0
    while consecutive < samples:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx, pci, uuid, name, mem_used, util = parts
            if int(idx) != physical_gpu:
                continue
            util = int(util)
            mem_used = int(mem_used)
            print(f"[preflight] GPU {idx} ({name}) util={util}% mem_used={mem_used}MiB")
            if util == 0 and mem_used <= memory_slack_mb:
                if accepted is None:
                    accepted = {
                        "physical_id": int(idx),
                        "pci": pci,
                        "uuid": uuid,
                        "name": name,
                        "memory_used_mb": mem_used,
                    }
                consecutive += 1
            else:
                consecutive = 0
                accepted = None
            break
        else:
            raise RuntimeError(f"physical GPU {physical_gpu} not found in nvidia-smi output")
        if consecutive < samples:
            time.sleep(interval)
    print(f"[preflight] Accepted GPU {accepted['physical_id']} / {accepted['uuid']} after {samples} idle samples")
    return accepted


def _load_calibration_texts(rows: int = 128) -> list[str]:
    parquet_path = Path("/monster/data/model/dataset/nm-calibration/llm.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Calibration parquet not found at {parquet_path}")
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()
    if rows > 0:
        texts = texts[:rows]
    return texts


def _extract_quantizable_weights(model) -> tuple[list[str], list[torch.Tensor]]:
    """Return (names, weights) for torch.nn.Linear modules that GPTQ would quantize."""

    names = []
    weights = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, torch.nn.Linear):
            continue
        # Match typical GPTQ quantizable modules; skip embeddings/lm_head if they appear as Linear.
        lowered = name.lower()
        if any(token in lowered for token in ("embed", "lm_head", "output_layer")):
            continue
        names.append(name)
        weights.append(module.weight.detach())
    return names, weights


def _slq_dynamic_allocation(
    names: list[str],
    weights: list[torch.Tensor],
    candidate_bits: list[int],
    budget: float,
    symmetric: bool,
    sample_size: int,
) -> tuple[dict, np.ndarray]:
    """Build a QuantizeConfig.dynamic dict and the raw assignment from per-weight SLQ sensitivity."""

    # Subsample large weights to keep the analysis fast and GPU-memory friendly.
    sampled_weights = []
    for w in weights:
        flat = w.reshape(-1)
        if flat.numel() > sample_size:
            # Deterministic-ish random sample (no torch RNG state side effects).
            perm = torch.randperm(flat.numel(), device=flat.device)[:sample_size]
            sampled = flat[perm]
        else:
            sampled = flat
        sampled_weights.append(sampled)

    costs = linear_sensitivity(sampled_weights, candidate_bits, symmetric=symmetric)
    assignment = allocate_bitwidth_ilp(costs, candidate_bits, budget=budget)
    dynamic = build_dynamic_bits(names, candidate_bits, assignment)
    return dynamic, assignment


def main() -> None:
    args = _parse_args()

    gpu_info = _idle_gate(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Verify the process sees the expected physical GPU as cuda:0.
    torch.cuda.init()
    visible_uuid = torch.cuda.get_device_properties(0).uuid
    gpu_uuid = gpu_info["uuid"]
    if gpu_uuid.startswith("GPU-"):
        gpu_uuid = gpu_uuid[4:]
    visible_uuid_str = str(visible_uuid)
    if visible_uuid_str.replace("-", "").lower() != gpu_uuid.replace("-", "").lower():
        print(f"[preflight] WARNING: visible GPU UUID {visible_uuid_str} does not match requested {gpu_info['uuid']}")

    candidate_bits = [int(b.strip()) for b in args.candidate_bits.split(",")]

    quantize_config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
        act_group_aware=args.act_group_aware,
        sym=args.sym,
        scale_search=args.scale_search,
        offload_to_disk=False,
    )

    print(f"[load] Loading dense model from {args.model_path} ...")
    model = GPTQModel.load(
        args.model_path,
        quantize_config=quantize_config,
        trust_remote_code=False,
        dtype="auto",
        device_map="auto",
    )

    # The loader places the dense model on CPU to save VRAM for the quant loop.
    # Move it to the target GPU for the SLQ weight-only analysis so the per-layer
    # sensitivity scan runs on the accelerator.
    print("[slq] Moving model to GPU for SLQ weight analysis ...")
    model.model = model.model.to("cuda:0")

    print("[slq] Extracting weights and running SLQ analyzer ...")
    names, weights = _extract_quantizable_weights(model.model)
    print(f"[slq] Analyzing {len(names)} quantizable modules")
    dynamic, assignment = _slq_dynamic_allocation(
        names,
        weights,
        candidate_bits,
        budget=args.budget,
        symmetric=args.sym,
        sample_size=args.sample_size,
    )

    # Persist recommendation before quantizing.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "slq_dynamic_bits.json", "w") as f:
        json.dump(dynamic, f, indent=2)

    assigned_bits = [int(candidate_bits[a]) for a in assignment]
    avg_bits = float(np.mean(assigned_bits))
    print(f"[slq] Average assigned bitwidth: {avg_bits:.3f}")
    values, counts = np.unique(assigned_bits, return_counts=True)
    print(f"[slq] Bitwidth histogram: {dict(zip(map(int, values), map(int, counts)))}")

    # Update the loaded model's config with the SLQ dynamic map and re-instantiate.
    quantize_config.dynamic = dynamic
    print(f"[quant] QuantizeConfig: {quantize_config}")

    calibration_texts = _load_calibration_texts(args.calibration_samples)
    print(f"[quant] Quantizing with {len(calibration_texts)} calibration samples ...")
    model.quantize(
        calibration_texts,
        calibration_concat_size=args.calibration_concat_size,
        batch_size=1,
        backend="auto",
    )

    print(f"[quant] Saving to {output} ...")
    model.save(str(output))

    print("[inference] Running generation smoke test ...")
    # Reload for inference so we exercise the full save/load path.
    model = None
    torch.cuda.empty_cache()
    q_model = GPTQModel.load(
        str(output),
        trust_remote_code=False,
        device_map="auto",
        backend="auto",
    )
    device = next(q_model.model.parameters()).device
    prompt = "The capital city of France is"
    inputs = q_model.tokenizer(prompt, return_tensors="pt").to(device)
    outputs = q_model.model.generate(**inputs, max_new_tokens=16, do_sample=False)
    generated = q_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[inference] Prompt: {prompt!r}")
    print(f"[inference] Generated: {generated!r}")

    print("[done]")


if __name__ == "__main__":
    main()
