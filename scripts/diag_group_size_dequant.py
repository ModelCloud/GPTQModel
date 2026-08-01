#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare dequantized group-size checkpoints back to the dense source weights.

Loads the dense Llama-3.2-1B model and each saved GPTQ checkpoint on CPU using
BACKEND.GPTQ_TORCH, calls module.dequantize_weight(), and reports RMSE/cosine
per module and aggregate statistics.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, help="Dense model path")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Saved GPTQ checkpoint label and path",
    )
    parser.add_argument("--output", default="/tmp/group_size_dequant.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="gptq_torch")
    parser.add_argument("--max-modules", type=int, default=0)
    return parser.parse_args()


def _load_dense(path: str, device: str):
    print(f"[dense] loading {path} ...")
    model = GPTQModel.load(path, device=device)
    return model.model if hasattr(model, "model") else model


def _load_quantized(path: str, device: str, backend: BACKEND):
    print(f"[quant] loading {path} with {backend.value} ...")
    model = GPTQModel.load(
        path,
        device=device,
        backend=backend,
    )
    return model.model if hasattr(model, "model") else model


def _dense_weight(module: torch.nn.Module) -> torch.Tensor:
    if isinstance(module, BaseQuantLinear):
        return None
    w = getattr(module, "weight", None)
    if w is None:
        return None
    return w.float()


def _compare_module(dense_mod: torch.nn.Module, quant_mod: torch.nn.Module, name: str) -> Dict[str, Any]:
    orig = _dense_weight(dense_mod)
    if orig is None or not isinstance(quant_mod, BaseQuantLinear):
        return None

    dequant = quant_mod.dequantize_weight().float()  # [in_features, out_features]
    # Original nn.Linear.weight is [out_features, in_features]
    dequant_t = dequant.t()  # [out_features, in_features]

    if dequant_t.shape != orig.shape:
        # Try the other orientation as a fallback (Conv1D / transposed layouts)
        if dequant.shape == orig.shape:
            dequant_t = dequant
        else:
            return {
                "name": name,
                "error": f"shape mismatch: orig {tuple(orig.shape)} vs dequant {tuple(dequant.shape)}",
            }

    # dequantize_weight may return a tensor on a different device; align and compute on that device.
    orig_dev = orig.to(device=dequant_t.device, dtype=torch.float32)
    diff = dequant_t - orig_dev
    mse = (diff * diff).mean().item()
    rmse = math.sqrt(mse)
    orig_l2 = torch.linalg.norm(orig_dev).item()
    rel_rmse = rmse / max(orig_l2 / math.sqrt(orig_dev.numel()), 1e-12)
    max_abs = diff.abs().max().item()
    cos = (
        torch.nn.functional.cosine_similarity(orig_dev.flatten(), dequant_t.flatten(), dim=0).item()
    )
    del orig_dev, dequant_t, dequant, diff
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "shape": list(orig.shape),
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "max_abs": max_abs,
        "cosine": cos,
    }


def main() -> int:
    args = _parse_args()
    backend = BACKEND(args.backend)
    device = args.device

    dense_model = _load_dense(args.dense, device)
    results: List[Dict[str, Any]] = []

    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        quant_model = _load_quantized(path, device, backend)

        comparisons: List[Dict[str, Any]] = []
        for (d_name, d_mod), (q_name, q_mod) in zip(
            dense_model.named_modules(), quant_model.named_modules()
        ):
            if d_name != q_name:
                continue
            if args.max_modules and len(comparisons) >= args.max_modules:
                break
            cmp = _compare_module(d_mod, q_mod, d_name)
            if cmp is not None:
                comparisons.append(cmp)

        # Aggregate across modules
        rel_rmse_vals = [c["rel_rmse"] for c in comparisons if "rel_rmse" in c]
        rmse_vals = [c["rmse"] for c in comparisons if "rmse" in c]
        cos_vals = [c["cosine"] for c in comparisons if "cosine" in c]
        results.append({
            "label": label,
            "path": path,
            "backend": backend.value,
            "modules": len(comparisons),
            "mean_rmse": sum(rmse_vals) / len(rmse_vals) if rmse_vals else None,
            "median_rmse": sorted(rmse_vals)[len(rmse_vals) // 2] if rmse_vals else None,
            "mean_rel_rmse": sum(rel_rmse_vals) / len(rel_rmse_vals) if rel_rmse_vals else None,
            "median_rel_rmse": sorted(rel_rmse_vals)[len(rel_rmse_vals) // 2] if rel_rmse_vals else None,
            "mean_cosine": sum(cos_vals) / len(cos_vals) if cos_vals else None,
            "median_cosine": sorted(cos_vals)[len(cos_vals) // 2] if cos_vals else None,
            "comparisons": comparisons,
        })

        del quant_model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== group-size dequant comparison ===")
    print(f"{'label':<12} {'mod':>4} {'mean_rmse':>11} {'med_rmse':>11} {'mean_rel':>11} {'med_rel':>11} {'mean_cos':>10}")
    for r in results:
        print(
            f"{r['label']:<12} {r['modules']:>4} "
            f"{r['mean_rmse']:>11.6f} {r['median_rmse']:>11.6f} "
            f"{r['mean_rel_rmse']*100:>10.4f}% {r['median_rel_rmse']*100:>10.4f}% "
            f"{r['mean_cosine']:>10.6f}"
        )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
