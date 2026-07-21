#!/usr/bin/env python3
"""Quantize one controlled AWQ ScaleSearch variant and persist auditable A/B metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


# Running a file under scripts/ otherwise lets unrelated editable checkouts win
# import resolution. Pin this benchmark to the repository that owns the script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import huggingface_hub  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers.utils import hub as transformers_hub  # noqa: E402


# Transformers 5.14 moved these Hub helpers out of transformers.utils.hub.
# Keep this benchmark process compatible without changing library-wide behavior.
for _hub_name in ("create_repo", "hf_hub_download", "list_repo_tree", "snapshot_download"):
    if not hasattr(transformers_hub, _hub_name):
        setattr(transformers_hub, _hub_name, getattr(huggingface_hub, _hub_name))

import gptqmodel  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import AWQConfig, FORMAT  # noqa: E402


# Refuse benchmark results produced by an unrelated editable checkout.
GPTQMODEL_SOURCE = Path(gptqmodel.__file__).resolve()
if REPO_ROOT not in GPTQMODEL_SOURCE.parents:
    raise RuntimeError(
        f"Benchmark must import GPT-QModel from {REPO_ROOT}, resolved {GPTQMODEL_SOURCE} instead."
    )


# Match the repository's standalone Llama 3.2 AWQ regression test dataset.
DEFAULT_CALIBRATION_DATASET = "/monster/data/model/dataset/nm-calibration"
DEFAULT_CALIBRATION_CONFIG = "LLM"
DEFAULT_QKV_MODULE_PATTERN = r".*\.self_attn\.(q_proj|k_proj|v_proj)$"
DEFAULT_NON_QKV_MODULE_PATTERN = r".*\.(self_attn\.o_proj|mlp\.(gate_proj|up_proj|down_proj))$"
SOURCE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _parse_args() -> argparse.Namespace:
    """Parse one independently runnable ScaleSearch variant configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--refine-steps", required=True, type=int)
    parser.add_argument(
        "--qkv-refine-steps",
        type=int,
        help="Optional refinement override for modules matched by --qkv-module-pattern.",
    )
    parser.add_argument(
        "--non-qkv-refine-steps",
        type=int,
        help="Optional refinement override for modules matched by --non-qkv-module-pattern.",
    )
    parser.add_argument("--qkv-module-pattern", default=DEFAULT_QKV_MODULE_PATTERN)
    parser.add_argument("--non-qkv-module-pattern", default=DEFAULT_NON_QKV_MODULE_PATTERN)
    parser.add_argument("--calibration-rows", type=int, default=512)
    parser.add_argument("--calibration-concat-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=SOURCE_DTYPES, default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_bytes(value: Any) -> bytes:
    """Encode calibration records deterministically for cross-run identity checks."""

    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


def _load_calibration(rows: int):
    """Load the fixed local calibration prefix and return it with a content fingerprint."""

    dataset = load_dataset(
        path=DEFAULT_CALIBRATION_DATASET,
        name=DEFAULT_CALIBRATION_CONFIG,
        split="train",
    )
    if rows <= 0 or rows > len(dataset):
        raise ValueError(f"calibration rows must be in [1, {len(dataset)}], got {rows}")

    selected = dataset.select(range(rows))
    digest = hashlib.sha256()
    for record in selected:
        digest.update(_json_bytes(dict(record)))
        digest.update(b"\n")
    return selected, digest.hexdigest()


def _finite_awq_losses(quant_logs: dict[str, list[dict[str, Any]]]) -> list[float]:
    """Extract comparable finite module reconstruction losses from AWQ logs."""

    losses = []
    for entry in quant_logs.get("awq", []):
        try:
            loss = float(entry.get("loss"))
        except (TypeError, ValueError):
            continue
        if torch.isfinite(torch.tensor(loss)).item():
            losses.append(loss)
    return losses


def _device_metadata() -> dict[str, Any]:
    """Describe the visible quantization GPU for hardware-controlled comparisons."""

    if not torch.cuda.is_available():
        return {"type": "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {
        "type": "cuda",
        "visible_device": 0,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _scale_search_dynamic(args: argparse.Namespace) -> dict[str, dict[str, int]] | None:
    """Translate experiment labels into configurable per-module AWQ overrides."""

    dynamic = {}
    if args.qkv_refine_steps is not None:
        dynamic[args.qkv_module_pattern] = {
            "scale_search_refine_steps": args.qkv_refine_steps,
        }
    if args.non_qkv_refine_steps is not None:
        dynamic[args.non_qkv_module_pattern] = {
            "scale_search_refine_steps": args.non_qkv_refine_steps,
        }
    return dynamic or None


def main() -> None:
    """Run quantization, save the checkpoint, and write a machine-readable summary."""

    args = _parse_args()
    source_dtype = SOURCE_DTYPES[args.dtype]
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    calibration, calibration_sha256 = _load_calibration(args.calibration_rows)
    quant_config = AWQConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        format=FORMAT.GEMM,
        device="cuda:0",
        offload_to_disk=False,
        scale_search_chunked_activations=True,
        scale_search_gpu_weight_restore=True,
        scale_search_refine_steps=args.refine_steps,
        dynamic=_scale_search_dynamic(args),
    )

    wall_start = time.perf_counter()
    model = GPTQModel.load(
        args.model,
        quantize_config=quant_config,
        dtype=source_dtype,
        attn_implementation="eager",
    )
    load_seconds = time.perf_counter() - wall_start

    quant_start = time.perf_counter()
    quant_logs = model.quantize(
        calibration,
        calibration_concat_size=args.calibration_concat_size,
        calibration_sort="desc",
        batch_size=args.batch_size,
        backend=BACKEND.AUTO,
    )
    quantize_seconds = time.perf_counter() - quant_start

    save_start = time.perf_counter()
    model.save(str(output))
    save_seconds = time.perf_counter() - save_start

    losses = _finite_awq_losses(quant_logs)
    summary = {
        "variant": args.variant,
        "model": str(Path(args.model).expanduser().resolve()),
        "output": str(output),
        "refine_steps": args.refine_steps,
        "qkv_refine_steps": args.qkv_refine_steps,
        "non_qkv_refine_steps": args.non_qkv_refine_steps,
        "scale_search_refine_dynamic": quant_config.dynamic,
        "quantization": {
            "method": "awq",
            "format": "gemm",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
        },
        "calibration": {
            "dataset": DEFAULT_CALIBRATION_DATASET,
            "config": DEFAULT_CALIBRATION_CONFIG,
            "raw_rows": args.calibration_rows,
            "concat_size": args.calibration_concat_size,
            "sort": "desc",
            "sha256": calibration_sha256,
        },
        "timing_seconds": {
            "load": load_seconds,
            "quantize": quantize_seconds,
            "save": save_seconds,
            "total": time.perf_counter() - wall_start,
        },
        "reconstruction_loss": {
            "count": len(losses),
            "mean": statistics.fmean(losses) if losses else None,
            "median": statistics.median(losses) if losses else None,
            "minimum": min(losses) if losses else None,
            "maximum": max(losses) if losses else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gptqmodel": getattr(gptqmodel, "__version__", None),
            "gptqmodel_source": str(GPTQMODEL_SOURCE),
            "device": _device_metadata(),
        },
    }
    summary_path = output / "scalesearch_ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
