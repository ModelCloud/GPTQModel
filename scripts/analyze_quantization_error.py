#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Scan dense model weights for quantization-sensitive modules and regions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate grouped quantization error for every materialized linear/embedding weight and emit "
            "JSON, Markdown, region mappings, and an optional module-level dynamic quantization config."
        )
    )
    parser.add_argument("--model", required=True, help="Dense Hugging Face model ID or local path.")
    parser.add_argument("--output-dir", required=True, help="Directory for reports and plans.")
    parser.add_argument("--quant-config", help="Existing quantization-config JSON. CLI values override it.")
    parser.add_argument("--method", default=None, help="Quantization method, for example gptq or rtn.")
    parser.add_argument("--format", default=None, help="Checkpoint format, for example gptq.")
    parser.add_argument("--bits", default=None, help="Target bit width or supported format-specific bit alias.")
    parser.add_argument("--group-size", type=int, default=None, help="Input-feature group size; -1 is per row.")
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lm-head", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--module-regex", help="Analyze only matching fully qualified module names.")
    parser.add_argument("--max-modules", type=int, help="Stop after this many analyzed modules.")
    parser.add_argument("--top-k", type=int, default=32, help="Modules retained in the Markdown summary.")
    parser.add_argument("--top-k-regions", type=int, default=128, help="Regions retained in the global mapping.")
    parser.add_argument("--regions-per-module", type=int, default=8)
    parser.add_argument("--include-endpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bad-group-rel-rmse", type=float, default=0.10)
    parser.add_argument("--recommendation-percentile", type=float, default=95.0)
    parser.add_argument("--min-recommendation-risk", type=float, default=20.0)
    parser.add_argument("--promotion-bits", type=int, default=8)
    parser.add_argument("--promotion-group-size", type=int, default=32)
    parser.add_argument(
        "--fusion-profile",
        choices=("none", "model_definition", "vllm_sglang"),
        default="model_definition",
        help=(
            "Expand direct recommendations using GPTQModel model-definition groups. "
            "`vllm_sglang` is accepted as a deprecated alias for `model_definition`."
        ),
    )
    parser.add_argument("--max-chunk-values", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-sample-values", type=int, default=1024 * 1024)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        help="Physical nvidia-smi index used for analysis; resolved to a UUID before importing Torch.",
    )
    parser.add_argument(
        "--load-device",
        default="cpu",
        help="Device passed to GPTQModel.load. Chunked analysis can still run on --physical-gpu.",
    )
    parser.add_argument("--gpu-memory-used-limit-mib", type=int, default=64)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--apply-plan",
        action="store_true",
        help="Explicitly merge non-endpoint plan rules into quantize_config.planned.json.",
    )
    return parser.parse_args()


def _run_nvidia_smi(*arguments: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _gpu_inventory() -> Dict[int, Dict[str, Any]]:
    output = _run_nvidia_smi(
        "--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    inventory = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 6)]
        if len(fields) != 7:
            continue
        index, bus_id, uuid, name, memory_total, memory_used, utilization = fields
        inventory[int(index)] = {
            "physical_index": int(index),
            "pci_bus_id": bus_id,
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory_total),
            "memory_used_mib": int(memory_used),
            "utilization_gpu_percent": int(utilization),
        }
    return inventory


def _compute_processes() -> list[Dict[str, Any]]:
    try:
        output = _run_nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    except subprocess.CalledProcessError:
        return []
    processes = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) == 4:
            uuid, pid, process_name, memory_used = fields
            processes.append(
                {
                    "uuid": uuid,
                    "pid": int(pid),
                    "process_name": process_name,
                    "memory_used_mib": int(memory_used),
                }
            )
    return processes


def _preflight_physical_gpu(
    physical_index: int,
    *,
    memory_used_limit_mib: int,
    allow_busy: bool,
) -> Dict[str, Any]:
    samples = []
    for sample_index in range(3):
        inventory = _gpu_inventory()
        if physical_index not in inventory:
            raise RuntimeError(
                f"Physical GPU {physical_index} does not exist; available indices are {sorted(inventory)}."
            )
        samples.append(inventory[physical_index])
        if sample_index < 2:
            time.sleep(1)
    target = samples[-1]
    foreign_processes = [
        process
        for process in _compute_processes()
        if process["uuid"] == target["uuid"] and process["pid"] != os.getpid()
    ]
    idle = all(
        sample["utilization_gpu_percent"] == 0
        and sample["memory_used_mib"] <= memory_used_limit_mib
        for sample in samples
    )
    if not allow_busy and (not idle or foreign_processes):
        raise RuntimeError(
            f"Physical GPU {physical_index} is not idle: samples={samples}, processes={foreign_processes}. "
            "Wait for an idle device or pass --allow-busy-gpu after reviewing the conflict."
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = target["uuid"]
    return {
        **target,
        "idle_samples": samples,
        "foreign_compute_processes": foreign_processes,
        "cuda_visible_devices": target["uuid"],
    }


def _parse_bits(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _load_quantize_config(args: argparse.Namespace):
    from gptqmodel import QuantizeConfig

    if args.quant_config:
        with Path(args.quant_config).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        qcfg = QuantizeConfig.from_quant_config(payload)
        overrides = {
            "method": args.method,
            "format": args.format,
            "bits": _parse_bits(args.bits),
            "group_size": args.group_size,
            "sym": args.sym,
            "desc_act": args.desc_act,
            "lm_head": args.lm_head,
        }
        if any(value is not None for value in overrides.values()):
            merged = qcfg.to_dict()
            merged.update({key: value for key, value in overrides.items() if value is not None})
            qcfg = QuantizeConfig.from_quant_config(merged)
        return qcfg

    kwargs = {
        "method": args.method or "gptq",
        "format": args.format or "gptq",
        "bits": _parse_bits(args.bits) if args.bits is not None else 4,
        "group_size": args.group_size if args.group_size is not None else 128,
        "sym": args.sym if args.sym is not None else True,
        "desc_act": args.desc_act,
        "lm_head": args.lm_head if args.lm_head is not None else False,
    }
    return QuantizeConfig(**kwargs)


def _runtime_metadata(torch, gpu_preflight: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "compute_device": "cpu",
    }
    if gpu_preflight is not None:
        properties = torch.cuda.get_device_properties(0)
        runtime.update(
            {
                "compute_device": "cuda:0",
                "visible_gpu_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "sm_count": properties.multi_processor_count,
                "visible_memory_bytes": properties.total_memory,
                "physical_gpu": gpu_preflight,
            }
        )
    return runtime


def _checkpoint_index_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Read layer-index evidence directly from a local safetensors index."""

    root = Path(model_path)
    if not root.is_dir():
        return None
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    if not index_paths:
        return None
    index_path = index_paths[0]
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return None
    layer_indices = set()
    for tensor_name in weight_map:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", tensor_name)
        if match:
            layer_indices.add(int(match.group(1)))
    ordered = sorted(layer_indices)
    return {
        "index_file": index_path.name,
        "tensor_key_count": len(weight_map),
        "layer_min": ordered[0] if ordered else None,
        "layer_max": ordered[-1] if ordered else None,
        "layer_count": len(ordered),
        "layer_indices": ordered,
        "input_embedding_key": "model.embed_tokens.weight" if "model.embed_tokens.weight" in weight_map else None,
        "lm_head_key": "lm_head.weight" if "lm_head.weight" in weight_map else None,
    }


def main() -> int:
    args = _parse_args()
    gpu_preflight = None
    if args.physical_gpu is not None:
        gpu_preflight = _preflight_physical_gpu(
            args.physical_gpu,
            memory_used_limit_mib=args.gpu_memory_used_limit_mib,
            allow_busy=args.allow_busy_gpu,
        )

    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

    import torch

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import (
        AnalysisConfig,
        AnalysisSelection,
        QuantizationAnalyzer,
        apply_analysis_plan,
        render_analysis_markdown,
        report_to_json,
    )

    qcfg = _load_quantize_config(args)
    analysis_config = AnalysisConfig(
        top_k=args.top_k,
        top_k_regions=args.top_k_regions,
        regions_per_module=args.regions_per_module,
        include_endpoints=args.include_endpoints,
        bad_block_rel_rmse_threshold=args.bad_group_rel_rmse,
        recommendation_percentile=args.recommendation_percentile,
        min_recommendation_risk=args.min_recommendation_risk,
        promotion_bits=args.promotion_bits,
        promotion_group_size=args.promotion_group_size,
        fusion_profile=args.fusion_profile,
        max_chunk_values=args.max_chunk_values,
        max_sample_values=args.max_sample_values,
    )
    dtype = "auto" if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    loaded = GPTQModel.load(
        args.model,
        quantize_config=qcfg,
        device=args.load_device,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
    )
    model = loaded.model if hasattr(loaded, "model") else loaded
    compute_device = "cuda:0" if gpu_preflight is not None else "cpu"
    module_groups = []
    group_source = None
    simple_layer_modules = getattr(loaded, "simple_layer_modules", None)
    if callable(simple_layer_modules):
        from gptqmodel.quantization.config import METHOD

        module_groups = simple_layer_modules(
            model_config=model.config,
            quantize_config=qcfg,
            is_awq_quantize=qcfg.method in {METHOD.AWQ, METHOD.PARO},
            include_capture_only=False,
        )
        group_source = f"{type(loaded).__module__}.{type(loaded).__name__}.module_tree"
    analyzer = QuantizationAnalyzer(
        qcfg,
        analysis_config,
        compute_device=compute_device,
        module_groups=module_groups,
        module_group_source=group_source,
    )
    turtle = getattr(loaded, "turtle_model", None)
    weight_resolver = None
    if callable(getattr(turtle, "checkpoint_tensors_for_submodule", None)):
        def _resolve_checkpoint_weight(_name, module):
            tensors = turtle.checkpoint_tensors_for_submodule(
                target_model=model,
                target_submodule=module,
                recurse=False,
            )
            return tensors.get("weight")

        weight_resolver = _resolve_checkpoint_weight
    report = analyzer.analyze_model(
        model,
        selection=AnalysisSelection(
            module_pattern=args.module_regex,
            max_modules=args.max_modules,
        ),
        weight_resolver=weight_resolver,
    )
    report["runtime"] = _runtime_metadata(torch, gpu_preflight)
    report["model"] = args.model
    report["weight_source"] = "lazy_turtle_checkpoint_stream" if weight_resolver else "materialized_model"
    checkpoint_index = _checkpoint_index_metadata(args.model)
    if checkpoint_index is not None:
        report["checkpoint_index"] = checkpoint_index
    report["markdown"] = render_analysis_markdown(report, top_k=analysis_config.top_k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_without_markdown = {key: value for key, value in report.items() if key != "markdown"}
    (output_dir / "quantization_analysis.json").write_text(
        report_to_json(report_without_markdown) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quantization_analysis.md").write_text(report["markdown"] + "\n", encoding="utf-8")
    (output_dir / "quantization_regions.json").write_text(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "config": report["config"],
                "regions": report["regions"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "quantization_plan.json").write_text(
        json.dumps(report["plan"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.apply_plan:
        planned, merge = apply_analysis_plan(qcfg, report["plan"])
        payload = planned.to_dict()
        payload.setdefault("meta", {})
        payload["meta"]["analysis_plan_merge"] = merge
        (output_dir / "quantize_config.planned.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(report["markdown"])
    print(
        f"\nWrote analysis artifacts to {output_dir.resolve()} "
        f"({report['summary']['analyzed_modules']} modules, {report['summary']['flagged_modules']} flagged)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
