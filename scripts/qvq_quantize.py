#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic QVQ preparation, quantization, and checkpoint save harness.

This is the single production-facing QVQ quantization harness.  Preparation
(ordinary activation Hessians, YAQA Sketch-B, output alignment, and modular
replay) remains inside GPTQModel's lifecycle and runs before layer commits.
Post-quantization quality measurement belongs in ``qvq_evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import (  # noqa: E402
    FORMAT,
    ModuleGranularReplayConfig,
    OutputAlignConfig,
    QVQConfig,
    YaqaConfig,
)
from gptqmodel.quantization.config import ChatTemplateConfig  # noqa: E402


QVQ_FORMATS = tuple(
    item.value
    for item in (
        FORMAT.QVQ,
        FORMAT.QVQ_V4,
        FORMAT.QVQ_V4_L18,
        FORMAT.QVQ_DUAL_V2,
        FORMAT.QVQ_V2B2_P32,
        FORMAT.QVQ_V2B4_P64,
    )
)


@dataclass(frozen=True)
class DatasetSlice:
    source: str
    config: str | None
    split: str
    row_start: int
    rows: int

    @property
    def row_stop(self) -> int:
        return self.row_start + self.rows

    @property
    def identity(self) -> tuple[str, str | None, str]:
        path = Path(self.source).expanduser()
        source = str(path.resolve()) if path.exists() else self.source
        return source, self.config, self.split


def _add_dataset_args(parser: argparse.ArgumentParser, prefix: str, *, required: bool = False) -> None:
    option = prefix.replace("_", "-")
    parser.add_argument(f"--{option}-dataset", required=required)
    parser.add_argument(f"--{option}-dataset-config")
    parser.add_argument(f"--{option}-dataset-split", default="train")
    parser.add_argument(f"--{option}-row-start", type=int, default=0)
    parser.add_argument(f"--{option}-rows", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Dense source checkpoint or Hub model ID.")
    parser.add_argument("--output", type=Path, required=True, help="New quantized checkpoint directory.")
    parser.add_argument("--report", type=Path, help="Run manifest; defaults inside the output directory.")
    parser.add_argument("--quant-config", type=Path, help="Complete QVQConfig JSON; convenience flags are ignored.")
    parser.add_argument("--bits", type=float, default=2.0)
    parser.add_argument("--format", choices=QVQ_FORMATS, default=FORMAT.QVQ.value)
    parser.add_argument("--bank-count", type=int, choices=(1, 2, 4))
    parser.add_argument("--rounding", choices=("block_ldlq", "yaqa"), default="yaqa")
    parser.add_argument("--device", default="cuda:0", help="QVQ execution owner; visible extra GPUs are automatic.")
    parser.add_argument("--layers", type=int, help="Quantize only the first N decoder layers; omitted means all.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concat-size", type=int, default=0, help="0 preserves natural calibration rows.")
    parser.add_argument("--calibration-sort", choices=("none", "asc", "desc"), default="desc")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--propagated-bank-selection", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--qvq-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record nested process-quant CUDA/host phases and shape counters in the run manifest.",
    )

    _add_dataset_args(parser, "calibration", required=True)
    _add_dataset_args(parser, "yaqa")
    _add_dataset_args(parser, "validation")
    _add_dataset_args(parser, "replay_search")
    _add_dataset_args(parser, "replay_confirmation")

    parser.set_defaults(calibration_rows=512)
    parser.add_argument("--yaqa-seed", type=int, default=0)
    parser.add_argument(
        "--yaqa-damping",
        "--yaqa-regularization",
        dest="yaqa_regularization",
        type=float,
        default=0.05,
        help="YAQA Hessian damping fallback; rate-specific QVQConfig overrides take precedence.",
    )
    parser.add_argument("--yaqa-minimum-sequences", type=int, default=512)
    parser.add_argument("--yaqa-batch-size", type=int, default=8)
    parser.add_argument("--yaqa-sequence-sort", choices=("none", "asc", "desc"), default="desc")
    parser.add_argument("--yaqa-sample-strategy", default="full")
    parser.add_argument("--yaqa-v2b2-family-mode", choices=("fixed_block_ldlq", "reselect"), default="reselect")
    parser.add_argument("--yaqa-no-activation-checkpointing", action="store_true")
    parser.add_argument("--yaqa-max-factor-bytes-per-pass", type=int)
    parser.add_argument("--yaqa-chat-template-weighting", action="store_true")
    parser.add_argument("--yaqa-chat-template-content-weight", type=float, default=0.97)

    parser.add_argument("--output-alignment", action="store_true")
    parser.add_argument("--output-alignment-lr", type=float, default=1e-5)
    parser.add_argument("--output-alignment-epochs", type=int, default=1)
    parser.add_argument("--output-alignment-train-batches", type=int, default=32)
    parser.add_argument("--output-alignment-validation-batches", type=int, default=16)
    parser.add_argument("--output-alignment-validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-alignment-minimum-improvement", type=float, default=0.0)

    parser.add_argument("--module-granular-replay", action="store_true")
    parser.add_argument(
        "--module-replay-subsets",
        nargs="+",
        default=("attention_qkvo",),
        choices=("attention_qk", "attention_vo", "attention_qkvo", "mlp_gate_up", "mlp_down", "mlp_gate_up_down"),
    )
    return parser


def _quant_log_rows(quant_log: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for rows in quant_log.values() for row in rows if isinstance(row, dict)]


def aggregate_qvq_process_telemetry(
    quant_log: dict[str, list[dict[str, Any]]],
) -> dict[str, object] | None:
    """Aggregate finalized per-module QVQ timers without another CUDA synchronization."""

    phases: dict[str, dict[str, float | int | None]] = {}
    counters: Counter[str] = Counter()
    modules: list[dict[str, object]] = []
    shapes: dict[str, dict[str, object]] = {}
    for row in _quant_log_rows(quant_log):
        telemetry = row.get("qvq_telemetry")
        if not isinstance(telemetry, dict):
            continue
        module_counters = telemetry.get("counters", {})
        counters.update(module_counters)
        input_features = int(module_counters.get("input_features", 0))
        output_features = int(module_counters.get("output_features", 0))
        shape_key = f"{output_features}x{input_features}"
        shape = shapes.setdefault(
            shape_key,
            {"modules": 0, "process_quant_seconds": 0.0, "phases": {}, "counters": Counter()},
        )
        shape["modules"] += 1
        shape["process_quant_seconds"] += float(row.get("time", 0.0))
        shape["counters"].update(module_counters)
        module_phases: dict[str, object] = {}
        for name, values in telemetry.get("phases", {}).items():
            module_values = {
                "calls": int(values["calls"]),
                "host_dispatch_ms": float(values["host_dispatch_ms"]),
                "gpu_ms": None if values["gpu_ms"] is None else float(values["gpu_ms"]),
            }
            module_phases[name] = module_values
            for target in (phases, shape["phases"]):
                aggregate = target.setdefault(
                    name,
                    {"calls": 0, "host_dispatch_ms": 0.0, "gpu_ms": 0.0},
                )
                aggregate["calls"] += module_values["calls"]
                aggregate["host_dispatch_ms"] += module_values["host_dispatch_ms"]
                if module_values["gpu_ms"] is None:
                    aggregate["gpu_ms"] = None
                elif aggregate["gpu_ms"] is not None:
                    aggregate["gpu_ms"] += module_values["gpu_ms"]
        modules.append(
            {
                "full_name": row.get("full_name"),
                "shape": shape_key,
                "process_quant_seconds": float(row.get("time", 0.0)),
                "phases": module_phases,
                "counters": dict(module_counters),
            }
        )
    if not modules:
        return None
    for shape in shapes.values():
        shape["counters"] = dict(shape["counters"])
    return {
        "timing_semantics": "inclusive; nested phase totals must not be added together",
        "process_quant_seconds": sum(float(module["process_quant_seconds"]) for module in modules),
        "phases": phases,
        "counters": dict(counters),
        "shapes": shapes,
        "modules": modules,
    }


def _automatic_bank_count(format_value: str) -> int:
    if format_value == FORMAT.QVQ_V2B2_P32.value:
        return 2
    if format_value in {FORMAT.QVQ_V4.value, FORMAT.QVQ_V2B4_P64.value}:
        return 4
    return 1


def build_quantize_config(args: argparse.Namespace) -> QVQConfig:
    if args.quant_config is not None:
        payload = json.loads(args.quant_config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("--quant-config must contain one JSON object")
        return QVQConfig(**payload)

    yaqa = YaqaConfig(
        seed=args.yaqa_seed,
        regularization=args.yaqa_regularization,
        minimum_sequences=args.yaqa_minimum_sequences,
        batch_size=args.yaqa_batch_size,
        sequence_sort=args.yaqa_sequence_sort,
        sample_strategy=args.yaqa_sample_strategy,
        v2b2_family_mode=args.yaqa_v2b2_family_mode,
        activation_checkpointing=not args.yaqa_no_activation_checkpointing,
        max_factor_bytes_per_pass=args.yaqa_max_factor_bytes_per_pass,
        chat_template=ChatTemplateConfig(
            enabled=args.yaqa_chat_template_weighting,
            content_weight=args.yaqa_chat_template_content_weight,
        ),
    )
    alignment = None
    if args.output_alignment:
        alignment = OutputAlignConfig(
            learning_rate=args.output_alignment_lr,
            epochs=args.output_alignment_epochs,
            maximum_train_batches=args.output_alignment_train_batches,
            maximum_validation_batches=args.output_alignment_validation_batches,
            validation_fraction=args.output_alignment_validation_fraction,
            minimum_relative_improvement=args.output_alignment_minimum_improvement,
        )
    replay = (
        ModuleGranularReplayConfig(subsets=tuple(args.module_replay_subsets))
        if args.module_granular_replay
        else None
    )
    return QVQConfig(
        bits=args.bits,
        format=FORMAT(args.format),
        bank_count=args.bank_count or _automatic_bank_count(args.format),
        rounding=args.rounding,
        device=args.device,
        propagated_bank_selection=args.propagated_bank_selection,
        yaqa=yaqa,
        output_alignment=alignment,
        module_granular_replay=replay,
        # YAQA preparation performs an exact full-model backward for Sketch-B.
        # A checkpoint-backed LazyTurtle shell cannot participate in autograd.
        offload_to_disk=args.rounding != "yaqa",
    )


def _slice_from_args(args: argparse.Namespace, prefix: str, *, fallback: DatasetSlice | None = None) -> DatasetSlice | None:
    source = getattr(args, f"{prefix}_dataset")
    rows = getattr(args, f"{prefix}_rows")
    if source is None and rows is None:
        return fallback
    if source is None:
        if fallback is None:
            raise ValueError(f"--{prefix.replace('_', '-')}-rows requires a dataset")
        source = fallback.source
    if rows is None:
        raise ValueError(f"--{prefix.replace('_', '-')}-dataset requires --{prefix.replace('_', '-')}-rows")
    config = getattr(args, f"{prefix}_dataset_config")
    split = getattr(args, f"{prefix}_dataset_split")
    start = getattr(args, f"{prefix}_row_start")
    if start < 0 or rows < 1:
        raise ValueError(f"{prefix} row start must be nonnegative and rows must be positive")
    return DatasetSlice(str(source), config, split, start, rows)


def validate_disjoint_slices(named_slices: dict[str, DatasetSlice | None]) -> None:
    present = [(name, item) for name, item in named_slices.items() if item is not None]
    for index, (left_name, left) in enumerate(present):
        for right_name, right in present[index + 1 :]:
            if left.identity != right.identity:
                continue
            if max(left.row_start, right.row_start) < min(left.row_stop, right.row_stop):
                raise ValueError(
                    f"Dataset slices `{left_name}` [{left.row_start}, {left.row_stop}) and `{right_name}` "
                    f"[{right.row_start}, {right.row_stop}) overlap"
                )


def load_dataset_slice(spec: DatasetSlice):
    path = Path(spec.source).expanduser()
    kwargs: dict[str, Any] = {"split": spec.split}
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            dataset = load_dataset("parquet", data_files={spec.split: str(path)}, **kwargs)
        elif suffix in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files={spec.split: str(path)}, **kwargs)
        else:
            raise ValueError(f"Unsupported local dataset file: {path}")
    else:
        if spec.config is not None:
            kwargs["name"] = spec.config
        dataset = load_dataset(str(path) if path.exists() else spec.source, **kwargs)
    if spec.row_stop > len(dataset):
        raise ValueError(
            f"Dataset slice [{spec.row_start}, {spec.row_stop}) exceeds `{spec.source}` length {len(dataset)}"
        )
    return dataset.select(range(spec.row_start, spec.row_stop))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.concat_size < 0:
        raise ValueError("batch size must be positive and concat size must be nonnegative")
    if args.layers is not None and args.layers < 1:
        raise ValueError("--layers must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output}")

    calibration_spec = _slice_from_args(args, "calibration")
    assert calibration_spec is not None
    yaqa_spec = _slice_from_args(args, "yaqa")
    validation_spec = _slice_from_args(args, "validation")
    replay_search_spec = _slice_from_args(args, "replay_search")
    replay_confirmation_spec = _slice_from_args(args, "replay_confirmation")
    config = build_quantize_config(args)
    if config.rounding == "yaqa" and yaqa_spec is None:
        yaqa_spec = calibration_spec
    if config.module_granular_replay is not None and (replay_search_spec is None or replay_confirmation_spec is None):
        raise ValueError("Module-granular replay requires explicit search and confirmation dataset slices")
    validate_disjoint_slices(
        {
            "calibration": calibration_spec,
            "yaqa": None if yaqa_spec == calibration_spec else yaqa_spec,
            "validation": validation_spec,
            "replay_search": replay_search_spec,
            "replay_confirmation": replay_confirmation_spec,
        }
    )

    streams = {
        "calibration": load_dataset_slice(calibration_spec),
        "yaqa": None if yaqa_spec is None or yaqa_spec == calibration_spec else load_dataset_slice(yaqa_spec),
        "validation": None if validation_spec is None else load_dataset_slice(validation_spec),
        "replay_search": None if replay_search_spec is None else load_dataset_slice(replay_search_spec),
        "replay_confirmation": (
            None if replay_confirmation_spec is None else load_dataset_slice(replay_confirmation_spec)
        ),
    }
    if config.rounding == "yaqa" and streams["yaqa"] is None:
        streams["yaqa"] = streams["calibration"]

    load_started = time.perf_counter()
    model = GPTQModel.load(
        args.model,
        quantize_config=config,
        dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    load_seconds = time.perf_counter() - load_started
    quant_started = time.perf_counter()
    telemetry_environment = "GPTQMODEL_QVQ_TELEMETRY"
    previous_telemetry = os.environ.get(telemetry_environment)
    if args.qvq_telemetry:
        os.environ[telemetry_environment] = "1"
    else:
        os.environ.pop(telemetry_environment, None)
    try:
        quant_log = model.quantize(
            streams["calibration"],
            calibration_concat_size=args.concat_size or None,
            calibration_sort=None if args.calibration_sort == "none" else args.calibration_sort,
            batch_size=args.batch_size,
            backend=BACKEND.QVQ,
            validation_calibration=streams["validation"],
            yaqa_calibration=streams["yaqa"],
            module_replay_search_calibration=streams["replay_search"],
            module_replay_confirmation_calibration=streams["replay_confirmation"],
            layer_scope=None if args.layers is None else slice(0, args.layers),
        )
    finally:
        if previous_telemetry is None:
            os.environ.pop(telemetry_environment, None)
        else:
            os.environ[telemetry_environment] = previous_telemetry
    quant_seconds = time.perf_counter() - quant_started
    save_started = time.perf_counter()
    model.save(str(output))
    save_seconds = time.perf_counter() - save_started

    report_path = args.report.expanduser().resolve() if args.report else output / "qvq_quantize_run.json"
    payload = {
        "model": args.model,
        "output": str(output),
        "commit": _git_commit(),
        "python": platform.python_version(),
        "python_gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "torch": torch.__version__,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
        "quantize_config": config.to_dict(),
        "datasets": {
            name: None if spec is None else asdict(spec)
            for name, spec in {
                "calibration": calibration_spec,
                "yaqa": yaqa_spec,
                "validation": validation_spec,
                "replay_search": replay_search_spec,
                "replay_confirmation": replay_confirmation_spec,
            }.items()
        },
        "layer_scope": "all" if args.layers is None else {"first_layers": args.layers},
        "seconds": {"load": load_seconds, "prepare_and_quantize": quant_seconds, "save": save_seconds},
        "quant_log_rows": len(_quant_log_rows(quant_log)),
        "telemetry": {
            "qvq_process_quant": aggregate_qvq_process_telemetry(quant_log),
            "lifecycle": {
                "aggregate": model.quant_region_timer.snapshot(),
                "layers": model.quant_region_timer.period_snapshots(),
            },
            "yaqa_sketch_b": next(
                (
                    row["yaqa_sketch_b_telemetry"]
                    for row in _quant_log_rows(quant_log)
                    if isinstance(row.get("yaqa_sketch_b_telemetry"), dict)
                ),
                None,
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
