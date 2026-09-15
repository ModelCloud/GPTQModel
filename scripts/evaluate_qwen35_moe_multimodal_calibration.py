#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Auditable, resumable Qwen3.5-MoE multimodal calibration harness.

The default command runs the frozen matrix.  Pass ``--dry-run`` to only write
its provenance.  Every non-dry-run invocation must
provide at least one existing image path with ``--image`` or ``--images``;
that requirement is checked before any worker or model is started.  A worker
is run in a fresh process per cell so an OOM or a broken model cannot
contaminate another cell.  This file deliberately does not import or patch a
Qwen model definition; it uses the public GPT-QModel load/quantize API and
the model's native processor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
SCRIPT_VERSION = "qwen35-mm-calibration-1"
CELL_ORDER = ("dense", "gptq-text", "gptq-multimodal", "awq-text", "awq-multimodal")
QUANT_CELLS = frozenset(CELL_ORDER[1:])
DEFAULT_PROMPTS = (
    "Describe the input and answer briefly.",
    "What is the main subject?",
)


def canonical_json(value: Any) -> str:
    """Serialize protocol data deterministically for hashing and manifests."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return contract violations; an empty list means an accepted manifest."""

    errors: list[str] = []
    required = (
        "schema_version",
        "success",
        "engine",
        "model",
        "task",
        "metric",
        "score",
        "samples",
        "expected_samples",
        "protocol_hash",
        "timing",
        "versions",
        "gpus",
        "command",
        "config",
        "raw_result",
        "log",
    )
    for key in required:
        if key not in manifest:
            errors.append(f"missing:{key}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if manifest.get("success") is not True:
        errors.append("success_not_true")
    if not _finite(manifest.get("score")):
        errors.append("score_not_finite")
    if manifest.get("samples") != manifest.get("expected_samples"):
        errors.append("sample_count")
    if not isinstance(manifest.get("protocol_hash"), str) or not manifest.get(
        "protocol_hash"
    ):
        errors.append("protocol_hash")
    for key in ("timing", "versions"):
        if not isinstance(manifest.get(key), Mapping):
            errors.append(f"{key}_not_object")
    versions = manifest.get("versions")
    if isinstance(versions, Mapping) and not versions.get("engine"):
        errors.append("versions_engine")
    gpus = manifest.get("gpus")
    if not isinstance(gpus, list):
        errors.append("gpus_not_list")
    else:
        for index, gpu in enumerate(gpus):
            physical_id = gpu.get("physical_id") if isinstance(gpu, Mapping) else None
            if not isinstance(physical_id, int) or physical_id < 0:
                errors.append(f"gpu_{index}_physical_id")
    return errors


def is_valid_manifest(manifest: Mapping[str, Any]) -> bool:
    return not validate_manifest(manifest)


def tensor_summary(
    value: Any, *, include_values: bool = False, max_values: int = 16
) -> dict[str, Any]:
    """Small deterministic summary for tensors/arrays without storing image data."""

    try:
        import torch
    except Exception:  # pragma: no cover - pure plan mode has no torch requirement
        torch = None
    if torch is not None and torch.is_tensor(value):
        tensor = value.detach().float().cpu()
        flat = tensor.reshape(-1)
        result: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "finite": bool(torch.isfinite(tensor).all().item())
            if tensor.numel()
            else True,
        }
        if tensor.numel():
            result.update(
                {
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                    "mean": float(flat.mean()),
                    "std": float(flat.std(unbiased=False)),
                    "l2": float(torch.linalg.vector_norm(flat)),
                    "absmax": float(flat.abs().max()),
                }
            )
            if include_values:
                result["values"] = flat[:max_values].tolist()
        return result
    if isinstance(value, (str, bytes, bytearray)):
        return {
            "type": type(value).__name__,
            "sha256": hashlib.sha256(
                bytes(value, "utf-8") if isinstance(value, str) else bytes(value)
            ).hexdigest(),
        }
    if isinstance(value, Sequence):
        values = list(value)
        result = {
            "shape": [len(values)],
            "numel": len(values),
            "type": type(value).__name__,
        }
        if values and all(isinstance(v, (int, float)) for v in values):
            finite_values = [float(v) for v in values if _finite(v)]
            result.update(
                {
                    "finite": len(finite_values) == len(values),
                    "min": min(finite_values),
                    "max": max(finite_values),
                    "mean": sum(finite_values) / len(finite_values),
                }
            ) if finite_values else result.update({"finite": True})
        if include_values:
            result["values"] = values[:max_values]
        return result
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def summarize_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in inputs.items():
        if key in {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "video_grid_thw",
        }:
            summary[key] = tensor_summary(value, include_values=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        else:
            summary[key] = tensor_summary(value)
    return summary


def _read_list(value: str | None, *, default: Sequence[str] = ()) -> list[str]:
    if not value:
        return list(default)
    path = Path(value)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [line.strip() for line in text.splitlines() if line.strip()]
        if not isinstance(parsed, list):
            raise ValueError(f"{path} must contain a JSON list")
        return [str(item) if not isinstance(item, Mapping) else item for item in parsed]  # type: ignore[list-item]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise ValueError(
            "list argument must be a JSON list, file, or comma-separated string"
        )
    return [str(item) for item in parsed]


def _validate_runtime_images(images: Sequence[str]) -> None:
    """Validate required multimodal inputs before starting runtime work."""

    if not images:
        raise ValueError(
            "actual multimodal evaluation requires at least one image; pass "
            "--image PATH or --images LIST (or use --dry-run to freeze the plan)"
        )
    values = [str(image) for image in images]
    invalid = [image for image in values if not image.strip()]
    missing = [
        image for image in values if image.strip() and not Path(image).is_file()
    ]
    if invalid or missing:
        details = []
        if invalid:
            details.append("empty image path")
        if missing:
            details.append("missing image path(s): " + ", ".join(missing))
        raise ValueError("actual multimodal evaluation requires existing image files (" + "; ".join(details) + ")")


def _key_values(items: Iterable[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    return values


def selected_cells(value: str | None) -> list[str]:
    cells = (
        CELL_ORDER
        if not value
        else tuple(x.strip().lower() for x in value.split(",") if x.strip())
    )
    unknown = sorted(set(cells) - set(CELL_ORDER))
    if unknown:
        raise ValueError(
            f"unknown cells {unknown}; choose from {', '.join(CELL_ORDER)}"
        )
    if len(set(cells)) != len(cells):
        raise ValueError("duplicate cell selection")
    return list(cells)


def build_protocol(
    args: argparse.Namespace, prompts: Sequence[str], images: Sequence[str]
) -> dict[str, Any]:
    return {
        "model": {"path": str(args.model_path), "revision": args.revision},
        "seed": args.seed,
        "prompts": list(prompts),
        "images": list(images),
        "quant": {
            "bits": args.bits,
            "group_size": args.group_size,
            "sym": args.sym,
            "batch_size": args.batch_size,
            "calibration_rows": args.calibration_rows,
            "calibration_concat_size": args.calibration_concat_size,
            "params": _key_values(args.quant_param),
        },
        "inference": {
            "dtype": args.dtype,
            "device_map": args.device_map,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "params": _key_values(args.inference_param),
        },
        "quant_output_root": str(args.quant_output_root)
        if args.quant_output_root
        else None,
        "trust_remote_code": bool(args.trust_remote_code),
        "debug": {
            "max_quant_layers": args.max_quant_layers,
            "stop_after_layer": args.stop_after_layer,
        },
        "script_version": SCRIPT_VERSION,
    }


def make_cell_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompts = _read_list(args.prompts, default=DEFAULT_PROMPTS)
    prompts.extend(args.prompt)
    images = _read_list(args.images) + list(args.image)
    if not prompts:
        raise ValueError("at least one prompt is required")
    if not args.dry_run:
        _validate_runtime_images(images)
    protocol = build_protocol(args, prompts, images)
    phash = protocol_hash(protocol)
    configs = []
    for cell in selected_cells(args.cells):
        calibration_modality = (
            "none"
            if cell == "dense"
            else ("multimodal" if cell.endswith("multimodal") else "text")
        )
        method = None if cell == "dense" else cell.split("-", 1)[0]
        configs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "cell": cell,
                "engine": cell,
                "method": method,
                "calibration_modality": calibration_modality,
                "inference_modality": "multimodal",
                "model_path": str(args.model_path),
                "revision": args.revision,
                "prompts": prompts,
                "images": images,
                "protocol": protocol,
                "protocol_hash": phash,
                "seed": args.seed,
                "quant": protocol["quant"],
                "inference": protocol["inference"],
                "debug": protocol["debug"],
            }
        )
    return configs


def _slug(cell: str) -> str:
    return cell.replace("/", "_").replace(" ", "_")


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.check_output(
                cmd, cwd=repo, text=True, stderr=subprocess.STDOUT
            ).strip()
        except Exception:
            return "unknown"

    return {
        "repo": str(repo),
        "head": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _write_report(
    root: Path,
    configs: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, str],
    command: str,
) -> None:
    lines = [
        "# Qwen3.5-MoE multimodal calibration",
        "",
        f"Command: `{command}`",
        "",
        "| Cell | Status | Protocol hash |",
        "|---|---|---|",
    ]
    for config in configs:
        cell = str(config["cell"])
        lines.append(
            f"| {cell} | {statuses.get(cell, 'pending')} | `{config['protocol_hash']}` |"
        )
    lines += [
        "",
        "Cells are process-isolated and resumable. A cell is accepted only when its normalized manifest has `success=true`, finite score, and complete sample coverage.",
        "",
        "Multimodal inputs are passed through the model's native processor; rendered prompts, token IDs, image-grid summaries, first language-layer activations, logits, and provenance are retained in raw output.",
        "",
        "`--dry-run` writes the frozen configs and provenance only; it does not fabricate scores.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_artifacts(
    root: Path, configs: Sequence[Mapping[str, Any]], repo: Path, command: str
) -> None:
    for name in (
        "configs",
        "raw",
        "manifests",
        "logs",
        "invalid_attempts",
        "source_patches",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    state = _git_state(repo)
    state["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root / "source_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if state["status"] != "unknown":
        (root / "source_patches" / "working_tree.diff").write_text(
            subprocess.run(
                ["git", "diff"], cwd=repo, text=True, capture_output=True
            ).stdout,
            encoding="utf-8",
        )
    for config in configs:
        (root / "configs" / f"{_slug(config['cell'])}.json").write_text(
            canonical_json(config) + "\n", encoding="utf-8"
        )
    _write_report(root, configs, {}, command)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _gpu_info() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        inventory: dict[str, dict[str, str]] = {}
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            for line in output.splitlines():
                index_value, uuid_value, driver, memory_total = (
                    field.strip() for field in line.split(",", 3)
                )
                item = {
                    "physical_index": index_value,
                    "uuid": uuid_value,
                    "driver": driver,
                    "memory_total_mib_nvidia_smi": memory_total,
                }
                inventory[index_value] = item
                inventory[uuid_value] = item
        except Exception:
            pass
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        result = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            physical_id = (
                visible[index].strip()
                if index < len(visible) and visible[index].strip()
                else str(index)
            )
            details = inventory.get(physical_id, {})
            result.append(
                {
                    "physical_id": int(details.get("physical_index", index)),
                    "visible_id": physical_id,
                    "uuid": details.get("uuid", str(getattr(props, "uuid", "unknown"))),
                    "model": props.name,
                    "driver": details.get("driver", "unknown"),
                    "cuda_runtime": getattr(torch.version, "cuda", None) or "unknown",
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total_mib": int(props.total_memory / (1024 * 1024)),
                    "peak_allocated_mib": round(
                        torch.cuda.max_memory_allocated(index) / (1024 * 1024), 3
                    ),
                    "peak_reserved_mib": round(
                        torch.cuda.max_memory_reserved(index) / (1024 * 1024), 3
                    ),
                }
            )
        return result
    except Exception:
        return []


def _move(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and not isinstance(value, (str, bytes)):
        try:
            return value.to(device)
        except Exception:
            return value
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _first_device(model: Any) -> Any:
    target = getattr(model, "model", model)
    get_input_embeddings = getattr(target, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        try:
            return get_input_embeddings().weight.device
        except Exception:
            pass
    try:
        return next(target.parameters()).device
    except Exception:
        try:
            import torch

            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"


def _conversation(prompt: str, image: str | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _load_image(path: str) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _prepare_sample(
    model: Any, prompt: str, image: str | None, multimodal: bool
) -> tuple[str, Mapping[str, Any]]:
    processor = getattr(model, "processor", None)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    proc_tokenizer = (
        getattr(processor, "tokenizer", None) if processor is not None else None
    )
    tokenizer = tokenizer or proc_tokenizer
    messages = _conversation(prompt, image if multimodal else None)
    if multimodal and processor is None:
        raise RuntimeError("multimodal cell requires model.processor")
    if processor is not None and multimodal:
        image_obj = _load_image(image) if image else None
        apply = getattr(processor, "apply_chat_template", None)
        if not callable(apply):
            raise RuntimeError("processor lacks apply_chat_template")
        try:
            rendered = apply(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[rendered],
                images=[image_obj] if image_obj is not None else None,
                return_tensors="pt",
                padding=True,
            )
        except (TypeError, ValueError):
            inputs = apply(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            rendered = apply(messages, tokenize=False, add_generation_prompt=True)
    elif tokenizer is not None:
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            rendered = apply(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            )
        else:
            rendered = prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    else:
        raise RuntimeError("model exposes neither processor nor tokenizer")
    if hasattr(inputs, "items"):
        inputs = dict(inputs.items())
    return str(rendered), inputs


def _find_first_language_layer(model: Any) -> tuple[str, Any] | tuple[None, None]:
    root = getattr(model, "model", model)
    candidates = []
    try:
        candidates = list(root.named_modules())
    except Exception:
        return None, None
    for name, module in candidates:
        norm = name.replace(".", "/")
        if (
            "language_model/layers/0" in norm or "language_model/model/layers/0" in norm
        ) and name:
            return name, module
    for name, module in candidates:
        if name.endswith("layers.0") and name:
            return name, module
    return None, None


def _forward_logits(model: Any, inputs: Mapping[str, Any]) -> Any:
    with __import__("torch").inference_mode():
        output = getattr(model, "model", model)(**inputs, return_dict=True)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None:
        raise RuntimeError("model forward returned no logits")
    return logits[..., -1, :].detach().float().cpu()


def _logit_metrics(
    candidate: Sequence[Sequence[float]], reference: Sequence[Sequence[float]] | None
) -> dict[str, Any]:
    import torch

    cand = torch.tensor(candidate, dtype=torch.float32)
    if not torch.isfinite(cand).all():
        return {
            "finite": False,
            "reference_available": reference is not None,
            "cosine": None,
            "mse": None,
            "kl": None,
        }
    if reference is None:
        return {
            "finite": True,
            "reference_available": False,
            "cosine": None,
            "mse": None,
            "kl": None,
        }
    ref = torch.tensor(reference, dtype=torch.float32)
    if cand.shape != ref.shape or not torch.isfinite(ref).all():
        raise ValueError(
            f"logit reference shape/finite mismatch: {tuple(cand.shape)} vs {tuple(ref.shape)}"
        )
    cosine = torch.nn.functional.cosine_similarity(cand, ref, dim=-1).mean()
    mse = torch.mean((cand - ref) ** 2)
    p = torch.nn.functional.log_softmax(cand, dim=-1)
    q = torch.nn.functional.softmax(ref, dim=-1)
    kl = torch.nn.functional.kl_div(p, q, reduction="batchmean")
    return {
        "finite": True,
        "reference_available": True,
        "compared_samples": int(cand.shape[0]),
        "cosine": float(cosine),
        "mse": float(mse),
        "kl": float(kl),
    }


def _run_model_cell(
    config: Mapping[str, Any], artifact_root: Path, command: str
) -> dict[str, Any]:
    """Run one real cell. Exceptions are handled by the worker wrapper."""

    cell_start = time.perf_counter()
    _set_seed(int(config["seed"]))
    images = list(config.get("images", []))
    inference_multimodal = (
        config.get("inference_modality", "multimodal") == "multimodal"
    )
    if inference_multimodal:
        _validate_runtime_images(images)
    import torch
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig

    cell = str(config["cell"])
    calibration_multimodal = config["calibration_modality"] == "multimodal"
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code")),
        "dtype": config["inference"].get("dtype", "auto"),
        "device_map": config["inference"].get("device_map", "auto"),
    }
    if config.get("revision"):
        kwargs["revision"] = config["revision"]
    qcfg = None
    if config["method"]:
        method = METHOD.GPTQ if config["method"] == "gptq" else METHOD.AWQ
        qcfg = QuantizeConfig(
            method=method,
            format=FORMAT.GPTQ if method == METHOD.GPTQ else FORMAT.GEMM,
            bits=config["quant"]["bits"],
            group_size=config["quant"]["group_size"],
            sym=config["quant"]["sym"],
            **config["quant"].get("params", {}),
        )
    if qcfg is not None and config["debug"].get("max_quant_layers") is not None:
        from transformers import AutoConfig

        native_config = AutoConfig.from_pretrained(
            config["model_path"],
            revision=config.get("revision"),
            trust_remote_code=bool(config.get("trust_remote_code")),
        )
        text_config = getattr(native_config, "text_config", native_config)
        layer_count = int(text_config.num_hidden_layers)
        layer_limit = int(config["debug"]["max_quant_layers"])
        if layer_limit <= 0 or layer_limit > layer_count:
            raise ValueError(
                f"max_quant_layers must be in [1, {layer_count}], got {layer_limit}"
            )
        qcfg.dynamic = {
            rf"-:^model\.language_model\.layers\.{index}\.": {}
            for index in range(layer_limit, layer_count)
        }

    load_start = time.perf_counter()
    if qcfg is None:
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        hf_model = AutoModelForImageTextToText.from_pretrained(
            config["model_path"], **kwargs
        ).eval()
        model = SimpleNamespace(
            model=hf_model,
            processor=AutoProcessor.from_pretrained(
                config["model_path"],
                revision=config.get("revision"),
                trust_remote_code=bool(config.get("trust_remote_code")),
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                config["model_path"],
                revision=config.get("revision"),
                trust_remote_code=bool(config.get("trust_remote_code")),
            ),
        )
    else:
        model = GPTQModel.load(config["model_path"], quantize_config=qcfg, **kwargs)
    startup_s = time.perf_counter() - load_start
    prompts = list(config["prompts"])
    calibration = []
    for index, prompt in enumerate(
        prompts[: int(config["quant"].get("calibration_rows") or len(prompts))]
    ):
        calibration.append(
            _conversation(
                prompt, images[index % len(images)] if calibration_multimodal else None
            )
        )
    quantization_s = 0.0
    save_reload_s = 0.0
    if qcfg is not None:
        from gptqmodel.looper.module_looper import StopMainLoop

        stop_after_layer = config["debug"].get("stop_after_layer")
        if stop_after_layer is not None:
            target = int(stop_after_layer)

            class Stopper:
                def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
                    if submodule_finalized and layer_idx >= target:
                        raise StopMainLoop

            model.layer_callback = Stopper()
        quantization_start = time.perf_counter()
        try:
            model.quantize(
                calibration,
                calibration_concat_size=config["quant"].get("calibration_concat_size"),
                batch_size=config["quant"].get("batch_size", 1),
            )
        except StopMainLoop as exc:
            raise RuntimeError(
                "debug quantization stop requested; no complete logits manifest produced"
            ) from exc
        quantization_s = time.perf_counter() - quantization_start
        output_root = config.get("protocol", {}).get("quant_output_root")
        if output_root:
            save_reload_start = time.perf_counter()
            from gptqmodel import BACKEND

            save_dir = Path(output_root) / cell
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(save_dir))
            reload_backend = (
                BACKEND.GPTQ_TORCH if config["method"] == "gptq" else BACKEND.AWQ_TORCH
            )
            model = GPTQModel.load(
                str(save_dir),
                backend=reload_backend,
                trust_remote_code=bool(config.get("trust_remote_code")),
                dtype=config["inference"].get("dtype", "auto"),
                device_map=config["inference"].get("device_map", "auto"),
            )
            save_reload_s = time.perf_counter() - save_reload_start
    layer_name, layer = _find_first_language_layer(model)
    captured: list[Any] = []
    hook = None
    if layer is not None:

        def capture(_module, inputs):
            value = inputs[0] if inputs else None
            if torch.is_tensor(value):
                captured.append(value.detach().float().cpu())

        hook = layer.register_forward_pre_hook(capture)
    rendered_samples = []
    logits = []
    task_start = time.perf_counter()
    try:
        for index, prompt in enumerate(prompts):
            image = images[index % len(images)] if inference_multimodal else None
            rendered, inputs = _prepare_sample(
                model, prompt, image, inference_multimodal
            )
            moved = _move(inputs, _first_device(getattr(model, "model", model)))
            output = _forward_logits(model, moved)
            logits.append(output[0].tolist())
            rendered_samples.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "image": image,
                    "rendered_prompt": rendered,
                    "input_ids": inputs.get("input_ids").detach().cpu().tolist()
                    if torch.is_tensor(inputs.get("input_ids"))
                    else inputs.get("input_ids"),
                    "inputs": summarize_inputs(inputs),
                    "pixel_values": summarize_inputs(
                        {"pixel_values": inputs.get("pixel_values")}
                    ).get("pixel_values")
                    if inputs.get("pixel_values") is not None
                    else None,
                    "image_grid": summarize_inputs(
                        {"image_grid_thw": inputs.get("image_grid_thw")}
                    ).get("image_grid_thw")
                    if inputs.get("image_grid_thw") is not None
                    else None,
                }
            )
    finally:
        if hook is not None:
            hook.remove()
    task_s = time.perf_counter() - task_start
    reference = None
    if cell != "dense":
        dense_raw = artifact_root / "raw" / "dense.json"
        if dense_raw.is_file():
            reference = json.loads(dense_raw.read_text(encoding="utf-8")).get("logits")
    metrics = _logit_metrics(logits, reference)
    if not metrics["finite"] or (
        cell != "dense" and not metrics["reference_available"]
    ):
        raise RuntimeError(
            "candidate logits are not finite or dense reference is unavailable"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "engine": cell,
        "model": config["model_path"],
        "revision": config.get("revision"),
        "protocol_hash": config["protocol_hash"],
        "success": True,
        "rendered_samples": rendered_samples,
        "logits": logits,
        "first_language_layer": {
            "name": layer_name,
            "input": tensor_summary(captured[0]) if captured else None,
        },
        "logit_metrics": metrics,
        "metric": "final_logits_cosine_to_dense",
        "score": 1.0 if cell == "dense" else metrics["cosine"],
        "samples": len(prompts),
        "expected_samples": len(prompts),
        "timing": {
            "startup_seconds": startup_s,
            "quantization_seconds": quantization_s,
            "save_reload_seconds": save_reload_s,
            "task_seconds": task_s,
            "wall_seconds": time.perf_counter() - cell_start,
        },
        "versions": {
            "evaluator": SCRIPT_VERSION,
            "engine": (
                getattr(__import__("transformers"), "__version__", "unknown")
                if cell == "dense"
                else getattr(
                    __import__("gptqmodel"), "__version__", "GPT-QModel-worktree"
                )
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "gptqmodel": getattr(__import__("gptqmodel"), "__version__", "unknown"),
        },
        "gpus": _gpu_info(),
        "command": command,
    }


def _worker(config_path: Path, artifact_root: Path, command: str) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cell = str(config["cell"])
    stem = _slug(cell)
    log_path = artifact_root / "logs" / f"{stem}.log"
    raw_path = artifact_root / "raw" / f"{stem}.json"
    invalid_path = (
        artifact_root / "invalid_attempts" / f"{stem}__{int(time.time())}.json"
    )
    try:
        if config.get("inference_modality", "multimodal") == "multimodal":
            _validate_runtime_images(list(config.get("images", [])))
        result = _run_model_cell(config, artifact_root, command)
        result.update(
            {
                "raw_result": str(raw_path),
                "config": str(config_path),
                "log": str(log_path),
            }
        )
        raw_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "success": True,
            "engine": cell,
            "model": config["model_path"],
            "task": "qwen35_moe_multimodal_calibration",
            "metric": result["metric"],
            "score": result["score"],
            "samples": result["samples"],
            "expected_samples": result["expected_samples"],
            "protocol_hash": config["protocol_hash"],
            "timing": {
                "launch_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **result["timing"],
            },
            "versions": result["versions"],
            "gpus": result["gpus"],
            "command": command,
            "config": str(config_path),
            "raw_result": str(raw_path),
            "log": str(log_path),
        }
        errors = validate_manifest(manifest)
        if errors:
            raise RuntimeError("manifest validation failed: " + ",".join(errors))
        (artifact_root / "manifests" / f"{stem}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "success": False,
            "cell": cell,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "command": command,
            "config": str(config_path),
            "log": str(log_path),
        }
        invalid_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/monster/data/model/Qwen3.5-35B-A3B")
    parser.add_argument("--revision", "--model-revision", default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/qwen35_moe_multimodal_calibration"),
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="comma-separated subset of dense,gptq-text,gptq-multimodal,awq-text,awq-multimodal",
    )
    parser.add_argument(
        "--prompts", help="JSON list, comma-separated list, or text/JSON file"
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--images", help="JSON list, comma-separated list, or text/JSON file"
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="one existing image path (repeatable; required unless --dry-run)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--quant-param", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--inference-param", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--calibration-rows", type=int, default=2)
    parser.add_argument("--calibration-concat-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-quant-layers", type=int, default=None)
    parser.add_argument("--stop-after-layer", type=int, default=None)
    parser.add_argument("--quant-output-root", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze configs and provenance without starting workers",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--_worker-config", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv if argv is not None else sys.argv[1:]),
        ]
    )
    root = args.artifact_root.resolve()
    if args._worker_config:
        return _worker(args._worker_config.resolve(), root, command)
    try:
        configs = make_cell_configs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    prepare_artifacts(root, configs, Path(__file__).resolve().parents[1], command)
    statuses: dict[str, str] = {}
    if not args.dry_run:
        for config in configs:
            cell = str(config["cell"])
            stem = _slug(cell)
            manifest_path = root / "manifests" / f"{stem}.json"
            if not args.no_resume and manifest_path.is_file():
                try:
                    old = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if (
                        is_valid_manifest(old)
                        and old.get("protocol_hash") == config["protocol_hash"]
                    ):
                        statuses[cell] = "resumed"
                        continue
                except Exception:
                    pass
                stale = (
                    root / "invalid_attempts" / f"{stem}__stale-{int(time.time())}.json"
                )
                manifest_path.replace(stale)
            config_path = root / "configs" / f"{stem}.json"
            log_path = root / "logs" / f"{stem}.log"
            worker_command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--artifact-root",
                str(root),
                "--_worker-config",
                str(config_path),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    worker_command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            statuses[cell] = "accepted" if completed.returncode == 0 else "failed"
    prepare_artifacts(root, configs, Path(__file__).resolve().parents[1], command)
    _write_report(root, configs, statuses, command)
    print(
        json.dumps(
            {
                "artifact_root": str(root),
                "protocol_hash": configs[0]["protocol_hash"],
                "cells": statuses or {c["cell"]: "planned" for c in configs},
            },
            indent=2,
        )
    )
    return (
        0
        if args.dry_run
        or all(status in {"accepted", "resumed"} for status in statuses.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
