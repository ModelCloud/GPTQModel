#!/usr/bin/env python3
"""Compare QVQ trellis topologies on full-width Llama attention blocks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from analyze_gptq_low_bit_grid import (
        capture_calibration_hessians,
        capture_forward,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        request_performance_qos,
        tensor_metrics,
    )
except ModuleNotFoundError:  # Package import used by unit tests.
    from scripts.analyze_gptq_low_bit_grid import (
        capture_calibration_hessians,
        capture_forward,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        request_performance_qos,
        tensor_metrics,
    )
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import (
    QVQLinearQuantizationResult,
    QVQQuantizationTelemetry,
    default_qvq_trellis_batch_size,
    prepare_qvq_input_hessian,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION
from gptqmodel.quantization.qvq_rates import normalize_qvq_rate
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b

QKVO_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
MLP_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
MODULE_SCOPE_SUFFIXES = {
    "qkvo": QKVO_SUFFIXES,
    "all-linear": (*QKVO_SUFFIXES, *MLP_SUFFIXES),
}
_QVQ_PREFIX_MANIFEST_KEY = "qvq_prefix_manifest"
_QVQ_PREFIX_SCHEMA_VERSION = 1
_QVQ_PREFIX_TENSOR_NAMES = frozenset(("trellis", "SU", "SV", "bias", "bank_ids", "bank_alt_id"))
ARM_CONFIG = {
    "v2": {"vector_size": 2, "trellis_window": 16, "dual_v2": False},
    "v2b2-p32": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
    },
    "v2b4-p64": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b4_p64": True,
        "bank_count": 4,
    },
    "dual-v2": {"vector_size": 2, "trellis_window": 16, "dual_v2": True},
    "v4": {"vector_size": 4, "trellis_window": 16, "dual_v2": False},
    "l18-v4": {"vector_size": 4, "trellis_window": 18, "dual_v2": False},
    "v2-yaqa": {"vector_size": 2, "trellis_window": 16, "dual_v2": False, "rounding": "yaqa"},
    "v2b2-p32-yaqa-fixed": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "fixed_block_ldlq",
    },
    "v2b2-p32-yaqa": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "reselect",
    },
    "v2b2-p32-yaqa-spectral": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "reselect",
        "yaqa_spectral_refinement": True,
    },
    "v2b2-p32-yaqa-spectral-fixed": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "fixed_block_ldlq",
        "yaqa_spectral_refinement": True,
    },
    "v2b2-p32-yaqa-spectral-push-fixed": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "fixed_block_ldlq",
        "yaqa_spectral_push": True,
    },
    "v2b2-p32-yaqa-spectral-push": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "reselect",
        "yaqa_spectral_push": True,
    },
}
DEFAULT_ARMS = ("v2", "v2b2-p32")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--module-scope",
        choices=tuple(MODULE_SCOPE_SUFFIXES),
        default="qkvo",
        help="Quantize attention Q/K/V/O only, or every Q/K/V/O and gate/up/down projection.",
    )
    parser.add_argument(
        "--all-linear-hessian-mode",
        choices=("staged", "layerwise", "dense-frozen"),
        default="staged",
        help=(
            "For all-linear sweeps, recapture downstream geometry in four global producer/consumer stages, "
            "replay every layer exactly, or retain the legacy dense-frozen diagnostic control."
        ),
    )
    parser.add_argument("--rates", nargs="+", type=float, default=(1, 1.5, 2, 2.5))
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_CONFIG), default=DEFAULT_ARMS)
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--evaluation-rows", type=int, default=64)
    parser.add_argument("--evaluation-row-offset", type=int, default=64)
    parser.add_argument("--yaqa-rows", type=int, default=512)
    parser.add_argument("--yaqa-row-offset", type=int)
    parser.add_argument("--yaqa-batch-size", type=int, default=8)
    parser.add_argument("--yaqa-seed", type=int, default=0)
    parser.add_argument("--yaqa-spectral-ranks", nargs="+", type=int, default=(8, 16, 32))
    parser.add_argument("--yaqa-spectral-lambdas", nargs="+", type=float, default=(0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--yaqa-spectral-push-alphas", nargs="+", type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument(
        "--yaqa-factor-cache",
        type=Path,
        help="Optional validated CPU cache shared by matched YAQA rate workers.",
    )
    parser.add_argument(
        "--prepare-yaqa-only",
        action="store_true",
        help="Collect and cache Sketch-B factors, then exit before Hessian capture and quantization.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional per-row truncation limit; omitted means each row's full tokenized length.",
    )
    parser.add_argument("--seed", type=int, default=18240)
    parser.add_argument("--trellis-batch-size", type=int)
    parser.add_argument(
        "--qvq-telemetry",
        action="store_true",
        help="Record nested QVQ phase timing and shape counters for each quantized module.",
    )
    return parser


def _padded_batch_chunks(
    encoded: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Split padded rows into batches while trimming padding-only edge columns."""

    attention_mask = encoded.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError("YAQA encoding must contain a rank-2 attention mask")
    if batch_size < 1:
        raise ValueError("YAQA batch size must be positive")
    batches = []
    for start in range(0, attention_mask.shape[0], batch_size):
        stop = min(start + batch_size, attention_mask.shape[0])
        mask = attention_mask[start:stop]
        active_columns = mask.ne(0).any(dim=0).nonzero(as_tuple=False).flatten()
        if active_columns.numel() == 0:
            raise ValueError("YAQA batch contains no valid tokens")
        column_start = int(active_columns[0])
        column_stop = int(active_columns[-1]) + 1
        batch = {}
        for name, value in encoded.items():
            if value.ndim >= 2 and tuple(value.shape[:2]) == tuple(attention_mask.shape):
                batch[name] = value[start:stop, column_start:column_stop].contiguous()
            else:
                batch[name] = value[start:stop].contiguous()
        batches.append(batch)
    return batches


def _yaqa_cache_metadata(args: argparse.Namespace, module_shapes: dict[str, list[int]], row_offset: int) -> dict:
    metadata = {
        "version": 1,
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "layers": args.layers,
        "module_shapes": module_shapes,
        "rows": args.yaqa_rows,
        "row_offset": row_offset,
        "batch_size": args.yaqa_batch_size,
        "seed": args.yaqa_seed,
        "max_length": args.max_length,
    }
    # Preserve compatibility with existing QKVO caches while making expanded
    # all-linear caches self-describing and impossible to reuse accidentally.
    if args.module_scope != "qkvo":
        metadata["version"] = 2
        metadata["module_scope"] = args.module_scope
    return metadata


def _load_yaqa_factor_cache(
    path: Path,
    *,
    expected_metadata: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        raise ValueError(f"YAQA factor cache metadata does not match this sweep: {path}")
    input_hessians = payload.get("input_hessians")
    output_hessians = payload.get("output_hessians")
    stats = payload.get("stats")
    expected_names = set(expected_metadata["module_shapes"])
    if not isinstance(input_hessians, dict) or not isinstance(output_hessians, dict) or not isinstance(stats, dict):
        raise ValueError(f"YAQA factor cache has an invalid payload: {path}")
    if set(input_hessians) != expected_names or set(output_hessians) != expected_names:
        raise ValueError(f"YAQA factor cache module names do not match this sweep: {path}")
    for name, shape in expected_metadata["module_shapes"].items():
        out_features, in_features = shape
        input_factor = input_hessians[name]
        output_factor = output_hessians[name]
        if (
            not isinstance(input_factor, torch.Tensor)
            or input_factor.device.type != "cpu"
            or input_factor.dtype != torch.float32
            or tuple(input_factor.shape) != (in_features, in_features)
            or not input_factor.is_contiguous()
        ):
            raise ValueError(f"YAQA input factor {name} has invalid storage or geometry")
        if (
            not isinstance(output_factor, torch.Tensor)
            or output_factor.device.type != "cpu"
            or output_factor.dtype != torch.float32
            or tuple(output_factor.shape) != (out_features, out_features)
            or not output_factor.is_contiguous()
        ):
            raise ValueError(f"YAQA output factor {name} has invalid storage or geometry")
    return input_hessians, output_hessians, stats


def _save_yaqa_factor_cache(
    path: Path,
    *,
    metadata: dict,
    input_hessians: dict[str, torch.Tensor],
    output_hessians: dict[str, torch.Tensor],
    stats: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(
        {
            "metadata": metadata,
            "input_hessians": input_hessians,
            "output_hessians": output_hessians,
            "stats": stats,
        },
        temporary,
    )
    temporary.replace(path)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one canonical CPU tensor without changing its dtype or shape."""

    owned = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(owned.view(torch.uint8).numpy().tobytes()).hexdigest()


def _save_qvq_prefix_artifact(
    path: Path,
    *,
    module_results: Mapping[str, QVQLinearQuantizationResult],
    bits: float,
    provenance: Mapping[str, object],
    codebook_version: str = PGC16_CODEBOOK_VERSION,
) -> dict[str, object]:
    """Atomically save a reusable packed V2B2-P32 prefix without dense weights."""

    normalized_bits = normalize_qvq_rate(bits)
    if normalized_bits > 3.5:
        raise ValueError("QVQ V2B2-P32 prefix artifacts support W1 through W3.5")
    if not module_results:
        raise ValueError("QVQ prefix artifacts require at least one quantized module")
    if not isinstance(provenance, Mapping):
        raise TypeError("QVQ prefix artifact provenance must be a mapping")
    try:
        normalized_provenance = json.loads(json.dumps(dict(provenance), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise TypeError("QVQ prefix artifact provenance must be JSON serializable") from error

    stored: dict[str, torch.Tensor] = {}
    module_manifest: dict[str, object] = {}
    for module_name, result in sorted(module_results.items()):
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("QVQ prefix artifact module names must be non-empty strings")
        if not isinstance(result, QVQLinearQuantizationResult):
            raise TypeError(f"QVQ prefix artifact module `{module_name}` has an invalid quantization result")
        payload = result.serialized_tensors()
        unexpected = set(payload) - _QVQ_PREFIX_TENSOR_NAMES
        required = {"trellis", "SU", "SV", "bank_ids", "bank_alt_id"}
        if unexpected or not required.issubset(payload):
            raise ValueError(
                f"QVQ prefix artifact module `{module_name}` has invalid tensors: "
                f"missing={sorted(required - set(payload))}, unexpected={sorted(unexpected)}"
            )
        in_features = int(result.SU.numel())
        out_features = int(result.SV.numel())
        tensor_manifest = {}
        owned_payload = {}
        for tensor_name, tensor in sorted(payload.items()):
            owned = tensor.detach().to(device="cpu").contiguous()
            if owned.is_floating_point() and not torch.isfinite(owned).all():
                raise ValueError(f"QVQ prefix artifact tensor `{module_name}.{tensor_name}` is non-finite")
            storage_name = f"{module_name}.{tensor_name}"
            stored[storage_name] = owned
            owned_payload[tensor_name] = owned
            tensor_manifest[tensor_name] = {
                "dtype": str(owned.dtype),
                "shape": list(owned.shape),
                "sha256": _tensor_sha256(owned),
            }
        # Validate the exact packed ownership boundary before publishing any
        # bytes. This catches malformed trellis/selector geometry even when a
        # caller manually constructs a quantization-result dataclass.
        QVQLinear(
            bits=normalized_bits,
            in_features=in_features,
            out_features=out_features,
            tensors={name: tensor.clone() for name, tensor in owned_payload.items()},
            vector_size=2,
            trellis_window=16,
            bank_count=2,
            v2b2_p32=True,
        )
        module_manifest[module_name] = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": "bias" in payload,
            "tensors": tensor_manifest,
        }

    manifest = {
        "schema_version": _QVQ_PREFIX_SCHEMA_VERSION,
        "format": "qvq_v2b2_p32",
        "bits": normalized_bits,
        "vector_size": 2,
        "trellis_window": 16,
        "bank_count": 2,
        "codebook_version": str(codebook_version),
        "provenance": normalized_provenance,
        "modules": module_manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.safetensors")
    try:
        save_file(
            stored,
            str(temporary),
            metadata={_QVQ_PREFIX_MANIFEST_KEY: json.dumps(manifest, sort_keys=True, separators=(",", ":"))},
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _load_qvq_prefix_artifact(
    path: Path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, torch.Tensor]]]:
    """Load and fully validate one packed V2B2-P32 prefix artifact on CPU."""

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        encoded_manifest = (handle.metadata() or {}).get(_QVQ_PREFIX_MANIFEST_KEY)
        if encoded_manifest is None:
            raise ValueError(f"QVQ prefix artifact has no manifest: {path}")
        try:
            manifest = json.loads(encoded_manifest)
        except json.JSONDecodeError as error:
            raise ValueError(f"QVQ prefix artifact manifest is invalid JSON: {path}") from error
        if not isinstance(manifest, dict):
            raise TypeError(f"QVQ prefix artifact manifest must be an object: {path}")
        fixed_contract = {
            "schema_version": _QVQ_PREFIX_SCHEMA_VERSION,
            "format": "qvq_v2b2_p32",
            "vector_size": 2,
            "trellis_window": 16,
            "bank_count": 2,
        }
        for field, expected in fixed_contract.items():
            if manifest.get(field) != expected:
                raise ValueError(f"QVQ prefix artifact has unsupported `{field}`: {manifest.get(field)!r}")
        bits = normalize_qvq_rate(manifest.get("bits"))
        if bits > 3.5:
            raise ValueError("QVQ prefix artifact rate exceeds the V2B2-P32 contract")
        if expected_provenance is not None:
            try:
                normalized_expected_provenance = json.loads(json.dumps(dict(expected_provenance), sort_keys=True))
            except (TypeError, ValueError) as error:
                raise TypeError("expected QVQ prefix provenance must be JSON serializable") from error
            if manifest.get("provenance") != normalized_expected_provenance:
                raise ValueError(f"QVQ prefix artifact provenance does not match this run: {path}")
        modules = manifest.get("modules")
        if not isinstance(modules, dict) or not modules:
            raise ValueError(f"QVQ prefix artifact has no module manifest: {path}")

        storage_names = set(handle.keys())
        expected_storage_names = set()
        loaded: dict[str, dict[str, torch.Tensor]] = {}
        for module_name, module_spec in modules.items():
            if not isinstance(module_name, str) or not isinstance(module_spec, dict):
                raise TypeError("QVQ prefix artifact module manifest is malformed")
            in_features = module_spec.get("in_features")
            out_features = module_spec.get("out_features")
            if (
                isinstance(in_features, bool)
                or not isinstance(in_features, int)
                or in_features < 16
                or in_features % 16
                or isinstance(out_features, bool)
                or not isinstance(out_features, int)
                or out_features < 16
                or out_features % 16
                or not isinstance(module_spec.get("bias"), bool)
            ):
                raise ValueError(f"QVQ prefix artifact module `{module_name}` has invalid geometry metadata")
            tensor_specs = module_spec.get("tensors")
            if not isinstance(tensor_specs, dict):
                raise TypeError(f"QVQ prefix artifact module `{module_name}` has no tensor manifest")
            required = {"trellis", "SU", "SV", "bank_ids", "bank_alt_id"}
            if not required.issubset(tensor_specs) or set(tensor_specs) - _QVQ_PREFIX_TENSOR_NAMES:
                raise ValueError(f"QVQ prefix artifact module `{module_name}` has an invalid tensor set")
            loaded[module_name] = {}
            for tensor_name, tensor_spec in tensor_specs.items():
                storage_name = f"{module_name}.{tensor_name}"
                expected_storage_names.add(storage_name)
                if not isinstance(tensor_spec, dict) or storage_name not in storage_names:
                    raise ValueError(f"QVQ prefix artifact tensor `{storage_name}` is missing or malformed")
                tensor = handle.get_tensor(storage_name).contiguous()
                if str(tensor.dtype) != tensor_spec.get("dtype") or list(tensor.shape) != tensor_spec.get("shape"):
                    raise ValueError(f"QVQ prefix artifact tensor `{storage_name}` violates its dtype/shape manifest")
                if _tensor_sha256(tensor) != tensor_spec.get("sha256"):
                    raise ValueError(f"QVQ prefix artifact tensor `{storage_name}` failed checksum validation")
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"QVQ prefix artifact tensor `{storage_name}` is non-finite")
                loaded[module_name][tensor_name] = tensor
        if storage_names != expected_storage_names:
            raise ValueError("QVQ prefix artifact contains unexpected serialized tensors")
    return manifest, loaded


def _install_qvq_prefix_artifact(
    model: torch.nn.Module,
    *,
    manifest: Mapping[str, object],
    module_tensors: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, QVQLinear]:
    """Transactionally replace matching dense modules with validated packed QVQLinear modules."""

    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or set(modules) != set(module_tensors):
        raise ValueError("QVQ prefix artifact module payload does not match its manifest")
    bits = normalize_qvq_rate(manifest.get("bits"))
    replacements = {}
    for module_name, module_spec in modules.items():
        current = model.get_submodule(module_name)
        if not isinstance(current, torch.nn.Linear):
            raise TypeError(f"QVQ prefix target `{module_name}` is not a dense torch.nn.Linear")
        if (
            not isinstance(module_spec, Mapping)
            or current.in_features != module_spec.get("in_features")
            or current.out_features != module_spec.get("out_features")
            or (current.bias is not None) != module_spec.get("bias")
        ):
            raise ValueError(f"QVQ prefix target `{module_name}` does not match the serialized geometry")
        device = current.weight.device
        dtype = current.weight.dtype
        tensors = {name: tensor.to(device=device) for name, tensor in module_tensors[module_name].items()}
        replacement = QVQLinear(
            bits=bits,
            in_features=current.in_features,
            out_features=current.out_features,
            name=module_name,
            tensors=tensors,
            dtype=dtype,
            out_dtype=dtype,
            codebook_version=str(manifest.get("codebook_version")),
            vector_size=2,
            trellis_window=16,
            bank_count=2,
            v2b2_p32=True,
        ).train(current.training)
        replacement.post_init()
        replacements[module_name] = replacement

    for module_name, replacement in replacements.items():
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
    return replacements


def _quantized_linear_modules(
    model: torch.nn.Module,
    *,
    layer_count: int,
    module_scope: str,
) -> dict[str, torch.nn.Linear]:
    """Select the exact decoder projection set requested by the comparison run."""

    try:
        suffixes = MODULE_SCOPE_SUFFIXES[module_scope]
    except KeyError as error:
        raise ValueError(f"unsupported module scope: {module_scope}") from error
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(suffixes)
    }
    expected = layer_count * len(suffixes)
    if len(modules) != expected:
        raise ValueError(
            f"expected {expected} {module_scope} modules in {layer_count} decoder layers, found {tuple(modules)}"
        )
    return modules


def _qkvo_modules(model: torch.nn.Module, *, layer_count: int) -> dict[str, torch.nn.Linear]:
    """Retain the original QKVO-only selector for callers importing this helper."""

    return _quantized_linear_modules(model, layer_count=layer_count, module_scope="qkvo")


def _shared_input_hessian_groups(
    model: torch.nn.Module,
    selected_modules: dict[str, torch.nn.Linear],
) -> tuple[tuple[str, ...], ...]:
    """Discover same-input projection groups from the instantiated module tree."""

    selected_names_by_id = {id(module): name for name, module in selected_modules.items()}
    groups = []
    seen: set[int] = set()
    for parent in model.modules():
        children_by_role = dict(parent.named_children())
        for child_roles in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
            children = tuple(children_by_role.get(role) for role in child_roles)
            if any(not isinstance(child, torch.nn.Linear) for child in children):
                continue
            child_ids = tuple(id(child) for child in children)
            if any(child_id not in selected_names_by_id for child_id in child_ids):
                continue
            if len(set(child_ids)) != len(child_ids) or seen.intersection(child_ids):
                raise ValueError("shared-input projection groups must contain distinct, disjoint modules")
            if len({child.in_features for child in children}) != 1:
                raise ValueError("shared-input projection groups must have one input width")
            groups.append(tuple(selected_names_by_id[child_id] for child_id in child_ids))
            seen.update(child_ids)
    return tuple(groups)


def _all_linear_dependency_stages(
    model: torch.nn.Module,
    selected_modules: dict[str, torch.nn.Linear],
    *,
    layerwise: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """Return Llama-style producer/consumer stages from the instantiated module tree.

    Q/K/V share the layer input, O consumes their attention result, gate/up
    share the post-attention input, and down consumes the gated product.  The
    stages are global across decoder layers so each calibration replay updates
    every consumer from all already-installed producer reconstructions.
    """

    selected_names_by_id = {id(module): name for name, module in selected_modules.items()}
    stages: list[list[str]] = [] if layerwise else [[], [], [], []]
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("all-linear staged replay requires decoder layers at model.layers")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        role_groups = (
            (attention, ("q_proj", "k_proj", "v_proj")),
            (attention, ("o_proj",)),
            (mlp, ("gate_proj", "up_proj")),
            (mlp, ("down_proj",)),
        )
        layer_stages: list[list[str]] = [[], [], [], []] if layerwise else stages
        for stage, (parent, roles) in zip(layer_stages, role_groups, strict=True):
            if not isinstance(parent, torch.nn.Module):
                raise ValueError("all-linear staged replay requires self_attn and mlp module groups")
            for role in roles:
                child = getattr(parent, role, None)
                name = selected_names_by_id.get(id(child))
                if not isinstance(child, torch.nn.Linear) or name is None:
                    raise ValueError(f"all-linear staged replay is missing selected projection `{role}`")
                stage.append(name)
        if layerwise:
            stages.extend(layer_stages)
    flattened = tuple(name for stage in stages for name in stage)
    if len(flattened) != len(set(flattened)) or set(flattened) != set(selected_modules):
        raise ValueError("all-linear staged replay did not cover the selected projections exactly once")
    return tuple(tuple(stage) for stage in stages)


def _joined(outputs: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    rows = {outputs[name].shape[0] for name in names}
    if len(rows) != 1:
        raise ValueError("selected module outputs do not share one held-out token geometry")
    return torch.cat([outputs[name] for name in names], dim=-1)


def _weight_metrics(dense: torch.Tensor, quantized: torch.Tensor) -> dict[str, float]:
    error = quantized.double() - dense.double()
    dense_norm = dense.double().norm().clamp_min(torch.finfo(torch.float64).eps)
    error_norm = error.norm()
    return {
        "mse": error.square().mean().item(),
        "relative_l2": (error_norm / dense_norm).item(),
        "sqnr_db": (20 * torch.log10(dense_norm / error_norm.clamp_min(torch.finfo(torch.float64).eps))).item(),
    }


def _aggregate_qvq_telemetry(module_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    """Aggregate already-finalized per-module telemetry without another synchronization."""

    phases: dict[str, dict[str, float | int | None]] = {}
    counters: Counter[str] = Counter()
    for metrics in module_metrics.values():
        telemetry = metrics.get("qvq_telemetry")
        if not isinstance(telemetry, dict):
            continue
        counters.update(telemetry.get("counters", {}))
        for name, values in telemetry.get("phases", {}).items():
            aggregate = phases.setdefault(name, {"calls": 0, "host_dispatch_ms": 0.0, "gpu_ms": 0.0})
            aggregate["calls"] += int(values["calls"])
            aggregate["host_dispatch_ms"] += float(values["host_dispatch_ms"])
            if values["gpu_ms"] is None:
                aggregate["gpu_ms"] = None
            elif aggregate["gpu_ms"] is not None:
                aggregate["gpu_ms"] += float(values["gpu_ms"])
    return {"phases": phases, "counters": dict(counters)}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _selector_metrics(histogram: list[int]) -> dict[str, object] | None:
    total = sum(histogram)
    if total == 0:
        return None
    probabilities = [count / total for count in histogram if count]
    return {
        "count": total,
        "histogram": histogram,
        "nonzero_fraction": sum(histogram[1:]) / total,
        "entropy_bits": -sum(probability * math.log2(probability) for probability in probabilities),
        "selector_bpw": 2 / 64,
    }


def _unpadded_evaluation_rows(
    encoded: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], ...]:
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("evaluation encoding must contain a rank-2 attention mask")
    rows = []
    for row_index in range(attention_mask.shape[0]):
        valid = attention_mask[row_index].ne(0).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError(f"evaluation row {row_index} contains no valid tokens")
        start = int(valid[0])
        stop = int(valid[-1]) + 1
        row = {}
        for name, value in encoded.items():
            if value.ndim >= 2 and tuple(value.shape[:2]) == tuple(attention_mask.shape):
                row[name] = value[row_index : row_index + 1, start:stop].to(device)
            else:
                row[name] = value[row_index : row_index + 1].to(device)
        if not bool(row["attention_mask"].ne(0).all()):
            raise ValueError(f"evaluation row {row_index} was not fully unpadded")
        rows.append(row)
    return tuple(rows)


@torch.inference_mode()
def _capture_forward_rows(
    model: torch.nn.Module,
    rows: tuple[dict[str, torch.Tensor], ...],
    modules: dict[str, torch.nn.Linear],
    *,
    capture_inputs: bool,
    layer_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    logits = []
    inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        row_logits, row_inputs, row_outputs = capture_forward(
            model,
            row,
            modules,
            capture_inputs=capture_inputs,
            layer_count=layer_count,
        )
        logits.append(row_logits)
        for name, value in row_inputs.items():
            inputs[name].append(value)
        for name, value in row_outputs.items():
            outputs[name].append(value)
    return (
        torch.cat(logits, dim=0),
        {name: torch.cat(values, dim=0) for name, values in inputs.items()},
        {name: torch.cat(values, dim=0) for name, values in outputs.items()},
    )


class _WeightedMetricAccumulator:
    """Merge row metrics without retaining full-vocabulary tensors."""

    def __init__(self) -> None:
        self.weight = 0
        self.template = None
        self.totals: dict[tuple[str, ...], float] = {}
        self.maxima: dict[tuple[str, ...], float] = {}
        self.booleans: dict[tuple[str, ...], bool] = {}
        self.shape: list[int] | None = None

    def add(self, metrics: dict, *, rows: int) -> None:
        if rows < 1:
            raise ValueError("streamed metric rows must be positive")
        if self.template is None:
            self.template = metrics
        self.weight += rows

        def visit(value, path: tuple[str, ...]):
            if isinstance(value, dict):
                for name, child in value.items():
                    visit(child, (*path, name))
            elif path == ("shape",):
                if self.shape is None:
                    self.shape = list(value)
                else:
                    if self.shape[1:] != list(value)[1:]:
                        raise ValueError("streamed metric feature shapes do not match")
                    self.shape[0] += int(value[0])
            elif isinstance(value, bool):
                self.booleans[path] = self.booleans.get(path, True) and value
            elif isinstance(value, (int, float)):
                if path[-1] in {"max", "max_abs_error"}:
                    self.maxima[path] = max(self.maxima.get(path, -math.inf), float(value))
                else:
                    self.totals[path] = self.totals.get(path, 0.0) + float(value) * rows

        visit(metrics, ())

    def result(self) -> dict:
        if self.template is None or self.weight < 1:
            raise ValueError("streamed metric accumulator is empty")

        def build(value, path: tuple[str, ...]):
            if isinstance(value, dict):
                return {name: build(child, (*path, name)) for name, child in value.items()}
            if path == ("shape",):
                return self.shape
            if isinstance(value, bool):
                return self.booleans[path]
            if isinstance(value, (int, float)):
                if path[-1] in {"max", "max_abs_error"}:
                    return self.maxima[path]
                return self.totals[path] / self.weight
            return value

        result = build(self.template, ())
        result["aggregation"] = "exact token-weighted means; row-weighted auxiliary quantiles"
        result["streamed_rows"] = self.weight
        return result

    def mean(self, *path: str) -> float | None:
        key = tuple(path)
        return None if self.weight == 0 or key not in self.totals else self.totals[key] / self.weight


@torch.inference_mode()
def _streaming_compare_models(
    dense_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    rows: tuple[dict[str, torch.Tensor], ...],
    dense_modules: dict[str, torch.nn.Linear],
    quantized_modules: dict[str, torch.nn.Linear],
    reconstructions: dict[str, torch.Tensor],
    *,
    layer_count: int,
    progress_label: str,
    module_scope: str = "qkvo",
) -> dict:
    module_names = tuple(dense_modules)
    local_accumulator = _WeightedMetricAccumulator()
    live_accumulator = _WeightedMetricAccumulator()
    layer_accumulators = {index: _WeightedMetricAccumulator() for index in range(layer_count)}
    logit_accumulator = _WeightedMetricAccumulator()
    for row_index, row in enumerate(rows, start=1):
        dense_logits, dense_inputs, dense_outputs = capture_forward(
            dense_model,
            row,
            dense_modules,
            capture_inputs=True,
            layer_count=layer_count,
        )
        quantized_logits, _, live_outputs = capture_forward(
            quantized_model,
            row,
            quantized_modules,
            capture_inputs=False,
            layer_count=layer_count,
        )
        token_count = dense_logits.shape[0]
        local_outputs = {
            name: F.linear(
                dense_inputs[name],
                reconstructions[name],
                None if dense_modules[name].bias is None else dense_modules[name].bias.detach().cpu().float(),
            )
            for name in module_names
        }
        dense_joined = _joined(dense_outputs, module_names)
        local_accumulator.add(
            tensor_metrics(
                dense_joined,
                _joined(local_outputs, module_names),
                normalize_distribution=True,
            ),
            rows=token_count,
        )
        live_accumulator.add(
            tensor_metrics(
                dense_joined,
                _joined(live_outputs, module_names),
                normalize_distribution=True,
            ),
            rows=token_count,
        )
        for layer_index, accumulator in layer_accumulators.items():
            accumulator.add(
                tensor_metrics(
                    dense_outputs[f"layer.{layer_index}.hidden"],
                    live_outputs[f"layer.{layer_index}.hidden"],
                    normalize_distribution=True,
                ),
                rows=token_count,
            )
        logit_accumulator.add(
            tensor_metrics(
                dense_logits,
                quantized_logits,
                normalize_distribution=False,
                include_top10=True,
            ),
            rows=token_count,
        )
        if row_index % 16 == 0 or row_index == len(rows):
            print(
                f"{progress_label}: eval {row_index}/{len(rows)} rows, "
                f"finalKL={logit_accumulator.mean('kl_forward', 'mean'):.6f} "
                f"top1={logit_accumulator.mean('top1_agreement'):.4f} "
                f"top5={logit_accumulator.mean('top5_overlap', 'mean'):.4f} "
                f"top10={logit_accumulator.mean('top10_overlap', 'mean'):.4f}",
                flush=True,
            )
    local_metrics = local_accumulator.result()
    live_metrics = live_accumulator.result()
    result = {
        "local_modules": local_metrics,
        "live_modules": live_metrics,
        "layers": {str(index): accumulator.result() for index, accumulator in layer_accumulators.items()},
        "logits": logit_accumulator.result(),
    }
    if module_scope == "qkvo":
        result["local_qkvo"] = local_metrics
        result["live_qkvo"] = live_metrics
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.layers < 1:
        raise ValueError("layer count must be positive")
    if args.prepare_yaqa_only and args.yaqa_factor_cache is None:
        raise ValueError("--prepare-yaqa-only requires --yaqa-factor-cache")
    if args.evaluation_row_offset < args.calibration_rows:
        raise ValueError("evaluation rows must be disjoint from calibration rows")
    yaqa_enabled = any(ARM_CONFIG[arm].get("rounding") == "yaqa" for arm in args.arms)
    yaqa_row_offset = (
        args.evaluation_row_offset + args.evaluation_rows if args.yaqa_row_offset is None else args.yaqa_row_offset
    )
    if yaqa_enabled and (
        args.yaqa_rows < 1
        or args.yaqa_batch_size < 1
        or yaqa_row_offset < args.evaluation_row_offset + args.evaluation_rows
    ):
        raise ValueError("YAQA rows must be positive and disjoint from calibration and evaluation rows")
    rates = tuple(normalize_qvq_rate(rate) for rate in args.rates)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    qos_requested = request_performance_qos()

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    config.num_hidden_layers = args.layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    modules = _quantized_linear_modules(model, layer_count=args.layers, module_scope=args.module_scope)
    module_names = tuple(modules)
    module_shapes = {name: list(module.weight.shape) for name, module in modules.items()}
    yaqa_input_hessians = {}
    yaqa_output_hessians = {}
    yaqa_stats = None
    yaqa_batches = None
    if yaqa_enabled:
        cache_metadata = _yaqa_cache_metadata(args, module_shapes, yaqa_row_offset)
        staged_yaqa_replay = args.module_scope == "all-linear" and args.all_linear_hessian_mode != "dense-frozen"
        if staged_yaqa_replay or args.yaqa_factor_cache is None or not args.yaqa_factor_cache.is_file():
            yaqa_encoded, yaqa_data_stats = load_nm_evaluation_batch(
                tokenizer,
                dataset_path=args.dataset,
                row_offset=yaqa_row_offset,
                rows=args.yaqa_rows,
                max_length=args.max_length,
            )
            yaqa_batches = _padded_batch_chunks(yaqa_encoded, batch_size=args.yaqa_batch_size)
        if args.yaqa_factor_cache is not None and args.yaqa_factor_cache.is_file():
            yaqa_input_hessians, yaqa_output_hessians, yaqa_stats = _load_yaqa_factor_cache(
                args.yaqa_factor_cache,
                expected_metadata=cache_metadata,
            )
            yaqa_stats = dict(yaqa_stats)
            yaqa_stats["cache"] = "loaded"
            yaqa_stats["cache_path"] = str(args.yaqa_factor_cache)
            print(f"Loaded validated YAQA Sketch-B factors from {args.yaqa_factor_cache}", flush=True)
        else:
            assert yaqa_batches is not None
            print(
                f"Capturing YAQA Sketch B from rows {yaqa_row_offset}:{yaqa_row_offset + args.yaqa_rows} "
                f"in {len(yaqa_batches)} batches",
                flush=True,
            )

            def sketch_progress(stats):
                print(
                    f"Sketch-B {stats['completed_batches']}/{stats['total_batches']} batches, "
                    f"{stats['completed_sequences']}/{args.yaqa_rows} rows, {stats['valid_tokens']} valid tokens",
                    flush=True,
                )

            yaqa_started = time.perf_counter()
            yaqa_input_hessians, yaqa_output_hessians, yaqa_stats = capture_yaqa_sketch_b(
                model,
                yaqa_batches,
                modules,
                device=device,
                seed=args.yaqa_seed,
                minimum_sequences=args.yaqa_rows,
                checkpoint_modules=tuple(model.model.layers),
                progress_callback=sketch_progress,
            )
            yaqa_stats["collection_seconds"] = time.perf_counter() - yaqa_started
            yaqa_stats["data"] = yaqa_data_stats
            yaqa_stats["row_offset"] = yaqa_row_offset
            yaqa_stats["batch_size"] = args.yaqa_batch_size
            yaqa_stats["cache"] = "collected"
            if args.yaqa_factor_cache is not None:
                _save_yaqa_factor_cache(
                    args.yaqa_factor_cache,
                    metadata=cache_metadata,
                    input_hessians=yaqa_input_hessians,
                    output_hessians=yaqa_output_hessians,
                    stats=yaqa_stats,
                )
                yaqa_stats["cache_path"] = str(args.yaqa_factor_cache)
                print(f"Saved YAQA Sketch-B factors to {args.yaqa_factor_cache}", flush=True)
        if staged_yaqa_replay:
            initial_yaqa_names = _all_linear_dependency_stages(
                model,
                modules,
                layerwise=args.all_linear_hessian_mode == "layerwise",
            )[0]
            yaqa_input_hessians = {name: yaqa_input_hessians[name] for name in initial_yaqa_names}
            yaqa_output_hessians = {name: yaqa_output_hessians[name] for name in initial_yaqa_names}
            yaqa_stats = dict(yaqa_stats)
            yaqa_stats["retained_initial_stage"] = "qkv"
            yaqa_stats["retained_initial_modules"] = len(initial_yaqa_names)
    if args.prepare_yaqa_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"metadata": cache_metadata, "stats": yaqa_stats}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("YAQA Sketch-B factor preparation complete", flush=True)
        return

    calibration, calibration_stats = load_nm_calibration_batches(
        tokenizer,
        config,
        dataset_path=args.dataset,
        rows=args.calibration_rows,
        concat_size=None,
        batch_size=1,
    )
    if calibration_stats["concat_size"] is not None or calibration_stats["batch_size"] != 1:
        raise RuntimeError("codec comparison requires independent, non-concatenated calibration rows at batch 1")
    if calibration_stats["prepared_sequences"] != args.calibration_rows:
        raise RuntimeError("codec comparison calibration did not preserve one sequence per source row")
    evaluation, evaluation_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=args.evaluation_row_offset,
        rows=args.evaluation_rows,
        max_length=args.max_length,
    )
    evaluation_rows = _unpadded_evaluation_rows(evaluation, device)

    print(f"Capturing {args.module_scope} calibration Hessians", flush=True)
    staged_all_linear = args.module_scope == "all-linear" and args.all_linear_hessian_mode != "dense-frozen"
    initial_hessian_modules = modules
    if staged_all_linear:
        first_stage = _all_linear_dependency_stages(
            model,
            modules,
            layerwise=args.all_linear_hessian_mode == "layerwise",
        )[0]
        has_block_ldlq_arm = any(ARM_CONFIG[arm].get("rounding", "block_ldlq") != "yaqa" for arm in args.arms)
        initial_hessian_modules = {name: modules[name] for name in first_stage} if has_block_ldlq_arm else {}
    shared_hessian_groups = _shared_input_hessian_groups(model, initial_hessian_modules)
    if initial_hessian_modules:
        hessians, sample_counts = capture_calibration_hessians(
            model,
            calibration,
            initial_hessian_modules,
            device=device,
            shared_input_groups=shared_hessian_groups,
            stop_after_module=initial_hessian_modules[first_stage[-1]] if staged_all_linear else None,
        )
    else:
        hessians, sample_counts = {}, {}
    original_weights = {name: module.weight.detach().cpu().float().clone() for name, module in modules.items()}
    print("Loading an immutable dense replay model for row-streamed metrics", flush=True)
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    dense_modules = _quantized_linear_modules(
        dense_model,
        layer_count=args.layers,
        module_scope=args.module_scope,
    )

    report = {
        "settings": {
            "model": str(args.model),
            "source_layers": source_layers,
            "tested_layers": args.layers,
            "module_scope": args.module_scope,
            "all_linear_hessian_mode": args.all_linear_hessian_mode,
            "modules": list(module_names),
            "module_shapes": module_shapes,
            "rates": list(rates),
            "arms": list(args.arms),
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "performance_qos_requested": qos_requested,
            "calibration": calibration_stats,
            "calibration_execution": "independent full rows; batch=1; no sequence concatenation",
            "calibration_samples": sample_counts,
            "calibration_hessian_capture": (
                (
                    "exact layerwise producer/consumer replay"
                    if args.all_linear_hessian_mode == "layerwise"
                    else "staged producer/consumer replay"
                )
                if staged_all_linear
                else "one dense-frozen model replay"
            ),
            "shared_input_hessian_groups": [list(group) for group in shared_hessian_groups],
            "evaluation": evaluation_stats,
            "evaluation_batch_size": 1,
            "evaluation_execution": "independent full rows; batch=1; no sequence concatenation",
            "yaqa_sketch_b": yaqa_stats,
            "yaqa_spectral_ranks": list(args.yaqa_spectral_ranks),
            "yaqa_spectral_lambdas": list(args.yaqa_spectral_lambdas),
            "yaqa_spectral_push_alphas": list(args.yaqa_spectral_push_alphas),
            "serialization": "disabled; dense reconstruction comparison",
        },
        "results": {},
    }

    for rate in rates:
        report["results"][str(rate)] = {}
        for arm in args.arms:
            started = time.perf_counter()
            geometry = dict(ARM_CONFIG[arm])
            if geometry.get("v2b2_p32") and rate > 3.5:
                report["results"][str(rate)][arm] = {
                    "status": "unsupported",
                    "reason": "V2B2-P32 supports W1 through W3.5",
                }
                print(f"Skipping W{rate:g} {arm}: V2B2-P32 supports W1 through W3.5", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                continue
            rounding = geometry.get("rounding", "block_ldlq")
            if geometry.get("yaqa_spectral_refinement", False):
                geometry["yaqa_spectral_ranks"] = tuple(args.yaqa_spectral_ranks)
                geometry["yaqa_spectral_lambdas"] = tuple(args.yaqa_spectral_lambdas)
            if geometry.get("yaqa_spectral_push", False):
                geometry["yaqa_spectral_ranks"] = tuple(args.yaqa_spectral_ranks)
                geometry["yaqa_spectral_push_alphas"] = tuple(args.yaqa_spectral_push_alphas)
            batch_size = args.trellis_batch_size or default_qvq_trellis_batch_size(
                rate,
                device,
                trellis_window=geometry["trellis_window"],
            )
            reconstructions: dict[str, torch.Tensor] = {}
            weight_metrics = {}
            selector_histogram = [0, 0, 0, 0]
            alternative_bank_histogram = [0, 0, 0, 0]
            staged_replay = args.module_scope == "all-linear" and args.all_linear_hessian_mode != "dense-frozen"
            dependency_stages = (
                _all_linear_dependency_stages(
                    model,
                    modules,
                    layerwise=args.all_linear_hessian_mode == "layerwise",
                )
                if staged_replay
                else (module_names,)
            )
            stage_reports = []
            completed_modules = 0
            print(f"Starting W{rate:g} {arm} with trellis batch {batch_size}", flush=True)
            for stage_index, stage_names in enumerate(dependency_stages):
                stage_started = time.perf_counter()
                stage_modules = {name: modules[name] for name in stage_names}
                if staged_replay:
                    role = ("qkv", "o", "gate_up", "down")[stage_index % 4]
                    stage_label = (
                        f"layer_{stage_index // 4}_{role}"
                        if args.all_linear_hessian_mode == "layerwise"
                        else role
                    )
                else:
                    stage_label = "dense_frozen"
                stage_sample_counts = {name: sample_counts[name] for name in stage_names if name in sample_counts}
                stage_yaqa_stats = None
                if rounding == "yaqa" and staged_replay and stage_index > 0:
                    if yaqa_batches is None:
                        raise RuntimeError("staged all-linear YAQA replay requires the disjoint Sketch-B rows")

                    def stage_sketch_progress(stats, label=stage_label):
                        print(
                            f"W{rate:g} {arm} staged Sketch-B {label}: "
                            f"{stats['completed_batches']}/{stats['total_batches']} batches, "
                            f"{stats['completed_sequences']}/{args.yaqa_rows} rows",
                            flush=True,
                        )

                    stage_input_hessians, stage_output_hessians, stage_yaqa_stats = capture_yaqa_sketch_b(
                        model,
                        yaqa_batches,
                        stage_modules,
                        device=device,
                        seed=args.yaqa_seed,
                        minimum_sequences=args.yaqa_rows,
                        checkpoint_modules=tuple(model.model.layers),
                        progress_callback=stage_sketch_progress,
                    )
                else:
                    stage_input_hessians = yaqa_input_hessians
                    stage_output_hessians = yaqa_output_hessians
                if rounding != "yaqa":
                    if staged_replay and stage_index > 0:
                        stage_shared_groups = _shared_input_hessian_groups(model, stage_modules)
                        stage_hessians, stage_sample_counts = capture_calibration_hessians(
                            model,
                            calibration,
                            stage_modules,
                            device=device,
                            shared_input_groups=stage_shared_groups,
                            stop_after_module=stage_modules[stage_names[-1]],
                        )
                    else:
                        stage_hessians = hessians
                    shared_hessian_totals = Counter(id(stage_hessians[name]) for name in stage_names)
                    shared_hessian_remaining = shared_hessian_totals.copy()
                    device_hessians: dict[int, torch.Tensor] = {}
                    input_preparations = {}

                for name in stage_names:
                    module = modules[name]
                    completed_modules += 1
                    module_started = time.perf_counter()
                    module_telemetry = QVQQuantizationTelemetry() if args.qvq_telemetry else None
                    if rounding == "yaqa":
                        quantization_hessian = stage_input_hessians[name].to(device)
                        input_preparation = None
                    else:
                        source_hessian = stage_hessians[name]
                        source_key = id(source_hessian)
                        quantization_hessian = device_hessians.get(source_key)
                        if quantization_hessian is None:
                            quantization_hessian = source_hessian.to(device)
                            if shared_hessian_totals[source_key] > 1:
                                device_hessians[source_key] = quantization_hessian
                        input_preparation = input_preparations.get(source_key)
                        if input_preparation is None and shared_hessian_totals[source_key] > 1:
                            input_preparation = prepare_qvq_input_hessian(
                                quantization_hessian,
                                seed=args.seed,
                                damp_percent=0.01,
                            )
                            input_preparations[source_key] = input_preparation
                    result = quantize_qvq_linear(
                        original_weights[name].to(device),
                        quantization_hessian,
                        bits=rate,
                        output_hessian=(
                            stage_output_hessians[name].to(device) if rounding == "yaqa" else None
                        ),
                        seed=args.seed,
                        trellis_batch_size=batch_size,
                        input_hessian_preparation=input_preparation,
                        telemetry=module_telemetry,
                        **geometry,
                    )
                    if rounding != "yaqa":
                        shared_hessian_remaining[source_key] -= 1
                        if shared_hessian_remaining[source_key] == 0:
                            device_hessians.pop(source_key, None)
                            input_preparations.pop(source_key, None)
                    reconstruction = result.weight.detach().cpu().float()
                    reconstructions[name] = reconstruction
                    weight_metrics[name] = _weight_metrics(original_weights[name], reconstruction)
                    weight_metrics[name]["shape"] = list(original_weights[name].shape)
                    weight_metrics[name]["parameter_count"] = original_weights[name].numel()
                    weight_metrics[name]["qvq_telemetry"] = result.telemetry
                    weight_metrics[name]["proxy_loss"] = float(result.proxy_loss)
                    weight_metrics[name]["kronecker_proxy_loss"] = (
                        None if result.kronecker_proxy_loss is None else float(result.kronecker_proxy_loss)
                    )
                    weight_metrics[name]["yaqa_spectral"] = {
                        "selected": result.yaqa_spectral_selected,
                        "method": result.yaqa_spectral_method,
                        "rank": result.yaqa_spectral_rank,
                        "lambda": result.yaqa_spectral_lambda,
                        "alpha": result.yaqa_spectral_alpha,
                        "svd_device": result.yaqa_spectral_svd_device,
                        "concentration": result.yaqa_spectral_concentration,
                        "oracle_losses": result.yaqa_spectral_oracle_losses,
                        "candidates": result.yaqa_spectral_candidates,
                        "absorption_efficiency": result.yaqa_spectral_absorption_efficiency,
                        "selector_churn": result.yaqa_spectral_selector_churn,
                        "family_changed": result.yaqa_spectral_family_changed,
                    }
                    if result.bank_ids is not None:
                        counts = torch.bincount(result.bank_ids.to(torch.int64).cpu(), minlength=4)
                        selector_histogram = [
                            current + int(count)
                            for current, count in zip(selector_histogram, counts.tolist(), strict=True)
                        ]
                    if result.bank_alt_id is not None:
                        alternative_bank_histogram[int(result.bank_alt_id.item())] += 1
                    print(
                        f"W{rate:g} {arm}: {completed_modules}/{len(modules)} {name} "
                        f"in {time.perf_counter() - module_started:.2f}s",
                        flush=True,
                    )
                    if result.telemetry is not None:
                        phase_times = ", ".join(
                            f"{phase}={values['gpu_ms']:.1f}ms"
                            for phase, values in result.telemetry["phases"].items()
                            if values["gpu_ms"] is not None
                        )
                        print(f"  QVQ telemetry [{list(original_weights[name].shape)}]: {phase_times}", flush=True)

                with torch.no_grad():
                    for name in stage_names:
                        module = modules[name]
                        module.weight.copy_(reconstructions[name].to(device=device, dtype=module.weight.dtype))
                stage_reports.append(
                    {
                        "name": stage_label,
                        "modules": list(stage_names),
                        "sample_counts": stage_sample_counts,
                        "yaqa_sketch_b": stage_yaqa_stats,
                        "seconds": time.perf_counter() - stage_started,
                    }
                )
                if rounding == "yaqa":
                    del stage_input_hessians, stage_output_hessians
                elif staged_replay and stage_index > 0:
                    del stage_hessians

            with torch.no_grad():
                for name, module in modules.items():
                    module.weight.copy_(reconstructions[name].to(device=device, dtype=module.weight.dtype))
            streamed_metrics = _streaming_compare_models(
                dense_model,
                model,
                evaluation_rows,
                dense_modules,
                modules,
                reconstructions,
                layer_count=args.layers,
                progress_label=f"W{rate:g} {arm}",
                module_scope=args.module_scope,
            )
            arm_report = {
                "seconds": time.perf_counter() - started,
                "trellis_batch_size": batch_size,
                "rounding": rounding,
                "hessian_mode": args.all_linear_hessian_mode if args.module_scope == "all-linear" else "dense-frozen",
                "quantization_stages": stage_reports,
                "effective_bpw": rate + (2 / 64 if geometry.get("v2b2_p32") or geometry.get("v2b4_p64") else 0),
                "bank_selectors": _selector_metrics(selector_histogram),
                "module_alternative_bank_histogram": (
                    alternative_bank_histogram if geometry.get("v2b2_p32") else None
                ),
                "qvq_telemetry": _aggregate_qvq_telemetry(weight_metrics) if args.qvq_telemetry else None,
                "weight": {
                    "modules": weight_metrics,
                    "mean_mse": _mean([metric["mse"] for metric in weight_metrics.values()]),
                    "mean_relative_l2": _mean([metric["relative_l2"] for metric in weight_metrics.values()]),
                },
                **streamed_metrics,
            }
            report["results"][str(rate)][arm] = arm_report
            logits = arm_report["logits"]
            layer_kl = _mean(
                [metrics["kl_forward"]["mean"] for metrics in arm_report["layers"].values()]
            )
            print(
                f"W{rate:g} {arm}: relL2={arm_report['weight']['mean_relative_l2']:.6f} "
                f"localKL={arm_report['local_modules']['kl_forward']['mean']:.6f} "
                f"liveKL={arm_report['live_modules']['kl_forward']['mean']:.6f} "
                f"layerKL={layer_kl:.6f} "
                f"logitKL={logits['kl_forward']['mean']:.6f} "
                f"top1={logits['top1_agreement']:.4f} "
                f"top5={logits['top5_overlap']['mean']:.4f} "
                f"top10={logits['top10_overlap']['mean']:.4f}",
                flush=True,
            )
            with torch.no_grad():
                for name, module in modules.items():
                    module.weight.copy_(original_weights[name].to(device=device, dtype=module.weight.dtype))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            del reconstructions, streamed_metrics
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
