# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Versioned, output-aware recovery owned by the P32 window deployment package.

B is stored in the *inner* output domain. Fits are solved in final-output
coordinates, then mapped back through the inverse SV/output transform. This
keeps the objective output-aware even when SV is not uniform.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from .qvq import repack_p32_planar_to_window, repack_p32_window_to_planar
from .qvq_codecs import pgc16_levels_for_version
from .rotation.hadamard_utils import matmul_hadU

CONTRACT = "p32-window-r8-v1:fp32-project,fp16-hidden,fp32-expand-add,existing-output-transform"
RANK8_BUFFERS = ("rank8_A", "rank8_B", "rank8_metadata")


@dataclass(frozen=True)
class P32WindowConfig:
    """Static graph policy. Unimplemented kernel choices fail explicitly."""

    abi_version: int = 3
    algorithm: str = "auto"
    recovery_mode: str = "off"
    recovery_kernel: str = "separate_reference"
    quality_mode: str = "fast"
    split_k: int = 1
    min_m: int = 1
    max_m: int = 4096

    def __post_init__(self):
        if self.abi_version != 3:
            raise ValueError("unsupported P32 window ABI version")
        if self.algorithm not in (
            "auto",
            "production_window",
            "hopper_direct_decode_mma",
            "hopper_m16",
        ):
            raise ValueError(
                "explicit direct-decode geometry is not exposed by this reference API"
            )
        if (
            type(self.split_k) is not int
            or self.split_k < 1
            or type(self.min_m) is not int
            or type(self.max_m) is not int
            or not 1 <= self.min_m <= self.max_m <= 4096
        ):
            raise ValueError("invalid split count or M range")
        if self.recovery_mode not in ("off", "on", "auto"):
            raise ValueError("invalid recovery mode")
        if self.quality_mode not in ("fast", "balanced", "quality"):
            raise ValueError("invalid quality mode")
        if self.recovery_kernel not in ("separate_reference", "fused_epilogue"):
            raise ValueError("fused recovery kernels are not implemented")


def _digest(tensors, metadata):
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for name, tensor in sorted(tensors.items()):
        if tensor is None:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _base(layer):
    if not layer.v2b2_p32 or layer.activation is not None:
        raise ValueError(
            "rank-8 recovery requires P32 with the existing A16 activation contract"
        )
    metadata = {
        "bits": layer.bits,
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "codebook_version": layer.codebook_version,
        "input_hadamard": layer.input_hadamard,
        "output_hadamard": layer.output_hadamard,
    }
    tensors = {
        "window_words": repack_p32_planar_to_window(layer.trellis, bits=layer.bits),
        "bank_ids": layer.bank_ids,
        "bank_alt_id": layer.bank_alt_id,
        "SU": layer.SU,
        "SV": layer.SV,
        "bias": layer.bias,
        "levels": pgc16_levels_for_version(layer.codebook_version),
    }
    return tensors, metadata


def _metadata(layer):
    return json.loads(bytes(layer.rank8_metadata.detach().cpu().tolist()).decode())


def _encode(metadata, device):
    return torch.tensor(
        list(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()),
        dtype=torch.uint8,
        device=device,
    )


def _versions(layer):
    names = (
        "trellis",
        "SU",
        "SV",
        "bias",
        "bank_ids",
        "bank_alt_id",
        *RANK8_BUFFERS,
    )
    return (
        layer.bits,
        layer.codebook_version,
        layer.input_hadamard,
        layer.output_hadamard,
        id(layer.activation),
    ) + tuple(
        (name, id(value), value._version)
        for name in names
        if (value := getattr(layer, name, None)) is not None
    )


def prepare_rank8(layer, config):
    """Validate/bind before graph capture; never hashes or synchronizes in forward."""
    if not isinstance(config, P32WindowConfig):
        raise TypeError("config must be P32WindowConfig")
    if not layer.v2b2_p32:
        raise ValueError("window configuration requires P32")
    if layer.training:
        raise ValueError("window recovery is inference-only")
    if layer.trellis.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare rank8 and kernel policy before CUDA Graph capture")
    if config.algorithm.startswith("hopper_"):
        if layer.trellis.device.type != "cuda":
            raise ValueError("explicit Hopper policy requires an SM90 CUDA device")
        properties = torch.cuda.get_device_properties(layer.trellis.device)
        if (
            (properties.major, properties.minor) != (9, 0)
            or not any(name in properties.name for name in ("H100", "H200"))
            or not layer.v2b2_p32
            or layer.activation is not None
            or layer.in_features % 256
            or layer.out_features % 256
            or layer.bits not in (2, 2.5, 3, 3.5)
        ):
            raise ValueError("unsupported explicit Hopper P32 contract")
        # Reuse the existing versioned window/selector cache; no new packing.
        layer._prepare_amd_p32_metadata(layer.trellis.device)
    enabled = False
    if config.recovery_mode != "off" and not (
        config.recovery_mode == "auto" and config.quality_mode == "fast"
    ):
        if layer.rank8_metadata is None:
            if config.recovery_mode == "on":
                raise ValueError("recovery requested without validated tensors")
        else:
            metadata = _metadata(layer)
            tensors, base_metadata = _base(layer)
            if metadata.get("base_hash") != _digest(tensors, base_metadata):
                raise ValueError("recovery base hash mismatch")
            if metadata.get("fit_contract") != CONTRACT or not metadata.get(
                "validated"
            ):
                raise ValueError("unvalidated or incompatible recovery contract")
            if (
                layer.rank8_A is None
                or layer.rank8_B is None
                or layer.rank8_A.shape != (layer.in_features, 8)
                or layer.rank8_B.shape != (8, layer.out_features)
            ):
                raise ValueError("invalid rank-8 tensor shapes")
            for tensor in (layer.rank8_A, layer.rank8_B):
                if (
                    tensor.dtype != torch.float16
                    or tensor.device != layer.trellis.device
                ):
                    raise ValueError(
                        "recovery tensors must be FP16 on the window device"
                    )
                if not torch.isfinite(tensor).all():
                    raise ValueError("non-finite recovery tensors")
            if metadata.get("factors_hash") != _digest(
                {"A": layer.rank8_A, "B": layer.rank8_B}, {}
            ):
                raise ValueError("recovery factors hash mismatch")
            enabled = (
                config.recovery_mode == "on"
                or config.quality_mode == "quality"
                or (
                    config.quality_mode == "balanced"
                    and metadata.get("selected", False)
                )
            )
    if enabled and getattr(layer, "_qvq_grouped_p32_delegate", None) is not None:
        raise ValueError(
            "prepare recovery before grouping; grouped recovery is not implemented"
        )
    if (
        enabled
        and layer.trellis.device.type == "cuda"
        and torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("rank8 FP32 reference requires CUDA matmul TF32 disabled")
    if (
        enabled
        and config.recovery_kernel == "fused_epilogue"
        and (
            layer.trellis.device.type != "cuda"
            or torch.cuda.get_device_capability(layer.trellis.device) != (9, 0)
            or layer.out_features > 16384
            or layer.out_features & (layer.out_features - 1)
        )
    ):
        raise ValueError(
            "rank8 fused epilogue requires SM90 and power-of-two N <= 16384"
        )
    grouped = getattr(layer, "_gptqmodel_qvq_grouped_runtime", None)
    if grouped is not None:
        if grouped._outputs is not None:
            raise RuntimeError(
                "cannot change rank8 mode during a sibling projection cycle"
            )
        grouped.invalidate()
    layer._p32_window_config = config
    layer._p32_rank8_enabled = enabled
    layer._p32_rank8_versions = _versions(layer) if enabled else None


def validate_rank8_state(layer):
    """Guard operator state before alternate dispatch paths can bypass correction."""
    if not getattr(layer, "_p32_rank8_enabled", False):
        return
    if layer.training:
        raise RuntimeError("window recovery is inference-only")
    if _versions(layer) != layer._p32_rank8_versions:
        raise RuntimeError(
            "window/recovery state changed; prepare recovery again before execution"
        )
    if layer.trellis.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("rank8 FP32 reference requires CUDA matmul TF32 disabled")


def add_rank8_correction(layer, transformed, base):
    """No recovery tensor access on the disabled branch. FP16 hidden is deliberate."""
    if not getattr(layer, "_p32_rank8_enabled", False):
        return base
    validate_rank8_state(layer)
    hidden = (transformed.float() @ layer.rank8_A.float()).half()
    correction = hidden.float() @ layer.rank8_B.float()
    return base.float() + correction


def fused_rank8_output(layer, transformed, base, compute_dtype):
    """Fuse expansion/addition into the existing numerical output-transform contract."""
    validate_rank8_state(layer)
    from ..utils.qvq_rank8_triton import rank8_output_epilogue

    hidden = (transformed.float() @ layer.rank8_A.float()).half()
    return rank8_output_epilogue(
        hidden,
        layer.rank8_B,
        base,
        layer._cached_cast("SV", compute_dtype, base.dtype),
        layer._cached_cast("bias", compute_dtype, base.dtype),
        hadamard=layer.output_hadamard,
    )


def qvq_p32_window_linear(layer, x, config=None):
    """One Python operator over a QVQLinear-owned window and optional recovery.

    Configure outside capture. Both branches use the module's production M
    dispatch, activation transform and output epilogue, with no model wrapper.
    """
    if config is not None:
        prepare_rank8(layer, config)
    return layer(x)


def _metrics(error):
    absolute = error.double().abs().flatten()
    return {
        "mse": float(absolute.square().mean()),
        "mae": float(absolute.mean()),
        "max": float(absolute.max()),
        "tail": float(torch.quantile(absolute, 0.99)),
    }


def _check_documents(train_ids, heldout_ids):
    if not train_ids or not heldout_ids or set(train_ids) & set(heldout_ids):
        raise ValueError("fit and held-out document IDs must be nonempty and disjoint")


@torch.no_grad()
def fit_rank8(
    layer,
    teacher,
    train_inputs,
    heldout_inputs,
    *,
    train_document_ids,
    heldout_document_ids,
    source_kind="calibration",
    minimum_improvement=0.01,
):
    """Finish a quantized module with two rank-8 fits; never consume eval benchmarks.

    Inputs must be original activations from disjoint calibration documents.
    The caller owns document provenance and row collection. The teacher must
    be the original FP linear module, before quantized replay overwrites it.
    Dense CPU FP64 SVD/lstsq is an initial bounded-calibration reference fitter.
    """
    if source_kind != "calibration":
        raise ValueError("only calibration activations may enter recovery fitting")
    _check_documents(train_document_ids, heldout_document_ids)
    if not 0 <= minimum_improvement < 1:
        raise ValueError("minimum_improvement must be in [0, 1)")
    if not isinstance(teacher, torch.nn.Linear) or teacher.training or layer.training:
        raise ValueError("fitting requires eval-mode original FP Linear and P32 module")
    if (teacher.in_features, teacher.out_features) != (
        layer.in_features,
        layer.out_features,
    ):
        raise ValueError("teacher shape mismatch")
    for inputs in (train_inputs, heldout_inputs):
        if (
            inputs.ndim != 2
            or not inputs.shape[0]
            or inputs.shape[1] != layer.in_features
            or not torch.isfinite(inputs).all()
        ):
            raise ValueError(
                "calibration inputs must be finite nonempty [rows,K] matrices"
            )
    if layer.rank8_metadata is not None:
        raise ValueError(
            "fit rank8 on a pristine quantized module, not an already fitted module"
        )
    tensors, base_metadata = _base(layer)
    if not torch.isfinite(layer.SV).all() or (layer.SV == 0).any():
        raise ValueError("recovery requires finite nonzero SV")
    old_config = getattr(layer, "_p32_window_config", P32WindowConfig())
    prepare_rank8(layer, P32WindowConfig())
    try:
        transformed = [
            layer.transform_input(x).reshape(-1, layer.in_features)
            for x in (train_inputs, heldout_inputs)
        ]
        base = [
            layer.forward_pretransformed(x, output_dtype=torch.float32)
            for x in transformed
        ]
        targets = [teacher(x).float() for x in (train_inputs, heldout_inputs)]
        if any(
            not torch.isfinite(value).all() for value in (*base, *targets, *transformed)
        ):
            raise ValueError("non-finite deployed calibration outputs or transforms")
        residual = [
            (target - output).double().cpu() for target, output in zip(targets, base)
        ]
        x_train = transformed[0].double().cpu()
        # Reduced-rank regression: SVD of the least-squares *predicted output*,
        # not the weight residual. Rank-deficient inputs use the SVD solver.
        candidates = []
        for objective in ("output_l2", "tail_weighted_output_l2"):
            weights = torch.ones((x_train.shape[0], 1), dtype=torch.float64)
            if objective == "tail_weighted_output_l2":
                energy = residual[0].square().mean(1, keepdim=True)
                weights = (1 + energy / energy.mean().clamp_min(1e-30)).sqrt()
            design = x_train * weights
            response = residual[0] * weights
            solution = torch.linalg.lstsq(
                design, response, driver="gelsd", rcond=1e-5
            ).solution
            _, _, vh = torch.linalg.svd(design @ solution, full_matrices=False)
            rank = min(8, vh.shape[0])
            a = torch.zeros((layer.in_features, 8), dtype=torch.float64)
            b = torch.zeros((8, layer.out_features), dtype=torch.float64)
            a[:, :rank] = solution @ vh[:rank].T
            b[:rank] = vh[:rank]
            # final correction = H(inner correction) * SV, so inner B =
            # H^T(final B / SV). transpose=True matters for composite widths.
            b = b / layer.SV.detach().double().cpu()
            if layer.output_hadamard:
                b = matmul_hadU(b, transpose=True)
            a = a.to(device=layer.trellis.device, dtype=torch.float16)
            b = b.to(device=layer.trellis.device, dtype=torch.float16)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                continue
            scores = []
            for x, target in zip(transformed, targets):
                inner = layer._inner_forward(x)
                hidden = (x.float() @ a.float()).half()
                corrected = inner.float() + hidden.float() @ b.float()
                # Invoke the deployed output transform rather than estimating
                # its rounding from the mathematical inverse used above.
                actual = layer._recover_output_compute_dtype(corrected, x.dtype)
                scores.append(_metrics(target - actual))
            if all(
                all(torch.isfinite(torch.tensor(v)) for v in score.values())
                for score in scores
            ):
                candidates.append((objective, a, b, scores))
        baseline = [_metrics(r) for r in residual]
        eligible = [
            c
            for c in candidates
            if all(
                s["mse"] < ref["mse"] and s["tail"] <= ref["tail"]
                for s, ref in zip(c[3], baseline)
            )
        ]
        selected = min(
            eligible, key=lambda c: (c[3][1]["mse"], c[3][1]["tail"]), default=None
        )
        report = {
            "fit_contract": CONTRACT,
            "rank": 8,
            "dtype": "float16",
            "input_domain": "p32_transformed",
            "output_domain": "pre_output_hadamard_sv",
            "base_hash": _digest(tensors, base_metadata),
            "teacher_hash": _digest(dict(teacher.named_parameters()), {}),
            "validated": selected is not None,
            "selected": selected is not None
            and all(
                score["mse"] < ref["mse"] * (1 - minimum_improvement)
                for score, ref in zip(selected[3], baseline)
            ),
            "source_kind": source_kind,
            "train_document_ids": list(train_document_ids),
            "heldout_document_ids": list(heldout_document_ids),
            "baseline": baseline,
            "candidates": {c[0]: c[3] for c in candidates},
            "minimum_improvement": minimum_improvement,
            "lstsq_rcond": 1e-5,
            "fit_device": str(layer.trellis.device),
            "activation_dtype": str(train_inputs.dtype),
            "torch_version": str(torch.__version__),
            "objective": None if selected is None else selected[0],
        }
        if selected is not None:
            _, a, b, _ = selected
            report["factors_hash"] = _digest({"A": a, "B": b}, {})
            # Ordinary versioned buffers are required for capture-safe guards.
            with torch.inference_mode(False):
                layer.rank8_A = a.clone()
                layer.rank8_B = b.clone()
                layer.rank8_metadata = _encode(report, layer.trellis.device)
        return report
    finally:
        prepare_rank8(layer, old_config)


def export_window_package(layer):
    """One window-only deployment payload; planar bytes are not duplicated."""
    tensors, metadata = _base(layer)
    recovery = None
    if layer.rank8_metadata is not None:
        old_config = getattr(layer, "_p32_window_config", P32WindowConfig())
        try:
            prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
            recovery = _metadata(layer)
            tensors.update(rank8_A=layer.rank8_A, rank8_B=layer.rank8_B)
        finally:
            prepare_rank8(layer, old_config)
    return {
        "metadata": metadata,
        "recovery": recovery,
        "tensors": {
            k: v.detach().cpu().contiguous().clone()
            for k, v in tensors.items()
            if v is not None
        },
    }


def load_window_package(package, *, device="cpu", config=None):
    from ..nn_modules.qlinear.qvq import QVQLinear

    metadata = dict(package["metadata"])
    tensors = {k: v.to(device) for k, v in package["tensors"].items()}
    recovery = package["recovery"]
    a, b = tensors.pop("rank8_A", None), tensors.pop("rank8_B", None)
    if recovery is not None and recovery.get("base_hash") != _digest(tensors, metadata):
        raise ValueError("window package base hash mismatch")
    if recovery is None and (a is not None or b is not None):
        raise ValueError("recovery factors require fitting metadata")
    levels = tensors.pop("levels")
    if not torch.equal(
        levels.cpu(), pgc16_levels_for_version(metadata["codebook_version"]).cpu()
    ):
        raise ValueError("window package codebook levels mismatch")
    tensors["trellis"] = repack_p32_window_to_planar(
        tensors.pop("window_words"), bits=metadata["bits"]
    )
    layer = QVQLinear(
        **metadata, tensors=tensors, bank_count=2, v2b2_p32=True, bias="bias" in tensors
    ).eval()
    if recovery is not None:
        layer.rank8_A, layer.rank8_B = a, b
        layer.rank8_metadata = _encode(recovery, device)
        layer.post_init()
        prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    prepare_rank8(layer, config or P32WindowConfig())
    return layer


def window_package_storage(packages):
    """Weight-count-weighted BPW and tensor bytes, excluding container headers."""
    records = []
    for package in packages:
        meta = package["metadata"]
        k, n = meta["in_features"], meta["out_features"]
        tensors = package["tensors"]
        base_bytes = sum(
            v.numel() * v.element_size()
            for name, v in tensors.items()
            if name not in ("rank8_A", "rank8_B")
        )
        recovery_bytes = sum(
            tensors[name].numel() * tensors[name].element_size()
            for name in ("rank8_A", "rank8_B")
            if name in tensors
        )
        records.append(
            {
                "weights": k * n,
                "window_bpw": 8 * base_bytes / (k * n),
                "recovered_bpw": 8 * (base_bytes + recovery_bytes) / (k * n),
                "rank8_delta_bpw": 16 * 8 * (k + n) / (k * n),
                "tensor_bytes": base_bytes + recovery_bytes,
            }
        )
    weights = sum(r["weights"] for r in records)
    total_bytes = sum(r["tensor_bytes"] for r in records)
    return {
        "modules": records,
        "tensor_bytes": total_bytes,
        "average_bpw": 8 * total_bytes / weights if weights else 0,
    }


@dataclass(frozen=True)
class Rank8Calibration:
    """Document-separated original activations supplied to the quantization job."""

    train_inputs: torch.Tensor
    heldout_inputs: torch.Tensor
    train_document_ids: tuple[str, ...]
    heldout_document_ids: tuple[str, ...]
    source_kind: str = "calibration"
    minimum_improvement: float = 0.01

    def __post_init__(self):
        if self.source_kind != "calibration":
            raise ValueError("only calibration activations may enter rank8 fitting")
        _check_documents(self.train_document_ids, self.heldout_document_ids)


def finish_rank8_quantization(
    result, original_weight, bias, calibration, *, bits, codebook_version
):
    """Called by the existing quantizer before returning its module result."""
    from dataclasses import replace

    from ..nn_modules.qlinear.qvq import QVQLinear

    n, k = original_weight.shape
    layer = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        bias=bias is not None,
        tensors=result.serialized_tensors(),
        v2b2_p32=True,
        bank_count=2,
        codebook_version=codebook_version,
        input_hadamard=result.input_hadamard,
        output_hadamard=result.output_hadamard,
    ).eval()
    layer.post_init()
    teacher = torch.nn.Linear(
        k,
        n,
        bias=bias is not None,
        device=original_weight.device,
        dtype=original_weight.dtype,
    ).eval()
    with torch.no_grad():
        teacher.weight.copy_(original_weight)
        if bias is not None:
            teacher.bias.copy_(bias)
    report = fit_rank8(
        layer,
        teacher,
        calibration.train_inputs,
        calibration.heldout_inputs,
        train_document_ids=calibration.train_document_ids,
        heldout_document_ids=calibration.heldout_document_ids,
        source_kind=calibration.source_kind,
        minimum_improvement=calibration.minimum_improvement,
    )
    return replace(
        result,
        rank8_A=layer.rank8_A,
        rank8_B=layer.rank8_B,
        rank8_metadata=layer.rank8_metadata,
        rank8_fit_report=report,
    )


def explicit_window_inner(layer, transformed, config):
    """Expose existing SM90 consumers; no copied A100 crossover or new decoder."""
    from ..utils.qvq_cuda import _pgc16_levels
    from ..utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma,
        qvq_p32_window_wgmma_single_large_m_packed,
    )

    rows = transformed.shape[0]
    if transformed.dtype != torch.float16 or not config.min_m <= rows <= config.max_m:
        raise ValueError("input dtype or M is outside the prepared Hopper policy")
    window, banks, alt_id = layer._prepare_amd_p32_metadata(transformed.device)
    if config.algorithm == "hopper_m16":
        return qvq_p32_window_wgmma_m16_tma(
            transformed.contiguous(),
            window,
            _pgc16_levels(transformed.device, layer.codebook_version),
            banks,
            layer.bits,
            out_features=layer.out_features,
            bank_alt_id=alt_id,
            split_count=config.split_k,
        )
    padded_rows = 16 if rows <= 16 else 32 if rows <= 32 else ((rows + 63) // 64) * 64
    padded = torch.nn.functional.pad(
        transformed, (0, 0, 0, padded_rows - rows)
    ).contiguous()
    return qvq_p32_window_wgmma_single_large_m_packed(
        padded,
        window,
        _pgc16_levels(transformed.device, layer.codebook_version),
        banks,
        layer.bits,
        out_features=layer.out_features,
        bank_alt_id=alt_id,
        split_count=config.split_k,
    )[:rows]


def window_tuning_key(layer, *, m, quality_mode, tp_world_size=1, tp_rank=0, build_id):
    """External tuner key; quality eligibility is resolved before latency tuning."""
    if not 0 <= tp_rank < tp_world_size or quality_mode not in (
        "fast",
        "balanced",
        "quality",
    ):
        raise ValueError("invalid quality/TP policy")
    device = layer.trellis.device
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        identity = (
            props.name,
            str(props.uuid),
            props.total_memory,
            props.multi_processor_count,
            props.major,
            props.minor,
        )
    else:
        identity = (device.type,)
    return (
        3,
        CONTRACT,
        build_id,
        identity,
        tp_world_size,
        tp_rank,
        layer.bits,
        layer.in_features,
        layer.out_features,
        m,
        quality_mode,
        bool(getattr(layer, "_p32_rank8_enabled", False)),
        layer.input_hadamard,
        layer.output_hadamard,
    )


def save_window_package(layer, path):
    """Serialize the unified package and report actual container bytes as well as BPW."""
    from pathlib import Path

    package = export_window_package(layer)
    torch.save(package, path)
    report = window_package_storage([package])
    report["serialized_bytes"] = Path(path).stat().st_size
    report["serialized_bpw"] = (
        8 * report["serialized_bytes"] / (layer.in_features * layer.out_features)
    )
    return report
