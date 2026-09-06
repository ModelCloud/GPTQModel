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
from dataclasses import asdict, dataclass, replace

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
    recovery_projection: str = "separate_reference"
    quality_mode: str = "fast"
    split_k: int = 1
    min_m: int = 1
    max_m: int = 8192
    block_m: int = 0
    block_n: int = 0
    block_k: int = 256
    pipeline_stages: int = 2
    warp_groups: int = 0
    chunk_m: int = 0

    def to_backend_config(self):
        """Lossless external tuning controls, including the versioned ABI."""
        return asdict(self)

    @classmethod
    def from_backend_config(cls, config):
        """Reject unknown fields/values instead of silently changing dispatch."""
        return cls(**config)

    def __post_init__(self):
        if any(
            type(v) is not int
            for v in (
                self.block_m,
                self.block_n,
                self.block_k,
                self.pipeline_stages,
                self.warp_groups,
                self.chunk_m,
            )
        ):
            raise ValueError("Hopper geometry must contain integers")
        if self.chunk_m not in (0, 4096) or (
            self.chunk_m and self.algorithm != "hopper_direct_decode_mma"
        ):
            raise ValueError("M chunking requires direct Hopper with chunk_m=0 or 4096")
        if self.block_k != 256 or self.pipeline_stages != 2:
            raise ValueError("current Hopper pipeline requires BK256 and two stages")
        if (self.block_m or self.block_n or self.warp_groups) and (
            self.algorithm != "hopper_direct_decode_mma"
            or self.block_m not in (32, 64, 128)
            or self.block_n not in (64, 128)
            or self.warp_groups not in (0, self.block_n // 64)
            or self.split_k != 1
        ):
            raise ValueError(
                "explicit Hopper geometry requires BM32/64/128, BN64/128 and split_k=1"
            )
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
            or not 1 <= self.min_m <= self.max_m <= 8192
        ):
            raise ValueError("invalid split count or M range")
        if self.recovery_mode not in ("off", "on", "auto"):
            raise ValueError("invalid recovery mode")
        if self.quality_mode not in ("fast", "balanced", "quality"):
            raise ValueError("invalid quality mode")
        if self.recovery_projection not in ("separate_reference", "input_fused", "tensor_core"):
            raise ValueError("unsupported rank8 projection implementation")
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
    grouped_delegate = getattr(layer, "_qvq_grouped_p32_delegate", None)
    if grouped_delegate is not None:
        state, consumer_index, _ = grouped_delegate
        state.prepare_rank8(consumer_index, layer, config)
        return
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
        # That cache validates the canonical selectors. The BF16 rescue branch
        # captured for narrow K must not repeat their host-side validation.
        layer._bank_ids_loaded = True
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
    if (
        enabled
        and layer.trellis.device.type == "cuda"
        and torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("rank8 FP32 reference requires CUDA matmul TF32 disabled")
    if (
        config.recovery_kernel == "fused_epilogue"
        and (
            layer.trellis.device.type != "cuda"
            or torch.cuda.get_device_capability(layer.trellis.device) != (9, 0)
            or layer.out_features > 16384
            or (layer.output_hadamard and layer.out_features & (layer.out_features - 1))
        )
    ):
        raise ValueError(
            "rank8 fused epilogue requires SM90 and N <= 16384; output Hadamard mode requires power-of-two N"
        )
    if (
        enabled
        and config.recovery_projection == "input_fused"
        and (
            layer.trellis.device.type != "cuda"
            or torch.cuda.get_device_capability(layer.trellis.device) != (9, 0)
            or layer.in_features > 16384
            or (layer.input_hadamard and layer.in_features & (layer.in_features - 1))
        )
    ):
        raise ValueError(
            "rank8 input producer requires SM90 and K <= 16384; input Hadamard mode requires power-of-two K"
        )
    if enabled and config.recovery_projection == "tensor_core" and (
        layer.trellis.device.type != "cuda"
        or torch.cuda.get_device_capability(layer.trellis.device) != (9, 0)
    ):
        raise ValueError("rank8 Tensor Core projection requires SM90")
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


def _project_rank8(layer, transformed):
    if (
        layer._p32_window_config.recovery_projection == "tensor_core"
        and transformed.dtype == torch.float16
    ):
        from ..utils.qvq_rank8_triton import rank8_tensor_core_projection

        return rank8_tensor_core_projection(transformed, layer.rank8_A)
    return (transformed.float() @ layer.rank8_A.float()).half()


def add_rank8_correction(layer, transformed, base, *, hidden=None):
    """No recovery tensor access on the disabled branch. FP16 hidden is deliberate."""
    if not getattr(layer, "_p32_rank8_enabled", False):
        return base
    validate_rank8_state(layer)
    if hidden is None:
        hidden = _project_rank8(layer, transformed)
    correction = hidden.float() @ layer.rank8_B.float()
    return base.float() + correction


def fused_rank8_output(layer, transformed, base, compute_dtype, *, hidden=None, output_dtype=None):
    """Fuse expansion/addition into the existing numerical output-transform contract."""
    validate_rank8_state(layer)
    from ..utils.qvq_rank8_triton import rank8_output_epilogue

    enabled = bool(getattr(layer, "_p32_rank8_enabled", False))
    if enabled and hidden is None:
        hidden = _project_rank8(layer, transformed)
    return rank8_output_epilogue(
        hidden,
        layer.rank8_B if enabled else None,
        base,
        layer._cached_cast("SV", compute_dtype, base.dtype),
        layer._cached_cast("bias", compute_dtype, base.dtype),
        hadamard=layer.output_hadamard,
        rank8_enabled=enabled,
        # Keep FP32 at the boundary when the surrounding operator may use
        # its range for a BF16 overflow retry. Stable power-of-two FP16 paths
        # already round only at their final store and need no extra cast.
        output_dtype=(
            torch.float16 if output_dtype == torch.float16
            and 2048 <= layer.in_features <= 16384
            and not layer.in_features & (layer.in_features - 1)
            else torch.float32
        ),
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


def _rank8_output_fit(
    x_train,
    residual,
    weights,
    *,
    max_solver_bytes,
    rcond,
    seed,
):
    """Fit a rank-8 output correction without an unbounded KxN workspace.

    Small modules retain the original deterministic least-squares/SVD reference
    path. For large K/N, a fixed-seed randomized output range produces an
    equivalent rank-8 factorization while keeping solver work proportional to
    ``(K + N) * rank`` instead of materializing a dense predicted ``rows x N``
    matrix and a full ``K x N`` least-squares solution.
    """
    if type(max_solver_bytes) is not int or max_solver_bytes < 1:
        raise ValueError("max_solver_bytes must be a positive integer")
    design = x_train * weights
    response = residual * weights
    rows, k = design.shape
    n = response.shape[1]
    rank = min(8, k, n)
    full_bytes = 8 * (k * n + rows * n)
    if full_bytes <= max_solver_bytes:
        solution = torch.linalg.lstsq(design, response, driver="gelsd", rcond=rcond).solution
        _, _, vh = torch.linalg.svd(design @ solution, full_matrices=False)
        rank = min(rank, vh.shape[0])
        a = solution @ vh[:rank].T
        b = vh[:rank]
        return a, b, "full_lstsq_svd"

    # Randomized range finding is deterministic for a fixed contract seed.
    # Oversampling is deliberately omitted: the deployed factor rank is fixed
    # at eight and the extra columns would only increase capture memory.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn((n, rank), dtype=response.dtype, generator=generator)
    q, _ = torch.linalg.qr(response @ omega, mode="reduced")
    output_basis, _ = torch.linalg.qr(response.T @ q, mode="reduced")
    a = torch.linalg.lstsq(
        design, response @ output_basis, driver="gelsd", rcond=rcond
    ).solution
    return a, output_basis.T, "randomized_output_range"


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
    max_solver_bytes=256 * 1024 * 1024,
):
    """Finish a quantized module with two rank-8 fits; never consume eval benchmarks.

    Inputs must be original activations from disjoint calibration documents.
    The caller owns document provenance and row collection. The teacher must
    be the original FP linear module, before quantized replay overwrites it.
    Dense CPU FP64 SVD/lstsq is used while its estimated workspace fits
    ``max_solver_bytes``. Larger projections use a fixed-seed output-range
    sketch whose working factors scale with rank rather than K*N.
    """
    if source_kind != "calibration":
        raise ValueError("only calibration activations may enter recovery fitting")
    _check_documents(train_document_ids, heldout_document_ids)
    if not 0 <= minimum_improvement < 1:
        raise ValueError("minimum_improvement must be in [0, 1)")
    if type(max_solver_bytes) is not int or max_solver_bytes < 1:
        raise ValueError("max_solver_bytes must be a positive integer")
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
            layer.forward_pretransformed(x, output_dtype=original.dtype).float()
            for x, original in zip(transformed, (train_inputs, heldout_inputs))
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
        solver_modes = {}
        for objective in ("output_l2", "tail_weighted_output_l2"):
            weights = torch.ones((x_train.shape[0], 1), dtype=torch.float64)
            if objective == "tail_weighted_output_l2":
                energy = residual[0].square().mean(1, keepdim=True)
                weights = (1 + energy / energy.mean().clamp_min(1e-30)).sqrt()
            fitted_a, fitted_b, solver_mode = _rank8_output_fit(
                x_train,
                residual[0],
                weights,
                max_solver_bytes=max_solver_bytes,
                rcond=1e-5,
                seed=0x51564 + (0 if objective == "output_l2" else 1),
            )
            solver_modes[objective] = solver_mode
            rank = min(8, fitted_a.shape[1], fitted_b.shape[0])
            a = torch.zeros((layer.in_features, 8), dtype=torch.float64)
            b = torch.zeros((8, layer.out_features), dtype=torch.float64)
            a[:, :rank] = fitted_a[:, :rank]
            b[:rank] = fitted_b[:rank]
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
            for x, target, original in zip(transformed, targets, (train_inputs, heldout_inputs)):
                inner = layer._inner_forward(x)
                hidden = (x.float() @ a.float()).half()
                corrected = inner.float() + hidden.float() @ b.float()
                # Invoke the deployed output transform rather than estimating
                # its rounding from the mathematical inverse used above.
                actual = layer._recover_output_compute_dtype(corrected, x.dtype).to(original.dtype).float()
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
            "max_solver_bytes": max_solver_bytes,
            "solver_modes": solver_modes,
            "fit_device": str(layer.trellis.device),
            "activation_dtype": str(train_inputs.dtype),
            "fit_output_boundary": "original_activation_dtype",
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


def _window_artifact_binding_digest(metadata, entries):
    """Hash artifact descriptors and file hashes in native-reproducible order."""
    binding = hashlib.sha256(b"qvq_p32_window_artifact-binding-v1\0")
    for key in (
        "bits",
        "codebook_version",
        "in_features",
        "out_features",
        "input_hadamard",
        "output_hadamard",
    ):
        binding.update(key.encode())
        binding.update(b"\0")
        binding.update(json.dumps(metadata[key], sort_keys=True, separators=(",", ":")).encode())
        binding.update(b"\0")
    for name in sorted(entries):
        entry = entries[name]
        binding.update(name.encode())
        binding.update(b"\0")
        for key in ("dtype", "shape", "bytes", "sha256"):
            binding.update(key.encode())
            binding.update(b"\0")
            binding.update(json.dumps(entry[key], sort_keys=True, separators=(",", ":")).encode())
            binding.update(b"\0")
    return binding.hexdigest()


def save_window_artifact(layer, directory):
    """Write a native-friendly unified artifact directory with a hash manifest.

    The artifact uses the exact tensors from :func:`export_window_package`;
    planar trellis words are never duplicated.  Files are immutable after the
    manifest is written, and the manifest records enough dtype/shape/hash data
    for a non-Python loader to validate the payload before touching a device.
    """
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    package = export_window_package(layer)
    entries = {}
    serialized_bytes = 0
    for name, tensor in package["tensors"].items():
        if not name or Path(name).name != name or Path(name).suffix:
            raise ValueError(f"invalid artifact tensor name: {name!r}")
        value = tensor.detach().cpu().contiguous()
        data = value.view(torch.uint8).numpy().tobytes()
        filename = f"{name}.bin"
        (directory / filename).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        entries[name] = {
            "file": filename,
            "dtype": str(value.dtype).split(".")[-1],
            "shape": list(value.shape),
            "bytes": len(data),
            "sha256": digest,
        }
        serialized_bytes += len(data)
    manifest = {
        "format": "qvq_p32_window_artifact",
        "version": 1,
        "metadata": package["metadata"],
        "recovery": package["recovery"],
        "tensors": entries,
    }
    # Bind the complete descriptor set (including every serialized payload
    # hash) to the module geometry and transform contract. Native loaders can
    # reproduce this delimiter-based digest without implementing Python's
    # tensor serialization or trusting a separately supplied base hash.
    manifest["payload_sha256"] = _window_artifact_binding_digest(
        package["metadata"], entries
    )
    manifest_path = directory / "manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    serialized_bytes += len(manifest_bytes)
    report = window_package_storage([package], serialized_bytes=serialized_bytes)
    report["artifact_directory"] = str(directory)
    report["manifest"] = str(manifest_path)
    return report


def load_window_artifact(directory, *, device="cpu", config=None):
    """Load and verify a hash-manifested native-friendly window artifact."""
    from pathlib import Path

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid window artifact manifest") from exc
    if manifest.get("format") != "qvq_p32_window_artifact" or manifest.get("version") != 1:
        raise ValueError("unsupported window artifact format")
    if not isinstance(manifest.get("metadata"), dict):
        raise TypeError("window artifact metadata is missing")
    entries = manifest.get("tensors")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("window artifact has no tensor manifest")
    payload_sha256 = manifest.get("payload_sha256")
    if payload_sha256 is not None and (
        not isinstance(payload_sha256, str)
        or payload_sha256 != _window_artifact_binding_digest(manifest["metadata"], entries)
    ):
        raise ValueError("window artifact payload binding mismatch")
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "uint8": torch.uint8,
    }
    tensors = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("invalid window artifact tensor name")
        if not isinstance(entry, dict) or entry.get("file") != f"{name}.bin":
            raise ValueError(f"invalid window artifact file entry: {name}")
        dtype = dtypes.get(entry.get("dtype"))
        shape = entry.get("shape")
        if dtype is None or not isinstance(shape, list) or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(f"invalid window artifact tensor metadata: {name}")
        try:
            raw = (directory / entry["file"]).read_bytes()
        except OSError as exc:
            raise ValueError(f"missing window artifact tensor: {name}") from exc
        if len(raw) != entry.get("bytes") or hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise ValueError(f"window artifact tensor hash mismatch: {name}")
        expected = int(torch.tensor([], dtype=dtype).element_size())
        count = 1
        for dim in shape:
            count *= dim
        if len(raw) != count * expected:
            raise ValueError(f"window artifact tensor byte count mismatch: {name}")
        tensors[name] = torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
    recovery = manifest.get("recovery")
    if recovery is not None and not {"rank8_A", "rank8_B"}.issubset(tensors):
        raise ValueError("window artifact recovery factors are incomplete")
    package = {
        "metadata": manifest.get("metadata"),
        "recovery": recovery,
        "tensors": tensors,
    }
    return load_window_package(package, device=device, config=config)


def window_package_storage(packages, *, serialized_bytes=None):
    """Return module and whole-model storage accounting for a unified package.

    Tensor BPW intentionally excludes the outer container/header bytes so it
    remains comparable with the quantizer's weight accounting.  Callers that
    have written a package can provide ``serialized_bytes`` to report the
    actual on-disk cost as a separate field.  A package's ``selected`` flag is
    read from its fitting metadata, so a model-wide quality selection can be
    accounted for without counting every available recovery tensor.
    """
    if serialized_bytes is not None and (
        type(serialized_bytes) is not int or serialized_bytes < 0
    ):
        raise ValueError("serialized_bytes must be a non-negative integer")
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
        recovery = package.get("recovery") or {}
        selected = bool(recovery.get("selected", False)) and recovery_bytes > 0
        weights = k * n
        records.append(
            {
                "weights": weights,
                "window_bpw": 8 * base_bytes / weights,
                "recovered_bpw": 8 * (base_bytes + recovery_bytes) / weights,
                "selected_bpw": 8 * (base_bytes + (recovery_bytes if selected else 0)) / weights,
                "rank8_delta_bpw": 16 * 8 * (k + n) / weights,
                "selected": selected,
                "window_tensor_bytes": base_bytes,
                "recovery_tensor_bytes": recovery_bytes,
                "tensor_bytes": base_bytes + recovery_bytes,
            }
        )
    weights = sum(r["weights"] for r in records)
    window_bytes = sum(r["window_tensor_bytes"] for r in records)
    recovery_bytes = sum(r["recovery_tensor_bytes"] for r in records)
    total_bytes = window_bytes + recovery_bytes
    selected_bytes = sum(
        r["window_tensor_bytes"]
        + (r["recovery_tensor_bytes"] if r["selected"] else 0)
        for r in records
    )
    report = {
        "modules": records,
        "weights": weights,
        "window_tensor_bytes": window_bytes,
        "recovery_tensor_bytes": recovery_bytes,
        "tensor_bytes": total_bytes,
        "window_average_bpw": 8 * window_bytes / weights if weights else 0,
        "recovered_average_bpw": 8 * total_bytes / weights if weights else 0,
        "selected_average_bpw": 8 * selected_bytes / weights if weights else 0,
        # Keep the historical name as the all-tensors (window + available
        # rank8) value; new callers should use the explicit names above.
        "average_bpw": 8 * total_bytes / weights if weights else 0,
        "rank8_modules": sum(r["recovery_tensor_bytes"] > 0 for r in records),
        "selected_rank8_modules": sum(r["selected"] for r in records),
    }
    if serialized_bytes is not None:
        report["serialized_bytes"] = serialized_bytes
        report["serialized_bpw"] = 8 * serialized_bytes / weights if weights else 0
    return report


@dataclass(frozen=True)
class Rank8Calibration:
    """Document-separated original activations supplied to the quantization job."""

    train_inputs: torch.Tensor
    heldout_inputs: torch.Tensor
    train_document_ids: tuple[str, ...]
    heldout_document_ids: tuple[str, ...]
    source_kind: str = "calibration"
    minimum_improvement: float = 0.01
    teacher_hash: str | None = None
    max_solver_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if self.source_kind != "calibration":
            raise ValueError("only calibration activations may enter rank8 fitting")
        _check_documents(self.train_document_ids, self.heldout_document_ids)
        if type(self.max_solver_bytes) is not int or self.max_solver_bytes < 1:
            raise ValueError("max_solver_bytes must be a positive integer")


def finish_rank8_quantization(
    result, original_weight, bias, calibration, *, bits, codebook_version
):
    """Called by the existing quantizer before returning its module result."""
    from dataclasses import replace

    rank8_A, rank8_B, rank8_metadata, report = fit_rank8_serialized_payload(
        result.serialized_tensors(),
        original_weight,
        bias,
        calibration,
        bits=bits,
        codebook_version=codebook_version,
        input_hadamard=result.input_hadamard,
        output_hadamard=result.output_hadamard,
    )
    return replace(
        result,
        rank8_A=rank8_A,
        rank8_B=rank8_B,
        rank8_metadata=rank8_metadata,
        rank8_fit_report=report,
    )


def fit_rank8_serialized_payload(
    serialized_tensors,
    original_weight,
    bias,
    calibration,
    *,
    bits,
    codebook_version,
    input_hadamard=True,
    output_hadamard=True,
):
    """Fit rank8 against an already selected serialized P32 candidate.

    Atomic grouped selection keeps candidate payloads as CPU snapshots and
    chooses the complete gate/up/down tuple only after replay.  This helper
    lets that finalizer fit the selected payload without manufacturing a
    second quantization result object; callers then append the accepted
    factors to the same payload atomically.
    """
    from ..nn_modules.qlinear.qvq import QVQLinear

    n, k = original_weight.shape
    device = original_weight.device
    payload = {
        name: tensor.to(device=device)
        for name, tensor in serialized_tensors.items()
        if name not in RANK8_BUFFERS
    }
    layer = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        bias=bias is not None,
        tensors=payload,
        v2b2_p32=True,
        bank_count=2,
        codebook_version=codebook_version,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    ).eval()
    layer.post_init()
    teacher = torch.nn.Linear(
        k,
        n,
        bias=bias is not None,
        device=original_weight.device,
        dtype=calibration.train_inputs.dtype,
    ).eval()
    with torch.no_grad():
        teacher.weight.copy_(original_weight)
        if bias is not None:
            teacher.bias.copy_(bias)
    if calibration.teacher_hash is not None and calibration.teacher_hash != _digest(
        dict(teacher.named_parameters()), {}
    ):
        raise ValueError("rank8 teacher state changed after activation capture")
    report = fit_rank8(
        layer,
        teacher,
        calibration.train_inputs.to(original_weight.device),
        calibration.heldout_inputs.to(original_weight.device),
        train_document_ids=calibration.train_document_ids,
        heldout_document_ids=calibration.heldout_document_ids,
        source_kind=calibration.source_kind,
        minimum_improvement=calibration.minimum_improvement,
        max_solver_bytes=calibration.max_solver_bytes,
    )
    return layer.rank8_A, layer.rank8_B, layer.rank8_metadata, report


def explicit_window_inner(layer, transformed, config):
    """Expose existing SM90 consumers; no copied A100 crossover or new decoder."""
    from ..utils.qvq_cuda import _pgc16_levels
    from ..utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma,
        qvq_p32_window_wgmma_single_large_m_packed,
        qvq_p32_window_wgmma_tuned,
    )

    rows = transformed.shape[0]
    if transformed.dtype != torch.float16 or not config.min_m <= rows <= config.max_m:
        raise ValueError("input dtype or M is outside the prepared Hopper policy")
    if config.chunk_m and rows > config.chunk_m:
        # Only the window launch is sliced. Input transform,
        # rank8 projection/addition and output transform retain their existing
        # full-operator boundaries; no term is duplicated across these slices.
        chunk_config = replace(config, min_m=1, max_m=config.chunk_m, chunk_m=0)
        return torch.cat(
            [
                explicit_window_inner(layer, chunk, chunk_config)
                for chunk in transformed.split(config.chunk_m, dim=0)
            ],
            dim=0,
        )
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
    padded_rows = (
        ((rows + config.block_m - 1) // config.block_m) * config.block_m
        if config.block_m
        else 16
        if rows <= 16
        else 32
        if rows <= 32
        else ((rows + 63) // 64) * 64
    )
    padded = torch.nn.functional.pad(
        transformed, (0, 0, 0, padded_rows - rows)
    ).contiguous()
    if config.block_m:
        return qvq_p32_window_wgmma_tuned(
            padded,
            window,
            _pgc16_levels(transformed.device, layer.codebook_version),
            banks,
            layer.bits,
            out_features=layer.out_features,
            bank_alt_id=alt_id,
            block_m=config.block_m,
            block_n=config.block_n,
        )[:rows]
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


def window_kernel_candidates(layer, *, m):
    """Enumerate implementations for a prepared quality policy, without timing.

    Native tuners and ZML can benchmark the same explicit candidates. This
    list does not discard a geometry because it lost at a different shape.
    Call prepare_rank8 for the requested quality mode before enumeration;
    each candidate still passes prepare_rank8 before execution/capture.
    """
    if type(m) is not int or not 1 <= m <= 8192:
        raise ValueError("window candidate enumeration requires M in [1,8192]")
    if not hasattr(layer, "_p32_window_config"):
        raise ValueError("prepare the module quality policy before enumerating kernels")
    if getattr(layer, "_p32_rank8_enabled", False):
        validate_rank8_state(layer)
    policy = replace(
        layer._p32_window_config,
        algorithm="production_window",
        block_m=0,
        block_n=0,
        warp_groups=0,
        split_k=1,
        chunk_m=0,
        min_m=m,
        max_m=m,
        recovery_kernel="separate_reference",
        recovery_projection="separate_reference",
    )
    candidates = [policy]
    if layer.trellis.device.type != "cuda":
        return tuple(candidates)
    props = torch.cuda.get_device_properties(layer.trellis.device)
    if (
        (props.major, props.minor) != (9, 0)
        or not any(name in props.name for name in ("H100", "H200"))
        or layer.activation is not None
        or layer.in_features % 256
        or layer.out_features % 256
        or layer.bits not in (2, 2.5, 3, 3.5)
    ):
        return tuple(candidates)
    candidates.extend(
        replace(policy, algorithm=name)
        for name in ("hopper_m16", "hopper_direct_decode_mma")
    )
    candidates.extend(
        replace(
            policy,
            algorithm="hopper_direct_decode_mma",
            block_m=bm,
            block_n=bn,
            warp_groups=bn // 64,
        )
        for bm in (32, 64, 128)
        for bn in (64, 128)
    )
    if m > 4096:
        candidates.extend(
            replace(c, chunk_m=4096)
            for c in tuple(candidates)
            if c.algorithm == "hopper_direct_decode_mma"
        )
    if layer.out_features <= 16384 and (
        not layer.output_hadamard
        or not layer.out_features & (layer.out_features - 1)
    ):
        candidates.extend(
            replace(c, recovery_kernel="fused_epilogue") for c in tuple(candidates)
        )
    if getattr(layer, "_p32_rank8_enabled", False):
        separate_candidates = tuple(candidates)
        candidates.extend(
            replace(c, recovery_projection="tensor_core") for c in separate_candidates
        )
        if layer.in_features <= 16384 and (
            not layer.input_hadamard
            or not layer.in_features & (layer.in_features - 1)
        ):
            candidates.extend(
                replace(c, recovery_projection="input_fused") for c in separate_candidates
            )
    return tuple(candidates)


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
    return window_package_storage(
        [package], serialized_bytes=Path(path).stat().st_size
    )
