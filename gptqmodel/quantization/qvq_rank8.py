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
from itertools import product

import torch

from .qvq import repack_p32_planar_to_window, repack_p32_window_to_planar
from .qvq_codecs import pgc16_levels_for_version
from .qvq_rates import qvq_transition_bits
from .rotation.hadamard_utils import matmul_hadU

CONTRACT = "p32-window-r8-v1:fp32-project,fp16-hidden,fp32-expand-add,existing-output-transform"
RANK8_BUFFERS = ("rank8_A", "rank8_B", "rank8_metadata")
RANK8_SWEEP_CANDIDATES = (2, 4, 6, 8, 12)


@dataclass(frozen=True)
class P32WindowConfig:
    """Static graph policy. Unimplemented kernel choices fail explicitly."""

    abi_version: int = 3
    algorithm: str = "auto"
    recovery_mode: str = "off"
    recovery_kernel: str = "separate_reference"
    recovery_projection: str = "separate_reference"
    arithmetic_signature: str = "reference_fp32_v1"
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
            raise ValueError("kernel geometry must contain integers")
        if self.algorithm == "amd_gfx950":
            if (
                self.chunk_m != 0
                or self.split_k != 1
                or self.block_m not in (16, 32, 64, 128, 256, 512, 1024)
                or self.block_n != 64
                or self.block_k not in (16, 32, 64)
                or self.warp_groups not in (4, 8)
                or self.pipeline_stages not in (1, 2, 3)
            ):
                raise ValueError(
                    "gfx950 geometry requires BM16/32/64/128/256/512/1024, "
                    "BN64, BK16/32/64, warps4/8, stages1/2/3 and split_k=1"
                )
        else:
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
            "ampere_window",
            "hopper_direct_decode_mma",
            "hopper_m16",
            "amd_gfx950",
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
        if self.recovery_projection not in (
            "separate_reference",
            "input_fused",
            "concurrent_reference",
            "tensor_core",
            "project_output_fused",
        ):
            raise ValueError("unsupported rank8 projection implementation")
        if self.recovery_kernel not in ("separate_reference", "fused_epilogue"):
            raise ValueError("fused recovery kernels are not implemented")
        if self.arithmetic_signature not in (
            "reference_fp32_v1",
            "unverified_fused_epilogue",
            "unverified_input_fused",
            "unverified_tensor_core",
            "unverified_project_output_fused",
        ):
            raise ValueError("unsupported rank8 arithmetic signature")


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
        "window_words": (
            layer.window_words
            if getattr(layer, "window_only", False)
            else repack_p32_planar_to_window(layer.trellis, bits=layer.bits)
        ),
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


def _validate_kernel_tuning_metadata(layer, tuning):
    """Validate a serialized tuning hint against the exact live payload."""
    if tuning is None:
        return
    if not isinstance(tuning, dict) or tuning.get("version") != 1:
        raise ValueError("invalid window kernel-tuning metadata")
    try:
        selected = P32WindowConfig.from_backend_config(tuning["selected"])
        identity = tuning["identity"]
        expected_state = identity["state_hash"]
        expected_factors = identity["factors_hash"]
        candidates = tuple(
            P32WindowConfig.from_backend_config(candidate)
            for candidate in identity["candidates"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid selected window kernel policy") from exc
    if selected not in candidates:
        raise ValueError("selected window kernel is absent from measured candidates")
    if tuning.get("quality_mode") != selected.quality_mode:
        raise ValueError("window kernel-tuning quality mode mismatch")
    if type(tuning.get("rank8_enabled")) is not bool:
        raise ValueError("invalid window kernel-tuning correction state")
    paired = tuning.get("candidate_recovery_overhead", [])
    if not isinstance(paired, list):
        raise ValueError("invalid per-candidate recovery timing metadata")
    target = tuning.get("max_recovery_overhead_percent")
    if target is not None and (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not torch.isfinite(torch.tensor(float(target)))
        or float(target) < 0
    ):
        raise ValueError("invalid recovery overhead promotion target")
    for entry in paired:
        if not isinstance(entry, dict) or "config" not in entry or "recovery_overhead" not in entry:
            raise ValueError("invalid per-candidate recovery timing metadata")
        try:
            candidate = P32WindowConfig.from_backend_config(entry["config"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid per-candidate recovery timing configuration") from exc
        if candidate not in candidates:
            raise ValueError("per-candidate recovery timing is not in measured candidates")
        overhead = entry["recovery_overhead"]
        if not isinstance(overhead, dict) or not all(
            isinstance(overhead.get(name), (int, float))
            and torch.isfinite(torch.tensor(float(overhead[name])))
            for name in ("overhead_us", "overhead_percent")
        ):
            raise ValueError("invalid per-candidate recovery timing values")
        eligible = entry.get("recovery_overhead_eligible")
        if eligible is not None and type(eligible) is not bool:
            raise ValueError("invalid per-candidate recovery timing eligibility")
    if expected_state != _digest(*_base(layer)):
        raise ValueError("window kernel-tuning state hash mismatch")
    if expected_factors is not None:
        if layer.rank8_A is None or layer.rank8_B is None:
            raise ValueError("window kernel-tuning factors are missing")
        if expected_factors != _digest({"A": layer.rank8_A, "B": layer.rank8_B}, {}):
            raise ValueError("window kernel-tuning factor hash mismatch")


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
        "window_words",
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
    runtime_device = layer.runtime_device()
    if (
        config.algorithm == "auto"
        and getattr(layer, "window_only", False)
        and runtime_device.type == "cuda"
        and torch.version.hip is None
    ):
        # A window-only deployment has no planar GEMV fallback. Resolve the
        # architecture's existing window consumer during preparation so the
        # default package policy remains graph-safe at replay time.
        properties = torch.cuda.get_device_properties(runtime_device)
        if (properties.major, properties.minor) == (9, 0) and any(
            name in properties.name for name in ("H100", "H200")
        ):
            config = replace(config, algorithm="hopper_m16")
        elif (properties.major, properties.minor) == (8, 0):
            config = replace(config, algorithm="ampere_window")
    if runtime_device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare rank8 and kernel policy before CUDA Graph capture")
    if runtime_device.type == "cuda" and torch.version.hip is None:
        # A policy change can expose the generic GEMV/BF16 rescue branch during
        # a later captured replay. Resolve both native handles while eager.
        from ..utils.qvq_cuda import prewarm_qvq_cuda

        prewarm_qvq_cuda()
    grouped_delegate = getattr(layer, "_qvq_grouped_p32_delegate", None)
    if grouped_delegate is not None:
        state, consumer_index, _ = grouped_delegate
        state.prepare_rank8(consumer_index, layer, config)
        return
    if config.algorithm == "ampere_window":
        if runtime_device.type != "cuda":
            raise ValueError("explicit Ampere policy requires an SM80 CUDA device")
        properties = torch.cuda.get_device_properties(runtime_device)
        if (properties.major, properties.minor) != (8, 0) or not layer.v2b2_p32:
            raise ValueError("unsupported explicit Ampere P32 contract")
        # Pack selectors and load the exact SM80 operator before capture;
        # split-count timing is performed by the external tuner, not here.
        layer._prepare_amd_p32_metadata(runtime_device)
        from ..utils.qvq_ampere_cuda import prewarm_qvq_ampere

        prewarm_qvq_ampere()
        layer._bank_ids_loaded = True
    elif config.algorithm == "amd_gfx950":
        if runtime_device.type != "cuda" or torch.version.hip is None:
            raise ValueError("explicit gfx950 policy requires a ROCm device")
        from ..utils.qvq_amd import qvq_p32_amd_supported

        if (
            not qvq_p32_amd_supported(runtime_device)
            or not layer.v2b2_p32
            or layer.activation is not None
            or layer.in_features % 16
            or layer.out_features % 16
            or layer.bits not in (2, 2.5, 3, 3.5)
        ):
            raise ValueError("unsupported explicit gfx950 P32 contract")
        # Selector/window preparation is the only host-side setup required by
        # the fused Triton consumer.  Its launch specialization is warmed by
        # the first eager call and must remain fixed for graph replay.
        layer._prepare_amd_p32_metadata(runtime_device)
        layer._bank_ids_loaded = True
    elif config.algorithm.startswith("hopper_"):
        if runtime_device.type != "cuda":
            raise ValueError("explicit Hopper policy requires an SM90 CUDA device")
        properties = torch.cuda.get_device_properties(runtime_device)
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
        layer._prepare_amd_p32_metadata(runtime_device)
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
                    or tensor.device != runtime_device
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
        and config.quality_mode in ("balanced", "quality")
        and config.arithmetic_signature != "reference_fp32_v1"
    ):
        raise ValueError(
            "unverified rank8 arithmetic is unavailable in balanced/quality mode"
        )
    if enabled and config.quality_mode in ("balanced", "quality"):
        metadata = _metadata(layer)
        if (
            metadata.get("audit_validated") is not True
            or not isinstance(metadata.get("audit_acceptance"), dict)
            or metadata["audit_acceptance"].get("accepted") is not True
        ):
            raise ValueError("rank8 recovery lacks independent audit confirmation")
    if (
        enabled
        and runtime_device.type == "cuda"
        and torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("rank8 FP32 reference requires CUDA matmul TF32 disabled")
    if (
        config.recovery_kernel == "fused_epilogue"
        and (
            runtime_device.type != "cuda"
            or torch.cuda.get_device_capability(runtime_device) != (9, 0)
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
            runtime_device.type != "cuda"
            or torch.cuda.get_device_capability(runtime_device) != (9, 0)
            or layer.in_features > 16384
            or (layer.input_hadamard and layer.in_features & (layer.in_features - 1))
        )
    ):
        raise ValueError(
            "rank8 input producer requires SM90 and K <= 16384; input Hadamard mode requires power-of-two K"
        )
    if enabled and config.recovery_projection == "tensor_core" and (
        runtime_device.type != "cuda"
        or torch.cuda.get_device_capability(runtime_device) != (9, 0)
    ):
        raise ValueError("rank8 Tensor Core projection requires SM90")
    if enabled and config.recovery_projection == "concurrent_reference":
        if (
            runtime_device.type != "cuda"
            or torch.cuda.get_device_capability(runtime_device) != (9, 0)
            or config.recovery_kernel not in ("separate_reference", "fused_epilogue")
        ):
            raise ValueError("concurrent rank8 projection requires SM90")
        # Stream/event allocation is preparation work and must complete before
        # a caller starts CUDA graph capture.  The actual matrix multiply is
        # warmed per (caller stream, M, K) on its first eager invocation.
        layer._rank8_concurrent_resources(runtime_device)
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
    if enabled and runtime_device.type == "cuda":
        # Keep factor conversion and allocator activity outside any captured
        # graph. The replay path reads these stable FP32 buffers directly.
        layer._cached_rank8_factor("A")
        layer._cached_rank8_factor("B")
    # A new policy or factor version must warm the exact projection shape
    # again before capture; retaining this set would permit a graph to bind a
    # stale A pointer/algorithm after preparation invalidated the old graph.
    if hasattr(layer, "_qvq_rank8_concurrent_warm"):
        layer._qvq_rank8_concurrent_warm.clear()


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
    if layer.runtime_device().type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("rank8 FP32 reference requires CUDA matmul TF32 disabled")


def _project_rank8(layer, transformed):
    if (
        layer._p32_window_config.recovery_projection == "tensor_core"
        and transformed.dtype == torch.float16
    ):
        from ..utils.qvq_rank8_triton import rank8_tensor_core_projection

        return rank8_tensor_core_projection(transformed, layer.rank8_A)
    project_a = layer._cached_rank8_factor("A") if transformed.device.type == "cuda" else layer.rank8_A.float()
    return (transformed.float() @ project_a).half()


def add_rank8_correction(layer, transformed, base, *, hidden=None):
    """No recovery tensor access on the disabled branch. FP16 hidden is deliberate."""
    if not getattr(layer, "_p32_rank8_enabled", False):
        return base
    validate_rank8_state(layer)
    if hidden is None:
        hidden = _project_rank8(layer, transformed)
    expand_b = layer._cached_rank8_factor("B") if hidden.device.type == "cuda" else layer.rank8_B.float()
    correction = hidden.float() @ expand_b
    return base.float() + correction


def fused_rank8_output(layer, transformed, base, compute_dtype, *, hidden=None, output_dtype=None):
    """Fuse expansion/addition into the existing numerical output-transform contract."""
    validate_rank8_state(layer)
    from ..utils.qvq_rank8_triton import rank8_output_epilogue

    enabled = bool(getattr(layer, "_p32_rank8_enabled", False))
    if (
        enabled
        and hidden is None
        and transformed.dtype == torch.float16
        and getattr(layer._p32_window_config, "recovery_projection", None)
        == "project_output_fused"
    ):
        from ..utils.qvq_rank8_triton import rank8_project_output_epilogue

        return rank8_project_output_epilogue(
            transformed,
            layer.rank8_A,
            layer.rank8_B,
            base,
            layer._cached_cast("SV", compute_dtype, base.dtype),
            layer._cached_cast("bias", compute_dtype, base.dtype),
            hadamard=layer.output_hadamard,
            output_dtype=(
                torch.float16 if output_dtype == torch.float16
                and 2048 <= layer.in_features <= 16384
                and not layer.in_features & (layer.in_features - 1)
                else torch.float32
            ),
        )
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


def rank8_audit_acceptance(audit_rows, *, minimum_improvement=0.0):
    """Apply the independent confirmation contract to per-document metrics.

    The fitter's train/held-out ``validated`` bit is deliberately not enough
    to ship factors.  Every independent document must have finite metrics and
    show the requested MSE improvement without worsening the tail error.  The
    returned record is JSON-friendly and is stored in the deployment report.
    """
    if not 0 <= minimum_improvement < 1:
        raise ValueError("minimum_improvement must be in [0, 1)")
    rows = list(audit_rows)
    if not rows:
        return {
            "accepted": False,
            "reason": "empty_independent_audit",
            "documents": [],
        }
    documents = []
    accepted = True
    seen_document_ids = set()
    for row in rows:
        try:
            if not isinstance(row, dict):
                raise TypeError("audit rows must be mappings")
            document_id = row.get("document_id")
            document_rows = row.get("rows", 0)
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id in seen_document_ids
                or type(document_rows) is not int
                or document_rows < 1
            ):
                raise ValueError("invalid audit document identity")
            seen_document_ids.add(document_id)
            baseline = row["baseline"]
            recovered = row["recovered"]
            values = tuple(
                float(baseline[name])
                for name in ("mse", "mae", "max", "tail")
            ) + tuple(
                float(recovered[name])
                for name in ("mse", "mae", "max", "tail")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid rank8 audit metrics") from exc
        finite = all(torch.isfinite(torch.tensor(value)) for value in values)
        nonnegative = all(value >= 0 for value in values)
        baseline_mse, _, _, baseline_tail, recovered_mse, _, _, recovered_tail = values
        mse_ok = recovered_mse <= baseline_mse * (1 - minimum_improvement)
        tail_ok = recovered_tail <= baseline_tail
        document_ok = bool(finite and nonnegative and mse_ok and tail_ok)
        accepted = accepted and document_ok
        documents.append(
            {
                "document_id": row.get("document_id"),
                "rows": document_rows,
                "finite": finite,
                "nonnegative": nonnegative,
                "mse_ok": bool(mse_ok),
                "tail_ok": bool(tail_ok),
                "accepted": document_ok,
            }
        )
    return {
        "accepted": bool(accepted),
        "reason": None if accepted else "independent_audit_regression",
        "documents": documents,
        "minimum_improvement": minimum_improvement,
    }


def apply_rank8_audit(layer, audit_rows, *, minimum_improvement=0.0):
    """Bind independent audit evidence or roll factors back exactly.

    This is intentionally the only promotion boundary used by exporters.  A
    fit that passed train/held-out selection but failed independent confirmation
    has its registered buffers cleared and is marked unavailable for dispatch.
    """
    if layer.rank8_metadata is None:
        gate = {
            "accepted": False,
            "reason": "fit_not_present",
            "documents": [],
        }
        if layer.rank8_A is not None or layer.rank8_B is not None:
            with torch.inference_mode(False):
                layer.rank8_A = None
                layer.rank8_B = None
        prepare_rank8(layer, P32WindowConfig())
        return gate
    metadata = _metadata(layer)
    fit_validated = bool(metadata.get("validated") and metadata.get("selected"))
    gate = rank8_audit_acceptance(
        audit_rows, minimum_improvement=minimum_improvement
    )
    gate["fit_validated"] = fit_validated
    if not fit_validated:
        gate["accepted"] = False
        gate["reason"] = "fit_selection_rejected"
    if not gate["accepted"]:
        with torch.inference_mode(False):
            layer.rank8_A = None
            layer.rank8_B = None
            layer.rank8_metadata = None
        prepare_rank8(layer, P32WindowConfig())
        return gate
    metadata["audit_acceptance"] = gate
    metadata["audit_validated"] = True
    metadata["validated"] = True
    with torch.inference_mode(False):
        layer.rank8_metadata = _encode(metadata, layer.runtime_device())
    return gate


@torch.no_grad()
def _independent_audit_rows(layer, teacher, calibration):
    """Evaluate the optional third calibration fold after fitting.

    The rows are kept document separated so the acceptance record can prove
    that every independent document improved.  This runs outside graph capture
    and is intentionally part of quantization finalization, before factors are
    copied into a deployment result.
    """
    inputs = calibration.audit_inputs
    if inputs is None:
        return []
    if len(calibration.audit_document_ids) != len(calibration.audit_row_counts):
        raise ValueError("independent audit document metadata is incomplete")
    old_config = getattr(layer, "_p32_window_config", P32WindowConfig())
    rows = []
    offset = 0
    try:
        for document_id, count in zip(
            calibration.audit_document_ids, calibration.audit_row_counts
        ):
            x = inputs[offset : offset + count].to(layer.runtime_device())
            offset += count
            prepare_rank8(layer, P32WindowConfig(recovery_mode="off"))
            baseline = layer(x).float()
            prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
            recovered = layer(x).float()
            target = teacher(x).float()
            rows.append(
                {
                    "document_id": document_id,
                    "rows": int(count),
                    "baseline": _metrics(target - baseline),
                    "recovered": _metrics(target - recovered),
                }
            )
        if offset != inputs.shape[0]:
            raise ValueError("independent audit row counts do not cover inputs")
        return rows
    finally:
        prepare_rank8(layer, old_config)


def _check_documents(train_ids, heldout_ids, *additional):
    if not train_ids or not heldout_ids or set(train_ids) & set(heldout_ids):
        raise ValueError("fit and held-out document IDs must be nonempty and disjoint")
    known = set(train_ids) | set(heldout_ids)
    for ids in additional:
        ids = set(ids)
        if ids & known:
            raise ValueError("calibration document IDs must be disjoint")
        known.update(ids)


def _rank8_output_fit(
    x_train,
    residual,
    weights,
    *,
    rank=8,
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
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be a positive integer")
    if type(max_solver_bytes) is not int or max_solver_bytes < 1:
        raise ValueError("max_solver_bytes must be a positive integer")
    design = x_train * weights
    response = residual * weights
    rows, k = design.shape
    n = response.shape[1]
    rank = min(rank, rows, k, n)
    full_bytes = 8 * (k * n + rows * n)
    if full_bytes <= max_solver_bytes:
        solution = torch.linalg.lstsq(design, response, driver="gelsd", rcond=rcond).solution
        _, _, vh = torch.linalg.svd(design @ solution, full_matrices=False)
        rank = min(rank, vh.shape[0])
        a = solution @ vh[:rank].T
        b = vh[:rank]
        return a, b, "full_lstsq_svd"

    # Randomized range finding is deterministic for a fixed contract seed.  The
    # range must be taken from the *predictable* residual P_X R, rather than
    # from R directly: output directions that the transformed activation
    # cannot predict waste the deployed rank.  Four oversampling columns make
    # the range estimate robust while the temporary state remains O(rank).
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sketch_rank = min(rank + 4, n, rows)
    omega = torch.randn((n, sketch_rank), dtype=response.dtype, generator=generator)
    response_sketch = response @ omega
    projected_sketch = design @ torch.linalg.lstsq(
        design, response_sketch, driver="gelsd", rcond=rcond
    ).solution
    q_z, _ = torch.linalg.qr(projected_sketch, mode="reduced")
    # SVD of Q_Z^T R is only sketch_rank x N.  It selects the best deployed
    # rank output directions inside the predictable range without materializing
    # the dense predicted rows x N matrix.
    _, _, vh = torch.linalg.svd(q_z.T @ response, full_matrices=False)
    output_basis = vh[:rank].T.contiguous()
    a = torch.linalg.lstsq(
        design, response @ output_basis, driver="gelsd", rcond=rcond
    ).solution
    return a, output_basis.T, "randomized_output_range"


@torch.no_grad()
def fit_rank_candidates(
    x_train,
    residual,
    *,
    ranks=RANK8_SWEEP_CANDIDATES,
    weights=None,
    max_solver_bytes=256 * 1024 * 1024,
    rcond=1e-5,
    seed=0x51564,
):
    """Fit preparation-time rank candidates with one output-aware contract.

    This is intentionally a solver/reporting API.  It returns FP64 CPU factors
    for each requested rank so a quantization job can compare 2/4/6/8/12 on
    held-out documents before deciding which rank is eligible for deployment.
    Only rank 8 is accepted by the current runtime package and kernels; the
    other candidates are not silently serialized as rank8 tensors.
    """
    if (
        not isinstance(x_train, torch.Tensor)
        or not isinstance(residual, torch.Tensor)
        or x_train.ndim != 2
        or residual.ndim != 2
        or x_train.shape[0] != residual.shape[0]
        or not x_train.shape[0]
        or not x_train.shape[1]
        or not residual.shape[1]
    ):
        raise ValueError("rank sweep requires finite nonempty [rows,K] and [rows,N] matrices")
    if not torch.isfinite(x_train).all() or not torch.isfinite(residual).all():
        raise ValueError("rank sweep inputs must be finite")
    normalized = tuple(ranks)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(type(r) is not int or r not in RANK8_SWEEP_CANDIDATES for r in normalized)
    ):
        raise ValueError(
            f"ranks must be distinct members of {RANK8_SWEEP_CANDIDATES}"
        )
    if weights is None:
        weights = torch.ones((x_train.shape[0], 1), dtype=torch.float64)
    if (
        not isinstance(weights, torch.Tensor)
        or weights.shape != (x_train.shape[0], 1)
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("rank sweep weights must be finite positive [rows,1]")
    x_cpu = x_train.detach().to(device="cpu", dtype=torch.float64).contiguous()
    residual_cpu = residual.detach().to(device="cpu", dtype=torch.float64).contiguous()
    weights_cpu = weights.detach().to(device="cpu", dtype=torch.float64).contiguous()
    result = {}
    for candidate_rank in normalized:
        a, b, solver_mode = _rank8_output_fit(
            x_cpu,
            residual_cpu,
            weights_cpu,
            rank=candidate_rank,
            max_solver_bytes=max_solver_bytes,
            rcond=rcond,
            seed=seed + candidate_rank,
        )
        predicted = (x_cpu * weights_cpu) @ a @ b
        weighted_error = residual_cpu * weights_cpu - predicted
        result[candidate_rank] = {
            "rank": candidate_rank,
            "effective_rank": int(a.shape[1]),
            "solver_mode": solver_mode,
            "A": a,
            "B": b,
            "weighted_fit": _metrics(weighted_error),
        }
    return result


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
    rank_candidates=(8,),
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
    rank_candidates = tuple(rank_candidates)
    if (
        not rank_candidates
        or 8 not in rank_candidates
        or len(set(rank_candidates)) != len(rank_candidates)
        or any(
            type(rank) is not int or rank not in RANK8_SWEEP_CANDIDATES
            for rank in rank_candidates
        )
    ):
        raise ValueError(
            f"rank_candidates must include distinct members of {RANK8_SWEEP_CANDIDATES}, including 8"
        )
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
        rank_sweep = {}
        for objective in ("output_l2", "tail_weighted_output_l2"):
            weights = torch.ones((x_train.shape[0], 1), dtype=torch.float64)
            if objective == "tail_weighted_output_l2":
                energy = residual[0].square().mean(1, keepdim=True)
                weights = (1 + energy / energy.mean().clamp_min(1e-30)).sqrt()
            objective_sweep = {}
            for candidate_rank in rank_candidates:
                fitted_a, fitted_b, solver_mode = _rank8_output_fit(
                    x_train,
                    residual[0],
                    weights,
                    rank=candidate_rank,
                    max_solver_bytes=max_solver_bytes,
                    rcond=1e-5,
                    seed=0x51564
                    + candidate_rank
                    + (0 if objective == "output_l2" else 1),
                )
                objective_sweep[str(candidate_rank)] = {
                    "rank": candidate_rank,
                    "effective_rank": min(
                        candidate_rank, fitted_a.shape[1], fitted_b.shape[0]
                    ),
                    "solver_mode": solver_mode,
                }
                effective_rank = min(
                    candidate_rank, fitted_a.shape[1], fitted_b.shape[0]
                )
                a64 = fitted_a[:, :effective_rank].contiguous()
                b64 = fitted_b[:effective_rank].contiguous()
                # final correction = H(inner correction) * SV, so inner B =
                # H^T(final B / SV). transpose=True matters for composite widths.
                b64 = b64 / layer.SV.detach().double().cpu()
                if layer.output_hadamard:
                    b64 = matmul_hadU(b64, transpose=True)
                a_eval = a64.to(device=layer.trellis.device, dtype=torch.float16)
                b_eval = b64.to(device=layer.trellis.device, dtype=torch.float16)
                if not torch.isfinite(a_eval).all() or not torch.isfinite(b_eval).all():
                    continue
                scores = []
                for x, target, original in zip(
                    transformed, targets, (train_inputs, heldout_inputs)
                ):
                    inner = layer._inner_forward(x)
                    hidden = (x.float() @ a_eval.float()).half()
                    corrected = inner.float() + hidden.float() @ b_eval.float()
                    # Invoke the deployed output transform rather than estimating
                    # its rounding from the mathematical inverse used above.
                    actual = layer._recover_output_compute_dtype(
                        corrected, x.dtype
                    ).to(original.dtype).float()
                    scores.append(_metrics(target - actual))
                if all(
                    all(torch.isfinite(torch.tensor(v)) for v in score.values())
                    for score in scores
                ):
                    objective_sweep[str(candidate_rank)]["scores"] = scores
                    if candidate_rank == 8:
                        solver_modes[objective] = solver_mode
                        a = torch.zeros((layer.in_features, 8), dtype=torch.float64)
                        b = torch.zeros((8, layer.out_features), dtype=torch.float64)
                        a[:, :effective_rank] = a64
                        b[:effective_rank] = b64
                        a = a.to(device=layer.trellis.device, dtype=torch.float16)
                        b = b.to(device=layer.trellis.device, dtype=torch.float16)
                        candidates.append((objective, a, b, scores))
            rank_sweep[objective] = objective_sweep
        baseline = [_metrics(r) for r in residual]
        for objective_sweep in rank_sweep.values():
            for candidate in objective_sweep.values():
                scores = candidate.get("scores")
                if scores is None:
                    candidate["eligible"] = False
                    candidate["selected_quality"] = False
                    continue
                candidate["eligible"] = all(
                    score["mse"] < reference["mse"]
                    and score["tail"] <= reference["tail"]
                    for score, reference in zip(scores, baseline)
                )
                candidate["selected_quality"] = candidate["eligible"] and all(
                    score["mse"] < reference["mse"] * (1 - minimum_improvement)
                    for score, reference in zip(scores, baseline)
                )
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
            "rank_candidates": list(rank_candidates),
            "rank_sweep": rank_sweep,
            "fit_device": str(layer.trellis.device),
            "activation_dtype": str(train_inputs.dtype),
            "fit_output_boundary": "original_activation_dtype",
            "torch_version": str(torch.__version__),
            "objective": None if selected is None else selected[0],
            "audit_validated": False,
            "audit_acceptance": None,
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
            if (
                recovery.get("audit_validated") is not True
                or not isinstance(recovery.get("audit_acceptance"), dict)
                or recovery["audit_acceptance"].get("accepted") is not True
            ):
                raise ValueError(
                    "rank8 export requires independent audit confirmation"
                )
            tensors.update(rank8_A=layer.rank8_A, rank8_B=layer.rank8_B)
        finally:
            prepare_rank8(layer, old_config)
    tuning = getattr(layer, "_p32_window_tuning", None)
    if tuning is not None:
        if not isinstance(tuning, dict) or tuning.get("version") != 1:
            raise ValueError("invalid window kernel-tuning metadata")
        try:
            P32WindowConfig.from_backend_config(tuning["selected"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid selected window kernel policy") from exc
        _validate_kernel_tuning_metadata(layer, tuning)
        # Normalize tuples from the device identity to JSON-compatible lists
        # before placing metadata in a portable torch/native package.
        tuning = json.loads(json.dumps(tuning, sort_keys=True))
    return {
        "metadata": metadata,
        "recovery": recovery,
        "kernel_tuning": tuning,
        "tensors": {
            k: v.detach().cpu().contiguous().clone()
            for k, v in tensors.items()
            if v is not None
        },
    }


def load_window_package(package, *, device="cpu", config=None, retain_planar=False):
    """Load a package with window-owned execution storage.

    CUDA loads release the temporary CPU planar reconstruction by default. A
    legacy/debug caller can request ``retain_planar=True`` explicitly; window
    kernels never depend on that copy.
    """
    if not isinstance(retain_planar, bool):
        raise TypeError("retain_planar must be a bool")
    from ..nn_modules.qlinear.qvq import QVQLinear

    metadata = dict(package["metadata"])
    tuning = package.get("kernel_tuning")
    if tuning is not None:
        if not isinstance(tuning, dict) or tuning.get("version") != 1:
            raise ValueError("invalid window kernel-tuning metadata")
        try:
            P32WindowConfig.from_backend_config(tuning["selected"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid selected window kernel policy") from exc
    tensors = {k: v.to(device) for k, v in package["tensors"].items()}
    recovery = package["recovery"]
    a, b = tensors.pop("rank8_A", None), tensors.pop("rank8_B", None)
    if recovery is not None and recovery.get("base_hash") != _digest(tensors, metadata):
        raise ValueError("window package base hash mismatch")
    if recovery is not None and (
        recovery.get("audit_validated") is not True
        or not isinstance(recovery.get("audit_acceptance"), dict)
        or recovery["audit_acceptance"].get("accepted") is not True
    ):
        raise ValueError("window package recovery lacks independent audit confirmation")
    if recovery is None and (a is not None or b is not None):
        raise ValueError("recovery factors require fitting metadata")
    levels = tensors.pop("levels")
    if not torch.equal(
        levels.cpu(), pgc16_levels_for_version(metadata["codebook_version"]).cpu()
    ):
        raise ValueError("window package codebook levels mismatch")
    window_words = tensors["window_words"]
    if str(device) == "cpu":
        planar = repack_p32_window_to_planar(window_words, bits=metadata["bits"])
        window_only = False
        tensors.pop("window_words")
    else:
        # The deployment package owns the continuous window on the execution
        # device.  Keep the canonical planar reconstruction CPU-side only for
        # legacy/debug access; Hopper/Ampere window consumers never repack or
        # retain a device planar copy.
        planar = repack_p32_window_to_planar(
            package["tensors"]["window_words"], bits=metadata["bits"]
        )
        window_only = True
    tensors["trellis"] = planar
    tensors["window_words"] = window_words
    layer = QVQLinear(
        **metadata,
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
        bias="bias" in tensors,
        window_only=window_only,
    ).eval()
    if recovery is not None:
        layer.rank8_A, layer.rank8_B = a, b
        layer.rank8_metadata = _encode(recovery, device)
        layer.post_init()
        prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    _validate_kernel_tuning_metadata(layer, tuning)
    layer._p32_window_tuning = tuning
    prepare_rank8(layer, config or P32WindowConfig())
    if window_only and not retain_planar:
        # The constructor needs a temporary planar tensor for legacy shape and
        # selector validation.  Release it after all setup so production CUDA
        # ownership is solely the continuous window payload.
        with torch.inference_mode(False):
            layer.trellis = None
        layer._validate_tensors()
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
        "kernel_tuning": package.get("kernel_tuning"),
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


def load_window_artifact(directory, *, device="cpu", config=None, retain_planar=False):
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
        "kernel_tuning": manifest.get("kernel_tuning"),
        "tensors": tensors,
    }
    return load_window_package(
        package, device=device, config=config, retain_planar=retain_planar
    )


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
    rank_candidates: tuple[int, ...] = (8,)
    # Optional third fold.  It is deliberately separate from train/held-out
    # fitting and is consumed only by the independent audit gate.
    audit_inputs: torch.Tensor | None = None
    audit_document_ids: tuple[str, ...] = ()
    audit_row_counts: tuple[int, ...] = ()

    def __post_init__(self):
        if self.source_kind != "calibration":
            raise ValueError("only calibration activations may enter rank8 fitting")
        _check_documents(self.train_document_ids, self.heldout_document_ids)
        if type(self.max_solver_bytes) is not int or self.max_solver_bytes < 1:
            raise ValueError("max_solver_bytes must be a positive integer")
        ranks = tuple(self.rank_candidates)
        if (
            not ranks
            or 8 not in ranks
            or len(set(ranks)) != len(ranks)
            or any(
                type(rank) is not int or rank not in RANK8_SWEEP_CANDIDATES
                for rank in ranks
            )
        ):
            raise ValueError(
                f"rank_candidates must include distinct members of {RANK8_SWEEP_CANDIDATES}, including 8"
            )
        object.__setattr__(self, "rank_candidates", ranks)
        audit_ids = tuple(self.audit_document_ids)
        counts = tuple(self.audit_row_counts)
        if self.audit_inputs is None:
            if audit_ids or counts:
                raise ValueError("audit document metadata requires audit_inputs")
        else:
            if (
                self.audit_inputs.ndim != 2
                or not self.audit_inputs.shape[0]
                or self.audit_inputs.shape[1] != self.train_inputs.shape[1]
                or not torch.isfinite(self.audit_inputs).all()
                or not audit_ids
                or len(audit_ids) != len(counts)
                or any(type(v) is not int or v < 1 for v in counts)
                or sum(counts) != self.audit_inputs.shape[0]
            ):
                raise ValueError("audit_inputs and document row counts are invalid")
            if any(not isinstance(v, str) or not v for v in audit_ids):
                raise ValueError("audit document IDs must be nonempty strings")
            if set(audit_ids) & (set(self.train_document_ids) | set(self.heldout_document_ids)):
                raise ValueError("audit documents must be disjoint from fit documents")
        object.__setattr__(self, "audit_document_ids", audit_ids)
        object.__setattr__(self, "audit_row_counts", counts)


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
        rank_candidates=calibration.rank_candidates,
    )
    if calibration.audit_inputs is not None and layer.rank8_metadata is not None:
        audit_gate = apply_rank8_audit(
            layer,
            _independent_audit_rows(layer, teacher, calibration),
            minimum_improvement=calibration.minimum_improvement,
        )
        report["audit_acceptance"] = audit_gate
        report["audit_validated"] = bool(audit_gate["accepted"])
        report["audit_input_hash"] = _digest(
            {"audit_inputs": calibration.audit_inputs}, {}
        )
        report["audit_document_ids"] = list(calibration.audit_document_ids)
        report["audit_row_counts"] = list(calibration.audit_row_counts)
        if audit_gate["accepted"]:
            metadata = _metadata(layer)
            metadata.update(
                audit_input_hash=report["audit_input_hash"],
                audit_document_ids=report["audit_document_ids"],
                audit_row_counts=report["audit_row_counts"],
            )
            with torch.inference_mode(False):
                layer.rank8_metadata = _encode(metadata, layer.runtime_device())
        report["validated"] = bool(report.get("validated") and audit_gate["accepted"])
        if not audit_gate["accepted"]:
            report["selected"] = False
    return layer.rank8_A, layer.rank8_B, layer.rank8_metadata, report


def explicit_window_inner(layer, transformed, config):
    """Dispatch one prepared, explicit window consumer by architecture."""
    from ..utils.qvq_cuda import _pgc16_levels
    from ..utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma,
        qvq_p32_window_wgmma_single_large_m_packed,
        qvq_p32_window_wgmma_tuned,
    )

    rows = transformed.shape[0]
    if transformed.dtype == torch.bfloat16:
        # The existing native window consumers are FP16-input kernels.  The
        # outer BF16 overflow-rescue branch may still reach this operator;
        # narrow that fixed-shape operand without reintroducing planar state.
        transformed = transformed.to(torch.float16)
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
    if config.algorithm == "amd_gfx950":
        from ..utils.qvq_amd import QVQAMDLaunchConfig, qvq_p32_amd

        return qvq_p32_amd(
            transformed.contiguous(),
            window,
            _pgc16_levels(transformed.device, layer.codebook_version),
            banks,
            layer.bits,
            out_features=layer.out_features,
            bank_alt_id=alt_id,
            output_fp32=True,
            cache_weight=False,
            launch_config=QVQAMDLaunchConfig(
                block_m=config.block_m,
                block_n=config.block_n,
                block_k=config.block_k,
                num_warps=config.warp_groups,
                num_stages=config.pipeline_stages,
            ),
        )
    if config.algorithm == "ampere_window":
        from ..utils.qvq_ampere_cuda import qvq_p32_window_ampere

        return qvq_p32_window_ampere(
            transformed.contiguous(),
            window,
            _pgc16_levels(transformed.device, layer.codebook_version),
            banks,
            layer.bits,
            out_features=layer.out_features,
            bank_alt_id=alt_id,
            split_count=config.split_k,
        )
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
    runtime_device = layer.runtime_device()
    if runtime_device.type != "cuda":
        base_algorithm = "production_window"
        props = None
    else:
        props = torch.cuda.get_device_properties(runtime_device)
        # A window-only artifact has no planar device fallback.  Keep the
        # baseline candidate on the architecture's graph-safe window consumer
        # so tuning cannot select ``production_window`` and later hit the
        # BF16/reference branch during CUDA Graph replay.
        base_algorithm = "production_window"
        if (
            getattr(layer, "window_only", False)
            and torch.version.hip is None
            and layer.activation is None
            and layer.bits in (2, 2.5, 3, 3.5)
            and layer.v2b2_p32
            and layer.in_features % 256 == 0
            and layer.out_features % 256 == 0
            and (props.major, props.minor) == (9, 0)
            and any(name in props.name for name in ("H100", "H200"))
        ):
            base_algorithm = "hopper_m16"
        elif (
            getattr(layer, "window_only", False)
            and torch.version.hip is None
            and layer.activation is None
            and layer.bits in (2, 2.5, 3, 3.5)
            and layer.v2b2_p32
            and (props.major, props.minor) == (8, 0)
        ):
            base_algorithm = "ampere_window"
    policy = replace(
        layer._p32_window_config,
        algorithm=base_algorithm,
        block_m=0,
        block_n=0,
        warp_groups=0,
        split_k=1,
        chunk_m=0,
        min_m=m,
        max_m=m,
        recovery_kernel="separate_reference",
        recovery_projection="separate_reference",
        arithmetic_signature="reference_fp32_v1",
    )
    candidates = [policy]
    if runtime_device.type != "cuda":
        return tuple(candidates)
    if (
        layer.activation is not None
        or layer.bits not in (2, 2.5, 3, 3.5)
        or not layer.v2b2_p32
    ):
        return tuple(candidates)
    if torch.version.hip is not None:
        from ..utils.qvq_amd import (
            qvq_p32_amd_kernel_candidates,
            qvq_p32_amd_supported,
        )

        if qvq_p32_amd_supported(layer.runtime_device()):
            # gfx950 has a Triton consumer with independent M/K/stage choices;
            # expose those exact launch controls to the same tuner contract.
            # Recovery implementation variants remain SM90-only until their
            # arithmetic and graph signatures are certified on ROCm.
            amd_candidates = tuple(
                replace(
                    policy,
                    algorithm="amd_gfx950",
                    block_m=launch.block_m,
                    block_n=launch.block_n,
                    block_k=launch.block_k,
                    warp_groups=launch.num_warps,
                    pipeline_stages=launch.num_stages,
                )
                for launch in qvq_p32_amd_kernel_candidates(
                    m, layer.out_features, layer.in_features
                )
            )
            if amd_candidates:
                return amd_candidates
    if (props.major, props.minor) == (8, 0):
        from ..utils.qvq_ampere_cuda import qvq_p32_window_ampere_kernel_candidates

        sm_count = int(props.multi_processor_count)
        splits = qvq_p32_window_ampere_kernel_candidates(
            (m, layer.in_features),
            out_features=layer.out_features,
            bits=layer.bits,
            sm_count=sm_count,
        )
        return tuple(
            replace(
                policy,
                algorithm="ampere_window",
                split_k=split,
            )
            for split in splits
        )
    if (
        (props.major, props.minor) != (9, 0)
        or not any(name in props.name for name in ("H100", "H200"))
        or layer.in_features % 256
        or layer.out_features % 256
    ):
        return tuple(candidates)
    hopper_algorithms = ("hopper_m16", "hopper_direct_decode_mma")
    if base_algorithm == "hopper_m16":
        hopper_algorithms = ("hopper_direct_decode_mma",)
    candidates.extend(replace(policy, algorithm=name) for name in hopper_algorithms)
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
    # Alternate epilogues have not yet passed the independent arithmetic
    # signature certification.  Keep them visible for fast/off experiments,
    # but do not make them eligible for balanced/quality correction tuning.
    allow_unverified = (
        not getattr(layer, "_p32_rank8_enabled", False)
        or layer._p32_window_config.quality_mode == "fast"
    )
    if allow_unverified and layer.out_features <= 16384 and (
        not layer.output_hadamard
        or not layer.out_features & (layer.out_features - 1)
    ):
        candidates.extend(
            replace(
                c,
                recovery_kernel="fused_epilogue",
                arithmetic_signature="unverified_fused_epilogue",
            )
            for c in tuple(candidates)
        )
    if getattr(layer, "_p32_rank8_enabled", False):
        # This uses the same reference FP32 projection as the separate path;
        # only its stream placement changes.  It is therefore eligible for
        # balanced/quality tuning once the caller has prepared the paired
        # stream/event resources and warmed the exact M/K shape.
        candidates.extend(
            replace(
                c,
                recovery_projection="concurrent_reference",
                arithmetic_signature="reference_fp32_v1",
            )
            for c in tuple(candidates)
            if c.recovery_projection == "separate_reference"
        )
    if getattr(layer, "_p32_rank8_enabled", False) and allow_unverified:
        separate_candidates = tuple(candidates)
        candidates.extend(
            replace(
                c,
                recovery_projection="tensor_core",
                arithmetic_signature="unverified_tensor_core",
            )
            for c in separate_candidates
        )
        if layer.in_features <= 16384 and (
            not layer.input_hadamard
            or not layer.in_features & (layer.in_features - 1)
        ):
            candidates.extend(
                replace(
                    c,
                    recovery_projection="input_fused",
                    arithmetic_signature="unverified_input_fused",
                )
                for c in separate_candidates
            )
        if layer.in_features <= 16384:
            candidates.extend(
                replace(
                    c,
                    recovery_kernel="fused_epilogue",
                    recovery_projection="project_output_fused",
                    arithmetic_signature="unverified_project_output_fused",
                )
                for c in separate_candidates
            )
    return tuple(candidates)


def grouped_window_kernel_candidates(layers, *, m):
    """Enumerate one shared-transform policy for a grouped P32 projection.

    SM80 grouped consumers need a split count per child because the optimum is
    a function of each child's ``N``.  Return candidate tuples in child order
    so the grouped runtime and an external tuner can benchmark/select one
    complete policy without collapsing the widths into a synthetic total.
    Hopper grouped consumers use the same policy tuple with child-local
    ordered split counts.  The grouped runtime consumes these policies when
    prepared; unsupported BM/BN geometry remains rejected explicitly rather
    than silently ignored.
    Enumeration is shape arithmetic only and performs no CUDA work.
    """

    children = tuple(layers)
    if len(children) not in (2, 3):
        raise ValueError("grouped window candidates require two or three children")
    if type(m) is not int or not 1 <= m <= 8192:
        raise ValueError("grouped window candidate enumeration requires M in [1,8192]")
    for child in children:
        if not hasattr(child, "_p32_window_config"):
            raise ValueError("prepare every grouped module policy before enumeration")
        if getattr(child, "_p32_rank8_enabled", False):
            validate_rank8_state(child)
        if (
            not child.v2b2_p32
            or child.in_features % 16
            or child.out_features % 16
        ):
            raise ValueError("grouped window candidates require aligned V2B2-P32 modules")
    reference = children[0]
    for child in children[1:]:
        if (
            child.runtime_device() != reference.runtime_device()
            or child.in_features != reference.in_features
            or child.bits != reference.bits
            or child.codebook_version != reference.codebook_version
        ):
            raise ValueError("grouped window candidates require one shared P32 shape/rate")

    def policy(child, *, algorithm, split_k):
        return replace(
            child._p32_window_config,
            algorithm=algorithm,
            block_m=0,
            block_n=0,
            warp_groups=0,
            split_k=split_k,
            chunk_m=0,
            min_m=m,
            max_m=m,
            # Grouped tuning changes only the window consumer geometry. Keep
            # each child's already-prepared rank8 arithmetic policy visible so
            # a grouped benchmark measures the requested fused/reference path
            # rather than silently replacing it with a separate correction.
            recovery_kernel=child._p32_window_config.recovery_kernel,
            recovery_projection=child._p32_window_config.recovery_projection,
            arithmetic_signature=child._p32_window_config.arithmetic_signature,
        )

    device = reference.runtime_device()
    if device.type != "cuda":
        return (
            tuple(
                policy(child, algorithm="production_window", split_k=1)
                for child in children
            ),
        )
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (8, 0):
        if (properties.major, properties.minor) != (9, 0):
            return (
                tuple(
                    policy(child, algorithm="production_window", split_k=1)
                    for child in children
                ),
            )

        # Hopper's grouped consumer exposes ordered split-K, while BM/BN
        # geometry is currently a single-child control.  Enumerate a compact,
        # shape-valid per-child split set so callers can benchmark complete
        # tuples without deriving a synthetic total-N policy.  Include the
        # measured H100 tuple first when one exists; on other SM90 devices the
        # same explicit alternatives remain available for local tuning.
        from ..utils.qvq_wgmma_cuda import qvq_h100_grouped_ordered_split_counts

        transition_bits = qvq_transition_bits(
            reference.bits, vector_size=reference.vector_size
        )
        measured = qvq_h100_grouped_ordered_split_counts(
            device_name=properties.name,
            compute_capability=(properties.major, properties.minor),
            in_features=reference.in_features,
            out_features=tuple(child.out_features for child in children),
            transition_bits=transition_bits,
        )
        k_tiles = reference.in_features // 16
        base_values = (1, 2, 4, 8)
        options = []
        for index, child in enumerate(children):
            values = list(base_values)
            if measured is not None:
                values.append(int(measured[index]))
            values = sorted(
                {
                    split
                    for split in values
                    if split >= 1
                    and k_tiles % split == 0
                    and (k_tiles // split) % 16 == 0
                }
            )
            if not values:
                raise ValueError("grouped Hopper shape has no valid split policy")
            options.append(tuple(values))
        split_tuples = list(product(*options))
        if measured is not None:
            measured = tuple(int(value) for value in measured)
            split_tuples.sort(key=lambda value: (value != measured, value))
        return tuple(
            tuple(
                policy(child, algorithm="hopper_m16", split_k=split)
                for child, split in zip(children, split_counts, strict=True)
            )
            for split_counts in split_tuples
        )

    from ..utils.qvq_ampere_cuda import (
        qvq_p32_window_ampere_grouped_kernel_candidates,
    )

    split_tuples = qvq_p32_window_ampere_grouped_kernel_candidates(
        (m, reference.in_features),
        out_features=tuple(child.out_features for child in children),
        bits=reference.bits,
        sm_count=int(properties.multi_processor_count),
    )
    return tuple(
        tuple(
            policy(child, algorithm="ampere_window", split_k=split)
            for child, split in zip(children, split_counts, strict=True)
        )
        for split_counts in split_tuples
    )


def window_tuning_key(layer, *, m, quality_mode, tp_world_size=1, tp_rank=0, build_id):
    """External tuner key; quality eligibility is resolved before latency tuning."""
    if not 0 <= tp_rank < tp_world_size or quality_mode not in (
        "fast",
        "balanced",
        "quality",
    ):
        raise ValueError("invalid quality/TP policy")
    device = layer.runtime_device()
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
