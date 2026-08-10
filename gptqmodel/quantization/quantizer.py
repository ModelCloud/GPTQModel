# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import contextlib
import os

import torch
import torch.nn as nn

from ..utils.logger import setup_logger
from .config import (
    AdaptiveClippingConfig,
    AdaptiveClippingMetric,
    BaseQuantizeConfig,
    ScaleSearchConfig,
    _normalize_quant_bits,
    resolve_quant_format,
)


try:
    from ..nn_modules.qlinear.pack_block_ext import find_params_batched_cpu as _find_params_batched_cpu
except Exception:
    _find_params_batched_cpu = None

from ..utils.backend import BACKEND
from ..utils.marlin import marlin_runtime_available, replace_parameter
from ..utils.torch import linalg_cholesky


log = setup_logger()

HF_OPTIMUM = "hf_optimum"

# Bound temporary candidate tensors while still amortizing eager CUDA launch
# overhead for the 128-column groups used by GPTQ.
SCALE_SEARCH_TARGET_ELEMENTS = 64 * 1024 * 1024
SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK = 16
# Correlated objectives benefit more from large GEMMs than elementwise
# objectives. Give them a larger, still-bounded workspace without increasing
# activation or MSE search memory.
CORRELATED_SCALE_SEARCH_TARGET_ELEMENTS = 128 * 1024 * 1024
CORRELATED_SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK = 80
# Adaptive clipping only searches a small, user-controlled candidate list. The q
# and diff temporaries are the dominant per-group allocations; cap chunk size so
# a long candidate list still has bounded peak memory. The default candidate list
# has 4 entries, so a cap of 4 lets the common case run in one chunk while
# keeping the per-candidate scratch small.
ADAPTIVE_CLIP_MAX_CANDIDATES_PER_CHUNK = 4

MARLIN_SCALE_SEARCH_METHODS = frozenset({
    ScaleSearchConfig.MARLIN,
    ScaleSearchConfig.MARLIN_MSE,
    ScaleSearchConfig.MARLIN_ACTIVATION,
})


def quantize(x, scale, zero, maxq, requires_groupwise_processing: bool):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    if requires_groupwise_processing:
        q = torch.clamp(torch.round(x / scale), -maxq, maxq)
        return scale * q
    else:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, qcfg: BaseQuantizeConfig, shape=1, name: str=None, region_timer=None):
        super(Quantizer, self).__init__()

        self.qcfg = qcfg
        self.region_timer = region_timer
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        self.name=name

    def requires_groupwise_processing(self) -> bool:
        return False

    @staticmethod
    def _scale_search_shrink_factors(candidate_count: int, grid: int, device: torch.device) -> torch.Tensor:
        """Build the FP32 shrink grid used by the pre-optimization GPTQ path."""

        return 1 - torch.arange(candidate_count, device=device, dtype=torch.float32) / grid

    # FIXME, optimum shouldn't call this directly, it should call hf_configure
    def configure(
        self,
        perchannel=False,
        grid=100,
        maxshrink=0.8,
        trits=False,
        bits: int | str | None = None, # for hf compat
        sym: bool | None = None, # for hf compat
    ):
        if self.name == HF_OPTIMUM:
            if bits is not None:
                self.qcfg.bits = _normalize_quant_bits(bits, format_value=resolve_quant_format(self.qcfg.format, self.qcfg.method))
            if sym is not None:
                self.qcfg.sym = sym

        if self.requires_groupwise_processing():
            maxq_value = 2 ** (self.qcfg.bits - 1) - 1
        else:
            maxq_value = 2 ** self.qcfg.bits - 1

        self.perchannel = perchannel
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            maxq_value = -1
        self._maxq_value = maxq_value
        self.maxq = torch.tensor(maxq_value)

    def _prepare_scale_search_hessian(
        self,
        hessian: torch.Tensor | None,
        *,
        method: ScaleSearchConfig,
        columns: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Prepare only the Hessian data required by the selected objective."""

        if hessian is None:
            return None
        if method in {ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.MARLIN_ACTIVATION}:
            # Activation search only consumes the diagonal. Avoiding a full
            # symmetric Hessian copy is especially important for 8192-wide MLPs.
            if hessian.ndim == 1 and hessian.shape == (columns,):
                importance = hessian.detach().to(device=device, dtype=torch.float32)
            elif hessian.ndim == 2 and hessian.shape == (columns, columns):
                importance = hessian.detach().diagonal().to(device=device, dtype=torch.float32)
            else:
                raise ValueError(
                    "Quantizer.find_params(): `hessian` must have shape "
                    f"({columns},) or ({columns}, {columns}) for activation scale search, "
                    f"got {tuple(hessian.shape)}."
                )
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
            diagonal_mean = importance.mean()
            valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
            safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
            normalized = importance / safe_mean
            return torch.where(valid, normalized, torch.ones_like(normalized))

        if method == ScaleSearchConfig.MARLIN_MSE:
            # Plain reconstruction MSE does not need Hessian or activation data.
            return None

        if method in {
            ScaleSearchConfig.HESSIAN,
            ScaleSearchConfig.HYBRID,
            ScaleSearchConfig.MARLIN,
        }:
            if hessian.ndim != 2 or hessian.shape != (columns, columns):
                raise ValueError(
                    "Quantizer.find_params(): `hessian` must have shape "
                    f"({columns}, {columns}) for {method.value} scale search, got {tuple(hessian.shape)}."
                )

            prepared = hessian.detach().to(device=device, dtype=torch.float32)
            prepared = torch.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
            prepared = (prepared + prepared.t()) * 0.5
            diagonal_mean = prepared.diagonal().clamp_min(0).mean()
            if not torch.isfinite(diagonal_mean) or diagonal_mean <= 0:
                return None
            prepared.div_(diagonal_mean)
            if method == ScaleSearchConfig.HYBRID:
                # Fold 50/50 diagonal shrinkage into the Hessian once instead of
                # allocating and reducing another candidate-sized squared-error
                # tensor for every scale-search chunk. Off-diagonal terms receive
                # half weight while diagonal terms retain their original weight.
                prepared.mul_(0.5)
                prepared.diagonal().mul_(2.0)
            return prepared

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    def _scale_search_error(
        self,
        error: torch.Tensor,
        *,
        method: ScaleSearchConfig,
        mse: float,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score one or more clipping candidates for every output row."""

        if method == ScaleSearchConfig.MSE or hessian is None:
            return error.abs_().pow_(mse).sum(dim=-1)

        error_fp32 = error.to(dtype=torch.float32)
        if method == ScaleSearchConfig.ACTIVATION:
            importance = hessian if hessian.ndim == 1 else hessian.diagonal().clamp_min(0)
            return error_fp32.square_().mul_(importance).sum(dim=-1)

        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            if error_fp32.ndim not in {2, 3}:
                raise ValueError(
                    f"{method.value.capitalize()} scale search expects a two-dimensional candidate tensor or "
                    "a three-dimensional candidate batch."
                )
            # Exact within a quantization group. For an ungrouped tensor, use a
            # bounded block-diagonal approximation so scale search does not turn
            # into an O(columns^2) allocation for every candidate.
            configured_group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
            columns = error_fp32.shape[-1]
            block_size = (
                min(columns, configured_group_size)
                if configured_group_size > 0
                else min(columns, 128)
            )
            objective = torch.zeros(error_fp32.shape[:-1], dtype=torch.float32, device=error_fp32.device)
            for start in range(0, columns, block_size):
                end = min(start + block_size, columns)
                block_error = error_fp32[..., start:end]
                block_hessian = hessian[start:end, start:end]
                block_objective = (block_error.matmul(block_hessian) * block_error).sum(dim=-1)
                objective.add_(block_objective)
            return objective.clamp_min_(0)

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    def _prepare_scale_search_hessian_batched(
        self,
        hessian: torch.Tensor | None,
        *,
        method: ScaleSearchConfig,
    ) -> torch.Tensor | None:
        """Prepare Hessian/diagonal data for batched multi-group scale search."""

        if hessian is None:
            return None

        if method in {ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.MARLIN_ACTIVATION}:
            if hessian.ndim == 1:
                hessian = hessian.unsqueeze(0)
            if hessian.ndim == 3:
                # A full group-Hessian was passed; activation search only needs the diagonal.
                hessian = hessian.diagonal(dim1=-2, dim2=-1).contiguous()
            if hessian.ndim != 2:
                raise ValueError(
                    f"Activation scale search expects a 2D group-diagonal tensor, got {tuple(hessian.shape)}."
                )
            importance = hessian.detach().to(dtype=torch.float32, device=hessian.device)
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
            # Normalize per-group to match the per-group find_params behavior.
            diagonal_mean = importance.mean(dim=-1, keepdim=True)
            valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
            safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
            normalized = importance / safe_mean
            return torch.where(valid, normalized, torch.ones_like(normalized))

        if method == ScaleSearchConfig.MARLIN_MSE:
            return None

        if method in {
            ScaleSearchConfig.HESSIAN,
            ScaleSearchConfig.HYBRID,
            ScaleSearchConfig.MARLIN,
        }:
            if hessian.ndim != 3:
                raise ValueError(
                    f"{method.value.capitalize()} batched scale search expects a 3D group-Hessian tensor, "
                    f"got {tuple(hessian.shape)}."
                )
            prepared = hessian.detach().to(dtype=torch.float32, device=hessian.device)
            prepared = torch.nan_to_num(prepared, nan=0.0, posinf=0.0, neginf=0.0)
            prepared = (prepared + prepared.transpose(-2, -1)) * 0.5
            diagonal = prepared.diagonal(dim1=-2, dim2=-1).clamp_min(0)
            # Normalize each group by its own diagonal mean to match per-group find_params.
            diagonal_mean = diagonal.mean(dim=-1, keepdim=True).unsqueeze(-1)
            valid = torch.isfinite(diagonal_mean) & (diagonal_mean > 0)
            safe_mean = torch.where(valid, diagonal_mean, torch.ones_like(diagonal_mean))
            prepared = prepared / safe_mean
            if method == ScaleSearchConfig.HYBRID:
                prepared.mul_(0.5)
                prepared.diagonal(dim1=-2, dim2=-1).mul_(2.0)
            return prepared

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    def _scale_search_error_batched(
        self,
        error: torch.Tensor,
        *,
        method: ScaleSearchConfig,
        mse: float,
        hessian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score candidates for every (output row, group) pair in one launch."""

        if method == ScaleSearchConfig.MSE or hessian is None:
            return error.abs_().pow_(mse).sum(dim=-1)

        error_fp32 = error.to(dtype=torch.float32)
        if method == ScaleSearchConfig.ACTIVATION:
            importance = hessian.unsqueeze(0).unsqueeze(0) if hessian.ndim == 2 else hessian
            return error_fp32.square_().mul_(importance).sum(dim=-1)

        if method in {ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID}:
            # error: [candidates, rows, groups, group_size]
            # hessian: [groups, group_size, group_size]
            # For HYBRID the Hessian is already folded into 0.5*(full + diagonal)
            # during _prepare_scale_search_hessian_batched, so the quadratic form
            # alone matches the per-group objective.
            projected = torch.einsum("crgi,gij->crgj", error_fp32, hessian)
            objective = (error_fp32 * projected).sum(dim=-1)
            return objective.clamp_min_(0)

        raise ValueError(f"Unsupported scale search method: `{method}`.")

    def _scale_search_candidate_chunk_size(
        self,
        x: torch.Tensor,
        candidate_count: int,
        method: ScaleSearchConfig,
    ) -> int:
        """Choose a bounded candidate batch that amortizes eager launch overhead."""

        configured_chunk_size = getattr(self.qcfg, "scale_search_candidate_chunk_size", None)
        if configured_chunk_size is not None:
            return max(1, min(candidate_count, int(configured_chunk_size)))

        if method in {
            ScaleSearchConfig.HESSIAN,
            ScaleSearchConfig.HYBRID,
            ScaleSearchConfig.MARLIN,
            ScaleSearchConfig.MARLIN_MSE,
            ScaleSearchConfig.MARLIN_ACTIVATION,
        }:
            target_elements = CORRELATED_SCALE_SEARCH_TARGET_ELEMENTS
            max_candidates = CORRELATED_SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK
        else:
            target_elements = SCALE_SEARCH_TARGET_ELEMENTS
            max_candidates = SCALE_SEARCH_MAX_CANDIDATES_PER_CHUNK
        elements_per_candidate = max(1, x.numel())
        return max(
            1,
            min(
                candidate_count,
                max_candidates,
                target_elements // elements_per_candidate,
            ),
        )

    def _marlin_scale_search_loss(
        self,
        W: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        hessian: torch.Tensor | None,
        *,
        method: ScaleSearchConfig,
        bits: int,
        group_size: int,
        sym: bool,
        dtype: torch.dtype,
        pack_dtype: torch.dtype,
        maxq_value: int,
        mse: float,
    ) -> torch.Tensor:
        """Score each scale/clip candidate using the packed Marlin kernel output.

        A synthetic activation matrix ``A_g`` is chosen so that the kernel-output
        MSE ``||A_g (W - Q)^T||^2`` matches the requested objective:

        - ``marlin`` / ``marlin_hessian``: ``A_g^T A_g = H_g`` from the Cholesky
          factor of the per-group Hessian (the dense quadratic form).
        - ``marlin_activation``: ``A_g`` is the diagonal sqrt of the per-channel
          activation importance (diagonal Hessian objective).
        - ``marlin_mse``: ``A_g`` is the identity, so the loss is plain weight
          reconstruction MSE evaluated through the packed kernel.

        The actual ``Y = Marlin(A_g, Q)`` path is used instead of ``A_g @ Q.T``,
        so the loss includes the kernel's sub-byte extraction, FP16/BF16 scale
        multiply, and accumulation order.
        """

        from ..nn_modules.qlinear.marlin import MarlinLinear

        if not sym:
            raise NotImplementedError(
                "Quantizer: `scale_search='marlin*'` currently requires symmetric quantization."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Quantizer: Marlin scale search requires a CUDA device.")
        if not marlin_runtime_available(dtype):
            raise RuntimeError(
                f"Quantizer: Marlin runtime not available for dtype `{dtype}`; "
                "cannot run kernel-aware scale search."
            )

        if method not in MARLIN_SCALE_SEARCH_METHODS:
            raise ValueError(f"Quantizer: unsupported Marlin scale-search method `{method.value}`.")

        # Mark that the kernel-aware path was reached; tests can assert this was
        # set and that no per-group fallback was used.
        self._marlin_scale_search_kernel_ran = True

        rows, num_groups, gs = W.shape
        C = scales.shape[0]
        # Keep all work on the same device as the incoming weights so multi-GPU
        # quantization threads do not silently cross devices.
        dev = W.device
        loss = torch.full((C, rows, num_groups), float("inf"), device=dev, dtype=torch.float32)

        # Canonicalize the optional per-group Hessian/importance tensor so the loop
        # can index it as ``hessian[g]`` for every sub-objective.
        if hessian is not None:
            if hessian.ndim == 1:
                # Single-group diagonal (activation-style) importance.
                hessian = hessian.unsqueeze(0)
            if hessian.ndim == 2:
                if hessian.shape == (num_groups, group_size):
                    # Batched diagonal importance; keep as-is for activation search.
                    pass
                elif hessian.shape == (group_size, group_size):
                    # Single full group Hessian; promote to a one-group batch.
                    hessian = hessian.unsqueeze(0)
                else:
                    raise ValueError(
                        "Quantizer: Marlin scale search expects a (num_groups, group_size) "
                        f"diagonal or (group_size, group_size) Hessian, got {tuple(hessian.shape)}."
                    )
            if hessian.ndim == 3 and (
                hessian.shape[0] != num_groups
                or hessian.shape[-2] != group_size
                or hessian.shape[-1] != group_size
            ):
                raise ValueError(
                    "Quantizer: Marlin scale search expects a (num_groups, group_size, group_size) "
                    f"Hessian, got {tuple(hessian.shape)} for W {tuple(W.shape)}."
                )

        # Use the same dtype for the kernel reference and the packed module.
        marlin_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16

        # Marlin only supports int32-packed weights.
        if pack_dtype != torch.int32:
            raise ValueError(
                f"Quantizer: Marlin scale search only supports int32 pack_dtype, got {pack_dtype}."
            )
        pack_factor = 32 // bits

        # The synthetic per-group activation has no column reordering, so the
        # scoring module must use desc_act=False even when the model config
        # enables activation ordering. Otherwise MarlinLinear would consume an
        # uninitialized g_idx and silently repack with a garbage permutation.
        marlin_desc_act = False

        for g in range(num_groups):
            W_g = W[:, g, :].detach().to(device=dev, dtype=marlin_dtype).contiguous()

            # Build the synthetic activation matrix for the requested objective.
            if method == ScaleSearchConfig.MARLIN_MSE:
                A_g = torch.eye(gs, device=dev, dtype=marlin_dtype)
                Y_ref = W_g.t()
                fallback_method = ScaleSearchConfig.MSE
                fallback_hessian = None
            elif method == ScaleSearchConfig.MARLIN_ACTIVATION:
                if hessian is None:
                    raise ValueError(
                        "Quantizer: `scale_search='marlin_activation'` requires activation importance (Hessian diagonal)."
                    )
                importance_g = hessian[g].to(device=dev, dtype=torch.float32)
                importance_g = torch.nan_to_num(importance_g, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
                # If the importance vector is degenerate, fall back to identity so the
                # kernel still returns a valid loss instead of zero for every candidate.
                if importance_g.sum().item() <= 0 or not torch.isfinite(importance_g).all():
                    A_g = torch.eye(gs, device=dev, dtype=marlin_dtype)
                else:
                    A_g = torch.diag(torch.sqrt(importance_g)).to(dtype=marlin_dtype)
                Y_ref = torch.matmul(A_g, W_g.t())
                fallback_method = ScaleSearchConfig.ACTIVATION
                fallback_hessian = importance_g
            else:  # ScaleSearchConfig.MARLIN (Hessian)
                if hessian is None:
                    raise ValueError(
                        "Quantizer: `scale_search='marlin'` requires a per-group Hessian."
                    )
                H_g = hessian[g].to(device=dev, dtype=torch.float32)

                # Add a small damping term to guarantee the group Hessian is positive-definite.
                diag_mean = H_g.diagonal().abs().mean()
                damping = max(1e-6, diag_mean * 1e-6)
                H_g = H_g + torch.eye(gs, device=H_g.device, dtype=H_g.dtype) * damping
                try:
                    L_g = linalg_cholesky(H_g)
                except RuntimeError:
                    H_g = torch.diag(H_g.diagonal().clamp_min(damping))
                    L_g = linalg_cholesky(H_g)

                # A_g^T A_g = H_g, so ``||A_g (w - q)^T||^2`` equals the Hessian quadratic form.
                A_g = L_g.transpose(-2, -1).to(dtype=marlin_dtype).contiguous()
                Y_ref = torch.matmul(A_g, W_g.t())
                fallback_method = ScaleSearchConfig.HESSIAN
                fallback_hessian = H_g

            s_g = scales[:, :, g].to(dtype=marlin_dtype).contiguous()
            z_g = zeros[:, :, g].to(dtype=marlin_dtype).contiguous()

            try:
                # Build the integer weight for all candidates in one GEMM-shaped tile.
                W_exp = W_g.unsqueeze(0)
                s_exp = s_g.unsqueeze(-1)
                z_exp = z_g.unsqueeze(-1)
                int_weight = ((W_exp + z_exp * s_exp) / s_exp).round().clamp(0, maxq_value).to(torch.int32)

                # Pack along the input-channel dimension: each int32 stores `pack_factor` input values
                # for one output channel, matching GPTQ/Marlin qweight layout.
                int_weight_t = int_weight.reshape(C * rows, gs).contiguous().T
                int_weight_t = int_weight_t.view(gs // pack_factor, pack_factor, C * rows)
                shifts = torch.arange(pack_factor, device=int_weight.device, dtype=torch.int32) * bits
                qweight = (int_weight_t << shifts.view(1, -1, 1)).sum(dim=1, dtype=torch.int32)

                scales_marlin = s_g.reshape(1, C * rows).to(dtype=marlin_dtype).contiguous()

                marlin = MarlinLinear(
                    bits=bits,
                    group_size=group_size,
                    desc_act=marlin_desc_act,
                    sym=sym,
                    in_features=gs,
                    out_features=C * rows,
                    bias=False,
                    pack_dtype=pack_dtype,
                    backend=BACKEND.GPTQ_MARLIN,
                    dtype=marlin_dtype,
                )
                replace_parameter(marlin, "qweight", qweight)
                replace_parameter(marlin, "scales", scales_marlin)
                marlin.post_init()

                Y_marlin = marlin(A_g)
                Y_marlin = Y_marlin.reshape(gs, C, rows)
                diff = Y_marlin - Y_ref.unsqueeze(1)
                loss[:, :, g] = diff.to(dtype=torch.float32).pow(2).sum(dim=0)
            except Exception as exc:
                # Fall back to the equivalent dense objective for this group so the
                # scale-search still returns usable parameters if Marlin packing
                # or the kernel fails for a shape.
                if not hasattr(self, "_marlin_scale_search_fallback_count"):
                    self._marlin_scale_search_fallback_count = 0
                self._marlin_scale_search_fallback_count += 1
                log.warn.once(
                    f"Marlin scale search ({method.value}) failed for at least one group "
                    f"(bits={bits}, group_size={group_size}), falling back to {fallback_method.value}: {exc}"
                )
                x_g = W_g.unsqueeze(0).expand(C, -1, -1)
                scale_g = s_g.unsqueeze(-1)
                zero_g = z_g.unsqueeze(-1)
                error = self._quantize_scale_search_candidates(
                    x_g,
                    scale_g,
                    zero_g,
                    maxq_value=maxq_value,
                )
                loss[:, :, g] = self._scale_search_error(
                    error,
                    method=fallback_method,
                    mse=mse,
                    hessian=fallback_hessian,
                )

        return loss

    def _effective_scale_search_bits(self, maxq_value: int) -> int:
        """Recover the bit width from the configured max quantized value."""

        if self.requires_groupwise_processing():
            return maxq_value.bit_length() + 1
        return (maxq_value + 1).bit_length() - 1

    def _quantize_scale_search_candidates(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        *,
        maxq_value: int,
    ) -> torch.Tensor:
        """Quantize one candidate batch and return the reconstruction error."""

        if maxq_value < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.div(x, scale)
        q.round_()
        maxq_f = float(maxq_value)
        if self.requires_groupwise_processing():
            return q.clamp_(-maxq_f, maxq_f).mul_(scale).sub_(x)
        return q.clamp_(-zero, maxq_f - zero).mul_(scale).sub_(x)

    def adaptive_clip_search(
        self,
        x: torch.Tensor,
        weight: bool = False,
        *,
        hessian: torch.Tensor | None = None,
        gptq_inverse_cholesky: torch.Tensor | None = None,
    ) -> None:
        """Search a per-row clipping threshold for ``x`` and set ``self.scale`` / ``self.zero``.

        Candidates are relative factors of each row's current ``max(|x|)``.
        For every candidate the clipped range is used to derive a scale and
        zero, the original ``x`` is quantized with that scale/zero, and the
        candidate with the lowest configured objective is kept. The default
        ``gptq_error`` objective simulates GPTQ's sequential error correction
        using its damped inverse-Cholesky factor. Candidate tensors are
        materialized in bounded chunks so a long candidate list does not blow
        up per-group memory.
        """

        cfg = self.qcfg.adaptive_clipping
        if not isinstance(cfg, AdaptiveClippingConfig) or not cfg.enabled or not cfg.per_group:
            return

        # Sanitize NaN/Inf without forcing a host/device sync (.all()).
        x = x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        dev = x.device
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())

        # ``find_params`` has already flattened/reshaped ``x`` to [rows, cols].
        rows = x.shape[0]
        zero = torch.zeros(rows, device=dev)
        # Single-pass min/max; avoid two full reductions.
        xmin_raw, xmax_raw = torch.aminmax(x, dim=-1)
        xmin = torch.minimum(xmin_raw, zero)
        xmax = torch.maximum(xmax_raw, zero)

        if self.qcfg.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            neg = xmin < 0
            xmin = torch.where(neg, -xmax, xmin)
        zero_range = (xmin == 0) & (xmax == 0)
        xmin = torch.where(zero_range, -torch.ones_like(xmin), xmin)
        xmax = torch.where(zero_range, torch.ones_like(xmax), xmax)
        maxabs = torch.maximum(torch.abs(xmin), torch.abs(xmax))
        maxabs = torch.where(maxabs <= 0, torch.ones_like(maxabs), maxabs)
        all_zero = (xmin_raw == 0) & (xmax_raw == 0)

        h_diag = None
        gptq_factor = None
        objective_geometry_valid = True
        if cfg.metric == AdaptiveClippingMetric.HESSIAN_DIAG.value:
            if hessian is not None:
                if hessian.ndim == 1 and hessian.shape[0] == x.shape[1]:
                    h_diag = hessian
                elif hessian.ndim == 2 and hessian.shape[0] == hessian.shape[1] == x.shape[1]:
                    h_diag = hessian.diagonal()
            if h_diag is not None:
                h_diag = h_diag.detach().to(device=dev, dtype=torch.float32)
                h_diag = torch.nan_to_num(h_diag, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0)
                objective_geometry_valid = bool(torch.any(h_diag > 0).item())
            else:
                objective_geometry_valid = False
        elif cfg.metric == AdaptiveClippingMetric.GPTQ_ERROR.value:
            factor = gptq_inverse_cholesky
            objective_geometry_valid = bool(
                factor is not None
                and factor.ndim == 2
                and factor.shape[0] == factor.shape[1] == x.shape[1]
                and torch.isfinite(factor).all().item()
                and (factor.diagonal() > 0).all().item()
            )
            if objective_geometry_valid:
                # GPTQ arithmetic is FP32 even when the saved module is FP16/BF16.
                # Keep a view when already colocated instead of copying a group
                # factor for every clipping search.
                gptq_factor = factor.detach().to(device=dev, dtype=torch.float32)

        if not objective_geometry_valid:
            # A missing calibration objective must never turn adaptive clipping
            # into an implicit weight-only optimizer. Candidate 1.0 exactly
            # reproduces the normal per-group min/max range after sanitization.
            candidates = torch.ones(1, dtype=torch.float32, device=dev)
            log.warn.once(
                f"Quantizer: `adaptive_clipping` metric `{cfg.metric}` requires valid calibration geometry "
                f"for module `{self.name}`; using the full unclipped range."
            )
        else:
            candidates = torch.tensor(cfg.candidates, dtype=torch.float32, device=dev)
        num_candidates = candidates.numel()

        # Keep the q / diff scratch tensors small; the candidate list is short so
        # extra chunk iterations are cheap, but the per-chunk temporaries scale
        # with rows * group_size and can dominate per-group peak memory.
        chunk_size = min(
            num_candidates,
            ADAPTIVE_CLIP_MAX_CANDIDATES_PER_CHUNK,
            self._scale_search_candidate_chunk_size(x, num_candidates, ScaleSearchConfig.MSE),
        )

        best_loss = torch.full((rows,), float("inf"), device=dev)
        best_scale = None
        best_zero = None

        for start in range(0, num_candidates, chunk_size):
            end = min(start + chunk_size, num_candidates)
            cand_chunk = candidates[start:end]

            # [chunk, rows]
            thresholds = cand_chunk.unsqueeze(1) * maxabs.unsqueeze(0)
            # Derive the clipped extrema directly from the pre-computed row min/max
            # without materializing a [chunk, rows, cols] ``x_clipped`` tensor.
            # For each row, clamping to [-t, t] raises the minimum to max(old_min, -t)
            # and lowers the maximum to min(old_max, t). The original code then
            # clamps the extrema toward zero so the asymmetric range always
            # contains zero; preserve that behavior.
            xmin_c = torch.maximum(xmin.unsqueeze(0), -thresholds).clamp_max(0)
            xmax_c = torch.minimum(xmax.unsqueeze(0), thresholds).clamp_min(0)

            # Keep zero inside the clipped range so asymmetric zero-points stay
            # valid and constant groups don't collapse to a zero-width range.
            if self.qcfg.sym:
                xmax_c = torch.maximum(torch.abs(xmin_c), xmax_c)
                neg_c = xmin_c < 0
                xmin_c = torch.where(neg_c, -xmax_c, xmin_c)
            zero_range_c = (xmin_c == 0) & (xmax_c == 0)
            xmin_c = torch.where(zero_range_c, -torch.ones_like(xmin_c), xmin_c)
            xmax_c = torch.where(zero_range_c, torch.ones_like(xmax_c), xmax_c)
            # Constant-zero rows have zero loss for every clipping candidate, so
            # the first candidate is selected. Force the full [-1, 1] range for
            # those rows to match the scale produced by the original materialized
            # ``x.clamp(-t, t)`` path (which sees the clipped tensor as all zeros).
            is_all_zero = all_zero.unsqueeze(0)
            xmin_c = torch.where(is_all_zero, -torch.ones_like(xmin_c), xmin_c)
            xmax_c = torch.where(is_all_zero, torch.ones_like(xmax_c), xmax_c)

            if maxq_value < 0:
                scale_c = xmax_c
                zero_c = xmin_c
            else:
                if self.requires_groupwise_processing():
                    scale_c = xmax_c / self.maxq
                    zero_c = torch.zeros_like(scale_c)
                else:
                    scale_c = (xmax_c - xmin_c) / self.maxq
                    if self.qcfg.sym:
                        zero_c = torch.full_like(scale_c, (self.maxq + 1) / 2)
                    else:
                        zero_c = torch.round(-xmin_c / scale_c)

            if gptq_factor is not None:
                loss_chunk = self._adaptive_clip_gptq_error(
                    x,
                    scale_c,
                    zero_c,
                    gptq_factor,
                    maxq_value=maxq_value,
                )
            else:
                # Quantize the *original* weights with each candidate's
                # scale/zero so the approximation loss is measured against the
                # unclipped signal.
                q = quantize(
                    x.unsqueeze(0),
                    scale_c.unsqueeze(2),
                    zero_c.unsqueeze(2),
                    maxq_value,
                    self.requires_groupwise_processing(),
                )
                diff = (x.unsqueeze(0) - q).float()
                diff.pow_(2)
                if h_diag is not None:
                    diff.mul_(h_diag)
                loss_chunk = diff.sum(dim=-1)
                del diff

            chunk_best, chunk_idx = loss_chunk.min(dim=0)
            take = chunk_best < best_loss
            best_loss = torch.where(take, chunk_best, best_loss)

            chunk_scale = scale_c.gather(0, chunk_idx.unsqueeze(0)).squeeze(0)
            chunk_zero = zero_c.gather(0, chunk_idx.unsqueeze(0)).squeeze(0)
            if best_scale is None:
                best_scale = chunk_scale
                best_zero = chunk_zero
            else:
                best_scale = torch.where(take, chunk_scale, best_scale)
                best_zero = torch.where(take, chunk_zero, best_zero)

        self.scale = best_scale
        self.zero = best_zero

    def _adaptive_clip_gptq_error(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        inverse_cholesky: torch.Tensor,
        *,
        maxq_value: int,
    ) -> torch.Tensor:
        """Return canonical GPTQ loss for each clipping candidate and row.

        Let ``U`` be the upper Cholesky factor of the damped inverse Hessian.
        For column ``j``, GPTQ uses ``E_j = (W_j - Q_j) / U_jj`` and updates
        every remaining column by ``W[:, j:] -= E_j outer U[j, j:]``. The
        candidate objective is therefore ``0.5 * sum_j(E_j ** 2)``. Simulating
        that recursion is essential: quantized values after the first column
        depend on both calibration activations and all earlier quantization
        errors, which a weight-only or diagonal objective cannot represent.
        """

        work = x.float().unsqueeze(0).expand(scale.shape[0], -1, -1).clone()
        factor = inverse_cholesky.float()
        loss = torch.zeros(scale.shape, device=x.device, dtype=torch.float32)
        groupwise = self.requires_groupwise_processing()
        for column in range(x.shape[1]):
            q = quantize(work[:, :, column], scale, zero, maxq_value, groupwise)
            error = (work[:, :, column] - q) / factor[column, column]
            loss.add_(0.5 * error.square())
            work[:, :, column:].sub_(error.unsqueeze(2) * factor[column, column:])
        return loss

    def find_params(
        self,
        x,
        weight=False,
        *,
        hessian: torch.Tensor | None = None,
        gptq_inverse_cholesky: torch.Tensor | None = None,
    ):
        # GPU scale-search algorithms (Triton and otherwise) assume a contiguous
        # [rows, num_groups, group_size] layout. gptq.py passes reshaped column
        # slices that may not be, so copy the local batch to avoid silent fallback
        # or strided-read errors without changing the quantized result.
        if isinstance(x, torch.Tensor) and not x.is_contiguous():
            x = x.contiguous()

        dev = x.device
        self.maxq = self.maxq.to(dev)
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())
            self._maxq_value = maxq_value

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        adaptive_clip_cfg = getattr(self.qcfg, "adaptive_clipping", None)
        use_adaptive_clip = (
            weight
            and self.perchannel
            and isinstance(adaptive_clip_cfg, AdaptiveClippingConfig)
            and adaptive_clip_cfg.enabled
            and adaptive_clip_cfg.per_group
            and not getattr(self.qcfg, "mock_quantization", False)
            and self.qcfg.group_size is not None
            and self.qcfg.group_size > 0
        )
        if use_adaptive_clip:
            mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
            scale_search_method = getattr(self.qcfg, "scale_search", None)
            if mse > 0.0 or scale_search_method is not None:
                log.warn.once(
                    "Quantizer: `adaptive_clipping` per_group is enabled; the `mse`/`scale_search` "
                    "range-search path is skipped because adaptive clipping already searches per-group ranges."
                )
            self.adaptive_clip_search(
                x,
                weight=weight,
                hessian=hessian,
                gptq_inverse_cholesky=gptq_inverse_cholesky,
            )
        else:
            zero = torch.zeros(x.shape[0], device=dev)
            xmin_raw, xmax_raw = torch.aminmax(x, dim=-1)
            xmin = torch.minimum(xmin_raw, zero)
            xmax = torch.maximum(xmax_raw, zero)

            if self.qcfg.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                xmin = torch.where(tmp, -xmax, xmin)
            tmp = (xmin == 0) & (xmax == 0)
            xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
            xmax = torch.where(tmp, torch.ones_like(xmax), xmax)

            if maxq_value < 0:
                self.scale = xmax
                self.zero = xmin
            else:
                if self.requires_groupwise_processing():
                    self.scale = xmax / self.maxq
                    self.zero = torch.zeros_like(self.scale)
                else:
                    self.scale = (xmax - xmin) / self.maxq
                    if self.qcfg.sym:
                        self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
                    else:
                        self.zero = torch.round(-xmin / self.scale)

            mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
            method = getattr(self.qcfg, "scale_search", None)
            if method is None and mse > 0:
                method = ScaleSearchConfig.MSE
            elif isinstance(method, str):
                method = ScaleSearchConfig(method)

            if method is not None and mse > 0.0:
                timer = getattr(self, "region_timer", None)
                timer_cm = (
                    timer.measure("scale_search", source=f"find_params {self.name}")
                    if timer is not None
                    else contextlib.nullcontext()
                )
                with timer_cm:
                    prepared_hessian = None
                    if method != ScaleSearchConfig.MSE:
                        prepared_hessian = self._prepare_scale_search_hessian(
                            hessian,
                            method=method,
                            columns=x.shape[1],
                            device=dev,
                        )
                    best = torch.full([x.shape[0]], float("inf"), device=dev)
                    candidate_count = int(self.maxshrink * self.grid)

                    chunk_size = self._scale_search_candidate_chunk_size(x, candidate_count, method)
                    # Preserve the original scalar candidate values while
                    # evaluating all rows and candidates in vectorized chunks.
                    shrink = self._scale_search_shrink_factors(candidate_count, self.grid, dev)
                    # Precompute scale/zero for every candidate once so the chunk loop
                    # only slices views instead of recomputing elementwise ranges.
                    p = shrink.view(-1, 1)
                    xmin_all = p * xmin.unsqueeze(0)
                    xmax_all = p * xmax.unsqueeze(0)
                    if self.requires_groupwise_processing():
                        scale_all = xmax_all / self.maxq
                    else:
                        scale_all = (xmax_all - xmin_all) / self.maxq
                    if self.qcfg.sym:
                        zero_all = self.zero.unsqueeze(0).expand_as(scale_all)
                    else:
                        zero_all = torch.round(-xmin_all / scale_all)

                    group_size = x.shape[1]
                    x_batch = x.unsqueeze(0)
                    for start in range(0, candidate_count, chunk_size):
                        end = min(start + chunk_size, candidate_count)
                        scale1 = scale_all[start:end]
                        zero1 = zero_all[start:end]
                        if (
                            method in MARLIN_SCALE_SEARCH_METHODS
                            and x.is_cuda
                            and group_size >= 64
                        ):
                            # Per-group K must be at least the Marlin GEMM's minimum
                            # thread_k tile (64); otherwise the temporary one-group
                            # MarlinLinear cannot be launched.
                            from ..nn_modules.qlinear.marlin import MarlinLinear

                            # Map Marlin sub-objectives to their dense fallback equivalents.
                            if method == ScaleSearchConfig.MARLIN_MSE:
                                fallback_method = ScaleSearchConfig.MSE
                            elif method == ScaleSearchConfig.MARLIN_ACTIVATION:
                                fallback_method = ScaleSearchConfig.ACTIVATION
                            else:
                                fallback_method = ScaleSearchConfig.HESSIAN

                            try:
                                bits = self._effective_scale_search_bits(maxq_value)
                                if bits in MarlinLinear.SUPPORTS_BITS and group_size in MarlinLinear.SUPPORTS_GROUP_SIZE:
                                    errors = self._marlin_scale_search_loss(
                                        x.unsqueeze(1),
                                        scale1.unsqueeze(2),
                                        zero1.unsqueeze(2),
                                        prepared_hessian,
                                        method=method,
                                        bits=bits,
                                        group_size=group_size,
                                        sym=self.qcfg.sym,
                                        dtype=x.dtype,
                                        pack_dtype=getattr(self.qcfg, "pack_dtype", torch.int32),
                                        maxq_value=maxq_value,
                                        mse=mse,
                                    ).squeeze(-1)
                                else:
                                    raise RuntimeError(
                                        f"Marlin scale search unsupported for bits={bits}, group_size={group_size}."
                                    )
                            except Exception as exc:
                                log.warn.once(
                                    f"Marlin scale search ({method.value}) failed, falling back to "
                                    f"{fallback_method.value}: {exc}"
                                )
                                effective_method = fallback_method
                                error = self._quantize_scale_search_candidates(
                                    x_batch,
                                    scale1.unsqueeze(2),
                                    zero1.unsqueeze(2),
                                    maxq_value=maxq_value,
                                )
                                errors = self._scale_search_error(
                                    error,
                                    method=effective_method,
                                    mse=mse,
                                    hessian=prepared_hessian,
                                )
                        else:
                            error = self._quantize_scale_search_candidates(
                                x_batch,
                                scale1.unsqueeze(2),
                                zero1.unsqueeze(2),
                                maxq_value=maxq_value,
                            )
                            effective_method = method
                            if method in MARLIN_SCALE_SEARCH_METHODS:
                                log.warn.once(
                                    f"Quantizer: `scale_search='{method.value}'` requires CUDA weights; "
                                    f"falling back to dense objective."
                                )
                                if method == ScaleSearchConfig.MARLIN_MSE:
                                    effective_method = ScaleSearchConfig.MSE
                                elif method == ScaleSearchConfig.MARLIN_ACTIVATION:
                                    effective_method = ScaleSearchConfig.ACTIVATION
                                else:
                                    effective_method = ScaleSearchConfig.HESSIAN
                            errors = self._scale_search_error(
                                error,
                                method=effective_method,
                                mse=mse,
                                hessian=prepared_hessian,
                            )
                        # torch.min returns the first index on ties. Combining one
                        # winner per chunk with a strict comparison across chunks
                        # exactly preserves the scalar loop's first-candidate rule.
                        chunk_best, chunk_index = errors.min(dim=0)
                        gather_index = chunk_index.unsqueeze(0)
                        chunk_scale = scale1.gather(0, gather_index).squeeze(0)
                        chunk_zero = zero1.gather(0, gather_index).squeeze(0)
                        take = chunk_best < best
                        best = torch.where(take, chunk_best, best)
                        self.scale = torch.where(take, chunk_scale, self.scale)
                        self.zero = torch.where(take, chunk_zero, self.zero)
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def find_params_batched(
        self,
        x: torch.Tensor,
        weight: bool = False,
        *,
        hessian: torch.Tensor | None = None,
        gptq_inverse_cholesky: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scales and zeros for all groups in x at once.

        x is expected to have shape [rows, num_groups, group_size] and
        hessian is either None, a [num_groups, group_size] diagonal for
        activation search, or a [num_groups, group_size, group_size] group
        Hessian for correlated objectives. ``gptq_inverse_cholesky`` is either
        the block factor [num_groups * group_size, num_groups * group_size] or
        per-group factors [num_groups, group_size, group_size]. Returns (scale,
        zero) of shape [rows, num_groups].
        """

        # The Triton activation/hessian scale-search kernels load x with
        # pointer+stride arithmetic that assumes a contiguous [rows, num_groups,
        # group_size] layout. gptq.py passes reshaped column slices that are
        # not contiguous, so copy the local batch to unlock the fast GPU path
        # without changing any quantized scale/zero values.
        if isinstance(x, torch.Tensor) and not x.is_contiguous():
            x = x.contiguous()

        if x.ndim != 3:
            raise ValueError(f"find_params_batched expects a 3D tensor, got {tuple(x.shape)}.")
        if not (weight and self.perchannel):
            raise ValueError("find_params_batched is only supported for per-channel weight quantization.")

        adaptive_clip_cfg = getattr(self.qcfg, "adaptive_clipping", None)
        if (
            isinstance(adaptive_clip_cfg, AdaptiveClippingConfig)
            and adaptive_clip_cfg.enabled
            and adaptive_clip_cfg.per_group
            and not getattr(self.qcfg, "mock_quantization", False)
        ):
            # Fallback to per-group find_params so the clipping search is applied.
            x = x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            scale_parts = []
            zero_parts = []
            for g in range(x.shape[1]):
                x_g = x[:, g, :].contiguous()
                h_g = None
                if hessian is not None:
                    if hessian.ndim == 2:
                        h_g = hessian[g]
                    elif hessian.ndim == 3:
                        h_g = hessian[g]
                inverse_g = None
                if gptq_inverse_cholesky is not None:
                    if gptq_inverse_cholesky.ndim == 2:
                        start = g * x.shape[2]
                        end = start + x.shape[2]
                        inverse_g = gptq_inverse_cholesky[start:end, start:end]
                    elif gptq_inverse_cholesky.ndim == 3:
                        inverse_g = gptq_inverse_cholesky[g]
                self.find_params(
                    x_g,
                    weight=True,
                    hessian=h_g,
                    gptq_inverse_cholesky=inverse_g,
                )
                scale_parts.append(self.scale.view(x_g.shape[0], 1))
                zero_parts.append(self.zero.view(x_g.shape[0], 1))
            scale = torch.cat(scale_parts, dim=1)
            zero = torch.cat(zero_parts, dim=1)
            # Keep the module buffers consistent with the returned batched tensors
            # instead of leaving them set to the final group's parameters.
            self.scale = scale
            self.zero = zero
            return scale, zero

        dev = x.device
        self.maxq = self.maxq.to(dev)
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())
            self._maxq_value = maxq_value

        rows, num_groups, group_size = x.shape

        # Min/max over the group dimension (last).
        zero = torch.zeros((rows, num_groups), device=dev)
        xmin_raw, xmax_raw = torch.aminmax(x, dim=-1)
        xmin = torch.minimum(xmin_raw, zero)
        xmax = torch.maximum(xmax_raw, zero)

        if self.qcfg.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            xmin = torch.where(tmp, -xmax, xmin)
        tmp = (xmin == 0) & (xmax == 0)
        xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
        xmax = torch.where(tmp, torch.ones_like(xmax), xmax)

        if maxq_value < 0:
            scale = xmax
            zero = xmin
        else:
            if self.requires_groupwise_processing():
                scale = xmax / self.maxq
                zero = torch.zeros_like(scale)
            else:
                scale = (xmax - xmin) / self.maxq
                if self.qcfg.sym:
                    zero = torch.full_like(scale, (self.maxq + 1) / 2)
                else:
                    zero = torch.round(-xmin / scale)

        mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
        method = getattr(self.qcfg, "scale_search", None)
        if method is None and mse > 0:
            method = ScaleSearchConfig.MSE
        elif isinstance(method, str):
            method = ScaleSearchConfig(method)

        if method is not None and mse > 0.0:
            timer = getattr(self, "region_timer", None)
            timer_cm = (
                timer.measure("scale_search", source=f"find_params_batched {self.name}")
                if timer is not None
                else contextlib.nullcontext()
            )
            with timer_cm:
                prepared_hessian = None
                if method != ScaleSearchConfig.MSE:
                    prepared_hessian = self._prepare_scale_search_hessian_batched(
                        hessian,
                        method=method,
                    )

                candidate_count = int(self.maxshrink * self.grid)

                if (
                    os.environ.get("GPTQMODEL_SCALE_SEARCH_CPU", "0") != "0"
                    and _find_params_batched_cpu is not None
                    and method in (ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.MSE)
                    and (method != ScaleSearchConfig.MSE or abs(mse - 2.0) < 1e-6)
                    and not self.requires_groupwise_processing()
                    and group_size <= 128
                    and maxq_value > 0
                    and x.is_cpu
                    and x.is_contiguous()
                    and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
                ):
                    try:
                        importance = prepared_hessian if method == ScaleSearchConfig.ACTIVATION else None
                        return _find_params_batched_cpu(
                            x,
                            xmin,
                            xmax,
                            importance,
                            self.grid,
                            self.maxshrink,
                            maxq_value,
                            self.qcfg.sym,
                            self.requires_groupwise_processing(),
                            method.value,
                            mse,
                        )
                    except Exception as e:
                        log.warn(f"CPU scale-search failed, falling back: {e}")

                # Fallback exact candidate-chunk loop. This is kept as the
                # strict-accuracy reference and is used when the Triton fast path
                # is disabled or unsupported.
                if candidate_count > 0:
                    shrink = self._scale_search_shrink_factors(candidate_count, self.grid, dev)
                    p = shrink.view(-1, 1, 1)
                    xmin_all = p * xmin.unsqueeze(0)
                    xmax_all = p * xmax.unsqueeze(0)
                    if self.requires_groupwise_processing():
                        scale_all = xmax_all / self.maxq
                    else:
                        scale_all = (xmax_all - xmin_all) / self.maxq
                    if self.qcfg.sym:
                        zero_all = zero.unsqueeze(0).expand(candidate_count, -1, -1)
                    else:
                        zero_all = torch.round(-xmin_all / scale_all)
                else:
                    scale_all = scale.unsqueeze(0)
                    zero_all = zero.unsqueeze(0)

                best = torch.full((rows, num_groups), float("inf"), device=dev)
                chunk_size = self._scale_search_candidate_chunk_size(x, candidate_count, method)

                x_batch = x.unsqueeze(0)
                for start in range(0, candidate_count, chunk_size):
                    end = min(start + chunk_size, candidate_count)
                    scale1 = scale_all[start:end]
                    zero1 = zero_all[start:end]
                    if (
                        method in MARLIN_SCALE_SEARCH_METHODS
                        and x.is_cuda
                        and group_size >= 64
                    ):
                        # Per-group K must be at least the Marlin GEMM's minimum
                        # thread_k tile (64); otherwise the temporary one-group
                        # MarlinLinear cannot be launched.
                        from ..nn_modules.qlinear.marlin import MarlinLinear

                        if method == ScaleSearchConfig.MARLIN_MSE:
                            fallback_method = ScaleSearchConfig.MSE
                        elif method == ScaleSearchConfig.MARLIN_ACTIVATION:
                            fallback_method = ScaleSearchConfig.ACTIVATION
                        else:
                            fallback_method = ScaleSearchConfig.HESSIAN

                        try:
                            bits = self._effective_scale_search_bits(maxq_value)
                            if bits in MarlinLinear.SUPPORTS_BITS and group_size in MarlinLinear.SUPPORTS_GROUP_SIZE:
                                errors = self._marlin_scale_search_loss(
                                    x,
                                    scale1,
                                    zero1,
                                    prepared_hessian,
                                    method=method,
                                    bits=bits,
                                    group_size=group_size,
                                    sym=self.qcfg.sym,
                                    dtype=x.dtype,
                                    pack_dtype=getattr(self.qcfg, "pack_dtype", torch.int32),
                                    maxq_value=maxq_value,
                                    mse=mse,
                                )
                            else:
                                raise RuntimeError(
                                    f"Marlin scale search unsupported for bits={bits}, group_size={group_size}."
                                )
                        except Exception as exc:
                            log.warn.once(
                                f"Marlin scale search ({method.value}) failed, falling back to "
                                f"{fallback_method.value}: {exc}"
                            )
                            effective_method = fallback_method
                            error = self._quantize_scale_search_candidates(
                                x_batch.expand(end - start, -1, -1, -1),
                                scale1.unsqueeze(-1),
                                zero1.unsqueeze(-1),
                                maxq_value=maxq_value,
                            )
                            errors = self._scale_search_error_batched(
                                error,
                                method=effective_method,
                                mse=mse,
                                hessian=prepared_hessian,
                            )
                    else:
                        error = self._quantize_scale_search_candidates(
                            x_batch.expand(end - start, -1, -1, -1),
                            scale1.unsqueeze(-1),
                            zero1.unsqueeze(-1),
                            maxq_value=maxq_value,
                        )
                        effective_method = method
                        if method in MARLIN_SCALE_SEARCH_METHODS:
                            log.warn.once(
                                f"Quantizer: `scale_search='{method.value}'` requires CUDA weights; "
                                f"falling back to dense objective."
                            )
                            if method == ScaleSearchConfig.MARLIN_MSE:
                                effective_method = ScaleSearchConfig.MSE
                            elif method == ScaleSearchConfig.MARLIN_ACTIVATION:
                                effective_method = ScaleSearchConfig.ACTIVATION
                            else:
                                effective_method = ScaleSearchConfig.HESSIAN
                        errors = self._scale_search_error_batched(
                            error,
                            method=effective_method,
                            mse=mse,
                            hessian=prepared_hessian,
                        )
                    chunk_best, chunk_index = errors.min(dim=0)
                    gather_index = chunk_index.unsqueeze(0)
                    chunk_scale = scale1.gather(0, gather_index).squeeze(0)
                    chunk_zero = zero1.gather(0, gather_index).squeeze(0)
                    take = chunk_best < best
                    best = torch.where(take, chunk_best, best)
                    scale = torch.where(take, chunk_scale, scale)
                    zero = torch.where(take, chunk_zero, zero)

        return scale, zero

    def quantize(self, x):
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        maxq_value = getattr(self, "_maxq_value", None)
        if maxq_value is None:
            maxq_value = int(self.maxq.item())
            self._maxq_value = maxq_value
        return quantize(x, self.scale, self.zero, maxq_value, self.requires_groupwise_processing())

    # def enabled(self):
    #     return self.maxq > 0

    # def ready(self):
    # return torch.all(self.scale != 0)

class QQQQuantizer(Quantizer):
    def requires_groupwise_processing(self) -> bool:
        return self.qcfg.group_size == -1 and self.qcfg.sym

__all__ = ["Quantizer"]
