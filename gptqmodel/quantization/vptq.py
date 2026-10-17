# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# VPTQ (Vector Post-Training Quantization) weight quantizer.
#
# Algorithm reference: https://github.com/microsoft/VPTQ (MIT licensed),
# "VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language
# Models" (https://arxiv.org/abs/2409.17066). The implementation below is a
# Torch-only port of the reference quantization recipe:
#
#   1. collect the per-module Hessian H = E[x^T x] from calibration data
#   2. optionally re-order input channels by descending Hessian diagonal and
#      normalize per-channel ranges
#   3. split each weight column into sub-vectors along `out_features` and
#      learn a per-group codebook with weighted k-means
#   4. run a GPTQ-style sequential pass that compensates each column's vector
#      quantization error with the inverse Hessian
#   5. optionally learn a residual codebook over the remaining error and
#      repeat the pass with a two-stage (main + residual) lookup

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers

from ..looper.named_module import NamedModule
from ..quantization.config import VPTQConfig
from ..utils.fallback import should_use_fallback
from ..utils.logger import setup_logger
from ..utils.vptq import (
    vptq_group_for_column,
    vptq_group_setting,
    vptq_index_bits,
    vptq_outlier_enabled,
)


log = setup_logger()

_HESSIAN_DTYPE = torch.float64
_DISTANCE_CHUNK_BYTES = 256 * 1024 * 1024
_EPS = 1e-12


def _matrix_to_vectors(data: torch.Tensor, vector_len: int) -> Tuple[torch.Tensor, int]:
    """Split a `[rows, cols]` matrix into `vector_len`-sized vectors along rows.

    Each input column becomes `ceil(rows / vector_len)` vectors, zero-padded at
    the end. Returns `(vectors, padded_rows)`.
    """

    rows, cols = data.shape
    pad = (-rows) % vector_len
    if pad:
        data = torch.cat(
            [data, torch.zeros((pad, cols), dtype=data.dtype, device=data.device)],
            dim=0,
        )
    vectors = data.permute(1, 0).reshape(cols, -1, vector_len).reshape(-1, vector_len)
    return vectors, rows + pad


def _vectors_to_matrix(vectors: torch.Tensor, cols: int, padded_rows: int, rows: int) -> torch.Tensor:
    """Inverse of :func:`_matrix_to_vectors`; drops the zero padding rows."""

    data = vectors.reshape(cols, padded_rows // vectors.shape[-1], vectors.shape[-1])
    data = data.permute(1, 0, 2).reshape(padded_rows, cols)
    return data[:rows]


def _chunked_assign(
    data: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_bytes: int = _DISTANCE_CHUNK_BYTES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each row of `data` to its nearest centroid.

    Returns `(assignment, squared_distance_to_assigned_centroid)`. Distance
    blocks are chunked so peak memory stays bounded for large codebooks.
    """

    num_rows = data.shape[0]
    num_centroids = centroids.shape[0]
    rows_per_chunk = max(1, chunk_bytes // max(1, num_centroids * data.element_size()))

    centroids_norm = (centroids * centroids).sum(dim=1)
    assignment = torch.empty(num_rows, dtype=torch.int64, device=data.device)
    residual = torch.empty(num_rows, dtype=data.dtype, device=data.device)

    for start in range(0, num_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, num_rows)
        block = data[start:end]
        # `||c||^2 - 2 x.c` is the expanded squared distance minus `||x||^2`.
        # The omitted term is constant per row, so it cannot change the argmin,
        # but the returned residual must add it back to be a true distance.
        distance = centroids_norm.unsqueeze(0) - 2.0 * (block @ centroids.t())
        best_distance, best_index = distance.min(dim=1)
        assignment[start:end] = best_index
        residual[start:end] = best_distance + (block * block).sum(dim=1)
    return assignment, residual


def _weighted_kmeans(
    data: torch.Tensor,
    num_centroids: int,
    weights: Optional[torch.Tensor] = None,
    *,
    iters: int,
    tol: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic weighted k-means (Lloyd) on `[num_vectors, vector_len]`.

    Always returns exactly `num_centroids` rows so the packed codebook keeps
    the size declared by the configuration. When there are fewer vectors than
    requested centroids (small layers with large codebooks), the missing
    entries are initialized as duplicates of existing vectors; they receive no
    assignments because ties resolve to the lowest index, so every vector ends
    up with its own centroid and the reconstruction is exact.
    """

    num_rows = data.shape[0]
    if num_centroids < 1:
        raise ValueError("VPTQ: k-means needs at least one vector.")

    generator = torch.Generator(device=data.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_rows, generator=generator, device=data.device)
    if num_centroids <= num_rows:
        init_index = permutation[:num_centroids]
    else:
        repeats = (num_centroids + num_rows - 1) // num_rows
        init_index = permutation.repeat(repeats)[:num_centroids]
    centroids = data[init_index].clone()

    if weights is None:
        weights = torch.ones(num_rows, dtype=data.dtype, device=data.device)
    else:
        weights = weights.to(dtype=data.dtype, device=data.device)

    assignment = None
    previous_inertia = None
    for _ in range(max(1, int(iters))):
        assignment, residual = _chunked_assign(data, centroids)
        inertia = (residual.double() * weights.double()).sum().item()

        weighted = data * weights.unsqueeze(1)
        sums = torch.zeros_like(centroids).index_add_(0, assignment, weighted)
        counts = torch.zeros(num_centroids, dtype=data.dtype, device=data.device).index_add_(
            0, assignment, weights
        )
        updated = sums / counts.clamp_min(_EPS).unsqueeze(1)
        empty = counts <= 0
        if bool(empty.any()):
            updated[empty] = centroids[empty]
        centroids = updated

        if previous_inertia is not None:
            denominator = max(abs(previous_inertia), _EPS)
            if abs(previous_inertia - inertia) <= tol * denominator:
                break
        previous_inertia = inertia

    return centroids, assignment


@dataclass
class VPTQQuantResult:
    """Per-module VPTQ codebooks and indices.

    Dictionary keys follow the VPTQ group layout: key `0` is the outlier
    codebook (when enabled), keys `1..group_num` are the non-outlier groups.
    """

    centroids: Dict[int, Optional[torch.Tensor]] = field(default_factory=dict)
    indices: Dict[int, Optional[torch.Tensor]] = field(default_factory=dict)
    res_centroids: Dict[int, Optional[torch.Tensor]] = field(default_factory=dict)
    res_indices: Dict[int, Optional[torch.Tensor]] = field(default_factory=dict)
    weight_scale: Optional[torch.Tensor] = None
    weight_bias: Optional[torch.Tensor] = None
    perm: Optional[torch.Tensor] = None
    outlier_size: int = 0
    group_size: int = 0
    group_num: int = 0
    index_bits: int = 0
    res_index_bits: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    quantized_weight: Optional[torch.Tensor] = None


class VPTQ:
    """Torch VPTQ quantizer for a single `nn.Linear` or `transformers.Conv1D` module."""

    def __init__(self, module: nn.Module, qcfg: Optional[VPTQConfig] = None):
        self.qcfg = qcfg if qcfg is not None else VPTQConfig()

        if isinstance(module, NamedModule):
            self._named_module = module
            self.layer = module.module
            self.name = module.name
        else:
            self._named_module = None
            self.layer = module
            self.name = getattr(module, "name", module.__class__.__name__)

        self._validate_module(self.layer)

        weight = self.layer.weight
        if isinstance(self.layer, transformers.Conv1D):
            # transformers.Conv1D stores `(in_features, out_features)`.
            rows, columns = weight.shape[1], weight.shape[0]
        else:
            rows, columns = weight.shape[0], weight.shape[1]
        self.rows = rows
        self.columns = columns

        self.dev = weight.device
        self.nsamples = 0
        self.fwd_counter = 0

        self.lock = threading.Lock()
        self.H2: Optional[torch.Tensor] = None
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}

        self.fallback = getattr(self.qcfg, "fallback", None)
        self.fallback_used = False
        self.expected_nsamples: Optional[float] = None
        self.quant_result: Optional[VPTQQuantResult] = None

    @staticmethod
    def _validate_module(module: nn.Module) -> None:
        if not isinstance(module, (nn.Linear, transformers.Conv1D)):
            raise ValueError(
                f"VPTQ supports only `nn.Linear` and `transformers.Conv1D` layers, got "
                f"`{module.__class__.__name__}`: the VPTQ runtime reconstructs a dense linear "
                "weight and cannot preserve convolutional semantics."
            )

    def free(self) -> None:
        self.H2 = None
        self._device_hessian_partials.clear()
        self._device_sample_counts.clear()

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor] = None) -> None:
        """Accumulate `H = sum(x^T x)` for one calibration batch."""

        del out
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()

        dev = inp.device
        batch_tokens = inp.shape[1]
        inp = inp.to(dtype=_HESSIAN_DTYPE)

        with self.lock:
            self.fwd_counter += 1
            partial = self._device_hessian_partials.get(dev)
            if partial is None:
                partial = torch.zeros((self.columns, self.columns), dtype=_HESSIAN_DTYPE, device=dev)
                self._device_hessian_partials[dev] = partial
            partial.addmm_(inp, inp.t(), beta=1.0, alpha=1.0)
            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_tokens
            self.nsamples += batch_tokens

    def materialize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """Merge per-device partials into `H = mean(x^T x)` on `target_device`."""

        if target_device is None:
            target_device = self.dev
        target_device = torch.device(target_device)

        total_samples = sum(self._device_sample_counts.values())
        if total_samples == 0:
            self.H2 = torch.zeros(
                (self.columns, self.columns), dtype=_HESSIAN_DTYPE, device=target_device
            )
            self.nsamples = 0
            return self.H2

        merged = torch.zeros((self.columns, self.columns), dtype=_HESSIAN_DTYPE, device=target_device)
        while self._device_hessian_partials:
            dev, partial = self._device_hessian_partials.popitem()
            if partial.device == target_device and partial.dtype == _HESSIAN_DTYPE:
                merged.add_(partial)
            else:
                merged.add_(partial.to(device=target_device, dtype=_HESSIAN_DTYPE))
            del partial

        merged.div_(float(total_samples))
        self.H2 = merged
        self.nsamples = total_samples
        self._device_sample_counts.clear()
        return self.H2

    def _layer_weight(self) -> torch.Tensor:
        weight = self.layer.weight.data
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        return weight.float()

    def _target_dtype(self) -> torch.dtype:
        layer = getattr(self, "layer", None)
        dtype = layer.weight.data.dtype if layer is not None else torch.float32
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            dtype = torch.float32
        return dtype

    @staticmethod
    def _hessian_is_usable(hessian: torch.Tensor) -> bool:
        """Report whether the Hessian holds any calibration information."""

        if not bool(torch.isfinite(hessian).all()):
            return False
        return float(torch.diagonal(hessian).abs().sum().item()) > 0.0

    @torch.no_grad()
    def quantize(self) -> Tuple[VPTQQuantResult, float, float]:
        """Run VPTQ and return `(result, duration_seconds, hessian_weighted_loss)`."""

        start = time.time()
        hessian = self.materialize_hessian(self.dev)
        weight = self._layer_weight().to(self.dev)
        result, loss = self.quantize_weight(weight, hessian)
        duration = time.time() - start
        result.stats["loss"] = loss
        result.stats["nsamples"] = self.nsamples
        result.stats["duration"] = duration
        self.quant_result = result
        return result, duration, loss

    @torch.no_grad()
    def quantize_weight(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[VPTQQuantResult, float]:
        """Quantize `weight` (`[out_features, in_features]`) with `hessian`."""

        qcfg = self.qcfg
        device = weight.device
        rows, columns = weight.shape
        vector_lens = [int(item) for item in qcfg.vector_lens]
        num_centroids = [int(item) for item in qcfg.num_centroids]
        num_res_centroids = [int(item) for item in qcfg.num_res_centroids]
        target_dtype = target_dtype if target_dtype is not None else self._target_dtype()
        self.fallback_used = False

        outlier_size, group_size, group_num = vptq_group_setting(
            columns,
            npercent=qcfg.npercent,
            group_size=qcfg.group_size,
            group_num=qcfg.group_num,
            enable_outlier=vptq_outlier_enabled(
                npercent=qcfg.npercent,
                outlier_vector_len=vector_lens[0],
                outlier_num_centroids=num_centroids[0],
            ),
        )
        self._outlier_size = outlier_size
        self._group_size = group_size

        hessian = hessian.to(device=device, dtype=_HESSIAN_DTYPE).clone()
        if not self._hessian_is_usable(hessian):
            # A module that never received calibration traffic has no Hessian at
            # all. The per-column rule for dead channels would zero the whole
            # weight matrix and silently destroy the layer, so degrade to plain
            # vector quantization instead.
            log.warn(
                f"VPTQ: `{self.name}` has no usable Hessian (samples={self.nsamples}); "
                "using the Hessian-free vector quantization fallback."
            )
            hessian = torch.eye(columns, dtype=_HESSIAN_DTYPE, device=device)
            self.fallback_used = True
        elif should_use_fallback(self.fallback, float(self.nsamples), self.expected_nsamples):
            log.warn(
                f"VPTQ: `{self.name}` observed `{self.nsamples}` calibration tokens, below the "
                "configured `fallback` threshold; using the Hessian-free vector quantization fallback."
            )
            hessian = torch.eye(columns, dtype=_HESSIAN_DTYPE, device=device)
            self.fallback_used = True

        dead = torch.diagonal(hessian) == 0
        if bool(dead.any()):
            hessian[dead, dead] = 1.0
            weight = weight.clone()
            weight[:, dead] = 0.0
        weight_original = weight.clone()

        mean_diag = torch.diagonal(hessian).mean()
        if float(mean_diag) > 0:
            hessian = hessian / mean_diag

        weight_scale = None
        weight_bias = None
        if qcfg.enable_norm:
            norm_dim = int(qcfg.norm_dim)
            weight_min = weight.amin(dim=norm_dim)
            weight_max = weight.amax(dim=norm_dim)
            scale = (weight_max - weight_min).clamp_min(_EPS)
            weight = (weight - weight_min.unsqueeze(norm_dim)) / scale.unsqueeze(norm_dim)
            weight_scale = scale
            weight_bias = weight_min

        kmeans_weight = None
        if qcfg.kmeans_mode == "hessian":
            kmeans_weight = torch.diagonal(hessian).to(weight.dtype).unsqueeze(0).expand(rows, columns).contiguous()

        perm = None
        if qcfg.enable_perm:
            perm = torch.argsort(torch.diagonal(hessian), descending=True)
            weight = weight[:, perm]
            hessian = hessian[perm][:, perm]
            if kmeans_weight is not None:
                kmeans_weight = kmeans_weight[:, perm]

        group_bounds = [(0, outlier_size)] + [
            (
                outlier_size + index * group_size,
                min(columns, outlier_size + (index + 1) * group_size),
            )
            for index in range(group_num)
        ]

        centroids: Dict[int, Optional[torch.Tensor]] = {}
        working_centroids: Dict[int, Optional[torch.Tensor]] = {}
        for cidx, (start, end) in enumerate(group_bounds):
            vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
            num_centroids_for_group = num_centroids[0] if cidx == 0 else num_centroids[1]
            if num_centroids_for_group <= 1 or end <= start:
                centroids[cidx] = None
                working_centroids[cidx] = None
                continue

            vectors, _ = _matrix_to_vectors(weight[:, start:end], vector_len)
            group_weights = None
            if kmeans_weight is not None:
                weight_vectors, _ = _matrix_to_vectors(kmeans_weight[:, start:end], vector_len)
                group_weights = weight_vectors.mean(dim=1)
            group_centroids, _ = _weighted_kmeans(
                vectors,
                num_centroids_for_group,
                group_weights,
                iters=qcfg.kmeans_iters,
                tol=qcfg.kmeans_tol,
                seed=qcfg.kmeans_seed,
            )
            # Round-trip through the storage dtype so quantization math and
            # runtime dequantization see identical codebook values.
            centroids[cidx] = group_centroids.to(dtype=target_dtype)
            working_centroids[cidx] = centroids[cidx].to(dtype=torch.float32)

        res_centroids: Dict[int, Optional[torch.Tensor]] = dict.fromkeys(centroids)
        working_res_centroids: Dict[int, Optional[torch.Tensor]] = dict.fromkeys(centroids)
        enable_residual = num_res_centroids[1] > 1

        quantized, qerror, indices, res_indices = self._vptq_pass(
            weight=weight,
            hessian=hessian,
            group_bounds=group_bounds,
            vector_lens=vector_lens,
            centroids=working_centroids,
            res_centroids=None,
            enable_residual=False,
        )

        if enable_residual:
            for cidx, (start, end) in enumerate(group_bounds):
                vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
                num_res = num_res_centroids[0] if cidx == 0 else num_res_centroids[1]
                if num_res <= 1 or end <= start:
                    continue
                residual_vectors, _ = _matrix_to_vectors(qerror[:, start:end], vector_len)
                group_weights = None
                if kmeans_weight is not None:
                    weight_vectors, _ = _matrix_to_vectors(kmeans_weight[:, start:end], vector_len)
                    group_weights = weight_vectors.mean(dim=1)
                group_res_centroids, _ = _weighted_kmeans(
                    residual_vectors,
                    num_res,
                    group_weights,
                    iters=qcfg.kmeans_iters,
                    tol=qcfg.kmeans_tol,
                    seed=qcfg.kmeans_seed,
                )
                res_centroids[cidx] = group_res_centroids.to(dtype=target_dtype)
                working_res_centroids[cidx] = res_centroids[cidx].to(dtype=torch.float32)

            quantized, qerror, indices, res_indices = self._vptq_pass(
                weight=weight,
                hessian=hessian,
                group_bounds=group_bounds,
                vector_lens=vector_lens,
                centroids=working_centroids,
                res_centroids=working_res_centroids,
                enable_residual=True,
            )

        if perm is not None:
            inverse_perm = torch.argsort(perm)
            quantized = quantized[:, inverse_perm]
            hessian = hessian[inverse_perm][:, inverse_perm]

        if qcfg.enable_norm:
            norm_dim = int(qcfg.norm_dim)
            # Mirror the runtime kernel: codebooks and scales are stored in the
            # model dtype, so apply the affine transform in that dtype too.
            scale = weight_scale.to(dtype=target_dtype)
            bias = weight_bias.to(dtype=target_dtype)
            quantized = (
                quantized.to(dtype=target_dtype) * scale.unsqueeze(norm_dim) + bias.unsqueeze(norm_dim)
            ).to(dtype=torch.float32)

        loss = self.hessian_weighted_error(
            weight_original=weight_original, quantized=quantized, hessian=hessian
        )

        result = VPTQQuantResult(
            centroids=centroids,
            indices=indices,
            res_centroids=res_centroids,
            res_indices=res_indices,
            weight_scale=weight_scale,
            weight_bias=weight_bias,
            perm=perm,
            outlier_size=outlier_size,
            group_size=group_size,
            group_num=group_num,
            index_bits=vptq_index_bits(num_centroids[1]),
            res_index_bits=vptq_index_bits(num_res_centroids[1]) if enable_residual else 0,
        )
        result.quantized_weight = quantized.to(dtype=target_dtype)
        result.stats["fallback"] = self.fallback_used
        return result, loss

    def _inverse_hessian_cholesky(self, hessian: torch.Tensor) -> torch.Tensor:
        """Return `U` with `U^T U = (H + damp * mean(diag(H)) I)^-1`, `U` upper."""

        damp = float(self.qcfg.percdamp) * float(torch.diagonal(hessian).mean())
        damped = hessian.clone()
        diagonal = torch.arange(damped.shape[0], device=damped.device)
        damped[diagonal, diagonal] += damp

        try:
            cholesky = torch.linalg.cholesky(damped)
        except RuntimeError as exc:
            raise RuntimeError(
                f"VPTQ: Hessian for `{self.name}` is not positive definite; "
                "increase `percdamp` or provide more calibration data."
            ) from exc
        inverse = torch.cholesky_inverse(cholesky)
        return torch.linalg.cholesky(inverse, upper=True)

    @torch.no_grad()
    def _vptq_pass(
        self,
        *,
        weight: torch.Tensor,
        hessian: torch.Tensor,
        group_bounds,
        vector_lens,
        centroids: Dict[int, Optional[torch.Tensor]],
        res_centroids: Optional[Dict[int, Optional[torch.Tensor]]],
        enable_residual: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Optional[torch.Tensor]], Dict[int, Optional[torch.Tensor]]]:
        """Sequential vector-quantization pass with Hessian error compensation."""

        device = weight.device
        rows, columns = weight.shape
        block_size = max(1, int(self.qcfg.block_size))

        index_tensors: Dict[int, Optional[torch.Tensor]] = {}
        res_index_tensors: Dict[int, Optional[torch.Tensor]] = {}
        for cidx, (start, end) in enumerate(group_bounds):
            codebook = centroids.get(cidx)
            if codebook is None or end <= start:
                index_tensors[cidx] = None
                res_index_tensors[cidx] = None
                continue
            vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
            num_indices = math.ceil(rows / vector_len)
            index_tensors[cidx] = torch.zeros((num_indices, end - start), dtype=torch.int64, device=device)
            res_codebook = None if res_centroids is None else res_centroids.get(cidx)
            if enable_residual and res_codebook is not None:
                res_index_tensors[cidx] = torch.zeros_like(index_tensors[cidx])
            else:
                res_index_tensors[cidx] = None

        inv_hessian = self._inverse_hessian_cholesky(hessian).to(dtype=torch.float32)
        diag = torch.diagonal(inv_hessian)
        if bool((diag <= 0).any()):
            raise RuntimeError(f"VPTQ: non-positive inverse-Hessian diagonal for `{self.name}`.")
        # GPTQ Cholesky formulation: `U` is the upper Cholesky factor of
        # H^-1, so `U_kk^2 == (H^-1)_kk`; dividing the column error by `U_kk`
        # is the update rule used by the reference GPTQ implementation.
        inv_scale = 1.0 / diag

        weight = weight.clone().float()
        quantized = torch.zeros_like(weight)
        qerror = torch.zeros_like(weight)

        for block_start in range(0, columns, block_size):
            block_end = min(block_start + block_size, columns)
            block_weight = weight[:, block_start:block_end].clone()
            block_qweight = torch.zeros_like(block_weight)
            block_error = torch.zeros_like(block_weight)
            block_inv = inv_hessian[block_start:block_end, block_start:block_end]

            for k in range(block_end - block_start):
                column_index = block_start + k
                column = block_weight[:, k : k + 1]
                q_column, index, res_index = self._quantize_column(
                    column,
                    column_index,
                    group_bounds=group_bounds,
                    vector_lens=vector_lens,
                    centroids=centroids,
                    res_centroids=res_centroids if enable_residual else None,
                )
                block_qweight[:, k : k + 1] = q_column

                cidx = vptq_group_for_column(
                    column_index, outlier_size=self._outlier_size, group_size=self._group_size
                )
                if index is not None and index_tensors.get(cidx) is not None:
                    group_start = group_bounds[cidx][0]
                    index_tensors[cidx][:, column_index - group_start] = index
                if res_index is not None and res_index_tensors.get(cidx) is not None:
                    group_start = group_bounds[cidx][0]
                    res_index_tensors[cidx][:, column_index - group_start] = res_index

                error = (column - q_column) * inv_scale[column_index]
                if k + 1 < block_end - block_start:
                    block_weight[:, k + 1 :] -= error @ block_inv[k : k + 1, k + 1 :]
                block_error[:, k : k + 1] = error

            quantized[:, block_start:block_end] = block_qweight
            qerror[:, block_start:block_end] = block_weight - block_qweight
            if block_end < columns:
                weight[:, block_end:] -= block_error @ inv_hessian[block_start:block_end, block_end:]

        return quantized, qerror, index_tensors, res_index_tensors

    def _quantize_column(
        self,
        column: torch.Tensor,
        column_index: int,
        *,
        group_bounds,
        vector_lens,
        centroids: Dict[int, Optional[torch.Tensor]],
        res_centroids: Optional[Dict[int, Optional[torch.Tensor]]],
    ):
        del group_bounds
        cidx = vptq_group_for_column(
            column_index, outlier_size=self._outlier_size, group_size=self._group_size
        )
        codebook = centroids.get(cidx)
        if codebook is None:
            return column.clone(), None, None
        codebook = codebook.to(dtype=column.dtype)

        vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
        rows = column.shape[0]
        vectors, padded_rows = _matrix_to_vectors(column, vector_len)
        assignment, _ = _chunked_assign(vectors, codebook)
        selected = codebook[assignment]

        res_assignment = None
        res_codebook = None if res_centroids is None else res_centroids.get(cidx)
        if res_codebook is not None:
            res_codebook = res_codebook.to(dtype=column.dtype)
            residual = vectors - selected
            res_assignment, _ = _chunked_assign(residual, res_codebook)
            selected = selected + res_codebook[res_assignment]

        quantized_column = _vectors_to_matrix(selected, 1, padded_rows, rows)
        return quantized_column, assignment, res_assignment

    @staticmethod
    @torch.no_grad()
    def hessian_weighted_error(
        *,
        weight_original: torch.Tensor,
        quantized: torch.Tensor,
        hessian: torch.Tensor,
    ) -> float:
        """Normalized Hessian-weighted reconstruction error `E[e^T e H] / E[w^T w H]`."""

        delta = quantized.to(torch.float64) - weight_original.to(torch.float64)
        hessian = hessian.to(torch.float64)
        rows = delta.shape[0]
        chunk = max(1, int(64 * 1024 * 1024 // max(1, delta.shape[1] * 8)))

        error_sum = 0.0
        weight_sum = 0.0
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            error_block = delta[start:end] @ hessian
            weight_block = weight_original[start:end].to(torch.float64) @ hessian
            error_sum += float((error_block * delta[start:end]).sum().item())
            weight_sum += float((weight_block * weight_original[start:end].to(torch.float64)).sum().item())

        denominator = max(abs(weight_sum), _EPS)
        return error_sum / denominator
