# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CUDA inference implementation for VPTQ linear layers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ...models._const import DEVICE
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from .vptq_torch import VPTQLinear


# Use the fused CUDA kernel for batches up to four rows.
_VPTQ_CUDA_FUSED_MAX_ROWS = 4


def _vptq_cuda_fused_max_rows(module: VPTQLinear) -> int:
    return _VPTQ_CUDA_FUSED_MAX_ROWS


class VPTQCudaLinear(VPTQLinear):
    """VPTQ layer using native CUDA ops, following upstream quant_gemm dispatch."""

    SUPPORTS_BACKENDS = [BACKEND.VPTQ_CUDA]
    SUPPORTS_METHODS = [METHOD.VPTQ]
    SUPPORTS_FORMATS = {FORMAT.VPTQ: 100}
    SUPPORTS_BITS = VPTQLinear.SUPPORTS_BITS
    SUPPORTS_SHARDS = VPTQLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = VPTQLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = VPTQLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = VPTQLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = VPTQLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_PACK_DTYPES = VPTQLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = VPTQLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = VPTQLinear.SUPPORTS_PLATFORM
    SUPPORTS_DTYPES = VPTQLinear.SUPPORTS_DTYPES
    QUANT_TYPE = "vptq_cuda"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.VPTQ_CUDA)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _dequantize_cuda(self) -> torch.Tensor:
        """Reconstruct weights with the native CUDA kernel for GEMM."""
        from ...utils.vptq_cuda import vptq_cuda_dequantize

        return vptq_cuda_dequantize(
            self.indices,
            self.centroids.weight,
            self.res_centroids.weight if self.enable_residual else None,
            self.outlier_indices if self.enable_outlier else None,
            self.outlier_centroids.weight if self.enable_outlier else None,
            self.inverse_perm if self.enable_perm else None,
            self.weight_scale if self.enable_norm else None,
            self.weight_bias if self.enable_norm else None,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            vector_len=self.vector_len,
            outlier_size=self.outlier_size,
            outlier_vector_len=self.outlier_vector_len,
            num_centroids=self.num_centroids,
            num_res_centroids=self.num_res_centroids,
            index_bits=self.index_bits,
            res_bits=self.res_index_bits,
            norm_dim=self.norm_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = getattr(self, "bias", None)
        rows = self.input_rows(x)
        if rows == 0:
            out = self.empty_linear_output(x)
            if self.adapter is not None:
                out = self.adapter.apply(x=x, out=out)
            return out

        if not x.is_cuda:
            raise RuntimeError("VPTQ_CUDA requires CUDA activations; select VPTQ_TORCH for CPU inference.")
        if self.indices.device != x.device or self.centroids.weight.device != x.device:
            raise RuntimeError("VPTQ_CUDA requires VPTQ buffers and activations on the same CUDA device.")

        fused_max_rows = _vptq_cuda_fused_max_rows(self)
        if x.dtype == self.centroids.weight.dtype and rows <= fused_max_rows:
            from ...utils.vptq_cuda import vptq_cuda_linear

            out = vptq_cuda_linear(
                x.contiguous(),
                self.indices,
                self.centroids.weight,
                self.res_centroids.weight if self.enable_residual else None,
                self.outlier_indices if self.enable_outlier else None,
                self.outlier_centroids.weight if self.enable_outlier else None,
                self.inverse_perm if self.enable_perm else None,
                self.weight_scale if self.enable_norm else None,
                self.weight_bias if self.enable_norm else None,
                bias,
                group_size=self.group_size,
                in_features=self.in_features,
                out_features=self.out_features,
                vector_len=self.vector_len,
                outlier_size=self.outlier_size,
                outlier_vector_len=self.outlier_vector_len,
                num_centroids=self.num_centroids,
                num_res_centroids=self.num_res_centroids,
                index_bits=self.index_bits,
                res_bits=self.res_index_bits,
                norm_dim=self.norm_dim,
            )
        else:
            # The fused GEMV repeats weight reconstruction across activation
            # rows. For larger M, reconstruct once and let cuBLAS reuse weight.
            weight = self._dequantize_cuda().to(dtype=x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(dtype=x.dtype)
            out = F.linear(x, weight, bias)

        if self.adapter is not None:
            out = self.adapter.apply(x=x, out=out)
        return out


__all__ = ["VPTQCudaLinear"]
