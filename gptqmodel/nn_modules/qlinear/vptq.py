# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Torch runtime for VPTQ (Vector Post-Training Quantization) checkpoints.
# Codebook / index layout follows the VPTQ reference implementation
# (https://github.com/microsoft/VPTQ, MIT licensed) so checkpoints stay
# interoperable with the upstream ecosystem.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.vptq import (
    unpack_vptq_indices,
    vptq_group_setting,
    vptq_index_bits,
    vptq_outlier_enabled,
    vptq_packed_words,
)
from . import WeightOnlyQuantLinear


log = setup_logger()


def _as_pair(value: Any, name: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"VPTQLinear: `{name}` must be a 2-item list/tuple, got `{value}`.")
    return [int(item) for item in value]


class VPTQLinear(WeightOnlyQuantLinear):
    """Weight-only VPTQ linear layer with a Torch dequantization kernel."""

    SUPPORTS_BACKENDS = [BACKEND.VPTQ_TORCH]
    SUPPORTS_METHODS = [METHOD.VPTQ]
    SUPPORTS_FORMATS = {FORMAT.VPTQ: 100}
    SUPPORTS_BITS = [1, 2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "vptq"

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        vector_lens=(8, 8),
        num_centroids=(4096, 4096),
        num_res_centroids=(-1, -1),
        npercent: float = 0.0,
        group_size: int = -1,
        group_num: int = -1,
        kmeans_mode: str = "",
        kmeans_iters: int = 100,
        kmeans_tol: float = 1e-5,
        kmeans_seed: int = 0,
        enable_norm: bool = False,
        norm_dim: int = 0,
        enable_perm: bool = False,
        block_size: int = 128,
        percdamp: float = 0.01,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            bits=int(bits),
            in_features=int(in_features),
            out_features=int(out_features),
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.VPTQ_TORCH),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            **kwargs,
        )

        if not bias and "bias" not in self._buffers and "bias" not in self._parameters:
            self.bias = None

        self.vector_lens = _as_pair(vector_lens, "vector_lens")
        self.num_centroids_pair = _as_pair(num_centroids, "num_centroids")
        self.num_res_centroids_pair = _as_pair(num_res_centroids, "num_res_centroids")
        self.npercent = float(npercent)
        self.kmeans_mode = kmeans_mode
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_tol = float(kmeans_tol)
        self.kmeans_seed = int(kmeans_seed)
        self.enable_norm = bool(enable_norm)
        self.norm_dim = int(norm_dim)
        self.enable_perm = bool(enable_perm)
        self.block_size = int(block_size)
        self.percdamp = float(percdamp)
        self.codebook_dtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16

        self.vector_len = self.vector_lens[1]
        self.outlier_vector_len = self.vector_lens[0] if self.vector_lens[0] > 0 else 0
        self.num_centroids = self.num_centroids_pair[1]
        self.num_res_centroids = self.num_res_centroids_pair[1]
        if self.num_res_centroids == 1:
            raise ValueError(
                "VPTQLinear: `num_res_centroids[1]` must be -1 (disabled) or a power-of-two "
                "integer of at least 2; a one-entry residual codebook has no index bits and "
                "cannot be packed."
            )
        self.enable_residual = self.num_res_centroids > 1
        self.enable_outlier = vptq_outlier_enabled(
            npercent=self.npercent,
            outlier_vector_len=self.outlier_vector_len,
            outlier_num_centroids=self.num_centroids_pair[0],
        )
        if self.npercent > 0 and not self.enable_outlier:
            raise ValueError(
                "VPTQLinear: `npercent` > 0 requires positive `vector_lens[0]` and `num_centroids[0]` > 1."
            )
        self.outlier_size, self.group_size, self.group_num = vptq_group_setting(
            self.in_features,
            npercent=self.npercent,
            group_size=group_size,
            group_num=group_num,
            enable_outlier=self.enable_outlier,
        )
        if self.outlier_size + self.group_num * self.group_size < self.in_features:
            raise ValueError(
                f"VPTQLinear: the resolved group layout covers "
                f"{self.outlier_size + self.group_num * self.group_size} of {self.in_features} input channels."
            )

        self.index_bits = vptq_index_bits(self.num_centroids)
        self.res_index_bits = vptq_index_bits(self.num_res_centroids) if self.enable_residual else 0
        self.num_indices = math.ceil(self.out_features / self.vector_len)
        self.row_padding = self.num_indices * self.vector_len - self.out_features
        self.packed_words = vptq_packed_words(self.group_size, self.index_bits + self.res_index_bits)

        if self.enable_outlier:
            self.outlier_num_centroids = self.num_centroids_pair[0]
            self.outlier_num_indices = math.ceil(self.out_features / self.outlier_vector_len)
            self.outlier_row_padding = (
                self.outlier_num_indices * self.outlier_vector_len - self.out_features
            )
        else:
            self.outlier_num_centroids = 0
            self.outlier_num_indices = 0
            self.outlier_row_padding = 0

        self.centroids = nn.Embedding(self.group_num, self.num_centroids * self.vector_len)
        self.centroids.weight.requires_grad_(False)
        if self.enable_residual:
            self.res_centroids = nn.Embedding(self.group_num, self.num_res_centroids * self.vector_len)
            self.res_centroids.weight.requires_grad_(False)
        else:
            self.res_centroids = None
        if self.enable_outlier:
            self.outlier_centroids = nn.Embedding(
                1, self.outlier_num_centroids * self.outlier_vector_len
            )
            self.outlier_centroids.weight.requires_grad_(False)
        else:
            self.outlier_centroids = None

        if register_buffers:
            self._allocate_buffers(bias=bias)
        elif not bias:
            # Keep `bias` defined so `forward`/`dequantize` can query it before
            # `pack()` allocates the real buffers.
            self.bias = None

    # ------------------------------------------------------------------
    # buffer management
    # ------------------------------------------------------------------
    def _allocate_buffers(self, *, bias: bool) -> None:
        """Create the codebook parameters and packed-index buffers.

        The quantization looper runs under `torch.inference_mode()`, and tensors
        born inside that mode keep inference-only bookkeeping even after a
        later `.data` assignment or device move, which breaks `.view()` and
        in-place updates during inference. Rebuild the storage explicitly
        outside inference mode.
        """

        dtype = self.codebook_dtype
        device = self.centroids.weight.device

        with torch.inference_mode(False):
            self.centroids = nn.Embedding(
                self.group_num, self.num_centroids * self.vector_len, dtype=dtype, device=device
            )
            self.centroids.weight.requires_grad_(False)
            if self.enable_residual:
                self.res_centroids = nn.Embedding(
                    self.group_num, self.num_res_centroids * self.vector_len, dtype=dtype, device=device
                )
                self.res_centroids.weight.requires_grad_(False)
            if self.enable_outlier:
                self.outlier_centroids = nn.Embedding(
                    1, self.outlier_num_centroids * self.outlier_vector_len, dtype=dtype, device=device
                )
                self.outlier_centroids.weight.requires_grad_(False)

            self.register_buffer(
                "indices",
                torch.zeros(
                    (self.group_num, self.num_indices, self.packed_words),
                    dtype=torch.int32,
                    device=device,
                ),
            )
            if self.enable_outlier:
                self.register_buffer(
                    "outlier_indices",
                    torch.zeros(
                        (1, self.outlier_num_indices, self.outlier_size),
                        # int32: an outlier codebook may hold more than 32768
                        # entries, and Torch gather cannot read a wrapped
                        # int16 the way the reference CUDA kernel reads uint16.
                        dtype=torch.int32,
                        device=device,
                    ),
                )
            if self.enable_norm:
                size = self.in_features if self.norm_dim == 0 else self.out_features
                self.register_buffer("weight_scale", torch.zeros(size, dtype=dtype, device=device))
                self.register_buffer("weight_bias", torch.zeros(size, dtype=dtype, device=device))
            if self.enable_perm:
                self.register_buffer(
                    "perm",
                    torch.arange(self.in_features, dtype=torch.int32, device=device),
                    persistent=True,
                )
            if bias:
                # `bias` may already exist as a plain attribute from `__init__`.
                self.__dict__.pop("bias", None)
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=dtype, device=device))
            else:
                self.bias = None

    def post_init(self):
        self._validate_loaded_state()
        super().post_init()

    def _validate_loaded_state(self) -> None:
        if self.indices.shape != (self.group_num, self.num_indices, self.packed_words):
            raise ValueError(
                f"VPTQLinear `{self.name}`: unexpected `indices` shape {tuple(self.indices.shape)}; "
                f"expected {(self.group_num, self.num_indices, self.packed_words)}."
            )

    # ------------------------------------------------------------------
    # packing API
    # ------------------------------------------------------------------
    def pack(self, linear: nn.Module = None, *, vptq_result=None, **kwargs):
        """Fill the VPTQ buffers from a :class:`VPTQQuantResult`.

        `linear` is accepted for API compatibility with other kernels; VPTQ
        quantization is Hessian-driven and is performed by `VPTQProcessor`, so
        the source module only contributes its bias.
        """

        del kwargs
        if vptq_result is None:
            raise ValueError(
                "VPTQLinear.pack requires a `vptq_result` produced by the VPTQ quantizer."
            )
        bias = None if linear is None else getattr(linear, "bias", None)
        if not hasattr(self, "indices"):
            self._allocate_buffers(bias=bias is not None)
        self.load_quant_result(vptq_result, bias=bias)

    @torch.no_grad()
    def load_quant_result(self, result, bias: Optional[torch.Tensor] = None) -> None:
        """Copy a `VPTQQuantResult` into this module's buffers."""

        device = self.centroids.weight.device
        codebooks = []
        for cidx in range(1, self.group_num + 1):
            codebook = result.centroids.get(cidx)
            if codebook is None:
                raise ValueError(f"VPTQLinear `{self.name}`: missing codebook for group {cidx}.")
            codebooks.append(codebook.to(device=device, dtype=self.codebook_dtype))
        self.centroids.weight.data.copy_(
            torch.stack(codebooks, dim=0).reshape(self.group_num, -1)
        )

        if self.enable_residual:
            res_codebooks = []
            for cidx in range(1, self.group_num + 1):
                codebook = result.res_centroids.get(cidx)
                if codebook is None:
                    raise ValueError(
                        f"VPTQLinear `{self.name}`: missing residual codebook for group {cidx}."
                    )
                res_codebooks.append(codebook.to(device=device, dtype=self.codebook_dtype))
            self.res_centroids.weight.data.copy_(
                torch.stack(res_codebooks, dim=0).reshape(self.group_num, -1)
            )

        if self.enable_outlier:
            codebook = result.centroids.get(0)
            if codebook is None:
                raise ValueError(f"VPTQLinear `{self.name}`: missing outlier codebook.")
            self.outlier_centroids.weight.data.copy_(
                codebook.to(device=device, dtype=self.codebook_dtype).reshape(1, -1)
            )
            outlier_indices = result.indices.get(0)
            if outlier_indices is None:
                raise ValueError(f"VPTQLinear `{self.name}`: missing outlier indices.")
            if int(outlier_indices.max().item()) >= self.outlier_num_centroids:
                raise ValueError(
                    f"VPTQLinear `{self.name}`: outlier index exceeds the "
                    f"{self.outlier_num_centroids}-entry outlier codebook."
                )
            self.outlier_indices.data.copy_(
                outlier_indices.to(device=device, dtype=torch.int32).reshape(
                    1, self.outlier_num_indices, self.outlier_size
                )
            )

        from ...utils.vptq import pack_vptq_indices

        index_rows = []
        for cidx in range(1, self.group_num + 1):
            group_indices = result.indices.get(cidx)
            if group_indices is None:
                raise ValueError(f"VPTQLinear `{self.name}`: missing indices for group {cidx}.")
            if group_indices.shape[1] < self.group_size:
                pad = torch.zeros(
                    (group_indices.shape[0], self.group_size - group_indices.shape[1]),
                    dtype=group_indices.dtype,
                    device=group_indices.device,
                )
                group_indices = torch.cat([group_indices, pad], dim=1)
            res_group_indices = None
            if self.enable_residual:
                res_group_indices = result.res_indices.get(cidx)
                if res_group_indices is None:
                    raise ValueError(
                        f"VPTQLinear `{self.name}`: missing residual indices for group {cidx}."
                    )
                if res_group_indices.shape[1] < self.group_size:
                    pad = torch.zeros(
                        (res_group_indices.shape[0], self.group_size - res_group_indices.shape[1]),
                        dtype=res_group_indices.dtype,
                        device=res_group_indices.device,
                    )
                    res_group_indices = torch.cat([res_group_indices, pad], dim=1)
            index_rows.append(
                pack_vptq_indices(
                    group_indices,
                    self.index_bits,
                    res_indices=res_group_indices,
                    res_bits=self.res_index_bits,
                )
            )
        self.indices.data.copy_(torch.stack(index_rows, dim=0).to(device=device, dtype=torch.int32))

        if self.enable_norm:
            self.weight_scale.data.copy_(
                result.weight_scale.to(device=device, dtype=self.codebook_dtype)
            )
            self.weight_bias.data.copy_(
                result.weight_bias.to(device=device, dtype=self.codebook_dtype)
            )
        if self.enable_perm:
            perm = result.perm
            if perm is None:
                raise ValueError(f"VPTQLinear `{self.name}`: missing permutation.")
            self.perm.data.copy_(perm.to(device=device, dtype=torch.int32))

        if bias is not None:
            if getattr(self, "bias", None) is None:
                raise ValueError(
                    f"VPTQLinear `{self.name}`: received a bias but the module was built without one."
                )
            self.bias.data.copy_(bias.detach().to(device=device, dtype=self.bias.dtype))

    # ------------------------------------------------------------------
    # runtime
    # ------------------------------------------------------------------
    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """Reconstruct the dense weight matrix from codebooks and indices."""

        device = self.centroids.weight.device
        centroids = self.centroids.weight.view(self.group_num, self.num_centroids, self.vector_len)

        indices, res_indices = unpack_vptq_indices(
            self.indices,
            self.index_bits,
            self.group_size,
            res_bits=self.res_index_bits,
        )

        flat_indices = (
            indices.reshape(self.group_num, -1, 1)
            .expand(-1, -1, self.vector_len)
            .reshape(self.group_num, -1, self.vector_len)
        )
        selected = torch.gather(centroids, 1, flat_indices)
        selected = selected.view(
            self.group_num, self.num_indices, self.group_size, self.vector_len
        ).permute(0, 1, 3, 2)
        qweight = selected.reshape(self.group_num, -1, self.group_size)
        qweight = qweight.permute(1, 0, 2).reshape(-1, self.group_num * self.group_size)

        if self.enable_residual:
            res_centroids = self.res_centroids.weight.view(
                self.group_num, self.num_res_centroids, self.vector_len
            )
            flat_res_indices = (
                res_indices.reshape(self.group_num, -1, 1)
                .expand(-1, -1, self.vector_len)
                .reshape(self.group_num, -1, self.vector_len)
            )
            selected_res = torch.gather(res_centroids, 1, flat_res_indices)
            selected_res = selected_res.view(
                self.group_num, self.num_indices, self.group_size, self.vector_len
            ).permute(0, 1, 3, 2)
            res_qweight = selected_res.reshape(self.group_num, -1, self.group_size)
            qweight = qweight + res_qweight.permute(1, 0, 2).reshape(
                -1, self.group_num * self.group_size
            )

        if self.row_padding > 0:
            qweight = qweight[: -self.row_padding]

        if self.enable_outlier:
            outlier_centroids = self.outlier_centroids.weight.view(
                1, self.outlier_num_centroids, self.outlier_vector_len
            )
            outlier_indices = self.outlier_indices.to(torch.int64)
            flat_outlier_indices = (
                outlier_indices.reshape(1, -1, 1)
                .expand(-1, -1, self.outlier_vector_len)
                .reshape(1, -1, self.outlier_vector_len)
            )
            selected_outlier = torch.gather(outlier_centroids, 1, flat_outlier_indices)
            selected_outlier = selected_outlier.view(
                1, self.outlier_num_indices, self.outlier_size, self.outlier_vector_len
            ).permute(0, 1, 3, 2)
            qweight_outlier = selected_outlier.reshape(-1, self.outlier_size)
            if self.outlier_row_padding > 0:
                qweight_outlier = qweight_outlier[: -self.outlier_row_padding]
            qweight = torch.cat([qweight_outlier, qweight], dim=1)

        if qweight.shape[1] > self.in_features:
            # The last group is zero-padded up to `group_size`; those padding
            # columns carry no quantized channel and must not reach the weight.
            qweight = qweight[:, : self.in_features]

        if self.enable_perm:
            qweight = qweight[:, torch.argsort(self.perm.to(torch.int64))]

        if self.enable_norm:
            if self.norm_dim == 0:
                qweight = qweight * self.weight_scale + self.weight_bias
            else:
                qweight = qweight * self.weight_scale.unsqueeze(1) + self.weight_bias.unsqueeze(1)

        if qweight.device != device:
            qweight = qweight.to(device)
        return qweight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize().to(dtype=x.dtype, device=x.device)
        bias = getattr(self, "bias", None)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        out = F.linear(x, weight, bias)
        if self.adapter is not None:
            out = self.adapter.apply(x=x, out=out)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={getattr(self, 'bias', None) is not None}, vector_lens={self.vector_lens}, "
            f"num_centroids={self.num_centroids_pair}, "
            f"num_res_centroids={self.num_res_centroids_pair}, "
            f"group_size={self.group_size}, group_num={self.group_num}, "
            f"npercent={self.npercent}"
        )

    # `select_quant_linear` validates the VPTQ contract instead of the generic
    # GPTQ-style bits/group-size contract.
    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = -1,
        desc_act: bool = False,
        sym: bool = True,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[Dict[str, Any]] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE | torch.device] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ):
        del group_size, desc_act, sym, dynamic, in_features, out_features
        ok, err = cls._validate_shared(
            pack_dtype=pack_dtype,
            dtype=dtype,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err
        if bits not in cls.SUPPORTS_BITS:
            return False, NotImplementedError(
                f"{cls} only supports `{cls.SUPPORTS_BITS}` nominal bits: actual bits = `{bits}`"
            )
        return True, None
