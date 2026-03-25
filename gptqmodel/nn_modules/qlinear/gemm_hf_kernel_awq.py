# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear
from .gemm_hf_kernel import HFKernelLinear


log = setup_logger()


class HFKernelAwqLinear(AWQuantLinear):
    """AWQ variant of HFKernelLinear — uses kernels-community gemm_int4 with AWQ weights."""

    QUANT_TYPE = "hf_kernel_awq"

    SUPPORTS_BACKENDS = [BACKEND.AWQ_HF_KERNEL]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 110}

    # inherit from HFKernelLinear
    SUPPORTS_BITS = HFKernelLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = HFKernelLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = HFKernelLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = HFKernelLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = HFKernelLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = HFKernelLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = HFKernelLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = HFKernelLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = HFKernelLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = HFKernelLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = HFKernelLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = HFKernelLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = HFKernelLinear.SUPPORTS_ADAPTERS
    REQUIRES_FORMAT_V2 = False

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    gemm_int4_forward_kernel = None
    KERNEL_REPO_ID = HFKernelLinear.KERNEL_REPO_ID

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.AWQ_HF_KERNEL)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            # Skip base buffer init, we need to manually init buffers for awq
            register_buffers=False,
            **kwargs,
        )

        self.linear_mode = None

        # Create awq buffers
        if register_buffers:
            pack_cols = max(1, self.out_features // self.pack_factor)
            qweight_shape = (self.in_features, pack_cols)
            group_size = max(int(self.group_size), 1)
            group_rows = max(1, math.ceil(self.in_features / group_size))

            self.register_buffer(
                "qweight",
                torch.zeros(qweight_shape, dtype=self.pack_dtype),
            )

            self.register_buffer(
                "qzeros",
                torch.zeros((group_rows, pack_cols), dtype=self.pack_dtype),
            )

            self.register_buffer(
                "scales",
                torch.zeros((group_rows, self.out_features), dtype=torch.float16),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        build_error = HFKernelLinear._hf_kernels_import_guard(cls.__name__)
        if build_error is not None:
            log.warning(str(build_error))
            return False, build_error

        if not HFKernelLinear._is_torch_release():
            msg = (
                f"HFKernelAwqLinear requires a release version of torch, "
                f"but found `{torch.__version__}`. "
                f"Please install a stable release (e.g. `pip install torch`)."
            )
            log.warning(msg)
            return False, RuntimeError(msg)

        try:
            from kernels import get_kernel

            repo_id = cls.KERNEL_REPO_ID
            try:
                cls.gemm_int4_forward_kernel = staticmethod(get_kernel(repo_id).gemm_int4_forward)
                log.info("HFKernelAwqLinear: loaded CPU gemm_4bit kernel from `%s`.", repo_id)
                return True, None
            except Exception:
                module, variant_name = HFKernelLinear._load_cpu_kernel_variant(repo_id)
                cls.gemm_int4_forward_kernel = staticmethod(module.gemm_int4_forward)
                log.info(
                    "HFKernelAwqLinear: loaded CPU gemm_4bit kernel from `%s` variant `%s`.",
                    repo_id,
                    variant_name,
                )
                return True, None
        except Exception as exc:  # pragma: no cover - best effort fallback
            cls.gemm_int4_forward_kernel = None
            log.warning(
                "Failed to load CPU gemm_4bit kernel from `%s`: %s. "
                "Please make sure `pip install -U kernels` is installed.",
                cls.KERNEL_REPO_ID,
                str(exc),
            )
            return False, exc

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return

        super().optimize()

    def convert_weight_packed_zp(self, block_n: int = 32):
        return HFKernelLinear.convert_weight_packed_zp(self, block_n=block_n)

    def transform_cpu(self):
        # Unpack AWQ weights directly to integer form
        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val).to(torch.uint8)
        izeros = torch.bitwise_and(izeros, max_val).to(torch.uint8)

        self.scales = self.scales.to(torch.bfloat16).contiguous()

        # AWQ has no g_idx — weights are already in natural order, just transpose
        # iweight: (in_features, out_features) -> (out_features, in_features)
        self.qweight = iweight.t().contiguous()
        self.qzeros = izeros.contiguous()

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
            self.convert_weight_packed_zp()
        else:
            raise NotImplementedError(
                "HFKernelAwqLinear only supports fused transforms on CPU devices."
            )

    def awq_weight_dequantize(self, device, dtype):
        return dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        ).to(device=device, dtype=dtype)

    @torch.no_grad()
    def _fused_op_forward(self, x):
        # AWQ has no g_idx reordering — skip ret_idx
        if x.device.type == "cpu":
            out = self.gemm_int4_forward_kernel(x, self.qweight, self.qzeros, self.scales, self.group_size)
        else:
            raise NotImplementedError
        return out

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if not self.training and not x.requires_grad and self.linear_mode is None and self.gemm_int4_forward_kernel is not None and x.device.type == "cpu":
            self.transform(x.device.type)
            self.linear_mode = "inference"
        elif self.linear_mode is None:
            self.linear_mode = "train"

        if self.linear_mode == "inference":
            out = self._fused_op_forward(x).reshape(out_shape)
        else:
            weight = self.awq_weight_dequantize(device=x.device, dtype=x.dtype)
            out = torch.matmul(x, weight).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["HFKernelAwqLinear"]
