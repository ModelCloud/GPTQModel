# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.mentaray import (
    apply_awq_mentaray_linear,
    awq_mentaray_repack,
    awq_to_mentaray_zero_points,
    mentaray_import_exception,
    mentaray_make_empty_g_idx,
    mentaray_make_workspace_new,
    mentaray_permute_bias,
    mentaray_permute_scales,
    mentaray_runtime_available,
    mentaray_runtime_error,
    replace_parameter,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


class AwqMentaRayLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_MENTARAY]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 91, FORMAT.MARLIN: 91}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "awq_mentaray"

    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers=False,
        **kwargs,
    ):
        self.max_par = 8
        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_MENTARAY),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        if register_buffers:
            self.register_parameter(
                "qweight",
                torch.nn.Parameter(
                    torch.empty(self.in_features, self.out_features // self.pack_factor, dtype=torch.int32),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "qzeros",
                torch.nn.Parameter(
                    torch.empty(self.in_features // self.group_size, self.out_features // self.pack_factor, dtype=torch.int32),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "scales",
                torch.nn.Parameter(
                    torch.empty(self.in_features // self.group_size, self.out_features, dtype=self.compute_dtype),
                    requires_grad=False,
                ),
            )

            if bias:
                self.register_buffer("bias", torch.zeros((out_features), dtype=self.compute_dtype))
            else:
                self.bias = None

        self.is_lm_head = False
        if kwargs.get("name") is not None and kwargs.get("lm_head_name") is not None:
            self.is_lm_head = kwargs["name"] == kwargs["lm_head_name"]

        if self.bits not in self.TYPE_MAP:
            raise ValueError(f"Unsupported num_bits = {self.bits}. Supported num_bits = {self.TYPE_MAP.keys()}")

        self.weight_type = self.TYPE_MAP[self.bits]

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if mentaray_import_exception is not None:
            return False, ImportError(mentaray_import_exception)
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("MentaRay kernel is not supported on ROCm.")
            device_count = torch.cuda.device_count() if visible is None else len(visible.split(","))
            has_sm80 = all(torch.cuda.get_device_capability(i) == (8, 0) for i in range(device_count))
            if not has_sm80:
                raise NotImplementedError("MentaRay kernel only supports compute capability 8.0.")

    def post_init(self):
        device = self.qweight.device

        if not mentaray_runtime_available(self.compute_dtype):
            raise ModuleNotFoundError(
                "MentaRay torch.ops kernels are not properly installed. Error: "
                + mentaray_runtime_error(self.compute_dtype)
            )

        self.workspace = mentaray_make_workspace_new(device)

        mentaray_qweight = awq_mentaray_repack(
            self.qweight,
            self.in_features,
            self.out_features,
            self.bits,
            dtype=self.compute_dtype,
        )
        replace_parameter(self, "qweight", mentaray_qweight)

        mentaray_scales = mentaray_permute_scales(
            self.scales,
            size_k=self.in_features,
            size_n=self.out_features,
            group_size=self.group_size,
        )
        replace_parameter(self, "scales", mentaray_scales)

        mentaray_zp = awq_to_mentaray_zero_points(
            self.qzeros,
            size_k=self.in_features // self.group_size,
            size_n=self.out_features,
            num_bits=self.bits,
        )
        replace_parameter(self, "qzeros", mentaray_zp)

        self.g_idx = mentaray_make_empty_g_idx(device)
        self.g_idx_sort_indices = mentaray_make_empty_g_idx(device)

        if hasattr(self, "bias") and self.bias is not None:
            self.bias.data = mentaray_permute_bias(self.bias)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        if hasattr(self, "g_idx_sort_indices") and self.g_idx_sort_indices is not None:
            buf.append(self.g_idx_sort_indices)
        if hasattr(self, "g_idx") and self.g_idx is not None:
            buf.append(self.g_idx)
        return buf

    def forward(self, x: torch.Tensor):
        assert hasattr(self, "workspace"), (
            "module.post_init() must be called before module.forward(). "
            "Use mentaray_post_init() on the whole model."
        )

        x = x.contiguous() if self.is_lm_head else x

        if self.scales.dtype != x.dtype:
            self.scales.data = self.scales.data.to(x.dtype)

        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = apply_awq_mentaray_linear(
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=self.weight_type,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            bias=self.bias,
            use_fp32_reduce=True,
        )

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["AwqMentaRayLinear"]
