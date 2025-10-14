# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from vllm at https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/gptq_marlin.py

import os
from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.marlin import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    replace_parameter,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


marlin_import_exception = None
try:
    import gptqmodel_marlin_kernels
except ImportError as e:
    marlin_import_exception = str(e)

log = setup_logger()


class AwqMarlinQuantLinear(AWQuantLinear):
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
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

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_marlin"

    # num_bits -> type
    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(
            self, bits: int,
            group_size: int,
            desc_act: bool,
            sym: bool,
            in_features: int,
            out_features: int,
            bias: bool = False,
            pack_dtype: torch.dtype = torch.int32,
            adapter: Adapter = None,
            register_buffers=False,
            **kwargs):
        if marlin_import_exception is not None:
            raise ValueError(
                f"Trying to use the marlin backend, but could not import the C++/CUDA dependencies with the following error: {marlin_import_exception}"
            )

        self.max_par = 8  # partitioning for large inputs

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.MARLIN),
            adapter=adapter,
            register_buffers=False,
            **kwargs)

        if register_buffers:
            self.register_parameter(
                "qweight",
                torch.nn.Parameter(
                    torch.empty(
                        self.in_features,
                        self.out_features // self.pack_factor,
                        dtype=torch.int32,
                    ),
                    requires_grad=False
                ),
            )
            self.register_parameter(
                "qzeros",
                torch.nn.Parameter(
                    torch.empty(
                        self.in_features // self.group_size,
                        self.out_features // self.pack_factor,
                        dtype=torch.int32,
                    ),
                    requires_grad=False
                )
            )

            self.register_parameter(
                "scales",
                torch.nn.Parameter(
                    torch.empty(
                        self.in_features // self.group_size,
                        self.out_features,
                        dtype=torch.float16,
                    ),
                    requires_grad=False
                )
            )

            if bias:
                self.register_buffer(
                    "bias",
                    torch.zeros(
                        (out_features),
                        dtype=torch.float16,
                    ),
                )
            else:
                self.bias = None

        self.is_lm_head = False
        if kwargs.get("name") is not None and kwargs.get("lm_head_name") is not None:
            self.is_lm_head = kwargs["name"] == kwargs["lm_head_name"]

        if self.bits not in self.TYPE_MAP:
            raise ValueError(f"Unsupported num_bits = {self.bits}. "
                             f"Supported num_bits = {self.TYPE_MAP.keys()}")

        self.weight_type = self.TYPE_MAP[self.bits]

    # def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
    #     if self.optimized:
    #         return
    #
    #     # compile dequantize
    #     self.forward = torch_compile(self.forward, backend=backend, mode=mode, fullgraph=fullgraph)
    #
    #     super().optimize()

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if marlin_import_exception is not None:
            return False, ImportError(marlin_import_exception)
        return cls._validate(**args)

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Marlin kernel is not supported on ROCm.")

            if CUDA_VISIBLE_DEVICES is None:
                has_cuda_v8 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))
            else:
                has_cuda_v8 = all(
                    torch.cuda.get_device_capability(i)[0] >= 8 for i in range(len(CUDA_VISIBLE_DEVICES.split(","))))
            if not has_cuda_v8:
                raise NotImplementedError("Marlin kernel only supports compute capability >= 8.0.")

    def post_init(self):
        device = self.qweight.device

        # Allocate marlin workspace
        self.workspace = marlin_make_workspace_new(device)

        # Repack weights from AWQ format to marlin format.
        marlin_qweight = gptqmodel_marlin_kernels.awq_marlin_repack(
            self.qweight,
            self.in_features,
            self.out_features,
            self.bits)
        replace_parameter(self, "qweight", marlin_qweight)

        # Permute scales from AWQ format to marlin format.
        marlin_scales = marlin_permute_scales(
            self.scales,
            size_k=self.in_features,
            size_n=self.out_features,
            group_size=self.group_size)
        replace_parameter(self, "scales", marlin_scales)

        # Permute zero-points from AWQ format to marlin format.
        marlin_zp = awq_to_marlin_zero_points(
            self.qzeros,
            size_k=self.in_features // self.group_size,
            size_n=self.out_features,
            num_bits=self.bits)
        replace_parameter(self, "qzeros", marlin_zp)

        # Not-used
        self.g_idx = marlin_make_empty_g_idx(device)
        self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if hasattr(self, "bias") and self.bias is not None:
            self.bias.data = marlin_permute_bias(self.bias)

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
            "Use marlin_post_init() on the whole model."
        )

        out = apply_awq_marlin_linear(
            input=x.contiguous() if self.is_lm_head else x,
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
        )

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["AwqMarlinQuantLinear"]
