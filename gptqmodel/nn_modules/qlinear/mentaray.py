# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.logger import setup_logger
from ...utils.mentaray import (
    _mentaray_capability_supported,
    _transform_param,
    apply_gptq_mentaray_linear,
    gptq_mentaray_repack,
    mentaray_import_exception,
    mentaray_is_k_full,
    mentaray_make_empty_g_idx,
    mentaray_make_workspace_new,
    mentaray_permute_bias,
    mentaray_permute_scales,
    mentaray_repeat_scales_on_all_ranks,
    mentaray_runtime_available,
    mentaray_runtime_error,
    mentaray_sort_g_idx,
    replace_parameter,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


class MentaRayLinear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_MENTARAY]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 91, FORMAT.MARLIN: 91}
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
    QUANT_TYPE = "mentaray"

    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
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
        register_buffers: bool = False,
        adapter: Adapter = None,
        **kwargs,
    ):
        if mentaray_import_exception is not None:
            raise ValueError(
                "Trying to use the MentaRay backend, but the runtime requirements were not met: "
                f"{mentaray_import_exception}"
            )

        if desc_act and group_size == -1:
            desc_act = False

        self.compute_dtype = kwargs.get("dtype") or torch.float16
        self.fp32 = env_flag("GPTQMODEL_MENTARAY_USE_FP32", default=env_flag("GPTQMODEL_MARLIN_USE_FP32", default=True))

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_MENTARAY),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        if not self.fp32:
            log.warn.once(
                "Kernel: GPTQMODEL_MENTARAY_USE_FP32 is disabled. MentaRay will use reduced-precision reduction."
            )

        if mentaray_repeat_scales_on_all_ranks(desc_act, self.group_size, is_row_parallel=False):
            scales_and_zp_size = self.in_features // self.group_size
        else:
            scales_and_zp_size = self.in_features // self.group_size

        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    self.in_features // self.pack_factor,
                    self.out_features,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                data=torch.empty(self.in_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(scales_and_zp_size, self.out_features, dtype=torch.float16),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(scales_and_zp_size, self.out_features // self.pack_factor, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
        else:
            self.bias = None

        self.is_lm_head = False
        if kwargs.get("name") is not None and kwargs.get("lm_head_name") is not None:
            self.is_lm_head = kwargs["name"] == kwargs["lm_head_name"]

        if (self.bits, sym) not in self.TYPE_MAP:
            raise ValueError(f"Unsupported quantization config: bits={self.bits}, sym={sym}")

        self.weight_type = self.TYPE_MAP[(self.bits, sym)]

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if mentaray_import_exception is not None:
            return False, ImportError(mentaray_import_exception)
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("MentaRay kernel is not supported on ROCm.")

            has_supported_cuda = all(
                _mentaray_capability_supported(*torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count())
            )
            if not has_supported_cuda:
                raise NotImplementedError("MentaRay kernel only supports compute capability 8.0.")

    def post_init(self):
        device = self.qweight.device

        if not mentaray_runtime_available(self.compute_dtype):
            raise ModuleNotFoundError(
                "MentaRay torch.ops kernels are not properly installed. Error: "
                + mentaray_runtime_error(self.compute_dtype)
            )

        self.is_k_full = mentaray_is_k_full(self.desc_act, is_row_parallel=False)
        self.workspace = mentaray_make_workspace_new(device)

        def transform_w_q(x):
            x.data = gptq_mentaray_repack(
                x.data.contiguous(),
                perm=self.g_idx_sort_indices,
                size_k=self.in_features,
                size_n=self.out_features,
                num_bits=self.bits,
                dtype=self.compute_dtype,
            )
            return x

        def transform_w_s(x):
            x.data = mentaray_permute_scales(
                x.data.contiguous(),
                size_k=self.in_features,
                size_n=self.out_features,
                group_size=self.group_size,
            )
            return x

        if self.desc_act:
            g_idx, g_idx_sort_indices = mentaray_sort_g_idx(getattr(self, "g_idx"))
            _transform_param(self, "g_idx", lambda _: g_idx)
            self.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(self, "g_idx", mentaray_make_empty_g_idx(device))
            self.g_idx_sort_indices = mentaray_make_empty_g_idx(device)

        setattr(self, "qzeros", mentaray_make_empty_g_idx(device))

        _transform_param(self, "qweight", transform_w_q)
        _transform_param(self, "scales", transform_w_s)

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
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        if x.dtype != self.scales.dtype:
            replace_parameter(self, "scales", self.scales.to(dtype=x.dtype))

        out = apply_gptq_mentaray_linear(
            input=x.contiguous() if self.is_lm_head else x,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=self.weight_type,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            is_k_full=self.is_k_full,
            bias=self.bias,
            use_fp32_reduce=self.fp32,
            use_atomics=False,
        )

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["MentaRayLinear"]
