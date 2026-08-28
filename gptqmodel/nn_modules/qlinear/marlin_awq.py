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
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.marlin import (
    apply_awq_marlin_linear,
    apply_awq_marlin_linear_padded,
    awq_marlin_repack,
    awq_to_marlin_zero_points,
    marlin_import_exception,
    marlin_is_tile_aligned,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_pad_awq_qweight,
    marlin_pad_awq_qzeros,
    marlin_pad_dim,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_runtime_available,
    marlin_runtime_error,
    replace_parameter,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


class AwqMarlinLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_MARLIN]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 90, FORMAT.MARLIN: 90}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

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
        self.max_par = 8  # partitioning for large inputs
        self.compute_dtype = kwargs.get("dtype") or torch.float16

        selected_backend = kwargs.pop("backend", BACKEND.AWQ_MARLIN)
        # Keep runtime padding opt-in until tail-shape performance is measured.
        if selected_backend in (BACKEND.AUTO, BACKEND.AUTO_TRAINABLE) and not marlin_is_tile_aligned(
            out_features, in_features
        ):
            raise NotImplementedError(
                "Automatic AWQ Marlin selection keeps tile-misaligned shapes "
                "on the next compatible backend; request AWQ_MARLIN explicitly "
                "to enable runtime tile padding."
            )

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=selected_backend,
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
                        dtype=self.compute_dtype,
                    ),
                    requires_grad=False
                )
            )

            if bias:
                self.register_buffer(
                    "bias",
                    torch.zeros(
                        (out_features),
                        dtype=self.compute_dtype,
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
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if marlin_import_exception is not None:
            return False, ImportError(marlin_import_exception)
        return True, None

    @classmethod
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err

        bits = args.get("bits", 4)
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if out_features is None:
            return True, None

        # AWQ packs N into int32 words; padding cannot repair a partial word.
        pack_factor = 32 // bits
        if out_features % pack_factor != 0:
            return False, NotImplementedError(
                "AWQ Marlin out_features must be divisible by "
                f"pack_factor={pack_factor}; got N={out_features}."
            )

        # Keep uint8 tails on fallback until zero-point thread configs are validated.
        if bits != 4 and in_features is not None and not marlin_is_tile_aligned(
            out_features, in_features
        ):
            return False, NotImplementedError(
                "AWQ Marlin runtime tile padding is enabled only for 4-bit "
                f"weights; got bits={bits}."
            )

        return True, None

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

        if not marlin_runtime_available(self.compute_dtype):
            raise ModuleNotFoundError(
                "Marlin torch.ops kernels are not properly installed. Error: "
                + marlin_runtime_error(self.compute_dtype)
            )

        # Allocate marlin workspace
        self.workspace = marlin_make_workspace_new(device)

        # group_size=K and group_size=-1 both mean one channelwise group.
        marlin_group_size = (
            -1
            if self.requested_group_size == self.in_features
            else self.requested_group_size
        )
        padded_n, padded_k = marlin_padded_nk(
            self.out_features,
            self.in_features,
            marlin_group_size,
        )
        self._marlin_tile_padding = (
            None
            if (padded_n, padded_k) == (self.out_features, self.in_features)
            else (padded_n, padded_k)
        )

        # Repack weights from AWQ format to marlin format.
        padded_qweight = marlin_pad_awq_qweight(
            self.qweight.contiguous(),
            self.out_features,
            self.in_features,
            padded_n,
            padded_k,
            self.bits,
        )
        marlin_qweight = awq_marlin_repack(
            padded_qweight,
            padded_k,
            padded_n,
            self.bits,
            dtype=self.compute_dtype)
        replace_parameter(self, "qweight", marlin_qweight)

        # Permute scales from AWQ format to marlin format.
        padded_scales = marlin_pad_scales(
            self.scales.contiguous(),
            self.out_features,
            self.in_features,
            padded_n,
            padded_k,
            marlin_group_size,
        )
        marlin_scales = marlin_permute_scales(
            padded_scales,
            size_k=padded_k,
            size_n=padded_n,
            group_size=marlin_group_size)
        replace_parameter(self, "scales", marlin_scales)

        # Permute zero-points from AWQ format to marlin format.
        padded_qzeros = marlin_pad_awq_qzeros(
            self.qzeros.contiguous(),
            self.out_features,
            self.in_features,
            padded_n,
            padded_k,
            marlin_group_size,
            self.bits,
        )
        padded_groups = 1 if marlin_group_size == -1 else padded_k // marlin_group_size
        marlin_zp = awq_to_marlin_zero_points(
            padded_qzeros,
            size_k=padded_groups,
            size_n=padded_n,
            num_bits=self.bits)
        replace_parameter(self, "qzeros", marlin_zp)

        # Not-used
        self.g_idx = marlin_make_empty_g_idx(device)
        self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if hasattr(self, "bias") and self.bias is not None:
            self.bias.data = marlin_permute_bias(
                marlin_pad_dim(self.bias, self.out_features, padded_n)
            )

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

        x = x.contiguous() if self.is_lm_head else x

        if self.scales.dtype != x.dtype:
            self.scales.data = self.scales.data.to(x.dtype)

        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # Aligned layers retain the original decode-sensitive call path.
        if self._marlin_tile_padding is None:
            out = apply_awq_marlin_linear(
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
            )
        else:
            out = apply_awq_marlin_linear_padded(
                tile_padding=self._marlin_tile_padding,
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
            )

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["AwqMarlinLinear"]
