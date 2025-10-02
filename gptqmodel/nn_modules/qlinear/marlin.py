# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from vllm at https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/gptq_marlin.py

from typing import List, Optional, Tuple

import numpy as np
import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.marlin import (
    _transform_param,
    apply_gptq_marlin_linear,
    gptq_marlin_repack,
    marlin_import_exception,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


class MarlinQuantLinear(BaseQuantLinear):
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
    QUANT_TYPE = "marlin"

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
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
            register_buffers: bool = False,
            adapter: Adapter = None,
            **kwargs):
        if marlin_import_exception is not None:
            raise ValueError(
                f"Trying to use the marlin backend, but could not import the C++/CUDA dependencies with the following error: {marlin_import_exception}"
            )

        # self.original_in_features = in_features
        # self.original_out_features = out_features

        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

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
            register_buffers=False, # do not register buffers in super()
            **kwargs)

        # toggle fp32 mode depending on MARLIN or MARLIN_FP16 backend
        self.fp32 = True if self.backend in [BACKEND.MARLIN, BACKEND.AUTO] else False

        if not self.fp32:
            log.warn.once(
                "Kernel: Marlin FP16 mode is activated with reduced accuracy. Use default Marlin model for improved inference quality.")

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(desc_act,
                                             self.group_size,
                                             is_row_parallel=False):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_size = self.in_features // self.group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_size = self.in_features // self.group_size

        # Quantized weights
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    self.in_features // self.pack_factor,
                    self.out_features,
                    dtype=torch.int32,
                ),
                requires_grad=False
            ),
        )

        # Activation order
        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(data=torch.empty(
                self.in_features,
                dtype=torch.int32,
            ), requires_grad=False),
        )

        # Scales
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(
                    scales_and_zp_size,
                    self.out_features,
                    dtype=torch.float16,
                ),
                requires_grad=False
            ),
        )

        # Quantized zero-points
        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    scales_and_zp_size,
                    self.out_features // self.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
        else:
            self.bias = None

        self.is_lm_head = False
        if kwargs.get("name") is not None and kwargs.get("lm_head_name") is not None:
            self.is_lm_head = kwargs["name"] == kwargs["lm_head_name"]

        if (self.bits, sym) not in self.TYPE_MAP:
            raise ValueError("Unsupported quantization config: "
                             f"bits={self.bits}, sym={sym}")

        self.weight_type = self.TYPE_MAP[(self.bits, sym)]

        # auto-optimize on post init
        # self.optimize()

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
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Marlin kernel is not supported on ROCm.")

            # Directly check capabilities of all currently visible CUDA devices
            has_cuda_v8 = all(
                torch.cuda.get_device_capability(i)[0] >= 8
                for i in range(torch.cuda.device_count())
            )
            if not has_cuda_v8:
                raise NotImplementedError("Marlin kernel only supports compute capability >= 8.0.")

    def post_init(self):
        device = self.qweight.device

        self.is_k_full = marlin_is_k_full(self.desc_act, is_row_parallel=False)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        def transform_w_q(x):
            x.data = gptq_marlin_repack(x.data.contiguous(),
                                        perm=self.g_idx_sort_indices,
                                        size_k=self.in_features,
                                        size_n=self.out_features,
                                        num_bits=self.bits)
            return x

        def transform_w_s(x):
            x.data = marlin_permute_scales(x.data.contiguous(),
                                           size_k=self.in_features,
                                           size_n=self.out_features,
                                           group_size=self.group_size)
            return x

        # Handle sorting for activation reordering if needed.
        if self.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(getattr(self, "g_idx"))
            _transform_param(self, "g_idx", lambda _: g_idx)
            self.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(self, "g_idx", marlin_make_empty_g_idx(device))
            self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        setattr(self, "qzeros", marlin_make_empty_g_idx(device))

        _transform_param(self, "qweight", transform_w_q)
        _transform_param(self, "scales", transform_w_s)

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
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            self.scales = self.scales.to(dtype=x.dtype)

        out = apply_gptq_marlin_linear(
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
            use_atomics=False, # reduces accuracy with slightly faster performance
        )

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


# Precompute permutations for Marlin weight and scale shuffling
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


def unpack_qzeros(qzeros):
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_zeros


def dequantize_qzeros(layer):
    qzeros = layer.qzeros
    unpacked_qzeros = unpack_qzeros(qzeros)
    group_size = layer.group_size
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)

    return unpacked_qzeros


__all__ = ["MarlinQuantLinear"]
