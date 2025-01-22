# Copyright 2025 ModelCloud
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

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from gptqmodel.nn_modules.qlinear import BaseQuantLinear

from ...models._const import DEVICE, PLATFORM
from ...utils.rocm import IS_ROCM


marlin_import_exception = None
try:
    import gptqmodel_marlin_kernels
except ImportError as e:
    marlin_import_exception = e


GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)

def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)

def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)

def marlin_make_workspace(output_size_per_partition: int,
                          device: torch.device) -> torch.Tensor:
    max_workspace_size = (output_size_per_partition //
                          GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(max_workspace_size,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)

def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices

def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)

# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_tensor(layer: torch.nn.Module, name: str,
                   new_t: torch.Tensor) -> None:
    # It is important to use resize_() here since it ensures
    # the same buffer is reused
    getattr(layer, name).resize_(new_t.shape)
    getattr(layer, name).copy_(new_t)
    del new_t

def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def apply_gptq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        num_bits: int,
        output_size_per_partition: int,
        input_size_per_partition: int,
        is_k_full: bool,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition, )

    output = gptqmodel_marlin_kernels.gptq_marlin_gemm(reshaped_x,
                                  weight,
                                  weight_scale,
                                  weight_zp,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  num_bits,
                                  reshaped_x.shape[0],
                                  output_size_per_partition,
                                  input_size_per_partition,
                                  is_k_full,
                                  False)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


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

    # for transformers/optimum tests compat
    QUANT_TYPE = "marlin"

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures: int, outfeatures: int,
                 bias: bool, **kwargs):
        if marlin_import_exception is not None:
            raise ValueError(
                f"Trying to use the marlin backend, but could not import the C++/CUDA dependencies with the following error: {marlin_import_exception}"
            )

        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures, outfeatures=outfeatures, **kwargs)

        self.original_infeatures = infeatures
        self.original_outfeatures = outfeatures

        self.pack_factor = 32 // bits  # packed into int32

        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # Normalize group_size
        if group_size != -1:
            group_size = group_size
        else:
            group_size = infeatures

        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(desc_act,
                                             group_size,
                                             is_row_parallel=False):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = infeatures // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = infeatures // group_size

        # Quantized weights
        qweight = Parameter(
            torch.empty(
                infeatures // self.pack_factor,
                outfeatures,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.pack_factor,
            },
        )

        # Activation order
        g_idx = Parameter(
            torch.empty(
                infeatures,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(
            g_idx,
            {
                "input_dim": 0,
                "ignore_warning": True
            },
        )

        # Scales
        scales = Parameter(
            torch.empty(
                scales_and_zp_size,
                outfeatures,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
            },
        )

        # Quantized zero-points
        qzeros = Parameter(
            torch.empty(
                scales_and_zp_size,
                outfeatures // self.pack_factor,
                dtype=torch.int32,
                # device="meta",
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.pack_factor,
            },
        )

        self.register_parameter("qweight", qweight)
        self.register_parameter("g_idx", g_idx)
        self.register_parameter("scales", scales)
        self.register_parameter("qzeros", qzeros)
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.is_k_full = marlin_is_k_full(desc_act, is_row_parallel=False)

        if bias:
            self.register_buffer("bias", torch.zeros((self.outfeatures), dtype=torch.half))
        else:
            self.bias = None

        self.is_lm_head = False
        if kwargs.get("name") is not None and kwargs.get("lm_head_name") is not None:
            self.is_lm_head = kwargs["name"] == kwargs["lm_head_name"]

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if marlin_import_exception is not None:
            return False, marlin_import_exception
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
                has_cuda_v8 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(len(CUDA_VISIBLE_DEVICES.split(","))))
            if not has_cuda_v8:
                raise NotImplementedError("Marlin kernel only supports compute capability >= 8.0.")

    def post_init(self):
        device = self.qweight.device
        # Allocate marlin workspace
        self.workspace = marlin_make_workspace(
            self.outfeatures, device)

        # Handle sorting for activation reordering if needed.
        if self.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(self.g_idx)
            self.g_idx_sort_indices = g_idx_sort_indices
            replace_tensor(self, "g_idx", g_idx)
        else:
            self.g_idx = marlin_make_empty_g_idx(device)
            self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        self.zp = marlin_make_empty_g_idx(device)

        # Repack weights from autogptq format to marlin format.
        marlin_qweight = gptqmodel_marlin_kernels.gptq_marlin_repack(
            self.qweight,
            self.g_idx_sort_indices,
            self.infeatures,
            self.outfeatures,
            self.bits)
        replace_tensor(self, "qweight", marlin_qweight)

        # Permute scales from autogptq format to marlin format.
        marlin_scales = marlin_permute_scales(
            self.scales,
            size_k=self.infeatures,
            size_n=self.outfeatures,
            group_size=self.group_size)
        replace_tensor(self, "scales", marlin_scales)

    def forward(self, A: torch.Tensor):
        if A.dtype != torch.float16:
            A = A.half()

        return apply_gptq_marlin_linear(
            input=A.contiguous() if self.is_lm_head else A,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.zp,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            num_bits=self.bits,
            output_size_per_partition=self.outfeatures,
            input_size_per_partition=self.infeatures,
            is_k_full=self.is_k_full,
            bias=self.bias)

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
