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

import torch
from typing import Optional, Tuple
from functools import partial
from ...models._const import DEVICE, PLATFORM
from ...adapter.adapter import Adapter, Lora
from .marlin import MarlinQuantLinear, replace_tensor
from ...utils.backend import BACKEND
from ...utils.scalar_type import scalar_types, ScalarType

machete_import_exception = None
try:
    import gptqmodel_machete_kernels
except ImportError as e:
    machete_import_exception = e

TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

MACHETE_PREPACKED_BLOCK_SHAPE = [64, 128]

def check_machete_supports_shape(in_features: int, out_featrues: int) \
    -> Tuple[bool, Optional[str]]:
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return False, "Input features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[0]}"
    if out_featrues % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return False, "Output features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[1]}"
    return True, None

def pack_quantized_values_into_int32(w_q: torch.Tensor,
                                     wtype: ScalarType,
                                     packed_dim: int = 0):
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    assert w_q_perm.shape[-1] % pack_factor == 0
    new_shape_perm[-1] //= pack_factor

    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    for i in range(pack_factor):
        res |= (w_q_perm[..., i::pack_factor] & mask) << wtype.size_bits * i

    return res.permute(inv_perm)


def unpack_quantized_values_into_int32(w_q: torch.Tensor,
                                       wtype: ScalarType,
                                       packed_dim: int = 0):
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    new_shape_perm[-1] *= pack_factor

    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    for i in range(pack_factor):
        res[..., i::pack_factor] = (w_q_perm >> wtype.size_bits * i) & mask

    return res.permute(inv_perm)

def machete_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None) -> torch.Tensor:
    return gptqmodel_machete_kernels.machete_mm(a, b_q, b_type.id, out_type, b_group_scales,
                                   b_group_zeros, b_group_size,
                                   b_channel_scales, a_token_scales, schedule)


def machete_prepack_B(
        b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
        group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
    return gptqmodel_machete_kernels.machete_prepack_B(b_q_weight, a_type, b_type.id,
                                          group_scales_type)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return gptqmodel_machete_kernels.permute_cols(a, perm)

class MacheteQuantLinear(MarlinQuantLinear):
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 128]
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

    # for transformers/optimum tests compat
    QUANT_TYPE = "machete"

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
        **kwargs):

        if machete_import_exception is not None:
            raise ValueError(
                f"Trying to use the machete backend, but could not import the C++/CUDA dependencies with the following error: {machete_import_exception}"
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
            backend=kwargs.pop("backend", BACKEND.MACHETE),
            adapter=adapter,
            **kwargs)

        _, err = check_machete_supports_shape(self.in_features, self.out_features)
        if err is not None:
            raise ValueError(
                f"check_machete_supports_shape failed, {err}"
            )

        self.quant_type = TYPE_MAP.get((bits, sym), None)

    def post_init(self):
        perm = torch.argsort(self.g_idx) \
            .to(torch.int)

        self.act_perm = lambda x: x[:, perm]
        if self.in_features % 8 == 0:
            self.act_perm = partial(permute_cols, perm=perm)

        x_unpacked = unpack_quantized_values_into_int32(self.qweight.data,
                                                        self.quant_type,
                                                        packed_dim=0)

        x_perm = x_unpacked[perm, :]
        self.qweight.data = pack_quantized_values_into_int32(x_perm,
                                                  self.quant_type,
                                                  packed_dim=0)

        machete_qweight = machete_prepack_B(self.qweight.data.t().contiguous().t(),
                                       a_type=self.scales.dtype,
                                       b_type=self.quant_type,
                                       group_scales_type=self.scales.dtype)

        replace_tensor(self, "qweight", machete_qweight)

        marlin_scales = self.scales.data.contiguous()

        replace_tensor(self, "scales", marlin_scales)

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            self.scales = self.scales.to(dtype=x.dtype)

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (self.out_features,)

        x_2d = self.act_perm(x_2d)
        output = machete_mm(a=x_2d,
                                b_q=self.qweight,
                                b_type=self.quant_type,
                                b_group_zeros=None,
                                b_group_scales=self.scales,
                                b_group_size=self.group_size)

        if self.bias is not None:
            output.add_(self.bias)  # In-place add

        return output.reshape(out_shape)
