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

from typing import Optional, Tuple

import torch
from gptqmodel.adapter.adapter import Adapter, Lora
from gptqmodel.models._const import DEVICE, PLATFORM
from .torch import TorchQuantLinear

from ...utils.logger import setup_logger
from ...utils.torch import HAS_XPU

logger = setup_logger()

BITS_DTYPE_MAPPING = {
    4: "int4_clip",
}

HAS_IPEX = False
IPEX_ERROR_LOG = None
try:
    from intel_extension_for_pytorch.llm.quantization import IPEXWeightOnlyQuantizedLinear, QuantDtype, QuantMethod

    HAS_IPEX = True
except BaseException:
    HAS_IPEX = False
    IPEX_ERROR_LOG = Exception

def ipex_dtype() -> torch.dtype:
    if not HAS_IPEX:
        raise ImportError("intel_extension_for_pytorch not installed. "
                          "Please install via `pip install intel_extension_for_pytorch`")

    return torch.float16 if HAS_XPU else torch.bfloat16


def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)

def convert_idx(self, g_idx, k):
    ret_idx = torch.zeros(k, dtype=int).to(g_idx.device)
    groups = k // self.blocksize
    remainder = k % self.blocksize
    g_idx_2 = g_idx * self.blocksize
    if remainder > 0:
        g_idx_2[g_idx == groups] += torch.arange(remainder).to(g_idx.device)
    arange_tensor = torch.arange(self.blocksize).to(g_idx.device)
    for i in range(groups):
        g_idx_2[g_idx == i] += arange_tensor
    ret_idx[g_idx_2] = torch.arange(k).to(g_idx.device)
    return ret_idx.to(torch.int32)

if HAS_IPEX:
    try:
        # monkey patch GPTQShuffle.convert_idx to use fixed convert_idx, fix the slow ipex generate issue
        from intel_extension_for_pytorch.nn.utils._quantize_convert import GPTQShuffle

        GPTQShuffle.convert_idx = convert_idx
    except ImportError:
        # if import GPTQShuffle failed, do nothing
        pass

class IPEXQuantLinear(TorchQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPORTS_ADAPTERS = [Lora]
    # for transformers/optimum tests compat
    QUANT_TYPE = "ipex"

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
        training=False,
        register_buffers: bool = True,
        **kwargs,
    ):
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
            register_buffers=register_buffers,
            **kwargs)

        # FIX ME IPEX CPU has no float16 support
        self.weight_dtype = torch.float16 if HAS_XPU else torch.bfloat16
        self.training = training
        self.ipex_linear = None  # None means not init, False means no ipex, else is good

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if not HAS_IPEX:
            return False, IPEX_ERROR_LOG
        return cls._validate(**args)

    def post_init(self):
        pass

    def init_ipex_linear(self, x: torch.Tensor):
        if not self.training and HAS_IPEX and not x.requires_grad:
            self.ipex_linear = IPEXWeightOnlyQuantizedLinear.from_weight(self.qweight, self.scales, self.qzeros,
                                                                     self.in_features, self.out_features, None, self.bias,
                                                                         self.group_size, self.g_idx, quant_method=QuantMethod.GPTQ_GEMM, dtype=QuantDtype.INT4)
            assert self.ipex_linear is not None
        else:
            self.ipex_linear = False

    def forward(self, x: torch.Tensor):
        if self.ipex_linear is None: # None is special value meaning ipex_linear init is not called yet
            self.init_ipex_linear(x)

        if self.ipex_linear:
            with torch.no_grad():
                outputs = self.ipex_linear(x)
            return outputs

        return super().forward(x)


# @torch.no_grad()
# def unpack_to_8bit_signed(qweight, qzeros, bits, g_idx=None):
#     wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
#     zeros = None
#     if not torch.all(torch.eq(qzeros, 2004318071 if bits == 4 else 0b01111111011111110111111101111111)):
#         zp_shape = list(qzeros.shape)
#         zp_shape[1] = zp_shape[1] * (32 // bits)
#
#         zeros = torch.bitwise_right_shift(
#             torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
#         ).to(torch.int16 if bits == 8 else torch.int8)
#         torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
#         if bits == 8:
#             zeros = zeros.to(torch.uint8)
#         zeros = zeros + 1
#         try:
#             zeros = zeros.reshape(zp_shape)
#         except Exception:
#             # zeros and scales have different iteam numbers.
#             # remove 1 (due to 0 + 1 in line 252)
#             zeros = zeros[zeros != 1]
#             zeros = zeros.reshape(zp_shape)
#
#     try:
#         r = torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1)
#     except BaseException as e:
#         print(e)
#     weight = torch.bitwise_right_shift(
#         r, wf.unsqueeze(-1)
#     ).to(torch.int16 if bits == 8 else torch.int8)
#     weight.bitwise_and_((2**bits) - 1)
#     weight = weight.view(-1, weight.shape[-1])
#
#     if g_idx is not None:
#         group_size = weight.shape[0] // qzeros.shape[0]
#         weight2 = weight.clone()
#         group_dict = {}
#         for i in range(len(g_idx)):
#             group_idx = g_idx[i].item()
#             if group_idx not in group_dict:
#                 target_idx = group_idx * group_size
#                 group_dict[group_idx] = 0
#             else:
#                 group_dict[group_idx] = group_dict[group_idx] + 1
#                 target_idx = group_idx * group_size + group_dict[group_idx]
#             weight2[target_idx] = weight[i]
#         weight = weight2
#
#     return weight, zeros
#
#
# # Copied from marlin.py
# @torch.no_grad()
# def dequantize_weight(qweight, qzeros, scales, bits):
#     unpacked_qweight, unpacked_qzeros = unpack_to_8bit_signed(qweight, qzeros, bits)
#     group_size = unpacked_qweight.shape[0] // scales.shape[0]
#     scales = scales.repeat_interleave(group_size, dim=0)
#     if unpacked_qzeros is not None:
#         unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
#     else:
#         unpacked_qzeros = torch.full_like(scales, 8 if bits == 4 else 128, dtype=torch.int32)
#     unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales
#
#     return unpacked_qweight, unpacked_qzeros


__all__ = ["IPEXQuantLinear"]
