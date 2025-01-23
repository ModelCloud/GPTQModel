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

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers

from gptqmodel.models._const import DEVICE, PLATFORM
from gptqmodel.nn_modules.qlinear import BaseQuantLinear

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

class IPEXQuantLinear(BaseQuantLinear):
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

    # for transformers/optimum tests compat
    QUANT_TYPE = "ipex"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        infeatures: int,
        outfeatures: int,
        bias: bool,
        kernel_switch_threshold=128,
        training=False,
        weight_dtype=None,
        **kwargs,
    ):
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures, outfeatures=outfeatures, **kwargs)

        if weight_dtype is None:
            weight_dtype = torch.float16 if HAS_XPU else torch.bfloat16

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size
        self.maxq = 2**self.bits - 1
        self.weight_dtype = weight_dtype
        self.init_ipex = False

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None

        self.kernel_switch_threshold = kernel_switch_threshold

        self.training = training

        # for training forward
        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)

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
                                                                    self.infeatures, self.outfeatures, None, self.bias,
                                                                    self.group_size, self.g_idx, quant_method=QuantMethod.GPTQ_GEMM, dtype=QuantDtype.INT4)

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = torch.round((W + scale_zeros[self.g_idx].T) / scales[self.g_idx].T).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        for row in range(qweight.shape[0]):
            i = row * (32 // self.bits)
            for j in range(32 // self.bits):
                qweight[row] |= intweight[i + j] << (self.bits * j)

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        for col in range(qzeros.shape[1]):
            i = col * (32 // self.bits)
            for j in range(32 // self.bits):
                qzeros[:, col] |= zeros[:, i + j] << (self.bits * j)

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x: torch.Tensor):
        if not self.init_ipex:
            self.init_ipex_linear(x)
            self.init_ipex = True

        if hasattr(self, "ipex_linear"):
            with torch.no_grad():
                outputs = self.ipex_linear(x)
            return outputs

        if self.wf.device != x.device:
            self.wf = self.wf.to(x.device)
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim : (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim : (i + 1) * num_dim]
                weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
            weights = torch.cat(weights, dim=1)
        out = torch.matmul(x, weights)
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out

        return out


@torch.no_grad()
def unpack_to_8bit_signed(qweight, qzeros, bits, g_idx=None):
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    zeros = None
    if not torch.all(torch.eq(qzeros, 2004318071 if bits == 4 else 0b01111111011111110111111101111111)):
        zp_shape = list(qzeros.shape)
        zp_shape[1] = zp_shape[1] * (32 // bits)

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
        if bits == 8:
            zeros = zeros.to(torch.uint8)
        zeros = zeros + 1
        try:
            zeros = zeros.reshape(zp_shape)
        except Exception:
            # zeros and scales have different iteam numbers.
            # remove 1 (due to 0 + 1 in line 252)
            zeros = zeros[zeros != 1]
            zeros = zeros.reshape(zp_shape)

    try:
        r = torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1)
    except BaseException as e:
        print(e)
    weight = torch.bitwise_right_shift(
        r, wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    weight.bitwise_and_((2**bits) - 1)
    weight = weight.view(-1, weight.shape[-1])

    if g_idx is not None:
        group_size = weight.shape[0] // qzeros.shape[0]
        weight2 = weight.clone()
        group_dict = {}
        for i in range(len(g_idx)):
            group_idx = g_idx[i].item()
            if group_idx not in group_dict:
                target_idx = group_idx * group_size
                group_dict[group_idx] = 0
            else:
                group_dict[group_idx] = group_dict[group_idx] + 1
                target_idx = group_idx * group_size + group_dict[group_idx]
            weight2[target_idx] = weight[i]
        weight = weight2

    return weight, zeros


# Copied from marlin.py
@torch.no_grad()
def dequantize_weight(qweight, qzeros, scales, bits):
    unpacked_qweight, unpacked_qzeros = unpack_to_8bit_signed(qweight, qzeros, bits)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    if unpacked_qzeros is not None:
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    else:
        unpacked_qzeros = torch.full_like(scales, 8 if bits == 4 else 128, dtype=torch.int32)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight, unpacked_qzeros


__all__ = ["IPEXQuantLinear", "dequantize_weight"]
