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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.logger import setup_logger

from ...models._const import DEVICE, PLATFORM

logger = setup_logger()

class TorchQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        infeatures: int,
        outfeatures: int,
        bias: bool,
        weight_dtype=torch.float16,
        **kwargs,
    ):
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures, outfeatures=outfeatures, **kwargs)

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.padded_infeatures = infeatures + (-infeatures % group_size)
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

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

        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)

    def post_init(self):
        if self.padded_infeatures != self.infeatures:
            self.qweight.resize_(self.padded_infeatures // 32 * self.bits, self.outfeatures)
            self.qzeros.resize_(
                math.ceil(self.padded_infeatures / self.group_size),
                self.outfeatures // 32 * self.bits
            )
            self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.outfeatures), )
            self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
                                      device=self.g_idx.device)

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
        if self.bits in [2, 4, 8]:
            bits_div = 32 // self.bits
            for row in range(qweight.shape[0]):
                for j in range(bits_div):
                    qweight[row] |= intweight[row * bits_div + j] << (self.bits * j)
        elif self.bits == 3:
            for row in range(qweight.shape[0]):
                row_offset = row * 10  # Cache row * 10
                row_offset_plus_10 = row_offset + 10  # Cache row * 10 + 10
                for j in range(10):
                    qweight[row] |= intweight[row_offset + j] << (3 * j)
                qweight[row] |= intweight[row_offset_plus_10] << 30
                row += 1
                qweight[row] |= (intweight[row_offset_plus_10] >> 2) & 1
                for j in range(10):
                    qweight[row] |= intweight[row_offset + j] << (3 * j + 1)
                qweight[row] |= intweight[row_offset_plus_10] << 31
                row += 1
                qweight[row] |= (intweight[row_offset_plus_10] >> 1) & 0x3
                for j in range(10):
                    qweight[row] |= intweight[row_offset + j] << (3 * j + 2)

        self.qweight = torch.from_numpy(qweight.astype(np.int32))

        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        if self.bits in [2, 4, 8]:
            bits_div = 32 // self.bits
            for col in range(qzeros.shape[1]):
                for j in range(bits_div):
                    qzeros[:, col] |= zeros[:, col * bits_div + j] << (self.bits * j)
        elif self.bits == 3:
            for col in range(qzeros.shape[1]):
                col_offset = col * 10  # Cache col * 10
                col_offset_plus_10 = col_offset + 10  # Cache col * 10 + 10
                for j in range(10):
                    qzeros[:, col] |= zeros[:, col_offset + j] << (3 * j)
                qzeros[:, col] |= zeros[:, col_offset_plus_10] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, col_offset_plus_10] >> 2) & 1
                for j in range(10):
                    qzeros[:, col] |= zeros[:, col_offset + j] << (3 * j + 1)
                qzeros[:, col] |= zeros[:, col_offset_plus_10] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, col_offset_plus_10] >> 1) & 0x3
                for j in range(10):
                    qzeros[:, col] |= zeros[:, col_offset + j] << (3 * j + 2)

        self.qzeros = torch.from_numpy(qzeros.astype(np.int32))

    def forward(self, x: torch.Tensor):
        if x.size(-1) != self.padded_infeatures:
            x = F.pad(x, (0, self.padded_infeatures - self.infeatures))

        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype
        out = self._forward(x, x_dtype, out_shape)
        return out

    def _forward(self, x, x_dtype, out_shape):
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        weights = self.dequantize_weight(num_itr=num_itr)

        out = torch.matmul(x, weights)
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

    @torch.no_grad()
    def dequantize_weight(self, num_itr=1):
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                self.wf.unsqueeze(0),
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            zeros = torch.bitwise_and(zeros, (2 ** self.bits) - 1).reshape(self.scales.shape)

            weight = torch.bitwise_and(
                torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                    self.wf.unsqueeze(-1),
                ).to(torch.int16 if self.bits == 8 else torch.int8),
                (2 ** self.bits) - 1
            )
        elif self.bits == 3:
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(
                -1, -1, -1, 12
            )
            zeros = zeros >> self.wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = torch.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            ).reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim: (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim: (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim: (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim: (i + 1) * num_dim].long()
                weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
            weights = torch.cat(weights, dim=1)

        return weights

__all__ = ["TorchQuantLinear"]