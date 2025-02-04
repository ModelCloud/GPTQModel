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

import torch
import torch.nn as nn
import torch.nn.functional as F
from gptqmodel.nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from gptqmodel.utils.logger import setup_logger

from ...models._const import DEVICE, PLATFORM
from ...quantization.config import EXTENSION

logger = setup_logger()

class EoRATorchQuantLinear(PackableQuantLinear):
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
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_EXTENSIONS = [EXTENSION.EORA] # <-- EoRA declration

    # for transformers/optimum tests compat
    QUANT_TYPE = "eora_torch"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool,
        pack_dtype: torch.dtype,
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
            register_buffers=True,
            **kwargs)

        # EoRA need to preallocate buffers for Lora_A and B weights so HF can load
        # self.register_buffer(
        #     "lora_A",
        #     t.zeros((0,0), dtype=self.pack_dtype), # <-- EoRA lora_A shape needs to be calculated using pass in_features/out_features or other eora math
        # )

        # EoRA need to preallocate buffers for Lora_A and B weights so HF can load
        # self.register_buffer(
        #     "lora_B",
        #     t.zeros((0, 0), dtype=self.pack_dtype), # <-- EoRA lora_A shape needs to be calculated using pass in_features/out_features or other eora math
        # )

        if self.group_size != self.in_features:
            self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        else:
            self.padded_infeatures = self.padded_infeatures

        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=torch.int32).unsqueeze(0)
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
        if self.padded_infeatures != self.in_features:
            self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
            self.qzeros.resize_(
                math.ceil(self.padded_infeatures / self.group_size),
                self.out_features // self.pack_dtype_bits * self.bits
            )
            self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
            self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
                                      device=self.g_idx.device)



    def forward(self, x: torch.Tensor):
        if x.size(-1) != self.padded_infeatures:
            x = F.pad(x, (0, self.padded_infeatures - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        out = self._forward(x, x.dtype, out_shape)
        return out

    def _forward(self, x, x_dtype, out_shape):
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        weights = self.dequantize_weight(num_itr=num_itr)

        # EoRA needs to apply A/B projection on to dequantized fp16 `weights`
        # here..... <-- EoRA A/B math with W (weights)

        out = torch.matmul(x, weights).reshape(out_shape).to(x_dtype)
        if self.bias is not None:
            out.add_(self.bias)
        return out

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

    def dequantize_weight(self, num_itr=1):
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        if self.bits in [2, 4, 8]:
            dtype = torch.int16 if self.bits == 8 else torch.int8
            zeros = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
                self.wf.unsqueeze(0),
            ).to(dtype)
            zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

            weight = torch.bitwise_and(
                torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                    self.wf.unsqueeze(-1),
                ).to(dtype),
                self.maxq
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

        return weight

__all__ = ["EoRATorchQuantLinear"]
