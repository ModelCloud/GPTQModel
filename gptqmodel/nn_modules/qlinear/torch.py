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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gptqmodel.adapter.adapter import Adapter, Lora
from gptqmodel.nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from gptqmodel.utils.logger import setup_logger

from ...models._const import DEVICE, PLATFORM


logger = setup_logger()

class TorchQuantLinear(PackableQuantLinear):
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
    SUPORTS_ADAPTERS = [Lora]
    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
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
            register_buffers=True,
            **kwargs)

        if self.group_size != self.in_features:
            self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        else:
            self.padded_infeatures = self.in_features

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

        super().post_init()

        self.wf = self.wf.to(device=self.qweight.device)


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

        out = torch.matmul(x, weights).reshape(out_shape)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        if self.bias is not None:
            out.add_(self.bias)

        return out.to(x_dtype)

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

    def dequantize_weight(self, num_itr=1):
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

        return weights

def dequantize_model(model: nn.Module):
    for name, module in model.model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchQuantLinear):
            raise ValueError(
                "Only models loaded using TorchQuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.TORCH."
            )

        if isinstance(module, TorchQuantLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = module.bias

            # Replace the module in the model
            parent = model.model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["TorchQuantLinear", "dequantize_model"]
