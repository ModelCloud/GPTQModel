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

# Adapted from turboderp exllama: https://github.com/turboderp/exllama

import math
from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from gptqmodel.nn_modules.qlinear import BaseQuantLinear

from ...models._const import DEVICE, PLATFORM

exllama_import_exception = None
try:
    from gptqmodel_exllama_kernels import make_q4, q4_matmul
except ImportError as e:
    exllama_import_exception = e

logger = getLogger(__name__)


# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
NON_TENSOR = torch.empty((1, 1), device="meta")


def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(qweight, qzeros, scales, g_idx if g_idx is not None else NON_TENSOR, device)


def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

    q4_matmul(x, q4, output)

    return output.view(outshape)


class ExllamaQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]

    # for transformers/optimum tests compat
    QUANT_TYPE = "exllama"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures: int, outfeatures: int, bias: bool,  **kwargs,):
        if exllama_import_exception is not None:
            raise ValueError(
                f"Trying to use the exllama backend, but could not import the C++/CUDA dependencies with the following error: {exllama_import_exception}"
            )
        self.group_size = group_size if group_size != -1 else infeatures
        # auto pad
        self.outfeatures = outfeatures + (-outfeatures % 32)
        self.infeatures = infeatures + (-infeatures % self.group_size)

        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=self.infeatures, outfeatures=self.outfeatures, **kwargs)

        self.bits = bits

        # backup original values
        self.original_outfeatures = outfeatures
        self.original_infeatures = infeatures

        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((self.original_infeatures // 32 * self.bits, self.original_outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(self.original_infeatures / self.group_size),
                    self.original_outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(self.original_infeatures / self.group_size), self.original_outfeatures),
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(self.original_infeatures)], dtype=torch.int32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(self.original_outfeatures, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if exllama_import_exception is not None:
            return False, exllama_import_exception
        return cls._validate(**args)

    def post_init(self):
        # resize due to padding after model weights have been loaded
        if self.outfeatures != self.original_outfeatures or self.infeatures != self.original_infeatures:
            self.qweight.resize_(self.infeatures // 32 * self.bits, self.outfeatures)
            self.qzeros.resize_(
                math.ceil(self.infeatures / self.group_size),
                self.outfeatures // 32 * self.bits
            )
            self.scales.resize_((math.ceil(self.infeatures / self.group_size), self.outfeatures),)
            self.g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32, device=self.g_idx.device)
            if self.bias is not None:
                self.bias.resize_(self.outfeatures)


        self.width = self.qweight.shape[1]

        # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
        self.q4 = ext_make_q4(
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx.to("cpu") if self._use_act_order else None,
            self.qweight.device.index,
        )

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
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.original_infeatures):
            intweight.append(
                torch.round((W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[
                    :, None
                ]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            for j in range(i, i + (32 // self.bits)):
                qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
            i += 32 // self.bits
            col += 1


        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        if x.dtype != torch.float16:
            logger.warning_once(
                f"Exllama kernel requires a float16 input activation, while {x.dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
            )

            x = x.half()

        # TODO: need to run checks to make sure there is no performance regression padding with F.pad
        # if infeatures is padded, we need to pad the input as well
        if x.size(-1) != self.infeatures:
            x = F.pad(x, (0, self.infeatures - self.original_infeatures))

        out = ext_q4_matmul(x, self.q4, self.width)

        if self.bias is not None:
            out.add_(self.bias)

        return out
