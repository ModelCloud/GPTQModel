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
import torch.nn.functional as F
import transformers
from packaging import version

from ...models._const import DEVICE, PLATFORM
from ...utils.logger import setup_logger
from . import BaseQuantLinear


try:
    import triton
    import triton.language as tl
    from triton import __version__ as triton_version

    from ..triton_utils.dequant import QuantLinearFunction
    from ..triton_utils.mixin import TritonModuleMixin
    if version.parse(triton_version) < version.parse("2.0.0"):
        raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")
    TRITON_AVAILABLE = True
except BaseException:
    TRITON_AVAILABLE = False
    class TritonModuleMixin:
        pass

TRITON_INSTALL_HINT = "Trying to use the triton backend, but it could not be imported. Please install triton by 'pip install gptqmodel[triton] --no-build-isolation'"
TRITON_XPU_INSTALL_HINT = "Trying to use the triton backend and xpu device, but it could not be imported. Please install triton by [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)"

logger = setup_logger()


class TritonV2QuantLinear(BaseQuantLinear, TritonModuleMixin):
    SUPPORTS_BITS = [2, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]

    # for transformers/optimum tests compat
    QUANT_TYPE = "tritonv2"

    """
    Triton v2 quantized linear layer.

    Calls dequant kernel (see triton_utils/dequant) to dequantize the weights then uses
    torch.matmul to compute the output whereas original `triton` quantized linear layer fused
    dequant and matmul into single kernel.add()
    """

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures, outfeatures, bias, **kwargs,):
        if not TRITON_AVAILABLE:
            raise ValueError(TRITON_INSTALL_HINT)
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
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if not TRITON_AVAILABLE:
            return False, ValueError(TRITON_INSTALL_HINT)

        device = args.get('device')

        if device == DEVICE.XPU and not triton_xpu_available():
            return False, ValueError(TRITON_XPU_INSTALL_HINT)

        return cls._validate(**args)

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
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

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

        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        for col in range(qzeros.shape[1]):
            i = col * (32 // self.bits)
            for j in range(32 // self.bits):
                qzeros[:, col] |= zeros[:, i + j] << (self.bits * j)

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        # if infeatures is padded, we need to pad the input as well
        if x.size(-1) != self.padded_infeatures:
            x = F.pad(x, (0, self.padded_infeatures - self.infeatures))

        out_shape = x.shape[:-1] + (self.outfeatures,)
        quant_linear_fn = QuantLinearFunction

        out = quant_linear_fn.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.maxq,
        )
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["TritonV2QuantLinear"]


def add(x: torch.Tensor, y: torch.Tensor):
    # don't put it on top-level to avoid crash if triton was not installed
    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                   y_ptr,  # *Pointer* to second input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y  # noqa: F841
    output = torch.empty_like(x)
    n_elements = output.numel()
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def triton_xpu_available():
    if not TRITON_AVAILABLE:
        return False
    size = 1024
    x = torch.rand(size, device='xpu:0')
    y = torch.rand(size, device='xpu:0')

    try:
        add(x, y)
        return True
    except Exception:
        return False


