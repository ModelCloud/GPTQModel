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
from packaging import version

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from .torch import TorchQuantLinear

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

log = setup_logger()


class TritonV2QuantLinear(TorchQuantLinear, TritonModuleMixin):
    SUPPORTS_BITS = [2, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA] # Intel XPU can use Triton but this has been validated
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.int16, torch.int8]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "tritonv2"

    """
    Triton v2 quantized linear layer.

    Calls dequant kernel (see triton_utils/dequant) to dequantize the weights then uses
    torch.matmul to compute the output whereas original `triton` quantized linear layer fused
    dequant and matmul into single kernel.add()
    """

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features,
        out_features,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        **kwargs,
    ):
        if not TRITON_AVAILABLE:
            raise ValueError(TRITON_INSTALL_HINT)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TRITON),
            adapter=adapter,
            register_buffers=True,
            **kwargs)

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if not TRITON_AVAILABLE:
            return False, ValueError(TRITON_INSTALL_HINT)

        device = args.get('device')

        if device == DEVICE.XPU and not triton_xpu_available():
            return False, ValueError(TRITON_XPU_INSTALL_HINT)

        return cls._validate(**args)

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_factor, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_factor
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)
        super().post_init()

    def forward(self, x):
        if self.training:
            return super().forward(x)

        # if in_features is padded, we need to pad the input as well
        # if x.size(-1) != self.padded_infeatures:
        #     x = F.pad(x, (0, self.padded_infeatures - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)

        out = QuantLinearFunction.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        ).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x.dtype)


__all__ = ["TritonV2QuantLinear"]

# test triton on XPU to ensure special Intel/Triton is installed as we cannot check based on triton package meta data
def triton_test_add(x: torch.Tensor, y: torch.Tensor):
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
        triton_test_add(x, y)
        return True
    except Exception:
        return False


