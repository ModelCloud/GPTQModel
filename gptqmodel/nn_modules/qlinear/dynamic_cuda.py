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

from typing import Optional, Tuple

import torch

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.logger import setup_logger

from ...models._const import DEVICE, PLATFORM


logger = setup_logger()


gptqmodel_cuda_import_exception = None
try:
    import gptqmodel_cuda_64  # noqa: E402
    import gptqmodel_cuda_256  # noqa: E402
except ImportError as e:
    gptqmodel_cuda_import_exception = e


class DynamicCudaQuantLinear(TorchQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False # TODO fix this
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]

    # for transformers/optimum tests compat
    QUANT_TYPE = "cuda"

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
            kernel_switch_threshold=128,
            **kwargs,
    ):
        if gptqmodel_cuda_import_exception is not None:
            raise ValueError(
                f"Trying to use the cuda backend, but could not import the C++/CUDA dependencies with the following error: {gptqmodel_cuda_import_exception}"
            )
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures,
                         outfeatures=outfeatures, bias=bias, weight_dtype=weight_dtype, **kwargs)

        self.kernel_switch_threshold = kernel_switch_threshold

        # use faster cuda_256 by default
        self.gptqmodel_cuda = gptqmodel_cuda_256

        # fall back to cuda_64
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            self.gptqmodel_cuda = gptqmodel_cuda_64

        assert infeatures % 64 == 0 and outfeatures % 64 == 0

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if gptqmodel_cuda_import_exception is not None:
            return False, gptqmodel_cuda_import_exception
        return cls._validate(**args)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        assert x.device.type == "cuda"

        if x.shape[0] >= self.kernel_switch_threshold:
            logger.warning_once(
               f"Cannot run on cuda kernel. Using torch forward() that may be slower. Shape: `{x.shape[0]}` >= `{self.kernel_switch_threshold}`")
            return self._forward(x, x_dtype, out_shape)

        out = torch.zeros((x.shape[0], self.outfeatures), device=x.device, dtype=torch.float32)
        if self.bits == 2:
            self.gptqmodel_cuda.vecquant2matmul(
                x.float(),
                self.qweight,
                out,
                self.scales.float(),
                self.qzeros,
                self.g_idx,
            )
        elif self.bits == 3:
            self.gptqmodel_cuda.vecquant3matmul(
                x.float(),
                self.qweight,
                out,
                self.scales.float(),
                self.qzeros,
                self.g_idx,
            )
        elif self.bits == 4:
            self.gptqmodel_cuda.vecquant4matmul(
                x.float(),
                self.qweight,
                out,
                self.scales.float(),
                self.qzeros,
                self.g_idx,
            )
        elif self.bits == 8:
            self.gptqmodel_cuda.vecquant8matmul(
                x.float(),
                self.qweight,
                out,
                self.scales.float(),
                self.qzeros,
                self.g_idx,
            )
        out = out.to(x_dtype)
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["DynamicCudaQuantLinear"]
