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

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear.torch import TorchQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger


log = setup_logger()


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
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    # for transformers/optimum tests compat
    QUANT_TYPE = "cuda"

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
            kernel_switch_threshold=128,
            **kwargs,
    ):
        if gptqmodel_cuda_import_exception is not None:
            raise ValueError(
                f"Trying to use the cuda backend, but could not import the C++/CUDA dependencies with the following error: {gptqmodel_cuda_import_exception}"
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
            backend=kwargs.pop("backend", BACKEND.CUDA),
            adapter=adapter,
            **kwargs)

        # assert in_features % 64 == 0 and out_features % 64 == 0

        self.kernel_switch_threshold = kernel_switch_threshold

        # use faster cuda_256 by default
        self.gptqmodel_cuda = gptqmodel_cuda_256

        # fall back to cuda_64
        if in_features % 256 != 0 or out_features % 256 != 0:
            self.gptqmodel_cuda = gptqmodel_cuda_64

        if self.bits == 4:
            self.qmatmul = self.gptqmodel_cuda.vecquant4matmul
        elif self.bits == 8:
            self.qmatmul = self.gptqmodel_cuda.vecquant8matmul
        elif self.bits == 2:
            self.qmatmul = self.gptqmodel_cuda.vecquant2matmul
        elif self.bits == 3:
            self.qmatmul = self.gptqmodel_cuda.vecquant3matmul

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if gptqmodel_cuda_import_exception is not None:
            return False, gptqmodel_cuda_import_exception
        return cls._validate(**args)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        # assert x.device.type == "cuda"

        # switch to torch kernel when input shape is >= kernel_switch_threshold
        # cuda is only optimized for < kernel_switch_threshold and will run slower than torch otherwise
        if x.shape[0] >= self.kernel_switch_threshold:
            # logger.warning_once(
            #   f"Input shape `{x.shape[0]}` >= `{self.kernel_switch_threshold}` is not optimized for cuda kernel: dynamic switching to torch kernel.")
            return self._forward(x, x.dtype, out_shape)

        out = torch.zeros((x.shape[0], self.out_features), device=x.device, dtype=torch.float32)
        self.qmatmul(
            x.to(dtype=torch.float32),
            self.qweight,
            out,
            self.scales.to(dtype=torch.float32),
            self.qzeros,
            self.g_idx,
        )

        out = out.reshape(out_shape)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        if self.bias is not None:
            out.add_(self.bias)

        return out.to(x.dtype)


__all__ = ["DynamicCudaQuantLinear"]
