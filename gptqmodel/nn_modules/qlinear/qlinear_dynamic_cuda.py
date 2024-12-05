# License: GPTQModel/licenses/LICENSE.apache

import torch
from gptqmodel.nn_modules.qlinear.qlinear_torch import TorchQuantLinear
from gptqmodel.utils.logger import setup_logger

from ...models._const import DEVICE

logger = setup_logger()

import gptqmodel_cuda_64
import gptqmodel_cuda_256


class DynamicCudaQuantLinear(TorchQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
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
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures,
                         outfeatures=outfeatures, bias=bias, weight_dtype=weight_dtype, **kwargs)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.gptqmodel_cuda_available = True

        self.gptqmodel_cuda = gptqmodel_cuda_256
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            self.gptqmodel_cuda = gptqmodel_cuda_64
        if infeatures % 64 != 0 or outfeatures % 64 != 0:
            self.gptqmodel_cuda_available = False

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        assert x.device.type == "cuda" 
        
        if x.shape[0] >= self.kernel_switch_threshold:
            logger.warning_once(
                "Does not meet the cuda kernel conditions, will use the non-optimized forward() with torch.")
            return super().forward(x)

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
