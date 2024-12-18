# License: GPTQModel/licenses/LICENSE.apache

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

    SUPPORTS_DEVICES = [DEVICE.CUDA]
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

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        assert x.device.type == "cuda"

        if x.shape[0] >= self.kernel_switch_threshold:
            logger.warning_once(
               f"Cannot run on cuda kernel. Using torch forward() that may be slower. Shape: `{x.shape[0]}` >= `{self.kernel_switch_threshold}`")
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
