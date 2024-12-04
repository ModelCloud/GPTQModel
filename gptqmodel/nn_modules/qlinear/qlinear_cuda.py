import torch
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.logger import setup_logger
from gptqmodel.nn_modules.qlinear.qlinear_torch import TorchQuantLinear
from ...models._const import DEVICE

logger = setup_logger()

cuda_import_exception = None
try:
    import gptqmodel_cuda_64
    import gptqmodel_cuda_256

    _gptqmodel_cuda_available = True
except ImportError as e:
    cuda_import_exception = e
    logger.warning("CUDA extension not installed.")
    gptqmodel_cuda_256 = None
    gptqmodel_cuda_64 = None
    _gptqmodel_cuda_available = False


class CudaQuantLinear(TorchQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_DEVICES = [DEVICE.CUDA]

    def __init__(
            self,
            bits: int,
            group_size: int,
            sym: bool,
            desc_act: bool,
            infeatures: int,
            outfeatures: int,
            bias: bool,
            kernel_switch_threshold=128,
            weight_dtype=torch.float16,
            **kwargs,
    ):
        super().__init__(bits=bits, group_size=group_size, sym=sym, desc_act=desc_act, infeatures=infeatures,
                         outfeatures=outfeatures, **kwargs)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.gptqmodel_cuda_available = _gptqmodel_cuda_available

        self.gptqmodel_cuda = gptqmodel_cuda_256
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            self.gptqmodel_cuda = gptqmodel_cuda_64
        if infeatures % 64 != 0 or outfeatures % 64 != 0:
            self.gptqmodel_cuda_available = False

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        x_dtype = x.dtype

        if x.device.type != "cuda":
            raise NotImplementedError(f"Unable to use cuda kernel. x.device.type is {x.device.type}")

        if not self.gptqmodel_cuda_available:
            raise ValueError(
                f"Trying to use the cuda backend, but could not import the C++/CUDA dependencies with the following error: {cuda_import_exception}"
            )

        if self.kernel_switch_threshold != 0 and x.shape[0] >= self.kernel_switch_threshold:
            raise ValueError(
                f"Trying to use the cuda backend, x.shape[0] is {x.shape[0]}, x.shape[0] cannot be greater than kernel_switch_threshold{self.kernel_switch_threshold}"
            )

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


__all__ = ["CudaQuantLinear"]
