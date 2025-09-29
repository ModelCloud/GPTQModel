# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from .awq_gemm import AwqGEMMQuantLinear


log = setup_logger()

try:
    from intel_extension_for_pytorch.llm.quantization import IPEXWeightOnlyQuantizedLinear

    assert hasattr(IPEXWeightOnlyQuantizedLinear, "from_weight"), "The minimum version for ipex is at least 2.4"
    IPEX_INSTALLED = True
except:
    IPEX_INSTALLED = False


class Awq_IPEXQuantLinear(AwqGEMMQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemm_ipex"

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
        assert IPEX_INSTALLED, \
            "Please install IPEX package with `pip install intel_extension_for_pytorch`."

        self.init_ipex = False

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.IPEX),
            adapter=adapter,
            **kwargs)

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        # awq only accepts float16
        self.scales = self.scales.to(dtype=torch.float16)

        device_type = self.qweight.device.type
        if device_type != "meta":
            assert device_type in ("cpu", "xpu")

        super().post_init()

    def init_ipex_linear(self):
        if not self.training:
            self.ipex_linear = IPEXWeightOnlyQuantizedLinear.from_weight(self.qweight, self.scales, self.qzeros,
                                                                         self.in_features, self.out_features, None,
                                                                         self.bias,
                                                                         self.group_size, None, quant_method=1, dtype=0)

    def forward(self, x: torch.Tensor):
        assert IPEX_INSTALLED, (
            "IPEX kernels could not be loaded. "
            "Please install with `pip install intel_extension_for_pytorch` and "
            "refer to the detial https://github.com/intel/intel-extension-for-pytorch/tree/main")

        if not self.init_ipex:
            self.init_ipex_linear()
            self.init_ipex = True

        out_shape = x.shape[:-1] + (self.out_features,)

        if hasattr(self, "ipex_linear"):
            with torch.inference_mode():
                out = self.ipex_linear(x)
        else:
            out = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.bits, self.group_size).to(x.dtype)
            out = torch.matmul(x, out)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def backward(self, grad_output):
        weights = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.bits, self.group_size).to(
            grad_output.dtype)
        batch_size = grad_output.shape[0]
        grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1))

        return grad_input, None, None, None, None, None, None, None

    def extra_repr(self) -> str:
        return ("in_features={}, out_features={}, bias={}, bits={}, group_size={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.bits,
            self.group_size,
        ))


__all__ = ["Awq_IPEXQuantLinear"]
