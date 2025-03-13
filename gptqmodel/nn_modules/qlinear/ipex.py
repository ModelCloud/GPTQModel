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

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from .torch import TorchQuantLinear

log = setup_logger()

BITS_DTYPE_MAPPING = {
    4: "int4_clip",
}

HAS_IPEX = False
IPEX_ERROR_LOG = None
try:
    from intel_extension_for_pytorch.llm.quantization import IPEXWeightOnlyQuantizedLinear, QuantDtype, QuantMethod

    HAS_IPEX = True
except BaseException:
    HAS_IPEX = False
    IPEX_ERROR_LOG = Exception

def ipex_dtype() -> torch.dtype:
    if not HAS_IPEX:
        raise ImportError("intel_extension_for_pytorch not installed. "
                          "Please install via `pip install intel_extension_for_pytorch`")

    return torch.float16


def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)

def convert_idx(self, g_idx, k):
    ret_idx = torch.zeros(k, dtype=int).to(g_idx.device)
    groups = k // self.blocksize
    remainder = k % self.blocksize
    g_idx_2 = g_idx * self.blocksize
    if remainder > 0:
        g_idx_2[g_idx == groups] += torch.arange(remainder).to(g_idx.device)
    arange_tensor = torch.arange(self.blocksize).to(g_idx.device)
    for i in range(groups):
        g_idx_2[g_idx == i] += arange_tensor
    ret_idx[g_idx_2] = torch.arange(k).to(g_idx.device)
    return ret_idx.to(torch.int32)

if HAS_IPEX:
    try:
        # monkey patch GPTQShuffle.convert_idx to use fixed convert_idx, fix the slow ipex generate issue
        from intel_extension_for_pytorch.nn.utils._quantize_convert import GPTQShuffle

        GPTQShuffle.convert_idx = convert_idx
    except ImportError:
        # if import GPTQShuffle failed, do nothing
        pass

class IPEXQuantLinear(TorchQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True

    SUPPORTS_TRAINING = True
    SUPPORTS_TRAINING_USE_TORCH_KERNEL = True

    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "ipex"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
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
            backend=kwargs.pop("backend", BACKEND.IPEX),
            **kwargs)

        self.weight_dtype = torch.float16

    @classmethod
    def validate(cls, bias: bool = False, adapter: Optional[Adapter] = None, **args) -> Tuple[bool, Optional[Exception]]:
        if not HAS_IPEX:
            return False, IPEX_ERROR_LOG
        return cls._validate(**args)

    def post_init(self):
        self.ipex_linear = IPEXWeightOnlyQuantizedLinear.from_weight(
            self.qweight,
            self.scales,
            self.qzeros,
            self.in_features,
            self.out_features,
            None,
            self.bias,
            self.group_size,
            self.g_idx,
            quant_method=QuantMethod.GPTQ_GEMM,
            dtype=QuantDtype.INT4)

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "ipex_linear") and self.ipex_linear is not None:
            buf.append(self.ipex_linear)
        return buf

    def forward(self, x: torch.Tensor):
        if self.training:
            return super().forward(x)

        if self.adapter:
            return self.adapter(x=x, out=self.ipex_linear(x))
        else:
            return self.ipex_linear(x)

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        # self.forward = torch_compile(self.forward, backend=backend, mode=mode, fullgraph=fullgraph)
        # torch.compile is incompatible with ipex woq linear, will enable it after we fix this issue.
        pass

__all__ = ["IPEXQuantLinear"]
