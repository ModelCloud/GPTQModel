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

# Adapted from turboderp exllama: https://github.com/turboderp/exllama

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import BaseQuantLinear

exllama_import_exception = None
try:
    from gptqmodel_exllama_kernels import make_q4, q4_matmul
except ImportError as e:
    exllama_import_exception = e

log = setup_logger()

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
NONE_TENSOR = torch.empty((1, 1), device="meta")


def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(qweight, qzeros, scales, g_idx if g_idx is not None else NONE_TENSOR, device)

class ExllamaQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "exllama"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

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
        if exllama_import_exception is not None:
            raise ValueError(
                f"Trying to use the exllama backend, but could not import the C++/CUDA dependencies with the following error: {exllama_import_exception}"
            )

        # backup original values
        # self.original_out_features = out_features
        # self.original_in_features = in_features
        #
        # # auto pad
        # group_size = group_size if group_size != -1 else in_features
        # out_features = out_features + (-out_features % 32)
        # in_features = in_features + (-in_features % group_size)
        # self.in_features_padding_size = in_features - self.original_in_features
        # self.in_features_padding_shape = (0, self.in_features_padding_size)

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym, desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.EXLLAMA_V1),
            adapter=adapter,
            register_buffers=True,
            register_buffers_in_features=in_features,
            register_buffers_out_feature=out_features,
            **kwargs)

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if exllama_import_exception is not None:
            return False, exllama_import_exception
        return cls._validate(**args)

    def post_init(self):
        # resize due to padding after model weights have been loaded
        # if self.out_features != self.original_out_features or self.in_features != self.original_in_features:
        #     self.qweight.resize_(self.in_features // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.in_features / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.in_features / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.in_features)], dtype=torch.int32, device=self.g_idx.device)
        #     if self.bias is not None:
        #         self.bias.resize_(self.out_features)

        # ext_make_q4 only accept float16 scales
        self.scales = self.scales.to(dtype=torch.float16)

        self.width = self.qweight.shape[1]

        # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
        self.q4 = ext_make_q4(
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx.to("cpu") if self._use_act_order else None,
            self.qweight.device.index,
        )

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "q4") and self.q4 is not None:
            buf.append(self.q4)
        return buf

    def ext_q4_matmul(self, x, q4, q4_width):
        """Matrix multiplication, returns x @ q4"""
        outshape = x.shape[:-1] + (q4_width,)
        x = x.view(-1, x.shape[-1])

        output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)
        q4_matmul(x, q4, output)

        if self.bias is not None:
            output.add_(self.bias)

        if self.adapter:
            output = self.adapter.apply(x=x, out=output)

        return output.view(outshape)

    def forward(self, x: torch.Tensor):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        x_dtype = x.dtype
        if x_dtype != torch.float16:
            #log.warn.once(
            #    f"Exllama kernel requires a float16 input activation, while {x.dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
            #)

            x = x.to(dtype=torch.float16)

        # TODO: need to run checks to make sure there is no performance regression padding with F.pad
        # if in_features is padded, we need to pad the input as well
        # if x.size(-1) != self.in_features:
        #     x = F.pad(x, self.in_features_padding_shape)

        out = self.ext_q4_matmul(x, self.q4, self.width)

        return out.to(x_dtype)
