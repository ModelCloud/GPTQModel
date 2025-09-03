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


import torch

from ..triton_utils.a100_qlinear import a100_qlinear
from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import PackableQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from packaging import version

log = setup_logger()

try:
    # TODO: triton is not compatible with free threading
    # if not has_gil_disabled():
    #     raise Exception("GIL is disabled so Triton is not (yet) compatible.")

    import triton
    import triton.language as tl
    from triton import __version__ as triton_version

    if version.parse(triton_version) < version.parse("2.0.0"):
        raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")
    TRITON_AVAILABLE = True
except BaseException:
    TRITON_AVAILABLE = False

TRITON_INSTALL_HINT = "Trying to use the triton backend, but it could not be imported. Please install triton by 'pip install gptqmodel[triton] --no-build-isolation'"
TRITON_XPU_INSTALL_HINT = "Trying to use the triton backend and xpu device, but it could not be imported. Please install triton by [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)"

class TritonA100QuantLinear(PackableQuantLinear):
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
    QUANT_TYPE = "triton_a100"

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
        register_buffers: bool = True,
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
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

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

        super().post_init()

        # torch benefits the most from torch.compile, enable it by default
        self.optimize()

    def optimize(self, backend: str = None, mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        if backend is None:
            # MPS doesn't support inductor.
            backend = "inductor" if self.list_buffers()[0].device.type != "mps" else "aot_eager"

        # # compile dequantize
        # self.dequantize_weight = torch_compile(self.dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)

        if self.adapter:
            self.adapter.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

        super().optimize()

    def forward(self, x: torch.Tensor):
        if self.training:
            return super().forward(x)

        out_shape = x.shape[:-1] + (self.out_features,)

        block_size_m = x.shape[0]
        # TODO test a100_qlinear
        out = a100_qlinear.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.group_size,
            block_size_m,
        ).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x.dtype)

__all__ = ["TritonA100QuantLinear"]
