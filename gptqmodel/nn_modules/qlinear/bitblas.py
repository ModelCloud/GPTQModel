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

import ctypes
import operator
import os
from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils import BACKEND
from ...utils.logger import setup_logger

log = setup_logger()

BITBLAS_TARGET = None
BITBLAS_DATABASE_PATH = None
BITBLAS_PROPAGATE_WEIGHTS = False

try:
    import bitblas  # noqa: F401
    BITBLAS_AVAILABLE = True
except Exception:
    BITBLAS_AVAILABLE = False

BITBLAS_INSTALL_HINT = "bitblas is not installed. Please install via `pip install bitblas`."


def import_bitblas():
    # print("import_bitblas() called")
    global BITBLAS_DATABASE_PATH, BITBLAS_TARGET

    # guard against bitblas pip whl incompatible env`
    import bitblas

    bitblas.set_log_level("INFO")

    if BITBLAS_TARGET is None:
        from .bitblas_target_detector import patched_auto_detect_nvidia_target

        bitblas.auto_detect_nvidia_target = patched_auto_detect_nvidia_target
        BITBLAS_TARGET = patched_auto_detect_nvidia_target(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
        os.environ["TVM_TARGET"] = f"{BITBLAS_TARGET}"
        print(f"BITBLAS_TARGET {BITBLAS_TARGET}")

    if BITBLAS_DATABASE_PATH is None:
        from bitblas.cache import get_database_path
        BITBLAS_DATABASE_PATH = f"{get_database_path()}_{bitblas.__version__}"
        print(f"BITBLAS_DATABASE_PATH: {BITBLAS_DATABASE_PATH}")

def unpack_qzeros(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i)) & 0xF

    return unpacked_zeros


class BitBLASQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [1, 2, 4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [16]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [16]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16]

    OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512]
    zeros_mode = "quantized"  # "original" or "rescale" or "quantized"
    TORCH_DTYPE = torch.float16
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }
    # for transformers/optimum tests compat
    QUANT_TYPE = "bitblas"

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
        enable_tuning: bool = True,
        fast_decoding: bool = True,
        propagate_b: bool = BITBLAS_PROPAGATE_WEIGHTS,
        opt_features: Union[int, List[int]] = OPT_FEATURES,
        layout: str = "nt",
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
            backend=kwargs.pop("backend", BACKEND.BITBLAS),
            adapter=adapter,
            register_buffers=False,
            **kwargs)

        import_bitblas()

        self._validate_parameters(group_size, in_features, out_features)

        self.opt_features = opt_features
        self.target = BITBLAS_TARGET
        self._configure_bitblas_matmul(
            enable_tuning, fast_decoding, bias, propagate_b, layout, bits
        )
        self._initialize_buffers(in_features, out_features, bias)
        self.reset_parameters()

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if not BITBLAS_AVAILABLE:
            return False, ValueError(BITBLAS_INSTALL_HINT)
        return cls._validate(**args)

    def _validate_parameters(
        self, group_size: int, in_features: int, out_features: int
    ):
        if in_features % group_size != 0:
            raise ValueError("`in_features` must be divisible by `group_size`.")

    def _initialize_buffers(self, in_features: int, out_features: int, bias: bool):
        self.register_buffer(
            "qweight",
            torch.zeros(
                self.bitblas_matmul.retrieve_weight_shape(),
                dtype=self.TORCH_STORAGE_DTYPE,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // self.group_size), dtype=self.TORCH_DTYPE
            ),
        )
        if self.zeros_mode == "quantized":
            storage_nbit = int("".join(c for c in self.STORAGE_DTYPE if c.isdigit()))
            self.register_buffer(
                "zeros",
                torch.zeros(
                    (in_features // self.group_size, out_features // storage_nbit * self.bits), dtype=self.TORCH_STORAGE_DTYPE
                ),
            )
        else:
            self.register_buffer(
                "zeros",
                torch.zeros(
                    (out_features, in_features // self.group_size), dtype=self.TORCH_DTYPE
                ),
            )

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=self.TORCH_DTYPE)
            )
        else:
            self.bias = None

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "zeros") and self.zeros is not None:
            buf.append(self.zeros)
        return buf

    def _configure_bitblas_matmul(
        self, enable_tuning: bool, fast_decoding: bool, bias: bool, propagate_b, layout, bits: int
    ):
        from bitblas import MatmulConfig

        # Assuming MatmulWeightOnlyDequantizeConfig and MatmulWeightOnlyDequantize are defined elsewhere
        bitblas_dtype = self.BITBLAS_DTYPES[self.TORCH_DTYPE]
        W_dtype = f"uint{bits}"
        matmul_config = MatmulConfig(
            M=self.opt_features,
            N=self.out_features,
            K=self.in_features,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.STORAGE_DTYPE,
            with_scaling=True,
            with_zeros=True,
            group_size=self.group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=self.zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=self.target)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                log.info(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                log.info("BitBLAS Operator created.")
        else:
            log.info("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def reset_parameters(self):
        # init for char
        self.qweight = torch.randint_like(
            self.qweight,
            0,
            2 ** (self.bits - 1) - 1,
            dtype=torch.int8,
            device=self.qweight.device,
        )
        nn.init.normal_(self.scales)
        nn.init.zeros_(self.zeros)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.q_params = None

    def post_init(self):
        # eliminate runtime overhead like exllama state
        param_list = [self.qweight, self.scales, self.zeros]
        if self.bitblas_matmul.config.with_bias:
            param_list.append(self.bias)
        self.q_params = [ctypes.c_void_p(arr.data_ptr()) for arr in param_list]

    def pack(self, linear, scales, zeros, g_idx=None):
        from bitblas.quantization.utils import general_compress

        W = linear.weight.data.clone()

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias:
            self.bias = linear.bias.clone().half()

        intweight = torch.round((W + scale_zeros[g_idx].T) / scales[g_idx].T).to(torch.int)

        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, i + (32 // self.bits)):
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1

        qweight = qweight.astype(np.int32)
        qweight = torch.from_numpy(qweight)
        qweight = qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(qweight.cpu()).cuda()
        self.qweight = qweight

        scales = self.scales.T.contiguous().view(self.TORCH_DTYPE)
        self.scales = scales

        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            for j in range(i, i + (32 // self.bits)):
                qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
            i += 32 // self.bits
            col += 1

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

        intzeros = unpack_qzeros(self.qzeros, self.bits).T.contiguous()
        if self.bitblas_matmul.config.zeros_mode == "original":
            self.zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            self.zeros = (
                torch.Tensor(
                    general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)
                )
                .to(self.qweight.device)
                .to(self.zeros.dtype)
                .contiguous()
            )
        else:
            raise ValueError(
                f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}"
            )

        if self.bias is not None:
            self.bias = self.bias.data.to(torch.float16).contiguous()

    def repack_from_gptq(self, gptq_module):
        from bitblas.quantization.utils import general_compress

        # qweight in gptq old quant linear stored with (out_features, in_features), should be transposed.
        qweight = gptq_module.qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(qweight.cpu()).cuda()
        self.qweight = qweight
        # scales in gptq old quant linear stored with (infeatures // group_size, outfeatures), should be transposed.
        scales = gptq_module.scales.T.contiguous().view(self.TORCH_DTYPE)
        self.scales = scales
        # qzeros should be de-quantized to int zeros.
        intzeros = unpack_qzeros(gptq_module.qzeros, self.bits).T.contiguous()
        if self.bitblas_matmul.config.zeros_mode == "original":
            self.zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            self.zeros = (
                torch.Tensor(
                    general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)
                )
                .to(self.qweight.device)
                .to(self.zeros.dtype)
                .contiguous()
            )
        else:
            raise ValueError(
                f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}"
            )
        if self.bias is not None:
            self.bias = gptq_module.bias.data.to(torch.float16).contiguous()

    def forward(self, A):
        if A.dtype != torch.float16:
            A = A.half()

        C = torch.empty(
            A.shape[:-1] + (self.scales.shape[0],), dtype=A.dtype, device=A.device
        )

        # m is the product of the last n - 1 dimensions of A
        m = ctypes.c_int32(reduce(operator.mul, A.shape[:-1], 1))
        self.bitblas_matmul.call_lib(
            ctypes.c_void_p(A.data_ptr()) , *self.q_params, ctypes.c_void_p(C.data_ptr()), m
        )

        if self.adapter:
            C = self.adapter.apply(x=A, out=C)

        return C


__all__ = ["BitBLASQuantLinear"]
