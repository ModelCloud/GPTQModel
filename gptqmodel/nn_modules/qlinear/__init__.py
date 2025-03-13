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
import copy
import math
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch as t  # conflict with torch.py
import torch.nn as nn
import transformers

from ...adapter.adapter import LORA_MERGED_WEIGHT_PATHS, Adapter
from ...models._const import DEVICE, PLATFORM
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger

log = setup_logger()

class BaseQuantLinear(nn.Module):
    SUPPORTS_BITS: List[int] = None
    SUPPORTS_GROUP_SIZE: List[int] = None
    SUPPORTS_DESC_ACT: List[bool] = None
    SUPPORTS_SYM: List[bool] = None
    SUPPORTS_SHARDS: bool = None
    SUPPORTS_TRAINING: bool = None

    # IPEX kernel will use Torch for training only and switches back to IPEX for eval/inference
    SUPPORTS_TRAINING_USE_TORCH_KERNEL: bool = False

    SUPPORTS_AUTO_PADDING: bool = None
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: List[int] = None
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: List[int] = None

    SUPPORTS_PACK_DTYPES: List[t.dtype] = None
    SUPPORTS_ADAPTERS: List[Adapter] = None
    SUPPORTS_DEVICES: List[DEVICE] = None
    SUPPORTS_PLATFORM: List[PLATFORM] = None

    SUPPORTS_DTYPES: List[t.dtype] = None

    def __init__(self,
                 bits: int,
                 group_size: int,
                 desc_act: bool,
                 sym: bool,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 pack_dtype: t.dtype,
                 backend: BACKEND,
                 adapter: Adapter,
                 name: str = None,
                 register_buffers: bool = False,
                 register_buffers_in_features: int = None,
                 register_buffers_out_features: int = None,
                 **kwargs):
        super().__init__()
        if name is None:
            name = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self.name = name # full path module name in model weights
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features
        self.bits = bits
        self.desc_act = desc_act
        self.pack_dtype = pack_dtype
        self.backend = backend
        self.maxq = 2 ** self.bits - 1
        self.pack_dtype = pack_dtype
        # we need to clone the adapter since passed in adapter may be shared
        # adapter tensors are lodaed inside adapter so they must be unique per module
        self.adapter =  copy.deepcopy(adapter)

        self.optimized = False

        if self.pack_dtype == t.int8:
            self.pack_dtype_bits = 8
            self.pack_np_dtype = np.int8 # qweight saved dtype
            self.pack_np_math_dtype = np.uint8 # pre-save math dtype
        elif self.pack_dtype == t.int16:
            self.pack_dtype_bits = 16
            self.pack_np_dtype = np.int16
            self.pack_np_math_dtype = np.uint16
        elif self.pack_dtype == t.int32:
            self.pack_dtype_bits = 32
            self.pack_np_dtype = np.int32
            self.pack_np_math_dtype = np.uint32
        elif self.pack_dtype == t.int64:
            self.pack_dtype_bits = 64
            self.pack_np_dtype = np.int64
            self.pack_np_math_dtype = np.uint64
        else:
            raise ValueError("Unsupported weight_dtype. Only int16 and int32 are supported.")

        # pack_factor is only used for bits 2, 4, and 8. bit3 3 does not use this variable.
        self.pack_factor = self.pack_dtype_bits // self.bits
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, in_features=in_features, out_features=out_features, pack_dtype=pack_dtype)
        if err:
            raise err

        # store qzero format
        self._qzeros_format = 1 # only valid values are 1 and 2 for GPTQ v1 GPTQ v2

        # most kernels share same buffers so they can share same register buffer code
        if register_buffers:
            # some kernels auto-pads in/out features
            in_features = self.in_features if not register_buffers_in_features else register_buffers_in_features
            out_features = self.out_features if not register_buffers_out_features else register_buffers_out_features

            self.register_buffer(
                "qweight",
                t.zeros((in_features // self.pack_dtype_bits * self.bits, out_features), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                t.zeros(
                    (
                        math.ceil(in_features / self.group_size),
                        out_features // self.pack_dtype_bits * self.bits,
                    ),
                    dtype=self.pack_dtype,
                ),
            )
            self.register_buffer(
                "scales",
                t.zeros(
                    (math.ceil(in_features / self.group_size), out_features),
                    dtype=t.float16,
                ),
            )
            self.register_buffer(
                "g_idx",
                t.tensor([i // self.group_size for i in range(in_features)], dtype=t.int32),
            )
            if bias:
                self.register_buffer("bias", t.zeros(out_features, dtype=t.float16))
            else:
                self.bias = None

        # load adapter if any
        if adapter is not None:
            if adapter.path in LORA_MERGED_WEIGHT_PATHS:
                print(f"Adapter (merged weights) lazy init: {self.adapter.name()}: {self.adapter}, module: {self.name}")

                # pre allocate buffers so accelerate can auto-bind merged weights in same tensor file as model
                self.register_buffer(
                    "lora_A",
                    t.zeros((in_features, adapter.rank), dtype=t.float16),
                )

                self.register_buffer(
                    "lora_B",
                    t.zeros((adapter.rank, out_features), dtype=t.float16),
                )
            else:
                pass
                # print(f"Adapter lazy init: {self.adapter.name()}: {self.adapter}, module: {self.name}")

            # TDOO: allow merged lora weights exist in gptq model safetensor file for direct loading
            # EoRA need to preallocate buffers for Lora_A and B weights so HF can load
            # self.register_buffer(
            #     "lora_A",
            #     torch.zeros((in_features, 128), dtype=torch.float16), # <-- EoRA lora_A shape needs to be calculated using pass in_features/out_features or other eora_test math
            # )
            #
            # # EoRA need to preallocate buffers for Lora_A and B weights so HF can load
            # self.register_buffer(
            #     "lora_B",
            #     torch.zeros((128, out_features), dtype=torch.float16), # <-- EoRA lora_A shape needs to be calculated using pass in_features/out_features or other eora_test math
            # )

    def list_buffers(self) -> List:
        buf = []
        if hasattr(self, "qweight") and self.qweight is not None:
            buf.append(self.qweight)
        if hasattr(self, "qzeros") and self.qzeros is not None:
            buf.append(self.qzeros)
        if hasattr(self, "scales") and self.scales is not None:
            buf.append(self.scales)
        if hasattr(self, "g_idx") and self.g_idx is not None:
            buf.append(self.g_idx)
        if hasattr(self, "bias") and self.bias is not None:
            buf.append(self.bias)

        return buf

    def qzero_format(self, format: int = None) -> int:
        # get
        if format is None:
            return self._qzeros_format

        # set
        if format not in [1, 2]:
            raise ValueError("Unsupported qzero format. Only 1 and 2 are supported.")

        self._qzeros_format = format
        return self._qzeros_format

    # override me, to perform post-weight load to device init
    def post_init(self):
        if self.adapter is not None:
            self.adapter.post_init(
                weight_key=self.name,
                device=self.list_buffers()[0].device,
                lora_A=getattr(self, "lora_A", None),
                lora_B=getattr(self, "lora_B", None))

    @classmethod
    # custom quant linear class can override this and add custom checks
    def validate(
            cls,
            bits: int,
            group_size: int,
            desc_act: bool,
            sym: bool,
            in_features:int=None,
            out_features:int=None,
            pack_dtype:t.dtype=None,
            dynamic:Optional[dict]=None,
            device:Optional[DEVICE]=None,
            trainable:Optional[bool]=None,
            adapter:Optional[Adapter]=None,
    ) -> Tuple[
        bool, Optional[Exception]]:
        return cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym,
                             in_features=in_features, out_features=out_features, pack_dtype=pack_dtype,
                             dynamic=dynamic, device=device, trainable=trainable, adapter=adapter)

    @classmethod
    # internal method and should not be overriden
    def verify_supports_params(cls):
        """
        Validate that SUPPORTS parameters are not None or empty lists, raising an exception if the validation fails.
        """
        base_supports_variables = [
            (name, value) for name, value in BaseQuantLinear.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value) and value is None
        ]
        child_supports_variables = [
            (name, value) for name, value in cls.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value)
        ]

        base_supports_variables.sort(key=lambda x: x[0])
        child_supports_variables.sort(key=lambda x: x[0])

        base_variable_names = {name for name, value in base_supports_variables}
        child_variable_names = {name for name, value in child_supports_variables}

        missing_variables = base_variable_names - child_variable_names

        if missing_variables:
            raise ValueError(
                f"{cls.__name__} these SUPPORTS variables are not overridden: {', '.join(sorted(missing_variables))}")

        for name, value in child_supports_variables:
            if not name.startswith("SUPPORTS") or callable(value):
                continue
            if value is None:
                raise ValueError(f"{cls.__name__}.{name} cannot be None.")

            # if isinstance(value, list) and not value:
            #     raise ValueError(f"{cls.__name__}.{name} cannot be an empty list.")

    @classmethod
    def _validate(cls, bits: int=4, group_size: int=128, desc_act: bool=False, sym: bool=False, pack_dtype:t.dtype=None, dynamic:Optional[dict]=None, in_features:int=None,
                  out_features:int=None, device:Optional[DEVICE]=None, trainable:Optional[bool]=None, adapter:Optional[Adapter]=None) -> Tuple[bool, Optional[Exception]]:
        cls.verify_supports_params()

        if adapter is not None and adapter.__class__ not in cls.SUPPORTS_ADAPTERS:
            err = f"{cls} does not support adapter: {adapter}"
            return False, NotImplementedError(err)

        if pack_dtype not in cls.SUPPORTS_PACK_DTYPES:
            err = f"{cls} does not support `pack_dtype`: {pack_dtype}"
            return False, NotImplementedError(err)

        if PLATFORM.ALL not in cls.SUPPORTS_PLATFORM and sys.platform not in cls.SUPPORTS_PLATFORM:
            err = f"{cls} does not support platform: {sys.platform}"
            return False, NotImplementedError(err)

        if DEVICE.ALL not in cls.SUPPORTS_DEVICES and device is not None:
            try:
                cls.validate_device(device)
            except NotImplementedError:
                e = f"{cls} does not support device: {device}"
                return False, NotImplementedError(e)

        if trainable and not cls.SUPPORTS_TRAINING:
            err = f"{cls} does not support training."
            return False, NotImplementedError(err)

        if bits not in cls.SUPPORTS_BITS:
            err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual bits = `{bits}`"
            return False, NotImplementedError(err)
        # valid group size is set of cls.SUPPORTS_GROUP_SIZE + in_features; group_size = -1 is alias for group_size == in_features
        if group_size not in cls.SUPPORTS_GROUP_SIZE and group_size != in_features:
            err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
            return False, NotImplementedError(err)
        if sym not in cls.SUPPORTS_SYM:
            err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}`"
            return False, NotImplementedError(err)
        if desc_act not in cls.SUPPORTS_DESC_ACT:
            err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}`"
            return False, NotImplementedError(err)
        if dynamic is not None:
            dynamic_bits = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_bits[pattern] = pattern_dict.get("bits", bits)
            if len(cls.SUPPORTS_BITS) == 1:
                err = f"{cls} not supported dynamic_bits, only support `{cls.SUPPORTS_BITS}` bits"
                return False, NotImplementedError(err)
            else:
                for layer, bits in dynamic_bits.items():
                    if bits not in cls.SUPPORTS_BITS:
                        err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual dynamic_bits = `{bits}` for layer `{layer}`"
                        return False, NotImplementedError(err)

            dynamic_group_size = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_group_size[pattern] = pattern_dict.get("group_size", group_size)
            for layer, group_size in dynamic_group_size.items():
                if group_size not in cls.SUPPORTS_GROUP_SIZE:
                    err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_sym = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_sym[pattern] = pattern_dict.get("sym", sym)
            for layer, sym in dynamic_sym.items():
                if sym not in cls.SUPPORTS_SYM:
                    err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_desc_act = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_desc_act[pattern] = pattern_dict.get("desc_act", desc_act)
            for layer, desc_act in dynamic_desc_act.items():
                if desc_act not in cls.SUPPORTS_DESC_ACT:
                    err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}` for layer `{layer}`"
                    return False, NotImplementedError(err)

        if in_features is not None:
            validate = all(in_features % in_fea == 0 for in_fea in cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `in_features` must be divisible by {cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)

            validate = in_features % group_size == 0 or cls.SUPPORTS_AUTO_PADDING
            if not validate:
                err = f"{cls}: `in_features` must be divisible by `group_size: {group_size}`."
                return False, NotImplementedError(err)
        if out_features is not None:
            validate = all(out_features % out_fea == 0 for out_fea in cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `out_features` must be divisible by {cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        assert isinstance(device, DEVICE)

        if device not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTS_DEVICES}`: actual device = `{device}`")

    # use optimize so we don't override native module.compile()
    # override me, to perform any torch.compile logic on the kernel pre forward
    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        self.optimized = True
        log.info.once(f"Optimize: `{self.__class__.__name__}` compilation triggered.")
        pass

    # overrides nn.module.train()
    def train(self, mode=True):
        old_mode = self.training

        if old_mode == mode:
            return self

        # Custom behavior when switching to training mode
        if mode:
            if not self.SUPPORTS_TRAINING:
                err = f"{self.__class__.__name__}: `{self.name}` switching to training mode."
                log.error(err)
                raise NotImplementedError(err)
            else:
                pass
                # log.info(f"{self.__class__.__name__}: `{self.name}` switching to training mode.")
        else:
            pass
            # log.info(f"{self.__class__.__name__}: `{self.name}` switching to eval mode.")

        return super().train(mode)

class PackableQuantLinear(BaseQuantLinear):
    def post_init(self, **kwargs):
        if self.bits in [2, 4, 8]:
            wf = t.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=t.int32).unsqueeze(0).to(
                device=self.g_idx.device)
        elif self.bits == 3:
            wf = t.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=t.int32,
            ).reshape(1, 3, 12).to(device=self.g_idx.device)

        # self.register_buffer("wf_unsqueeze_zero", wf.unsqueeze(0).to(device=self.g_idx.device))
        # self.register_buffer("wf_unsqueeze_neg_one", wf.unsqueeze(-1).to(device=self.g_idx.device))
        #
        self.wf_unsqueeze_zero = wf.unsqueeze(0).to(device=self.g_idx.device)
        self.wf_unsqueeze_neg_one = wf.unsqueeze(-1).to(device=self.g_idx.device)

        super().post_init(**kwargs)

    def list_buffers(self):
        return super().list_buffers() + [
            self.wf_unsqueeze_zero,
            self.wf_unsqueeze_neg_one,
        ]

    def dequantize_weight(self, num_itr: int = 1):
        if self.bits in [2, 4, 8]:
            zeros = t.bitwise_right_shift(
                t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
                self.wf_unsqueeze_zero  # self.wf.unsqueeze(0),
            ).to(self.dequant_dtype)
            zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

            weight = t.bitwise_and(
                t.bitwise_right_shift(
                    t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                    self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
                ).to(self.dequant_dtype),
                self.maxq
            )
        elif self.bits == 3:
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(
                -1, -1, -1, 12
            )
            zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = t.cat(
                [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                dim=2,
            ).reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if num_itr == 1:
            weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim: (i + 1) * num_dim]
                weight_i = weight[:, i * num_dim: (i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim: (i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim: (i + 1) * num_dim].long()
                weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
            weights = t.cat(weights, dim=1)

        return weights

    def pack(self, linear: nn.Module, scales: t.Tensor, zeros: t.Tensor, g_idx: t.Tensor=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.T

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.T.contiguous()
        zeros = zeros.T.contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=t.float16)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=t.float16)

        int_weight = t.round((W + scale_zeros[self.g_idx].T) / scales[self.g_idx].T).to(t.int32)
        int_weight = int_weight.T.contiguous()
        int_weight = int_weight.numpy().astype(self.pack_np_math_dtype)

        qweight = np.zeros((int_weight.shape[0] // self.pack_dtype_bits * self.bits, int_weight.shape[1]),
                           dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for row in range(qweight.shape[0]):
                for j in range(self.pack_factor):
                    qweight[row] |= int_weight[row * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            i = 0
            row = 0
            while row < qweight.shape[0]:
                for j in range(i, i + 10):
                    qweight[row] |= int_weight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= int_weight[i] << 30
                row += 1
                qweight[row] |= (int_weight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= int_weight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= int_weight[i] << 31
                row += 1
                qweight[row] |= (int_weight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= int_weight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1

        self.qweight = t.from_numpy(qweight.astype(self.pack_np_dtype))

        zeros = zeros.numpy().astype(self.pack_np_math_dtype)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // self.pack_dtype_bits * self.bits), dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for col in range(qzeros.shape[1]):
                for j in range(self.pack_factor):
                    qzeros[:, col] |= zeros[:, col * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            i = 0
            col = 0
            while col < qzeros.shape[1]:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1

        self.qzeros = t.from_numpy(qzeros.astype(self.pack_np_dtype))

        # assert
        # assert isinstance(self, TorchQuantLinear), f"type: {self.__class_}"
        # wq = linear.weight.data
        # wq_dequantized = self.dequantize_weight().T
        # print(f"------ WQ -----")
        # print(wq)
        # print(f"------ WQ Dequantized -----")
        # print(wq_dequantized)
        # assert t.equal(wq, wq_dequantized)

        # print("self qw", self.qweight, self.scales, self.qzeros)
