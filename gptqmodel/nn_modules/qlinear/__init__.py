# Copyright 2025 ModelCloud
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
import math
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch as t  # conflict with torch.py
import torch.nn as nn
import transformers

from ...models._const import DEVICE, PLATFORM
from ...quantization.config import Extension


class BaseQuantLinear(nn.Module):
    SUPPORTS_BITS: List[int] = None
    SUPPORTS_GROUP_SIZE: List[int] = None
    SUPPORTS_DESC_ACT: List[bool] = None
    SUPPORTS_SYM: List[bool] = None
    SUPPORTS_SHARDS: bool = None
    SUPPORTS_TRAINING: bool = None
    SUPPORTS_AUTO_PADDING: bool = None
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: List[int] = None
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: List[int] = None

    SUPPORTS_PACK_DTYPES: List[t.dtype] = None
    SUPPORTS_EXTENSIONS: List[Extension] = None
    SUPPORTS_DEVICES: List[DEVICE] = None
    SUPPORTS_PLATFORM: List[PLATFORM] = None

    def __init__(self,
                 name: str,
                 bits: int,
                 group_size: int,
                 desc_act: bool,
                 sym: bool,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 pack_dtype: t.dtype,
                 register_buffers: bool = False,
                 register_buffers_in_features: int = None,
                 register_buffers_out_features: int = None,
                 **kwargs):
        super().__init__()
        self.name = name # full path module name in model weights
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features
        self.bits = bits
        self.desc_act = desc_act
        self.pack_dtype = pack_dtype
        self.maxq = 2 ** self.bits - 1
        self.pack_dtype = pack_dtype

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

        self.pack_factor = self.pack_dtype_bits // self.bits
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, in_features=in_features, out_features=out_features, pack_dtype=pack_dtype)
        if err:
            raise err

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
                    dtype=t.float16,  # Scales are always float16
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
            extension:Optional[Extension]=None,
    ) -> Tuple[
        bool, Optional[Exception]]:
        return cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym,
                                      in_features=in_features, out_features=out_features, pack_dtype=pack_dtype,
                                      dynamic=dynamic, device=device, trainable=trainable, extension=extension)

    @classmethod
    # internal method and should not be overriden
    def verify_supports_params(cls):
        """
        Validate that SUPPORTS parameters are not None or empty lists, raising an exception if the validation fails.
        """
        base_supports_variables = [
            (name, value) for name, value in BaseQuantLinear.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value)
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
                  out_features:int=None, device:Optional[DEVICE]=None, trainable:Optional[bool]=None, extension:Optional[Extension]=None) -> Tuple[bool, Optional[Exception]]:
        cls.verify_supports_params()

        if extension is not None and extension.__class__ not in cls.SUPPORTS_EXTENSIONS:
            err = f"{cls} does not support extension: {extension}"
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
        if group_size not in cls.SUPPORTS_GROUP_SIZE:
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

    # override me
    def post_init(self):
        pass

class PackableQuantLinear(BaseQuantLinear):
    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=t.float16)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=t.float16)

        intweight = t.round((W + scale_zeros[self.g_idx].T) / scales[self.g_idx].T).to(t.int32)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(self.pack_np_math_dtype)

        qweight = np.zeros((intweight.shape[0] // self.pack_dtype_bits * self.bits, intweight.shape[1]),
                           dtype=self.pack_np_dtype)
        if self.bits in [2, 4, 8]:
            for row in range(qweight.shape[0]):
                for j in range(self.pack_factor):
                    qweight[row] |= intweight[row * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            for row in range(qweight.shape[0]):
                row_offset = row * 10  # Cache row * 10
                row_offset_plus_10 = row_offset + 10  # Cache row * 10 + 10
                for j in range(10):
                    qweight[row] |= intweight[row_offset + j] << (3 * j)
                qweight[row] |= intweight[row_offset_plus_10] << 30
                row += 1
                qweight[row] |= (intweight[row_offset_plus_10] >> 2) & 1
                for j in range(10):
                    qweight[row] |= intweight[row_offset + j] << (3 * j + 1)
                qweight[row] |= intweight[row_offset_plus_10] << 31
                row += 1
                qweight[row] |= (intweight[row_offset_plus_10] >> 1) & 0x3
                for j in range(10):
                    qweight[row] |= intweight[row_offset + j] << (3 * j + 2)

        self.qweight = t.from_numpy(qweight.astype(self.pack_np_dtype))

        zeros = zeros.numpy().astype(self.pack_np_math_dtype)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // self.pack_dtype_bits * self.bits),
                          dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for col in range(qzeros.shape[1]):
                for j in range(self.pack_factor):
                    qzeros[:, col] |= zeros[:, col * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            for col in range(qzeros.shape[1]):
                col_offset = col * 10  # Cache col * 10
                col_offset_plus_10 = col_offset + 10  # Cache col * 10 + 10
                for j in range(10):
                    qzeros[:, col] |= zeros[:, col_offset + j] << (3 * j)
                qzeros[:, col] |= zeros[:, col_offset_plus_10] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, col_offset_plus_10] >> 2) & 1
                for j in range(10):
                    qzeros[:, col] |= zeros[:, col_offset + j] << (3 * j + 1)
                qzeros[:, col] |= zeros[:, col_offset_plus_10] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, col_offset_plus_10] >> 1) & 0x3
                for j in range(10):
                    qzeros[:, col] |= zeros[:, col_offset + j] << (3 * j + 2)

        self.qzeros = t.from_numpy(qzeros.astype(self.pack_np_dtype))
