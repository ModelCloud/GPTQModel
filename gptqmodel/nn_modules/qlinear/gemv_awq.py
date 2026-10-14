# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Optional, Tuple

import torch
from torch import nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import pack_awq_columns
from ...utils.awq import awq_gemmv2_forward, awq_gemv_forward
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width
from ...utils.logger import setup_logger


log = setup_logger()

class AwqGEMVLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMV]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMV: 40}
    SUPPORTS_BITS = [4]
    # Both native AWQ GEMV implementations support only these normalized group
    # sizes. Keep -1 for kernel selection, then normalize it in validate() when
    # the layer's in_features value is available.
    SUPPORTS_GROUP_SIZE = [-1, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    # The M>8 path uses GEMMV2, whose tile is N=64. Keep the class-wide
    # contract at 64 so shape-dependent dispatch cannot select an unsupported
    # native launch.
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemv"

    @classmethod
    def validate(
        cls,
        bits: int,
        group_size: int = -1,
        desc_act: bool = False,
        sym: bool = True,
        in_features: int = None,
        out_features: int = None,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        """Validate the native group-size contract after resolving -1."""
        effective_group_size = in_features if group_size == -1 and in_features is not None else group_size
        native_group_sizes = [size for size in cls.SUPPORTS_GROUP_SIZE if size != -1]
        if effective_group_size != -1 and effective_group_size not in native_group_sizes:
            return False, NotImplementedError(
                f"{cls.__name__} native GEMV supports normalized group_size "
                f"{native_group_sizes}: actual group_size={effective_group_size}"
            )
        # Base validation cannot infer that -1 means the full input width.
        return super().validate(
            bits=bits,
            group_size=effective_group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )

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
        register_buffers: bool = False,
        **kwargs,
    ):
        backend = kwargs.pop("backend", BACKEND.AWQ_GEMV)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=backend,
            adapter=adapter,
            register_buffers=False,
            **kwargs)

        self.split_k_iters = 8

        if register_buffers:
            self.register_buffer(
                "qweight",
                torch.zeros((out_features, in_features // self.pack_factor), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    out_features,
                    calculate_zeros_width(in_features, self.group_size),
                    dtype=self.pack_dtype,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    out_features,
                    calculate_zeros_width(in_features, self.group_size) * self.pack_factor,
                    dtype=torch.float16,
                ),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
            else:
                self.bias = None

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

        super().post_init()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        if self.input_rows(x) == 0:
            # Avoid a native launch with M=0 while retaining the input rank.
            return self.empty_linear_output(x)
        inputs = x.reshape(-1, x.shape[-1])
        if not inputs.is_contiguous():
            # The CUDA entry point uses vectorized loads and requires a dense
            # 2D activation matrix.
            inputs = inputs.contiguous()

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()

        if inputs.shape[0] > 8:
            out = awq_gemmv2_forward(
                inputs,
                self.qweight,
                self.scales,
                self.qzeros,
                self.group_size,
                self.split_k_iters,
            )
        else:
            out = awq_gemv_forward(
                inputs, self.qweight, self.scales, self.qzeros, self.group_size
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor=None):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // self.bits
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(linear.in_features, self.group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        qscales[:, : scales.shape[1]] = scales
        self.register_buffer("scales", qscales)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.clone().half())
        else:
            self.bias = None

        # Gather each input column's group scale/zero in a single pass instead
        # of launching one tiny kernel per input column.  The GEMV layout keeps
        # scales as [out_features, groups], so the group index selects columns.
        group_index = torch.arange(self.in_features, device=linear.weight.device) // self.group_size
        intweight = torch.round(
            (linear.weight.data + scale_zeros.index_select(1, group_index))
            / self.scales.index_select(1, group_index)
        ).to(torch.int32)
        self.register_buffer("qweight", pack_awq_columns(intweight, self.bits))

        zeros = zeros.to(dtype=torch.int32)
        # Pad the group axis up to the packed width so the vectorized packer
        # reproduces the loop's out-of-range skips (those codes stay zero).
        padded_zeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, self.group_size) * pack_num),
            dtype=torch.int32,
            device=zeros.device,
        )
        padded_zeros[:, : zeros.shape[1]] = zeros
        self.register_buffer("qzeros", pack_awq_columns(padded_zeros, self.bits))

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, bits={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.bits,
                self.group_size,
            )
        )

__all__ = ["AwqGEMVLinear"]
