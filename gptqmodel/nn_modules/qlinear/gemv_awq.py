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

# The GEMV/GEMV_FAST/LLM_AWQ kernels are entry points into
# ``gptqmodel_ext/awq`` (``gemv_forward``/``gemmv2_forward``), which is a CUDA
# JIT extension: its sources include ``cuda_fp16.h`` and CUDA-specific inline
# asm, and the sources that hipify does translate still fail on wave32 (the
# warp-sync mask must be 64-bit).  It therefore never builds with hipcc on
# gfx1201 regardless of the include paths, so these kernels must not advertise
# ROCm support.  The declaration is enforced by ``SUPPORTS_DEVICES`` below rather
# than by ``validate_once``: packing is a pure Torch operation that must keep
# working for checkpoint conversion/inspection even on a ROCm host, and only a
# *placement* on ROCm is unsupported.  ``ModelLoader`` raises this message at
# load time so a ROCm user gets an actionable error instead of the generic
# "No valid quant linear".
# ``AwqGEMVTritonLinear`` below runs the *same* packed GEMV layout with a
# portable Triton kernel (``quantization/awq/modules/triton/gemv.py``), so a
# decode-shaped GEMV checkpoint is usable on ROCm too.  The three native classes
# stay CUDA-only and keep raising the message above; a ROCm user can now pick
# the Triton backend explicitly (``BACKEND.AWQ_GEMV_TRITON``) and AUTO stays off
# it so no existing checkpoint silently changes kernel.
AWQ_GEMV_ROCM_UNSUPPORTED = (
    "AWQ GEMV/GEMV_FAST/LLM_AWQ kernels require the native CUDA AWQ extension, "
    "which is not available on ROCm (it includes cuda_fp16.h and CUDA-only inline "
    "asm, and its hipified warp-sync reduction rejects wave32). Use the AWQ GEMM "
    "layout with `backend=AUTO` or `BACKEND.AWQ_GEMM_TRITON`, which has a Triton "
    "ROCm kernel, instead of FORMAT.GEMV/GEMV_FAST/LLM_AWQ. For a GEMV-layout "
    "checkpoint, `BACKEND.AWQ_GEMV_TRITON` runs the same packed layout with a "
    "portable Triton kernel on ROCm."
)


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

    SUPPORTS_DEVICES = [DEVICE.CUDA]
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

        if self.bias is not None:
            # Preserve the FP16 kernel result and bias until after addition:
            # rounding either operand to BF16 first can destroy cancellation.
            out = out.float() + self.bias.float()
        out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor=None):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros.float() * scales.float()

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
            (linear.weight.data.float() + scale_zeros.index_select(1, group_index))
            / self.scales.index_select(1, group_index).float()
        ).clamp_(0, (1 << self.bits) - 1).to(torch.int32)
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

__all__ = ["AWQ_GEMV_ROCM_UNSUPPORTED", "AwqGEMVLinear"]


class AwqGEMVTritonLinear(AwqGEMVLinear):
    """AWQ GEMV layout with a portable Triton decode kernel.

    The packed buffers are produced by :meth:`AwqGEMVLinear.pack`, so this class
    reads exactly the checkpoint layout the native CUDA kernel reads; only the
    compute path differs.  Weights stay packed between forwards.

    Explicit selection only: ``SUPPORTS_FORMATS`` priority is 0, which keeps the
    FORMAT.GEMV AUTO order unchanged (CUDA keeps the native kernel there, and
    ROCm keeps refusing the native one with the actionable message above).
    """

    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMV_TRITON]
    SUPPORTS_METHODS = [METHOD.AWQ]
    # 0 => usable format, never auto-selected. See select_quant_linear().
    SUPPORTS_FORMATS = {FORMAT.GEMV: 0}
    SUPPORTS_BITS = [4]
    # Same packed layout as the native GEMV: only these normalized group sizes.
    SUPPORTS_GROUP_SIZE = [-1, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    # The group axis is padded to a whole int32 word, so N must be a multiple
    # of 8 for the zero extraction to address complete words.
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    QUANT_TYPE = "awq_gemv_triton"

    @classmethod
    def validate(cls, bits, group_size=-1, desc_act=False, sym=True, in_features=None,
                 out_features=None, pack_dtype=None, dtype=None, dynamic=None,
                 device=None, trainable=None, adapter=None):
        """Reuse the native GEMV contract, then add the Triton kernel limits."""

        ok, error = super().validate(
            bits=bits,
            group_size=group_size,
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
        if not ok:
            return ok, error
        if out_features is not None and out_features % 8:
            return False, NotImplementedError(
                "AwqGEMVTritonLinear requires out_features divisible by 8 "
                f"(packed group words): actual={out_features}"
            )
        return True, None

    def post_init(self):
        # One packed GEMV layout is read for every row count; the fused kernel is
        # only a decode win, so wider batches use the same kernel rather than a
        # dequantize + matmul (no dense weight is ever materialized).
        super().post_init()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        for name in ("qweight", "qzeros", "scales"):
            tensor = getattr(self, name, None)
            if tensor is not None and tensor.device != x.device:
                raise RuntimeError(
                    f"AwqGEMVTritonLinear input and {name} must be on the same device: "
                    f"input={x.device}, {name}={tensor.device}"
                )

        from ...quantization.awq.modules.triton.gemv import awq_gemv_triton

        rows = self.input_rows(x)
        inputs = x.reshape(rows, x.shape[-1])
        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            # The packed AWQ codebook is FP16; compute there and cast back.
            inputs = inputs.half()
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        group_size = self.in_features if self.group_size == -1 else self.group_size
        out = awq_gemv_triton(
            inputs,
            self.qweight,
            self.scales,
            self.qzeros,
            group_size,
            output_dtype=torch.float16,
        )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        if self.bias is not None:
            out = out + self.bias
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)
        return out.reshape(out_shape)


__all__ = [
    "AWQ_GEMV_ROCM_UNSUPPORTED",
    "AwqGEMVLinear",
    "AwqGEMVTritonLinear",
]
