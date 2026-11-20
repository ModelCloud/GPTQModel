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
from ...quantization.awq.utils.packing_utils import pack_awq_columns, source_weight_out_in
from ...utils.awq import awq_gemmv2_forward, awq_gemv_forward
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width
from ...utils.logger import setup_logger
from ...utils.packing_math import scale_to_storage_dtype
from ._compile_support import COMPILE_LEVEL
from ._numerics import fp32_epilogue_context


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
# ``FORMAT.GEMV_FAST``/``FORMAT.LLM_AWQ`` are excluded from this refusal: their
# portable Torch readers declare ROCm and read the identical interleaved layout
# (issue #5109).
AWQ_GEMV_ROCM_UNSUPPORTED = (
    "AWQ FORMAT.GEMV requires the native CUDA AWQ extension, which is not "
    "available on ROCm (it includes cuda_fp16.h and CUDA-only inline asm, and "
    "its hipified warp-sync reduction rejects wave32). Use the AWQ GEMM layout "
    "with `backend=AUTO` or `BACKEND.AWQ_GEMM_TRITON`, which has a Triton ROCm "
    "kernel, or load this GEMV checkpoint with `BACKEND.AWQ_GEMV_TRITON`, which "
    "runs the same packed layout with a portable Triton kernel on ROCm. "
    "`FORMAT.GEMV_FAST`/`FORMAT.LLM_AWQ` pack and load through their portable "
    "Torch readers here."
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
    SUPPORTS_TRAINING = False
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

    # The bias is added after the kernel's FP16 result by a wide epilogue when
    # the stored width differs, so it keeps its checkpoint dtype.
    MODEL_DTYPE_BUFFERS = ()

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemv"

    # Measured on sm_80 (A100-class, torch 2.15.0.dev20260928+cu130): eager and an explicit
    # ``torch.cuda.graph`` capture both run with matching output. The native
    # ``awq.gemv_forward`` op has no fake/meta kernel, so ``torch.compile``
    # stops at EAGER.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = True

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

        # A biased layer asks the kernel to keep the complete reduction (and
        # every split-K partial) in FP32. The bias then joins the wide sum and
        # the result is rounded once, instead of rounding the product first and
        # adding the bias to that already-rounded value.
        wide_bias = self.bias is not None
        if inputs.shape[0] > 8:
            out = awq_gemmv2_forward(
                inputs,
                self.qweight,
                self.scales,
                self.qzeros,
                self.group_size,
                self.split_k_iters,
                fp32_output=wide_bias,
            )
        else:
            out = awq_gemv_forward(
                inputs,
                self.qweight,
                self.scales,
                self.qzeros,
                self.group_size,
                fp32_output=wide_bias,
            )

        if self.bias is not None:
            # ``out`` is the wide kernel result; add the stored bias
            # in place so only one allocation (the final cast) remains.
            out.add_(self.bias)
        out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor=None):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None

        pack_num = 32 // self.bits
        qscales = torch.zeros(
            (
                scales.shape[0],
                calculate_zeros_width(self.in_features, self.group_size) * pack_num,
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        # Reject an unrepresentable FP16 scale before replacing module state.
        qscales[:, : scales.shape[1]] = scale_to_storage_dtype(scales, qscales.dtype)
        stored_scales = qscales[:, : scales.shape[1]]
        # The checkpoint decodes with the stored FP16 scale, so build the
        # affine offset from that same grid. Mixing the FP32 source scale into
        # the offset shifts the zero point whenever the cast rounds (#5605).
        scale_zeros = zeros.float() * stored_scales.float()
        stored_bias = None if linear.bias is None else linear.bias.detach().clone()
        self.register_buffer("scales", qscales)
        if stored_bias is not None:
            # The bias is not a quantization operand. Store it at the source
            # width: ``forward`` adds it in a wide FP32 epilogue, so narrowing
            # it to FP16 here could only turn a finite value into ``inf``.
            self.register_buffer("bias", stored_bias)
        else:
            self.bias = None

        # Gather each input column's group scale/zero in a single pass instead
        # of launching one tiny kernel per input column.  The GEMV layout keeps
        # scales as [out_features, groups], so the group index selects columns.
        weight = source_weight_out_in(linear)
        group_index = torch.arange(self.in_features, device=weight.device) // self.group_size
        intweight = torch.round(
            (weight.float() + scale_zeros.index_select(1, group_index))
            / stored_scales.index_select(1, group_index).float()
        ).clamp_(0, (1 << self.bits) - 1).to(torch.int32)
        self.register_buffer("qweight", pack_awq_columns(intweight, self.bits))

        zeros = zeros.to(dtype=torch.int32)
        # Pad the group axis up to the packed width so the vectorized packer
        # reproduces the loop's out-of-range skips (those codes stay zero).
        padded_zeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(self.in_features, self.group_size) * pack_num),
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


class _AwqGemvTritonFn(torch.autograd.Function):
    """Differentiate the packed GEMV without retaining a dense weight."""

    @staticmethod
    def forward(ctx, inputs, qweight, scales, qzeros, group_size, output_dtype):
        from ...quantization.awq.modules.triton.gemv import awq_gemv_triton

        ctx.save_for_backward(qweight, scales, qzeros)
        ctx.group_size = group_size
        ctx.input_dtype = inputs.dtype
        return awq_gemv_triton(
            inputs, qweight, scales, qzeros, group_size, output_dtype=output_dtype,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        qweight, scales, qzeros = ctx.saved_tensors
        n, words = qweight.shape
        k = words * 8
        shifts = torch.arange(8, device=qweight.device, dtype=torch.int32) * 4
        groups = torch.arange(k, device=qweight.device) // ctx.group_size
        grad_input = torch.zeros(
            (grad_output.shape[0], k), device=grad_output.device, dtype=torch.float32,
        )
        # Unpack bounded output-column tiles only for backward. The temporary
        # dense tile is released each iteration; no expanded weight survives.
        for start in range(0, n, 128):
            stop = min(start + 128, n)
            codes = ((qweight[start:stop, :, None] >> shifts) & 15).reshape(stop - start, k)
            zeros = ((qzeros[start:stop, :, None] >> shifts) & 15).reshape(stop - start, -1)
            zeros = zeros.index_select(1, groups)
            weights = (codes - zeros).float() * scales[start:stop].index_select(1, groups).float()
            grad_input.addmm_(grad_output[:, start:stop].float(), weights)
        return grad_input.to(ctx.input_dtype), None, None, None, None, None


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

    # Measured on NVIDIA GB10 (sm_121), torch 2.11.0+cu130: dynamo, aot_eager,
    # inductor and reduce-overhead all pass, and an explicit CUDA-graph capture
    # of the eager forward succeeds.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.CUDAGRAPH
    SUPPORTS_GRAPH_CAPTURE = True
    # Only reachable through an explicit `BACKEND.AWQ_GEMV_TRITON` selection
    # (FORMAT priority 0), and no static-cache `generate()` win has been
    # measured for it, so the #4669 policy stays off (#4984).
    SUPPORTS_COMPILE = False
    SUPPORTS_COMPILE_EAGER_REASON = (
        "explicit-selection-only GEMV decode kernel; no static-cache generate "
        "win measured (#4984)"
    )

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
            out = self.empty_linear_output(x)
            if self.bias is not None and self.bias.requires_grad:
                out = (out.float() + self.bias.float()).to(x.dtype)
            return out

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
        output_dtype = torch.float32 if self.bias is not None else torch.float16
        if torch.is_grad_enabled() and inputs.requires_grad:
            out = _AwqGemvTritonFn.apply(
                inputs, self.qweight, self.scales, self.qzeros, group_size, output_dtype,
            )
        else:
            out = awq_gemv_triton(
                inputs, self.qweight, self.scales, self.qzeros, group_size,
                output_dtype=output_dtype,
            )
        if self.bias is not None:
            # ``out`` is the wide kernel result; add the stored bias
            # in place so only one allocation (the final cast) remains.
            out.add_(self.bias)
        if out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)
        return out.reshape(out_shape)


class AwqGEMVTorchLinear(AwqGEMVLinear):
    """Portable CPU reader for the packed ``FORMAT.GEMV`` layout (issue #4554).

    The native GEMV extension is CUDA-only and the Triton reader needs an
    accelerator, so a quantize run that finishes on CPU had no in-session kernel
    for a GEMV checkpoint and ``quantize()`` raised after the whole calibration.
    This kernel reads the exact same packed ``qweight``/``qzeros``/``scales``
    buffers and dequantizes them to a dense FP32 tile per forward, which keeps
    the checkpoint format unchanged and makes the in-session model runnable.

    It is a documented fallback, not an AUTO replacement: the priority-0
    ``SUPPORTS_FORMATS`` entry keeps the accelerator AUTO order (native CUDA
    GEMV, explicit Triton GEMV) untouched, and ``select_quant_linear`` only
    reaches it for a CPU-only target through ``PORTABLE_CPU_FALLBACK`` (ROCm
    uses the Triton reader, which keeps the weights packed).
    """

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH]
    SUPPORTS_METHODS = [METHOD.AWQ]
    # 0 => supported for this layout, never auto-selected on an accelerator.
    SUPPORTS_FORMATS = {FORMAT.GEMV: 0}
    SUPPORTS_BITS = [4]
    # Same packed layout as the native GEMV, so a CPU-packed checkpoint keeps
    # the contract the CUDA extension reads.
    SUPPORTS_GROUP_SIZE = [-1, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16]
    QUANT_TYPE = "awq_gemv_torch"
    # Consumed by select_quant_linear() when AUTO has no CPU kernel for GEMV.
    PORTABLE_CPU_FALLBACK = True

    # CPU-only kernel (`SUPPORTS_DEVICES` above), so the CPU measurement is the
    # declaration: on torch 2.16.0.dev20261021+cpu eager=True, graph_breaks=0,
    # and dynamo/aot_eager/inductor all pass. CUDA-graph capture is CUDA-only,
    # so it is not a CPU level.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_GRAPH_CAPTURE = True
    # CPU-only portable fallback for the packed GEMV layout: compiling a CPU
    # static-cache decode has no measured win, so the #4669 policy stays off
    # (#4984).
    SUPPORTS_COMPILE = False
    SUPPORTS_COMPILE_EAGER_REASON = (
        "CPU-only portable GEMV fallback; no measured static-cache win (#4984)"
    )

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.AWQ_TORCH)
        super().__init__(*args, **kwargs)

    def _dequantize_weight(self) -> torch.Tensor:
        """Return the packed GEMV words as a dense ``[in_features, out_features]``.

        Nibble ``i`` of word ``w`` holds input column ``w * pack_factor + i``;
        the group axis of ``qzeros``/``scales`` is padded to whole words, so the
        real group index selects the column. The product is formed in FP32, the
        same precision the native/Triton kernels accumulate in.
        """

        qweight = self.qweight
        qzeros = self.qzeros
        if qweight.dtype != torch.int32:
            qweight = qweight.to(torch.int32)
        if qzeros.dtype != torch.int32:
            qzeros = qzeros.to(torch.int32)

        out_features, words = qweight.shape
        in_features = words * self.pack_factor
        shifts = torch.arange(self.pack_factor, device=qweight.device, dtype=torch.int32) * self.bits
        code_mask = (1 << self.bits) - 1
        codes = ((qweight[:, :, None] >> shifts) & code_mask).reshape(out_features, in_features)
        zeros = ((qzeros[:, :, None] >> shifts) & code_mask).reshape(out_features, -1)

        groups = in_features // self.group_size
        group_index = torch.arange(in_features, device=qweight.device) // self.group_size
        zeros = zeros.index_select(1, group_index)
        scales = self.scales[:, :groups].index_select(1, group_index)
        weight = (codes.to(torch.float32) - zeros.to(torch.float32)) * scales.to(torch.float32)
        return weight.t().contiguous()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        inputs = x.reshape(-1, x.shape[-1])
        input_dtype = inputs.dtype

        weight = self._dequantize_weight()
        if weight.device != inputs.device:
            weight = weight.to(device=inputs.device)

        # One wide reduction and a single rounding at the end: the FP32
        # accumulator is what the accelerator GEMV kernels use, so the portable
        # path is a reference for them rather than a lower-precision shortcut.
        # An ambient autocast would narrow this explicit FP32 matmul and round
        # the product before the bias cancels it (#5613), so the epilogue runs
        # outside it and the caller's context is restored afterwards.
        with fp32_epilogue_context(inputs.device.type):
            output = torch.matmul(inputs.to(torch.float32), weight)
            if self.bias is not None:
                output = output + self.bias.to(device=output.device, dtype=torch.float32)
            output = output.to(dtype=input_dtype)

        if self.adapter:
            output = self.adapter.apply(x=inputs, out=output)

        return output.reshape(out_shape)


__all__ = [
    "AWQ_GEMV_ROCM_UNSUPPORTED",
    "AwqGEMVLinear",
    "AwqGEMVTritonLinear",
    "AwqGEMVTorchLinear",
]
