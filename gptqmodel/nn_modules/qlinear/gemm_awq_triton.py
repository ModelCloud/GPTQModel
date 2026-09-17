# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from contextlib import nullcontext
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear, empty_linear_output, input_rows
from ...quantization import FORMAT, METHOD
from ...quantization.awq.modules.triton.scheduler import (
    AwqTritonPlan,
    candidate_plans,
    clear_awq_triton_plan_cache,
    legacy_plan,
    mark_awq_triton_plan_warmed,
    select_awq_triton_plan,
    validate_fused_config,
)
from ...utils import has_gil_disabled
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.torch import HAS_XPU


# Shared runtime default: prefer accuracy first unless the user explicitly opts out.
FP32_ACCUM = env_flag("GPTQMODEL_FP32_ACCUM", default=True)


def _cuda_graph_capturing(device: torch.device) -> bool:
    """Query capture state without synchronization (and tolerate mock devices)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


class AwqGemmTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
        prefer_backend=None,
    ):
        # Only quantized weights and input metadata are needed for the
        # input-gradient backward; input and bias values are not retained.
        ctx.save_for_backward(qweight, qzeros, scales)
        ctx.input_shape = tuple(x.shape)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        function_input_dtype = x.dtype
        x = x.to(torch.float16)
        rows = input_rows(x)
        ctx.input_rows = rows
        if rows == 0:
            # Triton also requires a positive M dimension, so use the shared
            # empty path and avoid compiling/launching a zero-row kernel.
            return empty_linear_output(x, out_features)

        from ...quantization.awq.modules.triton.gemm import awq_dequantize_triton, awq_gemm_triton
        # Triton consumes [rows, features]; restore leading dimensions later.
        x_2d = x.reshape(rows, x.shape[-1])
        K = x_2d.shape[1]
        if qweight.ndim != 2 or qweight.shape[0] != K or qweight.shape[1] <= 0:
            raise ValueError("AWQ Triton qweight must be rank-2 with qweight.shape[0] == K")
        N = qweight.shape[1] * 8
        request = prefer_backend if isinstance(prefer_backend, dict) else {}
        fp32_accum = bool(request.get("fp32_accum", FP32_ACCUM))
        if qzeros.ndim != 2 or scales.ndim != 2 or qzeros.shape[0] <= 0:
            raise ValueError("AWQ Triton qzeros/scales must be non-empty rank-2 tensors")
        if K % qzeros.shape[0] or scales.shape[0] != qzeros.shape[0]:
            raise ValueError("AWQ Triton qzeros/scales group rows must divide K and match")
        if qzeros.shape[1] != qweight.shape[1] or scales.shape[1] != N:
            raise ValueError("AWQ Triton qweight/qzeros/scales output shapes must match")
        actual_group_size = K // qzeros.shape[0]
        # Check the public group-size declaration against the packed buffers
        # before asking the scheduler or kernel to interpret them.
        declared_group_size = K if group_size == -1 else group_size
        if declared_group_size != actual_group_size:
            raise ValueError(
                "AWQ Triton declared group_size does not match quantized buffers: "
                f"declared={declared_group_size}, actual={actual_group_size}"
            )
        original_input_dtype = request.get("input_dtype", function_input_dtype)
        final_output_dtype = request.get("output_dtype", original_input_dtype)
        capturing = _cuda_graph_capturing(x.device)
        plan = select_awq_triton_plan(
            M=rows, N=N, K=K, group_size=actual_group_size,
            device=x.device, input_dtype=original_input_dtype,
            compute_dtype=x.dtype, output_dtype=final_output_dtype,
            fp32_accum=fp32_accum, mode=request.get("schedule_mode"),
            explicit=request.get("schedule"), training=bool(request.get("training", False)),
            cuda_graph=capturing,
        )
        if not plan.fused:
            out = awq_dequantize_triton(qweight, scales, qzeros)
            out = torch.matmul(x_2d, out.to(x.dtype))
        else:
            out = awq_gemm_triton(
                x_2d,
                qweight,
                scales,
                qzeros,
                fp32_accum=fp32_accum,
                output_dtype=x.dtype,
                **plan.as_kwargs(),
            )

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)
        if not capturing:
            mark_awq_triton_plan_warmed(
                plan, M=rows, N=N, K=K, group_size=actual_group_size,
                device=x.device, input_dtype=original_input_dtype,
                compute_dtype=x.dtype, output_dtype=final_output_dtype,
                fp32_accum=fp32_accum,
                mode=request.get("schedule_mode"),
                explicit=request.get("schedule"),
                training=bool(request.get("training", False)),
            )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        qweight, qzeros, scales = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            if ctx.input_rows == 0:
                grad_input = grad_output.new_empty(ctx.input_shape)
                return grad_input, None, None, None, None, None, None, None, None
            from ...quantization.awq.modules.triton.gemm import awq_dequantize_triton

            weights = awq_dequantize_triton(qweight, scales, qzeros).to(grad_output.dtype)
            # Mirror forward's flatten/restore so backward supports every rank.
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.matmul(grad_output_2d, weights.transpose(-1, -2))
            grad_input = grad_input.reshape(ctx.input_shape)

        return grad_input, None, None, None, None, None, None, None, None


class AwqGEMMTritonLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMM_TRITON]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 50}
    SUPPORTS_BITS = [4]
    # The Triton kernel broadcasts one scale/zero per BK tile and has only
    # been validated for these packing group sizes.  G=16 is module metadata
    # from a different AWQ backend and must not reach this kernel.
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    # BK is at least 32 and every tile boundary must stay group-aligned.
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    # TODO: ROCM also has Triton support. Need to validate ROCM for triton
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "awq_gemm_triton"

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        from packaging import version
        from triton import __version__ as triton_version
        triton_v = version.parse(triton_version)

        if triton_v < version.parse("2.0.0"):
            raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")

        # GIL=0 is tested with Triton 3.4.0 and it works
        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            raise Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        return True, None

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
        fp32_accum = bool(kwargs.pop("fp32_accum", FP32_ACCUM))
        schedule_mode = kwargs.pop(
            "schedule_mode", kwargs.pop("awq_triton_schedule_mode", None)
        )
        schedule = kwargs.pop(
            "schedule", kwargs.pop("awq_triton_schedule", None)
        )
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_GEMM_TRITON),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)
        self.fp32_accum = fp32_accum
        self.awq_triton_schedule_mode = schedule_mode
        self.awq_triton_schedule = schedule

    def post_init(self):
        if self.scales is not None:
            self.scales = self.scales.to(dtype=torch.float16)
        super().post_init()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)

        # Quantized tensors are passed as raw device pointers to Triton; do
        # this check before entering its device context.
        for name in ("qweight", "qzeros", "scales"):
            tensor = getattr(self, name, None)
            if tensor is not None and tensor.device != x.device:
                raise RuntimeError(
                    f"AWQ Triton input and {name} must be on the same device: "
                    f"input={x.device}, {name}={tensor.device}"
                )

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        if not x.is_contiguous():
            x = x.contiguous()

        # Select from x.device instead of whichever accelerator is globally available.
        device_context = (
            torch.xpu.device(x.device)
            if x.device.type == "xpu" and HAS_XPU
            else torch.cuda.device(x.device)
            if x.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            with nullcontext() if self.training else torch.inference_mode():
                out = AwqGemmTritonFn.apply(
                    x,
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.bits,
                    self.group_size,
                    self.bias,
                    self.out_features,
                    {
                        "fp32_accum": getattr(self, "fp32_accum", FP32_ACCUM),
                        "schedule_mode": getattr(self, "awq_triton_schedule_mode", None),
                        "schedule": getattr(self, "awq_triton_schedule", None),
                        "training": self.training,
                        "input_dtype": input_dtype,
                        "output_dtype": input_dtype,
                    },
                )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)


__all__ = [
    "AwqGemmTritonFn",
    "AwqGEMMTritonLinear",
    "AwqTritonPlan",
    "candidate_plans",
    "clear_awq_triton_plan_cache",
    "legacy_plan",
    "mark_awq_triton_plan_warmed",
    "select_awq_triton_plan",
    "validate_fused_config",
]
