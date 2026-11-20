# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...extension import op as extension_op
from ...extension import register_fake
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from ...utils.logger import setup_logger
from ...utils.torch import dynamo_is_compiling
from . import AWQuantLinear, detach_cpu_operand
from ._awq_decode import AwqDecodeRangeMixin
from ._compile_support import COMPILE_LEVEL
from ._fused_train_state import (
    capture_fused_train_state,
    ensure_cpu_inference_layout,
    finalize_fused_train_state,
    fused_autograd_needs_dense,
    refinalize_fused_state_after_apply,
)
from ._numerics import matmul_fp32_reduce
from .torch_aten_kernel import TorchAtenLinear
from .torch_cpu_avx512 import _fake_gemm_int4_qzero_group_scaled, unpack_hf_int4


log = setup_logger()


class TorchCpuAvx512AwqLinear(AwqDecodeRangeMixin, AWQuantLinear):
    """AWQ W4A16 adapter for the shared AVX512-BF16 int4 GEMM."""

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH_CPU_AVX512]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 120}

    # ``transform_cpu`` widens scales *and* bias to FP32 in place, so both must
    # be restored before a checkpoint load or save.
    _FUSED_TRAIN_STATE_NAMES = ("qweight", "qzeros", "scales", "bias")
    # ``pack_awq_int4`` is a pure nibble permutation whose inverse is
    # ``rebuild_avx512_awq_qweight`` (pure Torch), so the checkpoint ``qweight``
    # is rebuilt on save instead of kept resident (#5013).
    _FUSED_TRAIN_STATE_REBUILD = "avx512_awq"
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [16, 32, 64, 128, 256, 512]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    # Training mode uses the dequantized matmul fallback in forward().
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [2]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = TorchAtenLinear.SUPPORTS_ADAPTERS
    # Shared with the BF16-only AVX512-BF16 GPTQ kernel. FP16 AWQ checkpoints
    # use the native ATen FP16 int4pack backend.
    SUPPORTS_DTYPES = [torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "torch_cpu_avx512_awq"

    # Measured on an x86-64 AVX512-BF16 host, torch 2.16.0.dev20261021+cpu:
    # eager=True, graph_breaks=0, dynamo/aot_eager/inductor all reach it. The
    # `_cpu_transform_lock` RLock around the one-time layout rewrite is skipped
    # while tracing. CPU-only, so no CUDA-graph capture applies.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for this kernel in this change; the INDUCTOR level is opt-in via "
        "optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = False

    gemm_int4_qzero_group_scaled_cpu_kernel = None
    pack_awq_int4_kernel = None

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
        adapter=None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.AWQ_TORCH_CPU_AVX512)
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
            register_buffers=register_buffers,
            **kwargs,
        )
        self._cpu_transformed = False
        self.linear_mode = None
        self._refresh_decode_range_fits()

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not has_avx512_bf16():
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            cls.pack_awq_int4_kernel = None
            err = RuntimeError(avx512_bf16_error())
            log.warning(str(err))
            return False, err

        try:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = extension_op(
                "int4_qzero_group_scaled_cpu", "gemm_int4_qzero_group_scaled_cpu_forward"
            )
            register_fake(
                "int4_qzero_group_scaled_cpu",
                "gemm_int4_qzero_group_scaled_cpu_forward",
                _fake_gemm_int4_qzero_group_scaled,
            )
            cls.pack_awq_int4_kernel = extension_op("int4_qzero_group_scaled_cpu", "pack_awq_int4")
        except Exception as exc:
            cls.gemm_int4_qzero_group_scaled_cpu_kernel = None
            cls.pack_awq_int4_kernel = None
            return False, exc
        return True, None

    def _cpu_fused_available(self) -> bool:
        if self.gemm_int4_qzero_group_scaled_cpu_kernel is not None:
            return True
        ok, _ = type(self).cached_validate_once()
        return ok

    def _fused_input_dtype(self, x: torch.Tensor) -> bool:
        return x.dtype == torch.bfloat16

    def _prepare_cpu_inference_layout(self, dtype) -> bool:
        """Install the AVX512 AWQ runtime layout outside a traced graph (#5515).

        This kernel's eager branch uses a device-only ``transform()`` and a
        string ``linear_mode``, so it cannot share the generic sequence.
        """

        if self.linear_mode is not None or self.training:
            return False
        if dtype != torch.bfloat16 or not self._cpu_fused_available():
            return False
        qweight = getattr(self, "qweight", None)
        if not isinstance(qweight, torch.Tensor) or qweight.device.type != "cpu":
            return False
        with self._cpu_transform_lock:
            if self.linear_mode is not None:
                return False
            capture_fused_train_state(self)
            self.transform("cpu")
            self.linear_mode = "inference"
        return True

    def post_init(self):
        # The decode range is read by a compiled graph as a Python constant, so
        # it must be computed before the first traced forward.
        self._refresh_decode_range_fits()
        super().post_init()

    def train(self, mode: bool = True):
        # forward() caches the fused-vs-dequantized dispatch decision. Reset it
        # on mode changes so an eval forward cannot pin later training forwards
        # to the no-grad fused kernel.
        if mode != self.training:
            self.linear_mode = None
        return super().train(mode=mode)

    def transform_cpu(self):
        if self._cpu_transformed:
            return
        if getattr(self, "_fused_train_state", None) is None:
            # A direct ``transform_cpu`` (outside ``forward``) still needs the
            # checkpoint snapshot so a later ``meta``/``to_empty`` move can
            # restore the checkpoint layout instead of the runtime one.
            capture_fused_train_state(self)

        self._dequant_scales_dtype = self.scales.dtype
        if self.pack_awq_int4_kernel is None:
            ok, err = type(self).cached_validate_once()
            if not ok:
                raise RuntimeError(str(err))

        # Keep the checkpoint scales in FP32. The shared AVX512 kernel uses
        # BF16 activations and weights with FP32 accumulation; rounding scales
        # to BF16 first would consume most of the inference error budget.
        # ``to`` returns the same tensor when the dtype already matches, which
        # would leave the operand as a view of the checkpoint mapping (#4785).
        scales = self.scales.to(torch.float32)
        if scales.data_ptr() == self.scales.data_ptr():
            scales = detach_cpu_operand(scales)
        self.scales = scales.contiguous()
        if self.bias is not None:
            bias = self.bias.to(torch.float32)
            if bias.data_ptr() == self.bias.data_ptr():
                bias = detach_cpu_operand(bias)
            self.bias = bias.contiguous()
        self.qweight, self.qzeros = self.pack_awq_int4_kernel(
            self.qweight,
            self.qzeros,
            self.out_features,
            self.in_features,
        )
        # ``g_idx`` still aliases the checkpoint mapping and would pin the whole
        # shard even though ``qweight``/``qzeros`` are now private (#4785).
        self.g_idx = detach_cpu_operand(getattr(self, "g_idx", None))
        self._cpu_transformed = True
        # ``qweight`` now holds the extension's packed nibbles: drop the
        # checkpoint clone if the pure-Torch inverse reproduces exactly the
        # bytes captured before the pack (#5013).
        finalize_fused_train_state(self)

    def _apply(self, fn, recurse: bool = True):
        """Keep the FP32 affine operands exact across dtype/device moves.

        ``transform_cpu`` packs the weight once and promotes ``scales``/``bias``
        to FP32 because the shared AVX512 kernel requires an FP32 ``absmax`` and
        accumulates in FP32. ``nn.Module._apply`` applies ``fn`` to every
        registered floating buffer, so ``model.to(torch.bfloat16)``/``.float()``/
        ``.half()`` used to round those operands through BF16 before restoring
        the dtype. BF16 keeps 8 explicit mantissa bits against the checkpoint's
        10, so the round trip lost scale precision permanently and the next
        forward raised ``absmax must be float32`` once the dtype was restored.

        Hold the exact tensors across ``super()._apply`` and reinstall them at
        the resulting device. Scales and zero points are quantization metadata,
        not expanded weights, so retaining them is allowed; the packed integer
        codes are untouched by a move and are never repacked here.

        A device or low-precision move invalidates the packed runtime layout
        itself. In that case the checkpoint snapshot (whose buffers ``fn``
        moved as well) is restored and the next forward re-transforms. This
        also keeps a ``meta`` move from retaining meta scales that
        ``to_empty`` cannot copy.
        """

        if not getattr(self, "_cpu_transformed", False):
            return super()._apply(fn, recurse=recurse)

        scales = getattr(self, "scales", None)
        previous_dtype = getattr(self, "_dequant_scales_dtype", None)
        if previous_dtype is None and isinstance(scales, torch.Tensor):
            previous_dtype = scales.dtype
        previous_device = scales.device if isinstance(scales, torch.Tensor) else None

        kept = {
            name: getattr(self, name)
            for name in ("scales", "bias")
            if isinstance(getattr(self, name, None), torch.Tensor)
        }
        result = super()._apply(fn, recurse=recurse)
        if refinalize_fused_state_after_apply(self, previous_dtype, previous_device):
            # A device or low-precision move invalidated the runtime layout.
            # The snapshot buffers moved with ``fn``, so the module is back on
            # the checkpoint layout and the next forward re-transforms. Never
            # reinstall the pre-move meta tensors: ``to_empty`` would try to
            # copy out of them.
            self._refresh_decode_range_fits()
            return result
        for name, value in kept.items():
            current = getattr(self, name, None)
            device = current.device if isinstance(current, torch.Tensor) else value.device
            restored = value.to(device=device, dtype=torch.float32).contiguous()
            if name in self._parameters:
                restored = torch.nn.Parameter(restored, requires_grad=False)
            setattr(self, name, restored)
        self._refresh_decode_range_fits()
        return result

    def clear_weight_cache(self) -> None:
        """Restore the FP32 kernel ABI after a load or explicit cache clear.

        ``_apply`` above preserves the operands exactly, but a checkpoint load
        or an external caller may still install a narrower ``scales``/``bias``.
        The packed codes are already in the kernel layout, so only the affine
        dtype needs re-establishing. The checkpoint snapshot is not restored
        here; only an explicit checkpoint load, export, or training resume does
        that.
        """

        self._refresh_decode_range_fits()
        if not getattr(self, "_cpu_transformed", False):
            return
        if self.scales.dtype != torch.float32:
            self.scales = self.scales.to(torch.float32).contiguous()
        if self.bias is not None and self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32).contiguous()

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
        else:
            raise NotImplementedError(
                "TorchCpuAvx512AwqLinear only supports fused transforms on CPU devices."
            )

    def dequantize_weight(self, num_itr: int = 1, dtype: torch.dtype | None = None):
        if not self._cpu_transformed:
            # ``dequantize_gemm`` multiplies the integer codes by the scales,
            # so a half-precision decode can overflow before the result is
            # widened. Convert the temporary scales first; the checkpoint
            # buffer keeps its loaded dtype.
            scales = self.scales
            if dtype is not None and scales.dtype != dtype:
                scales = scales.to(dtype=dtype)
            weight = dequantize_gemm(
                qweight=self.qweight,
                qzeros=self.qzeros,
                scales=scales,
                bits=self.bits,
                group_size=self.group_size,
            )
            return weight if dtype is None else weight.to(dtype=dtype)
        if num_itr != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__}.dequantize_weight only supports num_itr=1"
            )

        codes = unpack_hf_int4(
            self.qweight,
            out_features=self.out_features,
            in_features=self.in_features,
        )
        group_size = max(int(self.group_size), 1)
        groups = torch.arange(self.in_features, device=codes.device) // group_size
        zeros = self.qzeros.index_select(0, groups)
        scales = self.scales.index_select(0, groups)
        weight = (codes.t().float() - zeros.float()) * scales.float()
        return weight.to(self._dequant_scales_dtype if dtype is None else dtype)

    def awq_weight_dequantize(self, device, dtype):
        # The dense fallback's natural decode dtype is the checkpoint scale
        # dtype (or the transformed layout's recorded width). Widen only when
        # that dtype cannot hold the decoded weight, then cast to the activation
        # dtype as before.
        natural = self._dequant_scales_dtype if self._cpu_transformed else self.scales.dtype
        decode_dtype = self._decode_dtype_for_activation(natural)
        if decode_dtype == natural:
            weight = self.dequantize_weight()
        else:
            weight = self.dequantize_weight(dtype=decode_dtype)
        return weight.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _fused_op_forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        if x.device.type != "cpu":
            raise NotImplementedError

        if not self._fused_input_dtype(x):
            raise RuntimeError(
                f"{self.__class__.__name__} requires bfloat16 activations, got {x.dtype}."
            )
        zeros = self.qzeros if self.qzeros.is_contiguous() else self.qzeros.contiguous()
        scales = self.scales if self.scales.is_contiguous() else self.scales.contiguous()
        out = self.gemm_int4_qzero_group_scaled_cpu_kernel(
            x,
            self.qweight,
            zeros,
            scales,
            self.group_size,
            self.bias,
        )
        return out

    def _dense_weight_for_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Dense weight for a call that cannot use the fused op.

        ``awq_weight_dequantize`` decodes the live runtime layout when the
        module is transformed and the checkpoint layout otherwise, without
        restoring or rewriting either (#5509).
        """
        return self.awq_weight_dequantize(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        dense_override = False
        dense_weight = None
        if dynamo_is_compiling():
            # ``_cpu_transform_lock`` is an ``RLock``: dynamo cannot enter it (a
            # hard graph break), and the one-time ``transform()`` rewrites the
            # packed ``qweight``/``qzeros`` buffers in place, which a traced
            # graph must not do. The transform runs on the first eager forward
            # and in ``optimize()``'s warmup, so the graph only reads the
            # resolved ``linear_mode``.
            # A grad call must not take the fused op; decode instead (#5369, #5569).
            dense_override = fused_autograd_needs_dense(self, x_flat)
            if (
                self.linear_mode is None
                and not self.training
                and not x_flat.requires_grad
                and not fused_autograd_needs_dense(self, x_flat)
                and self._cpu_fused_available()
                and x_flat.device.type == "cpu"
                and self._fused_input_dtype(x_flat)
            ):
                # A cold trace must not bake in the dense branch: install the
                # runtime layout here and retrace into the fused path (#5515).
                ensure_cpu_inference_layout(self, x_flat.dtype)
            linear_mode = self.linear_mode
        else:
            with self._cpu_transform_lock:
                force_dense = fused_autograd_needs_dense(self, x_flat)
                if force_dense:
                    # The fused op has no backward, so this call needs a dense
                    # weight. Decode the live runtime layout instead of
                    # restoring the checkpoint layout: the restore rewrites the
                    # packed buffers a concurrent no-grad forward is reading,
                    # so alternating grad/no-grad calls raced (#5509).
                    dense_override = True
                    dense_weight = self._dense_weight_for_activation(x_flat)
                elif (
                    self.linear_mode is None
                    and not self.training
                    and x_flat.requires_grad
                ):
                    # A concurrent first no-grad forward may install the runtime
                    # layout while this call decodes the checkpoint layout.
                    # Snapshot the weight under the lock so that transform
                    # cannot swap the buffers mid-read (#5509).
                    dense_weight = self._dense_weight_for_activation(x_flat)
                if not force_dense and (
                    not self.training
                    and not x_flat.requires_grad
                    and self.linear_mode is None
                    and self._cpu_fused_available()
                    and x_flat.device.type == "cpu"
                    and self._fused_input_dtype(x_flat)
                ):
                    capture_fused_train_state(self)
                    self.transform("cpu")
                    self.linear_mode = "inference"
                elif not force_dense and self.linear_mode is None and self.training:
                    # Only a training-mode forward pins the dense path until
                    # `eval()` clears the latch. An eval-mode grad forward takes
                    # the dense path for this call only: latching here made every
                    # later no-grad forward dequantize the weight (#5392).
                    self.linear_mode = "train"
            linear_mode = self.linear_mode

        if linear_mode == "inference" and not dense_override:
            if not self._fused_input_dtype(x_flat):
                raise RuntimeError(
                    f"{self.__class__.__name__} was transformed for bfloat16 "
                    f"but received {x_flat.dtype} activations."
                )
            out = self._fused_op_forward(x_flat).reshape(out_shape)
        else:
            if dense_weight is None:
                dense_weight = self._dense_weight_for_activation(x_flat)
            weight = dense_weight
            if self.bias is not None:
                # transform_cpu() keeps the fused kernel's bias in FP32. Keep
                # the product and the bias in FP32 so a cancelling bias is
                # applied before the single output cast.
                out = matmul_fp32_reduce(x_flat, weight, self.bias).reshape(out_shape)
            else:
                out = torch.matmul(x_flat, weight).reshape(out_shape)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["TorchCpuAvx512AwqLinear"]
