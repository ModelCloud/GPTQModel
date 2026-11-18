# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter
from ...models._const import DEVICE
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear, detach_cpu_operand
from ._compile_support import COMPILE_LEVEL
from ._fused_train_state import (
    bind_runtime_qzeros,
    capture_fused_train_state,
    finalize_fused_train_state,
    refinalize_fused_state_after_apply,
    restore_fused_train_state,
)
from ._numerics import matmul_fp32_reduce
from .torch_aten_kernel import (
    TorchAtenLinear,
    _cpu_int4pack_scale_dtype,
    _cpu_int4pack_zero_offsets,
    _has_local_int4pack_cpu_ops,
)
from .torch_fused import (
    _cpu_int4pack_mm,
    _validate_cpu_int4pack_group_size,
    _validate_cpu_int4pack_out_features,
    pack_scales_and_zeros,
)


log = setup_logger()


class TorchAtenAwqLinear(AWQuantLinear):
    """AWQ CPU int4pack backend implemented with local ATen ops."""

    QUANT_TYPE = "awq_torch_aten_kernel"

    # Measured on CPU, torch 2.16.0.dev20261021+cpu: eager=True,
    # graph_breaks=0, dynamo/aot_eager/inductor all reach it for an unbiased
    # layer. The `_cpu_transform_lock` RLock around the one-time int4pack
    # rewrite is skipped while tracing, and `_refresh_fused_state` is
    # eager-only now. A biased FP16/BF16 layer needs an FP32 product, which
    # inductor cannot lower through a fused operand cast (#4068). CPU-only, so
    # no CUDA graph capture applies.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for this kernel in this change; the INDUCTOR level is opt-in via "
        "optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = False

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH_ATEN]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 110}

    SUPPORTS_BITS = TorchAtenLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = TorchAtenLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = TorchAtenLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = TorchAtenLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = TorchAtenLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = TorchAtenLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = TorchAtenLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = TorchAtenLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = TorchAtenLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = TorchAtenLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = TorchAtenLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = TorchAtenLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = TorchAtenLinear.SUPPORTS_ADAPTERS
    REQUIRES_FORMAT_V2 = False

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    gemm_int4_forward_kernel = None

    # ``transform_cpu`` keeps ``scales`` in the checkpoint dtype and layout, so
    # only the rewritten qweight/qzeros need a checkpoint snapshot.
    _FUSED_TRAIN_STATE_NAMES = ("qweight", "qzeros")
    # The ATen CPU int4pack permutation has a verified inverse, so the
    # checkpoint qweight is rebuilt on save instead of kept resident.
    _FUSED_TRAIN_STATE_REBUILD = "awq"

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
        kwargs.setdefault("backend", BACKEND.AWQ_TORCH_ATEN)
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
            register_buffers=False,
            **kwargs,
        )

        self.linear_mode = None
        self._cpu_transformed = False
        self._cpu_transform_dtype = None

        if register_buffers:
            pack_cols = max(1, self.out_features // self.pack_factor)
            qweight_shape = (self.in_features, pack_cols)
            group_size = max(int(self.group_size), 1)
            group_rows = max(1, math.ceil(self.in_features / group_size))

            self.register_buffer(
                "qweight",
                torch.zeros(qweight_shape, dtype=self.pack_dtype),
            )

            self.register_buffer(
                "qzeros",
                torch.zeros((group_rows, pack_cols), dtype=self.pack_dtype),
            )

            self.register_buffer(
                "scales",
                torch.zeros((group_rows, self.out_features), dtype=torch.float16),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = False,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE | torch.device] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ):
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err
        ok, err = _validate_cpu_int4pack_group_size(
            cls,
            group_size=group_size,
            in_features=in_features,
            device=device if device is not None else DEVICE.CPU,
        )
        if not ok:
            return ok, err
        return _validate_cpu_int4pack_out_features(
            cls,
            out_features=out_features,
            device=device if device is not None else DEVICE.CPU,
        )

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        ok, err = TorchAtenLinear.validate_once()
        if ok:
            cls.gemm_int4_forward_kernel = TorchAtenLinear.gemm_int4_forward_kernel
        else:
            cls.gemm_int4_forward_kernel = None
        return ok, err

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        super().optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def _fused_input_dtype(self, x: torch.Tensor) -> bool:
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if self._cpu_transform_dtype is not None:
            return x.dtype == self._cpu_transform_dtype
        return True

    def transform_cpu(self, dtype: torch.dtype | None = None):
        if self._cpu_transformed:
            return

        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val).to(torch.uint8)
        izeros = torch.bitwise_and(izeros, max_val).reshape(self.scales.shape).to(torch.uint8)

        # The op requires the activation and scales/zero-points to share one
        # dtype. Match the first fused activation instead of assuming the
        # checkpoint's FP16 scale buffer also implies FP16 model activations.
        target_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else self.scales.dtype
        if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            target_dtype = torch.bfloat16
        compute_dtype = _cpu_int4pack_scale_dtype(self.scales, izeros, self.bits, target_dtype)
        # Packed runtime metadata only: ``self.scales`` keeps the checkpoint
        # dtype so a later forward or save cannot persist a narrowed scale
        # (issue #4178).
        compute_scales = self.scales.to(compute_dtype).contiguous()
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(iweight.t().int(), 1).contiguous()
        self.qzeros = _cpu_int4pack_zero_offsets(izeros, compute_scales, self.bits).to(compute_dtype).contiguous()
        self.scales_and_zeros = pack_scales_and_zeros(compute_scales, self.qzeros)
        bind_runtime_qzeros(self)
        self._cpu_transform_dtype = target_dtype
        self._cpu_transformed = True
        # ``qweight`` now holds int4pack words: drop the checkpoint clone if
        # the inverse reproduces exactly the bytes captured before the pack.
        finalize_fused_train_state(self)
        # ``scales`` keeps the checkpoint dtype and order and ``bias`` is
        # untouched, so both are still views of the shard's ``safetensors``
        # mapping; either one alone pins every page the loader faulted in
        # (#4785). Detach them now that the runtime layout is built.
        self.scales = detach_cpu_operand(self.scales)
        self.bias = detach_cpu_operand(getattr(self, "bias", None))
        self.g_idx = detach_cpu_operand(getattr(self, "g_idx", None))

    def transform(self, device, dtype: torch.dtype | None = None):
        if device == "cpu":
            self.transform_cpu(dtype=dtype)
        else:
            raise NotImplementedError(
                "TorchAtenAwqLinear only supports fused transforms on CPU devices."
            )

    def awq_weight_dequantize(self, device, dtype):
        return dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        ).to(device=device, dtype=dtype)

    @torch.no_grad()
    def _fused_op_forward(self, x, out_dtype=None):
        if not torch.compiler.is_compiling():
            # Rebuilds ``scales_and_zeros`` from a ``tensor_cache_key``
            # (`is_inference`/`_version`/`data_ptr`), which dynamo cannot trace,
            # and writes module state a graph must not mutate. The eager warmup
            # already refreshed it for the current packed buffers.
            self._refresh_fused_state()
        if x.device.type != "cpu":
            raise NotImplementedError

        if not self._fused_input_dtype(x):
            raise RuntimeError(
                f"{self.__class__.__name__} fused ATen int4pack requires activation dtype "
                f"{self.scales_and_zeros.dtype}, got {x.dtype}."
            )
        if (
            out_dtype is None
            and self.bias is not None
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            # A caller that adds its own bias after this call needs the wide
            # product across the boundary; narrowing here overflows before the
            # outer epilogue can cancel it.
            out_dtype = torch.float32
        return _cpu_int4pack_mm(
            x,
            self.qweight,
            self.group_size,
            self.scales_and_zeros,
            out_dtype=out_dtype,
        )

    # The native affine operand is derived metadata, shared with GPTQ's ATen
    # implementation. Rebuild it after loads, moves or source mutations.
    _refresh_fused_state = TorchAtenLinear._refresh_fused_state

    def clear_weight_cache(self):
        self._fused_state_key = None
        self._refresh_fused_state()

    def _apply(self, fn, recurse: bool = True):
        """Re-finalize the checkpoint layout after an invalidating move.

        The AWQ affine offsets are scale-dtype coupled and the int4pack layout
        is device-specific, so a cross-device or low-precision move must restore
        the checkpoint snapshot instead of reinterpreting it in place. A
        same-device move to FP32 keeps the live layout.
        """

        scales = getattr(self, "scales", None)
        transformed = getattr(self, "_cpu_transformed", False) and isinstance(scales, torch.Tensor)
        # The packed layout is keyed by the activation dtype it was built for,
        # not by ``scales.dtype``: the checkpoint scale buffer now keeps its
        # loaded dtype while the affine metadata follows the activation dtype.
        runtime_dtype = getattr(self, "_cpu_transform_dtype", None) if transformed else None
        if runtime_dtype is None and transformed:
            runtime_dtype = scales.dtype
        device = scales.device if transformed else None
        result = super()._apply(fn, recurse=recurse)
        if refinalize_fused_state_after_apply(self, runtime_dtype, device):
            return result
        scales = getattr(self, "scales", None)
        if (
            transformed
            and isinstance(scales, torch.Tensor)
            and scales.dtype != runtime_dtype
            and scales.dtype in (torch.float16, torch.bfloat16)
        ):
            # No exportable snapshot: keep the live layout usable at the new
            # activation dtype. A later checkpoint load can re-finalize it.
            self._cpu_transform_dtype = scales.dtype
        return result

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self
        if mode and self.linear_mode == "inference":
            restore_fused_train_state(self)
            self.linear_mode = None
            self.scales_and_zeros = None
            self._cpu_transformed = False
            self._cpu_transform_dtype = None
        elif not mode and self.linear_mode == "train":
            # The dense-dequant path holds no derived state, so clearing the
            # latch lets the next forward choose the int4pack layout again.
            # Without this a forward taken while the module was in training
            # mode pinned it to dense dequantization for the rest of its life,
            # so a later `eval()` could never recover (#4907).
            self.linear_mode = None
        return super().train(mode=mode)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if torch.compiler.is_compiling():
            # ``_cpu_transform_lock`` is an ``RLock``: dynamo cannot enter it (a
            # hard graph break), and the one-time ``transform()`` rewrites the
            # packed int4pack buffers in place, which a traced graph must not
            # do. The transform runs on the first eager forward and in
            # ``optimize()``'s warmup, so the graph only reads the resolved
            # ``linear_mode``.
            linear_mode = self.linear_mode
        else:
            with self._cpu_transform_lock:
                if (
                    not self.training
                    and not x.requires_grad
                    and self.linear_mode is None
                    and _has_local_int4pack_cpu_ops()
                    and x.device.type == "cpu"
                    and self._fused_input_dtype(x)
                    # FP16 can overflow before a cancelling bias. BF16 has
                    # FP32's range, so it can use the fused kernel and add the
                    # bias once. The CPU fused op returns an FP32 product for a
                    # biased FP16 layer.
                    and (
                        self.bias is None
                        or x.dtype in (torch.bfloat16, torch.float16)
                    )
                ):
                    capture_fused_train_state(self)
                    self.transform(x.device.type, dtype=x.dtype)
                    self.linear_mode = "inference"
                elif self.linear_mode is None:
                    self.linear_mode = "train"
            linear_mode = self.linear_mode

        if linear_mode == "inference":
            if not self._fused_input_dtype(x):
                raise RuntimeError(
                    f"{self.__class__.__name__} was transformed for {self._cpu_transform_dtype} "
                    f"but received {x.dtype} activations. Keep one activation dtype per layer."
                )
            out = self._fused_op_forward(x)
            if self.bias is not None:
                out = (out.float() + self.bias.to(out.device, torch.float32)).to(x.dtype)
            out = out.reshape(out_shape)
        else:
            weight = self.awq_weight_dequantize(device=x.device, dtype=x.dtype)
            if self.bias is not None:
                out = matmul_fp32_reduce(x, weight, self.bias).reshape(out_shape)
            else:
                out = torch.matmul(x, weight).reshape(out_shape)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["TorchAtenAwqLinear"]
