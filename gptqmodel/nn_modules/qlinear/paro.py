# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Paro runtime implementation adapted from the Paro paper and public
# project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Paro CUDA-backed quantized linear layer."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq
from ...utils.backend import BACKEND
from ...utils.cpu_features import has_avx512_bf16
from ...utils.kernel import resolve_kernel_config
from ...utils.logger import setup_logger
from ...utils.paro import (
    apply_paro_rotation,
    apply_paro_rotation_autograd,
    apply_paro_rotation_vectorized,
    build_identity_rotation_buffers,
    is_identity_rotation,
    prewarm_paro_rotation_extension,
    validate_rotation_buffers,
)
from ...utils.rocm import IS_ROCM
from ...utils.torch import dynamo_is_compiling
from . import detach_cpu_operand
from ._compile_support import COMPILE_LEVEL
from ._fused_train_state import (
    bind_runtime_qzeros,
    capture_fused_train_state,
    discard_fused_train_state,
    finalize_fused_train_state,
    restore_fused_train_state,
)
from ._numerics import linear_wide_bias
from ._runtime_cache import tensor_cache_key
from .gemm_awq import (
    _awq_cuda_gemm_forward,
    _awq_cuda_gemm_forward_fp32_reduce,
)
from .torch_aten_kernel import _has_local_int4pack_cpu_ops
from .torch_aten_kernel_awq import TorchAtenAwqLinear
from .torch_awq import AwqTorchLinear
from .torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear


log = setup_logger()

# INT4 AWQ stores eight codes per int32, so `pack_awq` rejects a packed
# dimension that is not a multiple of eight. Paro reuses that packer for
# every path (CPU ATen, dense fallback, AWQ CUDA, Triton), so the alignment is
# a property of the format rather than of one kernel. Declaring it here keeps
# backend selection, the constructor, and the packer on one source of truth
# instead of accepting a width that can only fail later at packing.
PARO_OUT_FEATURES_ALIGN = 8

# Rotated activations benchmark faster with a shallower K split than generic AWQ.
_PARO_AWQ_SPLIT_K = 4

# The split-K AWQ GEMM and dequantize + cuBLAS cross over at a row count that
# depends on the projection shape. The dense route pays a fixed dequantize
# cost proportional to ``in_features * out_features`` (the dequantized weight
# bytes), while the native route pays one FP32 split-K partial slab per row.
# A global 640-row threshold (the #5164 fix for #5096) sent the 641-2000 row
# band to cuBLAS where the native kernel is still faster (issue #5210); the
# first shape-aware fit then clamped every small wide projection to a
# 2048-row floor and regressed the 1024-2047 row band (issue #5260).
#
# Measured on SM 8.9 (RTX 4090, torch 2.15.0+cu132, FP16, group_size 128,
# synthetic g128 layers), both routes interleaved in one process, min of
# nine to forty-one blocks. The crossover is ~800 rows for 896x4864, ~870 for
# 1536x8960, ~960 for 2048x8192, ~1340 for 4096x14336, ~1490 for 4096x4096,
# ~1900 for 2048x2048, ~2300 for 14336x4096 and 4864x896, ~2000 for
# 8192x2048 and ~3100 for 8960x1536. Small layers pay a fixed cuBLAS small-M
# penalty while large layers pay more dequantize work, so the size response
# is a V-shape in log size around the 4096x4096 anchor. The fit reproduces
# the wide and square crossovers within ~10%; the narrow tall shapes are
# conservative heuristics because cuBLAS switches kernels between adjacent
# row counts. It keeps the 641-row native choice from #5210 for the 7-8B
# shapes, and the 4096-row ceiling preserves #5096's wide-prefill escape.
# See docs/paro.md#shape-aware-crossover.
_PARO_AWQ_DENSE_FLOOR_ROWS = 768
_PARO_AWQ_DENSE_MAX_ROWS = 4096
_PARO_AWQ_DENSE_ROWS_AT_4096 = 1342
_PARO_AWQ_DENSE_ASPECT_EXP = 0.304
_PARO_AWQ_DENSE_SIZE_EXP = 0.357
_PARO_AWQ_DENSE_SMALL_SIZE_EXP = 0.098
_PARO_AWQ_DENSE_NARROW_OUT = 2048
_PARO_AWQ_NARROW_OUT_EXP = 0.219
_PARO_AWQ_DENSE_SIZE_ANCHOR = 4096 * 4096


def paro_awq_dense_min_rows(in_features: int, out_features: int) -> int:
    """Rows from which dequantize + cuBLAS beats the native split-K AWQ GEMM.

    The measured crossover has three drivers: the aspect ratio, the layer size
    ``in_features * out_features`` (the dequantized weight bytes), and the
    output width. Small layers pay a fixed cuBLAS small-M penalty, so the size
    term is a V-shape in log size around the 4096x4096 anchor: below it the
    threshold grows as the layer shrinks, above it the threshold grows with
    the dequantize work. A narrow output width makes the cuBLAS GEMM less
    efficient and raises the threshold further.

    It is a host-side integer decision, not a device probe, so the same shape
    always selects the same path. The floor keeps the 641-row native choice
    from #5210; the 4096-row ceiling preserves #5096's wide-prefill escape.
    """

    if in_features <= 0 or out_features <= 0:
        return _PARO_AWQ_DENSE_FLOOR_ROWS
    size = (in_features * out_features) / _PARO_AWQ_DENSE_SIZE_ANCHOR
    size_exp = (
        _PARO_AWQ_DENSE_SIZE_EXP
        if size >= 1.0
        else -_PARO_AWQ_DENSE_SMALL_SIZE_EXP
    )
    narrow = max(1.0, _PARO_AWQ_DENSE_NARROW_OUT / out_features)
    rows = (
        _PARO_AWQ_DENSE_ROWS_AT_4096
        * (in_features / out_features) ** _PARO_AWQ_DENSE_ASPECT_EXP
        * size**size_exp
        * narrow**_PARO_AWQ_NARROW_OUT_EXP
    )
    return int(
        min(
            _PARO_AWQ_DENSE_MAX_ROWS,
            max(_PARO_AWQ_DENSE_FLOOR_ROWS, round(rows)),
        )
    )


def awq_gemm_shape_supported(*, bits: int, in_features: int, out_features: int, groups: int) -> bool:
    """Whether the native AWQ GEMM accepts a packed layer of this shape.

    Mirrors ``validate_awq_gemm_args`` in
    ``gptqmodel_ext/awq/quantization/gemm_cuda_gen.cu``: uint4b codes only, a
    group size divisible by 32, an output width divisible by the 64-column CTA
    tile, and an output width divisible by the group size. ``groups`` is the
    packed scale-row count, so ``group_size`` is derived exactly as the kernel
    derives it from ``scaling_factors.size(0)``.
    """

    if bits != 4 or groups <= 0 or in_features <= 0 or out_features <= 0:
        return False
    if in_features % groups:
        return False
    group_size = in_features // groups
    if group_size % 32 or out_features % 64:
        return False
    return out_features % group_size == 0


def awq_gemm_bf16_supported(device: torch.device) -> bool:
    """Whether the native AWQ BF16 kernel can run on ``device`` (SM 8.0+)."""

    try:
        return torch.cuda.get_device_capability(device)[0] >= 8
    except Exception:  # noqa: BLE001 - device probe
        return False


class ParoLinear(AwqTorchLinear):
    """Run Paro inference by rotating inputs and reusing AWQ packed GEMM."""

    # The inherited AWQ payload plus the per-layer rotation state. A checkpoint
    # missing any of these loads a wrong rotation or a zero placeholder (the
    # issue measured 1.16-3.99 logit diffs).
    CHECKPOINT_REQUIRED_BUFFERS = (
        "qweight",
        "qzeros",
        "scales",
        "theta",
        "pairs",
        "channel_scales",
    )

    # Eager and an explicit ``torch.cuda.graph`` capture both run with matching
    # output (measured on sm_80, torch 2.15.0.dev20260928+cu130); the rotation
    # buffers are validated in ``post_init``.
    #
    # CPU, torch 2.16.0.dev20261021+cpu: eager=True, graph_breaks=0, and
    # dynamo/aot_eager/inductor all pass for the AVX512-BF16 path (bias
    # included) and for the bias-free ATen int4pack path. The
    # ``_cpu_transform_lock`` transforms and ``_ensure_cpu_layout`` rewrites are
    # skipped while tracing: the eager warmup builds the layout and the traced
    # graph only reads it (#4885 keeps the lock in eager). The AVX512 extension
    # op has a registered FakeTensor impl. A biased FP16/BF16 layer on the ATen
    # path asks for the wide product and adds the bias afterwards; the #4068
    # operand cast is a registered custom op (#5110), so that case traces too
    # (#5332).
    #
    # CUDA keeps ``EAGER``: the native AWQ GEMM entry point was not re-measured
    # on this CPU-only host, so it does not raise the base declaration.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_COMPILE_LEVEL_BY_DEVICE = {"cpu": COMPILE_LEVEL.INDUCTOR}
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for this kernel in this change; the INDUCTOR level is opt-in via "
        "optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = True

    SUPPORTS_BACKENDS = [BACKEND.PARO_CUDA]
    SUPPORTS_METHODS = [METHOD.PARO]
    SUPPORTS_FORMATS = {FORMAT.PARO: 55}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    # The rotation and fused AWQ GEMM branch on host-side row counts and call
    # custom ops, so a compiled decode loop cannot capture this forward. Do not
    # inherit the Torch-AWQ declaration.
    SUPPORTS_COMPILE = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    # The inherited INT4 AWQ packer writes eight codes per int32, so an output
    # width that is not a multiple of eight cannot be packed at all. Advertising
    # 1 here let backend selection accept such a layer and fail later inside
    # `pack_awq`; every Paro inference path consumes the same packed
    # layout, so the constraint is a declaration, not a per-kernel detail.
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [PARO_OUT_FEATURES_ALIGN]

    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "awq_paro"
    # Unlike Torch AWQ, the forward adds an activation-dtype runtime bias.
    MODEL_DTYPE_BUFFERS = ("bias",)
    # A CPU inference forward replaces the packed operands in place with the
    # kernel layout. ``scales``/``bias`` stay in their checkpoint dtype, so
    # only the two rewritten operands need a checkpoint snapshot.
    _FUSED_TRAIN_STATE_NAMES = ("qweight", "qzeros")

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
        krot: int = 8,
        fp32_accum: bool = None,
        cache_runtime_dtype: bool = None,
        auto_cache_bf16_runtime_dtype: bool = None,
        cache_rotation_dtype: bool = None,
        auto_cache_bf16_rotation_dtype: bool = None,
        **kwargs,
    ):
        """Initialize AWQ buffers plus the extra Paro rotation state."""
        self.krot = int(krot)
        if self.krot <= 0:
            raise ValueError(f"ParoLinear: `krot` must be positive, got {krot}.")
        # Backend selection already rejects this through
        # `SUPPORTS_OUT_FEATURES_DIVISIBLE_BY`, but a directly constructed layer
        # must fail here with the real reason instead of inside `pack_awq`.
        if int(out_features) % PARO_OUT_FEATURES_ALIGN:
            raise ValueError(
                f"ParoLinear: `out_features` must be a multiple of "
                f"{PARO_OUT_FEATURES_ALIGN}, got {out_features}. INT4 AWQ packs "
                "eight codes per int32, and Paro reuses that layout for every "
                "inference path, so a narrower output cannot be packed."
            )
        kernel = resolve_kernel_config(kwargs.get("kernel") or kwargs.get("backend"))
        self._paro_cpu_avx512 = kernel.paro_cpu_avx512
        self.fp32_accum = bool(kernel.fp32_accum if fp32_accum is None else fp32_accum)
        self.cache_runtime_dtype = bool(
            kernel.paro_cache_runtime_dtype if cache_runtime_dtype is None else cache_runtime_dtype
        )
        self.auto_cache_bf16_runtime_dtype = bool(
            kernel.paro_auto_cache_bf16_runtime_dtype
            if auto_cache_bf16_runtime_dtype is None
            else auto_cache_bf16_runtime_dtype
        )
        self.cache_rotation_dtype = bool(
            kernel.paro_cache_rotation_dtype if cache_rotation_dtype is None else cache_rotation_dtype
        )
        self.auto_cache_bf16_rotation_dtype = bool(
            kernel.paro_auto_cache_bf16_rotation_dtype
            if auto_cache_bf16_rotation_dtype is None
            else auto_cache_bf16_rotation_dtype
        )
        self._rotation_runtime_dtype: Optional[torch.dtype] = None
        self._rotation_runtime_device: Optional[torch.device] = None
        self._runtime_theta: Optional[torch.Tensor] = None
        self._runtime_channel_scales: Optional[torch.Tensor] = None
        self._rotation_runtime_key = None

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
            backend=kwargs.pop("backend", BACKEND.PARO_CUDA),
            **kwargs,
        )
        # One resident CPU layout at a time: ``"avx512"`` or ``"aten"``. The
        # checkpoint tensors are rebuilt for ``state_dict()``/save instead of
        # being kept next to the packed operands (issue #4814).
        self._cpu_runtime_layout: Optional[str] = None
        self._cpu_runtime_dtype: Optional[torch.dtype] = None
        self._cpu_layout_warning_logged = False
        self.scales_and_zeros = None
        self._register_rotation_buffers()
        self._rotation_identity = True
        # The dense/split-K crossover is a property of the projection shape,
        # so resolve it once instead of paying a float pow on every forward.
        self._awq_dense_min_rows = paro_awq_dense_min_rows(
            self.in_features, self.out_features
        )

    def _register_rotation_buffers(self) -> None:
        """Allocate the per-layer buffers that encode runtime rotations."""
        # Fresh runtime modules must start from a valid identity matching so the
        # fused kernel never sees duplicate pair indices before optimized
        # buffers are loaded from quantization or checkpoints.
        pairs, theta, channel_scales = build_identity_rotation_buffers(
            in_features=self.in_features,
            group_size=self.group_size,
            krot=self.krot,
            dtype=torch.float16,
        )

        if "theta" not in self._buffers:
            self.register_buffer("theta", theta)
        else:
            self.theta = theta

        if "pairs" not in self._buffers:
            self.register_buffer("pairs", pairs)
        else:
            self.pairs = pairs

        if "channel_scales" not in self._buffers:
            self.register_buffer("channel_scales", channel_scales)
        else:
            self.channel_scales = channel_scales

    def clear_weight_cache(self) -> None:
        """Drop packed-weight and rotation-derived state after a mutation."""
        super().clear_weight_cache()
        # The live packed layout stays authoritative for moves and explicit
        # clears; the export path and a checkpoint load restore the snapshot
        # explicitly. Once that snapshot is gone the live buffers are back in
        # checkpoint layout, so forget the layout flag with it.
        if getattr(self, "_fused_train_state", None) is None:
            self._clear_cpu_runtime_layout()
        self._clear_rotation_runtime_cache()
        # A meta/offloaded buffer has no values to inspect. Keep the shortcut
        # disabled until a materialized load or move clears the cache again.
        self._rotation_identity = (
            not self.theta.is_meta
            and not self.channel_scales.is_meta
            and is_identity_rotation(self.theta, self.channel_scales)
        )

    def _live_cpu_layout(self) -> Optional[str]:
        """The packed CPU layout the live buffers hold, or ``None``.

        ``restore_fused_train_state`` consumes the snapshot, so a missing
        snapshot means the buffers are back in checkpoint layout even if the
        flag has not been cleared yet.
        """

        if getattr(self, "_fused_train_state", None) is None:
            return None
        return self.__dict__.get("_cpu_runtime_layout")

    def _clear_cpu_runtime_layout(self) -> None:
        self._cpu_runtime_layout = None
        self._cpu_runtime_dtype = None
        self.scales_and_zeros = None

    def _restore_cpu_checkpoint_layout(self) -> bool:
        """Return the packed buffers to the checkpoint layout, if transformed."""

        if self._live_cpu_layout() is None:
            self._clear_cpu_runtime_layout()
            return False
        restored = restore_fused_train_state(self)
        self._clear_cpu_runtime_layout()
        return restored

    def _can_rebuild_checkpoint_qweight(self) -> bool:
        """Whether this layer's checkpoint ``qweight`` has a verified inverse."""

        layout = self.__dict__.get("_pending_cpu_layout") or self.__dict__.get(
            "_cpu_runtime_layout"
        )
        weight = getattr(self, "qweight", None)
        if layout not in ("avx512", "aten"):
            return False
        if not isinstance(weight, torch.Tensor):
            return False
        if weight.device.type != "cpu" or weight.dtype != torch.int32:
            return False
        if layout == "aten":
            from ._cpu_int4pack_layout import int4pack_block_size

            return int4pack_block_size() is not None
        return weight.dim() == 2

    def _rebuild_checkpoint_qweight(self) -> torch.Tensor:
        """Rebuild the checkpoint ``qweight`` from the live packed layout."""

        layout = self.__dict__.get("_cpu_runtime_layout")
        if layout == "avx512":
            from ...quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns
            from .torch_cpu_avx512 import unpack_hf_int4

            # The AVX512 packer consumes natural AWQ codes; its packed bytes
            # are the ``pack_hf_int4`` group layout, so the inverse is the HF
            # unpack followed by the AWQ word order.
            codes = unpack_hf_int4(self.qweight, self.out_features, self.in_features)
            return pack_awq_columns(codes.t().to(torch.int32), self.bits, order=AWQ_ORDER)
        if layout == "aten":
            from ._cpu_int4pack_layout import rebuild_awq_qweight

            return rebuild_awq_qweight(self)
        raise RuntimeError(
            f"{type(self).__name__} cannot rebuild its checkpoint qweight"
        )

    def _ensure_cpu_layout(self, layout: str, dtype: Optional[torch.dtype] = None) -> bool:
        """Pack the checkpoint buffers in place for one CPU kernel layout.

        Only one copy stays resident: the checkpoint tensors are captured as a
        non-persistent snapshot, then ``qweight``/``qzeros`` are rebound to the
        packed operands. ``finalize_fused_train_state`` drops the ``qweight``
        clone once the inverse reproduces it byte for byte, so ``state_dict()``
        still emits the checkpoint layout (issue #4814).
        """

        if layout == self._live_cpu_layout() and (
            layout != "aten" or self._cpu_runtime_dtype == dtype
        ):
            return True
        if self._live_cpu_layout() is not None:
            if not self._restore_cpu_checkpoint_layout():
                return False

        self.__dict__["_pending_cpu_layout"] = layout
        try:
            capture_fused_train_state(self)
            if layout == "avx512":
                packed_qweight, packed_qzeros = self._pack_cpu_avx512()
                scales_and_zeros = None
            else:
                packed_qweight, packed_qzeros, scales_and_zeros = self._pack_cpu_aten(dtype)
        except Exception as exc:
            # The transform is computed before any buffer is rebound, so a
            # failure leaves the checkpoint tensors untouched.
            discard_fused_train_state(self)
            self._clear_cpu_runtime_layout()
            if not self._cpu_layout_warning_logged:
                log.warning(
                    "Paro CPU %s layout unavailable for `%s`; using the dense fallback: %s",
                    layout,
                    self.name,
                    exc,
                )
                self._cpu_layout_warning_logged = True
            return False
        finally:
            self.__dict__.pop("_pending_cpu_layout", None)

        self.qweight = packed_qweight
        self.qzeros = packed_qzeros
        self._cpu_runtime_layout = layout
        self._cpu_runtime_dtype = dtype
        if layout == "aten":
            self.scales_and_zeros = scales_and_zeros
            bind_runtime_qzeros(self)
        finalize_fused_train_state(self)
        # The per-column operands keep the checkpoint dtype and stay the
        # loader's tensors, so they are views of the shard's ``safetensors``
        # mapping and pin every page the loader touched even after
        # ``qweight``/``qzeros`` were replaced (#4785/#4873). Detach them once
        # the runtime layout is built; values, dtype and order are unchanged.
        for operand in ("scales", "bias", "theta", "pairs", "channel_scales"):
            value = getattr(self, operand, None)
            if isinstance(value, torch.Tensor):
                setattr(self, operand, detach_cpu_operand(value))
        return True

    def _pack_cpu_avx512(self) -> Tuple[torch.Tensor, torch.Tensor]:
        from .torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear

        ok, err = TorchCpuAvx512AwqLinear.cached_validate_once()
        packer = TorchCpuAvx512AwqLinear.pack_awq_int4_kernel
        if not ok or packer is None:
            raise RuntimeError(str(err) if err is not None else "AVX512-BF16 packer unavailable")
        return packer(
            self.qweight.contiguous(),
            self.qzeros.contiguous(),
            self.out_features,
            self.in_features,
        )

    def _pack_cpu_aten(
        self, dtype: Optional[torch.dtype]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .torch_aten_kernel import _cpu_int4pack_scale_dtype, _cpu_int4pack_zero_offsets
        from .torch_fused import pack_scales_and_zeros

        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val).to(torch.uint8)
        izeros = torch.bitwise_and(izeros, max_val).reshape(self.scales.shape).to(torch.uint8)

        # The op requires the activation and scales/zero-points to share one
        # dtype. Match the first fused activation instead of assuming the
        # checkpoint's scale buffer also implies the activation dtype.
        target_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else self.scales.dtype
        if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            target_dtype = torch.bfloat16
        compute_dtype = _cpu_int4pack_scale_dtype(self.scales, izeros, self.bits, target_dtype)
        # Packed runtime metadata only: ``self.scales`` keeps the checkpoint
        # dtype so a later forward or save cannot persist a narrowed scale
        # (issue #4178).
        compute_scales = self.scales.to(compute_dtype).contiguous()
        packed_qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            iweight.t().int(), 1
        ).contiguous()
        packed_qzeros = _cpu_int4pack_zero_offsets(izeros, compute_scales, self.bits).to(
            compute_dtype
        ).contiguous()
        return packed_qweight, packed_qzeros, pack_scales_and_zeros(compute_scales, packed_qzeros)

    def _apply(self, fn, recurse: bool = True):
        """Restore checkpoint tensors before a device/dtype move.

        A CPU runtime layout is only valid for the device and dtype that built
        it. Restoring first also keeps ``.to()``/``.cuda()``/offload free of a
        packed operand whose layout another backend cannot read; the next
        forward re-transforms the module.
        """

        if self._live_cpu_layout() is not None:
            self._restore_cpu_checkpoint_layout()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        if mode and not self.training and self._live_cpu_layout() is not None:
            # Training reads the checkpoint tensors through the dense path.
            self._restore_cpu_checkpoint_layout()
        return super().train(mode=mode)

    def post_init(self) -> None:
        super().post_init()
        weight = getattr(self, "qweight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type == "cpu":
            # Resolve the CPU kernels before any torch.compile trace: their
            # validators load the JIT extensions through importlib, which dynamo
            # cannot trace. This only warms the process-wide caches.
            if self._paro_cpu_avx512 and has_avx512_bf16():
                TorchCpuAvx512AwqLinear.cached_validate_once()
            if _has_local_int4pack_cpu_ops():
                TorchAtenAwqLinear.cached_validate_once()
        # Rotation pairs are checkpoint data used as shared-memory indices by
        # the fused CUDA kernel. Validate them once after the weights load.
        validate_rotation_buffers(
            self.pairs,
            self.theta,
            self.channel_scales,
            group_size=self.group_size,
            in_features=self.in_features,
            krot=self.krot,
            layer_name=f"Paro `{self.name}`",
        )

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        """Load the fused rotation before ``torch.compile`` traces this layer.

        ``optimize()`` usually runs right after a checkpoint load, before any
        eager forward. Loading the JIT extension here keeps the compiled graph
        on the fused rotation op instead of the slower reference fallback.
        """
        weight = getattr(self, "qweight", None)
        prewarm_paro_rotation_extension(
            fused_rotation=True,
            group_size=self.group_size,
            krot=self.krot,
            device=weight.device if isinstance(weight, torch.Tensor) else None,
        )
        super().optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        """Paro relies on AWQ validation and needs no extra one-time checks here."""
        return True, None

    @classmethod
    def auto_select_priority(cls, fmt, dtype=None):
        """AUTO rank; demote the per-forward dense fallback on ROCm.

        ``_cuda_awq_kernel_supported`` is false on every HIP build because the
        AWQ GEMM extension is CUDA-only, so on ROCm this kernel always takes
        ``_forward_dense`` and re-dequantizes the whole packed weight on each
        call. ``ParoTritonLinear`` is the native ROCm path and must win AUTO
        selection there; the declared rank is unchanged on CUDA (#4914).
        """

        priority = super().auto_select_priority(fmt, dtype)
        if IS_ROCM:
            return min(priority, 0)
        return priority

    def extra_repr(self) -> str:
        """Expose Paro-specific fields in `repr(module)` for debugging."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}, "
            f"krot={self.krot}, awq_split_k={_PARO_AWQ_SPLIT_K}, "
            f"cache_runtime_dtype={self.cache_runtime_dtype}, "
            f"auto_cache_bf16={self.auto_cache_bf16_runtime_dtype}, "
            f"cache_rotation_dtype={self.cache_rotation_dtype}, "
            f"auto_cache_bf16_rotation={self.auto_cache_bf16_rotation_dtype}, "
            f"fp32_accum={self.fp32_accum}"
        )

    def _clear_rotation_runtime_cache(self) -> None:
        self._rotation_runtime_dtype = None
        self._rotation_runtime_device = None
        self._runtime_theta = None
        self._runtime_channel_scales = None
        self._rotation_runtime_key = None
        self._runtime_dtype_key = None
        self._runtime_scales = None
        self._runtime_bias = None

    def _runtime_dtype_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        runtime_key = (
            tensor_cache_key(self.scales),
            tensor_cache_key(self.bias),
            device,
            dtype,
        )
        if cache and self._runtime_dtype_key == runtime_key:
            return self._runtime_scales, self._runtime_bias

        scales = self.scales
        if (
            scales.device != device
            or scales.dtype != dtype
            or not scales.is_contiguous()
        ):
            scales = scales.to(device=device, dtype=dtype).contiguous()

        bias = self.bias
        if (
            bias is not None
            and (
                bias.device != device
                or bias.dtype != dtype
                or not bias.is_contiguous()
            )
        ):
            bias = bias.to(device=device, dtype=dtype).contiguous()

        if cache:
            self._runtime_dtype_key = runtime_key
            self._runtime_scales = scales
            self._runtime_bias = bias
        return scales, bias

    def _rotation_runtime_tensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert the rotation operands for this device/dtype, without caching."""

        theta = self.theta.to(device=device, dtype=dtype).contiguous()
        channel_scales = self.channel_scales.to(device=device, dtype=dtype).contiguous()
        return theta, channel_scales

    def _ensure_rotation_runtime_dtype(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if dynamo_is_compiling():
            # `tensor_cache_key` reads `is_inference()`/`_version`/`data_ptr()`,
            # all hard dynamo graph breaks, and publishing the cache is a state
            # write a traced graph must not perform. The eager warmup built the
            # tensors for this device/dtype; a cold trace converts them without
            # publishing, which stays traceable. Without this, every layer of a
            # real (non-identity) Paro checkpoint broke the graph twice on CPU
            # while the class declared `INDUCTOR` (#5115).
            theta = self._runtime_theta
            channel_scales = self._runtime_channel_scales
            if (
                theta is None
                or channel_scales is None
                or theta.device != device
                or theta.dtype != dtype
                or channel_scales.device != device
                or channel_scales.dtype != dtype
                or not theta.is_contiguous()
                or not channel_scales.is_contiguous()
            ):
                return self._rotation_runtime_tensors(device, dtype)
            return theta, channel_scales

        runtime_key = (
            tensor_cache_key(self.theta),
            tensor_cache_key(self.channel_scales),
            device,
            dtype,
        )
        if (
            self._rotation_runtime_key != runtime_key
            or self._runtime_theta is None
            or self._runtime_channel_scales is None
            or not self._runtime_theta.is_contiguous()
            or not self._runtime_channel_scales.is_contiguous()
        ):
            # This cache outlives the call that fills it: the bf16 auto-cache
            # serves every later forward, including grad-mode ones where the
            # fused rotation's autograd wrapper saves these tensors for
            # backward. Tensors created under `torch.inference_mode()` cannot
            # be saved that way, so an eval/`generate()` warm-up must not
            # poison the cache (#5317). `inference_mode(False)` restores the
            # enclosing grad mode, so an ordinary build is unchanged.
            with torch.inference_mode(False):
                self._runtime_theta, self._runtime_channel_scales = (
                    self._rotation_runtime_tensors(device, dtype)
                )
            self._rotation_runtime_device = device
            self._rotation_runtime_dtype = dtype
            self._rotation_runtime_key = runtime_key
        return self._runtime_theta, self._runtime_channel_scales

    def _rotation_parameters_need_grad(self) -> bool:
        """Whether an enabled gradient path consumes the rotation operands.

        An identity rotation is a no-op in value but not in its derivatives:
        the identity angle still has a nonzero derivative and a unit scale has
        a unit derivative (#5419). Dropping the rotation is only safe when no
        enabled gradient needs ``theta`` or ``channel_scales``.
        """

        if not torch.is_grad_enabled():
            return False
        return bool(self.theta.requires_grad or self.channel_scales.requires_grad)

    def _rotation_needs_autograd(self, x_flat: torch.Tensor) -> bool:
        """Whether this rotation must stay inside the autograd graph.

        The fused rotation op has no registered autograd kernel, so calling it
        while a gradient is needed silently returns ``None`` for the input and
        for every rotation parameter (issue #4958). The autograd wrapper runs
        the same fused kernel in its forward and implements the backward.
        """

        if not torch.is_grad_enabled():
            return False
        return bool(x_flat.requires_grad or self._rotation_parameters_need_grad())

    def _rotate_inputs(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Apply the learned input transform before quantized matmul."""
        if self._rotation_identity and not self._rotation_parameters_need_grad():
            # The identity is exact, so the shortcut is bit-identical for the
            # activations; it may only run when the rotation parameters are
            # frozen or the caller disabled grad (issue #5419).
            return x_flat
        needs_autograd = self._rotation_needs_autograd(x_flat)
        if (
            not needs_autograd
            and x_flat.device.type == "cpu"
            and not self.training
        ):
            theta, channel_scales = self._ensure_rotation_runtime_dtype(
                x_flat.device,
                x_flat.dtype,
            )
            return apply_paro_rotation_vectorized(
                x_flat,
                self.pairs,
                theta,
                scales=channel_scales,
                group_size=self.group_size,
            )
        use_cached_rotation_dtype = self.cache_rotation_dtype or (
            self.auto_cache_bf16_rotation_dtype and x_flat.dtype == torch.bfloat16
        )
        theta = self.theta
        channel_scales = self.channel_scales
        if use_cached_rotation_dtype:
            theta, channel_scales = self._ensure_rotation_runtime_dtype(x_flat.device, x_flat.dtype)
        if needs_autograd:
            return apply_paro_rotation_autograd(
                x_flat,
                self.pairs,
                theta,
                scales=channel_scales,
                group_size=self.group_size,
            )
        return apply_paro_rotation(
            x_flat,
            self.pairs,
            theta,
            scales=channel_scales,
            group_size=self.group_size,
        )

    def _forward_dense(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Fallback reference path: dequantize AWQ weights and run dense matmul."""
        # The dense path dequantizes the checkpoint tensors, so a CPU kernel
        # layout must be returned first.
        if self._live_cpu_layout() is not None:
            if dynamo_is_compiling():
                # Restoring rewrites the registered buffers, which a traced
                # graph must not do. A layer whose activation dtype or int4pack
                # availability does not match the warmup-built layout reaches
                # here; compiling before the first eager forward keeps the
                # checkpoint layout live and takes this dense path instead.
                raise RuntimeError(
                    f"{type(self).__name__}: a CPU runtime layout is live and no "
                    "traceable fused path was selected. Compile the layer before "
                    "its first eager forward, or run this layer eagerly."
                )
            self._restore_cpu_checkpoint_layout()
        # Keep the reconstructed grid, the reduction and the stored bias in
        # FP32, then round once.  Decoding to the activation dtype first makes
        # a finite FP16/BF16 result overflow before the bias can cancel it.
        compute_dtype = (
            torch.float32
            if x_flat.dtype in (torch.float16, torch.bfloat16)
            else x_flat.dtype
        )
        return self._dense_matmul_for_dtype(x_flat, compute_dtype)

    def _dense_matmul_for_dtype(self, x_flat: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize at ``compute_dtype`` and run the wide dense product."""
        weight = self._dequantize_weight_for_inference(x_flat, dtype=compute_dtype)
        return linear_wide_bias(
            x_flat,
            # AWQ reconstructs [in_features, out_features]; the wide epilogue
            # takes the F.linear operand order.
            weight.t(),
            self._bias_for_compute(device=x_flat.device),
        )

    def _forward_dense_batch(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Large-row dispatch: dequantize the packed grid, then cuBLAS.

        The activation dtype is the kernel's compute dtype, so the
        reconstructed weight stays narrow and only the reduction and the bias
        are wide, exactly like ``AwqGEMMLinear`` above its own row threshold.
        The weight exists only for this call and is not retained.
        """
        weight = self._dequantize_weight_for_inference(x_2d, dtype=x_2d.dtype)
        bias = self._bias_for_compute(device=x_2d.device)
        out = torch.mm(x_2d, weight, out_dtype=torch.float32)
        if bias is not None:
            out = out + bias.float()
        return out.to(dtype=x_2d.dtype)

    def _cuda_awq_kernel_supported(self, compute_dtype: torch.dtype) -> bool:
        """Whether the native AWQ GEMM can accept this layer's packed shape.

        The kernel's own validator (``validate_awq_gemm_args``) requires an
        output width that is a multiple of its 64-column CTA tile, a group size
        that is a multiple of 32, and an output width divisible by that group
        size. Paro advertises ``group_size=16`` and output widths that the
        shared INT4 packer accepts (multiples of eight) even though they miss
        this tile, because the CPU, ATen and dense paths support them; the
        native path is skipped for those configurations instead of raising from
        inside the kernel.

        The group size is derived from the packed tensors the kernel actually
        reads, not from the config value: ``group_size=-1`` packs one scale row
        per input channel, which the kernel turns into
        ``in_features / scales.size(0)``. The extension is also CUDA-only, so a
        HIP build always takes the portable path.
        """

        if IS_ROCM or self.qweight.device.type != "cuda":
            return False
        qweight, scales = self.qweight, self.scales
        if qweight.numel() == 0 or scales.numel() == 0:
            return False
        if self.bits != 4:
            # The native kernel decodes uint4b codes only.
            return False
        if not awq_gemm_shape_supported(
            bits=self.bits,
            in_features=self.in_features,
            out_features=qweight.size(1) * (32 // self.bits),
            groups=scales.size(0),
        ):
            return False
        if compute_dtype == torch.bfloat16 and not awq_gemm_bf16_supported(qweight.device):
            # ``validate_bf16_device`` rejects the BF16 kernel below SM 8.0.
            return False
        return True

    def _forward_cuda_awq_kernel(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Fast path that feeds rotated activations into the AWQ CUDA GEMM kernel."""
        if torch.is_grad_enabled() and x_flat.requires_grad:
            # The native AWQ GEMM op has no autograd kernel, so a gradient
            # taken through it is silently dropped (issue #4958). The dense
            # reference below is the differentiable path; inference keeps the
            # native kernel.
            return None
        compute_dtype = x_flat.dtype if x_flat.dtype in (torch.float16, torch.bfloat16) else torch.float16
        if not self._cuda_awq_kernel_supported(compute_dtype):
            return None
        kernel_input = (
            x_flat
            if x_flat.dtype == compute_dtype and x_flat.is_contiguous()
            else x_flat.to(device=x_flat.device, dtype=compute_dtype).contiguous()
        )
        use_cached_runtime_dtype = self.cache_runtime_dtype or (
            self.auto_cache_bf16_runtime_dtype and compute_dtype == torch.bfloat16
        )
        kernel_scales, kernel_bias = self._runtime_dtype_tensors(
            kernel_input.device,
            compute_dtype,
            cache=use_cached_runtime_dtype,
        )
        kernel_input_2d = kernel_input.reshape(-1, kernel_input.shape[-1])
        rows = kernel_input_2d.shape[0]
        if rows >= self._awq_dense_min_rows:
            # The split-K GEMM allocates one FP32 partial slab per split, so a
            # long prefill pays split_k x rows x out_features of transient
            # memory before the reduction. Dequantize + cuBLAS keeps only the
            # reconstructed weight beside the output and is faster from the
            # shape-aware crossover. The 32-bit store class of #5096 in the
            # native kernel is fixed independently in gptqmodel_ext/awq; the
            # 4096-row ceiling keeps that escape for every shape.
            return self._forward_dense_batch(kernel_input_2d)
        if self.bias is not None and self.fp32_accum:
            # The split-K partials are already FP32.  Ask for an FP32 result
            # and add the stored bias there, so a finite sum that overflows the
            # activation dtype is not rounded before the bias cancels it.
            out = _awq_cuda_gemm_forward_fp32_reduce(
                kernel_input_2d,
                self.qweight,
                kernel_scales,
                self.qzeros,
                _PARO_AWQ_SPLIT_K,
                kernel=self.kernel,
            )
            bias = self._bias_for_compute(device=x_flat.device).float()
            return (out + bias).to(dtype=x_flat.dtype)

        out = _awq_cuda_gemm_forward(
            kernel_input_2d,
            self.qweight,
            kernel_scales,
            self.qzeros,
            _PARO_AWQ_SPLIT_K,
            fp32_accum=self.fp32_accum,
            kernel=self.kernel,
        )
        if kernel_bias is not None:
            out = out + kernel_bias
        if out.dtype != x_flat.dtype:
            out = out.to(dtype=x_flat.dtype)
        return out

    def _forward_cpu_avx512(
        self,
        x_flat: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> Optional[torch.Tensor]:
        """Run the AVX512-BF16 int4 GEMM on the owned packed operands."""
        if not enabled or not self._cpu_avx512_eligible(x_flat):
            return None

        gemm = TorchCpuAvx512AwqLinear.gemm_int4_qzero_group_scaled_cpu_kernel
        if gemm is None:
            if dynamo_is_compiling():
                # Resolving the handle loads the JIT extension through
                # importlib, which dynamo cannot trace (a `setattr` on the
                # extension module is a hard graph break). The eager warmup
                # already answered for this host: a missing handle means the
                # AVX512 path is unavailable, so fall through to the ATen/dense
                # paths instead of resolving it inside the graph (#5115).
                return None
            ok, _ = TorchCpuAvx512AwqLinear.cached_validate_once()
            gemm = TorchCpuAvx512AwqLinear.gemm_int4_qzero_group_scaled_cpu_kernel
            if not ok or gemm is None:
                return None

        original_dtype = x_flat.dtype
        kernel_input = x_flat if x_flat.is_contiguous() else x_flat.contiguous()
        if dynamo_is_compiling():
            # The transform rewrites registered buffers and takes the RLock
            # dynamo cannot enter; a traced graph must not do either. The eager
            # warmup built this layout, so only read it. Without a warmup, fall
            # through to the ATen/dense paths.
            if self._live_cpu_layout() != "avx512":
                return None
            qweight = self.qweight
            qzeros = self.qzeros
            scales = self.scales.to(dtype=torch.float32).contiguous()
            bias = (
                self.bias.to(dtype=torch.float32).contiguous()
                if self.bias is not None
                else None
            )
        else:
            # The transform rewrites registered buffers, so it must not
            # interleave with `state_dict()`/`load_state_dict()`, which hold the
            # same lock. Only the transform and the operand capture need that
            # lock: the GEMM runs on these locals, so a concurrent save may
            # rebind the module attributes without tearing this product
            # (issue #4885).
            with self._cpu_transform_lock:
                if not self._ensure_cpu_layout("avx512"):
                    return None
                qweight = self.qweight
                qzeros = self.qzeros
                # The kernel requires FP32 affine operands. Widen per call so
                # the checkpoint dtype stays resident instead of a second FP32
                # copy (issue #4814).
                scales = self.scales.to(dtype=torch.float32).contiguous()
                bias = (
                    self.bias.to(dtype=torch.float32).contiguous()
                    if self.bias is not None
                    else None
                )
        output = gemm(
            kernel_input,
            qweight,
            qzeros,
            scales,
            self.group_size,
            bias,
        )
        if output.dtype != original_dtype:
            output = output.to(dtype=original_dtype)
        return output

    def _forward_cpu_aten(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Use ATen's native CPU int4pack GEMM for FP16/BF16 activations."""
        if (
            x_flat.device.type != "cpu"
            or self.training
            or x_flat.requires_grad
            or x_flat.dtype not in (torch.float16, torch.bfloat16)
            or self.bits != 4
            or getattr(self, "requested_group_size", self.group_size) == -1
            or self.group_size not in (32, 64, 128, 256)
        ):
            return None

        if dynamo_is_compiling():
            # The transform rewrites registered buffers and takes the RLock
            # dynamo cannot enter; a traced graph only reads the eager-built
            # layout. A biased layer asks for the FP32 product and adds the
            # bias afterwards, and the #4068 operand cast is a registered
            # custom op since #5110, so it traces as well.
            if (
                self._live_cpu_layout() != "aten"
                or self._cpu_runtime_dtype != x_flat.dtype
            ):
                return None
            int4pack_mm = getattr(self, "_paro_int4pack_mm", None)
            if int4pack_mm is None:
                return None
            qweight = self.qweight
            scales_and_zeros = self.scales_and_zeros
            bias = self.bias
        else:
            # Resolve the entry point at call time: the eager path keeps the
            # lazy lookup so a test or caller can substitute it, and the
            # compile path reads the reference captured here.
            from .torch_fused import _cpu_int4pack_mm as int4pack_mm

            self._paro_int4pack_mm = int4pack_mm
            if not _has_local_int4pack_cpu_ops():
                return None

            ok, _ = TorchAtenAwqLinear.cached_validate_once()
            if not ok:
                return None

            # The transform rewrites registered buffers, so it must not
            # interleave with `state_dict()`/`load_state_dict()`, which hold the
            # same lock. Capture the operands under it, then run the product on
            # those locals: a concurrent save may rebind the module attributes
            # without tearing this GEMM (issue #4885).
            with self._cpu_transform_lock:
                if not self._ensure_cpu_layout("aten", x_flat.dtype):
                    return None
                qweight = self.qweight
                scales_and_zeros = self.scales_and_zeros
                bias = self.bias
        try:
            # The layer has no bias: the parent adds it after return. Ask
            # for the wide product so a finite sum is not rounded to
            # infinity before the outer bias epilogue (issue #4471).
            out = int4pack_mm(
                x_flat,
                qweight,
                self.group_size,
                scales_and_zeros,
                out_dtype=torch.float32 if bias is not None else None,
            )
        except Exception as exc:
            # Fail closed to the dense reference instead of returning a
            # half-transformed module. Restoring rebinds registered buffers,
            # so it takes the same lock the save path holds.
            with self._cpu_transform_lock:
                self._restore_cpu_checkpoint_layout()
            if not self._cpu_layout_warning_logged:
                log.warning(
                    "Paro ATen CPU int4pack failed for `%s`; using dense fallback: %s",
                    self.name,
                    exc,
                )
                self._cpu_layout_warning_logged = True
            return None

        if bias is not None:
            # Accumulate the bias in FP32 before the visible dtype cast. ATen
            # returns a BF16/FP16 result, and a same-dtype add would round both
            # the bias and the sum again.
            out = (out.float() + bias.float()).to(dtype=x_flat.dtype)
        if out.dtype != x_flat.dtype:
            out = out.to(dtype=x_flat.dtype)
        return out

    def forward(self, x: torch.Tensor):
        """Rotate inputs, run quantized matmul, then apply adapters in input space."""
        original_shape = x.shape[:-1] + (self.out_features,)
        if self.input_rows(x) == 0:
            # Rotation and AWQ GEMM both require a positive row count; keep the
            # empty batch on the shared shape-preserving path instead.
            return self.empty_linear_output(x)
        # The fused rotation op requires a densely packed operand and rejects
        # a strided `reshape` view (for example `hidden_states[:, -1:, :]`).
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        rotated = self._rotate_inputs(x_flat)

        out = self._forward_cuda_awq_kernel(rotated)
        if out is None:
            out = self._forward_cpu_avx512(
                rotated,
                enabled=self._paro_cpu_avx512 and rotated.dtype == torch.bfloat16,
            )
        if out is None:
            out = self._forward_cpu_aten(rotated)
        if out is None:
            out = self._forward_dense(rotated)

        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)

        return out.reshape(original_shape)


__all__ = ["ParoLinear"]
