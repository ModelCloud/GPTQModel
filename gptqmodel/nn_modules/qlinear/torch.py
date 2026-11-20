# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import math
import warnings
import weakref
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, HAS_CUDA, PLATFORM
from ...nn_modules.qlinear import (
    BaseQuantLinear,
    PackableQuantLinear,
    _device_family,
    resolve_dequantize_dtype,
)
from ...quantization import FORMAT, METHOD
from ...utils.arch import cuda_arch as _arch_cuda_arch
from ...utils.arch import gemm_fused_dtypes as _arch_gemm_fused_dtypes
from ...utils.arch import gemm_fused_max_out_features as _arch_gemm_max_out_features
from ...utils.arch import is_cuda_fused as _arch_is_cuda_fused
from ...utils.arch import is_rocm_fused_arch, rocm_arch
from ...utils.backend import BACKEND
from ...utils.kernel import resolve_kernel_config
from ...utils.logger import setup_logger
from ...utils.planar_packing import planar_decode_rows
from ...utils.rocm import IS_ROCM
from ...utils.torch import dynamo_is_compiling, model_execution_device_scope, model_global_state_lock, torch_compile
from ._compile_support import COMPILE_LEVEL
from ._decode_dtype import _HALF_DTYPES, decode_dtype_for_scales, scales_fit_dtype
from ._g_idx import validate_g_idx_bounds
from ._numerics import (
    _wants_fp32_reduce,
    autocast_active,
    linear_needs_wide_bias,
    linear_wide_bias,
    matmul_fp32_reduce,
)
from ._runtime_cache import tensor_cache_key


try:
    from ..triton_utils.dequant import dequant as triton_dequant

    _TRITON_DEQUANT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton_dequant = None
    _TRITON_DEQUANT_AVAILABLE = False


try:
    from ..triton_utils.gptq_gemm import MAX_IN_FEATURES as _ROCM_GEMM_MAX_IN_FEATURES
    from ..triton_utils.gptq_gemm import SMALL_M_MAX as _ROCM_GEMM_SMALL_M_MAX
    from ..triton_utils.gptq_gemm import gate_limits as _ROCM_GEMM_GATE_LIMITS
except Exception:  # pragma: no cover - optional dependency
    _ROCM_GEMM_SMALL_M_MAX = 0
    _ROCM_GEMM_MAX_IN_FEATURES = 0
    _ROCM_GEMM_GATE_LIMITS = None




class _CompiledDequantizeWeight:
    """Run a compiled *unbound* ``dequantize_weight`` without pinning the module.

    ``torch.compile(self.dequantize_weight)`` stores the bound method inside the
    compiled wrapper, so the module ends up referencing itself through the
    instance attribute.  A GPU model then keeps its VRAM allocated after the last
    user reference is dropped, until a full cyclic ``gc.collect()`` (#4847).
    Compiling the class function and resolving the instance through a weak
    reference keeps the compiled fast path cycle-free.
    """

    __slots__ = ("_compiled", "_module_ref")

    def __init__(self, compiled, module):
        self._compiled = compiled
        self._module_ref = weakref.ref(module)

    def __call__(self, *args, **kwargs):
        module = self._module_ref()
        if module is None:
            raise RuntimeError("compiled `dequantize_weight` called after its module was released")
        # The compiled kernel is reachable outside the model-forward proxy
        # (e.g. a direct kernel call). Enter the module's device so inductor's
        # device-less lazy patterns cannot resolve to the current, excluded
        # GPU (#5101); the caller's device is restored on exit.
        with model_execution_device_scope(module, args, kwargs):
            return self._compiled(module, *args, **kwargs)

    def __repr__(self):
        return f"<compiled dequantize_weight {self._compiled!r}>"


log = setup_logger()


# ``gptqmodel.utils.arch`` owns the cached ``gcnArchName`` lookup. The alias is
# kept so existing imports of ``_rocm_arch_name`` inside this module (and by
# downstream users) keep working. It returns ``""`` rather than ``None`` because
# the historical signature always returned a string.
def _rocm_arch_name(device: torch.device) -> str:
    """Normalized ROCm architecture (``gfx950``) or ``""`` when unavailable."""
    return rocm_arch(device) or ""


def _fused_arch_key(device: torch.device) -> str:
    """Architecture token for the fused-kernel tables (``gfx950``/``sm_89``).

    The shared tables are keyed by the :func:`gptqmodel.utils.arch.tuning_family`
    vocabulary, so CUDA devices must be probed through the compute capability
    rather than ``gcnArchName`` (which is empty off ROCm).  ROCm behavior is
    unchanged: the ``gfx*`` string is returned when present.
    """

    arch = _rocm_arch_name(device)
    if arch:
        return arch
    return _arch_cuda_arch(device) or ""


# One-shot diagnostics for fallbacks: the forward path is hot, so each reason is
# logged at debug level at most once per process instead of every forward.
_LOGGED_FALLBACKS: set[str] = set()


def _log_fallback_once(reason: str, message: str) -> None:
    if reason in _LOGGED_FALLBACKS:
        return
    _LOGGED_FALLBACKS.add(reason)
    log.debug(message)


def fused_scale_pair_supported(
    scale_dtype: torch.dtype, activation_dtype: torch.dtype
) -> bool:
    """Whether the fused GPTQ kernels may read ``scale_dtype`` for this activation.

    ``gptq_gemv``/``gptq_gemm`` load every scale, widen it to FP32, form the
    decoded weight and round that product once to the activation dtype.  The
    stored scale operand therefore does not have to be the activation dtype: it
    only has to be one of the half dtypes the kernels were measured with.

    GPTQ checkpoints store FP16 scales (``_scales_for_storage``) while a BF16
    model runs BF16 activations, so that mixed pair is the one this admits
    (#5740); before it, every BF16 model missed the fused kernels and paid the
    dense dequantize + matmul fallback.  An equal pair keeps its previous
    behaviour, and any other pair (for example FP32 scales with BF16
    activations) keeps the dense path because the fused kernels are not
    measured for it.  Range safety is a separate check
    (``TorchLinear._decoded_weight_fits``), not part of this pair rule.
    """

    if scale_dtype == activation_dtype:
        return True
    return scale_dtype in _HALF_DTYPES and activation_dtype in _HALF_DTYPES


def _right_shift_unpack(values: torch.Tensor, shifts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if values.device.type == "xla":
        raise RuntimeError(
            "Torch/XLA does not provide an exact GPTQ word unpack for this operation. "
            "Use the Tenstorrent native unpack path."
        )
    if values.device.type != "npu":
        return torch.bitwise_right_shift(values, shifts).to(dtype)

    # CANN does not implement bitwise_right_shift. Use int64 arithmetic shift
    # equivalents on NPU so GPTQ unpacking avoids torch-npu's CPU fallback path.
    values_i64 = values.to(torch.int64)
    shifts_i64 = shifts.to(device=values.device, dtype=torch.int64)
    divisors = torch.pow(torch.full_like(shifts_i64, 2), shifts_i64)
    return torch.floor_divide(values_i64, divisors).to(dtype)


class _LinearWeightMetadata:
    """Tensor-like metadata shim for integrations that only inspect `weight` attrs."""

    def __init__(self, module: "TorchLinear", transposed: bool = False):
        # A strong back-reference here closes a module -> metadata -> module
        # cycle, so every quantized linear pins its buffers (and, on a GPU, its
        # VRAM) until a full cyclic GC (#4847).
        self._module_ref = weakref.ref(module)
        self._transposed = transposed

    @property
    def _module(self) -> "TorchLinear":
        module = self._module_ref()
        if module is None:
            raise RuntimeError("weight metadata accessed after its module was released")
        return module

    def _shape(self) -> torch.Size:
        shape = (self._module.out_features, self._module.in_features)
        if self._transposed:
            shape = (shape[1], shape[0])
        return torch.Size(shape)

    def _first_tensor(self) -> torch.Tensor | None:
        for name in ("qweight", "scales", "bias", "qzeros", "g_idx"):
            tensor = getattr(self._module, name, None)
            if tensor is not None:
                return tensor
        return None

    @property
    def device(self) -> torch.device:
        tensor = self._first_tensor()
        return tensor.device if tensor is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for name in ("bias", "scales", "qweight"):
            tensor = getattr(self._module, name, None)
            if tensor is not None:
                return tensor.dtype
        return torch.float16

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> torch.Size:
        return self._shape()

    @property
    def requires_grad(self) -> bool:
        return False

    @property
    def T(self) -> "_LinearWeightMetadata":
        return _LinearWeightMetadata(self._module, transposed=not self._transposed)

    def size(self, dim: int | None = None):
        shape = self._shape()
        return shape if dim is None else shape[dim]

    def __repr__(self) -> str:
        return (
            f"_LinearWeightMetadata(device={self.device}, dtype={self.dtype}, "
            f"shape={tuple(self.shape)})"
        )


class TorchLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 20, FORMAT.GPTQ_V2: 20, FORMAT.GPTQ_P: 20}
    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # CPU, torch 2.16.0.dev20261021+cpu: eager=True, graph_breaks=0,
    # dynamo/aot_eager/inductor all reach it, so the CPU forward is
    # ``INDUCTOR``. The weight-cache key (`is_inference`/`_version`/
    # `data_ptr`) and the `ret_idx` verdict used to break dynamo; both now have
    # a traced fast path that reads the eager-built state instead of re-keying
    # it (see `_stream_decode_qzeros`, `_stream_g_idx_long` and
    # `PackableQuantLinear._ret_idx_is_identity`).
    #
    # CUDA stays ``EAGER``: the streaming decode path (`_should_use_streaming`
    # -> `_stream_get_dequant_stream`) creates a `torch.cuda.Stream` and
    # enqueues copies on it, which dynamo cannot trace, and it is the default
    # for 2/4/8-bit CUDA layers. Explicit CUDA-graph capture of the eager
    # forward succeeds and matches eager.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_COMPILE_LEVEL_BY_DEVICE = {"cpu": COMPILE_LEVEL.INDUCTOR}
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for the portable Torch decode in this change; the INDUCTOR level is "
        "opt-in via optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

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
        format: Optional[FORMAT] = None,
        **kwargs,
    ):
        # Optional per-module opt-out from torch.compile. Validate here so a typo
        # cannot silently leave compilation enabled.
        compile_flag = kwargs.pop("compile", True)
        if not isinstance(compile_flag, bool):
            raise TypeError(f"`compile` must be bool, got {type(compile_flag).__name__}")
        self._compile_enabled = compile_flag
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=kwargs.pop("enable_wf_unsqueeze", True),
            format=format,
            **kwargs)

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8
        kernel = resolve_kernel_config(self)
        self._streaming_enabled = kernel.torch_streaming
        self._stream_tile_cols = kernel.torch_stream_tile
        self._stream_double_buffers = 2
        self._g_idx_long_cache = None
        self._g_idx_long_cache_state = None
        self._zeros_cache = None
        self._zeros_cache_state = None
        self._embedding_zeros = None
        self._embedding_zeros_key = None
        self._embedding_row_state = None
        self._stream_dequant_streams = {}
        self._cache_enabled = False
        # Range-safe decode dtypes, refreshed eagerly by ``clear_weight_cache``
        # so a traced graph can read them as constants (#5606).
        self._decode_dtype_cache = None
        self._decode_fit_cache = None
        self._decode_dtype_cache_key = None
        triton_flag = kernel.torch_triton_dequant
        if triton_flag is None:
            self._triton_dequant_enabled = _TRITON_DEQUANT_AVAILABLE
        else:
            self._triton_dequant_enabled = triton_flag and _TRITON_DEQUANT_AVAILABLE
        self._lookahead_enabled = False
        self._rocm_gemv_enabled = kernel.torch_rocm_gemv
        self._rocm_gemm_enabled = kernel.torch_rocm_gemm
        # Keyed by g_idx identity and version so a checkpoint load, move,
        # deepcopy, or in-place write cannot leave a stale dispatch choice.
        self._gptq_gemm_uniform_value = False
        self._gptq_gemm_uniform_id = None
        self._gptq_gemm_uniform_version = None
        self._lookahead_next = None
        self._weight_metadata = _LinearWeightMetadata(self)
        if kernel.torch_cache_weights:
            self.enable_weight_cache(True)
        if kernel.torch_lookahead:
            self.enable_lookahead(True)

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

    @classmethod
    def _validate(cls, **args) -> tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err

        bits = args.get("bits")
        pack_dtype = args.get("pack_dtype")
        if bits == 3 and pack_dtype not in (None, torch.int32):
            return False, NotImplementedError(
                "3-bit GPTQ packing requires torch.int32 words: "
                f"got pack_dtype={pack_dtype}"
            )
        return True, None

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)
        if _device_family(device) == DEVICE.XLA:
            # TT's Torch-XLA lowering converts int32 shifts through f32. The
            # portable wf-unpack path is therefore not numerically exact; use
            # the dedicated Tenstorrent kernel instead of silently wrong codes.
            raise NotImplementedError(
                "TorchLinear's packed-word unpack is not exact on Torch/XLA; "
                "a Tenstorrent-native GPTQ kernel is required."
            )

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

        super().post_init()

        # Packing often runs under InferenceMode, whose tensors have no version
        # counter. Keep this tiny metadata tensor normal so the dispatch cache
        # can cheaply detect in-place g_idx writes.
        if self.g_idx.is_inference():
            with torch.inference_mode(False):
                self.g_idx = self.g_idx.clone()

        validate_g_idx_bounds(
            self.g_idx,
            self.scales,
            layer_name=type(self).__name__,
        )

        self._gptq_gemm_uniform_group_for_current_g_idx()

        # torch benefits the most from torch.compile, enable it by default
        self.optimize()
        self.clear_weight_cache()

    @property
    def weight(self):
        metadata = getattr(self, "_weight_metadata", None)
        if metadata is None:
            # Rebuilt after unpickling: the weak back-reference is not serializable.
            metadata = self._weight_metadata = _LinearWeightMetadata(self)
        return metadata

    def dequantize_weight(self, num_itr: int = 1, dtype: torch.dtype | None = None):
        """Reconstruct the packed weight, preferably in ``dtype``.

        ``dtype`` is the caller's requested decode dtype (the base class takes
        the same argument).  It is honoured only when every decoded weight fits
        it; otherwise the range-safe decode keeps the reduction wide (#5606).
        """
        # Triton dequant currently handles the common single-iteration layout.
        # Multi-iteration requests (num_itr > 1) are routed to the torch path below.
        if (
            num_itr == 1
            and self._triton_dequant_enabled
            and self._can_use_triton_dequant()
        ):
            return self._dequantize_weight_triton(dtype=dtype)

        # Eval-time fast path for 2/4/8-bit torch dequant.
        # This is also the fallback when Triton is enabled but not eligible.
        if not self.training and self.bits in (2, 4, 8):
            return self._dequantize_weight_cached_248(num_itr=num_itr, dtype=dtype)

        return super().dequantize_weight(num_itr=num_itr, dtype=dtype)

    def optimize(self, backend: str = None, mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        device_type = self.list_buffers()[0].device.type
        if device_type == "npu":
            # torch.compile/inductor is not reliable on torch-npu in supported
            # deployments; the eager Torch path is the portable NPU backend.
            self.optimized = True
            log.info.once("Optimize: `TorchLinear` torch.compile skipped on NPU.")
            return

        if device_type == "xla":
            # TT-XLA already compiles StableHLO graphs through PJRT. Nesting
            # torch.compile's inductor backend here is unsupported.
            self.optimized = True
            log.info.once("Optimize: `TorchLinear` torch.compile skipped on XLA.")
            return

        if not self._compile_enabled:
            self.optimized = True
            log.info.once("Optimize: `TorchLinear` torch.compile disabled by `compile=False`.")
            return

        if backend is None:
            # MPS doesn't support inductor.
            backend = "inductor" if device_type != "mps" else "aot_eager"

        # Compile the unbound function: compiling the bound method would store a
        # strong reference to this module inside the compiled wrapper (#4847).
        # Construction imports dynamo and changes its process-global config, so
        # take the global-state writer while installing (#5127, #5268).
        with model_global_state_lock():
            compiled = torch_compile(type(self).dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)
            self.dequantize_weight = _CompiledDequantizeWeight(compiled, self)

            if self.adapter:
                self.adapter.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

            super().optimize()

    def __getstate__(self):
        """Drop the compiled ``dequantize_weight`` so the module stays picklable.

        ``optimize()`` replaces the bound method with a ``torch.compile``
        wrapper, which pickle cannot serialize. The class method is the eager
        fallback, so omitting the instance override restores it on unpickle and
        the module keeps running (uncompiled).
        """

        state = super().__getstate__()
        state.pop("dequantize_weight", None)
        state.pop("_weight_metadata", None)
        return state

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        # IPEX kernel will use Torch for training only and switches back to IPEX for eval/inference
        # If the kernel inherits Torch kernel only for training and can do its own inference in v1 (IPEX, Marlin) then
        # we can support training for all these v1 kernels by enabling this flag. We need to switch qzero states
        # by overriding module train() and swapping qzero back between v1 and v2 (Torch kernel requires v2)
        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            # training starts
            if mode:
                # one time clone v1 qzeros and save both v1 and v2 qzeros in memory
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                        self._stream_reset_cache()
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)
                        self._stream_reset_cache()

            # training switching to inference/eval
            else:
                if hasattr(self, "qzeros_data_v1"):
                    # switch qzero back to v1 for inference/eval
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)
                    self._stream_reset_cache()

        self.clear_weight_cache()
        return super().train(mode=mode)

    def forward(self, x: torch.Tensor):
        # if x.size(-1) != self.padded_infeatures:
        #     x = F.pad(x, (0, self.padded_infeatures - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        out = self._forward(x, out_shape)
        return out

    def _forward(self, x, out_shape):
        x = self._apply_rotation_to_input(x)
        if self._should_use_streaming(x):
            out = self._forward_streaming(x, out_shape)
        else:
            out = self._forward_eager(x, out_shape)

        return out

    def _forward_eager(self, x: torch.Tensor, out_shape):
        if self._can_use_rocm_gemv(x):
            from ..triton_utils.gptq_gemv import gptq_gemv

            out_dtype = self._fused_bias_out_dtype(x)
            out = gptq_gemv(
                x, self.qweight, self.scales, self.qzeros, self.g_idx,
                out_dtype=out_dtype, bits=self.bits,
            )
            out = self._apply_fused_bias(out, x, wide=out_dtype is not None).reshape(out_shape)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        if self._can_use_rocm_gemm(x):
            from ..triton_utils.gptq_gemm import gptq_gemm

            out_dtype = self._fused_bias_out_dtype(x)
            out = gptq_gemm(
                x, self.qweight, self.scales, self.qzeros, self.g_idx,
                out_dtype=out_dtype, bits=self.bits,
                uniform_group=self._gptq_gemm_uniform_group_for_current_g_idx(),
            )
            out = self._apply_fused_bias(out, x, wide=out_dtype is not None).reshape(out_shape)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        num_itr = self.g_idx.shape[0] // x.shape[-1]
        # Decode into the forward's activation dtype when every weight fits it:
        # the fallback narrows the weight to that dtype anyway, and decoding in
        # a different half dtype first rounds it twice (#5740).
        weights = self.dequantize_weight(
            num_itr=num_itr, dtype=self._forward_decode_dtype(x.dtype)
        )
        bias = self.bias
        if weights.dtype != x.dtype and not self._decoded_weight_fits(x.dtype):
            # The reconstructed weight is outside the activation dtype's finite
            # range, so narrowing it would turn it into `inf` (and `0 * inf`
            # into `NaN`). Promote the activation for this call, reduce wide,
            # and round once at the output (#5606). This is the same epilogue
            # the sibling dense fallbacks use.
            if weights.device != x.device:
                weights = weights.to(device=x.device)
            out = linear_wide_bias(x, weights.t(), bias).reshape(out_shape)
        elif x.device.type == "cpu" and x.dtype in (torch.float16, torch.bfloat16) and bias is not None:
            weights = weights.to(device=x.device)
            bias = bias.to(device=x.device)
            # CPU matmul rounds the product before a separate bias addition.
            # addmm keeps the reduction and bias in one epilogue. If checkpoint
            # metadata has a wider dtype, preserve its range through FP32.
            with torch.autocast(device_type="cpu", enabled=False):
                if weights.dtype == x.dtype and bias.dtype == x.dtype:
                    out = torch.addmm(bias, x, weights)
                else:
                    out = torch.addmm(bias.float(), x.float(), weights.float()).to(x.dtype)
            out = out.reshape(out_shape)
        else:
            if weights.device != x.device or weights.dtype != x.dtype:
                # Quantized modules can be staged on a different accelerator than the
                # caller tensor during multi-device kernel validation; matmul still
                # needs both operands on the same device and dtype.
                weights = weights.to(device=x.device, dtype=x.dtype)
            # Never narrow a stored wide bias here: rounding ``-65535`` to FP16
            # turns it into ``-inf``, and a low-precision reduction can round
            # away the partial sum the bias is supposed to cancel. The device
            # move is safe; ``matmul_fp32_reduce`` keeps a wide bias out of the
            # low-precision epilogue and both values meet in FP32.
            if bias is not None and bias.device != x.device:
                bias = bias.to(device=x.device)
            if bias is not None and not linear_needs_wide_bias(x, bias):
                bias = bias.to(dtype=x.dtype)
            out = matmul_fp32_reduce(x, weights, bias=bias).reshape(out_shape)
        del weights

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _fused_bias_out_dtype(self, x: torch.Tensor):
        """Request a wide fused-kernel output when a bias must be added first.

        ``gptq_gemv``/``gptq_gemm`` store into the output tensor's dtype, so a
        bias that cancels a large partial sum has to be applied to an FP32
        output: adding it after the FP16/BF16 store rounds (and can overflow)
        the partial sum first.  Autocast narrows the operands and lets the
        standalone addition promote instead, so it keeps that expression.
        """
        if self.bias is None or x.dtype not in (torch.float16, torch.bfloat16):
            return None
        if autocast_active(x.device.type):
            return None
        return torch.float32

    def _apply_fused_bias(self, out: torch.Tensor, x: torch.Tensor, *, wide: bool) -> torch.Tensor:
        """Add the bias to a fused-kernel output, keeping a wide partial sum."""
        bias = self.bias
        if bias is None:
            return out
        if bias.device != x.device:
            bias = bias.to(device=x.device)
        if not wide:
            # Preserve the previous in-place `out.add_(bias)` dtype even when
            # the stored bias has a wider dtype than the activations.
            return (out + bias).to(out.dtype)
        return (out + bias.float()).to(x.dtype)

    def _can_use_rocm_gemv(self, x: torch.Tensor) -> bool:
        return (
            self._rocm_gemv_enabled
            and x.shape == (1, self.in_features)
            and self._can_use_rocm_fused(x)
        )

    def _rocm_gemm_gate(self, x: torch.Tensor) -> tuple[int, int]:
        """Return ``(small_m_max, max_in_features)`` for this device's arch.

        Unregistered architectures yield ``(0, 0)`` so the fused GEMM stays off,
        matching the pre-RDNA4 behaviour.
        """
        if _ROCM_GEMM_GATE_LIMITS is None:
            return _ROCM_GEMM_SMALL_M_MAX, _ROCM_GEMM_MAX_IN_FEATURES
        return _ROCM_GEMM_GATE_LIMITS(_fused_arch_key(x.device))

    def _rocm_gemm_dtype_supported(self, x: torch.Tensor) -> bool:
        """Return whether the fused GEMM is a measured win for this activation dtype.

        RDNA4's fused small-M GEMM is a uniform win in FP16 (1.11x-2.53x on
        gfx1201) but not in BF16 (0.43x-1.42x, median 0.85x), so BF16 stays on
        the split path there.  CDNA keeps both dtypes (its historical values are
        unchanged).  Unregistered architectures have an empty set.
        """
        return str(x.dtype).removeprefix("torch.") in _arch_gemm_fused_dtypes(_fused_arch_key(x.device))

    def _rocm_gemm_out_features_supported(self, x: torch.Tensor) -> bool:
        """Return whether this output width is inside the measured fused win.

        ``cdna4``/``rdna4`` measured uniform wins out to their widest tested
        projection, so their cap is ``0`` (disabled) and this is a no-op.  Ada's
        fused tile only beats the split path while the grid stays narrow, so the
        cap is what excludes the wide-N regressions.
        """

        cap = _arch_gemm_max_out_features(_fused_arch_key(x.device))
        return cap <= 0 or self.out_features <= cap

    def _can_use_rocm_gemm(self, x: torch.Tensor) -> bool:
        small_m_max, max_in_features = self._rocm_gemm_gate(x)
        in_features_supported = (
            max_in_features <= 0 or self.in_features <= max_in_features
        )
        return (
            self._rocm_gemm_enabled
            and 2 <= x.shape[0] <= small_m_max
            and x.shape[1] == self.in_features
            and in_features_supported
            and self._rocm_gemm_out_features_supported(x)
            and self._rocm_gemm_dtype_supported(x)
            and self._can_use_rocm_fused(x)
        )

    def _can_use_rocm_fused(self, x: torch.Tensor) -> bool:
        """Preconditions shared by the fused GPTQ GEMV / small-M GEMM kernels.

        Enabled on ROCm for the registered ``gfx*`` families and on Ada
        (RTX 4090 / SM 8.9), whose launch entries and gates are measured in the
        `gptq_gemv`/`gptq_gemm` tables.  Every other CUDA family keeps the
        portable dequantize + matmul path.

        The scale dtype does not have to equal the activation dtype: the
        kernels widen every scale to FP32 before the product, so
        :func:`fused_scale_pair_supported` admits the FP16-scale/BF16-activation
        pair a GPTQ checkpoint produces for a BF16 model (#5740).  The range
        guard below still keeps a decode that would overflow the activation
        dtype on the dense path.
        """
        return (
            (IS_ROCM or _arch_is_cuda_fused(x.device))
            and self._triton_dequant_enabled
            and self._can_use_triton_dequant()
            and self.bits in (2, 4, 8)
            and self.pack_dtype == torch.int32
            and self.qzero_format() == 2
            and self.in_features >= 32
            and self.in_features % 32 == 0
            and self.out_features >= 32
            and self.out_features % 32 == 0
            and self.g_idx.numel() == self.in_features
            and x.is_contiguous()
            and not x.requires_grad
            and fused_scale_pair_supported(self.scales.dtype, x.dtype)
            # The fused kernels stage the decoded weight in the activation
            # dtype, so they are only range-safe when every decoded weight fits
            # it; otherwise the dense path keeps the reduction wide (#5606).
            and self._decoded_weight_fits(x.dtype)
            and self.qweight.device == x.device
            and self._rocm_arch_is_fused(x)
        )

    def _rocm_arch_is_fused(self, x: torch.Tensor) -> bool:
        """Architecture gate for the fused kernels, with a one-shot debug reason.

        Only architecture rejection is logged: the other preconditions (bits,
        packing, shape, training mode) are ordinary fallbacks that fire on every
        model and would be noise. Unregistered architectures keep the previous
        behaviour and silently use dequantize + matmul.
        """
        if not IS_ROCM:
            if _arch_is_cuda_fused(x.device):
                # The Ada/Blackwell entries live in the same tables; the shared precondition
                # list above already covers it.
                return True
            _log_fallback_once(
                "cuda-not-fused",
                "CUDA fused GPTQ kernel skipped: not an Ada/Blackwell CUDA host.",
            )
            return False
        arch = _rocm_arch_name(x.device)
        if not arch:
            _log_fallback_once(
                "rocm-arch-unknown",
                "ROCm fused GPTQ kernel skipped: device architecture is unknown.",
            )
            return False
        if not is_rocm_fused_arch(arch):
            _log_fallback_once(
                f"rocm-arch-{arch}",
                f"ROCm fused GPTQ kernel skipped: {arch} is not a registered fused architecture.",
            )
            return False
        return True

    def _forward_streaming(self, x: torch.Tensor, out_shape):
        tile = max(64, min(self._stream_tile_cols, self.out_features))
        total_tiles = math.ceil(self.out_features / tile)
        device = x.device

        out = torch.empty((x.shape[0], self.out_features), dtype=x.dtype, device=device)
        buffers = self._stream_get_workspace(device=device, dtype=x.dtype, tile=tile)
        widths = [0 for _ in range(self._stream_double_buffers)]

        compute_stream = torch.cuda.current_stream(device=device)
        wide_reduce = _wants_fp32_reduce(x)
        # A stored bias wider than the activation dtype must stay out of the
        # low-precision addmm epilogue, exactly like the eager path.
        wide_bias = linear_needs_wide_bias(x, self.bias)
        wide_epilogue = wide_reduce or wide_bias
        # Autocast narrows a fused `addmm` bias to the autocast dtype, so that
        # context keeps the post-loop addition, where type promotion applies.
        fuse_bias = not wide_epilogue and not autocast_active(device.type)
        bias = self.bias
        if bias is not None and bias.device != device:
            bias = bias.to(device=device)
        # A bias that must stay wide keeps its stored dtype so the per-tile
        # FP32 epilogue can add it; every other bias keeps the old cast.
        if bias is not None and not wide_bias:
            bias = bias.to(dtype=x.dtype)
        stream_dequant = self._stream_get_dequant_stream(device)
        zeros = self._stream_decode_qzeros()
        g_idx = self._stream_g_idx_long()
        stream_dequant.wait_stream(compute_stream)
        for tensor in (*buffers, zeros, g_idx, self.qweight, self.scales, self.wf_unsqueeze_neg_one):
            tensor.record_stream(stream_dequant)
        consumed = [None for _ in buffers]

        def schedule(tile_idx: int, buffer_idx: int):
            start = tile_idx * tile
            end = min(start + tile, self.out_features)
            with torch.cuda.stream(stream_dequant):
                if consumed[buffer_idx] is not None:
                    stream_dequant.wait_event(consumed[buffer_idx])
                widths[buffer_idx] = self._stream_dequantize_tile(
                    buffer=buffers[buffer_idx],
                    zeros=zeros,
                    g_idx=g_idx,
                    start=start,
                    end=end,
                    dtype=x.dtype,
                )

        schedule(0, 0)

        for tile_idx in range(total_tiles):
            buffer_idx = tile_idx % self._stream_double_buffers
            compute_stream.wait_stream(stream_dequant)
            width = widths[buffer_idx]
            start = tile_idx * tile

            out_slice = out.narrow(1, start, width)
            weight_tile = buffers[buffer_idx].narrow(1, 0, width)
            tile_bias = bias.narrow(0, start, width) if bias is not None else None
            if wide_epilogue:
                out_slice.copy_(matmul_fp32_reduce(x, weight_tile, bias=tile_bias))
            elif fuse_bias and tile_bias is not None:
                # Fuse the bias into the GEMM epilogue; rounding `out` before
                # `out.add_(bias)` overflows/cancels in the activation dtype.
                torch.addmm(tile_bias, x, weight_tile, beta=1.0, alpha=1.0, out=out_slice)
            else:
                out_slice.zero_()
                torch.addmm(out_slice, x, weight_tile, beta=0.0, alpha=1.0, out=out_slice)
            consumed[buffer_idx] = torch.cuda.Event()
            consumed[buffer_idx].record(compute_stream)

            next_tile = tile_idx + 1
            if next_tile < total_tiles:
                next_buffer_idx = next_tile % self._stream_double_buffers
                schedule(next_tile, next_buffer_idx)

        buffers.clear()
        out = out.reshape(out_shape)

        if bias is not None and not wide_epilogue and not fuse_bias:
            out.add_(bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _stream_dequantize_tile(
        self,
        buffer: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> int:
        width = end - start
        qweight_tile = self.qweight.narrow(1, start, width)
        weight = _right_shift_unpack(
            qweight_tile.unsqueeze(1).expand(-1, self.pack_factor, -1),
            self.wf_unsqueeze_neg_one,
            self.dequant_dtype,
        )
        weight = torch.bitwise_and(weight, self.maxq)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        weight = weight[:self.in_features]

        zeros_tile = zeros.narrow(1, start, width)
        scales_tile = self.scales.narrow(1, start, width)

        # Keep the multiply in a dtype that holds every decoded weight; the
        # stored half-precision scale would otherwise overflow here (#5606).
        decode_dtype = self._decode_dtype()
        if decode_dtype != self.scales.dtype:
            scales_tile = scales_tile.to(decode_dtype)
            zeros_tile = zeros_tile.to(decode_dtype)
        dequant = scales_tile[g_idx] * (weight - zeros_tile[g_idx])
        buffer.narrow(1, 0, width).copy_(dequant.to(dtype))
        return width

    def _decode_qzeros(self):
        zeros = _right_shift_unpack(
            self.qzeros.unsqueeze(2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
            self.dequant_dtype,
        )
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.qzeros.shape[0], -1)
        return zeros[:, :self.scales.shape[1]]

    def _stream_decode_qzeros(self):
        if dynamo_is_compiling():
            # `tensor_cache_key` reads `is_inference()`/`_version`/`data_ptr()`,
            # all hard dynamo graph breaks, and publishing a fresh key is an
            # attribute write a graph must not perform. The decoded zero-points
            # are runtime metadata derived from a constant buffer, so a traced
            # graph reads the already-built cache and only decodes directly
            # when the cache is empty (e.g. compile before any eager call).
            cached = self._zeros_cache
            if cached is not None:
                return cached
            return self._decode_qzeros()

        cache_state = (tensor_cache_key(self.qzeros), self.scales.shape)
        if self._zeros_cache is not None and self._zeros_cache_state == cache_state:
            return self._zeros_cache

        zeros = self._decode_qzeros()
        self._zeros_cache = zeros
        self._zeros_cache_state = cache_state
        return zeros

    def _stream_g_idx_long(self, target_device: torch.device = None):
        if target_device is None:
            if self.qweight is not None:
                target_device = self.qweight.device
            else:
                target_device = self.g_idx.device

        if dynamo_is_compiling():
            # Same reason as `_stream_decode_qzeros`: the cache key is
            # untraceable and the cache publication is a state write. A traced
            # graph reads the eager-built cache, or converts `g_idx` inline.
            cached = self._g_idx_long_cache
            if cached is not None and cached.device == target_device:
                return cached
            if self.g_idx.device == target_device:
                return self.g_idx.long()
            return self.g_idx.to(device=target_device, dtype=torch.long)

        cache_state = (tensor_cache_key(self.g_idx), target_device)
        if self._g_idx_long_cache is not None and self._g_idx_long_cache_state == cache_state:
            return self._g_idx_long_cache

        if self.g_idx.device == target_device:
            self._g_idx_long_cache = self.g_idx.long()
        else:
            non_blocking = self.g_idx.device.type == "cpu" and target_device.type in {"cuda", "xpu", "npu"}
            self._g_idx_long_cache = self.g_idx.to(
                device=target_device,
                dtype=torch.long,
                non_blocking=non_blocking,
            )

        self._g_idx_long_cache_state = cache_state
        return self._g_idx_long_cache

    def _stream_reset_cache(self):
        self._zeros_cache = None
        self._zeros_cache_state = None
        self._g_idx_long_cache = None
        self._g_idx_long_cache_state = None
        self._embedding_zeros = None
        self._embedding_zeros_key = None
        self._embedding_row_state = None

    def _maybe_offload_g_idx_to_cpu(self):
        if self.training or self.g_idx is None:
            return
        if self.g_idx.device.type not in {"cuda", "xpu", "npu"}:
            return
        # Keep original device g_idx when Triton dequant is active/usable.
        if self._triton_dequant_enabled and self._can_use_triton_dequant():
            return
        self.g_idx = self.g_idx.to(device="cpu")
        if self._g_idx_long_cache is not None:
            self._g_idx_long_cache_state = (tensor_cache_key(self.g_idx), self._g_idx_long_cache.device)

    def _device_cache_key(self, device: torch.device) -> int:
        if device.index is not None:
            return device.index
        return torch.cuda.current_device()

    def _stream_get_dequant_stream(self, device: torch.device) -> torch.cuda.Stream:
        key = self._device_cache_key(device)
        stream = self._stream_dequant_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._stream_dequant_streams[key] = stream
        return stream

    def _stream_get_workspace(
        self,
        device: torch.device,
        dtype: torch.dtype,
        tile: int,
    ) -> list[torch.Tensor]:
        return [
            torch.empty((self.in_features, tile), dtype=dtype, device=device)
            for _ in range(self._stream_double_buffers)
        ]

    def _should_use_streaming(self, x: torch.Tensor) -> bool:
        if not self._streaming_enabled:
            return False
        # Streaming decode assumes the 2/4/8-bit wf shift-buffer layout; 3-bit
        # and planar 5/6/7-bit widths use the eager dequant path instead.
        if self.bits not in (2, 4, 8):
            return False
        if x.device.type != "cuda":
            return False
        if not HAS_CUDA:
            return False
        if any(tensor.device != x.device for tensor in (self.qweight, self.qzeros, self.scales)):
            return False
        # Torch kernels with num_itr > 1 already tiled differently.
        if self.g_idx.shape[0] // x.shape[-1] != 1:
            return False
        if torch.is_grad_enabled() and (
            x.requires_grad or self.scales.requires_grad or (self.bias is not None and self.bias.requires_grad)
        ):
            return False
        if not self._decoded_weight_fits(x.dtype):
            # A decoded weight outside the activation dtype cannot be staged in
            # the x.dtype tile buffer; the eager path keeps the reduction wide.
            return False
        return True

    def enable_streaming(self, enabled: bool = True, tile_cols: int = None):
        self._streaming_enabled = enabled
        if tile_cols is not None:
            self._stream_tile_cols = tile_cols
        self._stream_reset_cache()
        return self

    def enable_weight_cache(self, enabled: bool = True):
        """Compatibility shim: dense inference weight caching is no longer supported."""
        self._cache_enabled = False
        self.clear_weight_cache()
        if enabled:
            warnings.warn(
                "Dense weight caching is no longer supported; weights remain quantized between forwards.",
                FutureWarning,
                stacklevel=2,
            )
        return self

    def clear_weight_cache(self):
        """Invalidate permitted zero-point and group-index metadata caches."""
        self._gptq_gemm_uniform_value = False
        self._gptq_gemm_uniform_id = None
        self._gptq_gemm_uniform_version = None
        self._stream_reset_cache()
        self._refresh_decode_dtypes()

    @property
    def _gptq_gemm_uniform_group(self) -> bool:
        return self._gptq_gemm_uniform_group_for_current_g_idx()

    @_gptq_gemm_uniform_group.setter
    def _gptq_gemm_uniform_group(self, value: bool) -> None:
        self._gptq_gemm_uniform_value = bool(value)

    def _gptq_gemm_uniform_group_for_current_g_idx(self) -> bool:
        """Return the fused-GEMM layout flag for the current ``g_idx``.

        The comparison itself is cheap, but it is still skipped on repeated
        forwards. Identity notices replacement, move, and deepcopy; ``_version``
        notices in-place writes.
        """
        g_idx = self.g_idx
        try:
            version = g_idx._version
        except RuntimeError:
            version = None
        if (
            id(g_idx) == self._gptq_gemm_uniform_id
            and version == self._gptq_gemm_uniform_version
        ):
            return self._gptq_gemm_uniform_value

        expected_g_idx = (
            torch.arange(
                self.in_features,
                device=g_idx.device,
                dtype=g_idx.dtype,
            )
            // self.group_size
        )
        self._gptq_gemm_uniform_value = bool(
            torch.equal(g_idx, expected_g_idx)
            and self.in_features % self.group_size == 0
        )
        self._gptq_gemm_uniform_id = id(g_idx)
        self._gptq_gemm_uniform_version = version
        return self._gptq_gemm_uniform_value

    def enable_lookahead(self, enabled: bool = True):
        """Compatibility shim: speculative dense-weight prefetching is disabled."""
        self._lookahead_enabled = False
        if enabled:
            warnings.warn(
                "Dense weight lookahead is no longer supported; dequantization is local to each forward.",
                FutureWarning,
                stacklevel=2,
            )
        return self

    def set_lookahead_next(self, module: "TorchLinear"):
        """Accept legacy lookahead configuration without retaining target modules."""
        if module is None or isinstance(module, TorchLinear):
            return self

        if isinstance(module, Iterable):
            targets = tuple(m for m in module if m is not None)
            for target in targets:
                if not isinstance(target, TorchLinear):
                    raise TypeError("lookahead targets must be TorchLinear modules or None")
            return self

        raise TypeError("lookahead target must be TorchLinear, iterable of TorchLinear, or None")

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

    def _can_use_triton_dequant(self) -> bool:
        if not _TRITON_DEQUANT_AVAILABLE:
            return False
        if self.training:
            return False
        if self.qweight is None or self.qzeros is None or self.scales is None or self.g_idx is None:
            return False
        if self.qweight.device.type != "cuda":
            return False
        if self.bits not in (2, 3, 4, 8):
            return False
        if self.bits == 3 and self.pack_dtype != torch.int32:
            return False
        # Planar 3-bit words do not match the continuous Triton decode layout.
        if self.planar:
            return False
        # Triton assumes complete zero-point words when computing group strides.
        if self.bits in (2, 4, 8) and self.out_features % self.pack_factor:
            return False
        if not (self.qweight.is_contiguous() and self.qzeros.is_contiguous() and self.scales.is_contiguous()):
            return False
        # g_idx is stored as int32 tensor; ensure it resides on the same device.
        if self.g_idx.device != self.qweight.device:
            return False
        return True

    def _dequantize_weight_triton(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        # Use the Triton helper to decode weights directly on device. The
        # kernel accumulates in FP32 but stores the requested dtype, so a half
        # scale that overflows must ask for the range-safe decode (#5606).
        dtype = self._resolve_decode_dtype(dtype)
        weights = triton_dequant(
            dtype,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        )
        return weights

    #: Activation dtypes whose range answer is precomputed for a traced graph.
    _DECODE_FIT_ACTIVATIONS = (torch.float16, torch.bfloat16, torch.float32)

    def _refresh_decode_dtypes(self) -> None:
        """Precompute the range-safe decode dtypes for the live scales.

        ``scales_fit_dtype`` runs a host reduction, which a traced graph cannot
        execute, so the answers are computed here and read back as plain Python
        constants (dynamo guards on them). ``clear_weight_cache()`` refreshes
        them on every checkpoint load, device/dtype move and training
        transition. ``None`` is the decoder's own dtype; the fit entries say
        whether the decoded weights are representable in an activation dtype.
        """
        scales = getattr(self, "scales", None)
        self._decode_dtype_cache = decode_dtype_for_scales(scales, self.bits)
        self._decode_fit_cache = {
            dtype: scales_fit_dtype(scales, self.bits, dtype)
            for dtype in self._DECODE_FIT_ACTIVATIONS
        }
        self._decode_dtype_cache_key = tensor_cache_key(scales)

    def _ensure_decode_caches(self) -> None:
        """Refresh the eager-built range answers after a scales change."""
        cached = self._decode_dtype_cache
        scales = getattr(self, "scales", None)
        # The range check is a host reduction, so it must not run inside a
        # traced graph or a CUDA-graph capture; both read the precomputed
        # answer instead (the eager warm-up built it).
        capturing = (
            scales is not None
            and scales.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
        if not dynamo_is_compiling() and not capturing:
            if cached is None or self._decode_dtype_cache_key != tensor_cache_key(scales):
                self._refresh_decode_dtypes()

    def _decode_dtype(self) -> torch.dtype:
        """Return a dtype that decodes every stored weight without range loss.

        The stored scale dtype is kept while it holds every decoded weight;
        otherwise the decode runs in FP32 (or FP64 for a pathological BF16
        scale), exactly like the base decoder (#5606).
        """
        self._ensure_decode_caches()
        cached = self._decode_dtype_cache
        if cached is not None:
            return cached
        return decode_dtype_for_scales(self.scales, self.bits)

    def _resolve_decode_dtype(self, dtype: torch.dtype | None) -> torch.dtype:
        """Pick the decode dtype for a caller-requested ``dtype``.

        A caller may ask for a dtype narrower than the range-safe decode; the
        dense forward asks for its activation dtype.  Honour the request only
        when every decoded weight fits it, and keep the range-safe decode
        (#5606) otherwise.
        """
        safe = self._decode_dtype()
        if dtype is None or dtype == safe:
            return safe
        if not self._decoded_weight_fits(dtype):
            return safe
        return dtype

    def _forward_decode_dtype(self, activation_dtype: torch.dtype) -> torch.dtype:
        """Decode dtype for the dense fallback of one forward.

        The fallback narrows the reconstructed weight to ``activation_dtype``
        anyway.  Decoding in a different half dtype first rounds every weight
        twice and then copies the whole matrix: a GPTQ checkpoint stores FP16
        scales, so a BF16 model decoded FP16 and narrowed to BF16, one bf16 ulp
        away from the single rounding the fused kernels perform (#5740).  Ask
        for the activation dtype when every decoded weight fits it; a decode
        that does not fit keeps the range-safe dtype.
        """
        if (
            activation_dtype in _HALF_DTYPES
            and self.scales.dtype != activation_dtype
            and self._decoded_weight_fits(activation_dtype)
        ):
            return activation_dtype
        return self._decode_dtype()

    def _decoded_weight_fits(self, activation_dtype: torch.dtype) -> bool:
        """Whether every decoded weight is representable in ``activation_dtype``.

        This gates narrowing a reconstructed weight to the forward's operand
        dtype: an overflowing decode must keep the reduction wide instead
        (#5606).
        """
        self._ensure_decode_caches()
        cached = self._decode_fit_cache
        if cached is not None:
            resolved = cached.get(activation_dtype)
            if resolved is not None:
                return resolved
        return scales_fit_dtype(self.scales, self.bits, activation_dtype)

    def _dequantize_weight_cached_248(
        self, num_itr: int = 1, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        self._init_wf_unsqueeze_buffers()
        zeros = self._stream_decode_qzeros()
        g_idx_long = self._stream_g_idx_long(target_device=self.qweight.device)
        self._maybe_offload_g_idx_to_cpu()

        # Decode in a dtype that holds every stored weight. A half-precision
        # scale overflows on the multiply here, before the caller widens the
        # result, so the eval fast path used to return `inf`/`NaN` (#5606).
        # The scale dtype is kept whenever it is safe, so a finite fast path is
        # bitwise unchanged.
        decode_dtype = self._resolve_decode_dtype(dtype)
        wide = decode_dtype != self.scales.dtype
        if wide:
            zeros = zeros.to(decode_dtype)

        weight = torch.bitwise_and(
            _right_shift_unpack(
                self.qweight.unsqueeze(1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
                self.dequant_dtype,
            ),
            self.maxq,
        )
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        if num_itr == 1:
            if wide:
                return self.scales.to(decode_dtype)[g_idx_long] * (
                    weight[:self.in_features].to(decode_dtype) - zeros[g_idx_long]
                )
            return self.scales[g_idx_long] * (weight[:self.in_features] - zeros[g_idx_long])

        num_dim = self.g_idx.shape[0] // num_itr
        out_dim = weight.shape[1] // num_itr
        scales = self.scales.to(decode_dtype) if wide else self.scales
        weights = []
        for i in range(num_itr):
            row_start = i * num_dim
            row_end = (i + 1) * num_dim
            col_start = i * out_dim
            col_end = (i + 1) * out_dim if i < (num_itr - 1) else weight.shape[1]

            scale_i = scales[:, col_start:col_end]
            weight_i = weight[row_start:row_end, col_start:col_end]
            zeros_i = zeros[:, col_start:col_end]
            g_idx_i = g_idx_long[row_start:row_end]
            if wide:
                weight_i = weight_i.to(decode_dtype)
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))

        return torch.cat(weights, dim=1)

def dequantize_model(model: PreTrainedModel, dtype: Optional[torch.dtype] = None):
    # Use the model's compute dtype, not the checkpoint storage dtype: GPTQ
    # restores fp16 scales even for a bf16 model, so ``scales.dtype`` would
    # turn a bf16 model into fp16 dense layers that fail in forward.
    target_dtype = None
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchLinear):
            raise ValueError(
                "Only models loaded using TorchLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.GPTQ_TORCH."
            )

        if isinstance(module, TorchLinear):
            if target_dtype is None:
                target_dtype = resolve_dequantize_dtype(model, module, dtype)
            # Create a new Linear layer with dequantized weights
            # Keep the quantized module's device: a dense layer built on CPU
            # inside a CUDA model fails the next forward with a device
            # mismatch. The shell is built on meta because both parameters
            # are replaced below.
            device = module.qweight.device
            new_module = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                device="meta",
            )
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to(device, target_dtype))
            new_module.bias = (
                nn.Parameter(module.bias.detach().to(device, target_dtype))
                if module.bias is not None else None
            )

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


class TorchQuantEmbeddings(TorchLinear):
    """Quantized embedding layer backed by the Torch GPTQ dequant path."""

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 20, FORMAT.GPTQ_V2: 20, FORMAT.GPTQ_P: 20}
    # Embeddings are selected by module role in create_quant_module(), and must
    # not be discovered as a general backend: their forward contract accepts
    # integer token IDs, not hidden states.
    SUPPORTS_BACKEND_SELECTION = False
    # CPU, torch 2.16.0.dev20261021+cpu: eager=True, graph_breaks=0, and
    # dynamo/aot_eager/inductor all pass for both paths: the whole-table
    # `F.embedding(dequantize_weight())` decode and the row-decode gather once
    # the eager warmup has built `_embedding_row_decode_state()`. A traced
    # forward reads that cached metadata instead of re-keying it with
    # `tensor_cache_key`, whose `is_inference()` is a hard graph break.
    #
    # CUDA keeps ``EAGER``: not re-measured on this CPU-only host.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_COMPILE_LEVEL_BY_DEVICE = {"cpu": COMPILE_LEVEL.INDUCTOR}
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for this kernel in this change; the INDUCTOR level is opt-in via "
        "optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = False
    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = []

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

    # Row decode reads the packed words that back the requested tokens and
    # writes only those rows; whole-table decode writes and rereads the complete
    # unquantized table. Both crossover points below are measured, not guessed.
    #
    # 1. Table size. A small table is cheaper to decode whole: the fixed launch
    #    cost of the row gather dominates. Measured FP16 single-token decode on
    #    an independent NVIDIA and an AMD ROCm host: full-table decode is 2.5x
    #    faster at a 32 MiB table (68.5 us vs 27.5 us), the paths cross at
    #    64 MiB (70.4 vs 89.3 us), and row decode is 2.7x faster at 128 MiB.
    # 2. Token count. Every extra token adds gather + scratch traffic
    #    proportional to its rows, while whole-table decode is constant. The
    #    measured crossover is ~1K tokens for realistic vocabularies
    #    (151936 x 3584: 83 us row vs 1615 us full at 1024 tokens; both ~1.5 ms
    #    at 1025 tokens).
    EMBEDDING_ROW_DECODE_LIMIT = 1024
    EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES = 64 * 1024 * 1024

    #: CPU row decode decodes bounded chunks so the transient code/scale
    #: tensors stay proportional to the chunk instead of the whole query. CPU
    #: has no kernel-launch amortization to recover, so the accelerator cutoffs
    #: above do not apply there (#5772).
    EMBEDDING_CPU_ROW_DECODE_CHUNK = 8192

    #: Dtype the forward returns. A quantized embedding's stored ``scales`` keep
    #: the checkpoint dtype (FP16 for GPTQ), which is not the model's compute
    #: dtype; returning it promoted a BF16 residual stream to FP32 and made a
    #: bf16 ``lm_head`` reject the activations (#5116). Set from the embedding
    #: the module replaced, or from the model dtype on a requantize.
    #:
    #: The value is held by a zero-element non-persistent buffer, not a plain
    #: attribute: ``.to()``/``.float()``/``.half()`` only convert parameters and
    #: buffers, so an attribute kept the pre-cast dtype and the embedding then
    #: fed BF16 activations into the FP32 layers ``model.float()`` produced
    #: (#5394).
    _EMBEDDING_OUTPUT_DTYPE_BUFFER = "_embedding_output_dtype_anchor"

    def set_embedding_output_dtype(self, dtype: Optional[torch.dtype]) -> None:
        """Record the dtype this embedding must return to the residual stream."""

        name = self._EMBEDDING_OUTPUT_DTYPE_BUFFER
        anchor = self._buffers.get(name)
        if dtype is None:
            # Reset to the scale dtype, which is the historical fallback and
            # keeps following later module casts.
            if anchor is not None:
                self._buffers[name] = torch.zeros(0, dtype=self.scales.dtype)
            return
        if anchor is None:
            self.register_buffer(name, torch.zeros(0, dtype=dtype), persistent=False)
        else:
            self._buffers[name] = anchor.to(dtype)

    def embedding_output_dtype(self) -> torch.dtype:
        """The dtype ``forward`` returns: the recorded one, else the scale dtype."""

        anchor = self._buffers.get(self._EMBEDDING_OUTPUT_DTYPE_BUFFER)
        return anchor.dtype if anchor is not None else self.scales.dtype

    def _embedding_row_decode_state(self):
        """Per-vocabulary index metadata for row decode, derived from packed tensors.

        Everything here is indexing metadata (word row per token, intra-word
        shift, group per token) plus the decoded zero-point table; no unpacked
        weight is retained. The state is keyed on the packed tensors so a
        replaced or mutated checkpoint invalidates it.
        """
        if dynamo_is_compiling():
            # `tensor_cache_key` reads `is_inference()`/`_version`/`data_ptr()`,
            # all hard dynamo graph breaks, and publishing the metadata is a
            # state write. The eager warmup built it for this constant
            # checkpoint, so a traced graph reads it; without a warmup it
            # returns None and the forward takes the whole-table
            # `F.embedding` path, which is traceable on its own.
            state = getattr(self, "_embedding_row_state", None)
            return None if state is None else state[1]
        key = (
            tensor_cache_key(self.qweight),
            tensor_cache_key(self.g_idx),
            tensor_cache_key(self.qzeros),
            self.bits,
            self.planar,
            self.pack_dtype,
        )
        state = getattr(self, "_embedding_row_state", None)
        if state is not None and state[0] == key:
            return state[1]

        device = self.qweight.device
        if self.g_idx.numel() != self.in_features:
            return None

        tokens = torch.arange(self.in_features, device=device, dtype=torch.int32)
        word_rows = None
        shifts = None
        if not self.planar:
            if self.bits in (2, 4, 8):
                # One storage word holds `pack_factor` consecutive codes, so the
                # word row and the intra-word shift follow the actual
                # `pack_dtype_bits` (8/16/32/64), not a fixed 32-bit width.
                word_rows = torch.div(
                    tokens, self.pack_factor, rounding_mode="floor"
                ).to(torch.int32)
                shifts = (tokens.remainder(self.pack_factor) * self.bits).to(torch.int32)
            else:
                # Continuous 3-bit GPTQ packs 32 codes into three 32-bit words;
                # `_validate()` admits this layout only for torch.int32 storage.
                word_rows = (torch.div(tokens, 32, rounding_mode="floor") * 3).to(torch.int32)

        groups = self.g_idx.to(device=device, dtype=torch.int32)
        num_groups = self.scales.shape[0]
        if int(groups.min()) < -num_groups or int(groups.max()) >= num_groups:
            return None
        if int(groups.min()) < 0:
            groups = torch.where(groups < 0, groups + num_groups, groups)
        groups = groups.to(torch.int32)

        metadata = {
            "word_rows": word_rows,
            "shifts": shifts,
            "groups": groups,
            "zeros": self._embedding_zero_points(),
        }
        self._embedding_row_state = (key, metadata)
        return metadata

    def _embedding_zero_points(self) -> torch.Tensor:
        """Decode the zero-point table once; it is permitted packed metadata.

        Kept apart from ``_zeros_cache`` (the streaming/GEMV layout) so a
        model that alternates between row decode and whole-table decode does
        not invalidate either decoding.
        """
        key = (tensor_cache_key(self.qzeros), self.scales.shape, self.bits, self.planar)
        if self._embedding_zeros is not None and self._embedding_zeros_key == key:
            return self._embedding_zeros
        self._embedding_zeros = self.unpack_zero_points()
        self._embedding_zeros_key = key
        return self._embedding_zeros

    def _embedding_row_codes(self, rows: torch.Tensor, state) -> torch.Tensor:
        """Decode packed codes for `rows` only, keeping `qweight` untouched."""
        qweight = self.qweight

        if self.planar:
            return planar_decode_rows(qweight, rows, self.bits)

        word_rows = state["word_rows"].index_select(0, rows)
        if self.bits in (2, 4, 8):
            words = qweight.index_select(0, word_rows)
            shifts = state["shifts"].index_select(0, rows).view(-1, 1)
            codes = torch.bitwise_and(words >> shifts, self.maxq)
            return codes.to(self.dequant_dtype)

        # Continuous 3-bit: 32 codes per 3-word block in the 10-1-10-1-10
        # layout, so a handful of codes straddle two words. Only the requested
        # rows are read; the neighbouring word is fetched per row.
        base = qweight.index_select(0, word_rows)
        word1 = qweight.index_select(0, word_rows + 1)
        word2 = qweight.index_select(0, word_rows + 2)
        idx = rows.remainder(32).view(-1, 1)
        codes = (base >> (idx * 3)) & 0x7
        codes = torch.where(idx == 10, ((base >> 30) & 0x3) | ((word1 & 0x1) << 2), codes)
        codes = torch.where((idx >= 11) & (idx <= 20), (word1 >> (1 + (idx - 11) * 3)) & 0x7, codes)
        codes = torch.where(idx == 21, ((word1 >> 31) & 0x1) | ((word2 & 0x3) << 1), codes)
        codes = torch.where(idx >= 22, (word2 >> (2 + (idx - 22) * 3)) & 0x7, codes)
        return codes.to(self.dequant_dtype)

    def _dequantize_embedding_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Dequantize only the requested embedding rows.

        The result is transient scratch for this forward: only the packed
        `qweight`/`qzeros`/`scales`/`g_idx` metadata persists afterwards.
        CPU queries are decoded in bounded chunks so the intermediate code and
        scale tensors stay proportional to the chunk, not the whole query.
        """
        self._init_wf_unsqueeze_buffers()
        state = self._embedding_row_decode_state()
        rows = input_ids.reshape(-1).to(device=self.qweight.device, dtype=torch.int32)
        decode_dtype = self._decode_dtype()
        chunk = (
            int(self.EMBEDDING_CPU_ROW_DECODE_CHUNK)
            if self.qweight.device.type == "cpu"
            else 0
        )
        if chunk <= 0 or rows.numel() <= chunk:
            return self._dequantize_embedding_row_chunk(rows, state, decode_dtype)
        result = torch.empty(
            rows.numel(), self.out_features, dtype=decode_dtype, device=rows.device
        )
        for start in range(0, rows.numel(), chunk):
            stop = min(start + chunk, rows.numel())
            result[start:stop] = self._dequantize_embedding_row_chunk(
                rows[start:stop], state, decode_dtype
            )
        return result

    def _dequantize_embedding_row_chunk(self, rows, state, decode_dtype):
        """Decode one bounded batch of packed embedding rows."""
        codes = self._embedding_row_codes(rows, state)
        groups = state["groups"].index_select(0, rows)
        scales = self.scales.index_select(0, groups)
        zeros = state["zeros"].index_select(0, groups)
        # The row gather multiplies in the scale dtype too; a half scale that
        # overflows must decode wide exactly like the whole-table path (#5606).
        if decode_dtype != self.scales.dtype:
            return scales.to(decode_dtype) * (codes.to(decode_dtype) - zeros.to(decode_dtype))
        return scales * (codes - zeros)

    def _embedding_row_decode_eligible(self, input_ids: torch.Tensor) -> bool:
        """Cheap gate for row decode; CPU and accelerator policies differ.

        The GPU cutoffs amortize kernel-launch cost and were measured on
        accelerators. CPU gathers only the queried rows, so it uses row decode
        whenever the query is no wider than the table; whole-table decode is
        reserved for queries with more lookups than vocabulary rows, where it
        decodes each row once instead of repeatedly (#5772).
        """
        if self.training or self.qweight is None:
            return False
        if input_ids.dtype not in (torch.int32, torch.int64):
            return False
        if input_ids.device != self.qweight.device:
            return False
        if self.qweight.device.type == "cpu":
            return input_ids.numel() <= self.in_features
        return (
            input_ids.numel() <= self.EMBEDDING_ROW_DECODE_LIMIT
            and self.in_features * self.out_features * self.scales.element_size()
            >= self.EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES
        )

    def forward(self, input_ids: torch.Tensor):
        output_dtype = self.embedding_output_dtype()
        if input_ids.numel() == 0:
            return torch.empty(
                (*input_ids.shape, self.out_features),
                dtype=output_dtype,
                device=self.scales.device,
            )
        if (
            self._embedding_row_decode_eligible(input_ids)
            and self._embedding_row_decode_state() is not None
        ):
            rows = self._dequantize_embedding_rows(input_ids)
            return rows.to(output_dtype).reshape(*input_ids.shape, self.out_features)
        return F.embedding(input_ids, self.dequantize_weight()).to(output_dtype)


__all__ = ["TorchLinear", "TorchQuantEmbeddings", "dequantize_model"]
