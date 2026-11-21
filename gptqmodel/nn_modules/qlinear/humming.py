# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Humming weight-only CUDA GEMM kernels for GPTQ and AWQ checkpoints.

Humming (https://github.com/vllm-project/humming) is an external, optional
JIT-compiled CUDA library for W<8-bit GEMM. This module is the GPTQModel
counterpart of vLLM's ``vllm/model_executor/layers/quantization/humming.py`` and
SGLang's ``sglang/srt/layers/quantization/humming.py``: it keeps the checkpoint
parameters registered under their usual GPTQ/AWQ names, and ``post_init()``
converts them once into humming's transformed operands.

Because the conversion happens after the whole layer is loaded, the runtime
tensors are no longer a checkpoint layout. Like the Marlin and Swordfish
sub-kernels, a Humming-loaded model is therefore inference-only: reload it from
the original checkpoint (or pack with a checkpoint backend) before saving.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear, GPTQQuantLinear, WeightOnlyQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.humming import (
    HUMMING_AWQ_BITS,
    HUMMING_GPTQ_BITS,
    HUMMING_GROUP_SIZES,
    create_humming_locks,
    humming_apply_linear,
    humming_import_exception,
    humming_runtime_error,
    humming_supports_device,
    humming_validate_device,
    prepare_humming_awq_tensors,
    prepare_humming_fp8_tensors,
    prepare_humming_gptq_tensors,
    prepare_humming_mxfp4_tensors,
)
from ...utils.logger import setup_logger
from ...utils.marlin import replace_parameter
from ...utils.rocm import IS_ROCM


log = setup_logger()

_HUMMING_ROCM_UNSUPPORTED = (
    "Humming (BACKEND.HUMMING) is CUDA-only: it ships NVIDIA SASS/PTX kernels "
    "and has no HIP port. Use a portable `*Torch`/`*Triton` backend on ROCm."
)

_HUMMING_CPU_UNSUPPORTED = (
    "Humming kernels are CUDA-only and convert their weights on the GPU in "
    "post_init(); a CPU/offloaded layer cannot run them. Use a portable "
    "`*Torch` backend for CPU inference."
)


class _HummingKernelMixin:
    """Shared Humming lifecycle: device gating, conversion, and forward."""

    # Runtime operands produced by post_init().
    humming_config = None
    weight_schema = None
    has_zero_point = False

    # Checkpoint tensor names that post_init() converts in place. GPTQ/AWQ keep
    # humming's operands in ``qweight``/``scales``; FP8 checkpoints already use
    # ``weight``/``weight_scale_inv`` and have no zero point at all.
    HUMMING_WEIGHT_ATTR = "qweight"
    HUMMING_SCALE_ATTR = "scales"
    HUMMING_ZERO_POINT_ATTR: Optional[str] = "qzeros"

    def _humming_prepare_device(self) -> torch.device:
        device = self.runtime_device()
        if device is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: no quantized tensor to convert; "
                "load the checkpoint (or call pack()) before post_init()."
            )
        if device.type != "cuda":
            raise NotImplementedError(f"{self.__class__.__name__}: {_HUMMING_CPU_UNSUPPORTED}")
        humming_validate_device(device)
        return device

    def _humming_replace(self, name: str, tensor: torch.Tensor) -> None:
        """Swap one checkpoint tensor for its transformed operand.

        ``bias`` is registered as a buffer by the quant-linear base classes while
        the packed tensors are parameters, and ``register_parameter`` refuses to
        shadow an existing buffer, so dispatch on the current registration.
        """

        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            replace_parameter(self, name, tensor)

    def _humming_install(
        self,
        *,
        device: torch.device,
        config,
        weight_schema,
        tensors: dict,
        zero_point: bool,
    ) -> None:
        """Replace the checkpoint tensors with humming's transformed operands."""

        self._humming_replace(self.HUMMING_WEIGHT_ATTR, tensors["weight"])
        self._humming_replace(self.HUMMING_SCALE_ATTR, tensors["weight_scale"])

        zero_point_attr = self.HUMMING_ZERO_POINT_ATTR
        if zero_point:
            if "zero_point" not in tensors:
                raise RuntimeError(
                    f"{self.__class__.__name__}: humming did not produce a `zero_point` "
                    "tensor for an asymmetric checkpoint."
                )
            self._humming_replace(zero_point_attr, tensors["zero_point"])
            self.has_zero_point = True
        elif zero_point_attr is not None:
            self._humming_replace(
                zero_point_attr,
                torch.empty(0, dtype=self.pack_dtype, device=device),
            )
            self.has_zero_point = False
        else:
            self.has_zero_point = False

        if self.bias is not None:
            bias = tensors.get("bias")
            if bias is None:
                bias = self.bias.data
            self._humming_replace(
                "bias", bias.to(dtype=self.compute_dtype).contiguous()
            )

        # Stream-K scratch is runtime-only state: keep it out of the state dict
        # but registered so device/dtype moves and offload hooks track it.
        if "locks" in self._buffers:
            self._buffers["locks"] = create_humming_locks(device)
        else:
            self.register_buffer("locks", create_humming_locks(device), persistent=False)

        self.humming_config = config
        self.weight_schema = weight_schema

    def _humming_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.humming_config is None:
            raise RuntimeError(
                f"{self.__class__.__name__} `{self.name}` has not been converted for "
                "humming yet; post_init() must run after the weights are loaded."
            )

        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        input_dtype = x.dtype
        # The native ABI reads one contiguous [M, K] activation.
        input_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if input_2d.dtype != self.compute_dtype:
            input_2d = input_2d.to(self.compute_dtype)

        zero_point = None
        if self.has_zero_point and self.HUMMING_ZERO_POINT_ATTR is not None:
            zero_point = getattr(self, self.HUMMING_ZERO_POINT_ATTR)

        output = humming_apply_linear(
            config=self.humming_config,
            inputs=input_2d,
            weight=getattr(self, self.HUMMING_WEIGHT_ATTR),
            weight_scale=getattr(self, self.HUMMING_SCALE_ATTR),
            zero_point=zero_point,
            bias=self.bias,
            locks=self.locks,
        )

        result = output.reshape(x.shape[:-1] + (self.out_features,))
        if result.dtype != input_dtype:
            result = result.to(input_dtype)
        if self.adapter is not None:
            result = self.adapter.apply(x=x, out=result)
        return result

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        # Architecture first: humming ships NVIDIA kernel images, so a ROCm host
        # can never run it even when the Python package imports.
        if IS_ROCM:
            return False, NotImplementedError(_HUMMING_ROCM_UNSUPPORTED)
        if humming_import_exception is not None:
            return False, ImportError(humming_runtime_error())
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE | torch.device):
        super().validate_device(device)
        if (device.type if isinstance(device, torch.device) else device) not in (
            "cuda",
            DEVICE.CUDA,
        ):
            return
        if IS_ROCM:
            raise NotImplementedError(_HUMMING_ROCM_UNSUPPORTED)
        targets = (
            (device,)
            if isinstance(device, torch.device)
            else tuple(
                torch.device("cuda", index) for index in range(torch.cuda.device_count())
            )
        )
        if not targets:
            raise NotImplementedError("Humming kernel requires CUDA.")
        for target in targets:
            if not humming_supports_device(target):
                humming_validate_device(target)


class HummingLinear(_HummingKernelMixin, GPTQQuantLinear):
    """GPTQ sub-kernel of the shared Humming backend.

    ``BACKEND.HUMMING`` plus the checkpoint method/format selects this class; no
    separate GPTQ-specific backend name is required.
    """

    SUPPORTS_BACKENDS = [BACKEND.HUMMING]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Priority 0 keeps explicit selection working while leaving AUTO on the
    # default kernels: humming JIT-compiles a kernel per layer shape on first
    # use, so it is opt-in rather than an automatic replacement.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = list(HUMMING_GPTQ_BITS)
    SUPPORTS_GROUP_SIZE = list(HUMMING_GROUP_SIZES)
    # Humming's GPTQ schema asserts against act-order checkpoints.
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_SHARDED_LOAD = True
    SUPPORTS_TRAINING = False
    # Tile padding is applied to the packed tensors in post_init(), not to the
    # checkpoint, so layers keep their logical shapes.
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "humming"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        register_buffers: bool = False,
        adapter: Adapter = None,
        format=None,
        **kwargs,
    ):
        if humming_import_exception is not None:
            raise ValueError(
                "Trying to use the humming backend, but the runtime requirements "
                f"were not met: {humming_runtime_error()}"
            )

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.HUMMING),
            adapter=adapter,
            register_buffers=False,
            format=format,
            **kwargs,
        )

        if self.planar:
            raise NotImplementedError(
                f"{self.__class__.__name__} cannot read the planar {self.bits}-bit GPTQ "
                "layout (format=gptq_p); use a checkpoint-format kernel instead."
            )

        # Checkpoint-compatible GPTQ buffer shapes so `from_quantized` loading
        # and manual `pack` write to the same fields. post_init() converts them
        # into humming's transformed operands.
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    (self.in_features * self.bits // self.pack_dtype_bits, self.out_features),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(
                torch.empty(self.in_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        grouped_rows = (self.in_features + self.group_size - 1) // self.group_size
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty((grouped_rows, self.out_features), dtype=self.compute_dtype),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    (grouped_rows, self.out_features * self.bits // self.pack_dtype_bits),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features,), dtype=self.compute_dtype))
        else:
            self.bias = None

        self.has_zero_point = False

    @classmethod
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err

        in_features = args.get("in_features")
        group_size = args.get("group_size")
        if (
            in_features is not None
            and group_size is not None
            and group_size not in (-1, in_features)
            and in_features % group_size != 0
        ):
            return False, NotImplementedError(
                f"{cls} requires in_features % group_size == 0, "
                f"got in_features={in_features}, group_size={group_size}"
            )
        return True, None

    def post_init(self):
        device = self._humming_prepare_device()

        qzeros = self.qzeros
        if not self.sym:
            if qzeros is None or qzeros.numel() == 0:
                raise NotImplementedError(
                    f"{self.__class__.__name__}: asymmetric (sym=False) Humming GPTQ "
                    "requires the checkpoint's qzeros tensor."
                )
            if self.qzero_format() == 1:
                # GPTQ v1 checkpoints store (zero_point - 1); humming reads the
                # logical zero points, so shift each field before packing.
                from ...utils.model_dequant import _correct_gptq_v1_qzeros

                # The unpack/repack helper works on CPU words, so return the
                # corrected zero points to the layer's device.
                qzeros = _correct_gptq_v1_qzeros(qzeros.data, self.bits).to(
                    device=device, dtype=self.pack_dtype
                )

        bias = None
        if self.bias is not None:
            bias = self.bias.data.to(device=device, dtype=self.compute_dtype).contiguous()

        config, weight_schema, tensors = prepare_humming_gptq_tensors(
            qweight=self.qweight.data,
            scales=self.scales.data,
            qzeros=qzeros,
            bias=bias,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=self.compute_dtype,
            device=device,
        )

        self._humming_install(
            device=device,
            config=config,
            weight_schema=weight_schema,
            tensors=tensors,
            zero_point=not self.sym,
        )
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._humming_forward(x)


class AwqHummingLinear(_HummingKernelMixin, AWQuantLinear):
    """AWQ sub-kernel of the shared Humming backend."""

    SUPPORTS_BACKENDS = [BACKEND.HUMMING]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 0}
    # Humming's AWQ converter only undoes the classic interleaved INT4 layout.
    SUPPORTS_BITS = list(HUMMING_AWQ_BITS)
    SUPPORTS_GROUP_SIZE = list(HUMMING_GROUP_SIZES)
    # AWQ has no activation reordering; the flag is accepted and ignored, like
    # the other AWQ sub-kernels.
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_SHARDED_LOAD = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_humming"

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        register_buffers: bool = False,
        adapter: Adapter = None,
        format=None,
        **kwargs,
    ):
        if humming_import_exception is not None:
            raise ValueError(
                "Trying to use the humming backend, but the runtime requirements "
                f"were not met: {humming_runtime_error()}"
            )

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.HUMMING),
            adapter=adapter,
            register_buffers=False,
            format=format,
            **kwargs,
        )

        # Classic AWQ GEMM layout: one int32 word per eight INT4 codes packed
        # along the output dimension.
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    (self.in_features, self.out_features * self.bits // self.pack_dtype_bits),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )
        grouped_rows = (self.in_features + self.group_size - 1) // self.group_size
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty((grouped_rows, self.out_features), dtype=self.compute_dtype),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    (grouped_rows, self.out_features * self.bits // self.pack_dtype_bits),
                    dtype=self.pack_dtype,
                ),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features,), dtype=self.compute_dtype))
        else:
            self.bias = None

        self.has_zero_point = False

    @classmethod
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err

        in_features = args.get("in_features")
        group_size = args.get("group_size")
        if (
            in_features is not None
            and group_size is not None
            and group_size not in (-1, in_features)
            and in_features % group_size != 0
        ):
            return False, NotImplementedError(
                f"{cls} requires in_features % group_size == 0, "
                f"got in_features={in_features}, group_size={group_size}"
            )
        return True, None

    def post_init(self):
        device = self._humming_prepare_device()

        bias = None
        if self.bias is not None:
            bias = self.bias.data.to(device=device, dtype=self.compute_dtype).contiguous()

        config, weight_schema, tensors = prepare_humming_awq_tensors(
            qweight=self.qweight.data,
            scales=self.scales.data,
            qzeros=self.qzeros,
            bias=bias,
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=self.compute_dtype,
            device=device,
        )

        self._humming_install(
            device=device,
            config=config,
            weight_schema=weight_schema,
            tensors=tensors,
            zero_point=bool(weight_schema.has_zero_point),
        )
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._humming_forward(x)


class HummingFP8Linear(_HummingKernelMixin, WeightOnlyQuantLinear):
    """FP8 (E4M3) sub-kernel of the shared Humming backend.

    The checkpoint layout is identical to ``TorchFP8Linear`` -- an E4M3
    ``weight`` plus a float32 ``weight_scale_inv`` -- so an FP8 checkpoint can be
    loaded by either backend. ``post_init()`` normalizes the stored inverse scale
    into humming's multiplier convention and hands the matrix to humming, which
    runs one FP8 GEMM instead of decoding the weights and calling a BLAS GEMM.
    """

    SUPPORTS_BACKENDS = [BACKEND.HUMMING]
    SUPPORTS_METHODS = [METHOD.FP8]
    # Priority 0: explicit selection only. Humming JIT-compiles one kernel per
    # layer shape, so it never silently replaces the default FP8 kernel.
    SUPPORTS_FORMATS = {FORMAT.FP8: 0}
    SUPPORTS_BITS = [8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # Same contract as the torch FP8 kernel: both buffers must be present or the
    # layer decodes to zeros.
    CHECKPOINT_REQUIRED_BUFFERS = ("weight", "weight_scale_inv")

    HUMMING_WEIGHT_ATTR = "weight"
    HUMMING_SCALE_ATTR = "weight_scale_inv"
    HUMMING_ZERO_POINT_ATTR = None

    QUANT_TYPE = "fp8"

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        register_buffers: bool = False,
        adapter: Adapter = None,
        format: str = "float8_e4m3fn",
        weight_scale_method: str = "row",
        weight_block_size: Optional[Tuple[int, int]] = None,
        weight_scale_semantics: str = "inverse",
        activation_quant_method: str = "none",
        **kwargs,
    ):
        if humming_import_exception is not None:
            raise ValueError(
                "Trying to use the humming backend, but the runtime requirements "
                f"were not met: {humming_runtime_error()}"
            )

        from .fp8 import (
            _normalize_fp8_activation_quant_method,
            _normalize_linear_scale_semantics,
        )

        self.fp8_format = format
        self.compute_dtype = kwargs.pop("dtype", None) or torch.float16
        self.weight_scale_method = weight_scale_method
        self.weight_block_size = (
            None if weight_block_size is None else tuple(int(v) for v in weight_block_size)
        )
        self.weight_scale_semantics = _normalize_linear_scale_semantics(weight_scale_semantics)
        self.activation_quant_method = _normalize_fp8_activation_quant_method(
            activation_quant_method
        )

        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.HUMMING),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            dtype=self.compute_dtype,
            **kwargs,
        )

        if register_buffers:
            self._allocate_buffers(bias=bias)
        else:
            self.bias = None

    @classmethod
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if args.get("activation_quant_method", "none") not in (None, "none"):
            return False, NotImplementedError(
                "Humming FP8 is weight-only (A16): per-token activation scaling "
                "needs humming's W8A8 input schema, which this backend does not "
                "expose yet."
            )
        if args.get("weight_scale_semantics", "inverse") not in ("inverse", "multiplier"):
            return False, ValueError(
                "Humming FP8 accepts only `inverse` or `multiplier` scale semantics."
            )
        return True, None

    def _scale_shape(self) -> Tuple[int, ...]:
        if self.weight_scale_method == "tensor":
            return (1,)
        if self.weight_scale_method == "row":
            return (self.out_features,)
        if self.weight_block_size is None:
            raise ValueError("Humming FP8 block scaling requires `weight_block_size`.")
        block_rows, block_cols = self.weight_block_size
        return (
            (self.out_features + block_rows - 1) // block_rows,
            (self.in_features + block_cols - 1) // block_cols,
        )

    def _allocate_buffers(self, *, bias: bool) -> None:
        fp8_dtype = getattr(torch, self.fp8_format, torch.float8_e4m3fn)
        weight = torch.zeros((self.out_features, self.in_features), dtype=fp8_dtype)
        scale = torch.ones(self._scale_shape(), dtype=torch.float32)

        if "weight" in self._buffers:
            self.weight = weight
        else:
            self.register_buffer("weight", weight)
        if "weight_scale_inv" in self._buffers:
            self.weight_scale_inv = scale
        else:
            self.register_buffer("weight_scale_inv", scale)

        if bias:
            bias_tensor = torch.zeros(self.out_features, dtype=torch.float32)
            if "bias" in self._buffers:
                self.bias = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def list_buffers(self):
        buffers = []
        for name in ("weight", "weight_scale_inv", "bias"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                buffers.append(tensor)
        return buffers

    def _expanded_block_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """Return the dense ``[out // bn, in // bk]`` block grid.

        Checkpoints may store a ceil-blocked grid (a partial last block), while
        humming indexes scales by whole blocks. Replicating the edge row/column
        is exact: the missing block rows/cols only cover padded weight rows.
        """

        if self.weight_block_size is None:
            raise ValueError("Humming FP8 block scaling requires `weight_block_size`.")
        block_n, block_k = self.weight_block_size
        dense_n = self.out_features // block_n
        dense_k = self.in_features // block_k
        if dense_n * block_n != self.out_features or dense_k * block_k != self.in_features:
            raise NotImplementedError(
                "Humming FP8 block scaling requires in/out features that are "
                f"multiples of the block size {self.weight_block_size}, got "
                f"({self.out_features}, {self.in_features})."
            )
        if tuple(scale.shape) == (dense_n, dense_k):
            return scale
        if scale.numel() == dense_n * dense_k:
            return scale.reshape(dense_n, dense_k)
        raise ValueError(
            "Humming FP8 block scale grid does not match the weight shape: "
            f"scale {tuple(scale.shape)} for weight "
            f"({self.out_features}, {self.in_features}) with block "
            f"{self.weight_block_size}."
        )

    def _multiplier_scale(self) -> torch.Tensor:
        scale = self.weight_scale_inv.detach().to(torch.float32)
        if self.weight_scale_semantics == "inverse":
            scale = torch.where(
                scale != 0,
                scale.reciprocal(),
                torch.zeros_like(scale),
            )
        return scale.contiguous()

    def post_init(self):
        device = self._humming_prepare_device()

        if self.weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise NotImplementedError(
                f"{self.__class__.__name__} needs an E4M3/E5M2 weight buffer, got "
                f"{self.weight.dtype}; convert the checkpoint first."
            )

        scale = self._multiplier_scale()
        if self.weight_scale_method == "block":
            scale = self._expanded_block_scale(scale)

        config, weight_schema, tensors = prepare_humming_fp8_tensors(
            weight=self.weight,
            weight_scale=scale,
            weight_scale_method=self.weight_scale_method,
            weight_block_size=self.weight_block_size,
            bias=self.bias,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=self.compute_dtype,
            device=device,
        )

        self._humming_install(
            device=device,
            config=config,
            weight_schema=weight_schema,
            tensors=tensors,
            zero_point=False,
        )
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._humming_forward(x)


class HummingMXFP4Linear(_HummingKernelMixin, WeightOnlyQuantLinear):
    """OCP MXFP4 (E2M1 weights + E8M0 block-32 scales) on the Humming backend.

    The buffers are MACaQ's checkpoint layout (``W_packed`` / ``W_scales``), which
    is byte-for-byte the layout humming consumes: low nibble = even column and
    ``value = code * 2 ** (e8m0 - 127)``. Only the plain OCP variant
    (``rotation="none"``, ``pre_mode="none"``) is accepted; a MACaQ-rotated
    checkpoint needs its activation Hadamard applied first.
    """

    SUPPORTS_BACKENDS = [BACKEND.HUMMING]
    SUPPORTS_METHODS = [METHOD.MACAQ]
    # Plain OCP MXFP4 only: the MACaQ-rotated variant carries a different frame.
    SUPPORTS_FORMATS = {FORMAT.MXFP4: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [32]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    CHECKPOINT_REQUIRED_BUFFERS = ("W_packed", "W_scales")

    HUMMING_WEIGHT_ATTR = "W_packed"
    HUMMING_SCALE_ATTR = "W_scales"
    HUMMING_ZERO_POINT_ATTR = None

    QUANT_TYPE = "mxfp4"

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.uint8,
        register_buffers: bool = False,
        adapter: Adapter = None,
        block_size: int = 32,
        had_block: int = 32,
        activation_precision: str = "w4a16",
        pre_mode: str = "none",
        rotation: str = "none",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if humming_import_exception is not None:
            raise ValueError(
                "Trying to use the humming backend, but the runtime requirements "
                f"were not met: {humming_runtime_error()}"
            )

        self.compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        self.block_size = int(block_size)
        self.had_block = int(had_block)
        self.activation_precision = activation_precision
        self.pre_mode = pre_mode
        self.rotation = rotation

        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=kwargs.pop("backend", BACKEND.HUMMING),
            adapter=adapter,
            register_buffers=False,
            pack_dtype=pack_dtype,
            dtype=self.compute_dtype,
            **kwargs,
        )

        if self.rotation != "none" or self.pre_mode != "none":
            raise NotImplementedError(
                "Humming MXFP4 consumes plain OCP MXFP4 only "
                "(rotation='none', pre_mode='none'); a MACaQ-rotated checkpoint "
                "must be converted first."
            )
        if self.block_size != 32:
            raise NotImplementedError(
                f"Humming MXFP4 requires 32-element scale blocks, got {self.block_size}."
            )
        if self.in_features % self.block_size:
            raise ValueError(
                f"Humming MXFP4 requires in_features divisible by {self.block_size}, "
                f"got {self.in_features}."
            )

        if register_buffers:
            self._allocate_buffers(bias=bias)
        else:
            self.bias = None

    @classmethod
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        if args.get("rotation", "none") != "none" or args.get("pre_mode", "none") != "none":
            return False, NotImplementedError(
                "Humming MXFP4 supports the plain OCP layout only "
                "(rotation='none', pre_mode='none')."
            )
        if args.get("block_size", 32) != 32:
            return False, NotImplementedError("Humming MXFP4 requires 32-element blocks.")
        return True, None

    def _allocate_buffers(self, *, bias: bool) -> None:
        groups = self.in_features // self.block_size
        for name, shape, dtype in (
            ("W_packed", (self.out_features, self.in_features // 2), torch.uint8),
            ("W_scales", (self.out_features, groups), torch.uint8),
        ):
            tensor = torch.zeros(shape, dtype=dtype)
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor)
        if bias:
            bias_tensor = torch.zeros(self.out_features, dtype=self.compute_dtype)
            if "bias" in self._buffers:
                self.bias = bias_tensor
            else:
                self.register_buffer("bias", bias_tensor)
        else:
            self.bias = None

    def list_buffers(self):
        buffers = []
        for name in ("W_packed", "W_scales", "bias"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                buffers.append(tensor)
        return buffers

    def post_init(self):
        device = self._humming_prepare_device()

        groups = self.in_features // self.block_size
        if tuple(self.W_packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError(
                f"{self.__class__.__name__}: unexpected W_packed shape "
                f"{tuple(self.W_packed.shape)} for ({self.out_features}, {self.in_features})."
            )
        if tuple(self.W_scales.shape) != (self.out_features, groups):
            raise ValueError(
                f"{self.__class__.__name__}: unexpected W_scales shape "
                f"{tuple(self.W_scales.shape)}; expected {(self.out_features, groups)}."
            )

        weight_scale = self.W_scales
        if weight_scale.dtype != torch.uint8:
            # Some writers store the E8M0 exponents with their float dtype.
            weight_scale = weight_scale.view(torch.uint8)

        config, weight_schema, tensors = prepare_humming_mxfp4_tensors(
            weight=self.W_packed,
            weight_scale=weight_scale,
            bias=self.bias,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=self.compute_dtype,
            device=device,
        )

        self._humming_install(
            device=device,
            config=config,
            weight_schema=weight_schema,
            tensors=tensors,
            zero_point=False,
        )
        super().post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._humming_forward(x)


__all__ = ["AwqHummingLinear", "HummingFP8Linear", "HummingLinear", "HummingMXFP4Linear"]


