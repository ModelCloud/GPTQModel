# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from functools import lru_cache
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.python import has_gil_disabled
from .torch import TorchLinear


log = setup_logger()


def _validate_g_idx_bounds(
    g_idx: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    layer_name: str = "TritonV2Linear",
) -> None:
    """Reject checkpoint ``g_idx`` that would index the group buffers out of bounds.

    The Triton dequant kernel resolves each row's group as
    ``groups = where(g_idx < 0, g_idx + num_groups, g_idx)`` and then loads
    ``scales``/``qzeros`` at that group with no upper-bound guard, where
    ``num_groups = scales.shape[0]``. A value outside ``[-num_groups, num_groups)``
    therefore indexes past the buffer, causing an out-of-bounds device read
    (denial of service, possible adjacent-memory disclosure) when an untrusted
    checkpoint is loaded. This mirrors the safe faulting of the Torch backend's
    ``scales[g_idx]`` gather, but validates once at load rather than per forward.
    """
    if g_idx is None or scales is None or g_idx.numel() == 0:
        return

    num_groups = scales.shape[0]
    g_min = int(g_idx.min())
    g_max = int(g_idx.max())
    if g_min < -num_groups or g_max >= num_groups:
        raise ValueError(
            f"{layer_name}: checkpoint g_idx is out of range for {num_groups} "
            f"scale group(s) (got min={g_min}, max={g_max}, valid "
            f"[{-num_groups}, {num_groups - 1}]); the group index would read "
            f"scales/qzeros out of bounds."
        )


class TritonV2Linear(TorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TRITON]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 40, FORMAT.GPTQ_V2: 40}
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    # TODO: ROCM also has Triton support. Need to validate ROCM for triton
    SUPPORTS_DEVICES = [DEVICE.CUDA]  # Intel XPU can use Triton but this has not been validated yet
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.int16, torch.int8]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "tritonv2"

    """
    Triton v2 quantized linear layer.

    Calls dequant kernel (see triton_utils/dequant) to dequantize the weights then uses
    torch.matmul to compute the output whereas original `triton` quantized linear layer fused
    dequant and matmul into single kernel.add()
    """

    def __init__(
            self,
            bits: int,
            group_size: int,
            desc_act: bool,
            sym: bool,
            in_features,
            out_features,
            bias: bool = False,
            pack_dtype: torch.dtype = torch.int32,
            adapter: Adapter = None,
            register_buffers: bool = True,
            **kwargs,
    ):
        # log.debug(f"triton register_buffers: {register_buffers}")
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TRITON),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        import triton  # noqa: F401  # validate Triton is importable
        import triton.language as tl  # noqa: F401  # ensure Triton language bindings load
        from packaging import version
        from triton import __version__ as triton_version

        from ..triton_utils.dequant import QuantLinearFunction  # noqa: F401  # dependency check for validate_once
        from ..triton_utils.mixin import TritonModuleMixin  # noqa: F401  # ensure mixin is available at runtime

        triton_v = version.parse(triton_version)

        if triton_v < version.parse("2.0.0"):
            raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")

        # GIL=0 is tested with Triton 3.4.0 and it works
        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            raise Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        device = args.get('device')

        # xpu requires extra runtime checks to see if triton actually works
        if device == DEVICE.XPU and not triton_xpu_available():
            return False, ValueError(
                "Trying to use the triton backend and xpu device, but it could not be imported. Please install triton by [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)")

        valid, error = cls._validate(**args)
        if not valid:
            return valid, error

        if args.get("bits") == 3:
            required = {
                "desc_act": False,
                "sym": True,
                "pack_dtype": torch.int32,
            }
            for name, expected in required.items():
                actual = args.get(name)
                if actual != expected:
                    return False, NotImplementedError(
                        f"{cls.__name__} 3-bit fused inference requires `{name}={expected}`, got `{actual}`."
                    )
            if args.get("dynamic"):
                return False, NotImplementedError(
                    f"{cls.__name__} 3-bit fused inference does not support dynamic per-layer quantization."
                )
            in_features = args.get("in_features")
            out_features = args.get("out_features")
            group_size = args.get("group_size", 128)
            effective_group_size = in_features if group_size == -1 else group_size
            if in_features is not None and (
                in_features % 32 != 0
                or effective_group_size is None
                or effective_group_size <= 0
                or in_features % effective_group_size != 0
            ):
                return False, NotImplementedError(
                    f"{cls.__name__} 3-bit fused inference requires in_features divisible by 32 and by "
                    f"group_size (or group_size=-1), got in_features={in_features}, group_size={group_size}."
                )
            if out_features is not None and out_features % 32 != 0:
                return False, NotImplementedError(
                    f"{cls.__name__} 3-bit fused inference requires out_features divisible by 32, "
                    f"got {out_features}."
                )

        return True, None

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_factor, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_factor
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)
        super().post_init()

        # The Triton dequant kernel indexes the per-group scales/qzeros buffers
        # with the checkpoint's g_idx and performs no upper-bound check (only a
        # negative-value wrap), so a crafted g_idx entry >= num_groups reads out
        # of bounds on the device (CWE-125). Validate once here at load — the
        # Torch backend's scales[g_idx] gather already faults safely on the same
        # input — instead of letting the raw value reach the kernel.
        _validate_g_idx_bounds(
            self.g_idx, self.scales, layer_name=type(self).__name__
        )
        if self.bits == 3:
            from ..triton_utils.three_bit import prepare_marlin_3bit, prepare_trilin_3bit, unpack_3bit

            expected_g_idx = torch.arange(
                self.in_features,
                dtype=self.g_idx.dtype,
                device=self.g_idx.device,
            ) // self.group_size
            if not torch.equal(self.g_idx, expected_g_idx):
                raise ValueError(
                    f"{type(self).__name__}: 3-bit fused inference requires natural group indices "
                    "for desc_act=False."
                )
            zeros = unpack_3bit(self.qzeros, axis=1, count=self.out_features)
            if not torch.all(zeros == 4):
                raise ValueError(
                    f"{type(self).__name__}: 3-bit symmetric GPTQ inference requires every zero point to equal 4."
                )
            self._trilin_native_3bit = prepare_trilin_3bit(self.qweight, self.scales, self.requested_group_size)
            marlin_state = prepare_marlin_3bit(self.qweight, self.scales, self.requested_group_size)
            if marlin_state is not None:
                self.register_buffer("_trilin_marlin_qweight", marlin_state.qweight, persistent=False)
                self.register_buffer("_trilin_marlin_scales", marlin_state.scales, persistent=False)
                self.register_buffer("_trilin_marlin_workspace", marlin_state.workspace, persistent=False)
                self.register_buffer("_trilin_marlin_empty", marlin_state.empty, persistent=False)

    def forward(self, x):
        from ..triton_utils.dequant import QuantLinearFunction

        if self.bits == 3 and not self.training:
            from ..triton_utils.three_bit import (
                LAYOUT_GPTQ,
                matmul_3bit,
                matmul_marlin_3bit,
                matmul_trilin_3bit,
            )

            capability = torch.cuda.get_device_capability(self.qweight.device)
            if capability >= (8, 0):
                out_shape = x.shape[:-1] + (self.out_features,)
                x_flat = x.reshape(-1, x.shape[-1])
                if x_flat.stride(-1) != 1:
                    x_flat = x_flat.contiguous()
                native_qweight = getattr(self, "_trilin_marlin_qweight", None)
                if (
                    x_flat.dtype in (torch.float16, torch.bfloat16)
                    and 0 < x_flat.shape[0] <= 16
                    and getattr(self, "_trilin_native_3bit", False)
                ):
                    out = matmul_trilin_3bit(
                        x_flat if x_flat.is_contiguous() else x_flat.contiguous(),
                        self.qweight,
                        self.scales,
                        bias=self.bias,
                        group_size=self.requested_group_size,
                    ).reshape(out_shape)
                elif x_flat.dtype == torch.float16 and native_qweight is not None:
                    out = matmul_marlin_3bit(
                        x_flat,
                        native_qweight,
                        self._trilin_marlin_scales,
                        self._trilin_marlin_workspace,
                        self._trilin_marlin_empty,
                        k=self.in_features,
                        n=self.out_features,
                        bias=self.bias,
                    ).reshape(out_shape)
                else:
                    out = matmul_3bit(
                        x_flat,
                        self.qweight,
                        self.scales,
                        layout=LAYOUT_GPTQ,
                        group_size=self.requested_group_size,
                    ).reshape(out_shape)

                    if self.bias is not None:
                        out.add_(self.bias)

                if self.adapter:
                    out = self.adapter.apply(x=x, out=out)

                return out.to(dtype=x.dtype)

        if self.training:
            return super().forward(x)

        # if in_features is padded, we need to pad the input as well
        # if x.size(-1) != self.padded_infeatures:
        #     x = F.pad(x, (0, self.padded_infeatures - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)

        out = QuantLinearFunction.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        ).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x.dtype)


__all__ = ["TritonV2Linear"]


# test triton on XPU to ensure special Intel/Triton is installed as we cannot check based on triton package meta data
def triton_test_add(x: torch.Tensor, y: torch.Tensor):
    import triton  # noqa: F401  # ensure Triton language bindings load
    import triton.language as tl  # noqa: F401  # ensure Triton language bindings load

    # don't put it on top-level to avoid crash if triton was not installed
    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                   y_ptr,  # *Pointer* to second input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y  # noqa: F841

    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@lru_cache
def triton_xpu_available():
    size = 1024
    x = torch.rand(size, device='xpu:0')
    y = torch.rand(size, device='xpu:0')

    try:
        triton_test_add(x, y)
        return True
    except Exception:
        return False
