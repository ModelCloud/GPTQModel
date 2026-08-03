# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os
from functools import lru_cache
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import FormatSupport
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.python import has_gil_disabled
from .torch import TorchLinear
from .utils import validate_mixed_precision_3bit_contract


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
    # Continuous gptq/gptq_v2 words decode at 2/3/4/8-bit. gptq_p is planar at
    # 3/5/6/7-bit and reuses the continuous word layout at 2/4/8-bit.
    # Continuous gptq_v2 3-bit additionally relayouts to planar once at
    # post_init (convert_to_planar) so the native Pangolin GEMV can serve
    # gptq_v2:3 checkpoints.
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.GPTQ: FormatSupport(priority=40, bits=(2, 3, 4, 8)),
        FORMAT.GPTQ_V2: FormatSupport(priority=40, bits=(2, 3, 4, 8)),
        FORMAT.GPTQ_P: FormatSupport(priority=40, bits=(2, 3, 4, 5, 6, 7, 8)),
    }
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

        valid, error = validate_mixed_precision_3bit_contract(
            kernel_name=cls.__name__,
            bits=args.get("bits"),
            desc_act=args.get("desc_act"),
            sym=args.get("sym"),
            pack_dtype=args.get("pack_dtype"),
            dynamic=args.get("dynamic"),
        )
        if not valid:
            return valid, error

        if args.get("bits") == 3 and args.get("format") != FORMAT.GPTQ_P:
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
        # Continuous gptq_v2 3-bit: relayout once to the planar layout when the
        # native Pangolin GEMV can serve it, so gptq_v2:3 checkpoints route
        # through the planar kernels instead of the continuous fused path.
        if self.bits == 3 and not self.planar:
            self._maybe_convert_continuous_3bit_to_planar()

        # The continuous 3-bit fused path checks below do not apply to planar
        # (gptq_p) modules, which decode through the planar Triton kernels.
        if self.bits == 3 and not self.planar:
            from ..triton_utils.three_bit import unpack_3bit

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

    def _maybe_convert_continuous_3bit_to_planar(self) -> bool:
        """Relayout continuous gptq_v2 3-bit buffers to planar when Pangolin can run.

        Runtime-only conversion: requires the packed buffers on a compute
        capability >= 8.0 CUDA device, block-uniform g_idx, and the Pangolin
        JIT extension. Otherwise the module keeps the continuous layout and
        the existing continuous 3-bit paths.
        """
        from ...utils.pangolin import ensure_pangolin_runtime_available
        from ..triton_utils.planar import _g_idx_block_uniform

        if _PANGOLIN_DISABLED:
            return False
        if not self.qweight.is_cuda:
            return False
        if torch.cuda.get_device_capability(self.qweight.device) < (8, 0):
            return False
        if not _g_idx_block_uniform(self.g_idx):
            return False
        if not ensure_pangolin_runtime_available():
            return False
        return self.convert_to_planar()

    def forward(self, x):
        from ..triton_utils.dequant import QuantLinearFunction

        if self.training:
            return super().forward(x)

        x = self._apply_rotation_to_input(x)

        if self.planar and not self.training:
            planar_out = self._forward_planar_triton(x)
            if planar_out is not None:
                return planar_out

        if self.bits == 3 and not self.planar and not self.training:
            from ..triton_utils.three_bit import LAYOUT_GPTQ, matmul_3bit

            capability = torch.cuda.get_device_capability(self.qweight.device)
            if capability >= (8, 0):
                out_shape = x.shape[:-1] + (self.out_features,)
                x_flat = x.reshape(-1, x.shape[-1])
                if x_flat.stride(-1) != 1:
                    x_flat = x_flat.contiguous()
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

    def _forward_planar_triton(self, x):
        from ..triton_utils.planar import (
            PLANAR_FUSED_MAX_M,
            PLANAR_GEMV_MAX_M,
            PLANAR_TRITON_BITS,
            planar_dequant,
            planar_gemv,
            planar_matmul,
        )

        if self.bits not in PLANAR_TRITON_BITS:
            return None

        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        # Parent modules (e.g. unused MoE experts) may call with zero rows.
        # https://github.com/ModelCloud/GPTQModel/issues/1361
        if x_flat.shape[0] == 0:
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # Packed planar buffers are padded to /32 once at pack/load; pad the
        # activation K to match and slice the output back to logical N.
        if self.padded_in_features != self.in_features:
            x_flat = torch.nn.functional.pad(x_flat, (0, self.padded_in_features - self.in_features))

        pangolin_out = self._forward_pangolin(x_flat)
        if pangolin_out is not None:
            out = pangolin_out
        elif x_flat.shape[0] <= PLANAR_GEMV_MAX_M:
            # Decode-shape inputs: fused GEMV reads only the packed words on
            # the weight side and skips the dense fp16 weight buffer entirely.
            out = planar_gemv(
                x_flat, self.qweight, self.scales, self.qzeros, self.g_idx, self.bits
            )
        elif x_flat.shape[0] <= PLANAR_FUSED_MAX_M:
            # Decode-shape inputs: fused dequant+matmul reads the packed words
            # once instead of round-tripping a dense fp16 weight through DRAM.
            out = planar_matmul(
                x_flat, self.qweight, self.scales, self.qzeros, self.g_idx, self.bits
            )
        else:
            weights = planar_dequant(
                x.dtype, self.qweight, self.scales, self.qzeros, self.g_idx, self.bits
            )
            out = torch.matmul(x_flat, weights)

        if out.shape[-1] != self.out_features:
            out = out[:, : self.out_features]
        out = out.reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x.dtype)

    def _forward_pangolin(self, x_flat):
        """Native CUDA register-decode GEMV for decode-shape planar inputs.

        Requirements: M in PANGOLIN_SUPPORTED_M, fp16/bf16 input with matching
        scales dtype, contiguous packed buffers, block-uniform g_idx, and the
        tensors resident on a compute capability >= 8.0 CUDA device with the
        JIT extension built. Returns None (Triton fallback) when any
        condition fails.
        """
        from ..triton_utils.planar import _g_idx_block_uniform
        from ...utils.pangolin import (
            PANGOLIN_MAX_M,
            PANGOLIN_SUPPORTED_M,
            ensure_pangolin_runtime_available,
            pangolin_gemv,
        )

        if _PANGOLIN_DISABLED:
            return None
        if x_flat.shape[0] < 1 or x_flat.shape[0] > PANGOLIN_MAX_M:
            return None
        if x_flat.shape[0] not in PANGOLIN_SUPPORTED_M:
            return None
        # M=32 is register/occupancy-limited; on tall layers (out_features < in_features)
        # the planar_dequant + cuBLAS fallback is faster. Route M=32 to native only when
        # there are enough output columns to occupy the device.
        if x_flat.shape[0] == 32 and self.out_features < self.in_features:
            return None
        if not x_flat.is_cuda:
            return None
        if x_flat.dtype not in (torch.float16, torch.bfloat16) or self.scales.dtype != x_flat.dtype:
            return None
        if not (
            self.qweight.is_contiguous()
            and self.qzeros.is_contiguous()
            and self.scales.is_contiguous()
            and self.g_idx.is_contiguous()
        ):
            return None
        if torch.cuda.get_device_capability(x_flat.device) < (8, 0):
            return None
        if not ensure_pangolin_runtime_available():
            return None
        if not _g_idx_block_uniform(self.g_idx):
            return None
        return pangolin_gemv(x_flat, self.qweight, self.scales, self.qzeros, self.g_idx, self.bits)


# Kill switch for the native planar GEMV; the Triton paths remain the fallback.
_PANGOLIN_DISABLED = os.environ.get("GPTQMODEL_PANGOLIN_DISABLE", "0") == "1"


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
