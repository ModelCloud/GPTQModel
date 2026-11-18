# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import source_weight_out_in
from ...utils.awq import (
    awq_fast_gemm_forward_prefill,
    awq_fast_gemv_forward_decode,
    awq_runtime_available,
    awq_runtime_error,
)
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width
from ...utils.model_dequant import _dequantize_awq_gemv_fast
from ...utils.packing_math import scale_to_storage_dtype, scaled_zero_offset
from ._compile_support import COMPILE_LEVEL


def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K] int32 codes.  Every permutation runs on the
    # input device, so packing no longer round-trips through host numpy.
    # Confine each code to its nibble before shifting, including narrow input
    # dtypes and negative values supplied directly to the layout helper.
    unpacked_qweight = unpacked_qweight.to(torch.int32) & 15
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]
    if N % interleave:
        raise ValueError(
            f"GEMV-fast out_features must be divisible by {interleave}: actual={N}"
        )
    if K % kstride:
        raise ValueError(
            f"GEMV-fast in_features must be divisible by {kstride}: actual={K}"
        )

    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = unpacked_qweight.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.permute(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
            Packed_Kernel[..., 0]
            | (Packed_Kernel[..., 1] << 4)
            | (Packed_Kernel[..., 2] << 8)
            | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    return Packed_Kernel.reshape(N // interleave, K).to(torch.int16).contiguous()


class AwqGEMVFastLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMV_FAST]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMV_FAST: 30}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [128]

    # The decode path calls the same native CUDA AWQ extension as
    # ``AwqGEMVLinear`` (``gemv_fast_forward_decode``/``gemm_fast_forward_prefill``),
    # so it has the same ROCm restriction; see ``AWQ_GEMV_ROCM_UNSUPPORTED``.
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16]

    # The bias is added after the kernel's FP16 result by a wide epilogue when
    # the stored width differs, so it keeps its checkpoint dtype.
    MODEL_DTYPE_BUFFERS = ()

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemv_fast"

    # Measured on sm_80 (A100-class, torch 2.15.0.dev20260928+cu130): eager and an explicit
    # ``torch.cuda.graph`` capture both run with matching output. The native
    # GEMV_FAST ops (gemv_fast_forward_decode / gemm_fast_forward_prefill) have
    # no fake/meta kernel, so ``torch.compile`` stops at EAGER.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = True

    zeros_name = "qzeros"

    def __init__(
            self,
            bits: int,
            group_size: int,
            sym: bool,
            desc_act: bool,
            in_features: int,
            out_features: int,
            bias: bool = False,
            pack_dtype: torch.dtype = torch.int16,
            adapter: Adapter = None,
            register_buffers: bool = False,
            **kwargs,
    ):
        backend = kwargs.pop("backend", BACKEND.AWQ_GEMV_FAST)
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
        self.interleave = 4

        # The GEMV_FAST packing is a bit special; qweight and scales/zeros do not use the same pack_dtype.
        # qweight is packaged using torch.int16.
        # scales/zeros are packaged using torch.int32.
        int32_pack_factor = 32 // self.bits

        if register_buffers:
            self.register_buffer(
                "qweight",
                torch.zeros((out_features // self.interleave, in_features // self.pack_factor * self.interleave),
                            dtype=self.pack_dtype),
            )
            self.register_buffer(
                self.zeros_name,
                torch.zeros(
                    calculate_zeros_width(in_features, self.group_size) * int32_pack_factor,
                    out_features,
                    dtype=torch.float16,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    calculate_zeros_width(in_features, self.group_size) * int32_pack_factor,
                    out_features,
                    dtype=torch.float16,
                ),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
            else:
                self.bias = None

    def forward(self, x: torch.Tensor):
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)
        # The availability probe resolves the JIT extension, so pay it once per
        # module. A failed probe is never memoised.
        if not getattr(self, "_runtime_ok", False):
            if not awq_runtime_available():
                raise ModuleNotFoundError(
                    "AWQ torch.ops kernels are not properly installed. Error: " + awq_runtime_error()
                )
            self._runtime_ok = True

        inputs = x
        inputs_dim = inputs.dim()

        batch_size = None
        n_tokens = None
        # Some linear layers in certain models (for example, opt-12m model.decoder.layers.0.fc1) receive inputs of shape `[B*S, H]`,
        # so it's necessary to check if the dimension is equal to 3.
        if inputs_dim == 3:
            batch_size, n_tokens, _ = inputs.shape

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.to(dtype=torch.float16)
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        self._ensure_runtime_buffers(device=inputs.device, dtype=inputs.dtype)

        zeros = self._runtime_zeros().contiguous()
        scales = self.scales.contiguous()
        if inputs_dim == 3 and batch_size < 8 and n_tokens == 1:
            out = awq_fast_gemv_forward_decode(
                inputs,
                self.qweight,
                scales,
                zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_fast_gemm_forward_prefill(
                inputs, self.qweight, scales, zeros
            )

        if self.bias is not None:
            # A CUDA FP16 add already accumulates in FP32 and rounds once, so
            # the single kernel is bitwise identical to widening both operands.
            # Keep the wide path for BF16 and mixed dtypes: rounding either
            # operand to BF16 before the add changes the result.
            if input_dtype == torch.float16 and self.bias.dtype == torch.float16:
                out = torch.add(out, self.bias)
            else:
                out = out.float() + self.bias.float()
        out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _ensure_runtime_buffers(self, *, device: torch.device, dtype: torch.dtype):
        if self.qweight.device != device or not self.qweight.is_contiguous():
            self.qweight = self.qweight.to(device=device).contiguous()

        zeros = self._runtime_zeros()
        if zeros.device != device or zeros.dtype != dtype:
            zeros = zeros.to(device=device, dtype=dtype).contiguous()
            if self.zeros_name == "qzeros":
                self.qzeros = zeros
            elif self.zeros_name == "scaled_zeros":
                self.scaled_zeros = zeros
            else:
                raise ValueError(f"Unsupported zeros buffer: {self.zeros_name}")

        if self.scales.device != device or self.scales.dtype != dtype:
            self.scales = self.scales.to(device=device, dtype=dtype).contiguous()

        if self.bias is not None and (
            self.bias.device != device or not self.bias.is_contiguous()
        ):
            # Move the bias, never narrow it: the forward epilogue adds a bias
            # stored at another width in FP32, and rewriting the buffer here
            # would silently turn a finite FP32 value into FP16 ``inf``.
            self.bias = self.bias.to(device=device).contiguous()

    def _runtime_zeros(self) -> torch.Tensor:
        if self.zeros_name == "qzeros":
            return self.qzeros
        if self.zeros_name == "scaled_zeros":
            return self.scaled_zeros
        raise ValueError(f"Unsupported zeros buffer: {self.zeros_name}")

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros.float() * scales.float()

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
        # GEMV_FAST/LLM_AWQ store FP16 offsets in the checkpoint. Reject an
        # unrepresentable offset before replacing any packed module state.
        offsets = scaled_zero_offset(qscales[:, : scales.shape[1]], zeros)
        # The bias is not a quantization operand. Store it at the source width:
        # ``forward`` adds it in a wide FP32 epilogue, so narrowing it to the
        # FP16 scale dtype here could only turn a finite value into ``inf``.
        stored_bias = None if linear.bias is None else linear.bias.detach().clone()
        self.register_buffer("scales", qscales.transpose(1, 0).contiguous())
        if stored_bias is not None:
            self.register_buffer("bias", stored_bias)
        else:
            self.bias = None

        # Gather each input column's group scale/zero in a single pass instead
        # of launching one tiny kernel per input column.
        weight = source_weight_out_in(linear)
        group_index = torch.arange(self.in_features, device=weight.device) // self.group_size
        intweight = torch.round(
            (weight.float() + scale_zeros.index_select(1, group_index))
            / qscales.index_select(1, group_index).float()
        ).clamp_(0, (1 << self.bits) - 1).to(torch.int32)
        self.register_buffer("qweight", pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        ))

        qzeros = torch.zeros_like(qscales)

        qzeros[:, : scales.shape[1]] = offsets
        self.register_buffer(self.zeros_name, qzeros.transpose(1, 0).contiguous())

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


class LLMAwqLinear(AwqGEMVFastLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMV_FAST]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.LLM_AWQ: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [128]

    # Inherits the native-CUDA-only restriction from ``AwqGEMVFastLinear``; see
    # ``gemv_awq.AWQ_GEMV_ROCM_UNSUPPORTED``.
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "llm-awq"

    # Same native GEMV_FAST ops as AwqGEMVFastLinear. Measured on sm_80 (A100-class, torch 2.15.0.dev20260928+cu130):
    # eager and an explicit ``torch.cuda.graph`` capture both run with matching
    # output; the missing fake/meta kernel keeps ``torch.compile`` at EAGER.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = True

    zeros_name = "scaled_zeros"


class AwqGEMVFastTorchLinear(AwqGEMVFastLinear):
    """Portable reader for the packed ``FORMAT.GEMV_FAST`` layout (#4736, #5109).

    The native GEMV_FAST extension is CUDA-only, so a quantize run that finishes
    on CPU -- or on ROCm, where hipcc cannot build that extension -- had no
    kernel for a GEMV_FAST checkpoint: ``quantize()`` raised after the whole
    calibration and the documented format/platform combination was unusable.
    This kernel reads the exact same interleaved INT16 ``qweight`` and FP16
    ``scales``/offset buffers and dequantizes them to a dense FP32 tile per
    forward, which keeps the checkpoint format unchanged and makes the
    in-session model runnable.

    It is a documented fallback, not an AUTO replacement: the priority-0
    ``SUPPORTS_FORMATS`` entry keeps the accelerator AUTO order untouched, and
    ``select_quant_linear`` only reaches it for a target whose device family
    has no AUTO kernel through ``PORTABLE_FALLBACK``.

    The per-forward dense tile is a bounded temporary. It is never cached on
    the module, so the packed INT16 checkpoint storage stays the only
    persistent weight representation.
    """

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH]
    SUPPORTS_METHODS = [METHOD.AWQ]
    # 0 => supported for this layout, never auto-selected by AUTO.
    SUPPORTS_FORMATS = {FORMAT.GEMV_FAST: 0}
    SUPPORTS_BITS = [4]
    # Same packed layout contract as the native GEMV_FAST kernel, so a
    # CPU-packed checkpoint stays loadable by the CUDA extension.
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [128]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16]
    QUANT_TYPE = "awq_gemv_fast_torch"
    # Consumed by select_quant_linear() when AUTO has no kernel for the layout
    # on the target device family.
    PORTABLE_FALLBACK = True

    # Measured with tests/kernels/compile_probe.py on torch 2.16 CPU and HIP
    # builds: eager=True, graph_breaks=0, and dynamo/aot_eager/inductor all
    # pass with output matching eager. Eager CUDA-graph capture is not claimed;
    # the per-forward dequantize allocates its scatter indices on every call.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_COMPILE = False
    SUPPORTS_COMPILE_EAGER_REASON = (
        "portable GEMV_FAST fallback; no measured static-cache win (#4984)"
    )
    SUPPORTS_GRAPH_CAPTURE = False

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.AWQ_TORCH)
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        inputs = x.reshape(-1, x.shape[-1])
        input_dtype = inputs.dtype

        # The portable reader used to run only on CPU, so the packed buffers
        # always matched the input. A partially placed or CPU-offloaded
        # accelerator module can still hold them on CPU, and dequantizing there
        # would move a dense weight over the bus. Move the packed words locally
        # instead of rewriting the module state from `forward`: the buffers
        # stay the checkpoint's, and the move stays inside this call.
        qweight, zeros, scales = self._packed_for(inputs.device)

        # Bounded temporary dequantization: the dense tile lives only for this
        # matmul and is never cached on the module.
        weight = _dequantize_awq_gemv_fast(
            qweight,
            zeros,
            scales,
            bits=self.bits,
            group_size=self.group_size,
        )
        if weight.device != inputs.device:
            weight = weight.to(device=inputs.device)

        # FP32 accumulation with a single rounding at the end, matching the
        # accelerator GEMV_FAST kernels' accumulator width.
        output = torch.matmul(inputs.to(torch.float32), weight.t())
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=torch.float32)
        output = output.to(dtype=input_dtype)

        if self.adapter:
            output = self.adapter.apply(x=inputs, out=output)

        return output.reshape(out_shape)

    def _packed_for(self, device: torch.device):
        """Return ``(qweight, zeros, scales)`` on ``device`` for one forward.

        Device only, and on the packed buffers: the checkpoint's stored dtype is
        never narrowed (unlike the native kernel's ``_ensure_runtime_buffers``),
        and no dense weight is derived here. The module state is not rewritten,
        so a compiled or captured forward sees no mutation on this path.
        """

        def _on(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.device == device else tensor.to(device=device)

        return _on(self.qweight), _on(self._runtime_zeros()), _on(self.scales)


class LLMAwqTorchLinear(AwqGEMVFastTorchLinear, LLMAwqLinear):
    """Portable reader for ``FORMAT.LLM_AWQ`` (GEMV_FAST layout, ``scaled_zeros``).

    It stays an ``LLMAwqLinear`` so layout-keyed consumers (MLX transfer) keep
    treating it as an LLM_AWQ module; the Torch ``forward`` comes first in MRO.
    """

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.LLM_AWQ: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [128]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int16]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16]
    # Must keep the `awq_` prefix: pack dispatch keys AWQ packers on it.
    QUANT_TYPE = "awq_llm_awq_torch"
    PORTABLE_FALLBACK = True

    # Same forward and measurement as AwqGEMVFastTorchLinear.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_COMPILE = False
    SUPPORTS_COMPILE_EAGER_REASON = (
        "portable GEMV_FAST fallback; no measured static-cache win (#4984)"
    )
    SUPPORTS_GRAPH_CAPTURE = False

    zeros_name = "scaled_zeros"


__all__ = ["AwqGEMVFastLinear", "LLMAwqLinear", "AwqGEMVFastTorchLinear", "LLMAwqTorchLinear"]
