# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm, pack_awq
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear
from ._runtime_cache import tensor_cache_key


log = setup_logger()


class AwqTorchLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 10}
    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "awq_torch"

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
            backend=kwargs.pop("backend", BACKEND.AWQ_TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        if register_buffers:
            if self.scales is not None and self.scales.dtype != self.compute_dtype:
                self.scales = self.scales.to(dtype=self.compute_dtype)
            if self.bias is not None and self.bias.dtype != self.compute_dtype:
                self.bias = self.bias.to(dtype=self.compute_dtype)
        self._cpu_packed_qweight = None
        self._cpu_packed_qzeros = None
        self._cpu_packed_scales = None
        self._cpu_packed_bias = None
        self._cpu_packed_key = None
        self._cpu_packed_warning_logged = False

    def post_init(self):
        if self.scales is not None and self.scales.dtype not in (torch.float16, torch.bfloat16):
            self.scales = self.scales.to(dtype=torch.float16)
        if self.bias is not None and self.bias.dtype not in (torch.float16, torch.bfloat16):
            self.bias = self.bias.to(dtype=torch.float16)
        self.clear_weight_cache()
        super().post_init()

    def train(self, mode: bool = True):
        self.clear_weight_cache()
        return super().train(mode=mode)

    def clear_weight_cache(self) -> None:
        self._cpu_packed_qweight = None
        self._cpu_packed_qzeros = None
        self._cpu_packed_scales = None
        self._cpu_packed_bias = None
        self._cpu_packed_key = None
        self._cpu_packed_warning_logged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor = None):
        del g_idx
        assert scales is not None and zeros is not None

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales

        scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
        self.register_buffer("scales", scales.clone().to(scale_dtype))
        if linear.bias is not None:
            bias_dtype = linear.bias.dtype if linear.bias.dtype in (torch.float16, torch.bfloat16) else scale_dtype
            self.register_buffer("bias", linear.bias.clone().to(bias_dtype))
        else:
            self.bias = None

        # Gather each input column's group scale/zero in a single pass instead
        # of launching one tiny kernel per input column.
        weight = linear.weight.data
        group_index = torch.arange(self.in_features, device=weight.device) // self.group_size
        intweight = torch.round(
            (weight.t() + scale_zeros.index_select(0, group_index))
            / self.scales.index_select(0, group_index)
        ).to(torch.int32).contiguous()

        qweight, qzeros = pack_awq(intweight, zeros.to(torch.int32), self.bits)

        self.register_buffer("qweight", qweight)
        self.register_buffer("qzeros", qzeros)

    def _ensure_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.scales.device != device or self.scales.dtype != dtype or not self.scales.is_contiguous():
            self.scales = self.scales.to(device=device, dtype=dtype).contiguous()
        if self.bias is not None and (
            self.bias.device != device or self.bias.dtype != dtype or not self.bias.is_contiguous()
        ):
            self.bias = self.bias.to(device=device, dtype=dtype).contiguous()

    def _dequantize_weight_for_inference(
        self,
        x_flat: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        )
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise AssertionError(f"weight {weight.dtype} is not float16 or bfloat16")
        if weight.dtype != dtype or weight.device != x_flat.device or not weight.is_contiguous():
            weight = weight.to(device=x_flat.device, dtype=dtype).contiguous()
        return weight

    def _cpu_avx512_eligible(self, x_flat: torch.Tensor) -> bool:
        return (
            x_flat.device.type == "cpu"
            and not self.training
            and not x_flat.requires_grad
            and self.bits == 4
            and self.group_size > 0
            and self.group_size <= 512
            and self.in_features % 2 == 0
            and self.in_features % self.group_size == 0
            and self.out_features % 32 == 0
            and x_flat.shape[-1] == self.in_features
            and x_flat.dtype == torch.bfloat16
        )

    def _ensure_cpu_packed(self) -> bool:
        key = (
            tensor_cache_key(self.qweight),
            tensor_cache_key(self.qzeros),
            tensor_cache_key(self.scales),
            tensor_cache_key(self.bias) if self.bias is not None else None,
            self.bits,
            self.group_size,
            self.out_features,
            self.in_features,
        )
        if self._cpu_packed_key == key and self._cpu_packed_qweight is not None:
            return True

        from .torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear

        ok, _ = TorchCpuAvx512AwqLinear.cached_validate_once()
        packer = TorchCpuAvx512AwqLinear.pack_awq_int4_kernel
        if not ok or packer is None:
            return False

        try:
            packed_qweight, packed_qzeros = packer(
                self.qweight.contiguous(),
                self.qzeros.contiguous(),
                self.out_features,
                self.in_features,
            )
        except Exception as exc:
            if not self._cpu_packed_warning_logged:
                log.warning(
                    "AWQ CPU AVX512 packing unavailable for `%s`; using dense fallback: %s",
                    self.name,
                    exc,
                )
                self._cpu_packed_warning_logged = True
            return False

        self._cpu_packed_qweight = packed_qweight
        self._cpu_packed_qzeros = packed_qzeros
        self._cpu_packed_scales = self.scales.to(dtype=torch.float32).contiguous()
        self._cpu_packed_bias = (
            self.bias.to(dtype=torch.float32).contiguous()
            if self.bias is not None
            else None
        )
        self._cpu_packed_key = key
        return True

    def _forward_cpu_avx512(
        self,
        x_flat: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> torch.Tensor | None:
        if not enabled or not self._cpu_avx512_eligible(x_flat) or not self._ensure_cpu_packed():
            return None

        from .torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear

        gemm = TorchCpuAvx512AwqLinear.gemm_int4_qzero_group_scaled_cpu_kernel
        if gemm is None:
            return None

        original_dtype = x_flat.dtype
        kernel_input = x_flat
        if not kernel_input.is_contiguous():
            kernel_input = kernel_input.contiguous()

        output = gemm(
            kernel_input,
            self._cpu_packed_qweight,
            self._cpu_packed_qzeros,
            self._cpu_packed_scales,
            self.group_size,
            self._cpu_packed_bias,
        )
        if output.dtype != original_dtype:
            output = output.to(dtype=original_dtype)
        return output

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.float16
        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or x_flat.device != device:
            x_flat = x_flat.to(device=device, dtype=compute_dtype)
        elif not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        self._ensure_runtime_dtype(device=device, dtype=compute_dtype)

        # This is the portable/reference AWQ implementation.  Its arithmetic
        # must remain the dense dequantize + matmul oracle; the BF16-only
        # AVX512 path is selected explicitly through TorchCpuAvx512AwqLinear.
        # Subclasses such as ParoQuant use the shared dequantize helper.
        weight = self._dequantize_weight_for_inference(x_flat, dtype=compute_dtype)
        output = torch.matmul(x_flat, weight)
        if self.bias is not None:
            output = output + self.bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        if output.dtype != input_dtype:
            output = output.to(dtype=input_dtype)

        output = output.reshape(original_shape)

        return output

__all__ = ["AwqTorchLinear"]
