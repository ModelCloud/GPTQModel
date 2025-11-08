# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter
from ...quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import TORCH_HAS_FUSED_OPS
from .torch_fused import Int4PackedOp, TorchFusedQuantLinear, pack_scales_and_zeros


log = setup_logger()


class TorchFusedAwqQuantLinear(TorchFusedQuantLinear):
    """Torch fused AWQ variant that reuses the GPTQ fused kernels via CPU int4 packing."""

    QUANT_TYPE = "torch_fused_awq"
    SUPPORTS_BITS = TorchFusedQuantLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = TorchFusedQuantLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = TorchFusedQuantLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = TorchFusedQuantLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = TorchFusedQuantLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = TorchFusedQuantLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = TorchFusedQuantLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = TorchFusedQuantLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = TorchFusedQuantLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = TorchFusedQuantLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = TorchFusedQuantLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = TorchFusedQuantLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = TorchFusedQuantLinear.SUPPORTS_ADAPTERS

    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = False

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
        kwargs.setdefault("backend", BACKEND.TORCH_FUSED_AWQ)
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        qweight_key = prefix + "qweight"
        awq_tensor = None
        if qweight_key in state_dict:
            candidate = state_dict[qweight_key]
            if self._is_awq_qweight_tensor(candidate):
                awq_tensor = candidate.to(self.pack_dtype).clone()
                placeholder = getattr(self, "qweight", None)
                if isinstance(placeholder, torch.Tensor) and placeholder.numel() == awq_tensor.numel():
                    state_dict[qweight_key] = torch.zeros_like(placeholder)
                else:
                    rows = max(1, self.in_features // self.pack_factor)
                    cols = self.out_features
                    state_dict[qweight_key] = torch.zeros(
                        (rows, cols),
                        dtype=self.pack_dtype,
                        device=awq_tensor.device,
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if awq_tensor is not None:
            state_dict[qweight_key] = awq_tensor
            device = getattr(self, "qweight", awq_tensor).device
            self.register_buffer(
                "qweight",
                awq_tensor.to(device=device, dtype=self.pack_dtype).contiguous(),
                persistent=True,
            )

    def _awq_qweight_shape(self):
        pack_cols = self.out_features // self.pack_factor
        return self.in_features, pack_cols

    def _is_awq_qweight_tensor(self, tensor: torch.Tensor) -> bool:
        if tensor is None or not torch.is_tensor(tensor) or tensor.dim() != 2:
            return False
        rows, cols = tensor.shape
        exp_rows, exp_cols = self._awq_qweight_shape()
        return rows == exp_rows and cols == exp_cols

    def _uses_awq_layout(self) -> bool:
        qweight = getattr(self, "qweight", None)
        return torch.is_tensor(qweight) and self._is_awq_qweight_tensor(qweight)

    def _transform_cpu_awq(self, dtype):
        if not self._uses_awq_layout():
            raise RuntimeError("AWQ state unavailable for CPU transform.")
        src_scales = self.scales
        if src_scales.dtype != torch.float16:
            src_scales = src_scales.to(torch.float16)
        src_scales = src_scales.contiguous()
        self.scales = src_scales.clone().to(dtype).contiguous()
        scale_fp32 = src_scales.to(torch.float32)
        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val)
        if izeros is not None:
            izeros = torch.bitwise_and(izeros, max_val)
        ret_idx = self._build_ret_idx()
        weight = iweight.index_select(0, ret_idx).t().contiguous()
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight.int(), 1).contiguous()

        if izeros is None:
            zeros = torch.zeros_like(scale_fp32)
        else:
            zero_offset = 1 << (self.bits - 1)
            zeros = (zero_offset - izeros.reshape_as(scale_fp32)).to(dtype=scale_fp32.dtype)
            zeros = zeros * scale_fp32
        self.scales = scale_fp32.to(dtype=dtype)
        self.qzeros = zeros.to(dtype=dtype)
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def _awq_weight_dense(self, device, dtype):
        if not self._uses_awq_layout():
            raise RuntimeError("AWQ dense weight requested without cached tensors.")
        dense = dequantize_gemm(
            self.qweight,
            self.qzeros,
            self.scales,
            self.bits,
            self.group_size,
        ).to(device=device, dtype=torch.float32)
        return dense.to(device=device, dtype=dtype)

    def transform_cpu(self, dtype):
        if self._uses_awq_layout():
            self._transform_cpu_awq(dtype)
            return
        super().transform_cpu(dtype)

    def transform(self, dtype, device):
        if device == "xpu" and self._uses_awq_layout():
            raise NotImplementedError("TorchFusedAwqQuantLinear AWQ layout is currently supported on CPU only.")
        super().transform(dtype, device)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        self._assert_supported_dtype(x_flat.dtype)
        if not self.training and not self.transformed and TORCH_HAS_FUSED_OPS:
            self.transform(x_flat.dtype, x_flat.device.type)
            self.transformed = True
            if x_flat.device.type == "cpu":
                self.torch_fused_op = Int4PackedOp(
                    self.qweight, self.scales_and_zeros, self.group_size
                ).eval()
                import torch._inductor.config as config
                config.freezing = True
                config.max_autotune = True

        if self.transformed:
            log.debug("awq calling fused op")
            out = self._fused_op_forward(x_flat)
        else:
            log.debug("awq dense path")
            if self._uses_awq_layout():
                weight = self._awq_weight_dense(device=x_flat.device, dtype=x_flat.dtype)
                out = torch.matmul(x_flat, weight)
            else:
                num_itr = self.g_idx.shape[0] // x_flat.shape[-1]
                weights = self.dequantize_weight(num_itr=num_itr).to(x_flat.dtype)
                out = torch.matmul(x_flat, weights)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)

        return out.reshape(out_shape)

    @torch.no_grad
    def _fused_op_forward(self, x):
        awq_active = self._uses_awq_layout()
        use_awq_fallback = awq_active and x.device.type == "cpu"
        if use_awq_fallback:
            log.debug("awq unfused fallback")
            weight = self._awq_weight_dense(device=x.device, dtype=x.dtype)
            return torch.matmul(x, weight)
        else:
            log.debug("awq fused")
        return super()._fused_op_forward(x)

    def _assert_supported_dtype(self, dtype: torch.dtype):
        if dtype not in self.SUPPORTS_DTYPES:
            supported = ", ".join(str(d) for d in self.SUPPORTS_DTYPES)
            raise TypeError(
                f"{self.__class__.__name__} only supports input dtypes [{supported}], but received {dtype}."
            )


__all__ = ["TorchFusedAwqQuantLinear"]
