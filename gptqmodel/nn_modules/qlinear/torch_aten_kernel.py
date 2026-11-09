# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ._fused_train_state import capture_fused_train_state, restore_fused_train_state
from ._runtime_cache import build_group_ret_idx, tensor_cache_key
from .torch_fused import _cpu_int4pack_mm, pack_scales_and_zeros


log = setup_logger()


def _has_local_int4pack_cpu_ops() -> bool:
    return (
        hasattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu")
        and hasattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu")
    )


def _cpu_int4pack_zero_offsets(
    zero_codes: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    # aten::_weight_int4pack_mm_for_cpu dequantizes as:
    #   scale * (signed_code - 2^(bits-1)) + zero_offset
    # Convert stored GPTQ zero codes so the fused kernel reproduces:
    #   scale * (code - zero_code)
    zero_center = 1 << (bits - 1)
    return (zero_center - zero_codes.float()) * scales.float()


def _cpu_int4pack_scale_dtype(scales, zero_codes, bits, target_dtype):
    # The fused op stores affine offsets instead of integer zero points. Check
    # their range before replacing checkpoint buffers with the packed layout.
    offsets = _cpu_int4pack_zero_offsets(zero_codes, scales.to(target_dtype), bits)
    # Rounded offsets can turn an exactly zero decoded weight into a nonzero
    # value. Preserve the affine identity as well as finite range.
    if not torch.equal(offsets.to(target_dtype).float(), offsets):
        return torch.float32
    for value in (scales, offsets):
        narrowed = value.to(target_dtype)
        if not torch.isfinite(narrowed).all() or ((value != 0) & (narrowed == 0)).any():
            return torch.float32
    return target_dtype


class TorchAtenLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_ATEN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 110, FORMAT.GPTQ_V2: 110}
    SUPPORTS_BITS = [4]
    # aten::_weight_int4pack_mm_for_cpu accepts 32, 64, 128, or 256. Keep the
    # selector aligned with the op contract so non-AVX512 hosts do not choose
    # this backend for a group size that fails on the first forward.
    SUPPORTS_GROUP_SIZE = [32, 64, 128, 256]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "torch_aten_kernel"

    gemm_int4_forward_kernel = None
    _fused_op_handles_bias = False

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
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH_ATEN),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=kwargs.pop("enable_wf_unsqueeze", True),
            **kwargs,
        )

        self.linear_mode = None
        self.dequant_dtype = torch.int8
        self._cpu_transformed = False
        self._cpu_transform_dtype = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _has_local_int4pack_cpu_ops():
            cls.gemm_int4_forward_kernel = None
            err = ImportError(
                "TorchAtenLinear requires aten::_convert_weight_to_int4pack_for_cpu and "
                "aten::_weight_int4pack_mm_for_cpu in this PyTorch build."
            )
            log.warning(str(err))
            return False, err

        cls.gemm_int4_forward_kernel = staticmethod(torch.ops.aten._weight_int4pack_mm_for_cpu)
        return True, None

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return

        super().optimize()

    def _cpu_fused_available(self) -> bool:
        return _has_local_int4pack_cpu_ops()

    def _fused_input_dtype(self, x: torch.Tensor) -> bool:
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if self._cpu_transform_dtype is not None:
            return x.dtype == self._cpu_transform_dtype
        return True

    def clear_weight_cache(self) -> None:
        """Drop derived state that depends on the loaded packed buffers.

        ``ret_idx`` and the cached identity flag stay valid only for the
        ``g_idx`` they were built from, and ``load_state_dict`` may copy that
        tensor in place. Once transformed, ``scales_and_zeros`` is likewise
        derived from the checkpoint's scales and zero points; a load copies
        those in place and would otherwise leave the fused op on stale data.
        The packed ``qweight`` is already in the ATen layout, so the derived
        metadata is rebuilt from it rather than from dense weights.
        """
        if restore_fused_train_state(self):
            self.linear_mode = None
            self.scales_and_zeros = None
            self._cpu_transformed = False
            self._cpu_transform_dtype = None

        self.ret_idx = None
        self._ret_idx_identity = None
        self._ret_idx_key = None
        self._fused_state_key = None
        self._refresh_fused_state()

    def _refresh_fused_state(self) -> None:
        """Rebuild derived int4pack metadata after its sources changed."""
        if not getattr(self, "_cpu_transformed", False):
            return
        scales = getattr(self, "scales", None)
        qzeros = getattr(self, "qzeros", None)
        if scales is None or qzeros is None:
            return
        key = (tensor_cache_key(scales), tensor_cache_key(qzeros))
        if getattr(self, "_fused_state_key", None) == key:
            return
        current = getattr(self, "scales_and_zeros", None)
        if not isinstance(current, torch.Tensor):
            return
        # Keep at least the original compute precision and the precision of
        # replacement metadata; narrowing affine offsets can lose cancellation.
        dtype = torch.promote_types(current.dtype, torch.promote_types(scales.dtype, qzeros.dtype))
        self.scales_and_zeros = pack_scales_and_zeros(
            scales.to(dtype), qzeros.to(dtype)
        )
        self._fused_state_key = key

    def _build_ret_idx(self) -> torch.Tensor:
        key = tensor_cache_key(self.g_idx)
        existing = getattr(self, "ret_idx", None)
        total = self.g_idx.shape[0]
        if (
            isinstance(existing, torch.Tensor)
            and existing.numel() == total
            and getattr(self, "_ret_idx_key", None) == key
        ):
            return existing

        device = self.g_idx.device
        group_size = max(int(self.group_size), 1)
        g_idx = self.g_idx.to(torch.int32)
        canonical_g_idx = torch.arange(total, dtype=torch.int32, device=device) // group_size
        if torch.equal(g_idx, canonical_g_idx):
            ret_idx = torch.arange(total, dtype=torch.int32, device=device)
            self.ret_idx = ret_idx
            self._ret_idx_identity = True
            self._ret_idx_key = key
            return ret_idx

        ret_idx = build_group_ret_idx(self.g_idx)
        self.ret_idx = ret_idx
        self._ret_idx_identity = None
        self._ret_idx_key = key
        return ret_idx

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        if mode and self.linear_mode == "inference":
            restore_fused_train_state(self)
            self.linear_mode = None
            self.scales_and_zeros = None
            self._cpu_transformed = False
            self._cpu_transform_dtype = None

        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            if mode:
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)
            else:
                if hasattr(self, "qzeros_data_v1"):
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)

        return super().train(mode=mode)

    def transform_cpu(self, dtype: torch.dtype | None = None):
        if self._cpu_transformed:
            return

        # ATen int4pack requires the activation and scales/zero-points to have
        # the same dtype. Choose that dtype from the first fused activation so
        # an FP16 checkpoint loaded into a BF16 model does not silently fall
        # back to dense dequantization on every token.
        target_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else self.scales.dtype
        if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            target_dtype = torch.bfloat16
        zero_codes = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(torch.uint8)
        zero_codes = torch.bitwise_and(zero_codes, self.maxq).reshape(self.scales.shape)
        compute_dtype = _cpu_int4pack_scale_dtype(self.scales, zero_codes, self.bits, target_dtype)
        self.scales = self.scales.to(compute_dtype).contiguous()

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(torch.uint8),
            self.maxq,
        )
        ret_idx = self._build_ret_idx()
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        if not self._ret_idx_is_identity():
            weight = weight.index_select(0, ret_idx)
        weight = weight.t().contiguous()
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight.int(), 1).contiguous()

        self.qzeros = _cpu_int4pack_zero_offsets(zero_codes, self.scales, self.bits).to(compute_dtype).contiguous()
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)
        self._cpu_transform_dtype = target_dtype
        self._cpu_transformed = True

    def transform(self, device, dtype: torch.dtype | None = None):
        if device == "cpu":
            self.transform_cpu(dtype=dtype)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if (
            not self.training
            and not x.requires_grad
            and self.linear_mode is None
            and self._cpu_fused_available()
            and x.device.type == "cpu"
            and self._fused_input_dtype(x)
        ):
            capture_fused_train_state(self)
            self.transform(x.device.type, dtype=x.dtype)
            self.linear_mode = "inference"
        elif self.linear_mode is None:
            self.linear_mode = "train"

        if self.linear_mode == "inference":
            if not self._fused_input_dtype(x):
                raise RuntimeError(
                    f"{self.__class__.__name__} was transformed for {self.scales.dtype} "
                    f"but received {x.dtype} activations. Keep one activation dtype per layer."
                )
            out = self._fused_op_forward(x).reshape(out_shape)
        else:
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
            out = torch.matmul(x, weights).reshape(out_shape)

        if self.bias is not None and not (
            self.linear_mode == "inference" and self._fused_op_handles_bias
        ):
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    @torch.no_grad
    def _fused_op_forward(self, x):
        self._refresh_fused_state()
        if not self._ret_idx_is_identity():
            x = x[:, self.ret_idx].contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()
        if x.device.type != "cpu":
            raise NotImplementedError

        if not self._fused_input_dtype(x):
            raise RuntimeError(
                f"{self.__class__.__name__} fused ATen int4pack requires activation dtype "
                f"{self.scales_and_zeros.dtype}, got {x.dtype}."
            )
        return _cpu_int4pack_mm(x, self.qweight, self.group_size, self.scales_and_zeros)

    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None


def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchAtenLinear):
            raise ValueError(
                "Only models loaded using TorchAtenLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.GPTQ_TORCH_ATEN"
            )

        if isinstance(module, TorchAtenLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

            parent = model
            if "." in name:
                parent_name, module_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["TorchAtenLinear", "dequantize_model"]
