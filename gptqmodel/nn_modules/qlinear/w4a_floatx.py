# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native GPTQ INT4 checkpoints with one-time floating weight staging."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from . import PackableQuantLinear


class W4AFP8Linear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_W4AFP8]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # A saved activation policy selects this backend. Ordinary GPTQ loads do not.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [128]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
    REQUIRES_FORMAT_V2 = True
    QUANT_TYPE = "w4afp8"

    def __init__(self, bits: int, group_size: int, sym: bool, desc_act: bool,
                 in_features: int, out_features: int, bias: bool = False,
                 pack_dtype: torch.dtype = torch.int32, register_buffers: bool = True,
                 **kwargs):
        super().__init__(
            bits=bits, group_size=group_size, sym=sym, desc_act=desc_act,
            in_features=in_features, out_features=out_features, bias=bias,
            pack_dtype=pack_dtype, register_buffers=register_buffers,
            backend=kwargs.pop("backend", BACKEND.GPTQ_W4AFP8),
            adapter=kwargs.pop("adapter", None), **kwargs,
        )
        self.register_buffer("_weight_e4m3", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            import triton  # noqa: F401

            from . import w4a_triton  # noqa: F401
        except (ImportError, OSError) as exc:
            return False, RuntimeError(f"{cls.__name__} needs Triton with FP8 support: {exc}")
        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = cls._validate(**args)
        if not ok:
            return ok, err
        device = args.get("device")
        if not torch.cuda.is_available():
            return False, RuntimeError(f"{cls.__name__} needs an NVIDIA GB10 CUDA device.")
        if isinstance(device, torch.device):
            ordinal = device.index if device.index is not None else torch.cuda.current_device()
        else:
            ordinal = torch.cuda.current_device()
        if torch.cuda.get_device_capability(ordinal) != (12, 1):
            return False, RuntimeError(f"{cls.__name__} is currently validated only for GB10 / SM121.")
        return cls.cached_validate_once()

    def pack(self, linear, scales, zeros, g_idx, **kwargs):
        super().pack(linear, scales, zeros, g_idx, **kwargs)
        # pack() stores logical zero points. Saved GPTQ-v1 conversion is a
        # separate serialization step, reversed by the loader before post_init.
        self.qzero_format(2)
        self._weight_e4m3 = self._weight_e4m3.new_empty(0)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._weight_e4m3 = self._weight_e4m3.new_empty(0)

    @torch.no_grad()
    def post_init(self):
        super().post_init()
        centered = self._centered_int4_codes()
        self._weight_e4m3 = centered.to(torch.float8_e4m3fn).contiguous()

    def _centered_int4_codes(self) -> torch.Tensor:
        """Read native GPTQ packing without changing the saved weight format."""
        if self.qweight.device.type == "meta":
            raise ValueError(f"{self.__class__.__name__} post_init requires loaded GPTQ weights.")
        expected_g_idx = torch.arange(self.in_features, device=self.g_idx.device, dtype=torch.int32) // 128
        if not torch.equal(self.g_idx, expected_g_idx):
            raise ValueError("W4AFP8 currently requires contiguous GPTQ groups and desc_act=False.")
        shifts = (torch.arange(8, device=self.qweight.device, dtype=torch.int32) * 4)
        codes = ((self.qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 15).reshape(
            self.in_features, self.out_features
        )
        zeros = ((self.qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 15).reshape(
            self.in_features // 128, self.out_features
        )
        if self.qzero_format() == 1:
            zeros = (zeros + 1) & 15
        if not bool((zeros == 8).all()):
            raise ValueError("W4AFP8 first release requires symmetric GPTQ zero point 8 in every group.")
        return codes.to(torch.int16) - zeros.repeat_interleave(128, dim=0).to(torch.int16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_e4m3.numel() != self.in_features * self.out_features:
            raise RuntimeError("W4AFP8 weight cache is absent; call post_init() after loading or packing.")
        from .w4a_activation import W4AActivation, pack_activation
        from .w4a_triton import fp8_linear, fp8_linear_prepacked

        if isinstance(x, W4AActivation):
            if x.mode != "w4afp8" or x.shape[-1] != self.in_features:
                raise ValueError("W4AFP8 requires a matching scale-aware FP8 activation.")
            needs_rotation = bool(self.online_full_had or self.online_partial_had)
            if x.rotation_applied and not needs_rotation:
                raise ValueError("This FP8 operand was rotated for a different Linear contract.")
            if needs_rotation and not x.rotation_applied:
                rotated = self._apply_rotation_to_input(x.decode(torch.float32))
                x = pack_activation(rotated, "w4afp8", model_dtype=x.model_dtype)
            result = fp8_linear_prepacked(
                x.codes, x.scales, self._weight_e4m3, self.scales, self.bias, torch.float32,
            )
            result = result.reshape(x.shape[:-1] + (self.out_features,))
            if not getattr(self, "_w4a_output_encoded", True):
                return result.to(x.model_dtype)
            return pack_activation(result, "w4afp8", model_dtype=x.model_dtype)
        if getattr(self, "_require_activation_stream", False):
            raise TypeError("This W4AFP8 Linear requires encoded FP8 input with token scales.")

        x = self._apply_rotation_to_input(x)
        return fp8_linear(x, self._weight_e4m3, self.scales, self.bias)
