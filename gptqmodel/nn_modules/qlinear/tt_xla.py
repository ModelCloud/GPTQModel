# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Packed 4-bit GPTQ linear for Torch/XLA on Tenstorrent PJRT."""

from __future__ import annotations

from typing import Optional

import torch

from ...adapter.adapter import Adapter
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.torch import require_xla_runtime
from ...utils.tt_packing import (
    TTPackedGPTQLayout,
    repack_gptq_for_tt,
    unpack_tt_byte_planes_bf16,
)
from ._g_idx import validate_g_idx_bounds
from ._runtime_cache import tensor_cache_key


class XLALinear(GPTQQuantLinear):
    """Exact packed GPTQ inference through TT's Torch/XLA PJRT plugin.

    The checkpoint remains in GPTQ words, then is repacked once after placement
    into output-major uint8 byte planes. Forward transiently expands nibbles to
    BF16 for matmul; it never persists dense or expanded-code weights.
    """

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TT_XLA]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 20, FORMAT.GPTQ_V2: 20}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    SUPPORTS_DEVICES = [DEVICE.XLA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "ttxla"

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
        adapter: Optional[Adapter] = None,
        register_buffers: bool = True,
        format: Optional[FORMAT] = None,
        **kwargs,
    ):
        if adapter is not None:
            raise NotImplementedError("XLALinear does not support adapters yet.")
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TT_XLA),
            adapter=None,
            register_buffers=register_buffers,
            format=format,
            **kwargs,
        )
        self._tt_layout: Optional[TTPackedGPTQLayout] = None
        self._tt_layout_key = None

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)
        target = device if isinstance(device, torch.device) else torch.device(device)
        if target.type != "xla":
            raise NotImplementedError(f"XLALinear requires a Torch/XLA target, got {device!r}.")

    @classmethod
    def validate_once(cls):
        try:
            runtime = require_xla_runtime()
            count = int(runtime.runtime.addressable_device_count())
        except Exception as exc:
            error = RuntimeError(
                "XLALinear requires an initialized Tenstorrent PJRT runtime."
            )
            error.__cause__ = exc
            return False, error
        if count < 1:
            return False, RuntimeError("Tenstorrent PJRT exposes no addressable devices.")
        return True, None

    def _placement_key(self):
        return (
            tensor_cache_key(self.qweight),
            tensor_cache_key(self.qzeros),
            tensor_cache_key(self.scales),
            tensor_cache_key(self.g_idx),
            self.qzero_format(),
        )

    def _build_tt_layout(self) -> TTPackedGPTQLayout:
        key = self._placement_key()
        if self._tt_layout is not None and self._tt_layout_key == key:
            return self._tt_layout

        # Construction and post_init run before placement, so this remains CPU
        # safe. BaseQuantLinear._apply invalidates this cache during .to(xla:N);
        # the post-placement rebuild uploads the packed-only layout to that card.
        device = self.qweight.device
        g_idx = self.g_idx.detach().to(device="cpu", dtype=torch.int64)
        validate_g_idx_bounds(g_idx, self.scales, layer_name=type(self).__name__)
        if g_idx.numel() != self.in_features:
            raise ValueError(f"g_idx length must equal {self.in_features}, got {g_idx.numel()}.")
        if int(g_idx.min()) < 0:
            g_idx = g_idx % self.scales.shape[0]
        self._tt_layout = repack_gptq_for_tt(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            g_idx=g_idx,
            bias=self.bias,
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
            bits=self.bits,
            pack_dtype=self.pack_dtype,
            sym=self.sym,
            planar=self.planar,
            qzero_format=self.qzero_format(),
            device=device,
        )
        self._tt_layout_key = key
        return self._tt_layout

    def post_init(self):
        self._build_tt_layout()

    def clear_weight_cache(self) -> None:
        self._tt_layout = None
        self._tt_layout_key = None

    def pack(self, linear, scales, zeros, g_idx, **kwargs):
        self.clear_weight_cache()
        super().pack(linear, scales, zeros, g_idx, **kwargs)
        self.clear_weight_cache()

    def dequantize_weight(self, num_itr: int = 1):
        if num_itr != 1:
            raise NotImplementedError("XLALinear supports only the single-group iteration layout.")
        layout = self._build_tt_layout()
        codes = unpack_tt_byte_planes_bf16(layout.weight_bytes)[:, : self.in_features]
        zero_codes = unpack_tt_byte_planes_bf16(layout.zero_bytes)[:, : self.out_features]
        zero_codes = torch.where(
            layout.zero_neg_one[:, : self.out_features],
            torch.full_like(zero_codes, -1.0),
            zero_codes,
        )
        g_idx = layout.g_idx.to(torch.int64)
        return layout.scales[g_idx] * (codes.transpose(0, 1) - zero_codes[g_idx])

    def forward(self, x: torch.Tensor):
        if x.device.type != "xla":
            raise RuntimeError(f"XLALinear requires an XLA activation tensor, got {x.device}.")
        if x.dtype != torch.bfloat16:
            raise NotImplementedError(f"XLALinear requires BF16 activations, got {x.dtype}.")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}.")

        if self.training:
            raise NotImplementedError("XLALinear is inference-only.")

        out_shape = x.shape[:-1] + (self.out_features,)
        flat = x.reshape(-1, self.in_features)
        layout = self._build_tt_layout()
        codes = unpack_tt_byte_planes_bf16(layout.weight_bytes)[:, : self.in_features]
        zero_codes = unpack_tt_byte_planes_bf16(layout.zero_bytes)[:, : self.out_features]
        zero_codes = torch.where(
            layout.zero_neg_one[:, : self.out_features],
            torch.full_like(zero_codes, -1.0),
            zero_codes,
        )
        g_idx = layout.g_idx.to(torch.int64)
        weights = layout.scales[g_idx] * (codes.transpose(0, 1) - zero_codes[g_idx])
        output = torch.matmul(flat, weights).reshape(out_shape)
        if self.bias is not None:
            # B5: TT transfer rounds FP16 to BF16. Use the packed BF16 copy so
            # the bias dtype is honest before it reaches PJRT.
            output = output + layout.bias
        return output


__all__ = ["XLALinear"]
