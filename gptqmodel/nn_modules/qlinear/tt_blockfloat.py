# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tenstorrent block-float inference linear layer.

The weights of a Tenstorrent accelerator are stored as blocks of 16 values that
share one exponent, with a fixed number of mantissa bits per value:
``bfloat4_b`` keeps 2 and ``bfloat8_b`` keeps 6. That is a storage format the
hardware unpacks natively, so a checkpoint written this way is already in the
layout the device wants and does not need a host-side requantisation step.

This module keeps that representation: the packed mantissa codes and the int8
exponents are the module's state, and a dense weight only exists for the
duration of one forward pass. Nothing is cached between calls, so the
quantized-storage invariant holds for this path exactly as it does for the
packed GPTQ kernels.

The Tenstorrent device kernels consume the codes directly. This module is the
portable fallback that makes a ``tt-blockfloat`` checkpoint loadable and
verifiable on any host, which is what the loader and the tests use.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import BaseQuantLinear


log = setup_logger()

#: Values per shared exponent. The hardware format fixes this; a checkpoint
#: whose block size differs cannot be fed to a device kernel unchanged.
BLOCKFLOAT_BLOCK = 16


class TTBlockfloatLinear(BaseQuantLinear):
    """Portable linear over Tenstorrent block-float weights.

    Storage is ``codes`` (two 4-bit codes per byte, or one 8-bit code) plus
    ``exponents`` (one int8 per block of ``block`` values along the input
    dimension). ``dequantize`` rebuilds the exact grid weights; the forward pass
    uses them and drops the reference.
    """

    SUPPORTS_BACKENDS = [BACKEND.TT_BLOCKFLOAT]
    SUPPORTS_METHODS = [METHOD.TT_BLOCKFLOAT]
    SUPPORTS_FORMATS = {FORMAT.TT_BLOCKFLOAT: 100}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    # One exponent covers ``block`` consecutive input features, so the input
    # dimension must be a whole number of blocks.
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [BLOCKFLOAT_BLOCK]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_PACK_DTYPES = [torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool,
        backend: BACKEND,
        adapter: Optional[Adapter] = None,
        name: str = None,
        dtype: Optional[torch.dtype] = None,
        block: int = BLOCKFLOAT_BLOCK,
        pack_dtype: torch.dtype = torch.uint8,
        **kwargs,
    ):
        if block != BLOCKFLOAT_BLOCK:
            raise NotImplementedError(
                f"{type(self).__name__} stores one exponent per {BLOCKFLOAT_BLOCK} "
                f"values; block={block} is not a hardware format"
            )
        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=backend,
            adapter=adapter,
            name=name,
            register_buffers=False,
            dtype=dtype,
            validate_kwargs={"pack_dtype": pack_dtype},
        )
        self.block = block
        self.pack_dtype = pack_dtype

        # Populated by ``pack`` or by the loader. The shapes here are the ones
        # the checkpoint declares; they are replaced on load.
        self.register_buffer(
            "codes",
            torch.zeros(0, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "code_shape",
            torch.zeros(2, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "exponents",
            torch.zeros(0, dtype=torch.int8),
            persistent=True,
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    # -- construction ------------------------------------------------------

    def pack(
        self,
        linear: nn.Module,
        scales=None,
        zeros=None,
        g_idx=None,
        **kwargs,
    ):
        """Quantise a dense ``linear`` onto the block-float grid."""

        from ...utils.tt_blockfloat_quant import quantize_blockfloat

        weight = linear.weight.detach().to(torch.float32)
        grid = quantize_blockfloat(weight, self.bits, block=self.block)[0]
        self.load_codes_from(grid)
        if getattr(linear, "bias", None) is not None:
            self.bias = linear.bias.detach().to(torch.float16).contiguous()
        return self

    def load_codes_from(self, grid_weight: torch.Tensor) -> None:
        """Store an already-on-grid weight as packed codes plus exponents."""

        from ...utils.tt_blockfloat_quant import decompose_blockfloat

        codes, exponents = decompose_blockfloat(
            grid_weight.detach().to(torch.float32), bits=self.bits, block=self.block
        )
        self.set_quantized_state(codes, exponents)

    def set_quantized_state(self, codes: torch.Tensor, exponents: torch.Tensor) -> None:
        """Install packed codes and exponents directly (used by the loader)."""

        from ...utils.tt_blockfloat_quant import _pack_codes

        codes = codes.to(torch.int64)
        if tuple(codes.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"{self.name}: code shape {tuple(codes.shape)} does not match "
                f"({self.out_features}, {self.in_features})"
            )
        expected = (self.out_features, self.in_features // self.block)
        if tuple(exponents.shape) != expected:
            raise ValueError(
                f"{self.name}: exponent shape {tuple(exponents.shape)} does not "
                f"match {expected} (one exponent per {self.block} values)"
            )
        self.codes = _pack_codes(codes, self.bits).contiguous()
        self.code_shape = torch.tensor(
            [self.out_features, self.in_features], dtype=torch.int64
        )
        self.exponents = exponents.to(torch.int8).contiguous()

    # -- numerics ----------------------------------------------------------

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Rebuild the exact grid weights as ``[out_features, in_features]``."""

        from ...utils.tt_blockfloat_quant import _unpack_codes, recompose_blockfloat

        if self.codes.numel() == 0:
            raise RuntimeError(f"{self.name}: no block-float weights loaded")
        shape = tuple(int(v) for v in self.code_shape)
        codes = _unpack_codes(self.codes, self.bits, shape)
        weight = recompose_blockfloat(
            codes, self.exponents, bits=self.bits, block=self.block
        )
        return weight.to(dtype)

    def dense_weight(self) -> torch.Tensor:
        """Dense view of the weights, computed on demand and never retained.

        Deliberately not named ``weight``: a ``weight`` attribute would read as
        a dense parameter to module-tree code that tests ``module.weight.dim()``
        and would invite callers to hold the expanded tensor.
        """

        return self.dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return self.empty_linear_output(x)

        out_dtype = x.dtype
        weight = self.dequantize(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        out = F.linear(x, weight, bias)
        # Release the temporary before the activation leaves this scope.
        del weight

        if self.adapter is not None:
            out = self.adapter.apply(x=x, out=out)
        return out.to(out_dtype)


__all__ = ["BLOCKFLOAT_BLOCK", "TTBlockfloatLinear"]
