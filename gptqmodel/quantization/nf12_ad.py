# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
"""NF12-AD quantization: lossless 12-bit BF16 weight encoding.

NF12 (NyoomFloat12) is Together AI's lossless 12-bit BF16 format. This
quantizer implements the AlpinDale *modified* tiled layout used by
https://github.com/dphnAI/nf12 and vendored under ``gptqmodel_ext/nf12``.

The only lossy step is casting non-BF16 module weights to BF16 with
round-to-nearest-even. ``encode`` is otherwise bit-exact, so a BF16 model
decodes back to the identical matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..utils import nf12_codec
from ..utils.logger import setup_logger


log = setup_logger()


@dataclass(frozen=True)
class NF12ADQuantResult:
    """Packed NF12-AD tensors for one weight.

    Only the quantized representation is retained; no dense or dequantized
    weight is stored.
    """

    packed: torch.Tensor  # uint8 [N, K * 3 / 2]
    escapes: torch.Tensor  # int16 [E, 8]
    exp_base: int
    n: int
    k: int

    @property
    def num_escapes(self) -> int:
        return self.escapes.shape[0]

    @property
    def storage_bytes(self) -> int:
        return self.packed.numel() + self.escapes.numel() * 2 + 1


def weight_to_matrix(module: nn.Module) -> torch.Tensor:
    """Return a module weight as the NF12 ``[out_features, in_features]`` matrix."""

    weight = module.weight.detach()
    if isinstance(module, _ConvNd):
        weight = weight.flatten(1)
    elif isinstance(module, transformers.pytorch_utils.Conv1D):
        weight = weight.T
    return weight


def quantize_nf12_ad_weight(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    exp_base: int | None = None,
) -> NF12ADQuantResult:
    """Quantize one 2-D weight to NF12-AD.

    Args:
        weight: ``[out_features, in_features]`` tensor in any floating dtype.
        dtype: storage dtype for the lossless encode. NF12 encodes BF16, so
            any other value is rejected by the codec.
        exp_base: optional forced exponent base (0..120). ``None`` performs
            the format's minimum-escape-base search.
    """

    if dtype != torch.bfloat16:
        raise ValueError("NF12-AD encodes bfloat16 weights; `dtype` must be torch.bfloat16")
    if weight.dim() != 2:
        raise ValueError(f"NF12-AD expects a 2-D weight, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    nf12_codec.check_shape(n, k)

    packed, escapes, base = nf12_codec.encode_weight(
        weight,
        dtype=dtype,
        exp_base=exp_base,
    )
    return NF12ADQuantResult(
        packed=packed.contiguous(),
        escapes=escapes.contiguous(),
        exp_base=int(base),
        n=n,
        k=k,
    )


class NF12ADQuantizer:
    """Quantize one ``nn.Linear``/``Conv1D``/``ConvNd`` module to NF12-AD."""

    def __init__(self, module: nn.Module, name: str | None = None):
        if not isinstance(
            module,
            (nn.Linear, _ConvNd, transformers.pytorch_utils.Conv1D),
        ):
            raise NotImplementedError(
                "NF12-AD supports nn.Linear, transformers.Conv1D, and ConvNd modules, "
                f"got {type(module).__name__}"
            )
        self.module = module
        self.name = name
        self.result: NF12ADQuantResult | None = None

    def quantize(self, *, exp_base: int | None = None) -> NF12ADQuantResult:
        matrix = weight_to_matrix(self.module)
        self.result = quantize_nf12_ad_weight(matrix, exp_base=exp_base)
        return self.result


__all__ = [
    "NF12ADQuantResult",
    "NF12ADQuantizer",
    "quantize_nf12_ad_weight",
    "weight_to_matrix",
]
