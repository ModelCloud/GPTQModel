# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Effective convolution padding for calibration capture.

The Hessian is built from an im2col view of the layer input, so that view has to
describe the convolution that will actually run. `nn.Unfold` only accepts
integer zero padding, which cannot express `padding="same"` or a non-zero
`padding_mode`, so callers apply the layer's own padding first and unfold the
already-padded input.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn


SUPPORTED_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def conv_padding_pairs(module: nn.Module) -> Tuple[Tuple[int, int], ...]:
    """Return the effective ``(left, right)`` padding per spatial dimension.

    ``padding="same"`` is resolved to the split PyTorch uses: the left side
    takes ``total // 2`` and the right side takes the remainder, where
    ``total = dilation * (kernel_size - 1)``. Stride and dilation are part of the
    calculation because the effective kernel width is
    ``dilation * (kernel - 1) + 1``.

    Raises ``ValueError`` for a setting whose geometry this module cannot
    reproduce, so capture fails before calibration instead of silently
    describing a different operator.
    """

    if isinstance(module, nn.Conv1d):
        kernel, stride, dilation, padding = (
            (module.kernel_size[0],),
            (module.stride[0],),
            (module.dilation[0],),
            module.padding,
        )
    elif isinstance(module, nn.Conv2d):
        kernel, stride, dilation, padding = (
            tuple(module.kernel_size),
            tuple(module.stride),
            tuple(module.dilation),
            module.padding,
        )
    else:
        raise TypeError(
            f"conv_padding_pairs expects nn.Conv1d or nn.Conv2d, got `{type(module).__name__}`"
        )

    if module.padding_mode not in SUPPORTED_PADDING_MODES:
        raise ValueError(
            f"Convolution capture does not support `{type(module).__name__}` "
            f"padding_mode=`{module.padding_mode}`; use "
            f"{', '.join(SUPPORTED_PADDING_MODES)}."
        )

    if isinstance(padding, str):
        mode = padding.lower()
        if mode == "valid":
            return tuple((0, 0) for _ in kernel)
        if mode != "same":
            raise ValueError(
                f"Convolution capture does not support `{type(module).__name__}` "
                f"padding=`{padding}`; use 'same', 'valid', or explicit integers."
            )
        pairs = []
        for kernel_size, stride_size, dilation_size in zip(kernel, stride, dilation):
            if stride_size != 1:
                raise ValueError(
                    f"Convolution capture cannot reproduce `{type(module).__name__}` "
                    f"padding='same' with stride={stride_size}: PyTorch only defines "
                    "'same' padding for stride 1."
                )
            total = dilation_size * (kernel_size - 1)
            pairs.append((total // 2, total - total // 2))
        return tuple(pairs)

    values = tuple(int(value) for value in padding)
    return tuple((value, value) for value in values)


def pad_mode(padding_mode: str) -> str:
    """Translate a convolution padding mode to its ``torch.nn.functional.pad`` name."""

    # `nn.Conv*d` calls its zero padding "zeros"; `F.pad` calls it "constant".
    # Every other supported mode has the same name in both.
    return "constant" if padding_mode == "zeros" else padding_mode


__all__ = ["SUPPORTED_PADDING_MODES", "conv_padding_pairs", "pad_mode"]
