# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ convolution capture must match the unfolded-input contract.

``QQQ._validate_module`` accepts convolution layers, so every accepted layout
must produce a Hessian whose width equals ``QQQ.columns`` and whose values match
the im2col reference. Grouped convolutions cannot satisfy that contract: the
unfolded input spans all input channels while the weight width is per-group, so
they are rejected before calibration instead of failing later in ``add_batch``.
"""

import pytest
import torch

from gptqmodel.quantization.config import QQQConfig
from gptqmodel.quantization.qqq import QQQ


def _conv_reference_hessian(x, layer):
    """Return ``2/N * (im2col(x) @ im2col(x)^T)`` in float64."""

    x = x.double()
    if isinstance(layer, torch.nn.Conv1d):
        cols = torch.nn.functional.unfold(
            x.unsqueeze(2),
            (1, layer.kernel_size[0]),
            dilation=(1, layer.dilation[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
        )
    else:
        cols = torch.nn.functional.unfold(
            x,
            layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )
    cols = cols.permute(1, 0, 2).flatten(1)
    return (2.0 / cols.shape[1]) * cols.matmul(cols.t())


def _normalized_error(actual, reference):
    return float(
        torch.linalg.vector_norm(actual.double() - reference)
        / torch.linalg.vector_norm(reference)
    )


@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation",
    [(3, 1, 0, 1), (3, 2, 1, 1), (3, 1, 2, 2)],
)
def test_conv1d_capture_matches_im2col_reference(kernel_size, stride, padding, dilation):
    torch.manual_seed(42)
    layer = torch.nn.Conv1d(
        2,
        3,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
    )
    qqq = QQQ(layer, QQQConfig())
    x = torch.randn(2, 2, 12, dtype=torch.float32)

    qqq.add_batch(x, None)
    H = qqq.materialize_hessian()

    assert qqq.columns == 2 * kernel_size
    assert H.shape == (qqq.columns, qqq.columns)
    assert _normalized_error(H, _conv_reference_hessian(x, layer)) <= 1e-6


def test_conv2d_capture_matches_im2col_reference():
    torch.manual_seed(42)
    layer = torch.nn.Conv2d(4, 3, 3, padding=1, stride=2, dtype=torch.float32)
    qqq = QQQ(layer, QQQConfig())
    x = torch.randn(1, 4, 7, 7, dtype=torch.float32)

    qqq.add_batch(x, None)
    H = qqq.materialize_hessian()

    assert qqq.columns == 36
    assert H.shape == (36, 36)
    assert _normalized_error(H, _conv_reference_hessian(x, layer)) <= 1e-6


@pytest.mark.parametrize(
    "layer",
    [
        torch.nn.Conv2d(4, 4, 3, groups=2),
        torch.nn.Conv1d(4, 4, 3, groups=2),
    ],
    ids=["conv2d_groups2", "conv1d_groups2"],
)
def test_grouped_convolutions_are_rejected_before_capture(layer):
    with pytest.raises(ValueError, match="grouped convolution"):
        QQQ(layer, QQQConfig())


def test_conv1d_quantization_preserves_weight_layout():
    torch.manual_seed(0)
    layer = torch.nn.Conv1d(2, 3, 3, dtype=torch.float32)
    qqq = QQQ(layer, QQQConfig())
    qqq.add_batch(torch.randn(2, 2, 9, dtype=torch.float32), None)
    qqq.quantizer.configure(bits=4, perchannel=True, groupsize=-1, sym=True)

    quantized = qqq.quantize(blocksize=6)[0]

    assert quantized.shape == layer.weight.shape
