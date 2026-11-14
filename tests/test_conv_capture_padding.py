# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Convolution calibration capture must describe the operator that actually runs.

``nn.Unfold`` only accepts integer zero padding, so a convolution with
``padding="same"`` or a non-zero ``padding_mode`` used to be captured with the
wrong geometry: ``"same"`` raised, and ``reflect``/``replicate``/``circular``
were silently replaced by zero padding. A wrong Hessian changes the quantized
weights without any error, so each case is pinned against an independent im2col
oracle built from the convolution's own output positions.

Both calibration-based quantizers that capture convolutions are covered: QQQ and
GPTQ. They share `gptqmodel.quantization.conv_padding` for the padding contract.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.quantization.config import QQQConfig, QuantizeConfig
from gptqmodel.quantization.conv_padding import conv_padding_pairs, pad_mode
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


QUANTIZERS = [
    pytest.param("qqq", id="qqq"),
    pytest.param("gptq", id="gptq"),
]


def _conv1d_patches(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Build the exact im2col rows PyTorch's Conv1d consumes."""

    (kernel,) = conv.kernel_size
    (dilation,) = conv.dilation
    total = dilation * (kernel - 1)
    if isinstance(conv.padding, str):
        if conv.padding == "valid":
            left = 0
            total = 0
        else:  # "same"; stride 1 is the only legal form
            left = total // 2
    else:
        (left,) = conv.padding
        total = 2 * left
    right = total - left

    if total:
        padded = F.pad(x, (left, right), mode=pad_mode(conv.padding_mode))
    else:
        padded = x
    windows = padded.unfold(-1, kernel, conv.stride[0])  # (N, C, L_out, kernel)
    # Channel-major, matching how the convolution flattens its weight.
    return windows.permute(0, 2, 1, 3).reshape(-1, kernel * x.shape[1])


def _conv2d_patches(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    kh, kw = conv.kernel_size
    dh, dw = conv.dilation
    if isinstance(conv.padding, str):
        if conv.padding == "valid":
            pad = (0, 0, 0, 0)
        else:
            top_total, left_total = dh * (kh - 1), dw * (kw - 1)
            pad = (left_total // 2, left_total - left_total // 2,
                   top_total // 2, top_total - top_total // 2)
    else:
        ph, pw = conv.padding
        pad = (pw, pw, ph, ph)

    padded = F.pad(x, pad, mode=pad_mode(conv.padding_mode)) if any(pad) else x
    # (N, C, H, W) -> (N, C, H_out, W, kh); the height dim is gone, so the
    # width is now dim 3.
    windows = padded.unfold(-2, kh, conv.stride[0])
    windows = windows.unfold(3, kw, conv.stride[1])  # (N, C, H_out, W_out, kh, kw)
    windows = windows.permute(0, 2, 3, 1, 4, 5)      # (N, H_out, W_out, C, kh, kw)
    return windows.reshape(-1, kh * kw * x.shape[1])


def _conv1d_output(patches: torch.Tensor, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """Apply the im2col rows with the conv's own weight, restoring (N, C, L)."""

    flat = conv.weight.reshape(conv.out_channels, -1)
    out = patches @ flat.t()
    return out.t().reshape(x.shape[0], conv.out_channels, -1)


def _conv2d_output(patches: torch.Tensor, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    flat = conv.weight.reshape(conv.out_channels, -1)
    out = patches @ flat.t()
    h = (x.shape[-2] + _total_pad(conv.padding[0], conv.dilation[0], conv.kernel_size[0])
         - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    w = (x.shape[-1] + _total_pad(conv.padding[1], conv.dilation[1], conv.kernel_size[1])
         - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    return out.t().reshape(x.shape[0], conv.out_channels, h, w)


def _total_pad(padding, dilation: int, kernel: int) -> int:
    if isinstance(padding, str):
        return 0 if padding == "valid" else dilation * (kernel - 1)
    return 2 * padding


def _reference_hessian(patches: torch.Tensor) -> torch.Tensor:
    rows = patches.float()
    return 2.0 * rows.t() @ rows / rows.shape[0]


def _capture(quantizer: str, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if quantizer == "qqq":
        task = QQQ(module, QQQConfig())
        task.add_batch(x, None)
        return task.materialize_hessian()

    task = GPTQ(module, QuantizeConfig(bits=4, group_size=-1, fallback=None))
    task.quantizer.configure(perchannel=True)
    task.add_batch(x, None)
    return task.finalize_hessian()


@pytest.mark.parametrize("quantizer", QUANTIZERS)
@pytest.mark.parametrize("kernel", [2, 3, 4], ids=["k2", "k3", "k4"])
def test_conv1d_same_padding_matches_the_convolution(quantizer, kernel):
    torch.manual_seed(7)
    x = torch.randn(1, 3, 9)
    conv = nn.Conv1d(3, 2, kernel, padding="same", bias=False)

    hessian = _capture(quantizer, conv, x)
    patches = _conv1d_patches(x, conv)

    # The oracle rows must reproduce the convolution before they can define a
    # Hessian: this catches an asymmetric left/right split.
    torch.testing.assert_close(_conv1d_output(patches, conv, x), conv(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hessian, _reference_hessian(patches), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quantizer", QUANTIZERS)
@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_conv1d_nonzero_padding_mode_matches_the_convolution(quantizer, mode):
    torch.manual_seed(11)
    x = torch.randn(1, 2, 8)
    conv = nn.Conv1d(2, 2, 3, padding=1, padding_mode=mode, bias=False)

    hessian = _capture(quantizer, conv, x)
    patches = _conv1d_patches(x, conv)

    torch.testing.assert_close(_conv1d_output(patches, conv, x), conv(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hessian, _reference_hessian(patches), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quantizer", QUANTIZERS)
def test_conv1d_valid_padding_ignores_no_tokens(quantizer):
    torch.manual_seed(13)
    x = torch.randn(1, 1, 7)
    conv = nn.Conv1d(1, 1, 3, padding="valid", bias=False)

    hessian = _capture(quantizer, conv, x)
    patches = _conv1d_patches(x, conv)

    assert patches.shape[0] == 5
    torch.testing.assert_close(hessian, _reference_hessian(patches), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("quantizer", QUANTIZERS)
@pytest.mark.parametrize("mode", ["zeros", "reflect"])
def test_conv2d_same_and_padding_mode_match_the_convolution(quantizer, mode):
    torch.manual_seed(17)
    x = torch.randn(1, 2, 6, 6)
    conv = nn.Conv2d(2, 3, 3, padding="same" if mode == "zeros" else 1,
                     padding_mode=mode, bias=False)

    hessian = _capture(quantizer, conv, x)
    patches = _conv2d_patches(x, conv)

    torch.testing.assert_close(_conv2d_output(patches, conv, x), conv(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hessian, _reference_hessian(patches), rtol=1e-5, atol=1e-5)


def test_conv1d_strided_same_padding_is_rejected_before_calibration():
    """PyTorch only defines ``same`` padding for stride 1."""

    with pytest.raises(ValueError, match="stride"):
        QQQ(nn.Conv1d(1, 1, 3, stride=2, padding="same"), QQQConfig())

    with pytest.raises(ValueError, match="stride"):
        GPTQ(nn.Conv1d(1, 1, 3, stride=2, padding="same"),
             QuantizeConfig(bits=4, group_size=-1, fallback=None))


def test_conv1d_unknown_padding_string_is_rejected():
    conv = nn.Conv1d(1, 1, 3, padding="valid")
    # Bypass the module constructor's own validation to reach the capture contract.
    conv.padding = "bogus"

    with pytest.raises(ValueError, match="padding"):
        QQQ(conv, QQQConfig())
    with pytest.raises(ValueError, match="padding"):
        GPTQ(conv, QuantizeConfig(bits=4, group_size=-1, fallback=None))


def test_conv1d_unknown_padding_mode_is_rejected():
    conv = nn.Conv1d(1, 1, 3, padding=1)
    conv.padding_mode = "bogus"

    with pytest.raises(ValueError, match="padding_mode"):
        QQQ(conv, QQQConfig())
    with pytest.raises(ValueError, match="padding_mode"):
        GPTQ(conv, QuantizeConfig(bits=4, group_size=-1, fallback=None))


def test_grouped_convolution_is_rejected_by_qqq():
    with pytest.raises(ValueError, match="grouped"):
        QQQ(nn.Conv1d(2, 2, 3, groups=2), QQQConfig())


def test_gptq_grouped_conv1d_capture_covers_every_group():
    """GPTQ supports groups, so its capture must keep each group's own channels."""

    torch.manual_seed(23)
    x = torch.randn(1, 2, 8)
    conv = nn.Conv1d(2, 2, 3, groups=2, padding=1, padding_mode="reflect", bias=False)

    hessian = _capture("gptq", conv, x)

    per_group = [x[:, index:index + 1] for index in range(conv.groups)]
    patches = torch.cat([_conv1d_patches(group, conv) for group in per_group], dim=0)

    # Each group's own weight slice must reproduce the grouped convolution.
    flat = conv.weight.reshape(conv.out_channels, -1)
    oracle = torch.cat(
        [(_conv1d_patches(group, conv) @ flat[row:row + 1].t()).t()
         for row, group in enumerate(per_group)],
        dim=0,
    ).unsqueeze(0)
    torch.testing.assert_close(oracle, conv(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hessian, _reference_hessian(patches), rtol=1e-6, atol=1e-6)


def test_conv_padding_pairs_matches_pytorch_same_padding():
    for kernel in (2, 3, 4, 5):
        conv = nn.Conv1d(1, 1, kernel, padding="same")
        (left, right), = conv_padding_pairs(conv)
        assert left + right == kernel - 1
        assert left == (kernel - 1) // 2

    dilated = nn.Conv1d(1, 1, 3, dilation=2, padding="same")
    (left, right), = conv_padding_pairs(dilated)
    assert (left, right) == (2, 2)
