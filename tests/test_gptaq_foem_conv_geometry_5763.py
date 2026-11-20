# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""GPTAQ/FOEM must calibrate Conv1d and grouped Conv2d like plain GPTQ (#5763).

The correction variants built their im2col view only for ungrouped Conv2d. A
Conv1d capture then called ``.t()`` on a three-dimensional input, and a grouped
Conv2d produced ``in_channels * kernel`` columns for a weight that only has
``in_channels // groups * kernel``. Plain GPTQ already handled both. All three
solvers now share ``conv_patch_matrix``.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.conv_padding import conv_padding_pairs, pad_mode
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


KERNEL = 3
# fp32 accumulation against a float64 oracle on large-magnitude fixtures.
REL_TOL = 1e-5


def _assert_close(actual, expected):
    error = ((actual - expected).norm() / expected.norm()).item()
    assert error < REL_TOL, f"relative Frobenius error {error:.3e} >= {REL_TOL:.1e}"


def _oracle_patches(layer, x):
    """Independent grouped im2col, built from the module's weight geometry."""

    groups = layer.groups
    x = x.reshape(x.shape[0] * groups, x.shape[1] // groups, *x.shape[2:])
    if isinstance(layer, nn.Conv1d):
        (left, right), = conv_padding_pairs(layer)
        # ``x`` is already [batch * groups, channels, length]; pad length only.
        x = F.pad(x, (left, right), mode=pad_mode(layer.padding_mode))
        return (
            x.unfold(2, KERNEL, 1)
            .permute(0, 2, 1, 3)
            .flatten(0, 1)
            .flatten(1)
        )
    (top, bottom), (left, right) = conv_padding_pairs(layer)
    x = F.pad(x, (left, right, top, bottom), mode=pad_mode(layer.padding_mode))
    return (
        x.unfold(2, KERNEL, 1)
        .unfold(3, KERNEL, 1)
        .permute(0, 2, 3, 1, 4, 5)
        .flatten(0, 2)
        .flatten(1)
    )


CASES = {
    "conv1d": (nn.Conv1d(2, 4, KERNEL, bias=False), torch.arange(20.0).reshape(1, 2, 10)),
    "conv1d-grouped": (
        nn.Conv1d(4, 4, KERNEL, padding=1, groups=2, bias=False),
        torch.arange(20.0).reshape(1, 4, 5),
    ),
    "conv2d-grouped": (
        nn.Conv2d(4, 4, KERNEL, groups=2, bias=False),
        torch.arange(100.0).reshape(1, 4, 5, 5),
    ),
    "conv2d": (nn.Conv2d(2, 4, KERNEL, bias=False), torch.arange(50.0).reshape(1, 2, 5, 5)),
}


def _task(kind, layer, native):
    wrapped = NamedModule(layer, name="conv", full_name="model.layers.0.conv", layer_index=0)
    wrapped.state[NATIVE_INPUTS_STATE_KEY] = [native]
    if kind is GPTAQ:
        kwargs = {"gptaq": GPTAQConfig(alpha=0.25)}
    else:
        kwargs = {"foem": FOEMConfig(alpha=0.25)}
    qcfg = QuantizeConfig(
        bits=4, group_size=-1, act_group_aware=False, offload_to_disk=False, **kwargs
    )
    task = kind(wrapped, qcfg)
    task.quantizer.configure(perchannel=True)
    return task


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_conv_patch_matrix_matches_the_grouped_oracle(case):
    """The shared helper folds groups and keeps the weight column count."""

    from gptqmodel.quantization.gptq import conv_patch_matrix

    layer, x = CASES[case]
    torch.manual_seed(0)
    expected = _oracle_patches(layer, x).double()
    actual = conv_patch_matrix(layer, x).double()

    torch.testing.assert_close(actual, expected)
    columns = layer.weight.shape[1] * KERNEL ** (x.dim() - 2)
    assert actual.shape[1] == columns
    if layer.groups == 1:
        # A grouped weight's output channels only see their own group's
        # patches, so one dense product is only meaningful when ungrouped.
        # The patches must reproduce the layer output before they are trusted.
        weight = layer.weight.detach().double().reshape(layer.weight.shape[0], -1)
        out = layer(x)
        spatial = out.shape[2:]
        dense = (actual @ weight.t()).reshape(x.shape[0], *spatial, -1)
        dense = dense.permute(0, -1, *range(1, 1 + len(spatial)))
        torch.testing.assert_close(dense, out.double(), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("kind", (GPTAQ, FOEM), ids=("gptaq", "foem"))
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_correction_capture_matches_the_conv_oracle(kind, case):
    """Hessian and native cross term follow the grouped im2col geometry."""

    layer, x = CASES[case]
    torch.manual_seed(0)
    native = x + 1.0
    task = _task(kind, layer, native)

    task.add_batch(x, None)

    quantized = _oracle_patches(layer, x).double()
    native_patches = _oracle_patches(layer, native).double()
    patches = quantized.shape[0]
    normalizer = 1.0 if kind is GPTAQ else float(patches)

    expected_hessian = (2.0 / normalizer) * (quantized.T @ quantized)
    expected_cross = (
        (2.0 / normalizer) * ((native_patches - quantized).T @ quantized)
    )

    assert task.columns == layer.weight.shape[1] * KERNEL ** (x.dim() - 2)
    _assert_close(task.H.double(), expected_hessian)
    _assert_close(task.dXXT.double(), expected_cross)


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_plain_gptq_uses_the_same_geometry(case):
    """The shared helper must not change plain GPTQ's Hessian."""

    layer, x = CASES[case]
    torch.manual_seed(0)
    task = GPTQ(
        NamedModule(layer, name="conv", full_name="model.layers.0.conv", layer_index=0),
        QuantizeConfig(bits=4, group_size=-1, act_group_aware=False, offload_to_disk=False),
    )
    task.quantizer.configure(perchannel=True)
    task.add_batch(x, None)

    patches = _oracle_patches(layer, x).double()
    expected = (2.0 / patches.shape[0]) * (patches.T @ patches)
    _assert_close(task.finalize_hessian().double(), expected)
