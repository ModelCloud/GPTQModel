# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Portable paths must keep finite reductions wide until the visible cast (#4471).

These tests compare against an independent float64 oracle, never against the
implementation under test. They cover the Torch portable paths: the Paro
CPU ATen child whose product feeds an outer bias epilogue, and the local Mamba
incremental state update whose half products feed an FP32 recurrent state.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import _has_local_int4pack_cpu_ops
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.backend import BACKEND


_MAMBA_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "gptqmodel/hf_kernels/mamba_ssm/ops/triton/selective_state_update.py"
)


def _rearrange(tensor, pattern):
    """Tiny einops stand-in for the module's append/insert-singleton patterns."""
    source, target = pattern.split("->")
    source_dims = source.split()
    target_dims = target.split()
    assert [dim for dim in target_dims if dim != "1"] == source_dims, pattern
    output = tensor
    for index, dim in enumerate(target_dims):
        if dim == "1":
            output = output.unsqueeze(index)
    return output


def _repeat(tensor, pattern, **kwargs):
    """Tiny einops stand-in for ``b g n -> b (g h) n``."""
    source, target = pattern.split("->")
    assert source.split() == ["b", "g", "n"], pattern
    assert target.split() == ["b", "(g", "h)", "n"], pattern
    return tensor.repeat_interleave(kwargs["h"], dim=1)


def _load_mamba(monkeypatch):
    try:
        import einops  # noqa: F401
    except ModuleNotFoundError:
        shim = ModuleType("einops")
        shim.rearrange = _rearrange
        shim.repeat = _repeat
        monkeypatch.setitem(sys.modules, "einops", shim)

    spec = importlib.util.spec_from_file_location(
        "issue_4471_selective_state_update", _MAMBA_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paro_cpu_aten_child_keeps_wide_product_for_outer_bias():
    """A finite pre-bias product must survive the parent's outer bias add."""
    if not _has_local_int4pack_cpu_ops():
        pytest.skip("native CPU int4pack ops are unavailable")

    in_features, out_features, group_size = 32, 16, 32
    qweight, qzeros = pack_awq(
        torch.ones((in_features, out_features), dtype=torch.int32),
        torch.full((in_features // group_size, out_features), 64, dtype=torch.int32),
        4,
    )
    layer = ParoLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        backend=BACKEND.PARO_CUDA,
    )
    layer.qweight.copy_(qweight)
    layer.qzeros.copy_(qzeros)
    layer.scales.fill_(8.0)
    layer.bias.data.fill_(-1024.0)
    layer.eval()

    x = torch.full((1, in_features), 258.0, dtype=torch.float16)
    weight = torch.full((in_features, out_features), 8.0, dtype=torch.float64)
    bias = torch.full((out_features,), -1024.0, dtype=torch.float64)
    expected = (x.double() @ weight + bias).to(x.dtype)

    actual = layer(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mamba_incremental_state_keeps_fp32_products_and_readout(monkeypatch):
    """Half products must not overflow before the FP32 state/reduction."""
    module = _load_mamba(monkeypatch)

    state = torch.zeros(1, 1, 1, dtype=torch.float32)
    x = torch.tensor([[32768.0]], dtype=torch.float16)
    dt = torch.ones_like(x)
    A = torch.zeros(1, 1, dtype=torch.float32)
    B = torch.tensor([[2.0]], dtype=torch.float16)
    C = torch.tensor([[0.5]], dtype=torch.float16)

    out = module.selective_state_update(state, x, dt, A, B, C)

    expected_state_2d = x.double() * dt.double() * B.double()
    expected_state = expected_state_2d.reshape(state.shape)
    expected_out = (expected_state_2d * C.double()).to(x.dtype)
    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state.to(state.dtype), rtol=0, atol=0)
