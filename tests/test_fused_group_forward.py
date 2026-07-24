# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import transformers

from gptqmodel.nn_modules.fused_group_forward import (
    FusedGroupForward,
    clear_fused_group_forward_caches,
    install_fused_group_forward,
)
from gptqmodel.nn_modules.hooked_linear import (
    HookedConv1D,
    HookedLinear,
)


def _make_hooked_linear(in_f: int, out_f: int, device: torch.device, dtype: torch.dtype):
    m = HookedLinear(in_f, out_f)
    m.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=dtype, device=device))
    if device.type != "meta":
        m.bias = nn.Parameter(torch.randn(out_f, dtype=dtype, device=device))
    else:
        m.bias = None
    return m


def _make_hooked_conv1d(nf: int, nx: int, device: torch.device, dtype: torch.dtype):
    base = transformers.Conv1D(nf, nx)
    base.weight = nn.Parameter(torch.randn(nx, nf, dtype=dtype, device=device))
    if base.bias is not None:
        base.bias.data = base.bias.data.to(dtype=dtype, device=device)
    return HookedConv1D.from_conv1d(base)


class Collector:
    def __init__(self):
        self.calls = []

    def __call__(self, module, inp, out):
        self.calls.append((module, inp[0].detach().clone(), out.detach().clone()))


def _fuse_members(members):
    fg = FusedGroupForward(None, members)
    for m in members:
        m._fused_group_forward = fg
    return fg


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_fused_linear_matches_separate(device, dtype):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    q = _make_hooked_linear(16, 16, device, dtype)
    k = _make_hooked_linear(16, 8, device, dtype)
    v = _make_hooked_linear(16, 8, device, dtype)

    c = Collector()
    q.forward_hook = c
    k.forward_hook = c
    v.forward_hook = c

    x = torch.randn(2, 3, 16, dtype=dtype, device=device)
    _fuse_members([q, k, v])

    out_q = q(x)
    out_k = k(x)
    out_v = v(x)

    assert out_q.shape == (2, 3, 16)
    assert out_k.shape == (2, 3, 8)
    assert out_v.shape == (2, 3, 8)

    expected_q = nn.functional.linear(x.to(q.weight.dtype), q.weight, q.bias)
    expected_k = nn.functional.linear(x.to(k.weight.dtype), k.weight, k.bias)
    expected_v = nn.functional.linear(x.to(v.weight.dtype), v.weight, v.bias)

    torch.testing.assert_close(out_q, expected_q)
    torch.testing.assert_close(out_k, expected_k)
    torch.testing.assert_close(out_v, expected_v)

    assert len(c.calls) == 3
    for module, inp, out in c.calls:
        assert inp.device == device
        assert out.device == device
        if module is q:
            torch.testing.assert_close(out, expected_q)
        elif module is k:
            torch.testing.assert_close(out, expected_k)
        elif module is v:
            torch.testing.assert_close(out, expected_v)


def test_fused_linear_cache_reuses_moved_input():
    q = _make_hooked_linear(16, 16, torch.device("cpu"), torch.float32)
    k = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)
    v = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)

    fg = _fuse_members([q, k, v])
    x = torch.randn(2, 3, 16)
    _ = q(x)
    assert len(fg._cache) == 1

    _ = k(x)
    _ = v(x)
    assert len(fg._cache) == 1


def test_fused_linear_weight_update():
    q = _make_hooked_linear(16, 16, torch.device("cpu"), torch.float32)
    k = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)
    v = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)

    fg = _fuse_members([q, k, v])
    x = torch.randn(2, 3, 16)

    out_q1 = q(x)
    new_weight = torch.randn_like(q.weight)
    assert fg.update_member_weight(q, new_weight)
    out_q2 = q(x)

    expected = nn.functional.linear(x, q.weight, q.bias)
    torch.testing.assert_close(out_q2, expected)
    assert not torch.allclose(out_q1, out_q2)


def test_fused_conv1d_matches_separate():
    q = _make_hooked_conv1d(16, 16, torch.device("cpu"), torch.float32)
    k = _make_hooked_conv1d(8, 16, torch.device("cpu"), torch.float32)
    v = _make_hooked_conv1d(8, 16, torch.device("cpu"), torch.float32)

    c = Collector()
    q.forward_hook = c
    k.forward_hook = c
    v.forward_hook = c

    _fuse_members([q, k, v])
    x = torch.randn(2, 3, 16)

    out_q = q(x)
    out_k = k(x)
    out_v = v(x)

    base_conv = transformers.Conv1D(16, 16)
    base_conv.weight = q.weight
    base_conv.bias = q.bias
    expected_q = base_conv(x)

    base_conv = transformers.Conv1D(8, 16)
    base_conv.weight = k.weight
    base_conv.bias = k.bias
    expected_k = base_conv(x)

    base_conv = transformers.Conv1D(8, 16)
    base_conv.weight = v.weight
    base_conv.bias = v.bias
    expected_v = base_conv(x)

    torch.testing.assert_close(out_q, expected_q)
    torch.testing.assert_close(out_k, expected_k)
    torch.testing.assert_close(out_v, expected_v)


def test_install_fused_group_forward_wires_layer():
    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = _make_hooked_linear(16, 16, torch.device("cpu"), torch.float32)
            self.k = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)
            self.v = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)

    layer = FakeLayer()
    blocks = [["q", "k", "v"]]
    count = install_fused_group_forward(layer, blocks)
    assert count == 1
    assert isinstance(layer.q._fused_group_forward, FusedGroupForward)
    assert layer.q._fused_group_forward is layer.k._fused_group_forward

    x = torch.randn(2, 3, 16)
    out_q = layer.q(x)
    out_k = layer.k(x)
    out_v = layer.v(x)

    torch.testing.assert_close(out_q, nn.functional.linear(x, layer.q.weight, layer.q.bias))
    torch.testing.assert_close(out_k, nn.functional.linear(x, layer.k.weight, layer.k.bias))
    torch.testing.assert_close(out_v, nn.functional.linear(x, layer.v.weight, layer.v.bias))


def test_clear_fused_group_forward_caches():
    q = _make_hooked_linear(16, 16, torch.device("cpu"), torch.float32)
    k = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)
    v = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)

    fg = _fuse_members([q, k, v])
    x = torch.randn(2, 3, 16)
    q(x)
    assert len(fg._cache) == 1
    assert fg.fused_weight_storage is not None
    assert clear_fused_group_forward_caches(q)
    assert not hasattr(q, "_fused_group_forward")
    assert len(fg._cache) == 0


def test_fused_linear_splice_view_matches_contiguous():
    q = _make_hooked_linear(16, 16, torch.device("cpu"), torch.float32)
    k = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)
    v = _make_hooked_linear(16, 8, torch.device("cpu"), torch.float32)

    fg_view = FusedGroupForward(None, [q, k, v], splice="view")
    for m in [q, k, v]:
        m._fused_group_forward = fg_view

    c = Collector()
    q.forward_hook = c
    k.forward_hook = c
    v.forward_hook = c

    x = torch.randn(2, 3, 16)
    out_q = q(x)
    out_k = k(x)
    out_v = v(x)

    expected_q = nn.functional.linear(x, q.weight, q.bias)
    expected_k = nn.functional.linear(x, k.weight, k.bias)
    expected_v = nn.functional.linear(x, v.weight, v.bias)

    torch.testing.assert_close(out_q, expected_q)
    torch.testing.assert_close(out_k, expected_k)
    torch.testing.assert_close(out_v, expected_v)

    # view mode returns strided slices
    assert not out_q.is_contiguous()
    assert not out_k.is_contiguous()
    assert not out_v.is_contiguous()

    # hook still receives correct inputs/outputs
    assert len(c.calls) == 3
    for module, inp, out in c.calls:
        assert inp.shape == x.shape
        if module is q:
            torch.testing.assert_close(out, expected_q)
        elif module is k:
            torch.testing.assert_close(out, expected_k)
        elif module is v:
            torch.testing.assert_close(out, expected_v)
