# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.lookahead import configure_default_lookahead
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear


def _mock_gptq_linear(bits: int, group_size: int, in_features: int, out_features: int) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor]:
    maxq = (1 << (bits - 1)) - 1
    weight = torch.randn((in_features, out_features), dtype=torch.float32)

    if group_size != -1:
        reshaped = weight.view(in_features // group_size, group_size, out_features)
        w_g = reshaped.permute(1, 0, 2).reshape(group_size, -1)
    else:
        w_g = weight

    scales = torch.maximum(
        w_g.abs().max(dim=0, keepdim=True).values,
        torch.full((1, w_g.shape[1]), 1e-6, device=w_g.device),
    )
    scales = scales / maxq

    q = torch.round(w_g / scales).clamp_(-maxq, maxq)
    ref = (q * scales).to(dtype=torch.float16)

    if group_size != -1:
        ref = ref.reshape(group_size, -1, out_features)
        ref = ref.permute(1, 0, 2).reshape(in_features, out_features)

        q = q.reshape(group_size, -1, out_features)
        q = q.permute(1, 0, 2).reshape(in_features, out_features)

    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = ref.t().contiguous()

    scales = scales.reshape(-1, out_features).contiguous()
    zeros = torch.zeros_like(scales, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // (
        group_size if group_size != -1 else in_features
    )

    return linear, scales, zeros, g_idx


@pytest.mark.cuda
@pytest.mark.parametrize("group_size", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_triton_large_group_sizes(group_size: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 not supported on this device")

    torch.cuda.set_device(0)

    bits = 4
    in_features = 4096
    out_features = 4096

    torch.manual_seed(0)

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)

    torch_module = TorchQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    torch_module.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    torch_module.post_init()

    try:
        triton_module = TritonV2QuantLinear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            bias=False,
        )
    except ValueError as err:
        pytest.skip(f"Triton backend unavailable: {err}")

    triton_module.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    triton_module.post_init()

    device = torch.device("cuda:0")
    torch_module = torch_module.to(device=device, dtype=dtype).eval()
    triton_module = triton_module.to(device=device, dtype=dtype).eval()

    batch = 8
    x = torch.randn((batch, in_features), device=device, dtype=dtype)

    with torch.inference_mode():
        torch_out = torch_module(x)
        triton_out = triton_module(x)

    torch_out = torch_out.to(torch.float32)
    triton_out = triton_out.to(torch.float32)

    assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)
    assert torch_out.abs().max() > 0

######### test_torch_weight_cache.py #########


def _make_module(device: torch.device):
    module = TorchQuantLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
    ).to(device)

    with torch.no_grad():
        module.qweight.zero_()
        module.qzeros.zero_()
        module.scales.fill_(1.0)
        module.bias.uniform_(-0.1, 0.1)

    module.qzero_format(format=2)
    module.post_init()
    module.eval()
    return module


def test_cached_forward_matches_baseline():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    module = _make_module(device)

    x = torch.randn(8, module.in_features, device=device, dtype=torch.float16)

    module.enable_weight_cache(False)
    ref = module(x)

    module.enable_weight_cache(True)
    module.clear_weight_cache()
    cached = module(x)

    torch.testing.assert_close(ref, cached)
    assert x.dtype in module._cached_weights
    assert module._cached_weights[x.dtype].device.type == device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required for lookahead prefetch test")
def test_lookahead_prefetch_single_step():
    device = torch.device("cuda")
    producer = _make_module(device)
    consumer = _make_module(device)

    producer.enable_lookahead(True).set_lookahead_next(consumer)
    consumer.enable_lookahead(True)

    x = torch.randn(4, producer.in_features, device=device, dtype=torch.float16)

    producer(x)
    assert torch.float16 in consumer._prefetched_weights

    consumer(x)
    assert torch.float16 not in consumer._prefetched_weights


def test_configure_default_lookahead_chain():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_module(device)
            self.k_proj = _make_module(device)
            self.v_proj = _make_module(device)
            self.o_proj = _make_module(device)

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_module(device)
            self.up_proj = _make_module(device)
            self.down_proj = _make_module(device)

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttn()
            self.mlp = DummyMLP()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DummyLayer()])

    model = DummyModel()
    for module in model.modules():
        if isinstance(module, TorchQuantLinear):
            module.enable_lookahead(True)

    configure_default_lookahead(model)

    layer = model.layers[0]
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    down_proj = layer.mlp.down_proj

    assert q_proj._lookahead_next == (gate_proj, up_proj, down_proj)
    assert q_proj._lookahead_enabled

    for module in (k_proj, v_proj, o_proj):
        assert module._lookahead_next is None
        assert not module._lookahead_enabled

    for module in (gate_proj, up_proj, down_proj):
        assert module._lookahead_next is None
        assert module._lookahead_enabled
