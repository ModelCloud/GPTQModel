# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.lookahead import configure_default_lookahead
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear


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
