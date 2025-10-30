# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.awq.utils.module import get_op_name


class _DummySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)


class _DummyMLP(nn.Module):
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Identity()
        # MoE-specific modules intentionally missing quantizable projections
        self.shared_experts = nn.Module()
        self.experts = nn.Module()


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _DummySelfAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _DummyMLP()


class _DummyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_DummyLayer()])


class _DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyStack()
        self.config = SimpleNamespace(n_routed_experts=2)


class _HybridGLMStub(BaseQModel):
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "q_norm:0:!", "k_proj:0", "k_norm:0:!", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "gate_proj": ("gate_proj",),
                "down_proj": ("down_proj",),
                "up_proj": ("up_proj",),
            },
        },
    ]
    dynamic_expert_index = "n_routed_experts"
    layer_modules_strict = False


def _build_stub_model():
    transformer = _DummyTransformer()
    qmodel = _HybridGLMStub.__new__(_HybridGLMStub)
    nn.Module.__init__(qmodel)
    qmodel.model = transformer
    # Minimal attribute used by awq_get_modules_for_scaling path
    qmodel.quantize_config = SimpleNamespace(offload_to_disk=False)
    return qmodel, transformer.model.layers[0]


def test_awq_scaling_handles_sparse_moe_layers():
    qmodel, layer = _build_stub_model()
    features = {
        "self_attn.q_proj": torch.randn(8, 4),
        "self_attn.k_proj": torch.randn(8, 4),
        "self_attn.v_proj": torch.randn(8, 4),
        "self_attn.o_proj": torch.randn(8, 4),
        "mlp.gate_proj": torch.randn(8, 4),
        "mlp.down_proj": torch.randn(8, 4),
        "mlp.up_proj": torch.randn(8, 4),
    }

    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    collected = [
        get_op_name(qmodel.model, submodule)
        for node in nodes
        for submodule in node["layers"]
    ]

    assert any(name.endswith("self_attn.q_proj") for name in collected)
    assert any(name.endswith("mlp.gate_proj") for name in collected)
    assert any(name.endswith("mlp.down_proj") for name in collected)
    assert any(name.endswith("mlp.up_proj") for name in collected)
