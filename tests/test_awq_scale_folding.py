# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for AWQ scale folding into normalization modules.

Two independent oracles are used:

* the pre-fold ``norm -> linear`` output computed by plain PyTorch modules;
* the closed-form folded weights ``(1 + w) / s - 1`` and ``fc_w * s``.
"""

import types

import pytest
import torch
from torch import nn
from transformers import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

from gptqmodel.models.definitions.gemma2 import Gemma2QModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.quantization.awq.quantize.scale import scale_ln_fcs


try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
except Exception:  # pragma: no cover - older transformers builds
    Qwen3_5RMSNorm = None


class _SyntheticQwen35MoeRMSNorm(nn.Module):
    """Future-style ``1 + weight`` norm that subclasses no known norm class."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states):
        output = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (output * (1.0 + self.weight.float())).type_as(hidden_states)


# Give the synthetic class the name the issue cites so name-based heuristics
# cannot accidentally pass the test.
_SyntheticQwen35MoeRMSNorm.__name__ = "Qwen3_5MoeRMSNorm"


def _norm_cases():
    cases = [
        ("rmsnorm", lambda dim: nn.RMSNorm(dim), 0.0),
        ("llama", lambda dim: LlamaRMSNorm(dim), 0.0),
        ("gemma", lambda dim: GemmaRMSNorm(dim), 1.0),
        ("gemma2", lambda dim: Gemma2RMSNorm(dim), 1.0),
        ("gemma3", lambda dim: Gemma3RMSNorm(dim), 1.0),
        ("qwen3_next", lambda dim: Qwen3NextRMSNorm(dim), 1.0),
        ("qwen3_5_moe", lambda dim: _SyntheticQwen35MoeRMSNorm(dim), 1.0),
    ]
    if Qwen3_5RMSNorm is not None:
        cases.append(("qwen3_5", lambda dim: Qwen3_5RMSNorm(dim), 1.0))
    return cases


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-12))


@pytest.mark.parametrize("name,norm_factory,offset", _norm_cases(), ids=lambda value: value if isinstance(value, str) else None)
def test_scale_ln_fcs_resolves_gain_offset_from_semantics(name, norm_factory, offset):
    torch.manual_seed(0)
    dim = 32
    norm = norm_factory(dim)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim) * 0.1)
    linear = nn.Linear(dim, dim, bias=False)

    scales = torch.rand(dim) + 0.5
    weight_before = norm.weight.detach().clone()
    linear_before = linear.weight.detach().clone()
    x = torch.randn(8, dim)
    with torch.no_grad():
        before = linear(norm(x))
        scale_ln_fcs(norm, [linear], scales)
        after = linear(norm(x))

    assert _relative_error(after, before) < 1e-6, name

    # Closed-form oracle for the stored weights, independent of the fold code.
    expected_norm = (weight_before + offset) / scales - offset
    torch.testing.assert_close(norm.weight.detach(), expected_norm, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        linear.weight.detach(),
        linear_before * scales.view(1, -1),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "dtype,tolerance",
    # BF16 stores ~8 significand bits, so the folded gain `(1 + w) / s - 1`
    # rounds more coarsely than FP16. The tolerance absorbs that storage
    # rounding only; the pre-fix wrong-semantics error is ~0.44.
    [(torch.float16, 2e-3), (torch.bfloat16, 1e-2)],
)
def test_scale_ln_fcs_offset_gain_folds_in_float32(dtype, tolerance):
    torch.manual_seed(0)
    dim = 32
    norm = Gemma3RMSNorm(dim).to(dtype)
    with torch.no_grad():
        norm.weight.copy_((torch.randn(dim, dtype=torch.float32) * 0.1).to(dtype))
    linear = nn.Linear(dim, dim, bias=False).to(dtype)

    scales = torch.rand(dim, dtype=torch.float32) + 0.5
    x = torch.randn(8, dim).to(dtype)
    weight_before = norm.weight.detach().clone()
    with torch.no_grad():
        before = linear(norm(x)).float()
        scale_ln_fcs(norm, [linear], scales)
        after = linear(norm(x)).float()

    assert _relative_error(after, before) < tolerance

    # The intermediate `1 + w` and the division are performed in float32; only
    # the final stored gain is rounded back to the serving dtype.
    expected = ((weight_before.float() + 1.0) / scales - 1.0).to(dtype)
    torch.testing.assert_close(norm.weight.detach(), expected, rtol=0.0, atol=0.0)


def _tiny_gemma2() -> Gemma2ForCausalLM:
    torch.manual_seed(0)
    return Gemma2ForCausalLM(Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )).eval()


def _bare_gemma2_qmodel(hf_model):
    qmodel = Gemma2QModel.__new__(Gemma2QModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    return qmodel


def test_gemma2_mlp_prev_op_is_pre_feedforward_layernorm():
    hf_model = _tiny_gemma2()
    qmodel = _bare_gemma2_qmodel(hf_model)
    layer = hf_model.model.layers[0]
    hidden = hf_model.config.hidden_size
    features = {
        name: torch.randn(4, hidden)
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }

    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})

    attention = next(node for node in nodes if any(module is layer.self_attn.q_proj for module in node["layers"]))
    mlp = next(node for node in nodes if any(module is layer.mlp.gate_proj for module in node["layers"]))

    assert attention["prev_op"] is layer.input_layernorm
    assert mlp["prev_op"] is layer.pre_feedforward_layernorm
    assert mlp["prev_op"] is not layer.post_attention_layernorm


def test_gemma2_mlp_scale_fold_is_output_neutral():
    hf_model = _tiny_gemma2()
    qmodel = _bare_gemma2_qmodel(hf_model)
    layer = hf_model.model.layers[0]
    hidden = hf_model.config.hidden_size
    features = {
        name: torch.randn(4, hidden)
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    mlp = next(node for node in nodes if any(module is layer.mlp.gate_proj for module in node["layers"]))

    x = torch.randn(4, hidden)
    scales = torch.rand(hidden) + 0.5
    with torch.no_grad():
        before = layer.mlp(layer.pre_feedforward_layernorm(x))
        scale_ln_fcs(mlp["prev_op"], mlp["layers"], scales)
        after = layer.mlp(layer.pre_feedforward_layernorm(x))

    assert _relative_error(after, before) < 1e-5


class _MoeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 2, bias=False)
        self.up_proj = nn.Linear(4, 2, bias=False)
        self.down_proj = nn.Linear(2, 4, bias=False)


class _MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = nn.Module()
        self.self_attn.q_norm = nn.LayerNorm(4)
        self.self_attn.k_norm = nn.LayerNorm(4)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(4, 4, bias=False))
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(4, 2, bias=False)
        self.mlp.experts = nn.ModuleList([_MoeExpert(), _MoeExpert()])


def test_moe_router_is_not_selected_as_prev_op():
    """The router is in the same non-quantized run but does not feed the experts."""
    qmodel = Qwen3MoeQModel.__new__(Qwen3MoeQModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = types.SimpleNamespace(
        config=types.SimpleNamespace(num_experts=2),
    )
    layer = _MoeLayer()
    features = {
        name: torch.randn(1, 3, 4)
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp")
    }
    for expert_index in range(2):
        prefix = f"mlp.experts.{expert_index}"
        features[f"{prefix}.gate_proj"] = torch.randn(1, 3, 4)
        features[f"{prefix}.up_proj"] = torch.randn(1, 3, 4)
        features[f"{prefix}.down_proj"] = torch.randn(1, 3, 2)

    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    gate_up = next(node for node in nodes if len(node["layers"]) == 4)

    assert gate_up["prev_op"] is layer.post_attention_layernorm
    assert gate_up["prev_op"] is not layer.mlp.gate
