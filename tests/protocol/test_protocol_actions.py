# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.quantization import GPTQConfig
from gptqmodel.quantization.protocol import (
    ExecutionPlan,
    OperationSpec,
    ProtocolActionError,
    QuantizeSpec,
    Rule,
    Stage,
    TargetSpec,
    apply_plan_actions,
    compile_plan_to_quantize_config,
    compile_protocol_yaml_to_quantize_config,
    compile_rule_actions,
    known_action_methods,
)


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int = 16, inter: int = 32):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, inter)
        self.up_proj = nn.Linear(hidden, inter)
        self.down_proj = nn.Linear(inter, hidden)

    def forward(self, x):
        hidden = self.input_layernorm(x)
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        attention = self.o_proj(query * torch.sigmoid(key) + value)
        x = x + attention
        mlp_input = self.post_attention_layernorm(x)
        mlp = self.down_proj(F.gelu(self.gate_proj(mlp_input)) * self.up_proj(mlp_input))
        return x + mlp


class _TinyModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_TinyBlock(hidden)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def _tiny_model() -> tuple[_TinyModel, torch.Tensor]:
    torch.manual_seed(0)
    model = _TinyModel().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) and module.bias is not None:
                module.bias.copy_(torch.randn_like(module.bias) * 0.1)
    input_ids = torch.randint(0, 32, (2, 8))
    return model, input_ids


def _action_plan(pattern: str, *, method: str = "smoothquant", args=None) -> ExecutionPlan:
    return ExecutionPlan(
        version=2,
        stages=(
            Stage(
                name="balance",
                rules=(
                    Rule(
                        match=pattern,
                        actions=(OperationSpec(method=method, args=dict(args or {"alpha": 0.5})),),
                    ),
                ),
            ),
        ),
    )


def _compilable_plan(method: str) -> ExecutionPlan:
    return ExecutionPlan(
        version=2,
        stages=(
            Stage(
                name="ptq",
                rules=(
                    Rule(
                        match="*",
                        actions=(OperationSpec(method=method),),
                        weight=TargetSpec(quantize=QuantizeSpec(method="gptq", args={"bits": 4})),
                    ),
                ),
            ),
        ),
    )


def test_compile_accepts_registered_action():
    assert known_action_methods() == ("smoothquant",)
    compiled = compile_rule_actions(
        (OperationSpec(method="smoothquant", args={"alpha": 0.5}),)
    )
    assert [action.method for action in compiled] == ["smoothquant"]

    config = compile_plan_to_quantize_config(_compilable_plan("smoothquant"))
    assert isinstance(config, GPTQConfig)
    assert config.bits == 4


def test_yaml_action_compiles():
    config = compile_protocol_yaml_to_quantize_config(
        """
version: 2
stages:
  - name: ptq
    rules:
      - match: "*"
        actions:
          - method: smoothquant
            alpha: 0.5
        weight:
          quantize:
            method: gptq
            bits: 4
"""
    )
    assert isinstance(config, GPTQConfig)
    assert config.bits == 4


@pytest.mark.parametrize("method", ["awq_balance", "calibrate_router", "does_not_exist"])
def test_unsupported_action_raises_and_lists_known_methods(method):
    with pytest.raises(NotImplementedError) as excinfo:
        compile_plan_to_quantize_config(_compilable_plan(method))

    message = str(excinfo.value)
    assert method in message
    assert "smoothquant" in message


def test_smoothquant_preserves_logits_and_moves_the_folded_pair():
    model, input_ids = _tiny_model()
    block = model.layers[0]
    with torch.no_grad():
        logits_before = model(input_ids).clone()

    q_before = block.q_proj.weight.detach().clone()
    k_before = block.k_proj.weight.detach().clone()
    v_before = block.v_proj.weight.detach().clone()
    o_before = block.o_proj.weight.detach().clone()
    norm_weight_before = block.input_layernorm.weight.detach().clone()
    norm_bias_before = block.input_layernorm.bias.detach().clone()

    apply_plan_actions(model, _action_plan(r".*\.q_proj$"), calibration=lambda: model(input_ids))

    with torch.no_grad():
        logits_after = model(input_ids)
    torch.testing.assert_close(logits_after, logits_before, rtol=1e-5, atol=1e-6)

    # The matched linear and its preceding norm both moved.
    assert not torch.equal(block.q_proj.weight, q_before)
    assert not torch.equal(block.input_layernorm.weight, norm_weight_before)
    assert not torch.equal(block.input_layernorm.bias, norm_bias_before)

    # k_proj and v_proj read the same norm output, so the fold must scale them
    # too; otherwise dividing the norm would change their outputs.
    assert not torch.equal(block.k_proj.weight, k_before)
    assert not torch.equal(block.v_proj.weight, v_before)

    # o_proj does not consume the norm output and must be untouched.
    assert torch.equal(block.o_proj.weight, o_before)


def test_smoothquant_does_not_change_the_quantized_block_output():
    model, input_ids = _tiny_model()
    block = model.layers[0]
    with torch.no_grad():
        hidden = block.input_layernorm(model.embed(input_ids))
        before = block.q_proj(hidden).clone()

    apply_plan_actions(model, _action_plan(r".*\.q_proj$"), calibration=lambda: model(input_ids))

    with torch.no_grad():
        hidden_after = block.input_layernorm(model.embed(input_ids))
        after = block.q_proj(hidden_after)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)


def test_smoothquant_unresolvable_match_raises():
    model, input_ids = _tiny_model()
    with pytest.raises(ProtocolActionError) as excinfo:
        apply_plan_actions(model, _action_plan(r".*\.o_proj$"), calibration=lambda: model(input_ids))

    message = str(excinfo.value)
    assert "layers.0.o_proj" in message
    assert "no norm output feeds it" in message


def test_smoothquant_requires_calibration():
    model, _ = _tiny_model()
    with pytest.raises(ProtocolActionError, match="requires calibration activations"):
        apply_plan_actions(model, _action_plan(r".*\.q_proj$"))


def test_smoothquant_rejects_unknown_arguments():
    model, input_ids = _tiny_model()
    with pytest.raises(ProtocolActionError, match="only `alpha` is supported"):
        apply_plan_actions(
            model,
            _action_plan(r".*\.q_proj$", args={"alpha": 0.5, "ratio": 0.7}),
            calibration=lambda: model(input_ids),
        )


def test_action_rule_that_matches_nothing_raises():
    model, input_ids = _tiny_model()
    with pytest.raises(ProtocolActionError, match="matched no modules"):
        apply_plan_actions(model, _action_plan(r".*does_not_exist$"), calibration=lambda: model(input_ids))
