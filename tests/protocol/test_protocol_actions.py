# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gptqmodel.quantization import GPTQConfig
from gptqmodel.quantization import protocol as protocol_module
from gptqmodel.quantization.protocol import (
    ExecutionPlan,
    OperationSpec,
    ProtocolActionError,
    QuantizeSpec,
    Rule,
    Stage,
    TargetSpec,
    apply_plan_actions,
    compile_and_apply_plan,
    compile_plan_to_quantize_config,
    compile_protocol_yaml_to_quantize_config,
    compile_rule_actions,
    known_action_methods,
    smoothquant,
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
    def __init__(self, vocab: int = 32, hidden: int = 16, layers: int = 1, tied: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_TinyBlock(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


class _ResidualReuseBlock(nn.Module):
    """The norm output is also added back functionally, with no module boundary."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        hidden = self.norm(x)
        # `hidden` is consumed by `proj` and also directly by the residual add.
        return self.o_proj(torch.cat((x + hidden, self.proj(hidden)), dim=-1))


class _MoEBlock(nn.Module):
    """The router is a weighted consumer; the experts read the norm output directly."""

    def __init__(self, hidden: int = 16, inter: int = 32, experts: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.up = nn.Parameter(torch.randn(experts, hidden, inter) * 0.2)
        self.down = nn.Parameter(torch.randn(experts, inter, hidden) * 0.2)

    def forward(self, x):
        hidden = self.norm(x)
        probs = self.gate(hidden).softmax(dim=-1)
        out = torch.zeros_like(hidden)
        for index in range(self.up.shape[0]):
            gate = probs[..., index].unsqueeze(-1)
            out = out + gate * ((hidden @ self.up[index]).relu() @ self.down[index])
        return x + out


class _MaskedModel(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden * 2, hidden)

    def forward(self, x, attention_mask=None):
        hidden = self.norm(x)
        return self.proj(torch.cat((x, hidden), dim=-1))


def _randomize_layer_norm_bias(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) and module.bias is not None:
                module.bias.copy_(torch.randn_like(module.bias) * 0.1)


def _tiny_model(*, layers: int = 1, tied: bool = False) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = _TinyModel(layers=layers, tied=tied).eval()
    _randomize_layer_norm_bias(model)
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


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _assert_unchanged(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, snapshot[name]), f"{name} changed"


def test_compile_accepts_registered_action():
    assert known_action_methods() == ("smoothquant",)
    compiled = compile_rule_actions((smoothquant(0.5),))
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


def test_compiler_rejects_invalid_action_arguments_instead_of_dropping_them():
    bad = Rule(
        match="*",
        actions=(OperationSpec(method="smoothquant", args={"alpha": 2.0}),),
        weight=TargetSpec(quantize=QuantizeSpec(method="gptq", args={"bits": 4})),
    )
    plan = ExecutionPlan(version=2, stages=(Stage(name="ptq", rules=(bad,)),))
    with pytest.raises(ProtocolActionError, match=r"must be in \[0, 1\]"):
        compile_plan_to_quantize_config(plan)


def test_resolution_is_per_layer_on_a_multi_layer_model():
    model, input_ids = _tiny_model(layers=6)
    capture = protocol_module._capture_smoothquant_statistics(model, lambda: model(input_ids))

    for layer in range(6):
        assert capture.predecessors[f"layers.{layer}.q_proj"] == {f"layers.{layer}.input_layernorm"}


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

    report = apply_plan_actions(model, _action_plan(r".*\.q_proj$"), calibration=lambda: model(input_ids))

    with torch.no_grad():
        logits_after = model(input_ids)
    torch.testing.assert_close(logits_after, logits_before, rtol=1e-5, atol=1e-6)
    assert report.applied == ["balance:[.*\\.q_proj$]:smoothquant"]
    assert report.skipped == []

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
    assert "no gain norm output feeds it" in message


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


def test_container_only_match_raises_with_a_useful_message():
    model, input_ids = _tiny_model()
    with pytest.raises(ProtocolActionError, match="matched no module with a 2-D weight"):
        apply_plan_actions(model, _action_plan(r"^layers$"), calibration=lambda: model(input_ids))


def test_broad_match_skips_containers_and_still_folds():
    model, input_ids = _tiny_model()
    with torch.no_grad():
        before = model(input_ids).clone()

    report = apply_plan_actions(model, _action_plan(r".*"), calibration=lambda: model(input_ids))

    with torch.no_grad():
        after = model(input_ids)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    assert report.applied == ["balance:[.*]:smoothquant"]
    assert any(reason.endswith("has no 2-D weight") for _name, reason in report.skipped)


def test_tied_head_is_refused_before_any_parameter_moves():
    model, input_ids = _tiny_model(tied=True)
    snapshot = _snapshot(model)

    with pytest.raises(ProtocolActionError, match="its weight is shared with"):
        apply_plan_actions(model, _action_plan(r"^lm_head$"), calibration=lambda: model(input_ids))

    _assert_unchanged(model, snapshot)


def test_functional_residual_consumer_is_detected_and_rolled_back():
    torch.manual_seed(0)
    block = _ResidualReuseBlock().eval()
    _randomize_layer_norm_bias(block)
    x = torch.randn(2, 8, 16)
    snapshot = _snapshot(block)

    plan = ExecutionPlan(
        version=2,
        stages=(Stage(name="balance", rules=(Rule(match=r"^proj$", actions=(smoothquant(0.5),)),)),),
    )
    with pytest.raises(ProtocolActionError) as excinfo:
        apply_plan_actions(block, plan, calibration=lambda: block(x))

    message = str(excinfo.value)
    assert "the model output changed" in message
    assert "the fold was rolled back" in message
    _assert_unchanged(block, snapshot)


def test_moe_experts_that_read_the_norm_output_directly_are_detected_and_rolled_back():
    torch.manual_seed(0)
    block = _MoEBlock().eval()
    _randomize_layer_norm_bias(block)
    x = torch.randn(2, 8, 16)
    snapshot = _snapshot(block)

    plan = ExecutionPlan(
        version=2,
        stages=(Stage(name="balance", rules=(Rule(match=r"^gate$", actions=(smoothquant(0.5),)),)),),
    )
    with pytest.raises(ProtocolActionError) as excinfo:
        apply_plan_actions(block, plan, calibration=lambda: block(x))

    assert "the model output changed" in str(excinfo.value)
    _assert_unchanged(block, snapshot)


def test_multi_stage_plan_is_atomic_when_a_later_action_is_unknown():
    model, input_ids = _tiny_model()
    snapshot = _snapshot(model)
    plan = ExecutionPlan(
        version=2,
        stages=(
            Stage(name="s1", rules=(Rule(match=r".*\.q_proj$", actions=(smoothquant(0.5),)),)),
            Stage(name="s2", rules=(Rule(match=r".*\.q_proj$", actions=(OperationSpec(method="awq_balance"),)),)),
        ),
    )

    with pytest.raises(NotImplementedError, match="awq_balance"):
        apply_plan_actions(model, plan, calibration=lambda: model(input_ids))

    _assert_unchanged(model, snapshot)


def test_multi_stage_plan_is_atomic_when_a_later_match_is_empty():
    model, input_ids = _tiny_model()
    snapshot = _snapshot(model)
    plan = ExecutionPlan(
        version=2,
        stages=(
            Stage(name="s1", rules=(Rule(match=r".*\.q_proj$", actions=(smoothquant(0.5),)),)),
            Stage(name="s2", rules=(Rule(match=r".*does_not_exist$", actions=(smoothquant(0.5),)),)),
        ),
    )

    with pytest.raises(ProtocolActionError, match="matched no modules"):
        apply_plan_actions(model, plan, calibration=lambda: model(input_ids))

    _assert_unchanged(model, snapshot)


def test_documented_flow_runs_end_to_end_from_one_plan():
    model, input_ids = _tiny_model()
    with torch.no_grad():
        before = model(input_ids).clone()

    plan = ExecutionPlan(
        version=2,
        stages=(
            Stage(name="balance", rules=(Rule(match=r".*\.q_proj$", actions=(smoothquant(0.5),)),)),
            Stage(
                name="ptq",
                rules=(
                    Rule(
                        match="*",
                        weight=TargetSpec(quantize=QuantizeSpec(method="gptq", args={"bits": 4})),
                    ),
                ),
            ),
        ),
    )

    config, report = compile_and_apply_plan(model, plan, calibration=lambda: model(input_ids))

    assert isinstance(config, GPTQConfig)
    assert config.bits == 4
    assert report.applied == ["balance:[.*\\.q_proj$]:smoothquant"]
    with torch.no_grad():
        after = model(input_ids)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)


def test_broad_documented_plan_applies_without_matching_a_container():
    model, input_ids = _tiny_model()
    plan = ExecutionPlan(
        version=2,
        stages=(
            Stage(name="ptq", rules=(Rule(match="*", actions=(smoothquant(0.5),)),)),
            Stage(
                name="ptq",
                rules=(
                    Rule(
                        match="*",
                        weight=TargetSpec(quantize=QuantizeSpec(method="gptq", args={"bits": 4})),
                    ),
                ),
            ),
        ),
    )

    config, report = compile_and_apply_plan(model, plan, calibration=lambda: model(input_ids))

    assert isinstance(config, GPTQConfig)
    assert report.applied
    assert report.skipped


def test_activation_statistics_ignore_padded_positions():
    torch.manual_seed(0)
    model = _MaskedModel().eval()
    _randomize_layer_norm_bias(model)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])

    # Two batches that differ only in the padded positions must produce the
    # same per-channel maxima when a mask is supplied.
    first = torch.randn(2, 4, 16)
    second = first.clone()
    second[1, 1:] = 1e3

    masked_a = protocol_module._capture_smoothquant_statistics(
        model, lambda: model(first, attention_mask=mask)
    )
    masked_b = protocol_module._capture_smoothquant_statistics(
        model, lambda: model(second, attention_mask=mask)
    )
    unmasked_b = protocol_module._capture_smoothquant_statistics(model, lambda: model(second))

    torch.testing.assert_close(masked_a.x_max["proj"], masked_b.x_max["proj"])
    assert not torch.equal(masked_b.x_max["proj"], unmasked_b.x_max["proj"])


def test_activation_statistics_use_each_batch_attention_mask():
    """Batch 2..N must be masked by their own padding, not batch 1's (#4559)."""

    torch.manual_seed(1)
    model = _MaskedModel().eval()
    _randomize_layer_norm_bias(model)
    first = torch.randn(2, 4, 16)
    second = torch.randn(2, 4, 16)
    # Positions padded by one batch and real in the other, so a stale mask
    # provably over- or under-counts the channel maximum.
    second[0, 1:] = 1e3
    second[1, 1:] = -1e3
    mask_a = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    mask_b = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])

    def single(x, mask):
        return protocol_module._capture_smoothquant_statistics(
            model, lambda: model(x, attention_mask=mask)
        ).x_max["proj"]

    expected = torch.maximum(single(first, mask_a), single(second, mask_b))

    forward = protocol_module._capture_smoothquant_statistics(
        model,
        lambda: (model(first, attention_mask=mask_a), model(second, attention_mask=mask_b)),
    ).x_max["proj"]
    reverse = protocol_module._capture_smoothquant_statistics(
        model,
        lambda: (model(second, attention_mask=mask_b), model(first, attention_mask=mask_a)),
    ).x_max["proj"]

    torch.testing.assert_close(forward, expected)
    torch.testing.assert_close(reverse, expected)


def test_activation_statistics_clear_a_mask_from_a_different_batch_shape():
    """A later batch with another sequence length must not fall back to unmasked."""

    torch.manual_seed(2)
    model = _MaskedModel().eval()
    _randomize_layer_norm_bias(model)
    first = torch.randn(2, 4, 16)
    second = torch.randn(2, 6, 16)
    second[0, 2:] = 1e3
    mask_a = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    mask_b = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])

    def single(x, mask):
        return protocol_module._capture_smoothquant_statistics(
            model, lambda: model(x, attention_mask=mask)
        ).x_max["proj"]

    expected = torch.maximum(single(first, mask_a), single(second, mask_b))
    combined = protocol_module._capture_smoothquant_statistics(
        model,
        lambda: (model(first, attention_mask=mask_a), model(second, attention_mask=mask_b)),
    ).x_max["proj"]

    torch.testing.assert_close(combined, expected)


def test_non_finite_activation_statistics_are_rejected_before_any_mutation():
    model, input_ids = _tiny_model()
    block = model.layers[0]
    with torch.no_grad():
        block.q_proj.weight[0, 0] = float("inf")
    snapshot = _snapshot(model)

    with pytest.raises(ProtocolActionError, match="non-finite"):
        apply_plan_actions(model, _action_plan(r".*\.q_proj$"), calibration=lambda: model(input_ids))

    _assert_unchanged(model, snapshot)
