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
from transformers import Gemma2Config, Gemma2ForCausalLM, Lfm3Config, Lfm3ForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.definitions.gemma2 import Gemma2QModel
from gptqmodel.models.definitions.lfm3 import LFM3QModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.quantization.awq.quantize.scale import (
    UnresolvableGainOffset,
    apply_scale,
    scale_ln_fcs,
)
from gptqmodel.quantization.awq.utils.module import get_op_name


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
        effective = scale_ln_fcs(norm, [linear], scales)
        after = linear(norm(x))

    assert _relative_error(after, before) < 1e-6, name

    # Closed-form oracle for the stored weights, independent of the fold code.
    # Offset-gain norms return the gauge-normalised scale that was folded.
    expected_norm = (weight_before + offset) / effective - offset
    torch.testing.assert_close(norm.weight.detach(), expected_norm, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        linear.weight.detach(),
        linear_before * effective.view(1, -1),
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
@pytest.mark.parametrize(
    "scale_peak",
    # Issue #4074 item 1: Gemma-3 MLP scales reach ~1e3. The original coverage
    # only used [0.5, 1.5], where `1 + w'` is still representable in bf16 and
    # the cancellation that shipped is invisible.
    [1.5, 100.0, 1000.0],
)
def test_scale_ln_fcs_offset_gain_folds_in_float32(dtype, tolerance, scale_peak):
    torch.manual_seed(0)
    dim = 32
    norm = Gemma3RMSNorm(dim).to(dtype)
    with torch.no_grad():
        norm.weight.copy_((torch.randn(dim, dtype=torch.float32) * 0.1).to(dtype))
    linear = nn.Linear(dim, dim, bias=False).to(dtype)

    scales = (torch.rand(dim, dtype=torch.float32) + 0.5) * scale_peak
    x = torch.randn(8, dim).to(dtype)
    weight_before = norm.weight.detach().clone()
    with torch.no_grad():
        before = linear(norm(x)).float()
        effective = scale_ln_fcs(norm, [linear], scales)
        after = linear(norm(x)).float()

    assert _relative_error(after, before) < tolerance

    # A uniform factor on the per-channel scale is a free gauge for per-group
    # symmetric quantisation, so an offset-gain fold must renormalise by its
    # peak and keep `max(s) <= 1`. Otherwise `(1 + w) / s - 1` approaches -1
    # and `1 + w'` is not representable in bf16 at Gemma-3 MLP scales.
    expected_effective = scales / scales.max() if float(scales.max()) > 1.0 else scales
    torch.testing.assert_close(effective, expected_effective, rtol=0.0, atol=0.0)
    assert float(effective.max()) <= 1.0 + 1e-6

    # The intermediate `1 + w` and the division are performed in float32; only
    # the final stored gain is rounded back to the serving dtype.
    expected = ((weight_before.float() + 1.0) / effective.float() - 1.0).to(dtype)
    torch.testing.assert_close(norm.weight.detach(), expected, rtol=0.0, atol=0.0)


def test_apply_scale_uses_the_gauge_for_captured_features_and_weights():
    """The fold and the captured input must share the same effective scale.

    Issue #4074 item 1: if only the stored gain is renormalised, the captured
    features are divided by the original scale while the weights were folded
    with the gauge-normalised one, so the quantization input no longer matches
    the block the scale search scored.
    """

    torch.manual_seed(0)
    dim = 32

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Gemma3RMSNorm(dim).to(torch.bfloat16)
            self.linear = nn.Linear(dim, dim, bias=False).to(torch.bfloat16)

        def forward(self, hidden_states):
            return self.linear(self.norm(hidden_states))

    block = _Block()
    with torch.no_grad():
        block.norm.weight.copy_((torch.randn(dim) * 0.1).to(torch.bfloat16))

    x = torch.randn(8, dim).to(torch.bfloat16)
    scales = (torch.rand(dim, dtype=torch.float32) + 0.5) * 1000.0
    effective = scales / scales.max()
    features = {"linear": x.clone()}
    linear_before = block.linear.weight.detach().clone()
    with torch.no_grad():
        before = block(x).float()
        apply_scale(
            block,
            [("norm", ["linear"], scales, None)],
            input_feat_dict=features,
        )
        after = block(x).float()

    assert _relative_error(after, before) < 1e-2
    torch.testing.assert_close(
        block.linear.weight.detach().float(),
        linear_before.float() * effective.view(1, -1),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        features["linear"].float(),
        x.float() / effective.view(1, -1),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    "name,norm_factory,offset",
    [case for case in _norm_cases() if case[2] == 1.0],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_offset_gain_fold_is_neutral_at_gemma3_mlp_scale(name, norm_factory, offset):
    """Every offset-gain norm family must survive the Gemma-3 MLP scale.

    Gemma-3 is the measured case; Gemma, Gemma 2, Qwen3-Next and Qwen3.5 share
    the `1 + weight` fold and are covered by the same gauge.
    """
    torch.manual_seed(0)
    dim = 32
    norm = norm_factory(dim).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_((torch.randn(dim, dtype=torch.float32) * 0.1).to(torch.bfloat16))
    linear = nn.Linear(dim, dim, bias=False).to(torch.bfloat16)
    scales = (torch.rand(dim, dtype=torch.float32) + 0.5) * 1000.0
    x = torch.randn(8, dim).to(torch.bfloat16)

    with torch.no_grad():
        before = linear(norm(x)).float()
        effective = scale_ln_fcs(norm, [linear], scales)
        after = linear(norm(x)).float()

    assert _relative_error(after, before) < 1e-2, name
    assert float(effective.max()) <= 1.0 + 1e-6


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


class _Lfm3Conv(nn.Module):
    """The depthwise-conv block LFM3 executes before the residual add."""

    def __init__(self, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states):
        return self.out_proj(torch.nn.functional.silu(self.in_proj(hidden_states)))


class _Lfm3FeedForward(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states):
        return self.w2(
            torch.nn.functional.silu(self.w1(hidden_states)) * self.w3(hidden_states)
        )


class _Lfm3DecoderLayer(nn.Module):
    """Mirror of `Lfm3DecoderLayer` with `use_sandwich_norm=True`.

    `post_operator_norm` / `post_ffn_norm` run on the block output after the
    residual add, so their output never feeds `conv.in_proj` or
    `feed_forward.w1` even though the module tree lists them first.
    """

    def __init__(self, hidden: int = 32, intermediate: int = 64):
        super().__init__()
        self.operator_norm = nn.RMSNorm(hidden)
        self.post_operator_norm = nn.RMSNorm(hidden)
        self.conv = _Lfm3Conv(hidden)
        self.ffn_norm = nn.RMSNorm(hidden)
        self.post_ffn_norm = nn.RMSNorm(hidden)
        self.feed_forward = _Lfm3FeedForward(hidden, intermediate)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.post_operator_norm(self.conv(self.operator_norm(hidden_states)))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_ffn_norm(self.feed_forward(self.ffn_norm(hidden_states)))
        return residual + hidden_states


def _bare_lfm3_qmodel():
    qmodel = LFM3QModel.__new__(LFM3QModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = types.SimpleNamespace(
        config=types.SimpleNamespace(num_experts=0),
    )
    return qmodel


def _lfm3_features(hidden: int = 32, intermediate: int = 64):
    return {
        "conv.in_proj": torch.randn(4, hidden),
        "conv.out_proj": torch.randn(4, hidden),
        "feed_forward.w1": torch.randn(4, hidden),
        "feed_forward.w3": torch.randn(4, hidden),
        "feed_forward.w2": torch.randn(4, intermediate),
    }


def test_lfm3_post_norms_are_declared_non_sources():
    assert LFM3QModel.get_post_norm_modules() == ["post_operator_norm", "post_ffn_norm"]

    from gptqmodel.models.definitions.lfm3_omni import LFM3OmniQModel
    from gptqmodel.models.definitions.lfm3_vl import LFM3VLQModel

    for definition in (LFM3VLQModel, LFM3OmniQModel):
        assert definition.get_post_norm_modules() == [
            "post_operator_norm",
            "post_ffn_norm",
        ], definition.__name__


def test_lfm3_sandwich_post_norms_are_not_fold_sources():
    torch.manual_seed(0)
    layer = _Lfm3DecoderLayer()
    qmodel = _bare_lfm3_qmodel()

    nodes = qmodel.awq_get_modules_for_scaling(layer, _lfm3_features(), {})
    conv = next(node for node in nodes if any(module is layer.conv.in_proj for module in node["layers"]))
    ffn = next(node for node in nodes if any(module is layer.feed_forward.w1 for module in node["layers"]))

    assert conv["prev_op"] is layer.operator_norm
    assert conv["prev_op"] is not layer.post_operator_norm
    assert ffn["prev_op"] is layer.ffn_norm
    assert ffn["prev_op"] is not layer.post_ffn_norm


def test_lfm3_sandwich_norm_fold_is_output_neutral():
    torch.manual_seed(0)
    layer = _Lfm3DecoderLayer()
    qmodel = _bare_lfm3_qmodel()
    nodes = qmodel.awq_get_modules_for_scaling(layer, _lfm3_features(), {})
    conv = next(node for node in nodes if any(module is layer.conv.in_proj for module in node["layers"]))
    ffn = next(node for node in nodes if any(module is layer.feed_forward.w1 for module in node["layers"]))

    x = torch.randn(4, 32)
    with torch.no_grad():
        before = layer(x)
        for node in (conv, ffn):
            scales = torch.rand(32) + 0.5
            scale_ln_fcs(node["prev_op"], node["layers"], scales)
        after = layer(x)

    # Folding into a post-norm changes the block output by ~0.57 on this
    # fixture; the data-flow source keeps it at floating-point noise.
    assert _relative_error(after, before) < 1e-5


def _tiny_lfm3_attention_model():
    torch.manual_seed(0)
    config = Lfm3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        use_attention_gate=True,
        use_sandwich_norm=True,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=0,
    )
    config._attn_implementation = "eager"
    model = Lfm3ForCausalLM(config).eval()
    layer = model.model.layers[0]

    features = {}

    def make_hook(name):
        def hook(_module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states")
            if isinstance(value, torch.Tensor):
                features[name] = value.detach().clone()

        return hook

    handles = []
    for name in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "feed_forward.w1",
        "feed_forward.w3",
        "feed_forward.w2",
    ):
        handles.append(
            layer.get_submodule(name).register_forward_pre_hook(
                make_hook(name), with_kwargs=True
            )
        )

    input_ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        model(input_ids)
    for handle in handles:
        handle.remove()
    return model, layer, input_ids, features


def test_lfm3_attention_fold_targets_qkv_and_compensates_sigmoid_gate():
    """`operator_norm` feeds Q/K/V and the sigmoid gate, never `o_proj`.

    The legacy tree named `gate_proj`, so `o_proj` was folded into the attention
    input norm while the live `g_proj` gate was left uncompensated.
    """

    model, layer, input_ids, features = _tiny_lfm3_attention_model()
    qmodel = _bare_lfm3_qmodel()
    qmodel.__dict__["model"] = model
    qmodel.__dict__["effective_module_tree"] = LFM3QModel.module_tree
    qmodel.__dict__["module_tree"] = LFM3QModel.module_tree

    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    node = next(node for node in nodes if node["prev_op"] is layer.operator_norm)
    names = [get_op_name(layer, module) for module in node["layers"]]

    assert names == [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ]
    assert node["_scale_consumers"] == ["self_attn.g_proj"]
    for candidate in nodes:
        assert "self_attn.o_proj" not in [
            get_op_name(layer, module) for module in candidate["layers"]
        ]

    AWQProcessor._validate_scale_group(
        layer,
        node["prev_op"],
        node["layers"],
        node["inp"],
        node["module2inspect"],
        node["_scale_consumers"],
    )

    scales = torch.linspace(0.3, 2.0, 32)
    with torch.no_grad():
        before = model(input_ids).logits.clone()
    o_proj_before = layer.self_attn.o_proj.weight.detach().clone()
    gate_before = layer.self_attn.g_proj.weight.detach().clone()

    apply_scale(
        layer,
        [("operator_norm", names, scales, None)],
        consumer_map={"operator_norm": node["_scale_consumers"]},
    )

    with torch.no_grad():
        after = model(input_ids).logits

    assert _relative_error(after, before) < 1e-5
    # `o_proj` is quantized plainly, so its weights stay untouched; the gate is
    # a declared consumer and must absorb the inverse scale.
    assert torch.equal(layer.self_attn.o_proj.weight, o_proj_before)
    assert not torch.equal(layer.self_attn.g_proj.weight, gate_before)


def _scaled_before_norm_class():
    """A norm that scales before normalizing; its gain is not foldable."""

    try:
        from transformers.models.muse_spark.modeling_muse_spark import (
            MuseSparkTextInputScaledRMSNorm,
        )

        return MuseSparkTextInputScaledRMSNorm
    except Exception:  # pragma: no cover - older transformers builds
        class _ScaledBeforeNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(dim))
                self.eps = eps

            def forward(self, hidden_states):
                scaled = hidden_states.float() * (1.0 + self.weight).float()
                normed = scaled * torch.rsqrt(
                    scaled.pow(2).mean(-1, keepdim=True) + self.eps
                )
                return normed.type_as(hidden_states)

        return _ScaledBeforeNorm


def test_probe_failure_skips_scale_instead_of_guessing_from_source():
    torch.manual_seed(0)
    dim = 32
    norm_cls = _scaled_before_norm_class()
    norm = norm_cls(dim)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim) * 0.1)
    linear = nn.Linear(dim, dim, bias=False)
    scales = torch.rand(dim) + 0.5

    # The old source-text fallback read `1.0 + self.weight` here and folded as
    # an offset gain, changing the linear output by ~0.11.
    with pytest.raises(UnresolvableGainOffset):
        scale_ln_fcs(norm, [linear], scales)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = norm_cls(dim)
            self.linear = nn.Linear(dim, dim, bias=False)

        def forward(self, hidden_states):
            return self.linear(self.norm(hidden_states))

    block = _Block()
    with torch.no_grad():
        block.norm.weight.copy_(torch.randn(dim) * 0.1)
    norm_before = block.norm.weight.detach().clone()
    linear_before = block.linear.weight.detach().clone()
    x = torch.randn(8, dim)
    features = {"linear": x.clone()}
    with torch.no_grad():
        before = block(x)
        apply_scale(
            block,
            [("norm", ["linear"], scales, None)],
            input_feat_dict=features,
        )
        after = block(x)

    assert torch.equal(block.norm.weight, norm_before)
    assert torch.equal(block.linear.weight, linear_before)
    assert torch.equal(features["linear"], x)
    assert torch.equal(after, before)
