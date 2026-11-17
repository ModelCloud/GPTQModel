# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Hymba AWQ regressions (#4891).

`nvidia/Hymba-1.5B-Base` needs remote code and CUDA-only mamba kernels, and the
model class is not part of this transformers release, so the cases here are
synthesised with the exact module names `HymbaQModel.module_tree` declares:

* `input_layernorm` -> `mamba` -> `pre_moe_layernorm` -> `moe`,
* a `HymbaBlock`-style `forward(self, hidden_states, past_key_value=None, **kwargs)`
  that reads `attention_mask`, `position_ids` and `kv_last_layer` out of the
  catch-all.
"""

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.utils.module import get_op_name
from tests.test_awq_scaling_groups import (
    _features,
    _foldable_groups,
    _worst_relative_change,
)


HIDDEN = 32
INNER = 24
VOCAB = 64


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INNER, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INNER, bias=False)
        self.down_proj = nn.Linear(INNER, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, 2, bias=False)
        self.experts = nn.ModuleList([_Expert()])

    def forward(self, x):
        probs = torch.softmax(self.gate(x), dim=-1)
        return self.experts[0](x) * probs[..., :1]


class _Mamba(nn.Module):
    """Mixer with the same dense leaves as `HymbaMixer`."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(HIDDEN, 2 * INNER, bias=False)
        self.conv1d = nn.Conv1d(
            2 * INNER, 2 * INNER, kernel_size=2, padding=1, groups=2 * INNER
        )
        self.x_proj = nn.ModuleList([nn.Linear(INNER, 4, bias=False)])
        self.dt_proj = nn.ModuleList([nn.Linear(4, INNER, bias=False)])
        self.out_proj = nn.Linear(INNER, HIDDEN, bias=False)

    def forward(self, x):
        length = x.shape[1]
        z = self.in_proj(x).transpose(1, 2)
        z = self.conv1d(z)[..., :length].transpose(1, 2)
        gate, up = z.chunk(2, dim=-1)
        up = self.dt_proj[0](torch.tanh(self.x_proj[0](up)))
        return self.out_proj(nn.functional.silu(gate) * up)


class _HymbaBlock(nn.Module):
    """`HymbaDecoderLayer` shape: a `**kwargs`-only mixer handoff."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(HIDDEN)
        self.mamba = _Mamba()
        self.pre_moe_layernorm = nn.LayerNorm(HIDDEN)
        self.moe = _Moe()

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        # The real block reads these out of `**kwargs`; a name-based filter that
        # drops them turns this into a `KeyError`. Validate the whole required
        # set explicitly instead of leaving the reads as unused lookups.
        required = ("attention_mask", "kv_last_layer", "position_ids")
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise KeyError(missing[0])
        attention_mask = kwargs["attention_mask"]
        residual = hidden_states + self.mamba(self.input_layernorm(hidden_states))
        hidden_states = residual + self.moe(self.pre_moe_layernorm(residual))
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[..., :1]
        return hidden_states, past_key_value


class _HymbaInner(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_HymbaBlock() for _ in range(layers)])
        self.final_layernorm = nn.LayerNorm(HIDDEN)
        # Bare parameter on the decoder root, exactly like Hymba's meta tokens.
        self.memory_tokens = nn.Parameter(torch.zeros(4, HIDDEN))


class _HymbaForCausalLM(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.model = _HymbaInner(layers=layers)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = None

    def forward(self, input_ids):
        hidden = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
        for layer in self.model.layers:
            hidden = layer(
                hidden,
                past_key_value=None,
                attention_mask=None,
                position_ids=position_ids,
                kv_last_layer=None,
                use_swa=False,
            )[0]
        return _Out(self.lm_head(self.model.final_layernorm(hidden)))


def _synthetic_hymba():
    """A tiny Hymba-shaped model plus the definition bound to it."""

    torch.manual_seed(0)
    hf_model = _HymbaForCausalLM().eval()
    qmodel = HymbaQModel.__new__(HymbaQModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    tree = HymbaQModel._resolve_effective_module_tree(hf_model, None)
    qmodel.__dict__["effective_module_tree"] = tree
    qmodel.__dict__["module_tree"] = tree
    return hf_model, qmodel


class _VarKwargsModule(nn.Module):
    def forward(self, hidden_states, past_key_value=None, **kwargs):
        return hidden_states, past_key_value


class _NamedKwargsModule(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states


def _processor():
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace()
    processor._get_root_rotary = lambda: None
    return processor


# --------------------------------------------------------------------------- #
# #4891 item 1: `_sanitize_kwargs` must not empty a `**kwargs` forward.
# --------------------------------------------------------------------------- #


def test_sanitize_kwargs_keeps_every_key_for_a_var_keyword_forward():
    processor = _processor()
    mask = torch.ones(1, 4)
    kwargs = {
        "attention_mask": mask,
        "position_ids": torch.arange(4).unsqueeze(0),
        "kv_last_layer": None,
        "use_swa": True,
        "_awq_rotary_position_ids": torch.arange(4).unsqueeze(0),
    }

    sanitized = processor._sanitize_kwargs(kwargs, _VarKwargsModule())

    assert sanitized == kwargs
    # Processor bookkeeping must not reach the module's catch-all.
    internal = processor._sanitize_kwargs(
        {**kwargs, "_awq_feature_kwargs": {"q_proj": {}}}, _VarKwargsModule()
    )
    assert internal == kwargs


def test_sanitize_kwargs_still_filters_a_named_signature():
    processor = _processor()
    mask = torch.ones(1, 4)

    sanitized = processor._sanitize_kwargs(
        {"attention_mask": mask, "kv_last_layer": None}, _NamedKwargsModule()
    )

    assert sanitized == {"attention_mask": mask}


def test_prepare_module_forward_promotes_a_var_keyword_block_input():
    processor = _processor()
    module = _VarKwargsModule()
    module.add_module("proj", nn.Linear(4, 4))
    x = torch.randn(6, 4)
    position_ids = torch.arange(6).unsqueeze(0)

    prepared_x, prepared_kwargs = processor._prepare_module_forward(
        x, module, {"attention_mask": torch.ones(1, 6), "position_ids": position_ids}
    )

    # `HymbaBlock` packs `[batch, seq, hidden]`; a 2-D replay input crashes it.
    assert prepared_x.shape == (1, 6, 4)
    assert prepared_kwargs["attention_mask"].shape == (1, 6)
    assert torch.equal(prepared_kwargs["position_ids"], position_ids)


# --------------------------------------------------------------------------- #
# #4891 item 2: the tree must name the norm the layer actually owns.
# --------------------------------------------------------------------------- #


def test_hymba_tree_declares_the_norm_the_layer_actually_owns():
    plan = ModulePlan.from_tree(HymbaQModel.module_tree)

    assert plan.find("post_attention_layernorm") is None, (
        "HymbaDecoderLayer has no post_attention_layernorm; declaring it left the "
        "routed-expert gate/up block without a producer"
    )
    assert plan.find("pre_moe_layernorm") is not None

    layer = _HymbaBlock()
    missing = []
    for spec in plan.specs:
        if "{expert_index}" in spec.template:
            continue
        try:
            layer.get_submodule(spec.template)
        except AttributeError:
            missing.append(spec.template)
    assert not missing, f"Hymba tree declares modules the layer does not own: {missing}"


def test_hymba_mixer_declares_its_dense_leaves_and_no_fold_source():
    plan = ModulePlan.from_tree(HymbaQModel.module_tree)

    in_proj = plan.find("mamba.in_proj")
    assert in_proj is not None
    assert "nosrc" in in_proj.flags, (
        "the conv/SSM path between in_proj and out_proj is not homogeneous, so "
        "in_proj must never be a fold source"
    )
    for leaf in ("mamba.conv1d", "mamba.x_proj.0", "mamba.dt_proj.0"):
        spec = plan.find(leaf)
        assert spec is not None and not spec.quantize, f"{leaf} must be declared `:!`"


def test_hymba_gate_up_group_reads_pre_moe_layernorm():
    hf_model, qmodel = _synthetic_hymba()
    layer = hf_model.model.layers[0]
    plan = ModulePlan.from_tree(qmodel.module_tree)

    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
    groups = {
        get_op_name(layer, node["prev_op"]): [
            get_op_name(layer, module) for module in node["layers"]
        ]
        for node in nodes
        if node.get("prev_op") is not None
    }

    assert groups["input_layernorm"] == ["mamba.in_proj"]
    assert groups["pre_moe_layernorm"] == [
        "moe.experts.0.gate_proj",
        "moe.experts.0.up_proj",
    ]
    assert groups["moe.experts.0.up_proj"] == ["moe.experts.0.down_proj"]
    # The router reads the same normed input, so the fold has to scale it too.
    gate_up = next(
        node for node in nodes if get_op_name(layer, node["prev_op"]) == "pre_moe_layernorm"
    )
    assert gate_up["_scale_consumers"] == ["moe.gate"]


def test_hymba_awq_fold_is_output_preserving():
    hf_model, qmodel = _synthetic_hymba()
    plan = ModulePlan.from_tree(qmodel.module_tree)
    ids = torch.randint(0, VOCAB, (2, 6))

    worst = 0.0
    for layer in hf_model.model.layers:
        nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
        groups = _foldable_groups(layer, nodes)
        assert groups, "Hymba must build AWQ scaling groups"
        worst = max(worst, _worst_relative_change(hf_model, ids, layer, groups))

    assert worst < 1e-5, (
        f"folding the Hymba AWQ groups changed the logits by {worst:.3e}; the "
        "routed-expert gate/up group was folded into the wrong norm"
    )
