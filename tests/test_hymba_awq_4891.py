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

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.utils.module import get_op_name
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
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


# --------------------------------------------------------------------------- #
# #5098: a KV-reuse consumer layer must not need the producer's KV to score
#        `input_layernorm -> mamba.in_proj`.
# --------------------------------------------------------------------------- #


class _ReuseMamba(nn.Module):
    """`HymbaBlock` mixer with the native `reuse_kv` contract.

    The shapes follow the real `cuda_kernels_forward`: the projection is
    `[batch, latent, seq]`, the padding mask is applied with
    `attention_mask.unsqueeze(1)`, and a `reuse_kv` mixer asserts that it
    received the producing layer's K/V (#5098).
    """

    def __init__(self, reuse_kv: bool):
        super().__init__()
        self.reuse_kv = reuse_kv
        self.in_proj = nn.Linear(HIDDEN, 2 * INNER, bias=False)
        self.out_proj = nn.Linear(INNER, HIDDEN, bias=False)

    def forward(self, x, *, attention_mask=None, position_ids=None, kv_last_layer=None, use_swa=False, use_cache=False):
        del position_ids, use_swa, use_cache
        projected = self.in_proj(x).transpose(1, 2)
        if projected.shape[-1] > 1 and attention_mask is not None and (attention_mask == 0).any():
            projected = projected * attention_mask.unsqueeze(1).to(projected.dtype)
        if self.reuse_kv:
            assert kv_last_layer is not None
        gate, up = projected.chunk(2, dim=1)
        return self.out_proj(nn.functional.silu(gate.transpose(1, 2)) * up.transpose(1, 2))


class _ReuseBlock(nn.Module):
    """`HymbaDecoderLayer` for a KV producer/consumer pair."""

    def __init__(self, reuse_kv: bool):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(HIDDEN)
        self.mamba = _ReuseMamba(reuse_kv=reuse_kv)
        self.pre_moe_layernorm = nn.LayerNorm(HIDDEN)
        self.moe = _Moe()

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        mask = kwargs["attention_mask"]
        residual = hidden_states + self.mamba(
            self.input_layernorm(hidden_states),
            attention_mask=mask,
            position_ids=kwargs["position_ids"],
            kv_last_layer=kwargs["kv_last_layer"],
            use_swa=kwargs["use_swa"],
            use_cache=kwargs["use_cache"],
        )
        hidden_states = residual + self.moe(self.pre_moe_layernorm(residual))
        if mask is not None:
            hidden_states = hidden_states * mask[..., :1]
        return hidden_states, past_key_value


class _ReuseInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        # Layer 1 produces the K/V layer 2 consumes, like `kv_reuse_group=[[1, 2]]`.
        self.layers = nn.ModuleList([_ReuseBlock(reuse_kv=True), _ReuseBlock(reuse_kv=False), _ReuseBlock(reuse_kv=True)])
        self.final_layernorm = nn.LayerNorm(HIDDEN)
        self.memory_tokens = nn.Parameter(torch.zeros(4, HIDDEN))


class _ReuseCausalLM(nn.Module):
    """Threads `kv_last_layer` exactly like `HymbaModel.forward` does."""

    def __init__(self):
        super().__init__()
        self.model = _ReuseInner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = None

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
        kv_last_layer = None
        for layer in self.model.layers:
            hidden = layer(
                hidden,
                past_key_value=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_last_layer=kv_last_layer,
                use_swa=False,
                use_cache=False,
            )[0]
            kv_last_layer = (torch.zeros(1, INNER, 1, 8), torch.zeros(1, INNER, 1, 8))
        return _Out(self.lm_head(self.model.final_layernorm(hidden)))


def _reuse_hymba():
    torch.manual_seed(0)
    hf_model = _ReuseCausalLM().eval()
    qmodel = HymbaQModel.__new__(HymbaQModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    tree = HymbaQModel._resolve_effective_module_tree(hf_model, None)
    qmodel.__dict__["effective_module_tree"] = tree
    qmodel.__dict__["module_tree"] = tree
    return hf_model, qmodel


def _hymba_processor(qmodel, hf_model, *, fallback=None):
    return AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=HIDDEN, sym=True, fallback=fallback),
        calibration=CalibrationConfig(batch_size=1),
        gptq_model=qmodel,
        model=hf_model,
    )


def _mamba_group(layer, qmodel):
    plan = ModulePlan.from_tree(qmodel.module_tree)
    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})
    return next(
        node for node in nodes
        if get_op_name(layer, node["prev_op"]) == "input_layernorm"
    )


def test_hymba_kv_reuse_group_scores_the_projection_not_the_mixer():
    hf_model, qmodel = _reuse_hymba()
    consumer = hf_model.model.layers[2]
    assert consumer.mamba.reuse_kv

    group = _mamba_group(consumer, qmodel)

    assert [get_op_name(consumer, target) for target in group["layers"]] == ["mamba.in_proj"]
    assert group["module2inspect"] is consumer.mamba.in_proj, (
        "replaying `HymbaBlock` requires the producer layer's `kv_last_layer`, "
        "which the per-linear capture never records (#5098); the group has one "
        "Linear member, so score that projection"
    )


def test_hymba_kv_reuse_group_search_completes_without_cross_layer_kv():
    hf_model, qmodel = _reuse_hymba()
    consumer = hf_model.model.layers[2]
    processor = _hymba_processor(qmodel, hf_model)
    group = _mamba_group(consumer, qmodel)

    prev_name, targets, scales, loss = processor._search_best_scale(
        consumer,
        prev_op=group["prev_op"],
        layers=group["layers"],
        inp=group["inp"],
        module2inspect=group["module2inspect"],
        kwargs=dict(group["kwargs"]),
    )

    assert prev_name == "input_layernorm"
    assert targets == ("mamba.in_proj",)
    assert scales.shape == (HIDDEN,)
    assert torch.isfinite(scales).all()
    assert float(loss) >= 0.0


def test_hymba_mixer_replay_without_cross_layer_kv_uses_the_configured_fallback(caplog):
    """A native replay assertion must fall back instead of aborting `quantize()`."""

    hf_model, qmodel = _reuse_hymba()
    consumer = hf_model.model.layers[2]
    processor = _hymba_processor(qmodel, hf_model, fallback={})
    group = _mamba_group(consumer, qmodel)
    state = _AWQLayerState(layer_module=consumer, modules={
        "mamba.in_proj": NamedModule(
            consumer.mamba.in_proj, name="mamba.in_proj",
            full_name="model.layers.2.mamba.in_proj", layer_index=2,
        ),
    })
    features = {"mamba.in_proj": torch.randn(4, HIDDEN)}
    # Re-introduce the pre-#5098 boundary: replay the whole `HymbaBlock`. The
    # captured kwargs carry no `kv_last_layer`, exactly like the real capture.
    broken = dict(
        group,
        module2inspect=consumer.mamba,
        kwargs={
            "attention_mask": None,
            "position_ids": torch.arange(4).unsqueeze(0),
            "kv_last_layer": None,
            "use_swa": False,
            "use_cache": False,
        },
    )
    processor.gptq_model.awq_get_modules_for_scaling = lambda *_args: [broken]
    quantized = []

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(processor, "_layer_input_features", lambda _state: features)
        patcher.setattr(
            processor,
            "apply_quant",
            lambda modules, scales_list: quantized.append((set(modules), scales_list)),
        )
        processor._quantize_layer(2, state)

    assert (set(features), []) in quantized
    assert state.quantized
    assert "configured plain-quantization fallback" in caplog.text
