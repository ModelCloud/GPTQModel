# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.integrations import use_experts_implementation

from gptqmodel.models import auto
from gptqmodel.models.definitions.intern_s2_mobius import (
    InternS2MobiusMoELifecycleHooks,
    InternS2MobiusQModel,
)
from gptqmodel.utils.structure import LazyTurtle
from model_test import ModelTest


def test_intern_s2_mobius_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="interns2_mobius")
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )
    assert auto.check_and_get_model_definition("intern-s2-mobius-fixture") is InternS2MobiusQModel


def test_intern_s2_mobius_decoder_and_auxiliary_module_expansion():
    config = SimpleNamespace(num_experts=3, model_type="interns2_mobius")
    modules = InternS2MobiusQModel.full_layer_modules(config)
    flat = {name for block in modules for name in block}
    assert "mlp.shared_expert.gate_proj" in flat
    assert "mlp.experts.0.gate_proj" not in flat
    assert "experts.0.gate_proj" in flat
    assert "experts.2.down_proj" in flat
    assert InternS2MobiusQModel.extract_layers_node() == [
        "model.language_model.layers",
        "model.language_model.meta_mlp",
    ]


def test_intern_s2_mobius_auxiliary_cache_isolated_and_aggregated():
    class Block(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    meta_mlp = nn.ModuleList([Block(), Block()])
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.meta_mlp = meta_mlp

    qmodel = object.__new__(InternS2MobiusQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel._intern_aux_capture_handles = []
    qmodel._intern_aux_capture_sinks = {}
    qmodel._intern_aux_capture_active = None
    qmodel._intern_aux_capture_context = None
    qmodel._intern_aux_active_block_index = None
    qmodel.begin_auxiliary_input_capture(processor="p")
    for layer_index in (0, 1, 4, 5):
        qmodel.before_layer_forward(
            layer=nn.Identity(),
            layer_index=layer_index,
            batch_index=0,
            processor="p",
            layer_input=[],
            additional_inputs={},
            target_device=torch.device("cpu"),
        )
        meta_mlp[layer_index % 2](torch.ones(1, 2, 4))

    caches = qmodel.finalize_auxiliary_input_capture(processor="p")
    assert set(caches) == {
        "model.language_model.meta_mlp.0",
        "model.language_model.meta_mlp.1",
    }
    assert caches["model.language_model.meta_mlp.0"].layer_inputs[0][0].shape == (4, 4)
    assert caches["model.language_model.meta_mlp.1"].layer_inputs[0][0].shape == (4, 4)
    qmodel.close_auxiliary_input_capture()


def test_intern_s2_mobius_root_moe_subset_paths():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 3, bias=False)
            self.up_proj = nn.Linear(4, 3, bias=False)
            self.down_proj = nn.Linear(3, 4, bias=False)

    block = nn.Module()
    block.gate = nn.Linear(4, 2, bias=False)
    block.experts = nn.ModuleList([Expert(), Expert()])
    subset = {
        "experts.0.gate_proj": block.experts[0].gate_proj,
        "experts.0.up_proj": block.experts[0].up_proj,
    }
    hooks = InternS2MobiusMoELifecycleHooks()

    assert hooks.get_moe_block_for_subset(block, InternS2MobiusQModel, subset) is block
    assert hooks._extract_moe_block_prefix(subset, block) == ""
    assert hooks.get_subset_execution_order(
        ordered_module_names=list(subset),
        moe_block_prefix="",
        experts_attr_name="experts",
        shared_expert_attr_name=None,
    ) == ["experts"]


def test_intern_s2_mobius_defuses_and_resolves_packed_experts(tmp_path):
    @use_experts_implementation
    class PackedExperts(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.num_experts = 2
            self.gate_up_proj = nn.Parameter(torch.randn(2, 6, 4))
            self.down_proj = nn.Parameter(torch.randn(2, 4, 3))

        def forward(self, hidden_states, top_k_index=None, top_k_weights=None):
            del top_k_index, top_k_weights
            return hidden_states

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = PackedExperts(SimpleNamespace())

    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    root.model.language_model.meta_mlp = nn.ModuleList([Block()])
    InternS2MobiusQModel.after_defuser_conversion(root)
    experts = root.model.language_model.meta_mlp[0].experts
    assert isinstance(experts[0].gate_proj, nn.Linear)
    assert isinstance(experts[1].up_proj, nn.Linear)
    assert isinstance(experts[0].down_proj, nn.Linear)

    save_file(
        {
            "model.language_model.meta_mlp.0.experts.gate_up_proj": torch.arange(48.0).reshape(2, 6, 4),
            "model.language_model.meta_mlp.0.experts.down_proj": torch.arange(24.0).reshape(2, 4, 3),
        },
        tmp_path / "model.safetensors",
    )
    turtle = LazyTurtle(
        model_local_path=str(tmp_path),
        config=SimpleNamespace(),
        module_tree=InternS2MobiusQModel.module_tree,
        target_model=root,
    )
    gate_source = turtle._resolve_checkpoint_tensor_source(
        "model.language_model.meta_mlp.0.experts.1.gate_proj",
        "weight",
    )
    up_source = turtle._resolve_checkpoint_tensor_source(
        "model.language_model.meta_mlp.0.experts.1.up_proj",
        "weight",
    )
    down_source = turtle._resolve_checkpoint_tensor_source(
        "model.language_model.meta_mlp.0.experts.1.down_proj",
        "weight",
    )
    assert gate_source[:3] == (
        "model.language_model.meta_mlp.0.experts.gate_up_proj",
        1,
        0,
    )
    assert up_source[:3] == (
        "model.language_model.meta_mlp.0.experts.gate_up_proj",
        1,
        1,
    )
    assert down_source[:3] == (
        "model.language_model.meta_mlp.0.experts.down_proj",
        1,
        None,
    )


class TestInternS2Mobius(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Intern-S2-Mobius"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    TORCH_DTYPE = torch.bfloat16
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_SINGLE_GPU = False
    EVAL_BATCH_SIZE = 1
    # Official model-card MMLU-Pro result; this is a sourced baseline, not a
    # locally measured quantized score.  Protocol/tokenizer differences can
    # make direct lm-eval comparisons non-equivalent.
    EVAL_TASKS_SLOW = {
        "mmlu_pro": {
            "chat_template": True,
            # Evalution reports MMLU-Pro as exact-match choice labels.  The
            # 0.8905 value is the model card's OpenCompass result, so it is a
            # traceable reference only; those protocols are not equivalent.
            "em,choice_label": {"value": 0.8905, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_intern_s2_mobius(self):
        self.quantize_and_evaluate()
