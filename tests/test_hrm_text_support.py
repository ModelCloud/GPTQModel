# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import torch
import torch.nn as nn

from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models import auto
from gptqmodel.models.definitions.hrm_text import HrmTextQModel
from gptqmodel.utils.model import get_layers_with_prefixes
from gptqmodel.utils.structure import LazyTurtle


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _DummyStack(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.layers = nn.ModuleList([_DummyLayer() for _ in range(count)])
        self.final_norm = nn.Identity()


class _DummyHrmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.rotary_emb = nn.Identity()
        self.model.L_module = _DummyStack(2)
        self.model.H_module = _DummyStack(3)


def test_hrm_text_model_type_selects_hrm_text_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="hrm_text")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/hrm_text") is HrmTextQModel


def test_get_layers_with_prefixes_flattens_hrm_stacks_in_order():
    model = _DummyHrmModel()

    layers, layer_names = get_layers_with_prefixes(model, HrmTextQModel.extract_layers_node())

    assert len(layers) == 5
    assert layer_names == [
        "model.L_module.layers.0",
        "model.L_module.layers.1",
        "model.H_module.layers.0",
        "model.H_module.layers.1",
        "model.H_module.layers.2",
    ]


def test_hrm_text_module_tree_expands_layer_and_base_paths():
    model = _DummyHrmModel()

    assert HrmTextQModel.extract_layers_node() == [
        "model.L_module.layers",
        "model.H_module.layers",
    ]
    assert HrmTextQModel.get_base_modules(model) == [
        "model.embed_tokens",
        "model.rotary_emb",
        "model.L_module.final_norm",
        "model.H_module.final_norm",
    ]
    assert HrmTextQModel.get_modules_with_direct_meta_tensors(model) == ["model"]


def test_hrm_text_module_tree_aliases_reach_lazy_turtle_prefixes():
    layer_prefixes, _ = LazyTurtle._build_moe_alias_specs(HrmTextQModel.module_tree)

    assert layer_prefixes == (
        ("model", "L_module", "layers"),
        ("model", "H_module", "layers"),
    )


def test_hrm_text_direct_meta_materialization_hook_targets_root_stack():
    outer_model = _DummyHrmModel()
    outer_model.model.register_parameter(
        "z_L_init",
        nn.Parameter(torch.empty(4, device="meta"), requires_grad=False),
    )

    calls = []

    class _FakeTurtle:
        def materialize_direct_meta_tensors(self, *, target_model, target_submodule, device=None):
            calls.append((target_model, target_submodule, device))
            target_submodule.register_parameter(
                "z_L_init",
                nn.Parameter(torch.zeros(4, device=device or "cpu"), requires_grad=False),
            )
            return 1

    instance = object.__new__(HrmTextQModel)
    nn.Module.__init__(instance)
    instance._turtle_lock = threading.RLock()
    instance.model = outer_model
    instance.turtle_model = _FakeTurtle()

    synced = instance.shell_direct_meta_materialize(
        target_submodule=outer_model.model,
        device=torch.device("cpu"),
    )

    assert synced == 1
    assert not outer_model.model.z_L_init.is_meta
    assert outer_model.model.z_L_init.device.type == "cpu"
    assert calls == [(outer_model, outer_model.model, torch.device("cpu"))]


def test_stage_input_capture_materializes_hrm_direct_meta_tensor_modules():
    outer_model = _DummyHrmModel()
    outer_model.model.register_parameter(
        "z_L_init",
        nn.Parameter(torch.empty(4, device="meta"), requires_grad=False),
    )

    calls = []

    class _FakeGPTQModel:
        def __init__(self, model):
            self.model = model

        def get_modules_with_direct_meta_tensors(self, model):
            return HrmTextQModel.get_modules_with_direct_meta_tensors(model)

        def shell_direct_meta_materialize(self, *, target_submodule, device=None):
            calls.append((target_submodule, device))
            target_submodule.register_parameter(
                "z_L_init",
                nn.Parameter(torch.zeros(4, device=device or "cpu"), requires_grad=False),
            )

    stage = StageInputsCapture(SimpleNamespace(gptq_model=_FakeGPTQModel(outer_model)))

    stage._materialize_modules_with_direct_meta_tensors(torch.device("cpu"))

    assert not outer_model.model.z_L_init.is_meta
    assert outer_model.model.z_L_init.device.type == "cpu"
    assert calls == [(outer_model.model, torch.device("cpu"))]
