# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import torch
import torch.nn as nn
import transformers

from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models import auto
from gptqmodel.models.definitions.hrm_text import HrmTextQModel
from gptqmodel.models.ports.native.auto import CONFIG_MAPPING
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


def test_hrm_text_model_type_selects_hrm_text_definition(tmp_path):
    CONFIG_MAPPING["hrm_text"]().save_pretrained(tmp_path)

    assert auto.check_and_get_model_definition(str(tmp_path)) is HrmTextQModel


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
        def materialize_direct_meta_tensors(
            self, *, target_model, target_submodule, device=None, module_path=None
        ):
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


def _tiny_hrm_text_model():
    config = transformers.HrmTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        H_cycles=2,
        L_cycles=2,
        L_bp_cycles=[1, 1],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return transformers.HrmTextForCausalLM(config).eval()


def _hrm_wrapper(model):
    instance = object.__new__(HrmTextQModel)
    nn.Module.__init__(instance)
    instance._turtle_lock = threading.RLock()
    instance.model = model
    instance.turtle_model = None
    instance.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    return instance


def test_hrm_text_calibration_captures_every_recurrent_invocation():
    """The L stack runs H_cycles*L_cycles times, the H stack H_cycles times."""

    model = _tiny_hrm_text_model()
    instance = _hrm_wrapper(model)
    layers, layer_names = get_layers_with_prefixes(
        model, HrmTextQModel.extract_layers_node()
    )
    assert layer_names == [
        "model.L_module.layers.0",
        "model.L_module.layers.1",
        "model.H_module.layers.0",
        "model.H_module.layers.1",
    ]

    example = {
        "input_ids": torch.randint(0, 64, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    instance.build_quantization_input_cache(
        layers=layers,
        calibration_data=[example],
        use_cache=False,
        layer_names=layer_names,
    )

    per_layer = instance._looped_per_layer
    assert [len(per_layer[id(layer)]["layer_inputs"]) for layer in layers] == [4, 4, 2, 2]

    # Each L-stack invocation receives a different recurrent state, so a single
    # captured chain pass cannot represent the calibration distribution.
    l0_inputs = [batch[0] for batch in per_layer[id(layers[0])]["layer_inputs"]]
    flattened = torch.nn.functional.cosine_similarity(
        l0_inputs[0].flatten(), l0_inputs[-1].flatten(), dim=0
    )
    assert flattened.item() < 0.999

    # H0's input is `z_H + final_norm(z_L)`, not the L1 output the flattened
    # chain would hand it.
    l1_outputs = per_layer[id(layers[1])]["layer_inputs"]
    h0_inputs = per_layer[id(layers[2])]["layer_inputs"]
    assert len(h0_inputs) == 2
    for h0_input, l1_input in zip(h0_inputs, l1_outputs):
        assert not torch.allclose(
            h0_input[0], l1_input[0], rtol=0.0, atol=1e-6
        ), "H0 received the flattened L1 output instead of z_H + final_norm(z_L)"


def test_hrm_text_pre_quantize_serves_each_layers_own_invocations():
    model = _tiny_hrm_text_model()
    instance = _hrm_wrapper(model)
    layers, layer_names = get_layers_with_prefixes(
        model, HrmTextQModel.extract_layers_node()
    )
    example = {
        "input_ids": torch.randint(0, 64, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    cache = instance.build_quantization_input_cache(
        layers=layers,
        calibration_data=[example],
        use_cache=False,
        layer_names=layer_names,
    )

    for layer in layers:
        instance.pre_quantize(layer)
        assert cache.layer_inputs is instance._looped_per_layer[id(layer)]["layer_inputs"]
        assert len(cache.attention_masks) == len(cache.layer_inputs)


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

        def shell_direct_meta_materialize(
            self, *, target_submodule, device=None, module_path=None
        ):
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
