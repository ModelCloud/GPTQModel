# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native LFM3/VL and package-owned Omni consumer and AWQ regressions."""

from typing import cast

import pytest
import torch
from torch import nn
from transformers import Lfm3Config, Lfm3ForCausalLM, Lfm3VlConfig, Lfm3VlForConditionalGeneration

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.lfm3 import LFM3QModel
from gptqmodel.models.definitions.lfm3_omni import LFM3OmniQModel
from gptqmodel.models.definitions.lfm3_vl import LFM3VLQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.lfm3_omni.modeling_lfm3_omni import Lfm3OmniForConditionalGeneration
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, Fallback, QuantizeConfig
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes
from tests.test_lfm3_omni_local_port import tiny_config as tiny_omni_config


DEFINITIONS: dict[str, type[LFM3QModel]] = {
    "lfm3": LFM3QModel, "lfm3_vl": LFM3VLQModel, "lfm3_omni": LFM3OmniQModel,
}
LAYOUTS = [
    ("lfm3", "native"), ("lfm3", "legacy_gate"), ("lfm3", "ungated"),
    ("lfm3_vl", "native"), ("lfm3_vl", "ungated"),
    ("lfm3_omni", "native"), ("lfm3_omni", "unsandwiched"),
]
MLP = {"feed_forward.w1", "feed_forward.w3", "feed_forward.w2"}
QUANTIZED = {
    "conv": {"conv.in_proj", "conv.out_proj"} | MLP,
    "full_attention": {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"} | MLP,
    "sliding_attention": {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"} | MLP,
}
IDS = torch.tensor([[0, 1, 9, 4, 6], [1, 7, 3, 8, 2]])
MASK = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])


@pytest.fixture(autouse=True)
def bounded_cpu_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            yield
    finally:
        torch.set_num_threads(previous)


def _fixture(kind, *, device="cpu", layout="native", dtype=torch.float32, fallback=None):
    config = Lfm3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["conv", "full_attention", "sliding_attention"],
        conv_L_cache=3, conv_bias=True, attention_bias=True, sliding_window=8,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        use_attention_gate=layout != "ungated", use_sandwich_norm=layout not in {"ungated", "unsandwiched"},
    )
    config._attn_implementation = "eager"
    with torch.device(device):
        if kind == "lfm3":
            model = Lfm3ForCausalLM(config)
        elif kind == "lfm3_vl":
            model = Lfm3VlForConditionalGeneration(Lfm3VlConfig(
                text_config=config,
                vision_config={
                    "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "patch_size": 2,
                    "temporal_patch_size": 1, "layer_types": ["conv", "full_attention"],
                },
                projector_hidden_size=48, image_token_id=60, video_token_id=61,
                vision_start_token_id=62, vision_end_token_id=63,
            ))
        else:
            omni_config = tiny_omni_config(bias=True, sandwich=config.use_sandwich_norm)
            config = type(omni_config.text_config)(**config.to_dict())
            omni_config.text_config = config
            model = Lfm3OmniForConditionalGeneration(omni_config)
    model = cast(nn.Module, model).to(dtype=dtype).eval()
    definition = DEFINITIONS[kind]
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    decoder_layers, prefixes = cast(
        tuple[list[nn.Module], list[str]],
        get_layers_with_prefixes(model, definition.extract_layers_node()),
    )
    layers = dict(zip(prefixes, decoder_layers))
    if layout == "legacy_gate":
        for layer in layers.values():
            if hasattr(layer, "self_attn"):
                attention = layer.get_submodule("self_attn")
                gate = attention.get_submodule("g_proj")
                del attention.g_proj
                attention.gate_proj = gate
                # Preserve the native forward without a second state_dict owner.
                object.__setattr__(attention, "g_proj", gate)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=fallback)
    # Compile the real decoder contract without unrelated media/tokenizer I/O.
    wrapper = definition.__new__(definition)
    nn.Module.__init__(wrapper)
    tree = definition._resolve_effective_module_tree(model, qcfg)
    wrapper.__dict__.update(
        model=model, quantize_config=qcfg, effective_module_tree=tree,
        module_tree=tree, _module_plan=ModulePlan.from_tree(tree), qlinear_kernel=None,
    )
    return model, wrapper, layers, config


def _consumer_blocks(wrapper, *, planning=False) -> list[list[str]]:
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    looper.processors = []
    looper._layer_modules_by_prefix = {}
    blocks = looper.layer_modules_for_prefix(wrapper.extract_layers_node()[0], planning=planning)
    assert blocks is not None
    return blocks


def _logits(model):
    output = model(IDS, attention_mask=MASK, use_cache=False)
    return output.logits if hasattr(output, "logits") else output


@pytest.mark.parametrize("kind,layout", LAYOUTS)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_real_decoder_owners_tags_and_public_consumer(kind, device, layout):
    model, wrapper, layers, config = _fixture(kind, device=device, layout=layout)
    plan = wrapper.module_plan()
    blocks = _consumer_blocks(wrapper)
    selected = {name for block in blocks for name in block}
    full = {name for block in _consumer_blocks(wrapper, planning=True) for name in block}
    assert len(layers) == 3
    assert selected == set.union(*QUANTIZED.values())
    gates = {"lfm3": ("g_proj", "gate_proj"), "lfm3_vl": ("g_proj",), "lfm3_omni": ("gate_proj",)}[kind]
    conv_blocks = [["conv.in_proj"], ["conv.conv:!"], ["conv.out_proj"]]
    if kind == "lfm3_vl":
        conv_blocks = [["conv.in_proj"], ["conv.out_proj"], ["conv.conv:!"]]
    assert _consumer_blocks(wrapper, planning=True) == [
        ["operator_norm:!"], ["post_operator_norm:!"],
        *conv_blocks,
        ["self_attn.q_proj", "self_attn.q_norm:!", "self_attn.k_proj", "self_attn.k_norm:!",
         *(f"self_attn.{gate}:!" for gate in gates), "self_attn.v_proj"],
        ["self_attn.o_proj"], ["ffn_norm:!"], ["post_ffn_norm:!"],
        ["feed_forward.w1", "feed_forward.w3"], ["feed_forward.w2"],
    ]
    for name in ("conv.conv", *(f"self_attn.{gate}" for gate in gates)):
        spec = plan.find(name)
        assert spec is not None, f"undeclared source-dtype owner: {name}"
        assert "!" in spec.flags and not spec.quantize and not spec.capture_only
        assert name not in selected
        assert f"{name}:!" in full
    for gate in {"g_proj", "gate_proj"} - set(gates):
        assert plan.find(f"self_attn.{gate}") is None
    assert config.layer_types is not None
    for layer_type, layer in zip(config.layer_types, layers.values()):
        owners = {key.rsplit(".", 1)[0] for key in layer.state_dict()}
        dense = {"operator_norm", "ffn_norm"}
        if config.use_sandwich_norm:
            dense |= {"post_operator_norm", "post_ffn_norm"}
        if layer_type == "conv":
            dense.add("conv.conv")
            conv = layer.get_submodule("conv.conv")
            assert isinstance(conv, nn.Conv1d)
            assert conv.groups == config.hidden_size
            assert conv.weight.shape == (32, 1, 3)
        else:
            dense |= {"self_attn.q_norm", "self_attn.k_norm"}
            if config.use_attention_gate:
                gate = "gate_proj" if kind == "lfm3_omni" or layout == "legacy_gate" else "g_proj"
                dense.add(f"self_attn.{gate}")
                gate_module = layer.get_submodule(f"self_attn.{gate}")
                assert isinstance(gate_module, nn.Linear)
                assert gate_module.weight.shape == (4, 32)
        assert owners == QUANTIZED[layer_type] | dense
        assert set(find_modules(layer)) & selected == QUANTIZED[layer_type]
        for owner in owners:
            spec = plan.find(owner)
            assert spec is not None, owner
            assert spec.quantize == (owner in QUANTIZED[layer_type])
        indices = {name: i for i, block in enumerate(blocks) for name in block}
        if layer_type == "conv":
            assert indices["conv.in_proj"] < indices["conv.out_proj"]
        else:
            assert indices["self_attn.q_proj"] < indices["self_attn.o_proj"]
        assert indices["feed_forward.w1"] == indices["feed_forward.w3"] < indices["feed_forward.w2"]
    if device == "cpu":
        with torch.inference_mode():
            assert torch.isfinite(_logits(model)).all()


@pytest.mark.parametrize("kind,layout", LAYOUTS)
@pytest.mark.parametrize(
    "dtype,fallback", [(torch.float32, None), (torch.bfloat16, Fallback())], ids=["fp32", "bf16-rtn-fallback"],
)
def test_awq_quantizes_and_packs_every_projection_and_preserves_excluded_owners(
    monkeypatch, kind, layout, dtype, fallback,
):
    model, wrapper, layers, config = _fixture(kind, layout=layout, dtype=dtype, fallback=fallback)
    selected = {name for block in _consumer_blocks(wrapper) for name in block}
    processor = AWQProcessor(
        tokenizer=None, qcfg=cast(QuantizeConfig, wrapper.quantize_config), calibration=CalibrationConfig(),
        gptq_model=wrapper, model=model, calculate_w_wq_diff=False,
    )
    monkeypatch.setattr(processor, "draw_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(processor, "log_new_row", lambda *args, **kwargs: None)
    assert config.layer_types is not None
    for index, ((prefix, layer), layer_type) in enumerate(zip(layers.items(), config.layer_types)):
        features, handles = {}, []

        def record(name):
            def capture(module, args):
                features[f"{prefix}.{name}"] = args[0].detach().clone()
            return capture

        try:
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    handles.append(module.register_forward_pre_hook(record(name)))
            with torch.inference_mode():
                assert torch.isfinite(_logits(model)).all()
        finally:
            for handle in handles:
                handle.remove()
        assert {f"{prefix}.{name}" for name in QUANTIZED[layer_type]} <= features.keys()
        local_features = {name.removeprefix(prefix + "."): value for name, value in features.items()}
        attention = getattr(layer, "self_attn", None)
        if kind == "lfm3_vl":
            # VL calls the generic planner directly, so it keeps the
            # `operator_norm` fold and its explicit value-to-output fold.
            fold_sources = {"operator_norm", "ffn_norm", "feed_forward.w3"}
            if attention is not None and attention.v_proj.out_features == attention.o_proj.in_features:
                fold_sources.add("self_attn.v_proj")
        elif layer_type == "conv":
            # Conv layers have no gate, so both input norms fold.
            fold_sources = {"operator_norm", "ffn_norm"}
        else:
            # An attention layer's `operator_norm` feeds the non-quantized
            # sigmoid gate; the shipped contract folds only `ffn_norm` there.
            fold_sources = {"ffn_norm"}
        groups = wrapper.awq_get_modules_for_scaling(layer, local_features, {})
        assert {node["prev_op"] for node in groups} == {layer.get_submodule(name) for name in fold_sources}
        if layer_type != "conv":
            assert [node["prev_op"] for node in groups if attention.o_proj in node["layers"]] == (
                [attention.v_proj] if "self_attn.v_proj" in fold_sources else []
            )
            ffn_group = next(node for node in groups if node["prev_op"] is layer.ffn_norm)
            assert ffn_group["layers"] == [layer.feed_forward.w1, layer.feed_forward.w3]
            if "operator_norm" in fold_sources:
                norm_group = next(node for node in groups if node["prev_op"] is layer.operator_norm)
                assert norm_group["layers"] == [attention.q_proj, attention.k_proj, attention.v_proj]
                gate = "gate_proj" if kind == "lfm3_omni" or layout == "legacy_gate" else "g_proj"
                assert norm_group.get("_scale_consumers", []) == (
                    [f"self_attn.{gate}"] if config.use_attention_gate else []
                )
            else:
                assert all(node["prev_op"] is not layer.operator_norm for node in groups)
        elif kind != "lfm3_vl":
            conv_group = next(node for node in groups if node["prev_op"] is layer.operator_norm)
            assert conv_group["layers"] == [layer.conv.in_proj]
        # Folds also divide a linear source's bias and compensate non-quantized
        # readers of the folded norm. VL keeps the compensated attention gate;
        # base LFM3/LFM3-Omni leave `g_proj` untouched by suppressing the fold.
        folded_extras = {f"{name}.bias" for name in fold_sources}
        folded_extras |= {f"{name}.weight" for node in groups for name in node.get("_scale_consumers") or []}
        before = {key: value.clone() for key, value in layer.state_dict().items()}
        state = processor._get_layer_state(index)
        state.layer_module = layer
        named = {
            f"{prefix}.{name}": NamedModule(module, name, f"{prefix}.{name}", index)
            for name, module in find_modules(layer).items() if name in selected
        }
        assert {module.name for module in named.values()} == QUANTIZED[layer_type]
        state.modules = dict(named)

        def interrupted_capture(state):
            raise RuntimeError("fixture capture interrupted")

        monkeypatch.setattr(processor, "_layer_input_features", interrupted_capture)
        with pytest.raises(RuntimeError, match="fixture capture interrupted"):
            processor._quantize_layer(index, state)
        assert not state.quantized
        for key, value in before.items():
            torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
        monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
        processor._quantize_layer(index, state)
        assert state.quantized
        for module in named.values():
            assert {"q_scales", "q_zeros", "wq"} <= module.state.keys()
            assert not torch.equal(before[f"{module.name}.weight"], module.weight)
            packed = AwqTorchLinear(
                bits=4, group_size=16, sym=False, desc_act=False,
                in_features=module.in_features, out_features=module.out_features,
                bias=module.bias is not None,
            )
            packed.pack(module.module, module.state["q_scales"], module.state["q_zeros"])
            assert isinstance(packed.qweight, torch.Tensor) and isinstance(packed.qzeros, torch.Tensor)
            assert packed.qweight.dtype == packed.qzeros.dtype == torch.int32
            assert packed.qweight.numel() * 8 == module.weight.numel()
        mutable_weights = {f"{name}.weight" for name in QUANTIZED[layer_type] | fold_sources} | folded_extras
        for key, value in before.items():
            if key not in mutable_weights:
                torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
                assert layer.state_dict()[key].dtype == dtype
        after = {key: value.clone() for key, value in layer.state_dict().items()}
        processor._quantize_layer(index, state)
        for key, value in after.items():
            torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(_logits(model)).all()
