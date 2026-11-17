# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Live Kimi-K2.5 decoder ownership, MoE consumers, and conservative plain AWQ."""

import hashlib
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.kimi_k25 import KimiK25QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from tests.test_kimi_k25_local_port import released_state, save_processor, split_expert_state, tiny_config


def test_pinned_checkpoint_inventory_matches_real_released_layout(tmp_path):
    import gptqmodel.models.ports.kimi_k25 as port

    directory = Path(port.__file__).parent
    fixture = json.loads((directory / "checkpoint_ownership.json").read_text())
    upstream = directory / "upstream_config.json"
    # The Hub source has no terminal newline; the repository text file does.
    assert hashlib.sha256(upstream.read_bytes().removesuffix(b"\n")).hexdigest() == fixture["config_sha256"]
    keys = set()
    for group in fixture["patterns"]:
        lmin, lmax = group.get("layers_inclusive", [0, 0])
        emin, emax = group.get("experts_inclusive", [0, 0])
        keys.update(group["template"].format(layer=l, expert=e)
                    for l, e in itertools.product(range(lmin, lmax + 1), range(emin, emax + 1)))
    assert len(keys) == fixture["key_count"] == 208550
    expected = set()
    for key in keys:
        layer = re.search(r"\.layers\.(\d+)\.", key)
        block = re.search(r"\.blocks\.(\d+)\.", key)
        expert = re.search(r"\.experts\.(\d+)\.", key)
        if (layer and int(layer[1]) >= 2) or (block and int(block[1]) >= 1) or (expert and int(expert[1]) >= 4):
            continue
        # The released checkpoint physically packs each routed expert into
        # weight_packed/scale/shape; all three belong to its logical Linear.
        key = re.sub(r"\.weight_(?:packed|scale|shape)$", ".weight", key)
        expected.add(key)
    with torch.device("meta"):
        model = KimiK25QModel.loader.from_config(tiny_config())
    assert set(released_state(model)) == expected
    (tmp_path / "config.json").write_bytes(upstream.read_bytes())
    config = KimiK25QModel.loader.load_config(tmp_path)
    assert config.text_config.model_type == "deepseek_v3"
    assert config.vision_config.hidden_size == 1152
    assert config.vision_config.num_hidden_layers == 27
    assert config.text_config.n_routed_experts == 384


def test_explicit_central_registry_case_for_integrator():
    import gptqmodel.models.ports.kimi_k25 as port

    case = json.loads((Path(port.__file__).parent / "registry_plan.json").read_text())
    assert KimiK25QModel.extract_layers_node() == case["layers"]
    assert KimiK25QModel.build_layer_modules(KimiK25QModel.module_tree, include_capture_only=True) == case["blocks"]
    assert KimiK25QModel.fusable is None


@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
@pytest.mark.parametrize("first_k_dense_replace", [0, 1], ids=["all_moe", "dense_then_moe"])
def test_every_real_checkpoint_owner_and_moe_consumer(tmp_path, q_lora_rank, first_k_dense_replace):
    assert MODEL_MAP["kimi_k25"] is KimiK25QModel
    config = tiny_config(q_lora_rank, first_k_dense_replace)
    with torch.device("meta"):
        model = KimiK25QModel.loader.from_config(config)
    expected_keys = set(split_expert_state(model))
    assert _convert_model_with_defuser(KimiK25QModel, model, cleanup_original=True)
    assert set(model.state_dict()) == expected_keys
    media = save_processor(tmp_path)
    wrapper = KimiK25QModel(
        model=model, quantized=False, quantize_config=None, model_local_path=str(tmp_path), tokenizer=media.tokenizer,
    )
    plan = wrapper.module_plan()
    simple = {
        name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block
    }
    attention = {"self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj", "self_attn.o_proj"}
    attention |= {"self_attn.q_proj"} if q_lora_rank is None else {"self_attn.q_a_proj", "self_attn.q_b_proj"}
    for i, layer in enumerate(model.model.language_model.layers):
        mlp = (
            {f"mlp.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")}
            if i < first_k_dense_replace else
            {f"mlp.experts.{e}.{leaf}" for e in range(4) for leaf in ("gate_proj", "up_proj", "down_proj")}
            | {f"mlp.shared_experts.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")}
        )
        found = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, f"layer {i}: unowned {key}"
            if isinstance(layer.get_submodule(owner), torch.nn.Linear):
                assert spec.quantize and owner in simple, owner
                found.add(owner)
            else:
                assert not spec.quantize and set(spec.flags) & {"!", "?"}, owner
        assert found == attention | mlp
    assert wrapper.extract_layers_node() == ["model.language_model.layers"]
    assert set(wrapper.get_base_modules(model)) >= {
        "model.language_model.embed_tokens", "model.language_model.norm", "model.language_model.rotary_emb",
        "model.vision_tower", "model.mm_projector",
    }
    assert wrapper.pre_lm_head_norm_module == "model.language_model.norm"
    for name in ("self_attn.q_a_layernorm", "self_attn.kv_a_layernorm", "mlp.gate"):
        assert not plan.find(name).quantize
        assert name not in simple
        assert plan.find(name + ".undeclared") is None
    assert plan.find("self_attn.q_a_proj_suffix") is None
    assert plan.roles("self_attn.q_b_proj") == ("q",)
    assert plan.roles("self_attn.kv_b_proj") == ("k", "v")
    assert plan.roles("mlp.experts.3.gate_proj") == ("gate",)
    assert plan.roles("mlp.shared_experts.up_proj") == ("up",)
    assert plan.expert_containers() == ("mlp.experts",)
    assert plan.shared_expert_containers() == ("mlp.shared_experts",)
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    assert looper._extract_moe_group_key("mlp.experts.2.up_proj") == "mlp.experts.2"
    assert looper._extract_moe_group_key("mlp.shared_experts.up_proj") is None
    assert wrapper.fusable is None


@pytest.mark.parametrize("q_lora_rank", [None, 16], ids=["dense_q", "split_q"])
@pytest.mark.parametrize("fallback", [None, {}], ids=["no_fallback", "fallback"])
def test_plain_awq_real_projection_pack_infer_and_safetensors_retry(tmp_path, monkeypatch, q_lora_rank, fallback):
    torch.manual_seed(90)
    config = tiny_config(
        q_lora_rank, hidden_size=32, intermediate_size=32, moe_intermediate_size=16, kv_lora_rank=16,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
    )
    model = KimiK25QModel.loader.from_config(config).eval().half()
    _convert_model_with_defuser(KimiK25QModel, model, cleanup_original=True)
    media = save_processor(tmp_path)
    wrapper = KimiK25QModel(
        model=model, quantized=False, quantize_config=None, model_local_path=str(tmp_path), tokenizer=media.tokenizer,
    )
    processor = AWQProcessor(
        tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=16, fallback=fallback),
        calibration=CalibrationConfig(), gptq_model=wrapper, model=model, calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    plan = wrapper.module_plan()
    packed_modules = {}
    original = {name: value.clone() for name, value in model.state_dict().items()}
    for i, layer in enumerate(model.model.language_model.layers):
        state = processor._get_layer_state(i)
        state.layer_module = layer
        state.modules = {
            name: NamedModule(module, name, f"model.language_model.layers.{i}.{name}", i)
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear) and plan.find(name).quantize
        }
        # Plain quantization also quantizes experts absent from activation samples.
        monkeypatch.setattr(processor, "_layer_input_features", lambda _: {})
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        named_modules = dict(state.modules)
        processor._quantize_layer(i, state)
        assert state.quantized
        for name, named in named_modules.items():
            linear = named.module
            assert named.state["q_scales"].numel() and named.state["q_zeros"].numel()
            packed = AwqTorchLinear(
                bits=4, group_size=16, sym=False, desc_act=False,
                in_features=linear.in_features, out_features=linear.out_features, register_buffers=True,
            )
            packed.pack(linear, named.state["q_scales"], named.state["q_zeros"])
            assert packed.qweight.dtype == torch.int32 and packed.qweight.numel()
            x = torch.linspace(-1, 1, linear.in_features).half().expand(2, -1)
            with torch.inference_mode():
                torch.testing.assert_close(packed(x), linear(x), rtol=2e-3, atol=2e-3)
            parent, _, leaf = name.rpartition(".")
            setattr(layer.get_submodule(parent), leaf, packed)
            packed_modules[f"model.language_model.layers.{i}.{name}"] = (linear.in_features, linear.out_features)
    assert len(packed_modules) == (26 if q_lora_rank is None else 28)
    for name, expected in original.items():
        if name.removesuffix(".weight") not in packed_modules:
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 6, 9]])
    with torch.inference_mode():
        expected = model(ids).logits
        assert torch.isfinite(expected).all() and expected.std() > .01
    output = tmp_path / "model.safetensors"
    save_file({key: value.contiguous() for key, value in model.state_dict().items()}, output, metadata={"format": "pt"})
    restored = KimiK25QModel.loader.from_config(config).eval().half()
    _convert_model_with_defuser(KimiK25QModel, restored, cleanup_original=True)
    for path, (in_features, out_features) in packed_modules.items():
        parent, _, leaf = path.rpartition(".")
        setattr(restored.get_submodule(parent), leaf, AwqTorchLinear(
            bits=4, group_size=16, sym=False, desc_act=False,
            in_features=in_features, out_features=out_features, register_buffers=True,
        ))
    state = load_file(output)
    broken = dict(state)
    broken[next(name for name in state if name.endswith(".qweight"))] = torch.ones(3, 3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="size mismatch"):
        restored.load_state_dict(broken, strict=True)
    restored.load_state_dict(state, strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)
    for name, value in state.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
