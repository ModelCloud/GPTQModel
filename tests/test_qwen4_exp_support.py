# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.qwen4_exp import Qwen4ExpQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.model import apply_no_placement_to_device_map, simple_dispatch_model
from gptqmodel.utils.structure import LazyTurtle


def _outer_config(num_experts=3):
    return SimpleNamespace(text_config=SimpleNamespace(num_experts=num_experts))


class _TinyNoPlacementModel(nn.Module):
    _no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        layer = self.model.language_model.layers[1]
        layer.proj = nn.Linear(4, 4)
        layer.ple = nn.Module()
        layer.ple.ple_embedding = nn.Module()
        layer.ple.ple_embedding.ngram_embedding = nn.Embedding(8, 4)
        layer.ple.key_proj = nn.Linear(4, 4)


def test_qwen4_exp_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="qwen4_exp")
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/qwen3.8-flash-next") is Qwen4ExpQModel


def test_qwen4_exp_quantized_load_keeps_ple_embedding_on_cpu():
    model = _TinyNoPlacementModel()
    layer_name = "model.language_model.layers.1"
    embedding_name = f"{layer_name}.ple.ple_embedding.ngram_embedding"

    device_map = apply_no_placement_to_device_map(model, {layer_name: "cuda:1"})

    assert layer_name not in device_map
    assert device_map[f"{layer_name}.proj"] == "cuda:1"
    assert device_map[f"{layer_name}.ple.key_proj"] == "cuda:1"
    assert device_map[embedding_name] == "cpu"
    assert not any(
        embedding_name.startswith(f"{name}.")
        for name in device_map
        if name != embedding_name
    )


def test_qwen4_exp_dispatch_does_not_move_cpu_ple_under_gpu_parent(monkeypatch):
    import accelerate

    model = _TinyNoPlacementModel()
    layer_name = "model.language_model.layers.1"
    embedding_name = f"{layer_name}.ple.ple_embedding.ngram_embedding"
    device_map = {layer_name: "cuda:1", embedding_name: "cpu"}
    added_hooks = []

    monkeypatch.setattr(accelerate.utils.modeling, "find_tied_parameters", lambda model: [])
    monkeypatch.setattr(accelerate.utils.modeling, "retie_parameters", lambda model, tied: None)
    monkeypatch.setattr(
        accelerate,
        "cpu_offload_with_hook",
        lambda *args, **kwargs: pytest.fail("no-placement PLE must remain resident on CPU"),
    )
    monkeypatch.setattr(
        accelerate.hooks,
        "add_hook_to_module",
        lambda module, hook: added_hooks.append((module, hook)),
    )

    simple_dispatch_model(model, device_map)

    assert len(added_hooks) == 1
    assert added_hooks[0][0] is model.model.language_model.layers[1]
    assert added_hooks[0][1].place_submodules is False


def test_qwen4_exp_module_tree_quantizes_selected_attention_and_mlp_linears():
    modules = Qwen4ExpQModel.simple_layer_modules(
        _outer_config(),
        SimpleNamespace(dynamic=None),
    )
    flat = {name for block in modules for name in block}
    captured = {
        name
        for block in Qwen4ExpQModel.full_layer_modules(
            _outer_config(),
            include_capture_only=True,
        )
        for name in block
    }

    assert Qwen4ExpQModel.extract_layers_node() == ["model.language_model.layers"]
    assert Qwen4ExpQModel.pre_lm_head_norm_module == "model.language_model.hyper_connection_mixer"
    assert Qwen4ExpQModel.out_of_model_tensors == {"prefixes": ["mtp"]}
    assert Qwen4ExpQModel.hf_conversion_model_type_alias == "qwen4_exp_text"

    assert modules[:7] == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["linear_attn.in_proj_qkv"],
        ["linear_attn.in_proj_z"],
        ["linear_attn.out_proj"],
        ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"],
        ["mlp.shared_expert.down_proj"],
    ]
    assert len(modules) == 9
    assert flat == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.0.down_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.1.up_proj",
        "mlp.experts.1.down_proj",
        "mlp.experts.2.gate_proj",
        "mlp.experts.2.up_proj",
        "mlp.experts.2.down_proj",
    }

    for name in (
        "self_attn.indexer.index_qk_proj:!",
        "self_attn.indexer.q_layernorm:!",
        "linear_attn.conv1d:!",
        "linear_attn.in_proj_b:!",
        "linear_attn.in_proj_a:!",
        "linear_attn.norm:!",
        "mlp.gate:!",
        "mlp.shared_expert_gate:!",
    ):
        assert name in captured

    shared = [name for block in modules for name in block if name.startswith("mlp.shared_expert.")]
    assert shared == [
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    ]


def _tiny_qwen4_exp_text_model():
    transformers = pytest.importorskip("transformers")
    config_cls = getattr(transformers, "Qwen4ExpTextConfig", None)
    model_cls = getattr(transformers, "Qwen4ExpForCausalLM", None)
    if config_cls is None or model_cls is None:
        pytest.skip("Installed Transformers does not provide Qwen4Exp")

    config = config_cls(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )
    return model_cls(config).eval()


def test_qwen4_exp_outer_model_reuses_text_checkpoint_conversion_map():
    transformers = pytest.importorskip("transformers")
    if getattr(transformers, "Qwen4ExpConfig", None) is None:
        pytest.skip("Installed Transformers does not provide Qwen4Exp")

    outer_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp"))
    reversed_map = Qwen4ExpQModel.resolve_hf_conversion_map_reversed(target_model=outer_model)

    assert reversed_map is not None
    assert any(
        "ngram_embedding.weight" in converter.source_patterns
        and "ngram_embedding.shard_*.weight" in converter.target_patterns
        for converter in reversed_map
        if hasattr(converter, "source_patterns") and hasattr(converter, "target_patterns")
    )


def test_qwen4_exp_outer_lazy_materialization_loads_ple_shards(tmp_path):
    transformers = pytest.importorskip("transformers")
    config_cls = getattr(transformers, "Qwen4ExpConfig", None)
    model_cls = getattr(transformers, "Qwen4ExpForConditionalGeneration", None)
    if config_cls is None or model_cls is None:
        pytest.skip("Installed Transformers does not provide Qwen4Exp")

    text_config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 32,
        "layer_types": ["qwen_sparse_attention", "linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 2,
        "hc_count": 2,
        "hc_lowrank": 8,
        "ple_layer_ids": [2],
        "ple_embed_dim": 16,
        "ple_conv_kernel_size": 2,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 32,
        "make_ngram_vocab_size_divisible_by": 2,
        "split_ngram_parts": 2,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": 32,
        "num_position_embeddings": 16,
    }
    config = config_cls(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    source = model_cls(config).eval()
    expected = source.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight.detach().clone()
    source.save_pretrained(tmp_path)

    with torch.device("meta"):
        shell = model_cls(config).eval()
    assert _convert_model_with_defuser(Qwen4ExpQModel, shell, cleanup_original=False)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(tmp_path),
        config=shell.config,
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=Qwen4ExpQModel.module_tree,
        hf_conversion_map_reversed=Qwen4ExpQModel.resolve_hf_conversion_map_reversed(target_model=shell),
        target_model=shell,
    )
    assert turtle is not None

    layer = shell.model.language_model.layers[1]
    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=layer,
        device=torch.device("cpu"),
        module_path="model.language_model.layers.1",
        show_progress=False,
    )

    actual = layer.ple.ple_embedding.ngram_embedding.weight
    assert actual.device.type == "cpu"
    assert not any(parameter.is_meta for parameter in layer.parameters())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_exp_defuser_preserves_model_type_and_forward():
    model = _tiny_qwen4_exp_text_model()
    input_ids = torch.tensor([[1, 7, 8, 2]])
    original_model_type = model.config.model_type
    packed_experts = model.model.layers[0].mlp.experts
    assert hasattr(packed_experts, "gate_up_proj")

    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits

    assert _convert_model_with_defuser(Qwen4ExpQModel, model, cleanup_original=False) is True
    experts = model.model.layers[0].mlp.experts

    assert model.config.model_type == original_model_type
    assert not hasattr(experts, "gate_up_proj")
    assert isinstance(experts[0].gate_proj, torch.nn.Linear)
    assert isinstance(experts[0].up_proj, torch.nn.Linear)
    assert isinstance(experts[0].down_proj, torch.nn.Linear)

    with torch.inference_mode():
        actual = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
