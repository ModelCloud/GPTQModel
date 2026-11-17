# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Dream checkpoint ownership and conservative AWQ admission."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.dream import DreamQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.dream.configuration_dream import DreamConfig
from gptqmodel.models.ports.dream.modeling_dream import DreamModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig


QUANTIZED = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
NORMS = {"input_layernorm", "post_attention_layernorm"}
# Audited against all 339 keys of Dream-org/Dream-v0-Instruct-7B's pinned index:
# 12 tensors/layer * 28 layers + embedding, final norm and head.
LAYER_KEYS = {name + ".weight" for name in QUANTIZED | NORMS} | {
    "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}


def model_config(kv_heads=2, **kwargs):
    return DreamConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv_heads,
                       pad_token_id=0, mask_token_id=31, max_position_embeddings=32, **kwargs)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_real_state_dict_owners_tags_roles_and_aliases(kv_heads, method):
    with torch.device("meta"):
        model = DreamModel(model_config(kv_heads))
    _convert_model_with_defuser(DreamQModel, model, cleanup_original=False)
    wrapper = DreamQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} | {
        f"model.layers.{index}.{key}" for index in range(2) for key in LAYER_KEYS}
    assert set(model.state_dict()) == expected
    assert len(model.state_dict()) == 27
    blocks = DreamQModel.simple_layer_modules(model.config, SimpleNamespace(method=method, dynamic=None))
    assert {name for block in blocks for name in block} == QUANTIZED
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize is (owner in QUANTIZED)
            assert spec.quantize or "!" in spec.flags
    for name, role in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")]:
        assert role in plan.find("self_attn." + name).flags
    assert plan.find("self_attn.q_proj.undeclared") is None
    assert plan.find("self_attn.q_proj_extra") is None
    assert wrapper.awq_get_modules_for_scaling(model.model.layers[0], {}, {}) == []
    assert wrapper.support_batch_quantize is False


def test_published_checkpoint_meta_geometry_and_complete_key_inventory():
    config = DreamConfig(vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                         num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4,
                         rope_theta=1000000.0, max_position_embeddings=131072)
    with torch.device("meta"):
        model = DreamModel(config)
    assert set(model.state_dict()) == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} | {
        f"model.layers.{index}.{key}" for index in range(28) for key in LAYER_KEYS}
    assert len(model.state_dict()) == 339
    assert model.model.layers[0].self_attn.k_proj.weight.shape == (512, 3584)


@pytest.mark.parametrize("fallback", [None, "quantize"])
def test_awq_empty_groups_reach_real_quantization_and_pack_every_linear(fallback):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(29)
        model = DreamModel(model_config()).half().eval()
        wrapper = DreamQModel(model=model, quantized=False, quantize_config=None)
        processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=16),
                                 calibration=CalibrationConfig(), gptq_model=wrapper, model=model,
                                 calculate_w_wq_diff=False)
        processor.draw_progress = lambda *args, **kwargs: None
        processor.log_new_row = lambda *args, **kwargs: None
        processor.fallback = fallback
        ids = torch.tensor([[1, 2, 3]])
        for index, layer in enumerate(model.model.layers):
            norms = {name: layer.get_submodule(name).weight.detach().clone() for name in NORMS}
            state = processor._get_layer_state(index)
            state.layer_module = layer
            modules = {
                f"model.layers.{index}.{name}": NamedModule(layer.get_submodule(name), name, f"model.layers.{index}.{name}", index)
                for name in sorted(QUANTIZED)}
            weights = {name: named.weight.detach().clone() for name, named in modules.items()}
            state.modules = modules.copy()
            # Real projection captures populate the normal AWQ task representation.
            handles = []
            for name, named in modules.items():
                processor.tasks[name] = {"inputs": []}
                def capture(module, args, name=name):
                    processor.tasks[name]["inputs"].append(args[0].detach())
                handles.append(named.module.register_forward_pre_hook(capture))
            try:
                model(ids)
            finally:
                for handle in handles:
                    handle.remove()
            processor._quantize_layer(index, state)
            assert state.quantized
            for name, named in modules.items():
                named.stream_sync()
                assert named.state["q_scales"].numel() and named.state["q_zeros"].numel()
                assert not torch.equal(named.weight, weights[name])
                packed = AwqTorchLinear(bits=4, group_size=16, sym=False, desc_act=False,
                                        in_features=named.in_features, out_features=named.out_features,
                                        bias=named.bias is not None, dtype=torch.float16)
                packed.pack(named.module, named.state["q_scales"], named.state["q_zeros"])
                assert packed.qweight.dtype == torch.int32
                assert packed.qzeros.dtype == torch.int32
                parent_path, child_name = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_path), child_name, packed)
            for name, expected in norms.items():
                torch.testing.assert_close(layer.get_submodule(name).weight, expected, atol=0, rtol=0)
        assert sum(isinstance(module, AwqTorchLinear) for module in model.modules()) == 14
        assert torch.isfinite(model(ids).logits).all()
    finally:
        torch.set_num_threads(old_threads)
